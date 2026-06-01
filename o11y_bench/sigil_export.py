"""Export completed o11y-bench job artifacts to Sigil experiments.

This is an **optional, opt-in** feature. It is only triggered by the
``--sigil-experiment-export`` flag on ``o11y_bench job`` or by the standalone
``o11y_bench sigil-export`` subcommand, and it requires the Sigil Python SDK
(install it with ``mise run sigil:setup``). The SDK is imported lazily, so
importing this module — and the rest of o11y-bench — works fine when the SDK is
not installed.

It rides on the Sigil SDK's generic experiment runner (``ExperimentRun``): each
completed trial becomes a Sigil generation (the agent transcript) plus a score
(the verifier reward), attributed to one external experiment ``run_id`` so runs
can be browsed and compared in Sigil.
"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from grading.models import Message as BenchMessage
from grading.transcript_parser import parse_transcript
from reporting.report_data import (
    agent_result_metrics,
    agent_seconds,
    load_task_categories,
    load_trials,
    reward_counts_as_pass,
    trial_reasoning_effort,
)


@dataclass(slots=True)
class SigilExportOptions:
    api: str = "http://localhost:8080"
    tenant: str = "fake"
    run_id: str = ""
    name: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=lambda: ["o11y-bench"])
    sdk_path: Path | None = None


@dataclass(slots=True)
class SigilExportResult:
    run_id: str
    generation_count: int
    score_count: int
    url: str


class SigilLiveExporter:
    """Live exporter for an o11y-bench job.

    It creates the Sigil experiment before Harbor starts, exports each completed
    trial as soon as its result artifact is available, then marks the experiment
    terminal after Harbor exits.
    """

    def __init__(
        self,
        job_dir: Path,
        tasks_dir: Path,
        options: SigilExportOptions,
        *,
        poll_interval_seconds: float = 2.0,
    ) -> None:
        self.job_dir = job_dir.resolve()
        self.tasks_dir = tasks_dir.resolve()
        self.options = options
        self.poll_interval_seconds = poll_interval_seconds
        self.run_id = options.run_id or _default_run_id(self.job_dir)
        self._sdk: Any | None = None
        self._client: Any | None = None
        self._run: Any | None = None
        self._categories: dict[str, str] = {}
        self._exported_trial_ids: set[str] = set()
        self._generation_count = 0
        self._score_count = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._error: Exception | None = None
        self.last_result: SigilExportResult | None = None

    @property
    def url(self) -> str:
        if self._client is not None:
            return str(self._client.experiment_url(self.run_id))
        return f"{self.options.api.rstrip('/')}/a/grafana-sigil-app/evaluation/experiments/{self.run_id}"

    def start(self) -> None:
        self._connect()
        self._thread = threading.Thread(
            target=self._poll_loop, name="sigil-live-export", daemon=True
        )
        self._thread.start()

    def finish(self, *, status: str = "succeeded", error: str = "") -> SigilExportResult:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.poll_interval_seconds * 2))
        self.scan_once()
        if self._client is not None:
            self._client.complete_experiment(
                self.run_id,
                status=status,
                score_count=self._score_count,
                error=error,
                metadata={
                    "source": "o11y-bench",
                    "job_name": self.job_dir.name,
                    "exported_generations": self._generation_count,
                    "exported_scores": self._score_count,
                },
            )
            self._client.shutdown()
        if self._error is not None:
            raise RuntimeError(f"Sigil live export failed: {self._error}") from self._error
        self.last_result = SigilExportResult(
            run_id=self.run_id,
            generation_count=self._generation_count,
            score_count=self._score_count,
            url=self.url,
        )
        return self.last_result

    def export_existing(self) -> SigilExportResult:
        self._connect()
        self.scan_once()
        return self.finish(status="succeeded")

    def scan_once(self) -> None:
        if self._run is None:
            return
        if not self.job_dir.exists():
            return
        for trial in load_trials(self.job_dir):
            trial_dir = Path(str(trial["__result_path"])).parent
            trial_id = trial_dir.name
            reward = ((trial.get("verifier_result") or {}).get("rewards") or {}).get("reward")
            if trial_id in self._exported_trial_ids or reward is None:
                continue
            self._export_trial(trial, trial_dir, reward)

    # --- setup -------------------------------------------------------------- #

    def _connect(self) -> None:
        if self._run is not None:
            return
        sdk = _import_sdk(self.options.sdk_path)
        self._sdk = sdk
        self._client = _make_client(sdk, self.options)
        self._categories = load_task_categories(self.tasks_dir)
        self._create_or_update_experiment()
        # ExperimentRun handles run_id tagging on generations + deterministic,
        # idempotent score ids; we drive it manually (no context manager) so it
        # stays open across Harbor's lifetime.
        self._run = sdk.ExperimentRun(
            client=self._client,
            run_id=self.run_id,
            name=self.options.name or f"o11y-bench {self.job_dir.name}",
            dataset={"id": "o11y-bench"},
            candidate={"job_name": self.job_dir.name},
            upload="continuous",
            agent_name="o11y-bench",
        )

    def _create_or_update_experiment(self) -> None:
        sdk, client = self._sdk, self._client
        if sdk is None or client is None:
            return
        tags = _normalized_tags(self.options.tags)
        name = self.options.name or f"o11y-bench {self.job_dir.name}"
        try:
            client.create_experiment(
                sdk.CreateExperimentRequest(
                    run_id=self.run_id,
                    name=name,
                    source="external",
                    description=self.options.description,
                    tags=tags,
                    metadata={
                        "source": "o11y-bench",
                        "job_name": self.job_dir.name,
                        "job_dir": str(self.job_dir),
                        "tasks_dir": str(self.tasks_dir),
                    },
                )
            )
        except sdk.ConflictError:
            # Re-running an export against an existing run: reopen it so we can
            # append any newly completed trials, then re-finalize on finish().
            client.update_experiment(
                self.run_id,
                sdk.UpdateExperimentRequest(
                    name=name,
                    description=self.options.description,
                    tags=tags,
                    status="running",
                ),
            )

    # --- per-trial export --------------------------------------------------- #

    def _poll_loop(self) -> None:
        while not self._stop.wait(self.poll_interval_seconds):
            try:
                self.scan_once()
            except Exception as exc:
                with self._lock:
                    if self._error is None:
                        self._error = exc
                self._stop.set()

    def _export_trial(self, trial: dict[str, Any], trial_dir: Path, reward: Any) -> None:
        sdk, run = self._sdk, self._run
        if sdk is None or run is None:
            return
        task_name = str(trial.get("task_name") or trial_dir.name.split("__", 1)[0])
        trial_id = trial_dir.name
        category = self._categories.get(task_name, "unknown")
        generation_id = _stable_id("o11y-gen", self.run_id, trial_id)
        conversation_id = _stable_id("o11y-conv", self.run_id, trial_id)

        # One conversation per trial; start_generation tags the generation with
        # the experiment run_id and captures its id for the score below.
        run.reset_capture(conversation_id=conversation_id)
        with run.start_generation(
            _generation_start(
                sdk, trial, trial_dir, generation_id, conversation_id, task_name, category
            )
        ) as recorder:
            recorder.set_result(
                _generation_result(sdk, trial, trial_dir, generation_id, conversation_id, task_name)
            )

        run.add_scores(
            [_score_output(sdk, trial, trial_dir, reward, self.job_dir.name, task_name, category)],
            item=sdk.DatasetItem(id=task_name),
            generation_ids=run.produced_generation_ids,
            trial_id=trial_id,
        )
        self._exported_trial_ids.add(trial_id)
        self._generation_count += 1
        self._score_count = int(run.accepted_scores)


def export_job_to_sigil(
    job_dir: Path, tasks_dir: Path, options: SigilExportOptions
) -> SigilExportResult:
    exporter = SigilLiveExporter(job_dir, tasks_dir, options, poll_interval_seconds=0.0)
    result = exporter.export_existing()
    if result.generation_count == 0:
        raise ValueError(f"no completed scored trial result.json files found under {job_dir}")
    return result


# --- generation + score mapping -------------------------------------------- #


def _score_output(
    sdk: Any,
    trial: dict[str, Any],
    trial_dir: Path,
    reward: Any,
    job_name: str,
    task_name: str,
    task_category: str,
) -> Any:
    cost_usd, n_input_tokens, n_cache_tokens, n_output_tokens = agent_result_metrics(trial)
    total_tokens = n_input_tokens + n_cache_tokens + n_output_tokens
    return sdk.ScoreOutput(
        evaluator_id="o11y-bench.verifier",
        evaluator_version=_evaluator_version(trial),
        score_key="reward",
        value=sdk.ScoreValue(number=float(reward)),
        passed=reward_counts_as_pass(trial),
        explanation=_score_explanation(trial_dir),
        metadata={
            "source": "o11y-bench",
            "job_name": job_name,
            "task_id": task_name,
            "task_category": task_category,
            "cost_usd": cost_usd,
            "input_tokens": n_input_tokens,
            "cache_tokens": n_cache_tokens,
            "output_tokens": n_output_tokens,
            "total_tokens": total_tokens,
            "wall_time_seconds": agent_seconds(trial),
            "model": _model_name(trial),
            "reasoning_effort": trial_reasoning_effort(trial),
            "result_path": str(trial_dir / "result.json"),
        },
    )


def _make_client(sdk: Any, options: SigilExportOptions) -> Any:
    api = options.api.rstrip("/")
    return sdk.Client(
        sdk.ClientConfig(
            api=sdk.ApiConfig(endpoint=api),
            generation_export=sdk.GenerationExportConfig(
                protocol="http",
                endpoint=f"{api}/api/v1/generations:export",
                auth=sdk.AuthConfig(mode="tenant", tenant_id=options.tenant),
            ),
        )
    )


def _generation_start(
    sdk: Any,
    trial: dict[str, Any],
    trial_dir: Path,
    generation_id: str,
    conversation_id: str,
    task_name: str,
    task_category: str,
) -> Any:
    cost_usd, *_ = agent_result_metrics(trial)
    transcript = parse_transcript(trial_dir / "agent")
    system_prompt, _input_messages, _output_messages = _split_transcript_messages(
        sdk, transcript.messages
    )
    model_provider, model_name = _model_ref(trial)
    return sdk.GenerationStart(
        id=generation_id,
        conversation_id=conversation_id,
        conversation_title=task_name,
        model=sdk.ModelRef(provider=model_provider, name=model_name),
        operation_name="o11y-bench trial",
        agent_version=trial_reasoning_effort(trial),
        system_prompt=system_prompt,
        tags={
            "source": "o11y-bench",
            "task_id": task_name,
            "task_category": task_category,
        },
        metadata={
            "source": "o11y-bench",
            "task_id": task_name,
            "task_category": task_category,
            "trial_id": trial_dir.name,
            "cost_usd": cost_usd,
        },
        started_at=_parse_datetime((trial.get("agent_execution") or {}).get("started_at")),
    )


def _generation_result(
    sdk: Any,
    trial: dict[str, Any],
    trial_dir: Path,
    generation_id: str,
    conversation_id: str,
    task_name: str,
) -> Any:
    _cost_usd, n_input_tokens, n_cache_tokens, n_output_tokens = agent_result_metrics(trial)
    transcript = parse_transcript(trial_dir / "agent")
    _system_prompt, input_messages, output_messages = _split_transcript_messages(
        sdk, transcript.messages
    )
    if not input_messages:
        input_messages = [sdk.user_text_message(f"Run o11y-bench task {task_name}.")]
    if not output_messages:
        output_messages = [sdk.assistant_text_message("No assistant transcript was captured.")]
    model_provider, model_name = _model_ref(trial)
    return sdk.Generation(
        id=generation_id,
        conversation_id=conversation_id,
        conversation_title=task_name,
        model=sdk.ModelRef(provider=model_provider, name=model_name),
        response_model=_model_name(trial),
        input=input_messages,
        output=output_messages,
        usage=sdk.TokenUsage(
            input_tokens=n_input_tokens,
            cache_read_input_tokens=n_cache_tokens,
            output_tokens=n_output_tokens,
            total_tokens=n_input_tokens + n_cache_tokens + n_output_tokens,
        ),
        metadata={"trial_id": trial_dir.name, "run_id": generation_id},
    )


def _split_transcript_messages(
    sdk: Any, messages: list[BenchMessage]
) -> tuple[str, list[Any], list[Any]]:
    system_parts: list[str] = []
    input_messages: list[Any] = []
    output_messages: list[Any] = []
    for message in messages:
        if message.role == "system":
            if message.content:
                system_parts.append(message.content)
            continue
        mapped = _map_transcript_message(sdk, message)
        if mapped is None:
            continue
        if message.role == "user":
            input_messages.append(mapped)
        else:
            output_messages.append(mapped)
    return "\n\n".join(system_parts), input_messages, output_messages


def _map_transcript_message(sdk: Any, message: BenchMessage) -> Any | None:
    parts = []
    if message.thinking_content:
        parts.append(sdk.thinking_part(message.thinking_content))
    if message.content:
        parts.append(sdk.text_part(message.content))
    if message.tool_calls:
        for call in message.tool_calls:
            parts.append(
                sdk.tool_call_part(
                    sdk.ToolCall(
                        id=call.id,
                        name=call.name,
                        input_json=json.dumps(call.arguments).encode(),
                    )
                )
            )
    if message.tool_results:
        for result in message.tool_results:
            parts.append(
                sdk.tool_result_part(
                    sdk.ToolResult(
                        tool_call_id=result.tool_call_id,
                        content=result.content,
                    )
                )
            )
    if not parts:
        return None
    role = sdk.MessageRole.USER if message.role == "user" else sdk.MessageRole.ASSISTANT
    if message.role == "tool":
        role = sdk.MessageRole.TOOL
    return sdk.Message(role=role, parts=parts)


def _score_explanation(trial_dir: Path) -> str:
    details_path = trial_dir / "verifier" / "grading_details.json"
    if not details_path.exists():
        return ""
    try:
        details = json.loads(details_path.read_text())
    except Exception:
        return ""
    explanations = [
        str(value)
        for key, value in details.items()
        if isinstance(key, str) and key.startswith("explanation:") and value
    ]
    return "\n".join(explanations)


def _evaluator_version(trial: dict[str, Any]) -> str:
    checksum = trial.get("task_checksum")
    if isinstance(checksum, str) and checksum:
        return checksum
    return "unknown"


def _model_name(trial: dict[str, Any]) -> str:
    model_info = (trial.get("agent_info") or {}).get("model_info") or {}
    return str(model_info.get("name") or "")


def _model_ref(trial: dict[str, Any]) -> tuple[str, str]:
    model_info = (trial.get("agent_info") or {}).get("model_info") or {}
    provider = str(model_info.get("provider") or "")
    name = str(model_info.get("name") or "")
    if not provider and "/" in name:
        provider, name = name.split("/", 1)
    return provider, name


def _default_run_id(job_dir: Path) -> str:
    digest = hashlib.sha1(str(job_dir.resolve()).encode()).hexdigest()[:10]
    return f"o11y-bench-{job_dir.name}-{digest}"


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha1("\x1f".join(parts).encode()).hexdigest()[:24]
    return f"{prefix}-{digest}"


def _normalized_tags(tags: list[str]) -> list[str]:
    out = []
    for tag in ["o11y-bench", *tags]:
        normalized = tag.strip()
        if normalized and normalized not in out:
            out.append(normalized)
    return out


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _import_sdk(sdk_path: Path | None = None) -> Any:
    """Imports the Sigil SDK lazily, with a friendly opt-in error if it's absent.

    ``sdk_path`` (or ``SIGIL_SDK_PYTHON_PATH``) is an optional escape hatch to
    point at a local SDK checkout without installing it; normally the SDK is
    installed into the venv via ``mise run sigil:setup``.
    """

    if sdk_path is not None:
        sys.path.insert(0, str(sdk_path.resolve()))
    try:
        import sigil_sdk
    except ImportError as exc:
        raise RuntimeError(
            "Sigil export is an opt-in feature and needs the Sigil Python SDK, which is "
            "not installed in this environment.\n"
            "Enable it with:\n"
            "    mise run sigil:setup\n"
            "or point at a local SDK checkout with --sigil-sdk-path / SIGIL_SDK_PYTHON_PATH."
        ) from exc
    return sigil_sdk
