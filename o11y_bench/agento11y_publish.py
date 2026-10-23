"""Stream completed Harbor trials to Agent Observability experiments."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from grading.transcript_parser import parse_transcript
from reporting.report_data import (
    agent_result_metrics,
    agent_seconds,
    grading_details,
    load_trials,
    reward_counts_as_pass,
    trial_reasoning_effort,
)

from .agento11y_suite import DEFAULT_SUITE_ID

_STATE_FILE = ".agento11y-publish.json"


@dataclass(slots=True)
class Agento11yPublishOptions:
    """Environment-driven settings for one benchmark experiment."""

    experiment_id: str = ""
    suite_id: str = DEFAULT_SUITE_ID
    suite_version: str = "latest_published"
    name: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=lambda: ["o11y-bench"])

    @classmethod
    def from_env(cls) -> Agento11yPublishOptions:
        raw_tags = os.getenv("AGENTO11Y_EXPERIMENT_TAGS", "")
        tags = [tag.strip() for tag in raw_tags.split(",") if tag.strip()]
        return cls(
            experiment_id=os.getenv("AGENTO11Y_EXPERIMENT_ID", "").strip(),
            suite_id=os.getenv("AGENTO11Y_SUITE_ID", DEFAULT_SUITE_ID).strip() or DEFAULT_SUITE_ID,
            suite_version=os.getenv("AGENTO11Y_SUITE_VERSION", "latest_published").strip()
            or "latest_published",
            name=os.getenv("AGENTO11Y_EXPERIMENT_NAME", "").strip(),
            description=os.getenv("AGENTO11Y_EXPERIMENT_DESCRIPTION", "").strip(),
            tags=["o11y-bench", *tags],
        )


@dataclass(frozen=True, slots=True)
class Agento11yPublishResult:
    experiment_id: str
    trial_count: int
    score_count: int
    url: str


class Agento11yLivePublisher:
    """Publish each scored Harbor trial as soon as its result artifact appears."""

    def __init__(
        self,
        job_dir: Path,
        tasks_dir: Path,
        *,
        model: str,
        agent_name: str,
        agent_version: str,
        options: Agento11yPublishOptions | None = None,
        planned_trial_count: int | None = None,
        poll_interval_seconds: float = 1.0,
    ) -> None:
        self.job_dir = job_dir.resolve()
        self.tasks_dir = tasks_dir.resolve()
        self.model = model
        self.agent_name = agent_name
        self.agent_version = agent_version
        self.options = options or Agento11yPublishOptions.from_env()
        self.planned_trial_count = planned_trial_count
        self.poll_interval_seconds = poll_interval_seconds
        self.experiment_id = self.options.experiment_id or _default_experiment_id(self.job_dir)
        self._sdk: Any | None = None
        self._experiments: Any | None = None
        self._experiment: Any | None = None
        self._suite_cases: dict[str, Any] = {}
        self._exported: set[str] = set()
        self._attempts: dict[str, int] = {}
        self._trial_count = 0
        self._score_count = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._scan_lock = threading.Lock()
        self._error: Exception | None = None
        self.last_result: Agento11yPublishResult | None = None
        self._load_state()

    @property
    def url(self) -> str:
        if self._experiment is not None:
            return str(self._experiment.url)
        base = os.getenv("AGENTO11Y_GRAFANA_URL", "").rstrip("/")
        return f"{base}/a/grafana-agento11y-app/experiments/runs/{self.experiment_id}"

    def start(self) -> None:
        self._connect()
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="agento11y-live-publish",
            daemon=True,
        )
        self._thread.start()

    def finish(self, *, succeeded: bool, error: str = "") -> Agento11yPublishResult:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.poll_interval_seconds * 3))

        publish_error = self._error
        try:
            self.scan_once()
        except Exception as exc:
            publish_error = publish_error or exc

        failure: RuntimeError | None = None
        if publish_error is not None:
            failure = RuntimeError(f"Agent Observability live publishing failed: {publish_error}")
        elif not succeeded:
            failure = RuntimeError(error or "Harbor benchmark failed")

        result_url = self.url
        if self._experiment is not None:
            if failure is None:
                self._experiment.__exit__(None, None, None)
            else:
                self._experiment.__exit__(RuntimeError, failure, None)
        self.last_result = Agento11yPublishResult(
            experiment_id=self.experiment_id,
            trial_count=self._trial_count,
            score_count=self._score_count,
            url=result_url,
        )
        if publish_error is not None:
            assert failure is not None
            raise failure from publish_error
        return self.last_result

    def publish_existing(self) -> Agento11yPublishResult:
        self._connect()
        self.scan_once()
        return self.finish(succeeded=True)

    def scan_once(self) -> None:
        if self._experiment is None or not self.job_dir.exists():
            return
        with self._scan_lock:
            for result in load_trials(self.job_dir):
                result_path = Path(str(result["__result_path"]))
                trial_dir = result_path.parent
                harbor_trial_id = trial_dir.name
                rewards = (result.get("verifier_result") or {}).get("rewards") or {}
                if harbor_trial_id in self._exported or rewards.get("reward") is None:
                    continue
                self._publish_trial(result, trial_dir, rewards)
                self._exported.add(harbor_trial_id)
                self._trial_count += 1
                self._save_state()

    def _connect(self) -> None:
        if self._experiment is not None:
            return
        sdk, experiments = _import_agento11y()
        provider, model_name = _model_ref(self.model)
        experiment = experiments.experiment_from_suite(
            self.options.suite_id,
            version=self.options.suite_version,
            name=self.options.name or f"o11y-bench {self.job_dir.name}",
            experiment_id=self.experiment_id,
            candidate={
                "agent_name": self.agent_name,
                "agent_version": self.agent_version,
                "model_provider": provider,
                "model_name": model_name,
            },
            description=self.options.description,
            tags=_dedupe(
                [*self.options.tags, provider, model_name, trial_reasoning_label(self.model)]
            ),
            planned_trial_count=self.planned_trial_count,
            metadata={
                "source": "o11y-bench",
                "job_name": self.job_dir.name,
                "framework": "harbor",
                "transcript_format": "ATIF",
            },
        )
        experiment.__enter__()
        self._sdk = sdk
        self._experiments = experiments
        self._experiment = experiment
        self._suite_cases = {case.test_case_id: case for case in experiment.suite.cases}

    def _poll_loop(self) -> None:
        while not self._stop.wait(self.poll_interval_seconds):
            try:
                self.scan_once()
            except Exception as exc:
                self._error = exc
                self._stop.set()

    def _publish_trial(
        self,
        result: dict[str, Any],
        trial_dir: Path,
        rewards: dict[str, Any],
    ) -> None:
        assert self._sdk is not None
        assert self._experiments is not None
        assert self._experiment is not None
        task_id = str(result.get("task_name") or trial_dir.name.split("__", 1)[0])
        case = self._suite_cases.get(task_id)
        if case is None:
            raise ValueError(f"Harbor result references task {task_id!r}, absent from stored suite")
        attempt = self._attempt_for(task_id, trial_dir.name)
        cost, input_tokens, cache_tokens, output_tokens = agent_result_metrics(result)
        provider, model_name = _result_model_ref(result, self.model)
        conversation_id = self._experiments.stable_id("conv", self.experiment_id, task_id, attempt)
        generation_id = self._experiments.stable_id("gen", self.experiment_id, task_id, attempt)
        final_output = _final_output(trial_dir)
        usage = self._sdk.TokenUsage(
            input_tokens=input_tokens,
            cache_read_input_tokens=cache_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + cache_tokens + output_tokens,
        )
        self._experiment.client.record_generation(
            generation_id,
            conversation_id=conversation_id,
            input_text=str(case.input),
            output_text=final_output,
            model_provider=provider,
            model_name=model_name,
            agent_name=self.agent_name,
            agent_version=self.agent_version,
            operation_name="harbor-trial",
            usage=usage,
            tags={
                "experiment.run_id": self.experiment_id,
                "task_id": task_id,
                "task_category": case.category,
            },
            metadata={
                "source": "o11y-bench",
                "harbor_trial_id": trial_dir.name,
                "reasoning_effort": trial_reasoning_effort(result),
            },
        )
        self._experiment.client.flush_generations()

        version = str(result.get("task_checksum") or "unknown")
        score_specs = _score_specs(result, trial_dir, rewards, version)
        with self._experiment.trial(
            case,
            attempt=attempt,
            metadata={
                "source": "o11y-bench",
                "harbor_trial_id": trial_dir.name,
                "reasoning_effort": trial_reasoning_effort(result),
            },
        ) as trial:
            trial.bind_generation(generation_id, conversation_id=conversation_id)
            trial.set_usage(
                cost=cost if cost > 0 else None,
            )
            for score in score_specs:
                trial.record_evaluation(score)
                self._score_count += trial.flush()
            for name, path in _artifact_paths(trial_dir):
                trial.artifact(name, path=str(path))

        duration_ms = max(0, int(agent_seconds(result) * 1000))
        self._experiment.client.update_trial(
            self.experiment_id,
            trial.trial_id,
            status="completed",
            duration_ms=duration_ms,
            conversation_id=conversation_id,
            cost=cost if cost > 0 else None,
        )

    def _attempt_for(self, task_id: str, harbor_trial_id: str) -> int:
        key = f"{task_id}\x1f{harbor_trial_id}"
        existing = self._attempts.get(key)
        if existing is not None:
            return existing
        used = [
            attempt
            for stored, attempt in self._attempts.items()
            if stored.startswith(f"{task_id}\x1f")
        ]
        attempt = max(used, default=0) + 1
        self._attempts[key] = attempt
        self._save_state()
        return attempt

    def _load_state(self) -> None:
        path = self.job_dir / _STATE_FILE
        try:
            payload = json.loads(path.read_text())
        except OSError, json.JSONDecodeError:
            return
        if payload.get("experiment_id") != self.experiment_id:
            return
        self._exported = {str(item) for item in payload.get("exported_trials", [])}
        self._attempts = {
            str(key): int(value) for key, value in (payload.get("attempts") or {}).items()
        }
        self._trial_count = len(self._exported)
        self._score_count = int(payload.get("score_count") or 0)

    def _save_state(self) -> None:
        if not self.job_dir.exists():
            return
        payload = {
            "experiment_id": self.experiment_id,
            "exported_trials": sorted(self._exported),
            "attempts": self._attempts,
            "score_count": self._score_count,
        }
        (self.job_dir / _STATE_FILE).write_text(json.dumps(payload, indent=2) + "\n")


def _score_specs(
    result: dict[str, Any],
    trial_dir: Path,
    rewards: dict[str, Any],
    version: str,
) -> list[Any]:
    _, experiments = _import_agento11y()
    details = grading_details(trial_dir)
    check_names, rubric_names = _evaluator_names(trial_dir, result)
    final_value = float(rewards["reward"])
    scores = [
        experiments.EvaluationResult(
            evaluator=experiments.Evaluator(
                evaluator_id="o11y-bench.verifier",
                version=version,
                kind=experiments.EvaluatorKind.CUSTOM.value,
            ),
            value=final_value,
            passed=reward_counts_as_pass(result),
            explanation=_overall_explanation(details),
            score_key="final",
            metadata={"source": "o11y-bench"},
        )
    ]
    score_values = {name: value for name, value in rewards.items() if name != "reward"}
    for name, value in details.items():
        if name == "score" or name.startswith("explanation:") or name in score_values:
            continue
        if isinstance(value, (bool, int, float)):
            score_values[name] = value
    for name, raw_value in score_values.items():
        try:
            value = float(raw_value)
        except TypeError, ValueError:
            continue
        is_check = name in check_names or name in {"checks_passed", "validators_passed"}
        kind = (
            experiments.EvaluatorKind.DETERMINISTIC.value
            if is_check
            else experiments.EvaluatorKind.LLM_JUDGE.value
        )
        prefix = "check" if is_check else "rubric"
        scores.append(
            experiments.EvaluationResult(
                evaluator=experiments.Evaluator(
                    evaluator_id=f"o11y-bench.{prefix}.{_slug(name)}",
                    version=version,
                    kind=kind,
                ),
                value=value,
                passed=value >= 1.0,
                explanation=str(details.get(f"explanation:{name}") or ""),
                score_key=f"{prefix}.{_slug(name)}",
                metadata={
                    "source": "o11y-bench",
                    "criterion": name,
                    "outcome": str(details.get(f"outcome:{name}") or ""),
                    "evaluator_runtime": "harbor-verifier",
                    "declared_in_suite": name in check_names or name in rubric_names,
                },
            )
        )
    return scores


def _evaluator_names(trial_dir: Path, result: dict[str, Any]) -> tuple[set[str], set[str]]:
    task_name = str(result.get("task_name") or trial_dir.name.split("__", 1)[0])
    problem_path = trial_dir / "config.json"
    task_root: Path | None = None
    try:
        config = json.loads(problem_path.read_text())
        raw_task_path = (config.get("task") or {}).get("path")
        if isinstance(raw_task_path, str) and raw_task_path:
            task_root = Path(raw_task_path)
    except OSError, json.JSONDecodeError:
        pass
    candidates = [
        task_root / "tests" / "problem.yaml" if task_root is not None else None,
        trial_dir.parent.parent / "tasks" / task_name / "tests" / "problem.yaml",
    ]
    payload: dict[str, Any] = {}
    for candidate in candidates:
        if candidate is None or not candidate.exists():
            continue
        loaded = yaml.safe_load(candidate.read_text())
        if isinstance(loaded, dict):
            payload = loaded
            break
    checks = {
        str(item.get("name"))
        for item in payload.get("checks", [])
        if isinstance(item, dict) and item.get("name")
    }
    rubrics = {
        str(item.get("criterion"))
        for item in payload.get("rubric", [])
        if isinstance(item, dict) and item.get("criterion")
    }
    return checks, rubrics


def _overall_explanation(details: dict[str, Any]) -> str:
    explanations = [
        str(value)
        for key, value in details.items()
        if str(key).startswith("explanation:") and value
    ]
    return "\n".join(explanations)


def _final_output(trial_dir: Path) -> str:
    transcript = parse_transcript(trial_dir / "agent")
    for message in reversed(transcript.messages):
        if message.role == "assistant" and message.content:
            return message.content
    return ""


def _artifact_paths(trial_dir: Path) -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = []
    agent_dir = trial_dir / "agent"
    for name, pattern in (
        ("harbor-trajectory", "trajectory.json"),
        ("harbor-transcript", "transcript.jsonl"),
    ):
        matches = sorted(agent_dir.rglob(pattern)) if agent_dir.exists() else []
        if matches:
            paths.append((name, matches[0]))
    grading = trial_dir / "verifier" / "grading_details.json"
    if grading.exists():
        paths.append(("grading-details", grading))
    return paths


def _result_model_ref(result: dict[str, Any], fallback: str) -> tuple[str, str]:
    model_info = (result.get("agent_info") or {}).get("model_info") or {}
    provider = str(model_info.get("provider") or "")
    name = str(model_info.get("name") or fallback)
    fallback_provider, fallback_name = _model_ref(fallback)
    if "/" in name:
        embedded_provider, name = name.split("/", 1)
        provider = provider or embedded_provider
    return provider or fallback_provider, name or fallback_name


def _model_ref(model: str) -> tuple[str, str]:
    provider, separator, name = model.partition("/")
    return (provider, name) if separator else ("", model)


def _default_experiment_id(job_dir: Path) -> str:
    digest = hashlib.sha1(str(job_dir.resolve()).encode()).hexdigest()[:10]
    return f"o11y-bench-{job_dir.name}-{digest}"


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "score"
    if len(normalized) <= 80:
        return normalized
    digest = hashlib.sha1(value.encode()).hexdigest()[:10]
    return f"{normalized[:69]}-{digest}"


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def trial_reasoning_label(model: str) -> str:
    return f"candidate:{model}"


def _import_agento11y() -> tuple[Any, Any]:
    try:
        import agento11y
        from agento11y import experiments
    except ImportError as exc:
        raise RuntimeError(
            "Agent Observability publishing requires agento11y>=0.12.0. "
            "Run `mise run agento11y:setup` to install the published SDK."
        ) from exc
    return agento11y, experiments
