#!/usr/bin/env python3
"""Shared data-loading and summarization helpers for Harbor reports."""

import html
import json
import os
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from .model_costs import estimate_cost_usd
from .report_paths import normalize_repo_path

MODEL_LABELS: dict[str, str] = {
    "claude-sonnet-4-6": "Sonnet 4.6",
    "claude-sonnet-4-5": "Sonnet 4.5",
    "claude-opus-4-6": "Opus 4.6",
    "claude-haiku-4-5-20251001": "Haiku 4.5",
    "claude-haiku-4-5": "Haiku 4.5",
    "gpt-5.4": "GPT 5.4",
    "gpt-5.4-2026-03-05": "GPT 5.4",
    "gpt-5.4-mini": "GPT 5.4 Mini",
    "gpt-5.4-nano": "GPT 5.4 Nano",
    "gpt-5.2": "GPT 5.2",
    "gpt-5.2-2025-12-11": "GPT 5.2",
    "gpt-5.2-codex": "GPT 5.2 Codex",
    "gpt-5.1-codex-mini": "GPT 5.1 Codex Mini",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "gemini-3-flash-preview": "Gemini 3 Flash",
    "gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash Lite",
}

HIGH_SCORE_THRESHOLD = 0.7
MEDIUM_SCORE_THRESHOLD = 0.4
INVALID_INFRA_MARKERS = (
    "docker compose command failed",
    "dependency failed to start",
    "grafana not reachable",
    "container ",
    "environment ",
    "network ",
    "image ",
)
ROOT = Path(__file__).resolve().parent.parent

JsonDict = dict[str, Any]
TrialArtifactStatus = Literal["complete", "retryable", "stale", "corrupt"]


def pretty_model(name: str) -> str:
    return MODEL_LABELS.get(name, name)


def pretty_variant(name: str, reasoning_effort: str) -> str:
    label = pretty_model(name)
    if reasoning_effort == "off":
        return label
    return f"{label} ({reasoning_effort})"


def load_task_categories(tasks_dir: Path) -> dict[str, str]:
    categories: dict[str, str] = {}
    for toml_path in tasks_dir.rglob("task.toml"):
        task_name = toml_path.parent.name
        try:
            with open(toml_path, "rb") as file_obj:
                data = tomllib.load(file_obj)
        except Exception:
            categories[task_name] = "unknown"
            continue
        metadata = data.get("metadata") or {}
        categories[task_name] = str(metadata.get("category", "unknown"))
    return categories


def load_job_config(job_dir: Path) -> JsonDict | None:
    config_path = job_dir / "config.json"
    try:
        data = json.loads(config_path.read_text())
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def job_tasks_dir(job_dir: Path) -> Path | None:
    config = load_job_config(job_dir)
    if not config:
        return None
    datasets = config.get("datasets")
    if not isinstance(datasets, list):
        return None
    for dataset in datasets:
        if not isinstance(dataset, dict):
            continue
        raw_path = dataset.get("path")
        if isinstance(raw_path, str) and raw_path:
            tasks_dir = normalize_repo_path(ROOT, raw_path)
            if tasks_dir.exists():
                return tasks_dir
    return None


def job_task_count(job_dir: Path) -> int | None:
    tasks_dir = job_tasks_dir(job_dir)
    if tasks_dir is None:
        return None
    return sum(1 for path in tasks_dir.iterdir() if path.is_dir() and not path.name.startswith("."))


def job_n_attempts(job_dir: Path) -> int | None:
    config = load_job_config(job_dir)
    if not config:
        return None
    value = config.get("n_attempts")
    return value if isinstance(value, int) and value > 0 else None


def job_expected_trial_count(job_dir: Path) -> int | None:
    n_tasks = job_task_count(job_dir)
    n_attempts = job_n_attempts(job_dir)
    if n_tasks is None or n_attempts is None:
        return None
    return n_tasks * n_attempts


def trial_task_name(trial_dir: Path, trial: JsonDict | None) -> str:
    if trial is not None:
        task_name = trial.get("task_name")
        if isinstance(task_name, str) and task_name:
            return task_name
    return trial_dir.name.split("__", 1)[0]


def is_interrupted_trial(trial: JsonDict) -> bool:
    exc = trial.get("exception_info")
    if not isinstance(exc, dict):
        return False
    if exc.get("exception_type") == "CancelledError":
        return True

    message = exc.get("exception_message")
    if not isinstance(message, str):
        return False

    return message.strip() in {"CancelledError", "asyncio.exceptions.CancelledError"}


def is_missing_reward_trial(trial: JsonDict) -> bool:
    exc = trial.get("exception_info")
    if not isinstance(exc, dict):
        return False

    if exc.get("exception_type") == "RewardFileNotFoundError":
        return True

    message = exc.get("exception_message")
    if not isinstance(message, str):
        return False

    return message.strip().lower() in {"rewardfilenotfounderror", "no reward file found"}


def is_agent_timeout_trial(trial: JsonDict) -> bool:
    exc = trial.get("exception_info")
    if not isinstance(exc, dict):
        return False

    if exc.get("exception_type") == "AgentTimeoutError":
        return True

    message = exc.get("exception_message")
    if not isinstance(message, str):
        return False

    return message.startswith("Agent execution timed out after ")


def is_nonzero_agent_exit_trial(trial: JsonDict) -> bool:
    exc = trial.get("exception_info")
    if not isinstance(exc, dict):
        return False

    if exc.get("exception_type") == "NonZeroAgentExitCodeError":
        return True

    message = exc.get("exception_message")
    if not isinstance(message, str):
        return False

    return message.startswith("Agent exited with code ")


def classify_trial_artifact(
    trial_dir: Path,
    trial: JsonDict | None,
    *,
    task_checksums: dict[str, str] | None = None,
) -> TrialArtifactStatus:
    if trial is None:
        result_path = trial_dir / "result.json"
        if not result_path.exists():
            return "retryable"
        return "corrupt"

    if not trial.get("agent_info") or not trial_task_name(trial_dir, trial):
        return "corrupt"

    if task_checksums is not None:
        task_name = trial_task_name(trial_dir, trial)
        current_checksum = task_checksums.get(task_name)
        stored_checksum = trial.get("task_checksum")
        if current_checksum is None:
            return "stale"
        if not isinstance(stored_checksum, str) or stored_checksum != current_checksum:
            return "stale"

    if (
        is_invalid_infra_trial(trial)
        or is_interrupted_trial(trial)
        or is_missing_reward_trial(trial)
    ):
        return "retryable"

    return "complete"


def load_trials(jobs_dir: Path) -> list[JsonDict]:
    trials: list[JsonDict] = []
    result_paths: list[Path] = []
    for root, dirs, files in os.walk(jobs_dir, followlinks=True):
        # Ignore hidden/staging suite folders so archived reruns do not affect reports.
        dirs[:] = [name for name in dirs if not name.startswith(".")]
        for name in files:
            if name == "result.json":
                result_paths.append(Path(root) / name)
    for result_path in sorted(result_paths):
        # Skip job-level result.json (sits directly in the job dir, depth=1)
        parts = result_path.relative_to(jobs_dir).parts
        if len(parts) < 2:
            continue
        try:
            data = json.loads(result_path.read_text())
        except Exception:
            continue
        if not data.get("agent_info") or not data.get("task_name"):
            continue
        data["__result_path"] = str(result_path.resolve())
        trials.append(data)
    return trials


def parse_datetime(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def agent_seconds(trial: JsonDict) -> float:
    ae = trial.get("agent_execution") or {}
    t0 = parse_datetime(ae.get("started_at"))
    t1 = parse_datetime(ae.get("finished_at"))
    if t0 and t1:
        return (t1 - t0).total_seconds()
    return 0.0


def is_invalid_infra_trial(trial: JsonDict) -> bool:
    if is_nonzero_agent_exit_trial(trial):
        return True

    if trial.get("agent_result") is not None:
        return False
    if trial.get("agent_execution") is not None:
        return False
    if trial.get("verifier") is not None:
        return False
    exc = trial.get("exception_info")
    if not exc:
        return False
    text = json.dumps(exc).lower() if isinstance(exc, dict) else str(exc).lower()
    return any(marker in text for marker in INVALID_INFRA_MARKERS)


def trial_reasoning_effort(trial: JsonDict) -> str:
    metadata = (trial.get("agent_result") or {}).get("metadata") or {}
    reasoning_effort = metadata.get("reasoning_effort")
    if isinstance(reasoning_effort, str) and reasoning_effort:
        return reasoning_effort

    result_path = trial.get("__result_path")
    if not result_path:
        return "off"

    trial_config_path = Path(result_path).parent / "config.json"
    try:
        config = json.loads(trial_config_path.read_text())
    except Exception:
        return "off"

    agent_config = config.get("agent") or {}
    model_options = agent_config.get("model_options") or {}
    agent_kwargs = agent_config.get("kwargs") or {}
    config_reasoning_effort = model_options.get("reasoning_effort")
    if not isinstance(config_reasoning_effort, str) or not config_reasoning_effort:
        config_reasoning_effort = agent_kwargs.get("reasoning_effort")
    if isinstance(config_reasoning_effort, str) and config_reasoning_effort:
        return config_reasoning_effort
    return "off"


def variant_key(trial: JsonDict) -> tuple[str, str]:
    model = trial["agent_info"]["model_info"]["name"]
    return model, trial_reasoning_effort(trial)


def rubric_passed(grading: JsonDict) -> bool:
    scored_criteria: list[float] = []
    for key, val in grading.items():
        if key in (
            "score",
            "checks_passed",
            "rubric_passed",
            "validators_passed",
            "criteria_passed",
        ) or key.startswith("explanation:"):
            continue
        try:
            scored_criteria.append(float(val))
        except TypeError, ValueError:
            continue

    if scored_criteria:
        return all(score >= 1.0 for score in scored_criteria)

    return float(grading.get("score", 0.0)) >= 1.0


def grading_counts_as_pass(trial: JsonDict, grading: JsonDict) -> bool:
    return rubric_passed(grading) and not is_agent_timeout_trial(trial)


def reward_counts_as_pass(trial: JsonDict) -> bool:
    rewards = (trial.get("verifier_result") or {}).get("rewards") or {}
    reward = rewards.get("reward")
    return reward == 1.0 and not is_agent_timeout_trial(trial)


def escape_html(text: object) -> str:
    return html.escape(str(text))


def agent_result_metrics(result: JsonDict) -> tuple[float, int, int, int]:
    agent_result = result.get("agent_result") or {}
    n_input_tokens = agent_result.get("n_input_tokens") or 0
    n_cache_tokens = agent_result.get("n_cache_tokens") or 0
    n_output_tokens = agent_result.get("n_output_tokens") or 0
    cost_usd = agent_result.get("cost_usd")
    if cost_usd is None or float(cost_usd) <= 0:
        model_info = (result.get("agent_info") or {}).get("model_info") or {}
        model_name = model_info.get("name") or ""
        estimated_cost_usd = estimate_cost_usd(
            str(model_name),
            int(n_input_tokens),
            int(n_cache_tokens),
            int(n_output_tokens),
        )
        if estimated_cost_usd is not None:
            cost_usd = estimated_cost_usd
    if cost_usd is None:
        cost_usd = 0.0
    return float(cost_usd), int(n_input_tokens), int(n_cache_tokens), int(n_output_tokens)


def count_tool_calls(transcript: list[JsonDict]) -> int:
    return sum(
        1
        for msg in transcript
        if msg.get("type") == "assistant" and isinstance(msg.get("message"), dict)
        for block in msg["message"].get("content", [])
        if isinstance(block, dict) and block.get("type") == "tool_use"
    )


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m{secs:02d}s"


def format_cost(cost_usd: float) -> str:
    if cost_usd < 0.01:
        return f"${cost_usd * 1000:.2f}m"
    return f"${cost_usd:.3f}"


def format_compact_count(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 1_000_000:
        compact = f"{value / 1_000_000:.1f}".rstrip("0").rstrip(".")
        return f"{compact}M"
    if abs_value >= 1_000:
        compact = f"{value / 1_000:.1f}".rstrip("0").rstrip(".")
        return f"{compact}k"
    return f"{value:.0f}"


def score_color_class(score: float) -> str:
    if score >= HIGH_SCORE_THRESHOLD:
        return "text-green-600"
    if score >= MEDIUM_SCORE_THRESHOLD:
        return "text-yellow-600"
    return "text-red-600"


def score_bar_class(score: float) -> str:
    if score >= HIGH_SCORE_THRESHOLD:
        return "bg-green-500"
    if score >= MEDIUM_SCORE_THRESHOLD:
        return "bg-yellow-500"
    return "bg-red-500"


# ── Shared helpers used by report, run_report, compare_report ─────────────


def trial_to_row(result: JsonDict, transcript: list[JsonDict] | None = None) -> Any:
    """Build a TrialRow from a flat result dict (+ optional transcript for tool call count)."""
    reward = ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    cost_usd, n_input_tokens, n_cache_tokens, n_output_tokens = agent_result_metrics(result)
    trial_dir = Path(result["__result_path"]).parent if result.get("__result_path") else Path(".")
    status = classify_trial_artifact(trial_dir, result)

    return {
        "task_name": result.get("task_name", "unknown"),
        "score": float(reward) if reward is not None else None,
        "cost_usd": cost_usd,
        "agent_secs": agent_seconds(result),
        "n_input_tokens": n_input_tokens,
        "n_output_tokens": n_output_tokens,
        "n_cache_tokens": n_cache_tokens,
        "tool_calls": count_tool_calls(transcript) if transcript else 0,
        "invalid_infra": status != "complete",
        "counts_as_pass": reward_counts_as_pass(result),
    }


def resolve_tasks_dir(job_dir: Path, tasks_dir: Path | None = None) -> Path | None:
    """Resolve the tasks directory from explicit arg, job config, or well-known paths."""
    if tasks_dir is not None and tasks_dir.exists():
        return tasks_dir
    from_config = job_tasks_dir(job_dir)
    if from_config is not None:
        return from_config
    for candidate in [
        ROOT / "tasks",
        job_dir.parent.parent / "tasks",
        job_dir.parent / "tasks",
    ]:
        if candidate.exists():
            return candidate
    return None


def trial_model_display(first_result: JsonDict, result_path: Path | None = None) -> str:
    """Extract a human-readable model display string from the first trial result."""
    model_info = (first_result.get("agent_info") or {}).get("model_info") or {}
    model_name = model_info.get("name") or ""
    if not model_name:
        meta = (first_result.get("agent_result") or {}).get("metadata") or {}
        model_name = meta.get("model", "unknown") if isinstance(meta, dict) else "unknown"

    enriched = first_result
    if result_path is not None:
        enriched = {**first_result, "__result_path": str(result_path)}
    reasoning_effort = trial_reasoning_effort(enriched)
    return pretty_variant(model_name, reasoning_effort)
