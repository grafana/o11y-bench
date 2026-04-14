"""Resume/repair logic: checksums, staleness detection, archival."""

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from reporting.report_data import classify_trial_artifact, is_interrupted_trial
from reporting.report_paths import normalize_repo_path

from .config import ROOT, JobSpec, load_job_config_overrides

RESUME_ARCHIVE_DIRNAME = ".resume-pruned"
PREVIEW_NAME_LIMIT = 10


@dataclass(frozen=True)
class ResumeTrialAction:
    trial_dir: Path
    reason: str


@dataclass(frozen=True)
class ResumeRepairPlan:
    job_dir: Path
    config_notes: tuple[str, ...]
    actions: tuple[ResumeTrialAction, ...]


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def task_problem_yaml_path(task_dir: Path) -> Path:
    return task_dir / "tests" / "problem.yaml"


def compute_task_checksums(tasks_dir: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    if not tasks_dir.is_dir():
        return checksums
    for task_path in sorted(tasks_dir.iterdir()):
        if task_path.is_dir() and not task_path.name.startswith("."):
            problem_yaml_path = task_problem_yaml_path(task_path)
            if problem_yaml_path.exists():
                checksums[task_path.name] = hash_file(problem_yaml_path)
    return checksums


def expected_resume_fields(spec: JobSpec) -> dict[str, Any]:
    overrides = load_job_config_overrides()
    retry = ((overrides.get("orchestrator") or {}).get("retry") or {}).get("max_retries")
    return {
        "job_name": spec.job_name,
        "jobs_dir": str(spec.jobs_dir.resolve()),
        "n_attempts": spec.n_attempts,
        "model_name": spec.model,
        "reasoning_effort": spec.reasoning_effort,
        "tasks_path": str(spec.tasks_dir.resolve()),
        "agent_name": spec.agent,
        "agent_import_path": spec.agent_import_path if not spec.agent else None,
        "environment_type": "docker",
        "override_cpus": spec.override_cpus,
        "override_memory_mb": spec.override_memory_mb,
        "override_storage_mb": spec.override_storage_mb,
        "n_concurrent_trials": spec.n_concurrent,
        "retry_max_retries": retry,
    }


def normalize_saved_path(value: Any) -> Any:
    if isinstance(value, str) and value:
        return str(normalize_repo_path(ROOT, value))
    return value


def load_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError, OSError:
        return None
    return payload if isinstance(payload, dict) else None


def ensure_environment_config(config: dict[str, Any]) -> dict[str, Any]:
    environment = config.get("environment")
    if isinstance(environment, dict):
        return environment
    environment = {}
    config["environment"] = environment
    return environment


def saved_resume_fields(config: dict[str, Any]) -> dict[str, Any]:
    agent = (config.get("agents") or [None])[0] or {}
    datasets = (config.get("datasets") or [None])[0] or {}
    environment = config.get("environment") or {}
    orchestrator = config.get("orchestrator") or {}
    retry = config.get("retry") or orchestrator.get("retry") or {}
    return {
        "job_name": config.get("job_name"),
        "jobs_dir": normalize_saved_path(config.get("jobs_dir")),
        "n_attempts": config.get("n_attempts"),
        "model_name": agent.get("model_name"),
        "reasoning_effort": ((agent.get("kwargs") or {}).get("reasoning_effort") or "off"),
        "tasks_path": normalize_saved_path(datasets.get("path")),
        "agent_name": agent.get("name"),
        "agent_import_path": agent.get("import_path"),
        "environment_type": environment.get("type"),
        "override_cpus": environment.get("override_cpus"),
        "override_memory_mb": environment.get("override_memory_mb"),
        "override_storage_mb": environment.get("override_storage_mb"),
        "n_concurrent_trials": config.get("n_concurrent_trials")
        or orchestrator.get("n_concurrent_trials"),
        "retry_max_retries": retry.get("max_retries"),
    }


def semantic_resume_mismatches(
    saved_fields: dict[str, Any], expected_fields: dict[str, Any]
) -> list[str]:
    allowed_drift = {"jobs_dir", "n_concurrent_trials", "tasks_path"}
    return sorted(
        key
        for key, expected_value in expected_fields.items()
        if key not in allowed_drift and saved_fields.get(key) != expected_value
    )


def patch_job_concurrency_config(
    job_dir: Path, expected_n_concurrent: int, *, dry_run: bool = False
) -> str | None:
    config_path = job_dir / "config.json"
    if not config_path.exists():
        return None
    config = json.loads(config_path.read_text())
    current_n_concurrent = config.get("n_concurrent_trials")
    use_top_level = current_n_concurrent is not None
    orchestrator = config.setdefault("orchestrator", {})
    if current_n_concurrent is None:
        current_n_concurrent = orchestrator.get("n_concurrent_trials")
    if current_n_concurrent == expected_n_concurrent:
        return None
    if dry_run:
        return f"Would patch {job_dir.name} config for resume: n_concurrent {current_n_concurrent} -> {expected_n_concurrent}"
    if use_top_level:
        config["n_concurrent_trials"] = expected_n_concurrent
    else:
        orchestrator["n_concurrent_trials"] = expected_n_concurrent
    config_path.write_text(json.dumps(config, indent=4) + "\n")
    return f"Patched {job_dir.name} config for resume: n_concurrent {current_n_concurrent} -> {expected_n_concurrent}"


def patch_job_paths_for_resume(
    job_dir: Path, expected_jobs_dir: Path, *, dry_run: bool = False
) -> list[str]:
    notes: list[str] = []
    config_path = job_dir / "config.json"
    if not config_path.exists():
        return notes

    config = json.loads(config_path.read_text())
    expected_jobs_dir_text = str(expected_jobs_dir.resolve())
    current_jobs_dir = normalize_saved_path(config.get("jobs_dir"))
    if current_jobs_dir != expected_jobs_dir_text:
        verb = "Would patch" if dry_run else "Patched"
        if not dry_run:
            config["jobs_dir"] = expected_jobs_dir_text
            config_path.write_text(json.dumps(config, indent=4) + "\n")
        notes.append(
            f"{verb} {job_dir.name} config for resume: jobs_dir {current_jobs_dir} -> {expected_jobs_dir_text}"
        )
    current_trials_dir = str(job_dir.resolve())
    affected_trial_configs = 0
    skipped_trial_configs = 0
    for trial_dir in sorted(
        path for path in job_dir.iterdir() if path.is_dir() and "__" in path.name
    ):
        trial_config_path = trial_dir / "config.json"
        if not trial_config_path.exists():
            continue
        trial_config = load_json_object(trial_config_path)
        if trial_config is None:
            skipped_trial_configs += 1
            continue
        if normalize_saved_path(trial_config.get("trials_dir")) == current_trials_dir:
            continue
        if not dry_run:
            trial_config["trials_dir"] = current_trials_dir
            trial_config_path.write_text(json.dumps(trial_config, indent=4) + "\n")
        affected_trial_configs += 1

    if affected_trial_configs:
        verb = "Would patch" if dry_run else "Patched"
        notes.append(
            f"{verb} {affected_trial_configs} trial config(s) for resume: trials_dir -> {current_trials_dir}"
        )
    if skipped_trial_configs:
        verb = "Would skip" if dry_run else "Skipped"
        notes.append(
            f"{verb} {skipped_trial_configs} unreadable trial config(s) {'during' if dry_run else 'while repairing'} {('resume repair' if dry_run else job_dir.name)}"
        )
    return notes


def patch_environment_delete(job_dir: Path, *, dry_run: bool = False) -> list[str]:
    expected_delete = (load_job_config_overrides().get("environment") or {}).get("delete")
    if expected_delete is None:
        return []
    updated_configs = 0
    for config_path in [job_dir / "config.json", *sorted(job_dir.glob("*__*/config.json"))]:
        config = load_json_object(config_path)
        if config is None:
            continue
        environment = ensure_environment_config(config)
        if environment.get("delete") == expected_delete:
            continue
        updated_configs += 1
        if dry_run:
            continue
        environment["delete"] = expected_delete
        config_path.write_text(json.dumps(config, indent=4) + "\n")
    if not updated_configs:
        return []
    verb = "Would patch" if dry_run else "Patched"
    return [
        f"{verb} {updated_configs} config(s) for resume: environment.delete -> {expected_delete}"
    ]


def make_resume_archive_dir(job_dir: Path) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    archive_dir = job_dir / RESUME_ARCHIVE_DIRNAME / stamp
    archive_dir.mkdir(parents=True, exist_ok=True)
    return archive_dir


def archive_trial_dir(trial_dir: Path, archive_dir: Path, *, reason: str) -> bool:
    if not trial_dir.exists():
        return False

    reason_dir = archive_dir / reason
    reason_dir.mkdir(parents=True, exist_ok=True)
    target = reason_dir / trial_dir.name
    suffix = 1
    while target.exists():
        target = reason_dir / f"{trial_dir.name}-{suffix}"
        suffix += 1

    try:
        trial_dir.rename(target)
    except FileNotFoundError:
        return False
    return True


def plan_job_dir_for_resume(
    job_dir: Path,
    expected_fields: dict[str, Any],
    task_checksums: dict[str, str] | None = None,
) -> ResumeRepairPlan:
    config_path = job_dir / "config.json"
    if not config_path.exists():
        return ResumeRepairPlan(job_dir=job_dir, config_notes=(), actions=())

    config = json.loads(config_path.read_text())
    saved_fields = saved_resume_fields(config)
    mismatches = semantic_resume_mismatches(saved_fields, expected_fields)
    if mismatches:
        mismatch_text = ", ".join(mismatches)
        raise RuntimeError(
            f"Cannot auto-resume {job_dir.name}: incompatible saved config fields: {mismatch_text}"
        )

    config_notes_list = patch_job_paths_for_resume(
        job_dir, Path(expected_fields["jobs_dir"]), dry_run=True
    )
    concurrency_note = patch_job_concurrency_config(
        job_dir, expected_fields["n_concurrent_trials"], dry_run=True
    )
    if concurrency_note:
        config_notes_list.append(concurrency_note)
    config_notes_list.extend(patch_environment_delete(job_dir, dry_run=True))
    config_notes = tuple(config_notes_list)

    actions: list[ResumeTrialAction] = []
    for trial_dir in sorted(
        path for path in job_dir.iterdir() if path.is_dir() and "__" in path.name
    ):
        result_path = trial_dir / "result.json"
        if not result_path.exists():
            actions.append(ResumeTrialAction(trial_dir=trial_dir, reason="incomplete"))
            continue

        try:
            trial_result = json.loads(result_path.read_text())
        except json.JSONDecodeError:
            actions.append(ResumeTrialAction(trial_dir=trial_dir, reason="corrupt"))
            continue

        status = classify_trial_artifact(trial_dir, trial_result, task_checksums=task_checksums)
        match status:
            case "retryable":
                reason = "interrupted" if is_interrupted_trial(trial_result) else "retryable"
                actions.append(ResumeTrialAction(trial_dir=trial_dir, reason=reason))
            case "stale":
                actions.append(ResumeTrialAction(trial_dir=trial_dir, reason="stale"))
            case "corrupt":
                actions.append(ResumeTrialAction(trial_dir=trial_dir, reason="corrupt"))
            case "complete":
                pass
            case _:
                raise AssertionError(f"Unhandled resume status {status!r}")

    return ResumeRepairPlan(
        job_dir=job_dir,
        config_notes=config_notes,
        actions=tuple(actions),
    )


def format_resume_plan(plan: ResumeRepairPlan) -> list[str]:
    lines = list(plan.config_notes)

    if not plan.actions:
        lines.append(f"Resume preview {plan.job_dir.name}: no trial dirs need rerun")
        return lines

    lines.append(
        f"Resume preview {plan.job_dir.name}: {len(plan.actions)} trial dir(s) would be archived and rerun"
    )
    grouped: dict[str, list[str]] = {}
    for action in plan.actions:
        grouped.setdefault(action.reason, []).append(action.trial_dir.name)

    for reason in sorted(grouped):
        names = grouped[reason]
        preview_names = ", ".join(names[:PREVIEW_NAME_LIMIT])
        hidden_count = len(names) - PREVIEW_NAME_LIMIT
        if hidden_count > 0:
            preview_names = f"{preview_names}, ... +{hidden_count} more"
        lines.append(f"  {reason}: {len(names)} ({preview_names})")

    lines.append(f"  archive target: {plan.job_dir / RESUME_ARCHIVE_DIRNAME}/<timestamp>/...")
    return lines


def repair_job_dir_for_resume(
    job_dir: Path,
    expected_fields: dict[str, Any],
    task_checksums: dict[str, str] | None = None,
) -> list[str]:
    notes: list[str] = []
    if not (job_dir / "config.json").exists():
        return notes

    plan = plan_job_dir_for_resume(job_dir, expected_fields, task_checksums)
    notes.extend(patch_job_paths_for_resume(job_dir, Path(expected_fields["jobs_dir"])))
    concurrency_note = patch_job_concurrency_config(job_dir, expected_fields["n_concurrent_trials"])
    if concurrency_note:
        notes.append(concurrency_note)
    notes.extend(patch_environment_delete(job_dir))

    if not plan.actions:
        return notes

    archive_dir = make_resume_archive_dir(job_dir)
    archived_counts: dict[str, int] = {
        "corrupt": 0,
        "incomplete": 0,
        "interrupted": 0,
        "retryable": 0,
        "stale": 0,
    }
    for action in plan.actions:
        archived_counts[action.reason] += int(
            archive_trial_dir(action.trial_dir, archive_dir, reason=action.reason)
        )

    archived_incomplete = archived_counts["incomplete"] + archived_counts["corrupt"]
    if archived_incomplete:
        notes.append(
            f"Repaired {job_dir.name}: archived {archived_incomplete} incomplete/corrupt trial dirs"
        )
    if archived_counts["retryable"]:
        notes.append(
            f"Repaired {job_dir.name}: archived {archived_counts['retryable']} retryable trial dirs"
        )
    if archived_counts["interrupted"]:
        notes.append(
            f"Repaired {job_dir.name}: archived {archived_counts['interrupted']} interrupted trial dirs"
        )
    if archived_counts["stale"]:
        notes.append(
            f"Repaired {job_dir.name}: archived {archived_counts['stale']} stale trials (task changed)"
        )
    notes.append(f"Preserved repaired trials under {archive_dir.relative_to(job_dir)}")
    return notes
