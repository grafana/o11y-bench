"""Core execution engine: single jobs and full suite."""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import yaml

from grading.env_context import VerifierContext
from grading.judge import grading_model
from grading.models import Problem
from grading.transcript_parser import parse_transcript
from grading.verifier import GradeResult, grade, read_truth, write_outputs
from reporting import report as suite_report
from reporting import run_report
from reporting.report_paths import (
    run_report_output_path,
)

from .agento11y_publish import Agento11yLivePublisher, Agento11yPublishResult
from .config import (
    PROVIDERS,
    ROOT,
    STANDARD_SUITE,
    JobSpec,
    SuiteOpts,
    build_suite_job_spec,
    provider_variants,
    resolve_suite_dir,
)
from .harbor import (
    build_command,
    build_resume_command,
    run_preflight,
    write_builtin_prompt_template,
)
from .harbor import run as run_harbor
from .regrade_stack import (
    problem_reads_final_state,
    problem_requires_live_stack,
    running_regrade_stack,
)
from .resume import (
    ResumeTrialAction,
    compute_task_checksums,
    expected_resume_fields,
    format_resume_plan,
    plan_job_dir_for_resume,
    repair_job_dir_for_resume,
)
from .scenario_clock import bound_scenario_time, resolve_scenario_time


@dataclass
class JobResult:
    status: str  # "fresh", "up_to_date", "repair_only", "ran", "dry_run"
    job_name: str
    needs_repair: bool = False
    needs_harbor: bool = False
    repair_actions: int = 0
    missing_trials: int = 0
    harbor_exit_code: int | None = None
    report_path: Path | None = None
    agento11y_result: Agento11yPublishResult | None = None


def _print_job_summary(result: JobResult) -> None:
    details: list[str] = []
    if result.needs_repair and result.repair_actions:
        details.append(f"repaired {result.repair_actions}")
    if result.needs_harbor and result.missing_trials:
        details.append(f"ran {result.missing_trials} trial(s)")

    if result.status == "up_to_date":
        message = "up to date"
    elif result.status == "dry_run":
        message = ", ".join(details) if details else "up to date"
        message = f"dry run, {message}"
    else:
        message = ", ".join(details) if details else result.status

    if result.report_path:
        message = f"{message} | report: {result.report_path}"
    if result.agento11y_result:
        message = f"{message} | Agent Observability: {result.agento11y_result.url}"

    print(f"{result.job_name}: {message}")


def _make_agento11y_publisher(spec: JobSpec, job_dir: Path) -> Agento11yLivePublisher:
    agent_name = os.getenv("AGENTO11Y_AGENT_NAME", "").strip()
    if not agent_name:
        agent_name = spec.agent or spec.agent_import_path.split(":", 1)[0].rsplit(".", 1)[-1]
    agent_version = os.getenv("AGENTO11Y_AGENT_VERSION", "").strip() or "unknown"
    return Agento11yLivePublisher(
        job_dir,
        spec.tasks_dir,
        model=spec.model,
        agent_name=agent_name,
        agent_version=agent_version,
        planned_trial_count=len(_selected_task_names(spec)) * spec.n_attempts,
    )


def _run_harbor_with_agento11y(
    command: list[str],
    *,
    publisher: Agento11yLivePublisher | None,
) -> tuple[int, Agento11yPublishResult | None]:
    if publisher is None:
        return run_harbor(command, forward_signals=False), None

    publisher.start()
    exit_code = 1
    error = ""
    try:
        exit_code = run_harbor(command, forward_signals=False)
    except BaseException as exc:
        error = str(exc)
        raise
    finally:
        result = publisher.finish(succeeded=exit_code == 0 and not error, error=error)
    return exit_code, result


def _selected_task_names(spec: JobSpec) -> list[str]:
    if spec.task_names:
        return list(spec.task_names)
    if not spec.tasks_dir.is_dir():
        return []
    return sorted(
        d.name for d in spec.tasks_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    )


def _count_usable_trials(
    job_dir: Path, task_names: list[str], actions: tuple[ResumeTrialAction, ...]
) -> dict[str, int]:
    """Count existing trials per task, subtracting those queued for archival."""
    counts = {name: 0 for name in task_names}
    selected = set(task_names)
    if not job_dir.exists():
        return counts

    for trial_dir in job_dir.iterdir():
        if not trial_dir.is_dir() or "__" not in trial_dir.name:
            continue
        result_path = trial_dir / "result.json"
        if not result_path.exists():
            continue
        try:
            task_name = json.loads(result_path.read_text()).get("task_name")
        except json.JSONDecodeError, OSError:
            continue
        if isinstance(task_name, str) and task_name in selected:
            counts[task_name] += 1

    action_counts: dict[str, int] = {}
    for action in actions:
        name = action.trial_dir.name.split("__", 1)[0]
        action_counts[name] = action_counts.get(name, 0) + 1

    return {name: max(0, count - action_counts.get(name, 0)) for name, count in counts.items()}


def _stamp_checksums(job_dir: Path, task_checksums: dict[str, str]) -> None:
    """Write current task checksums into trial result.json files."""
    for trial_dir in sorted(d for d in job_dir.iterdir() if d.is_dir() and "__" in d.name):
        result_path = trial_dir / "result.json"
        if not result_path.exists():
            continue
        try:
            payload = json.loads(result_path.read_text())
        except json.JSONDecodeError:
            continue
        task_name = payload.get("task_name")
        if isinstance(task_name, str) and task_name in task_checksums:
            checksum = task_checksums[task_name]
            if payload.get("task_checksum") != checksum:
                payload["task_checksum"] = checksum
                result_path.write_text(json.dumps(payload, indent=4) + "\n")


def _read_json_dict(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _write_json_dict(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=4) + "\n")


def _portable_artifact_path(path: Path, fallback: Path) -> str:
    try:
        portable = path.resolve().relative_to(ROOT)
    except ValueError:
        portable = fallback
    return portable.as_posix()


def _portable_jobs_dir(job_dir: Path) -> str:
    return _portable_artifact_path(job_dir.parent, Path("jobs") / job_dir.parent.name)


def _portable_job_dir(job_dir: Path) -> str:
    return _portable_artifact_path(job_dir, Path("jobs") / job_dir.parent.name / job_dir.name)


def _portable_task_root(tasks_dir: Path) -> str:
    return _portable_artifact_path(tasks_dir, Path("tasks"))


def _portable_task_path(tasks_dir: Path, task_name: str) -> str:
    return _portable_artifact_path(tasks_dir / task_name, Path("tasks") / task_name)


def _configured_task_name(payload: dict[str, Any]) -> str | None:
    task = payload.get("task")
    if isinstance(task, dict):
        task_path = task.get("path")
        if isinstance(task_path, str) and task_path:
            return Path(task_path).name
    return None


def _trial_task_name(trial_dir: Path, task_name: str | None) -> str:
    if isinstance(task_name, str) and task_name:
        return task_name
    return trial_dir.name.split("__", 1)[0]


def _resolved_trial_task_name(
    trial_dir: Path,
    *,
    result_payload: dict[str, Any] | None = None,
    config_payload: dict[str, Any] | None = None,
) -> str:
    task_name = result_payload.get("task_name") if isinstance(result_payload, dict) else None
    if not isinstance(task_name, str) or not task_name:
        task_name = (
            _configured_task_name(config_payload) if isinstance(config_payload, dict) else None
        )
    return _trial_task_name(trial_dir, task_name)


def _sanitize_trial_config_payload(
    payload: dict[str, Any],
    *,
    job_dir: Path,
    trial_dir: Path,
    tasks_dir: Path,
    task_name: str | None,
) -> bool:
    changed = False
    task_name = _trial_task_name(trial_dir, task_name or _configured_task_name(payload))
    portable_job_dir = _portable_job_dir(job_dir)
    portable_task_dir = _portable_task_path(tasks_dir, task_name)

    if payload.get("trials_dir") != portable_job_dir:
        payload["trials_dir"] = portable_job_dir
        changed = True

    task = payload.get("task")
    if isinstance(task, dict) and task.get("path") != portable_task_dir:
        task["path"] = portable_task_dir
        changed = True

    return changed


def _sanitize_trial_result_payload(
    payload: dict[str, Any], *, job_dir: Path, trial_dir: Path, tasks_dir: Path
) -> bool:
    changed = False
    task_name = _trial_task_name(trial_dir, payload.get("task_name"))
    portable_job_dir = _portable_job_dir(job_dir)
    portable_task_dir = _portable_task_path(tasks_dir, task_name)
    portable_trial_dir = f"{portable_job_dir}/{trial_dir.name}"
    portable_trial_uri = f"file:{portable_trial_dir}"

    if payload.get("trial_uri") != portable_trial_uri:
        payload["trial_uri"] = portable_trial_uri
        changed = True

    config_payload = payload.get("config")
    if isinstance(config_payload, dict):
        if _sanitize_trial_config_payload(
            config_payload,
            job_dir=job_dir,
            trial_dir=trial_dir,
            tasks_dir=tasks_dir,
            task_name=task_name,
        ):
            changed = True

    task = payload.get("task")
    if isinstance(task, dict) and task.get("path") != portable_task_dir:
        task["path"] = portable_task_dir
        changed = True

    task_id = payload.get("task_id")
    if isinstance(task_id, dict) and task_id.get("path") != portable_task_dir:
        task_id["path"] = portable_task_dir
        changed = True

    return changed


def _extract_gcx_version(job_dir: Path) -> str | None:
    """Read the gcx version from the first trial's agent_info, if present."""
    for trial_dir in _iter_trial_dirs(job_dir):
        result = _read_json_dict(trial_dir / "result.json")
        if not isinstance(result, dict):
            continue
        version = (result.get("agent_info") or {}).get("version")
        if isinstance(version, str):
            for line in version.splitlines():
                if line.startswith("gcx version"):
                    return line
    return None


def _uses_gcx_agent(config_payload: dict[str, Any]) -> bool:
    agents = config_payload.get("agents")
    if not isinstance(agents, list):
        return False
    return any(
        isinstance(a, dict) and "gcx_opencode_agent" in (a.get("import_path") or "") for a in agents
    )


def _sanitize_job_artifacts(job_dir: Path, tasks_dir: Path) -> None:
    config_path = job_dir / "config.json"
    if config_path.exists():
        config_payload = _read_json_dict(config_path)
        if isinstance(config_payload, dict):
            changed = False
            portable_jobs_dir = _portable_jobs_dir(job_dir)
            portable_tasks_dir = _portable_task_root(tasks_dir)
            if config_payload.get("jobs_dir") != portable_jobs_dir:
                config_payload["jobs_dir"] = portable_jobs_dir
                changed = True
            datasets = config_payload.get("datasets")
            if isinstance(datasets, list):
                for dataset in datasets:
                    if not isinstance(dataset, dict):
                        continue
                    if dataset.get("path") != portable_tasks_dir:
                        dataset["path"] = portable_tasks_dir
                        changed = True
            # get gcx_version from one of the tasks and put it in the job-level config.json
            if "gcx_version" not in config_payload and _uses_gcx_agent(config_payload):
                gcx_version = _extract_gcx_version(job_dir)
                if gcx_version:
                    config_payload["gcx_version"] = gcx_version
                    changed = True
            if changed:
                _write_json_dict(config_path, config_payload)

    for trial_dir in _iter_trial_dirs(job_dir):
        trial_config_path = trial_dir / "config.json"
        result_path = trial_dir / "result.json"
        trial_config = _read_json_dict(trial_config_path) if trial_config_path.exists() else None
        result_payload = _read_json_dict(result_path) if result_path.exists() else None
        task_name = _resolved_trial_task_name(
            trial_dir,
            result_payload=result_payload,
            config_payload=trial_config,
        )
        if isinstance(trial_config, dict) and _sanitize_trial_config_payload(
            trial_config,
            job_dir=job_dir,
            trial_dir=trial_dir,
            tasks_dir=tasks_dir,
            task_name=task_name,
        ):
            _write_json_dict(trial_config_path, trial_config)

        if isinstance(result_payload, dict) and _sanitize_trial_result_payload(
            result_payload, job_dir=job_dir, trial_dir=trial_dir, tasks_dir=tasks_dir
        ):
            _write_json_dict(result_path, result_payload)


def finalize_job_dir(job_dir: Path, tasks_dir: Path, task_checksums: dict[str, str]) -> Path | None:
    """Stamp checksums and generate per-job HTML report."""
    if not job_dir.exists():
        return None
    _stamp_checksums(job_dir, task_checksums)
    _sanitize_job_artifacts(job_dir, tasks_dir)
    if not run_report.load_trials(job_dir):
        return None
    report_path = run_report_output_path(job_dir)
    run_report.write_report(job_dir, tasks_dir=tasks_dir, output=report_path)
    return report_path


def _iter_job_dirs(target_dir: Path) -> list[Path]:
    if (target_dir / "config.json").exists():
        return [target_dir]
    return sorted(
        path for path in target_dir.iterdir() if path.is_dir() and (path / "config.json").exists()
    )


def _iter_trial_dirs(job_dir: Path) -> list[Path]:
    return sorted(path for path in job_dir.iterdir() if path.is_dir() and "__" in path.name)


def _write_regrade_outputs(
    trial_dir: Path, result: GradeResult, *, task_checksum: str | None
) -> None:
    write_outputs(trial_dir / "verifier", result)
    result_path = trial_dir / "result.json"
    payload = json.loads(result_path.read_text())
    if task_checksum:
        payload["task_checksum"] = task_checksum
    payload.setdefault("verifier_result", {}).setdefault("rewards", {})["reward"] = result.score
    now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    payload["verifier"] = {"started_at": now, "finished_at": now}
    result_path.write_text(json.dumps(payload, indent=4) + "\n")


def regrade_job_dir(
    job_dir: Path,
    *,
    tasks_dir: Path,
    task_checksums: dict[str, str],
    quiet: bool = False,
    offline: bool = False,
    n_concurrent: int = 1,
) -> Path | None:
    model = grading_model()
    scenario_time_iso = resolve_scenario_time()

    updated = 0
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    trials_by_task: dict[str, list[Path]] = {}
    for trial_dir in _iter_trial_dirs(job_dir):
        result_path = trial_dir / "result.json"
        task_name = (
            json.loads(result_path.read_text()).get("task_name") if result_path.exists() else None
        )
        if isinstance(task_name, str) and task_name:
            trials_by_task.setdefault(task_name, []).append(trial_dir)
        else:
            skipped.append({"trial": trial_dir.name, "reason": "no task name in result.json"})

    def regrade_task(task_name: str, trial_dirs: list[Path]) -> tuple[int, list[Any], list[Any]]:
        task_rows: list[dict[str, Any]] = []
        task_skipped: list[dict[str, str]] = []
        problem_path = tasks_dir / task_name / "tests" / "problem.yaml"
        if not problem_path.exists():
            task_skipped = [
                {"trial": d.name, "reason": f"task not in {tasks_dir}"} for d in trial_dirs
            ]
            return 0, task_rows, task_skipped
        problem = Problem(**yaml.safe_load(problem_path.read_text()))
        task_offline = offline or problem_reads_final_state(problem)
        regrade_group = partial(
            _regrade_trial_group,
            trial_dirs=trial_dirs,
            problem=problem,
            model=model,
            task_checksum=task_checksums.get(task_name),
            offline=task_offline,
            rows=task_rows,
            skipped=task_skipped,
        )
        if task_offline or not problem_requires_live_stack(problem):
            return regrade_group(ctx=None), task_rows, task_skipped
        with running_regrade_stack(
            task_dir=tasks_dir / task_name,
            trial_dir=trial_dirs[0],
            scenario_time_iso=scenario_time_iso,
        ) as ctx:
            return regrade_group(ctx=ctx), task_rows, task_skipped

    # Each task boots its own stack, so tasks grade concurrently; the clock is shared.
    with bound_scenario_time(scenario_time_iso), ThreadPoolExecutor(n_concurrent) as pool:
        futures = [pool.submit(regrade_task, *item) for item in sorted(trials_by_task.items())]
        try:
            for future in futures:
                count, task_rows, task_skipped = future.result()
                updated += count
                rows += task_rows
                skipped += task_skipped
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    rows.sort(key=lambda row: row["trial"])
    skipped.sort(key=lambda row: row["trial"])

    (job_dir / "regrade.json").write_text(
        json.dumps({"model": model, "offline": offline, "rows": rows, "skipped": skipped}, indent=2)
        + "\n"
    )
    report_path = finalize_job_dir(job_dir, tasks_dir, task_checksums)
    if not quiet:
        changed = sum(1 for row in rows if row["before"] != row["after"])
        print(
            f"{job_dir.name}: regraded {updated} trial(s), {changed} item(s) changed, "
            f"{len(skipped)} skipped (see regrade.json)"
        )
        if report_path:
            print(f"Report: {report_path}")
    return report_path


def _regrade_trial_group(
    *,
    trial_dirs: list[Path],
    problem: Problem,
    model: str,
    task_checksum: str | None,
    offline: bool,
    rows: list[dict[str, Any]],
    skipped: list[dict[str, str]],
    ctx: VerifierContext | None,
) -> int:
    updated = 0
    for trial_dir in trial_dirs:
        logs_dir = trial_dir / "agent"
        if not logs_dir.exists():
            skipped.append({"trial": trial_dir.name, "reason": "no agent transcript"})
            continue
        details_path = trial_dir / "verifier" / "grading_details.json"
        before: dict[str, Any] = (
            json.loads(details_path.read_text()) if details_path.exists() else {}
        )
        truth = read_truth(trial_dir / "verifier")
        if offline and truth is None:
            skipped.append({"trial": trial_dir.name, "reason": "no verifier/truth.json"})
            continue
        result = grade(
            problem,
            parse_transcript(logs_dir),
            model,
            ctx=ctx,
            truth=truth if offline else None,
            previous=before if offline else None,
        )
        _write_regrade_outputs(trial_dir, result, task_checksum=task_checksum)
        rows.extend(
            {
                "trial": trial_dir.name,
                "task": problem.id,
                "item": item.name,
                "before": before.get(item.name),
                "after": item.score,
                "outcome": item.outcome,
                "quote": item.quote,
                "explanation": item.explanation,
            }
            for item in result.items
        )
        updated += 1
    return updated


def execute_regrade(
    target_dir: Path,
    *,
    tasks_dir: Path,
    quiet: bool = False,
    offline: bool = False,
    n_concurrent: int = 1,
) -> None:
    task_checksums = compute_task_checksums(tasks_dir)
    job_dirs = _iter_job_dirs(target_dir)
    if not job_dirs:
        raise SystemExit(f"No job dirs found under {target_dir}")

    if not offline:
        run_preflight(quiet=quiet)
    for job_dir in job_dirs:
        regrade_job_dir(
            job_dir,
            tasks_dir=tasks_dir,
            task_checksums=task_checksums,
            quiet=quiet,
            offline=offline,
            n_concurrent=n_concurrent,
        )

    if len(job_dirs) > 1:
        suite_report.write_report(target_dir, tasks_dir=tasks_dir, quiet=quiet)


def execute_job(
    spec: JobSpec,
    *,
    dry_run: bool = False,
    quiet: bool = False,
    agento11y_publish: bool = False,
) -> JobResult:
    """Single-pass: plan, repair, run harbor, finalize."""
    task_checksums = compute_task_checksums(spec.tasks_dir)
    job_dir = spec.jobs_dir / spec.job_name
    publisher = (
        _make_agento11y_publisher(spec, job_dir) if agento11y_publish and not dry_run else None
    )
    scenario_time_iso = resolve_scenario_time()
    os.environ["O11Y_SCENARIO_TIME_ISO"] = scenario_time_iso
    if spec.agent and not dry_run:
        write_builtin_prompt_template(spec, scenario_time_iso)
    task_names = _selected_task_names(spec)
    n_tasks = len(task_names)

    # Plan
    if not (job_dir / "config.json").exists():
        missing = n_tasks * spec.n_attempts
        if not quiet:
            print(
                f"{spec.job_name}: fresh, {n_tasks} task(s) x {spec.n_attempts} = {missing} trial(s)"
            )
        if dry_run:
            result = JobResult(
                "dry_run", spec.job_name, missing_trials=missing, needs_harbor=missing > 0
            )
            if quiet:
                _print_job_summary(result)
            return result
        fresh_harbor_exit_code, fresh_publish_result = _run_harbor_with_agento11y(
            build_command(spec, quiet=quiet), publisher=publisher
        )
        report_path = finalize_job_dir(job_dir, spec.tasks_dir, task_checksums)
        result = JobResult(
            "fresh",
            spec.job_name,
            needs_harbor=True,
            missing_trials=missing,
            harbor_exit_code=fresh_harbor_exit_code,
            report_path=report_path,
            agento11y_result=fresh_publish_result,
        )
        if quiet:
            _print_job_summary(result)
        return result

    fields = expected_resume_fields(spec)
    plan = plan_job_dir_for_resume(job_dir, fields, task_checksums)
    usable = _count_usable_trials(job_dir, task_names, plan.actions)
    missing = sum(max(0, spec.n_attempts - c) for c in usable.values())
    needs_repair = bool(plan.actions or plan.config_notes)
    needs_harbor = missing > 0

    if not quiet:
        for line in format_resume_plan(plan):
            print(line)
        if missing > 0:
            print(f"  missing trials to run: {missing}")

    if dry_run:
        result = JobResult(
            "dry_run",
            spec.job_name,
            needs_repair=needs_repair,
            needs_harbor=needs_harbor,
            repair_actions=len(plan.actions),
            missing_trials=missing,
        )
        if quiet:
            _print_job_summary(result)
        return result

    if not needs_repair and not needs_harbor:
        existing_publish_result = publisher.publish_existing() if publisher is not None else None
        result = JobResult("up_to_date", spec.job_name, agento11y_result=existing_publish_result)
        if quiet:
            _print_job_summary(result)
        return result

    if needs_repair:
        repair_notes = repair_job_dir_for_resume(job_dir, fields, task_checksums)
        if not quiet:
            for note in repair_notes:
                print(f"  {note}")

    rerun_harbor_exit_code: int | None = None
    publish_result: Agento11yPublishResult | None = None
    if needs_harbor:
        rerun_harbor_exit_code, publish_result = _run_harbor_with_agento11y(
            build_resume_command(
                str(job_dir / "config.json"),
                quiet=quiet,
                harbor_args=spec.harbor_args,
            ),
            publisher=publisher,
        )
    elif publisher is not None:
        publish_result = publisher.publish_existing()

    report_path = finalize_job_dir(job_dir, spec.tasks_dir, task_checksums)
    if report_path and not quiet:
        print(f"Report: {report_path}")

    result = JobResult(
        "ran",
        spec.job_name,
        needs_repair=needs_repair,
        needs_harbor=needs_harbor,
        repair_actions=len(plan.actions),
        missing_trials=missing,
        harbor_exit_code=rerun_harbor_exit_code,
        report_path=report_path,
        agento11y_result=publish_result,
    )
    if quiet:
        _print_job_summary(result)
    return result


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------


def _run_provider_queue(provider: str, suite_dir: Path, opts: SuiteOpts) -> list[JobResult]:
    results: list[JobResult] = []
    for model, reasoning_effort in provider_variants(provider):
        spec = build_suite_job_spec(suite_dir, provider, model, reasoning_effort, opts)
        if not opts.quiet:
            print(f"[{provider}] {model} ({reasoning_effort})")
        result = execute_job(
            spec,
            dry_run=opts.dry_run,
            quiet=opts.quiet,
            agento11y_publish=opts.agento11y_publish,
        )
        results.append(result)
        if result.harbor_exit_code and result.harbor_exit_code != 0:
            print(f"[{provider}] Harbor failed (exit {result.harbor_exit_code})", file=sys.stderr)
            break
    return results


def execute_suite(opts: SuiteOpts) -> None:
    """Run all STANDARD_SUITE variants with per-provider parallelism."""
    suite_dir = resolve_suite_dir(opts.jobs_dir, allow_resume=opts.resume)
    os.environ["O11Y_SCENARIO_TIME_ISO"] = resolve_scenario_time()

    print(f"Suite dir: {suite_dir}")
    print(f"Mode: {'resume' if opts.resume else 'fresh'}")
    print(f"Providers: {len(PROVIDERS)} ({', '.join(PROVIDERS)})")
    print(
        f"Concurrency: {opts.n_concurrent} | Attempts: {opts.n_attempts} | Variants: {len(STANDARD_SUITE)}"
    )
    if opts.dry_run:
        print("Dry run: showing plan only")

    if not opts.dry_run:
        run_preflight(quiet=opts.quiet)

    start = time.time()
    all_results: list[JobResult] = []
    failed = False

    with ThreadPoolExecutor(max_workers=len(PROVIDERS)) as pool:
        futures = {
            pool.submit(_run_provider_queue, provider, suite_dir, opts): provider
            for provider in PROVIDERS
        }
        for future in as_completed(futures):
            provider = futures[future]
            try:
                results = future.result()
                all_results.extend(results)
                if any(r.harbor_exit_code and r.harbor_exit_code != 0 for r in results):
                    failed = True
            except Exception as exc:
                print(f"[{provider}] failed: {exc}", file=sys.stderr)
                failed = True

    elapsed = time.time() - start

    if opts.dry_run:
        up_to_date = sum(1 for r in all_results if not r.needs_repair and not r.needs_harbor)
        needs_run = sum(1 for r in all_results if r.needs_harbor)
        repair_only = sum(1 for r in all_results if r.needs_repair and not r.needs_harbor)
        print(
            f"\nSummary: {len(all_results)} variants | up to date: {up_to_date} | run: {needs_run} | repair: {repair_only}"
        )
        print(
            f"  trials to run: {sum(r.missing_trials for r in all_results)} | to archive: {sum(r.repair_actions for r in all_results)}"
        )

    print(f"Finished in {elapsed:.1f}s")
    if not opts.dry_run:
        suite_report.write_report(suite_dir, tasks_dir=opts.tasks_dir, quiet=opts.quiet)
    if failed:
        raise SystemExit(1)
