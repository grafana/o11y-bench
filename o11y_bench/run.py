"""Core execution engine: single jobs and full suite."""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from grading.models import Problem
from grading.transcript_parser import parse_transcript
from grading.verifier import grade
from reporting import report as suite_report
from reporting import run_report
from reporting.report_paths import (
    run_report_output_path,
)

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
from .harbor import build_command, build_resume_command, run_preflight
from .harbor import run as run_harbor
from .regrade_stack import problem_requires_live_stack, running_regrade_stack
from .resume import (
    ResumeTrialAction,
    compute_task_checksums,
    expected_resume_fields,
    format_resume_plan,
    plan_job_dir_for_resume,
    repair_job_dir_for_resume,
)
from .scenario_clock import bound_scenario_time, ensure_scenario_time


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

    print(f"{result.job_name}: {message}")


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
            _configured_task_name(config_payload)
            if isinstance(config_payload, dict)
            else None
        )
    return _trial_task_name(trial_dir, task_name)


def _sanitize_trial_config_payload(
    payload: dict[str, Any], *, job_dir: Path, trial_dir: Path, tasks_dir: Path, task_name: str | None
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
    trial_dir: Path,
    *,
    score: float,
    rewards: dict[str, float | int],
    explanations: dict[str, str],
    task_checksum: str | None,
) -> None:
    verifier_dir = trial_dir / "verifier"
    verifier_dir.mkdir(parents=True, exist_ok=True)
    (verifier_dir / "reward.txt").write_text(str(score))

    details: dict[str, Any] = dict(rewards)
    for criterion, explanation in explanations.items():
        details[f"explanation:{criterion}"] = explanation
    (verifier_dir / "grading_details.json").write_text(json.dumps(details, indent=2))

    result_path = trial_dir / "result.json"
    payload = json.loads(result_path.read_text())
    if task_checksum:
        payload["task_checksum"] = task_checksum
    payload.setdefault("verifier_result", {}).setdefault("rewards", {})["reward"] = score
    now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    payload["verifier"] = {"started_at": now, "finished_at": now}
    result_path.write_text(json.dumps(payload, indent=4) + "\n")


def regrade_job_dir(
    job_dir: Path,
    *,
    tasks_dir: Path,
    task_checksums: dict[str, str],
    quiet: bool = False,
) -> Path | None:
    model = os.getenv("GRADING_MODEL", "claude-haiku-4-5-20251001")
    scenario_time_iso = ensure_scenario_time(job_dir, prefer_existing=True)

    updated = 0
    trials_by_task: dict[str, list[Path]] = {}
    for trial_dir in _iter_trial_dirs(job_dir):
        result_path = trial_dir / "result.json"
        if not result_path.exists():
            continue
        payload = json.loads(result_path.read_text())
        task_name = payload.get("task_name")
        if isinstance(task_name, str) and task_name:
            trials_by_task.setdefault(task_name, []).append(trial_dir)

    for task_name, trial_dirs in sorted(trials_by_task.items()):
        problem_path = tasks_dir / task_name / "tests" / "problem.yaml"
        if not problem_path.exists():
            continue
        with open(problem_path) as file_obj:
            problem = Problem(**yaml.safe_load(file_obj))
        with bound_scenario_time(scenario_time_iso):
            if problem_requires_live_stack(problem):
                with running_regrade_stack(
                    task_dir=tasks_dir / task_name,
                    trial_dir=trial_dirs[0],
                    scenario_time_iso=scenario_time_iso,
                ):
                    updated += _regrade_trial_group(
                        trial_dirs=trial_dirs,
                        task_name=task_name,
                        problem=problem,
                        model=model,
                        task_checksums=task_checksums,
                    )
            else:
                updated += _regrade_trial_group(
                    trial_dirs=trial_dirs,
                    task_name=task_name,
                    problem=problem,
                    model=model,
                    task_checksums=task_checksums,
                )

    report_path = finalize_job_dir(job_dir, tasks_dir, task_checksums)
    if not quiet:
        print(f"{job_dir.name}: regraded {updated} trial(s)")
        if report_path:
            print(f"Report: {report_path}")
    return report_path


def _regrade_trial_group(
    *,
    trial_dirs: list[Path],
    task_name: str,
    problem: Problem,
    model: str,
    task_checksums: dict[str, str],
) -> int:
    updated = 0
    for trial_dir in trial_dirs:
        logs_dir = trial_dir / "agent"
        if not logs_dir.exists():
            continue
        transcript = parse_transcript(logs_dir)
        score, rewards, _checks_passed, explanations = grade(problem, transcript, model)
        _write_regrade_outputs(
            trial_dir,
            score=score,
            rewards=rewards,
            explanations=explanations,
            task_checksum=task_checksums.get(task_name),
        )
        updated += 1
    return updated


def execute_regrade(target_dir: Path, *, tasks_dir: Path, quiet: bool = False) -> None:
    task_checksums = compute_task_checksums(tasks_dir)
    job_dirs = _iter_job_dirs(target_dir)
    if not job_dirs:
        raise SystemExit(f"No job dirs found under {target_dir}")

    run_preflight(quiet=quiet)
    for job_dir in job_dirs:
        regrade_job_dir(job_dir, tasks_dir=tasks_dir, task_checksums=task_checksums, quiet=quiet)

    if len(job_dirs) > 1:
        suite_report.write_report(target_dir, tasks_dir=tasks_dir, quiet=quiet)


def execute_job(spec: JobSpec, *, dry_run: bool = False, quiet: bool = False) -> JobResult:
    """Single-pass: plan, repair, run harbor, finalize."""
    task_checksums = compute_task_checksums(spec.tasks_dir)
    job_dir = spec.jobs_dir / spec.job_name
    scenario_time_iso = ensure_scenario_time(
        job_dir,
        default_iso=os.environ.get("O11Y_SCENARIO_TIME_ISO"),
    )
    os.environ["O11Y_SCENARIO_TIME_ISO"] = scenario_time_iso
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
        fresh_harbor_exit_code = run_harbor(build_command(spec, quiet=quiet), forward_signals=False)
        report_path = finalize_job_dir(job_dir, spec.tasks_dir, task_checksums)
        result = JobResult(
            "fresh",
            spec.job_name,
            needs_harbor=True,
            missing_trials=missing,
            harbor_exit_code=fresh_harbor_exit_code,
            report_path=report_path,
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
        result = JobResult("up_to_date", spec.job_name)
        if quiet:
            _print_job_summary(result)
        return result

    if needs_repair:
        repair_notes = repair_job_dir_for_resume(job_dir, fields, task_checksums)
        if not quiet:
            for note in repair_notes:
                print(f"  {note}")

    rerun_harbor_exit_code: int | None = None
    if needs_harbor:
        rerun_harbor_exit_code = run_harbor(
            build_resume_command(str(job_dir / "config.json"), quiet=quiet),
            forward_signals=False,
        )

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
        result = execute_job(spec, dry_run=opts.dry_run, quiet=opts.quiet)
        results.append(result)
        if result.harbor_exit_code and result.harbor_exit_code != 0:
            print(f"[{provider}] Harbor failed (exit {result.harbor_exit_code})", file=sys.stderr)
            break
    return results


def execute_suite(opts: SuiteOpts) -> None:
    """Run all STANDARD_SUITE variants with per-provider parallelism."""
    suite_dir = resolve_suite_dir(opts.jobs_dir, allow_resume=opts.resume)
    os.environ["O11Y_SCENARIO_TIME_ISO"] = ensure_scenario_time(
        suite_dir,
        default_iso=os.environ.get("O11Y_SCENARIO_TIME_ISO"),
    )

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
