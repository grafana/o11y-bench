"""CLI entry point: subcommands for run, suite, and finalize."""

import argparse
import os
import sys
from pathlib import Path

from reporting.report_paths import latest_job_dir, normalize_repo_path
from scripts.sync_tasks import is_materialized_tasks_dir, materialize_specs_path

from .config import (
    DEFAULT_AGENT_IMPORT_PATH,
    DEFAULT_N_ATTEMPTS,
    DEFAULT_N_CONCURRENT,
    ROOT,
    TASKS_DIR,
    JobSpec,
    SuiteOpts,
    make_job_name,
)
from .harbor import build_command_from_args, run_preflight
from .harbor import run as run_harbor
from .resume import compute_task_checksums
from .run import execute_job, execute_regrade, execute_suite, finalize_job_dir
from .scenario_clock import SCENARIO_TIME_ENV, current_scenario_time_iso


def _extract_option(args: list[str], *names: str) -> str | None:
    for i, arg in enumerate(args):
        if arg in names and i + 1 < len(args):
            return args[i + 1]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(prog="o11y_bench", description="o11y-bench benchmark runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- run: passthrough to harbor ---
    run_parser = subparsers.add_parser("run", help="Wraps `harbor run` with preflight")
    run_parser.add_argument("--skip-preflight", action="store_true")
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--quiet", action="store_true")
    run_parser.add_argument("harbor_args", nargs=argparse.REMAINDER)

    # --- suite: all standard variants ---
    suite_parser = subparsers.add_parser("suite", help="Run the standard full suite")
    suite_parser.add_argument("--jobs-dir", type=Path)
    suite_parser.add_argument("--path", type=Path)
    suite_parser.add_argument("--no-resume", action="store_true")
    suite_parser.add_argument("--dry-run", action="store_true")
    suite_parser.add_argument("--quiet", action="store_true")
    suite_parser.add_argument(
        "--agento11y-publish",
        action="store_true",
        help="Publish each completed trial to an Agent Observability experiment",
    )
    suite_parser.add_argument("--n-attempts", type=int, default=DEFAULT_N_ATTEMPTS)
    suite_parser.add_argument("--n-concurrent", type=int, default=DEFAULT_N_CONCURRENT)
    suite_parser.add_argument("--override-cpus", type=int)
    suite_parser.add_argument("--override-memory-mb", type=int)
    suite_parser.add_argument("--override-storage-mb", type=int)

    # --- job: single variant with resume/repair ---
    job_parser = subparsers.add_parser("job", help="Run or resume a single benchmark variant")
    job_parser.add_argument("--model", required=True)
    job_parser.add_argument("--agent", "-a", help="Harbor built-in agent name (e.g. opencode)")
    job_parser.add_argument(
        "--agent-import-path",
        default=DEFAULT_AGENT_IMPORT_PATH,
        help="Python import path for a custom Harbor agent class",
    )
    job_parser.add_argument("--reasoning-effort", default="off")
    job_parser.add_argument("--jobs-dir", type=Path, default=ROOT / "jobs")
    job_parser.add_argument("--job-name")
    job_parser.add_argument("--path", type=Path)
    job_parser.add_argument("--n-attempts", type=int, default=DEFAULT_N_ATTEMPTS)
    job_parser.add_argument("--n-concurrent", type=int, default=DEFAULT_N_CONCURRENT)
    job_parser.add_argument("--override-cpus", type=int)
    job_parser.add_argument("--override-memory-mb", type=int)
    job_parser.add_argument("--override-storage-mb", type=int)
    job_parser.add_argument(
        "--task-name",
        action="append",
        default=[],
        help="Run only the named task(s); may be repeated",
    )
    job_parser.add_argument("--dry-run", action="store_true")
    job_parser.add_argument("--quiet", action="store_true")
    job_parser.add_argument(
        "--agento11y-publish",
        action="store_true",
        help="Publish each completed trial to an Agent Observability experiment",
    )

    # --- finalize: stamp checksums + generate per-job report ---
    finalize_parser = subparsers.add_parser(
        "finalize", help="Stamp checksums and generate job report"
    )
    finalize_parser.add_argument("--jobs-dir", type=Path, default=ROOT / "jobs")
    finalize_parser.add_argument("--path", type=Path)
    finalize_parser.add_argument("--job-name")
    finalize_parser.add_argument("harbor_args", nargs=argparse.REMAINDER)

    # --- regrade: rerun verifier only against existing transcripts ---
    regrade_parser = subparsers.add_parser(
        "regrade", help="Rerun verifier against existing transcripts without rerunning agents"
    )
    regrade_parser.add_argument("--jobs-dir", type=Path, default=ROOT / "jobs")
    regrade_parser.add_argument("--path", type=Path)
    regrade_parser.add_argument("--job-name")
    regrade_parser.add_argument("--quiet", action="store_true")
    regrade_parser.add_argument(
        "--offline",
        action="store_true",
        help="Grade from each trial's saved verifier/truth.json; no stack, no live reads",
    )
    regrade_parser.add_argument(
        "--n-concurrent",
        type=int,
        default=1,
        help="Tasks to regrade at once, each on its own stack",
    )

    args = _parse_args(parser)

    match args.command:
        case "run":
            _cmd_run(args)
        case "suite":
            _cmd_suite(args)
        case "job":
            _cmd_job(args)
        case "finalize":
            _cmd_finalize(args)
        case "regrade":
            _cmd_regrade(args)


def _parse_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    raw_args = sys.argv[1:]
    if raw_args[:1] != ["job"]:
        return parser.parse_args(raw_args)

    args, harbor_args = parser.parse_known_args(raw_args)
    args.harbor_args = tuple(arg for arg in harbor_args if arg != "--")
    return args


def _cmd_run(args: argparse.Namespace) -> None:
    """Preflight → harbor run (passthrough)."""
    harbor_args = [a for a in args.harbor_args if a != "--"]

    if not args.skip_preflight:
        run_preflight(quiet=args.quiet)

    command = build_command_from_args(harbor_args, quiet=args.quiet)
    if not os.environ.get(SCENARIO_TIME_ENV):
        os.environ[SCENARIO_TIME_ENV] = current_scenario_time_iso()

    if args.dry_run:
        print("Harbor command:", " ".join(command))
        return

    exit_code = run_harbor(command)
    if exit_code != 0:
        raise SystemExit(exit_code)


def _cmd_suite(args: argparse.Namespace) -> None:
    execute_suite(
        SuiteOpts(
            jobs_dir=args.jobs_dir,
            resume=not args.no_resume,
            dry_run=args.dry_run,
            quiet=args.quiet,
            n_attempts=args.n_attempts,
            n_concurrent=args.n_concurrent,
            tasks_dir=_resolve_tasks_path(args.path),
            agento11y_publish=args.agento11y_publish,
            override_cpus=args.override_cpus,
            override_memory_mb=args.override_memory_mb,
            override_storage_mb=args.override_storage_mb,
        )
    )


def _cmd_job(args: argparse.Namespace) -> None:
    provider, _, model_name = args.model.partition("/")
    if not provider or not model_name:
        raise SystemExit("--model must include provider/model, e.g. anthropic/claude-sonnet-4-6")
    if args.agent and args.agent_import_path != DEFAULT_AGENT_IMPORT_PATH:
        raise SystemExit("Use either --agent or --agent-import-path, not both")

    job_name = args.job_name or make_job_name(
        provider,
        model_name,
        args.reasoning_effort,
        args.n_attempts,
        agent=args.agent,
        agent_import_path=args.agent_import_path,
    )

    spec = JobSpec(
        jobs_dir=normalize_repo_path(ROOT, args.jobs_dir),
        job_name=job_name,
        tasks_dir=_resolve_tasks_path(args.path),
        model=args.model,
        agent=args.agent,
        agent_import_path=args.agent_import_path,
        reasoning_effort=args.reasoning_effort,
        n_attempts=args.n_attempts,
        n_concurrent=args.n_concurrent,
        override_cpus=args.override_cpus,
        override_memory_mb=args.override_memory_mb,
        override_storage_mb=args.override_storage_mb,
        task_names=tuple(args.task_name),
        harbor_args=tuple(getattr(args, "harbor_args", ())),
    )

    if not args.dry_run:
        run_preflight(quiet=args.quiet)

    execute_kwargs = {"dry_run": args.dry_run, "quiet": args.quiet}
    if getattr(args, "agento11y_publish", False):
        execute_kwargs["agento11y_publish"] = True
    result = execute_job(spec, **execute_kwargs)
    if result.agento11y_result and not args.quiet:
        published = result.agento11y_result
        print(
            f"Published {published.trial_count} trial(s) and {published.score_count} score(s) "
            f"to Agent Observability experiment {published.experiment_id}"
        )
        print(f"View in Agent Observability: {published.url}")
    if result.harbor_exit_code and result.harbor_exit_code != 0:
        raise SystemExit(result.harbor_exit_code)


def _cmd_finalize(args: argparse.Namespace) -> None:
    """Stamp checksums into result.json and generate per-job HTML report."""
    harbor_args = [a for a in args.harbor_args if a != "--"]
    jobs_dir = normalize_repo_path(
        ROOT, _extract_option(harbor_args, "--jobs-dir", "-o") or args.jobs_dir
    )
    tasks_arg = _extract_option(harbor_args, "--path", "-p", "--dataset", "-d")
    tasks_dir = _resolve_tasks_path(Path(tasks_arg) if tasks_arg else args.path)
    job_name = _extract_option(harbor_args, "--job-name") or args.job_name

    latest = latest_job_dir(jobs_dir, job_name)
    if latest is None:
        return

    task_checksums = compute_task_checksums(tasks_dir)
    report_path = finalize_job_dir(latest, tasks_dir, task_checksums)
    if report_path:
        print(f"Report: {report_path}")


def _cmd_regrade(args: argparse.Namespace) -> None:
    tasks_dir = _resolve_tasks_path(args.path)
    target = normalize_repo_path(ROOT, args.jobs_dir)
    if args.job_name:
        latest = latest_job_dir(target, args.job_name)
        if latest is None:
            raise SystemExit(f"No job dir found for --job-name {args.job_name!r} under {target}")
        target = latest
    execute_regrade(
        target,
        tasks_dir=tasks_dir,
        quiet=args.quiet,
        offline=args.offline,
        n_concurrent=args.n_concurrent,
    )


def _resolve_tasks_path(path: Path | None) -> Path:
    if path is None:
        return TASKS_DIR
    resolved = normalize_repo_path(ROOT, path)
    if is_materialized_tasks_dir(resolved):
        return resolved
    if not resolved.exists():
        raise SystemExit(f"--path not found: {resolved}")
    return materialize_specs_path(resolved)


if __name__ == "__main__":
    main()
