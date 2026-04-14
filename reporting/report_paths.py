#!/usr/bin/env python3
"""Shared path helpers for Harbor job and suite reports."""

from pathlib import Path

SUITE_PREFIX = "full-suite-"
RUN_REPORT_NAME = "run_report.html"
SUITE_REPORT_NAME = "comparison.html"


def normalize_cli_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def normalize_repo_path(root: Path, path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve(strict=False)


def is_suite_dir(path: Path) -> bool:
    return path.name.startswith(SUITE_PREFIX)


def latest_job_dir(jobs_dir: Path, job_name: str | None) -> Path | None:
    if job_name:
        candidate = jobs_dir / job_name
        if candidate.exists():
            return candidate

    job_dirs = sorted(
        (path for path in jobs_dir.iterdir() if path.is_dir() and (path / "result.json").exists()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not job_dirs:
        return None
    return job_dirs[0]


def latest_suite_dir(jobs_root: Path) -> Path | None:
    if not jobs_root.exists():
        return None

    suite_dirs = [path for path in jobs_root.iterdir() if path.is_dir() and is_suite_dir(path)]
    if not suite_dirs:
        return None

    return max(suite_dirs, key=lambda path: (_suite_sort_key(path), path.name))


def _suite_sort_key(path: Path) -> int:
    mtimes = [path.stat().st_mtime_ns]

    report_paths = [path / SUITE_REPORT_NAME]
    for report_path in report_paths:
        if report_path.exists():
            mtimes.append(report_path.stat().st_mtime_ns)

    for job_dir in path.iterdir():
        if not job_dir.is_dir():
            continue
        result_path = job_dir / "result.json"
        if result_path.exists():
            mtimes.append(result_path.stat().st_mtime_ns)

    return max(mtimes)


def run_report_output_path(job_dir: Path, output: Path | None = None) -> Path:
    return output or (job_dir / RUN_REPORT_NAME)


def suite_report_output_path(jobs_dir: Path, output: Path | None = None) -> Path:
    if output is not None:
        return output
    if is_suite_dir(jobs_dir):
        return jobs_dir / SUITE_REPORT_NAME
    return jobs_dir / "report.html"
