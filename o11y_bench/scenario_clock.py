import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

SCENARIO_TIME_ENV = "O11Y_SCENARIO_TIME_ISO"
SCENARIO_TIME_FILE = "scenario_time.txt"
SCENARIO_TIME_RE = re.compile(r"Current time:\s*([0-9T:\-+.Z]+)")


def current_scenario_time_iso(now: datetime | None = None) -> str:
    current = datetime.now(UTC) if now is None else now.astimezone(UTC)
    return current.replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_scenario_time_iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def scenario_time_path(target_dir: Path) -> Path:
    return target_dir / SCENARIO_TIME_FILE


def load_saved_scenario_time(target_dir: Path) -> str | None:
    for candidate in (target_dir, target_dir.parent):
        path = scenario_time_path(candidate)
        if path.exists():
            saved = path.read_text().strip()
            if saved:
                return saved
    return None


def infer_scenario_time_from_job(job_dir: Path) -> str | None:
    if not job_dir.exists():
        return None
    for trial_dir in sorted(
        path for path in job_dir.iterdir() if path.is_dir() and "__" in path.name
    ):
        instruction_path = trial_dir / "agent" / "instruction.txt"
        if not instruction_path.exists():
            continue
        match = SCENARIO_TIME_RE.search(instruction_path.read_text())
        if match:
            return match.group(1)
    return None


def ensure_scenario_time(
    target_dir: Path,
    *,
    default_iso: str | None = None,
    prefer_existing: bool = False,
) -> str:
    if prefer_existing:
        scenario_time_iso = (
            load_saved_scenario_time(target_dir)
            or infer_scenario_time_from_job(target_dir)
            or os.environ.get(SCENARIO_TIME_ENV, "").strip()
            or default_iso
            or current_scenario_time_iso()
        )
    else:
        scenario_time_iso = (
            os.environ.get(SCENARIO_TIME_ENV, "").strip()
            or load_saved_scenario_time(target_dir)
            or infer_scenario_time_from_job(target_dir)
            or default_iso
            or current_scenario_time_iso()
        )
    target_dir.mkdir(parents=True, exist_ok=True)
    scenario_time_path(target_dir).write_text(f"{scenario_time_iso}\n")
    return scenario_time_iso


@contextmanager
def bound_scenario_time(scenario_time_iso: str) -> Iterator[None]:
    previous = os.environ.get(SCENARIO_TIME_ENV)
    os.environ[SCENARIO_TIME_ENV] = scenario_time_iso
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(SCENARIO_TIME_ENV, None)
        else:
            os.environ[SCENARIO_TIME_ENV] = previous
