import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

SCENARIO_TIME_ENV = "O11Y_SCENARIO_TIME_ISO"


def current_scenario_time_iso(now: datetime | None = None) -> str:
    current = datetime.now(UTC) if now is None else now.astimezone(UTC)
    return current.replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_scenario_time_iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def resolve_scenario_time() -> str:
    # Default to real "now" so Tempo's ingest/search time windows stay aligned; telemetry
    # is regenerated each stack boot, so the absolute timestamp is incidental.
    return os.environ.get(SCENARIO_TIME_ENV, "").strip() or current_scenario_time_iso()


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
