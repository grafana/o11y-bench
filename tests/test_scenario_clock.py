import os
from datetime import UTC, datetime, timedelta, timezone

from o11y_bench.scenario_clock import (
    bound_scenario_time,
    current_scenario_time_iso,
    parse_scenario_time_iso,
    resolve_scenario_time,
)


def test_resolve_scenario_time_returns_env_floored_to_the_minute(monkeypatch) -> None:
    monkeypatch.setenv("O11Y_SCENARIO_TIME_ISO", "2026-04-04T10:05:14Z")

    assert resolve_scenario_time() == "2026-04-04T10:05:00Z"


def test_the_clock_is_floored_to_the_minute_in_utc() -> None:
    last_instant = datetime(2026, 10, 20, 10, 5, 59, 999999, tzinfo=UTC)
    assert current_scenario_time_iso(last_instant) == "2026-10-20T10:05:00Z"

    plus_two = datetime(2026, 10, 20, 12, 5, 59, tzinfo=timezone(timedelta(hours=2)))
    assert current_scenario_time_iso(plus_two) == "2026-10-20T10:05:00Z"


def test_resolve_scenario_time_uses_now_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("O11Y_SCENARIO_TIME_ISO", raising=False)

    age = datetime.now(UTC) - parse_scenario_time_iso(resolve_scenario_time())
    assert timedelta(0) <= age < timedelta(minutes=1, seconds=5)


def test_bound_scenario_time_restores_previous_env(monkeypatch) -> None:
    monkeypatch.setenv("O11Y_SCENARIO_TIME_ISO", "2026-04-04T10:05:14Z")

    with bound_scenario_time("2026-04-04T11:00:00Z"):
        assert os.environ["O11Y_SCENARIO_TIME_ISO"] == "2026-04-04T11:00:00Z"

    assert os.environ["O11Y_SCENARIO_TIME_ISO"] == "2026-04-04T10:05:14Z"


def test_bound_scenario_time_clears_env_when_previously_unset(monkeypatch) -> None:
    monkeypatch.delenv("O11Y_SCENARIO_TIME_ISO", raising=False)

    with bound_scenario_time("2026-04-04T11:00:00Z"):
        assert os.environ["O11Y_SCENARIO_TIME_ISO"] == "2026-04-04T11:00:00Z"

    assert "O11Y_SCENARIO_TIME_ISO" not in os.environ
