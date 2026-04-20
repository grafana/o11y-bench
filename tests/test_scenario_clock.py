import os

from o11y_bench.scenario_clock import (
    bound_scenario_time,
    parse_scenario_time_iso,
    resolve_scenario_time,
)


def test_resolve_scenario_time_returns_env_when_set(monkeypatch) -> None:
    monkeypatch.setenv("O11Y_SCENARIO_TIME_ISO", "2026-04-04T10:05:14Z")

    assert resolve_scenario_time() == "2026-04-04T10:05:14Z"


def test_resolve_scenario_time_uses_now_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("O11Y_SCENARIO_TIME_ISO", raising=False)

    parse_scenario_time_iso(resolve_scenario_time())


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
