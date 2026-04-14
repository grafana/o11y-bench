import os
from pathlib import Path

from o11y_bench.scenario_clock import (
    bound_scenario_time,
    ensure_scenario_time,
    infer_scenario_time_from_job,
)


def test_ensure_scenario_time_persists_existing_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("O11Y_SCENARIO_TIME_ISO", "2026-04-04T10:05:14Z")

    value = ensure_scenario_time(tmp_path)

    assert value == "2026-04-04T10:05:14Z"
    assert (tmp_path / "scenario_time.txt").read_text().strip() == value


def test_ensure_scenario_time_prefers_saved_value_for_regrade(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("O11Y_SCENARIO_TIME_ISO", "2026-04-04T12:00:00Z")
    (tmp_path / "scenario_time.txt").write_text("2026-04-04T10:05:14Z\n")

    value = ensure_scenario_time(tmp_path, prefer_existing=True)

    assert value == "2026-04-04T10:05:14Z"
    assert (tmp_path / "scenario_time.txt").read_text().strip() == value


def test_bound_scenario_time_restores_previous_env(monkeypatch) -> None:
    monkeypatch.setenv("O11Y_SCENARIO_TIME_ISO", "2026-04-04T10:05:14Z")

    with bound_scenario_time("2026-04-04T11:00:00Z"):
        assert os.environ["O11Y_SCENARIO_TIME_ISO"] == "2026-04-04T11:00:00Z"

    assert os.environ["O11Y_SCENARIO_TIME_ISO"] == "2026-04-04T10:05:14Z"


def test_infer_scenario_time_from_existing_job_artifact(tmp_path: Path) -> None:
    trial_dir = tmp_path / "task__abc123" / "agent"
    trial_dir.mkdir(parents=True)
    (trial_dir / "instruction.txt").write_text(
        "<context>\nCurrent time: 2026-04-04T10:05:14Z\n</context>\n"
    )

    assert infer_scenario_time_from_job(tmp_path) == "2026-04-04T10:05:14Z"
