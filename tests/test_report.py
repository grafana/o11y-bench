import json
import os
from pathlib import Path

import pytest

from reporting import report, report_data


def _trial(model: str, reasoning_effort: str, task_name: str, reward: float) -> dict:
    return {
        "agent_info": {"model_info": {"name": model}},
        "agent_result": {
            "cost_usd": 1.5,
            "n_input_tokens": 1_000,
            "n_output_tokens": 250,
            "n_cache_tokens": 100,
            "metadata": {"reasoning_effort": reasoning_effort},
        },
        "agent_execution": {
            "started_at": "2026-03-18T00:00:00Z",
            "finished_at": "2026-03-18T00:01:00Z",
        },
        "task_name": task_name,
        "verifier_result": {"rewards": {"reward": reward}},
    }


def test_aggregate_keeps_reasoning_variants_separate() -> None:
    trials = [
        _trial("gpt-5.4", "off", "task-a", 1.0),
        _trial("gpt-5.4", "high", "task-a", 0.0),
    ]

    data = report.aggregate(trials, {"task-a": "prometheus_query"})

    labels = {model["label"]: model for model in data["models"]}
    assert labels["GPT 5.4"]["pass_rate"] == 1.0
    assert labels["GPT 5.4"]["pass_hat_rate"] == 1.0
    assert labels["GPT 5.4 (high)"]["pass_rate"] == 0.0
    assert labels["GPT 5.4 (high)"]["pass_hat_rate"] == 0.0


def test_aggregate_uses_pass_hat_k_as_primary_metric() -> None:
    trials = [
        _trial("model-a", "off", "task-a", 1.0),
        _trial("model-a", "off", "task-a", 0.0),
        _trial("model-b", "off", "task-a", 1.0),
        _trial("model-b", "off", "task-a", 1.0),
    ]

    data = report.aggregate(trials, {"task-a": "prometheus_query"})

    assert [model["name"] for model in data["models"]] == ["model-b", "model-a"]
    assert data["models"][0]["pass_hat_rate"] == 1.0
    assert data["models"][1]["pass_hat_rate"] == 0.0
    assert data["models"][1]["pass_rate"] == 1.0


def test_load_trials_follows_symlinked_job_dirs(tmp_path) -> None:
    real_job = tmp_path / "real-job"
    real_trial = real_job / "task-a__abc123"
    real_trial.mkdir(parents=True)
    (real_trial / "result.json").write_text(
        json.dumps(
            {
                "agent_info": {"model_info": {"name": "gpt-5.4"}},
                "task_name": "task-a",
                "verifier_result": {"rewards": {"reward": 1.0}},
            }
        )
    )

    selection = tmp_path / "selection"
    selection.mkdir()
    (selection / "linked-job").symlink_to(real_job, target_is_directory=True)

    trials = report.load_trials(selection)

    assert len(trials) == 1
    assert trials[0]["task_name"] == "task-a"


def test_latest_suite_dir_prefers_most_recent_suite(tmp_path: Path) -> None:
    jobs_root = tmp_path / "jobs"
    older = jobs_root / "full-suite-20260323-090000"
    newer = jobs_root / "full-suite-20260323-100000"
    older.mkdir(parents=True)
    newer.mkdir()
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))

    assert report.latest_suite_dir(jobs_root) == newer


def test_aggregate_excludes_pre_agent_infra_failures() -> None:
    invalid_trial = {
        "agent_info": {"model_info": {"name": "gpt-5.4"}},
        "agent_result": None,
        "agent_execution": None,
        "verifier": None,
        "task_name": "task-a",
        "exception_info": {
            "exception_message": "Docker compose command failed for environment task-a",
        },
        "verifier_result": {"rewards": {"reward": 0.0}},
    }
    valid_trial = _trial("gpt-5.4", "off", "task-a", 1.0)

    data = report.aggregate([invalid_trial, valid_trial], {"task-a": "prometheus_query"})

    model = data["models"][0]
    assert model["n_tasks"] == 1
    assert model["pass_hat_rate"] == 1.0
    assert model["pass_rate"] == 1.0


def test_aggregate_excludes_nonzero_agent_exit_trials() -> None:
    bad_trial = _trial("gpt-5.4", "off", "task-a", 0.0)
    bad_trial["exception_info"] = {
        "exception_type": "NonZeroAgentExitCodeError",
        "exception_message": "Agent exited with code 1",
    }
    valid_trial = _trial("gpt-5.4", "off", "task-a", 1.0)

    data = report.aggregate([bad_trial, valid_trial], {"task-a": "prometheus_query"})

    model = data["models"][0]
    assert model["n_tasks"] == 1
    assert model["n_valid_trials"] == 1
    assert model["pass_hat_rate"] == 1.0
    assert model["pass_rate"] == 1.0


@pytest.mark.parametrize(
    ("traceback",),
    [
        ("asyncio.exceptions.CancelledError\nAgentTimeoutError",),
        ("RewardFileNotFoundError\nAgentTimeoutError",),
    ],
)
def test_aggregate_counts_retryable_agent_exceptions_as_valid(traceback: str) -> None:
    timeout_trial = _trial("gpt-5.4", "off", "task-a", 0.0)
    timeout_trial["exception_info"] = {
        "exception_type": "AgentTimeoutError",
        "exception_message": "Agent execution timed out after 600.0 seconds",
        "exception_traceback": traceback,
    }
    timeout_trial["__result_path"] = str(
        Path("/tmp/jobs/full-suite-123/openai-gpt-5-4-off-k3/task-a__abc/result.json")
    )

    data = report.aggregate([timeout_trial], {"task-a": "prometheus_query"})

    model = data["models"][0]
    assert model["n_valid_trials"] == 1
    assert model["expected_trials"] == 1


def test_aggregate_treats_timeout_full_score_as_failure_for_pass_metrics() -> None:
    timeout_trial = _trial("gpt-5.4", "off", "task-a", 1.0)
    timeout_trial["exception_info"] = {
        "exception_type": "AgentTimeoutError",
        "exception_message": "Agent execution timed out after 600.0 seconds",
    }
    timeout_trial["__result_path"] = str(
        Path("/tmp/jobs/full-suite-123/openai-gpt-5-4-off-k3/task-a__abc/result.json")
    )

    data = report.aggregate([timeout_trial], {"task-a": "prometheus_query"})

    model = data["models"][0]
    assert model["n_valid_trials"] == 1
    assert model["mean_score"] == 1.0
    assert model["pass_hat_rate"] == 0.0
    assert model["pass_rate"] == 0.0


@pytest.mark.parametrize(
    ("model_options", "expected_label"),
    [
        ({"reasoning_effort": "high"}, "GPT 5.4 Nano (high)"),
        ({}, "GPT 5.4 Nano"),
    ],
)
def test_aggregate_resolves_reasoning_effort_from_trial_config(
    tmp_path, model_options: dict, expected_label: str
) -> None:
    trial_dir = tmp_path / "task-a__abc123"
    trial_dir.mkdir()
    (trial_dir / "config.json").write_text(json.dumps({"agent": {"model_options": model_options}}))

    trial = _trial("gpt-5.4-nano", "off", "task-a", 1.0)
    trial["agent_result"]["metadata"] = {}
    trial["__result_path"] = str(trial_dir / "result.json")

    data = report.aggregate([trial], {"task-a": "prometheus_query"})

    assert data["models"][0]["label"] == expected_label


@pytest.mark.parametrize(
    ("model_name", "n_input", "n_cache", "n_output", "expected_cost"),
    [
        (
            "gpt-5.4-mini",
            2_000,
            500,
            400,
            ((1_500 * 0.75) + (500 * 0.075) + (400 * 4.50)) / 1_000_000,
        ),
        (
            "gpt-5.4-nano-2026-03-17",
            3_000,
            1_000,
            800,
            ((2_000 * 0.20) + (1_000 * 0.02) + (800 * 1.25)) / 1_000_000,
        ),
    ],
)
def test_agent_result_metrics_uses_snapshot_fallback_pricing(
    model_name: str, n_input: int, n_cache: int, n_output: int, expected_cost: float
) -> None:
    result = {
        "agent_info": {"model_info": {"name": model_name}},
        "agent_result": {
            "cost_usd": 0.0,
            "n_input_tokens": n_input,
            "n_cache_tokens": n_cache,
            "n_output_tokens": n_output,
        },
    }

    cost_usd, got_input, got_cache, got_output = report_data.agent_result_metrics(result)

    assert cost_usd == expected_cost
    assert (got_input, got_cache, got_output) == (n_input, n_cache, n_output)


def test_write_report_warns_on_mixed_task_checksums(tmp_path: Path, monkeypatch, capsys) -> None:
    jobs_dir = tmp_path / "jobs"
    output = tmp_path / "report.html"
    trial_a = jobs_dir / "job-a" / "task-a__abc123"
    trial_b = jobs_dir / "job-b" / "task-a__def456"
    trial_a.mkdir(parents=True)
    trial_b.mkdir(parents=True)

    payload_a = _trial("gpt-5.4", "off", "task-a", 1.0)
    payload_a["task_checksum"] = "checksum-a"
    payload_b = _trial("claude-sonnet-4-6", "off", "task-a", 1.0)
    payload_b["task_checksum"] = "checksum-b"
    (trial_a / "result.json").write_text(json.dumps(payload_a))
    (trial_b / "result.json").write_text(json.dumps(payload_b))

    monkeypatch.setattr(
        report,
        "refresh_run_reports",
        lambda jobs_dir, tasks_dir=None, quiet=False: (0, 0),
    )
    monkeypatch.setattr(report, "render_html", lambda data: "<html></html>")

    report.write_report(jobs_dir, output=output)

    captured = capsys.readouterr()
    assert "mixed checksums across models" in captured.err
    assert output.exists()


def test_aggregate_uses_job_config_n_attempts_for_expected_trials(tmp_path: Path) -> None:
    trial_dir = tmp_path / "job-a" / "task-a__abc123"
    trial_dir.mkdir(parents=True)
    (trial_dir.parent / "config.json").write_text(json.dumps({"n_attempts": 3}))

    trial = _trial("claude-haiku-4-5-20251001", "off", "task-a", 1.0)
    trial["__result_path"] = str(trial_dir / "result.json")

    data = report.aggregate([trial], {f"task-{i}": "prometheus_query" for i in "abc"})

    assert data["models"][0]["n_valid_trials"] == 1
    assert data["models"][0]["expected_trials"] == 9
