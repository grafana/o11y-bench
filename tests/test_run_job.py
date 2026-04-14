import argparse
import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from o11y_bench import cli, config, harbor, run


def test_build_command_produces_valid_harbor_invocation() -> None:
    spec = config.JobSpec(
        jobs_dir=config.ROOT / "jobs",
        job_name="test-job",
        tasks_dir=config.TASKS_DIR,
        model="openai/gpt-5.4-mini",
        reasoning_effort="off",
        n_attempts=3,
        n_concurrent=8,
    )
    command = harbor.build_command(spec)

    assert command[:4] == ["uv", "run", "harbor", "run"]
    assert "--yes" in command
    assert "openai/gpt-5.4-mini" in command
    assert "test-job" in command


def test_build_command_maps_task_filters_to_harbor_include_task_name() -> None:
    spec = config.JobSpec(
        jobs_dir=config.ROOT / "jobs",
        job_name="test-job",
        tasks_dir=config.TASKS_DIR,
        model="openai/gpt-5.4-mini",
        reasoning_effort="off",
        n_attempts=1,
        n_concurrent=1,
        task_names=("promql-error-rate", "dashboard-create-service-overview"),
    )
    command = harbor.build_command(spec)

    assert "--include-task-name" in command
    assert "--task-name" not in command
    assert command.count("--include-task-name") == 2
    assert "promql-error-rate" in command
    assert "dashboard-create-service-overview" in command


def test_build_resume_command_uses_saved_job_config() -> None:
    command = harbor.build_resume_command("/tmp/job/config.json")

    assert command == ["uv", "run", "harbor", "run", "--yes", "--config", "/tmp/job/config.json"]


def test_build_command_from_args_adds_repo_defaults() -> None:
    command = harbor.build_command_from_args(["--model", "openai/gpt-5.4-mini"])

    assert command[:4] == ["uv", "run", "harbor", "run"]
    assert "--yes" in command
    assert str(config.JOB_CONFIG) in command
    assert str(config.TASKS_DIR) in command
    assert config.DEFAULT_AGENT_IMPORT_PATH in command


def test_execute_job_resumes_from_saved_config(monkeypatch, tmp_path) -> None:
    job_dir = tmp_path / "jobs" / "resume-job"
    job_dir.mkdir(parents=True)
    (job_dir / "config.json").write_text("{}\n")

    spec = config.JobSpec(
        jobs_dir=tmp_path / "jobs",
        job_name="resume-job",
        tasks_dir=tmp_path / "tasks",
        model="openai/gpt-5.4-mini",
        reasoning_effort="off",
        n_attempts=3,
        n_concurrent=2,
    )

    monkeypatch.setattr(run, "compute_task_checksums", lambda tasks_dir: {})
    monkeypatch.setattr(run, "ensure_scenario_time", lambda *args, **kwargs: "2026-04-04T12:00:00Z")
    monkeypatch.setattr(run, "_selected_task_names", lambda spec: ["demo-task"])
    monkeypatch.setattr(
        run,
        "plan_job_dir_for_resume",
        lambda job_dir, fields, task_checksums: SimpleNamespace(
            job_dir=job_dir,
            config_notes=(),
            actions=(),
        ),
    )
    monkeypatch.setattr(run, "_count_usable_trials", lambda *args, **kwargs: {"demo-task": 0})
    monkeypatch.setattr(run, "repair_job_dir_for_resume", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        run,
        "finalize_job_dir",
        lambda job_dir, tasks_dir, task_checksums: job_dir / "run_report.html",
    )

    harbor_calls: list[list[str]] = []
    monkeypatch.setattr(
        run, "run_harbor", lambda command, **kwargs: harbor_calls.append(command) or 0
    )

    result = run.execute_job(spec, quiet=True)

    assert result.harbor_exit_code == 0
    assert harbor_calls == [
        ["uv", "run", "harbor", "run", "--yes", "--config", str(job_dir / "config.json"), "--quiet"]
    ]


@pytest.mark.parametrize(
    ("dry_run", "expected_events", "expected_preflight_calls"),
    [
        (False, ["preflight", "execute"], [True]),
        (True, ["execute"], []),
    ],
)
def test_cmd_job_preflight_respects_dry_run(
    monkeypatch, dry_run: bool, expected_events: list[str], expected_preflight_calls: list[bool]
) -> None:
    events: list[str] = []
    preflight_calls: list[bool] = []
    execute_calls: list[tuple[config.JobSpec, bool, bool]] = []

    def fake_run_preflight(*, quiet: bool = False) -> None:
        preflight_calls.append(quiet)
        events.append("preflight")

    monkeypatch.setattr(cli, "run_preflight", fake_run_preflight)

    def fake_execute_job(spec: config.JobSpec, *, dry_run: bool = False, quiet: bool = False):
        execute_calls.append((spec, dry_run, quiet))
        events.append("execute")
        return run.JobResult(status="fresh", job_name=spec.job_name)

    monkeypatch.setattr(cli, "execute_job", fake_execute_job)

    args = argparse.Namespace(
        model="openai/gpt-5.4-mini",
        agent=None,
        agent_import_path=config.DEFAULT_AGENT_IMPORT_PATH,
        reasoning_effort="high",
        jobs_dir=config.ROOT / "jobs",
        job_name="branch-safe-job",
        path=None,
        n_attempts=1,
        n_concurrent=2,
        override_cpus=None,
        override_memory_mb=None,
        override_storage_mb=None,
        task_name=[],
        dry_run=dry_run,
        quiet=True,
    )

    cli._cmd_job(args)

    assert events == expected_events
    assert preflight_calls == expected_preflight_calls
    assert len(execute_calls) == 1
    spec, executed_dry_run, quiet = execute_calls[0]
    assert spec.job_name == "branch-safe-job"
    assert spec.model == "openai/gpt-5.4-mini"
    assert executed_dry_run is dry_run
    assert quiet is True


def test_cmd_job_auto_job_name_includes_builtin_agent(monkeypatch) -> None:
    monkeypatch.setattr(cli, "run_preflight", lambda *, quiet=False: None)

    execute_calls: list[config.JobSpec] = []

    def fake_execute_job(spec: config.JobSpec, *, dry_run: bool = False, quiet: bool = False):
        execute_calls.append(spec)
        return run.JobResult(status="dry_run", job_name=spec.job_name)

    monkeypatch.setattr(cli, "execute_job", fake_execute_job)

    args = argparse.Namespace(
        model="openai/gpt-5.4-nano",
        agent="opencode",
        agent_import_path=config.DEFAULT_AGENT_IMPORT_PATH,
        reasoning_effort="off",
        jobs_dir=config.ROOT / "jobs",
        job_name=None,
        path=None,
        n_attempts=3,
        n_concurrent=1,
        override_cpus=None,
        override_memory_mb=None,
        override_storage_mb=None,
        task_name=["query-cpu-metrics"],
        dry_run=True,
        quiet=True,
    )

    cli._cmd_job(args)

    assert len(execute_calls) == 1
    assert execute_calls[0].job_name == "openai-gpt-5-4-nano-off-opencode-k3"


def test_cmd_job_auto_job_name_includes_custom_agent_import_path(monkeypatch) -> None:
    monkeypatch.setattr(cli, "run_preflight", lambda *, quiet=False: None)

    execute_calls: list[config.JobSpec] = []

    def fake_execute_job(spec: config.JobSpec, *, dry_run: bool = False, quiet: bool = False):
        execute_calls.append(spec)
        return run.JobResult(status="dry_run", job_name=spec.job_name)

    monkeypatch.setattr(cli, "execute_job", fake_execute_job)

    args = argparse.Namespace(
        model="openai/gpt-5.4-nano",
        agent=None,
        agent_import_path="custom_agents.my_agent:MyAgent",
        reasoning_effort="off",
        jobs_dir=config.ROOT / "jobs",
        job_name=None,
        path=None,
        n_attempts=3,
        n_concurrent=1,
        override_cpus=None,
        override_memory_mb=None,
        override_storage_mb=None,
        task_name=["query-cpu-metrics"],
        dry_run=True,
        quiet=True,
    )

    cli._cmd_job(args)

    assert len(execute_calls) == 1
    assert execute_calls[0].job_name == "openai-gpt-5-4-nano-off-custom-agents-my-agent-myagent-k3"


def test_cmd_job_rejects_agent_and_agent_import_path_together() -> None:
    args = argparse.Namespace(
        model="openai/gpt-5.4-nano",
        agent="opencode",
        agent_import_path="custom_agents.my_agent:MyAgent",
        reasoning_effort="off",
        jobs_dir=config.ROOT / "jobs",
        job_name=None,
        path=None,
        n_attempts=3,
        n_concurrent=1,
        override_cpus=None,
        override_memory_mb=None,
        override_storage_mb=None,
        task_name=[],
        dry_run=True,
        quiet=True,
    )

    with pytest.raises(SystemExit, match="Use either --agent or --agent-import-path"):
        cli._cmd_job(args)


def test_regrade_job_dir_updates_verifier_outputs(monkeypatch, tmp_path) -> None:
    tasks_dir = tmp_path / "tasks"
    problem_dir = tasks_dir / "demo-task" / "tests"
    problem_dir.mkdir(parents=True)
    problem_dir.joinpath("problem.yaml").write_text(
        "\n".join(
            [
                "id: demo-task",
                "category: prometheus_query",
                "statement: demo",
                "checks: []",
                "rubric:",
                "- criterion: The final response is accurate.",
                "  weight: 1.0",
            ]
        )
        + "\n"
    )

    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "config.json").write_text(
        json.dumps({"agents": [{"model_name": "claude-haiku-4-5-20251001"}]})
    )

    trial_dir = job_dir / "demo-task__abc123"
    (trial_dir / "agent").mkdir(parents=True)
    (trial_dir / "result.json").write_text(json.dumps({"task_name": "demo-task"}) + "\n")

    monkeypatch.setattr(
        run,
        "parse_transcript",
        lambda logs_dir: object(),
    )
    monkeypatch.setattr(
        run,
        "grade",
        lambda problem, transcript, model: (
            1.0,
            {
                "score": 1.0,
                "checks_passed": 1,
                "rubric_passed": 1,
                "The final response is accurate.": 1.0,
            },
            True,
            {"The final response is accurate.": "looks good"},
        ),
    )
    monkeypatch.setattr(
        run,
        "finalize_job_dir",
        lambda job_dir, tasks_dir, task_checksums: job_dir / "run_report.html",
    )

    report_path = run.regrade_job_dir(
        job_dir,
        tasks_dir=tasks_dir,
        task_checksums={"demo-task": "checksum-123"},
        quiet=True,
    )

    assert report_path == job_dir / "run_report.html"
    assert (trial_dir / "verifier" / "reward.txt").read_text() == "1.0"
    grading = json.loads((trial_dir / "verifier" / "grading_details.json").read_text())
    assert grading["The final response is accurate."] == 1.0
    result = json.loads((trial_dir / "result.json").read_text())
    assert result["task_checksum"] == "checksum-123"
    assert result["verifier_result"]["rewards"]["reward"] == 1.0


def test_regrade_job_dir_reuses_live_stack_once_per_task(monkeypatch, tmp_path) -> None:
    tasks_dir = tmp_path / "tasks"
    task_dir = tasks_dir / "demo-task"
    problem_dir = task_dir / "tests"
    problem_dir.mkdir(parents=True)
    (task_dir / "environment").mkdir()
    (task_dir / "environment" / "setup.json").write_text("{}\n")
    problem_dir.joinpath("problem.yaml").write_text(
        "\n".join(
            [
                "id: demo-task",
                "category: prometheus_query",
                "statement: demo",
                "checks: []",
                "rubric:",
                "- criterion: The final response is accurate.",
                "  weight: 1.0",
                "  fact:",
                "    kind: query",
                "    backend: prometheus",
                "    query: up",
            ]
        )
        + "\n"
    )

    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "config.json").write_text(json.dumps({"agents": [{"model_name": "test-model"}]}))

    for suffix in ("abc123", "def456"):
        trial_dir = job_dir / f"demo-task__{suffix}"
        (trial_dir / "agent").mkdir(parents=True)
        (trial_dir / "result.json").write_text(json.dumps({"task_name": "demo-task"}) + "\n")

    stack_calls: list[tuple[str, str]] = []

    def fake_running_regrade_stack(*, task_dir, trial_dir, scenario_time_iso):
        stack_calls.append((task_dir.name, trial_dir.name))
        return nullcontext()

    monkeypatch.setattr(run, "parse_transcript", lambda logs_dir: object())
    monkeypatch.setattr(
        run,
        "grade",
        lambda problem, transcript, model: (
            1.0,
            {
                "score": 1.0,
                "checks_passed": 0,
                "rubric_passed": 1,
                "The final response is accurate.": 1.0,
            },
            True,
            {"The final response is accurate.": "looks good"},
        ),
    )
    monkeypatch.setattr(run, "running_regrade_stack", fake_running_regrade_stack)
    monkeypatch.setattr(
        run,
        "finalize_job_dir",
        lambda job_dir, tasks_dir, task_checksums: job_dir / "run_report.html",
    )

    run.regrade_job_dir(
        job_dir,
        tasks_dir=tasks_dir,
        task_checksums={"demo-task": "checksum-123"},
        quiet=True,
    )

    assert stack_calls == [("demo-task", "demo-task__abc123")]
