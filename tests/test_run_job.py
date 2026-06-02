import argparse
import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from o11y_bench import cli, config, harbor, run


def _option_values(command: list[str], option: str) -> list[str]:
    return [command[i + 1] for i, arg in enumerate(command[:-1]) if arg == option]


def test_build_command_produces_valid_harbor_invocation() -> None:
    spec = config.JobSpec(
        jobs_dir=config.ROOT / "jobs",
        job_name="test-job",
        tasks_dir=config.TASKS_DIR,
        model="openai/gpt-5.4-mini",
        reasoning_effort="off",
        n_attempts=3,
        n_concurrent=8,
        harbor_args=("--ak", "temperature=0"),
    )
    command = harbor.build_command(spec)

    assert command[:4] == ["uv", "run", "harbor", "run"]
    assert "--yes" in command
    assert "openai/gpt-5.4-mini" in command
    assert "test-job" in command
    assert "temperature=0" in _option_values(command, "--ak")


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
        harbor_args=("--ak", "temperature=0"),
    )

    monkeypatch.setattr(run, "compute_task_checksums", lambda tasks_dir: {})
    monkeypatch.setattr(
        run, "resolve_scenario_time", lambda *args, **kwargs: "2026-04-04T12:00:00Z"
    )
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

    harbor_calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run_harbor(command: list[str], **kwargs: object) -> int:
        harbor_calls.append((command, kwargs))
        return 0

    monkeypatch.setattr(run, "run_harbor", fake_run_harbor)

    result = run.execute_job(spec, quiet=True)

    assert result.harbor_exit_code == 0
    assert len(harbor_calls) == 1
    command, kwargs = harbor_calls[0]
    assert str(job_dir / "config.json") in _option_values(command, "--config")
    assert "temperature=0" in _option_values(command, "--ak")
    assert "--quiet" in command
    assert kwargs == {"forward_signals": False}


def test_run_harbor_with_live_export_starts_before_harbor_and_finishes_after(monkeypatch) -> None:
    events: list[str] = []

    class FakeExporter:
        def start(self):
            events.append("start")

        def finish(self, *, status: str, error: str = ""):
            events.append(f"finish:{status}:{error}")

    def fake_run_harbor(command: list[str], **kwargs: object) -> int:
        events.append("harbor")
        assert command == ["harbor", "run"]
        assert kwargs == {"forward_signals": False}
        return 0

    monkeypatch.setattr(run, "run_harbor", fake_run_harbor)

    exit_code = run._run_harbor_with_live_export(
        ["harbor", "run"],
        sigil_exporter=FakeExporter(),
    )

    assert exit_code == 0
    assert events == ["start", "harbor", "finish:succeeded:"]


def test_finalize_job_dir_sanitizes_artifact_paths(monkeypatch, tmp_path) -> None:
    tasks_dir = tmp_path / "tasks"
    task_dir = tasks_dir / "promql-error-rate"
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "tests" / "problem.yaml").write_text("prompt: test\n")

    jobs_dir = tmp_path / "nested" / "suite"
    job_dir = jobs_dir / "openai-gpt-5-4-nano-off-k1"
    trial_dir = job_dir / "promql-error-rate__abc123"
    trial_dir.mkdir(parents=True)

    (job_dir / "config.json").write_text(
        json.dumps(
            {
                "job_name": job_dir.name,
                "jobs_dir": str(jobs_dir.resolve()),
                "datasets": [{"path": str(tasks_dir.resolve())}],
            }
        )
    )
    (trial_dir / "config.json").write_text(
        json.dumps(
            {
                "task": {"path": str(task_dir.resolve())},
                "trials_dir": str(job_dir.resolve()),
            }
        )
    )
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "trial_uri": trial_dir.resolve().as_uri(),
                "task_name": "promql-error-rate",
                "task_id": {"path": str(task_dir.resolve())},
                "task_checksum": "old",
                "config": {
                    "task": {"path": str(task_dir.resolve())},
                    "trials_dir": str(job_dir.resolve()),
                },
                "task": {"path": str(task_dir.resolve())},
                "agent_info": {"model_info": {"name": "openai/gpt-5.4-nano"}},
                "agent_result": {"metadata": {"reasoning_effort": "off"}},
            }
        )
    )

    monkeypatch.setattr(run.run_report, "load_trials", lambda job_dir: [{}])
    monkeypatch.setattr(run.run_report, "write_report", lambda *args, **kwargs: None)

    report_path = run.finalize_job_dir(job_dir, tasks_dir, {"promql-error-rate": "new"})

    saved_job = json.loads((job_dir / "config.json").read_text())
    saved_trial = json.loads((trial_dir / "config.json").read_text())
    saved_result = json.loads((trial_dir / "result.json").read_text())

    assert report_path == job_dir / "run_report.html"
    assert saved_job["jobs_dir"] == "jobs/suite"
    assert saved_job["datasets"][0]["path"] == "tasks"
    assert saved_trial["task"]["path"] == "tasks/promql-error-rate"
    assert saved_trial["trials_dir"] == "jobs/suite/openai-gpt-5-4-nano-off-k1"
    assert (
        saved_result["trial_uri"]
        == "file:jobs/suite/openai-gpt-5-4-nano-off-k1/promql-error-rate__abc123"
    )
    assert saved_result["config"]["task"]["path"] == "tasks/promql-error-rate"
    assert saved_result["config"]["trials_dir"] == "jobs/suite/openai-gpt-5-4-nano-off-k1"
    assert saved_result["task_id"]["path"] == "tasks/promql-error-rate"
    assert saved_result["task"]["path"] == "tasks/promql-error-rate"
    assert saved_result["task_checksum"] == "new"


def test_finalize_job_dir_preserves_full_task_id_in_trial_config(monkeypatch, tmp_path) -> None:
    tasks_dir = tmp_path / "tasks"
    task_name = "traceql-discover-orders-error-attributes"
    task_dir = tasks_dir / task_name
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "tests" / "problem.yaml").write_text("prompt: test\n")

    jobs_dir = tmp_path / "nested" / "suite"
    job_dir = jobs_dir / "openai-gpt-5-4-nano-off-k1"
    trial_dir = job_dir / "traceql-discover-orders-error-at__abc1234"
    truncated_task_name = trial_dir.name.split("__", 1)[0]
    truncated_task_path = str((tasks_dir / truncated_task_name).resolve())
    trial_dir.mkdir(parents=True)

    (job_dir / "config.json").write_text(
        json.dumps(
            {
                "job_name": job_dir.name,
                "jobs_dir": str(jobs_dir.resolve()),
                "datasets": [{"path": str(tasks_dir.resolve())}],
            }
        )
    )
    (trial_dir / "config.json").write_text(
        json.dumps(
            {
                "task": {"path": truncated_task_path},
                "trials_dir": str(job_dir.resolve()),
            }
        )
    )
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "trial_uri": trial_dir.resolve().as_uri(),
                "task_name": task_name,
                "task_id": {"path": str(task_dir.resolve())},
                "task_checksum": "old",
                "config": {
                    "task": {"path": truncated_task_path},
                    "trials_dir": str(job_dir.resolve()),
                },
                "task": {"path": str(task_dir.resolve())},
                "agent_info": {"model_info": {"name": "openai/gpt-5.4-nano"}},
                "agent_result": {"metadata": {"reasoning_effort": "off"}},
            }
        )
    )

    monkeypatch.setattr(run.run_report, "load_trials", lambda job_dir: [{}])
    monkeypatch.setattr(run.run_report, "write_report", lambda *args, **kwargs: None)

    run.finalize_job_dir(job_dir, tasks_dir, {task_name: "new"})

    saved_trial = json.loads((trial_dir / "config.json").read_text())
    saved_result = json.loads((trial_dir / "result.json").read_text())

    assert saved_trial["task"]["path"] == f"tasks/{task_name}"
    assert saved_result["config"]["task"]["path"] == f"tasks/{task_name}"
    assert saved_result["task"]["path"] == f"tasks/{task_name}"
    assert saved_result["task_id"]["path"] == f"tasks/{task_name}"


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


def test_main_passes_unknown_job_args_through_to_harbor(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["o11y_bench", "job", "--model", "openai/gpt-5.4-nano", "--ak", "temperature=0"],
    )
    monkeypatch.setattr(cli, "run_preflight", lambda *, quiet=False: None)

    execute_calls: list[config.JobSpec] = []

    def fake_execute_job(spec: config.JobSpec, *, dry_run: bool = False, quiet: bool = False):
        execute_calls.append(spec)
        return run.JobResult(status="dry_run", job_name=spec.job_name)

    monkeypatch.setattr(cli, "execute_job", fake_execute_job)

    cli.main()

    assert len(execute_calls) == 1
    assert execute_calls[0].harbor_args == ("--ak", "temperature=0")


def test_main_rejects_unknown_args_before_job_subcommand(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["o11y_bench", "--quiet", "job", "--model", "openai/gpt-5.4-nano"],
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code != 0


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


def test_cmd_job_passes_live_sigil_exporter_when_flag_enabled(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "run_preflight", lambda *, quiet=False: None)
    monkeypatch.setenv("SIGIL_EVAL_ENDPOINT", "https://grafana-stack.grafana.net")

    execute_calls: list[tuple[config.JobSpec, object]] = []
    exporter_calls: list[tuple[object, object, object]] = []

    def fake_execute_job(
        spec: config.JobSpec,
        *,
        dry_run: bool = False,
        quiet: bool = False,
        sigil_exporter=None,
    ):
        execute_calls.append((spec, sigil_exporter))
        return run.JobResult(
            status="ran",
            job_name=spec.job_name,
            report_path=spec.jobs_dir / spec.job_name / "run_report.html",
        )

    def fake_live_exporter(job_dir, tasks_dir, options):
        exporter_calls.append((job_dir, tasks_dir, options))
        return "live-exporter"

    monkeypatch.setattr(cli, "execute_job", fake_execute_job)
    monkeypatch.setattr(cli, "SigilLiveExporter", fake_live_exporter)

    args = argparse.Namespace(
        model="openai/gpt-5.4-nano",
        agent="opencode",
        agent_import_path=config.DEFAULT_AGENT_IMPORT_PATH,
        reasoning_effort="off",
        jobs_dir=config.ROOT / "jobs",
        job_name=None,
        path=None,
        n_attempts=1,
        n_concurrent=1,
        override_cpus=None,
        override_memory_mb=None,
        override_storage_mb=None,
        task_name=["query-cpu-metrics"],
        dry_run=False,
        quiet=True,
        sigil_experiment_export=True,
        sigil_api="https://sigil-prod-us-central1.grafana.net",
        sigil_tenant="grafana-cloud-tenant",
        sigil_auth_mode="basic",
        sigil_auth_token="grafana-cloud-access-policy-token",
        sigil_run_id="o11y-run-1",
        sigil_name="o11y run",
        sigil_description="",
        sigil_tag=["o11y-bench", "opencode"],
    )

    cli._cmd_job(args)

    assert len(execute_calls) == 1
    assert execute_calls[0][1] == "live-exporter"
    assert len(exporter_calls) == 1
    job_dir, tasks_dir, options = exporter_calls[0]
    assert job_dir == config.ROOT / "jobs" / "openai-gpt-5-4-nano-off-opencode-k1"
    assert tasks_dir == config.TASKS_DIR
    assert options.api == "https://sigil-prod-us-central1.grafana.net"
    assert options.tenant == "grafana-cloud-tenant"
    assert options.auth_mode == "basic"
    assert options.auth_token == "grafana-cloud-access-policy-token"
    assert options.run_id == "o11y-run-1"
    assert options.tags == ["o11y-bench", "opencode"]

    captured = capsys.readouterr()
    assert "Publishing results to Grafana AI o11y" in captured.out
    assert "Sigil stack URL: https://grafana-stack.grafana.net" in captured.out
    assert "Sigil ingest endpoint: https://sigil-prod-us-central1.grafana.net" in captured.out
    assert "grafana-cloud-access-policy-token" not in captured.out


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
