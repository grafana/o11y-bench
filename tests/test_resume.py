import json
from pathlib import Path

from o11y_bench import resume


def _write_resume_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    tasks_dir = tmp_path / "tasks"
    (tasks_dir / "demo-task" / "tests").mkdir(parents=True)
    (tasks_dir / "demo-task" / "tests" / "problem.yaml").write_text("prompt: test\n")

    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / "demo-job"
    trial_dir = job_dir / "demo-task__abc1234"
    trial_dir.mkdir(parents=True)

    (job_dir / "config.json").write_text(
        json.dumps({"jobs_dir": str(jobs_dir.resolve()), "datasets": []}, indent=4)
    )
    (trial_dir / "config.json").write_text(
        json.dumps(
            {
                "trials_dir": "jobs/demo-job",
                "task": {"path": "tasks/demo-task"},
            },
            indent=4,
        )
    )

    return tasks_dir, jobs_dir, job_dir


def test_patch_job_paths_for_resume_restores_absolute_trial_paths(tmp_path: Path) -> None:
    tasks_dir, jobs_dir, job_dir = _write_resume_fixture(tmp_path)

    notes = resume.patch_job_paths_for_resume(job_dir, jobs_dir, tasks_dir)

    payload = json.loads((job_dir / "demo-task__abc1234" / "config.json").read_text())
    assert notes == [
        "Patched 1 trial config(s) for resume: "
        f"trials_dir -> {job_dir.resolve()}, task.path -> {tasks_dir.resolve()}/<task>"
    ]
    assert payload["trials_dir"] == str(job_dir.resolve())
    assert payload["task"]["path"] == str((tasks_dir / "demo-task").resolve())


def test_patch_job_paths_for_resume_dry_run_reports_needed_trial_path_repairs(
    tmp_path: Path,
) -> None:
    tasks_dir, jobs_dir, job_dir = _write_resume_fixture(tmp_path)

    notes = resume.patch_job_paths_for_resume(job_dir, jobs_dir, tasks_dir, dry_run=True)

    assert notes == [
        "Would patch 1 trial config(s) for resume: "
        f"trials_dir -> {job_dir.resolve()}, task.path -> {tasks_dir.resolve()}/<task>"
    ]
    payload = json.loads((job_dir / "demo-task__abc1234" / "config.json").read_text())
    assert payload["trials_dir"] == "jobs/demo-job"
    assert payload["task"]["path"] == "tasks/demo-task"


def test_patch_job_paths_for_resume_uses_full_task_id_from_result_json(tmp_path: Path) -> None:
    tasks_dir = tmp_path / "tasks"
    task_name = "traceql-discover-orders-error-attributes"
    (tasks_dir / task_name / "tests").mkdir(parents=True)
    (tasks_dir / task_name / "tests" / "problem.yaml").write_text("prompt: test\n")

    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / "demo-job"
    trial_dir = job_dir / "traceql-discover-orders-error-at__abc1234"
    trial_dir.mkdir(parents=True)

    (job_dir / "config.json").write_text(
        json.dumps({"jobs_dir": str(jobs_dir.resolve()), "datasets": []}, indent=4)
    )
    (trial_dir / "config.json").write_text(
        json.dumps(
            {
                "trials_dir": "jobs/demo-job",
                "task": {"path": "tasks/traceql-discover-orders-error-at"},
            },
            indent=4,
        )
    )
    (trial_dir / "result.json").write_text(json.dumps({"task_name": task_name}, indent=4))

    resume.patch_job_paths_for_resume(job_dir, jobs_dir, tasks_dir)

    payload = json.loads((trial_dir / "config.json").read_text())
    assert payload["task"]["path"] == str((tasks_dir / task_name).resolve())


def test_patch_job_paths_for_resume_matches_relative_job_config_paths(tmp_path: Path) -> None:
    task_name = "traceql-discover-orders-error-attributes"
    jobs_dir = tmp_path / "jobs"
    tasks_dir = tmp_path / "tasks"
    (tasks_dir / task_name / "tests").mkdir(parents=True)
    (tasks_dir / task_name / "tests" / "problem.yaml").write_text("prompt: test\n")

    job_dir = jobs_dir / "demo-job"
    trial_dir = job_dir / "traceql-discover-orders-error-at__abc1234"
    trial_dir.mkdir(parents=True)

    (job_dir / "config.json").write_text(
        json.dumps(
            {"jobs_dir": str(jobs_dir.resolve()), "datasets": [{"path": "tasks"}]},
            indent=4,
        )
    )
    (trial_dir / "config.json").write_text(
        json.dumps(
            {
                "trials_dir": str(job_dir.resolve()),
                "task": {"path": str((tasks_dir / "traceql-discover-orders-error-at").resolve())},
            },
            indent=4,
        )
    )
    (trial_dir / "result.json").write_text(json.dumps({"task_name": task_name}, indent=4))

    notes = resume.patch_job_paths_for_resume(job_dir, jobs_dir, tasks_dir)

    payload = json.loads((trial_dir / "config.json").read_text())
    assert notes == [
        "Patched 1 trial config(s) for resume: "
        f"trials_dir -> {job_dir.resolve()}, task.path -> tasks/<task>"
    ]
    assert payload["trials_dir"] == str(job_dir.resolve())
    assert payload["task"]["path"] == f"tasks/{task_name}"
