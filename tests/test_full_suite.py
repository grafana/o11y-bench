import json
from pathlib import Path

from o11y_bench import config, resume


def _archived_trial_dirs(job_dir: Path) -> list[Path]:
    archive_root = job_dir / resume.RESUME_ARCHIVE_DIRNAME
    if not archive_root.exists():
        return []
    return sorted(path for path in archive_root.rglob("*") if path.is_dir() and "__" in path.name)


def _make_job(tmp_path: Path, *, n_concurrent: int = 4) -> tuple[Path, dict]:
    """Create a job dir with config and return (job_dir, expected_fields)."""
    suite_dir = tmp_path / "suite"
    job_dir = suite_dir / "openai-gpt-5-4-mini-off-k3"
    job_dir.mkdir(parents=True)
    job_config = {
        "job_name": job_dir.name,
        "jobs_dir": str(suite_dir.resolve()),
        "n_attempts": 3,
        "orchestrator": {
            "type": "local",
            "n_concurrent_trials": n_concurrent,
            "retry": {"max_retries": 1},
        },
        "environment": {
            "type": "docker",
            "override_cpus": None,
            "override_memory_mb": None,
            "override_storage_mb": None,
        },
        "agents": [
            {
                "import_path": config.DEFAULT_AGENT_IMPORT_PATH,
                "model_name": "openai/gpt-5.4-mini",
                "kwargs": {"reasoning_effort": "off"},
            }
        ],
        "datasets": [{"path": str(config.TASKS_DIR.resolve())}],
    }
    (job_dir / "config.json").write_text(json.dumps(job_config))
    spec = config.JobSpec(
        jobs_dir=suite_dir,
        job_name=job_dir.name,
        tasks_dir=config.TASKS_DIR,
        model="openai/gpt-5.4-mini",
        reasoning_effort="off",
        n_attempts=3,
        n_concurrent=8,
        override_cpus=None,
        override_memory_mb=None,
        override_storage_mb=None,
    )
    expected = resume.expected_resume_fields(spec)
    return job_dir, expected


VALID_TRIAL_RESULT = {
    "agent_info": {"model_info": {"name": "openai/gpt-5.4-mini"}},
    "agent_result": {"metadata": {"reasoning_effort": "off"}},
    "agent_execution": {"started_at": "2026-01-01T00:00:00Z"},
    "verifier": {"status": "ok"},
    "verifier_result": {"rewards": {"reward": 1.0}},
}

INFRA_FAILURE_RESULT = {
    "agent_info": {"model_info": {"name": "openai/gpt-5.4-mini"}},
    "agent_result": None,
    "agent_execution": None,
    "verifier": None,
    "exception_info": {"exception_message": "Grafana not reachable at http://localhost:30000"},
    "verifier_result": {"rewards": {"reward": 0.0}},
}


def test_repair_patches_concurrency_and_paths(tmp_path: Path) -> None:
    """Resume repair patches config fields and trial dirs when suite was moved."""
    job_dir, expected = _make_job(tmp_path, n_concurrent=4)

    # Simulate moved suite: stale jobs_dir in config
    saved_config = json.loads((job_dir / "config.json").read_text())
    stale_source = tmp_path / "old-suite"
    saved_config["jobs_dir"] = str(stale_source.resolve())
    (job_dir / "config.json").write_text(json.dumps(saved_config))

    # Trial with stale trials_dir + one with broken config
    trial_dir = job_dir / "query-cpu-metrics__abc123"
    trial_dir.mkdir()
    (trial_dir / "config.json").write_text(
        json.dumps(
            {
                "trials_dir": str((stale_source / job_dir.name).resolve()),
                "environment": {"delete": True},
            }
        )
    )
    (trial_dir / "result.json").write_text(
        json.dumps({**VALID_TRIAL_RESULT, "task_name": "query-cpu-metrics"})
    )
    broken_trial = job_dir / "query-memory-usage__def456"
    broken_trial.mkdir()
    (broken_trial / "config.json").write_text("{broken")

    notes = resume.repair_job_dir_for_resume(job_dir, expected)
    saved_job = json.loads((job_dir / "config.json").read_text())
    saved_trial = json.loads((trial_dir / "config.json").read_text())

    assert any("n_concurrent 4 -> 8" in n for n in notes)
    assert any("jobs_dir" in n for n in notes)
    assert any("trial config(s)" in n for n in notes)
    assert saved_job["orchestrator"]["n_concurrent_trials"] == 8
    assert saved_job["jobs_dir"] == str((tmp_path / "suite").resolve())
    assert saved_job["environment"]["delete"] is False
    assert saved_trial["trials_dir"] == str(job_dir.resolve())
    assert saved_trial["environment"]["delete"] is False


def test_plan_marks_environment_delete_drift_for_repair(tmp_path: Path) -> None:
    job_dir, expected = _make_job(tmp_path)
    saved_config = json.loads((job_dir / "config.json").read_text())
    saved_config["environment"]["delete"] = True
    (job_dir / "config.json").write_text(json.dumps(saved_config))

    plan = resume.plan_job_dir_for_resume(job_dir, expected)

    assert plan.actions == ()
    assert plan.config_notes


def test_repair_normalizes_non_dict_environment_config(tmp_path: Path) -> None:
    job_dir, expected = _make_job(tmp_path)
    trial_dir = job_dir / "task-a__abc123"
    trial_dir.mkdir()
    (trial_dir / "config.json").write_text(
        json.dumps({"trials_dir": str(job_dir.resolve()), "environment": None})
    )
    (trial_dir / "result.json").write_text(
        json.dumps({**VALID_TRIAL_RESULT, "task_name": "task-a"})
    )

    resume.repair_job_dir_for_resume(job_dir, expected)

    repaired_job = json.loads((job_dir / "config.json").read_text())
    repaired_trial = json.loads((trial_dir / "config.json").read_text())
    assert repaired_job["environment"]["type"] == "docker"
    assert repaired_job["environment"]["delete"] is False
    assert repaired_trial["environment"] == {"delete": False}


def test_repair_archives_incomplete_and_retryable_trials(tmp_path: Path) -> None:
    """Incomplete (no result.json) and infra-failure trials are archived; valid ones kept."""
    job_dir, expected = _make_job(tmp_path)

    incomplete = job_dir / "task-a__abc123"
    retryable = job_dir / "task-b__def456"
    valid = job_dir / "task-c__ghi789"
    incomplete.mkdir()
    retryable.mkdir()
    valid.mkdir()
    (retryable / "result.json").write_text(
        json.dumps({**INFRA_FAILURE_RESULT, "task_name": "task-b"})
    )
    (valid / "result.json").write_text(json.dumps({**VALID_TRIAL_RESULT, "task_name": "task-c"}))

    resume.repair_job_dir_for_resume(job_dir, expected)

    assert not incomplete.exists()
    assert not retryable.exists()
    assert valid.exists()
    assert sorted(p.name for p in _archived_trial_dirs(job_dir)) == [
        "task-a__abc123",
        "task-b__def456",
    ]


def test_repair_archives_nonzero_agent_exit_trials(tmp_path: Path) -> None:
    job_dir, expected = _make_job(tmp_path)

    retryable = job_dir / "task-a__abc123"
    valid = job_dir / "task-b__def456"
    retryable.mkdir()
    valid.mkdir()
    (retryable / "result.json").write_text(
        json.dumps(
            {
                **VALID_TRIAL_RESULT,
                "task_name": "task-a",
                "exception_info": {
                    "exception_type": "NonZeroAgentExitCodeError",
                    "exception_message": "Agent exited with code 1",
                },
            }
        )
    )
    (valid / "result.json").write_text(json.dumps({**VALID_TRIAL_RESULT, "task_name": "task-b"}))

    resume.repair_job_dir_for_resume(job_dir, expected)

    assert not retryable.exists()
    assert valid.exists()
    assert [p.name for p in _archived_trial_dirs(job_dir)] == ["task-a__abc123"]


def test_repair_archives_stale_trials_when_task_changed(tmp_path: Path) -> None:
    """Trials with outdated task checksums are archived; matching ones kept."""
    job_dir, expected = _make_job(tmp_path)

    stale = job_dir / "task-a__abc123"
    valid = job_dir / "task-b__def456"
    removed = job_dir / "task-c__ghi789"
    for d, task, checksum in [
        (stale, "task-a", "old-checksum"),
        (valid, "task-b", "current-checksum"),
        (removed, "task-c", "removed-task-checksum"),
    ]:
        d.mkdir()
        (d / "result.json").write_text(
            json.dumps({**VALID_TRIAL_RESULT, "task_name": task, "task_checksum": checksum})
        )

    resume.repair_job_dir_for_resume(
        job_dir,
        expected,
        task_checksums={"task-a": "new-checksum", "task-b": "current-checksum"},
    )

    assert not stale.exists()
    assert valid.exists()
    assert not removed.exists()
    assert sorted(p.name for p in _archived_trial_dirs(job_dir)) == [
        "task-a__abc123",
        "task-c__ghi789",
    ]
