import json

from reporting import compare_report


def test_load_job_uses_artifact_reasoning_effort_and_excludes_invalid_infra(tmp_path) -> None:
    job_dir = tmp_path / "job"
    valid_trial = job_dir / "task-a__good"
    (valid_trial / "verifier").mkdir(parents=True)
    (valid_trial / "result.json").write_text(
        json.dumps(
            {
                "agent_info": {"model_info": {"name": "gpt-5.4-mini", "provider": "openai"}},
                "agent_result": {
                    "cost_usd": 1.0,
                    "n_input_tokens": 10,
                    "n_output_tokens": 5,
                    "n_cache_tokens": 0,
                    "metadata": {},
                },
                "agent_execution": {
                    "started_at": "2026-03-18T00:00:00.000000Z",
                    "finished_at": "2026-03-18T00:01:00.000000Z",
                },
                "task_name": "task-a",
                "verifier_result": {"rewards": {"reward": 1.0}},
            }
        )
    )
    (valid_trial / "config.json").write_text(
        json.dumps({"agent": {"model_options": {"reasoning_effort": "high"}}})
    )
    (valid_trial / "verifier" / "grading_details.json").write_text(
        json.dumps({"score": 1.0, "rubric_passed": 1, "criterion": 1.0})
    )

    invalid_trial = job_dir / "task-a__infra"
    (invalid_trial / "verifier").mkdir(parents=True)
    (invalid_trial / "result.json").write_text(
        json.dumps(
            {
                "agent_info": {"model_info": {"name": "gpt-5.4-mini", "provider": "openai"}},
                "agent_result": None,
                "agent_execution": None,
                "verifier": None,
                "task_name": "task-a",
                "exception_info": {
                    "exception_message": "Docker compose command failed for environment task-a",
                },
                "verifier_result": {"rewards": {"reward": 0.0}},
            }
        )
    )
    (invalid_trial / "verifier" / "grading_details.json").write_text(
        json.dumps({"score": 0.0, "rubric_passed": 0, "criterion": 0.0})
    )

    data = compare_report.load_job(job_dir, tasks_dir=None)

    assert data["model_display"] == "GPT 5.4 Mini (high)"
    assert data["total_tasks"] == 1
    assert data["tasks_passed"] == 1
    assert data["tasks_consistent"] == 1
    assert data["pass_hat_rate"] == 1.0
