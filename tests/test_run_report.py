import json

import pytest

from reporting import run_report
from reporting.report_data import rubric_passed


@pytest.mark.parametrize(
    ("grading", "expected"),
    [
        (
            {
                "score": 1.0,
                "checks_passed": 1,
                "rubric_passed": 0,
                "check-a": 1.0,
                "rubric-a": 1.0,
                "explanation:check-a": "ok",
                "explanation:rubric-a": "ok",
            },
            True,
        ),
        (
            {
                "score": 0.5,
                "checks_passed": 1,
                "check-a": 1.0,
                "rubric-a": 0.0,
                "explanation:check-a": "ok",
                "explanation:rubric-a": "partial",
            },
            False,
        ),
    ],
)
def test_rubric_passed_requires_every_scored_item(
    grading: dict[str, object], expected: bool
) -> None:
    assert rubric_passed(grading) is expected


def test_generate_report_uses_trial_config_reasoning_effort(tmp_path) -> None:
    tasks_dir = tmp_path / "tasks"
    task_dir = tasks_dir / "task-a"
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text('[metadata]\ncategory = "prometheus_query"\n')

    job_dir = tmp_path / "job"
    trial_dir = job_dir / "task-a__abc123"
    (trial_dir / "agent").mkdir(parents=True)
    (trial_dir / "verifier").mkdir(parents=True)
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "agent_info": {"model_info": {"name": "gpt-5.4-nano", "provider": "openai"}},
                "agent_result": {"metadata": {}},
                "agent_execution": {
                    "started_at": "2026-03-18T00:00:00.000000Z",
                    "finished_at": "2026-03-18T00:01:00.000000Z",
                },
                "task_name": "task-a",
                "verifier_result": {"rewards": {"reward": 1.0}},
            }
        )
    )
    (trial_dir / "config.json").write_text(
        json.dumps({"agent": {"model_options": {"reasoning_effort": "high"}}})
    )
    (trial_dir / "verifier" / "grading_details.json").write_text(
        json.dumps({"score": 1.0, "rubric_passed": 1, "check-a": 1.0})
    )

    html = run_report.generate_report(job_dir, tasks_dir=tasks_dir)

    assert "GPT 5.4 Nano (high)" in html


def test_load_atif_trajectory_accepts_legacy_harbor_results(tmp_path) -> None:
    trajectory = tmp_path / "trajectory.json"
    trajectory.write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.6",
                "session_id": "legacy-run",
                "agent": {"name": "o11y-bench", "version": "1.0.0"},
                "steps": [
                    {"step_id": 1, "source": "user", "message": "Find errors"},
                    {
                        "step_id": 2,
                        "source": "agent",
                        "message": "(tool use)",
                        "tool_calls": [
                            {
                                "tool_call_id": "call-1",
                                "function_name": "query_loki",
                                "arguments": {"query": "{service=\"api\"}"},
                            }
                        ],
                        "observation": {
                            "results": [
                                {
                                    "source_call_id": "call-1",
                                    "content": "error log",
                                }
                            ]
                        },
                    },
                ],
                "final_metrics": {
                    "total_prompt_tokens": 10,
                    "total_completion_tokens": 5,
                    "total_tool_calls": 1,
                    "reasoning_effort": "off",
                },
            }
        )
    )

    transcript = run_report._load_atif_trajectory(trajectory)

    assert [message["type"] for message in transcript] == ["user", "assistant", "tool_result"]
    assert transcript[1]["message"]["content"][0]["name"] == "query_loki"
    assert transcript[2]["content"] == "error log"
