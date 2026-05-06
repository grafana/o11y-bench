import json

from grading.transcript_parser import parse_atif_trajectory


def test_parse_atif_trajectory_accepts_legacy_harbor_results(tmp_path) -> None:
    (tmp_path / "trajectory.json").write_text(
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
                                "arguments": {"query": '{service="api"}'},
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

    transcript = parse_atif_trajectory(tmp_path)

    assert [message.role for message in transcript.messages] == ["user", "assistant", "tool"]
    assert transcript.messages[1].tool_calls is not None
    assert transcript.messages[1].tool_calls[0].name == "query_loki"
    assert transcript.messages[2].tool_results is not None
    assert transcript.messages[2].tool_results[0].content == "error log"
