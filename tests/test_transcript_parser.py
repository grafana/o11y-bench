import json

from grading.transcript_parser import parse_atif_trajectory, parse_transcript


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


def test_the_judge_never_sees_the_agents_thinking(tmp_path) -> None:
    (tmp_path / "trajectory.json").write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.7",
                "session_id": "thinking-run",
                "agent": {"name": "o11y-bench", "version": "1.0.0"},
                "steps": [
                    {"step_id": 1, "source": "user", "message": "Which service is slowest?"},
                    {
                        "step_id": 2,
                        "source": "agent",
                        "message": "payment-service is slowest.",
                        "reasoning_content": "SECRET-CHAIN-OF-THOUGHT order-service looks worse",
                    },
                ],
                "final_metrics": {"total_prompt_tokens": 1, "total_completion_tokens": 1},
            }
        )
    )

    transcript = parse_atif_trajectory(tmp_path)

    assert "SECRET-CHAIN-OF-THOUGHT" not in transcript.to_text()
    assert transcript.final_answer() == "payment-service is slowest."


def test_a_claude_code_stream_gives_the_answer_once_and_marks_failed_tool_calls(tmp_path) -> None:
    events = [
        {"type": "user", "message": {"content": "Which service is slow?"}},
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "tool_use", "id": "c1", "name": "query_logs", "input": {"q": "x"}}
                ]
            },
        },
        {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "c1",
                        "content": "boom",
                        "is_error": True,
                    }
                ]
            },
        },
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "order-service."}]}},
        # Claude Code repeats its last message as the run's result.
        {"type": "result", "result": "order-service."},
    ]
    (tmp_path / "claude-code.txt").write_text("\n".join(json.dumps(e) for e in events))

    transcript = parse_transcript(tmp_path)

    assert transcript.final_answer() == "order-service."
    text = transcript.to_text()
    assert text.count("order-service.") == 1
    assert "✗" in text and "boom" in text
