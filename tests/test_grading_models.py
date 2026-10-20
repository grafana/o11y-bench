import pytest

from grading.models import Message, ToolCall, ToolResult, Transcript

REPORT = "Root cause: payment-service connection pool exhausted. " * 5


@pytest.mark.parametrize(
    ("messages", "final_answer"),
    [
        pytest.param(
            [
                Message("user", "q"),
                Message("assistant", "Let me search.", [ToolCall("a", "s", {})]),
            ],
            "",
            id="a planning turn with tool calls is not an answer",
        ),
        pytest.param(
            [Message("user", "q"), Message("assistant", REPORT), Message("assistant", "Done.")],
            REPORT.strip() + "\n\nDone.",
            id="a closing note joins the report",
        ),
        pytest.param(
            [Message("user", "q"), Message("assistant", REPORT), Message("user", "and now?")],
            "",
            id="a later user message ends the answer",
        ),
    ],
)
def test_final_answer_is_the_closing_text_without_tool_calls(
    messages: list[Message], final_answer: str
) -> None:
    transcript = Transcript(messages)

    assert transcript.final_answer() == final_answer
    # The judge is told where the answer is only when there is one.
    assert ("Agent (final response)" in transcript.to_text()) == bool(final_answer)


def test_a_long_investigation_is_trimmed_but_keeps_what_the_judge_needs() -> None:
    messages = [Message("user", "Investigate the checkout service failure.")]
    for i in range(40):
        call = ToolCall(f"t{i}", "query_logs", {"query": "error" * 1000})
        messages.append(Message("assistant", f"step {i} " + "detail " * 200, tool_calls=[call]))
        messages.append(
            Message("tool", tool_results=[ToolResult(f"t{i}", f"log-line-{i} " * 1000)])
        )
    messages.append(Message("assistant", "The checkout failure is caused by payment timeouts."))

    text = Transcript(messages).to_text()

    # The raw transcript is over 650k characters.
    assert len(text) < 150_000
    assert text.startswith("User: Investigate the checkout service failure.")
    assert "step 0 detail" in text
    assert "log-line-39" in text and "log-line-0 " not in text
    assert text.endswith(
        "Agent (final response): The checkout failure is caused by payment timeouts."
    )
