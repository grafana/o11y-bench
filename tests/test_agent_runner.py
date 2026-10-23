import pytest
from harbor.models.trajectories import Trajectory

from agents import agent_runner


class _FakeResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self.headers: dict[str, str] = {}


class _FakeError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response_status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        response_code = status_code if response_status_code is None else response_status_code
        self.response = _FakeResponse(response_code) if response_code is not None else None


def test_enforce_step_limit_allows_cap_boundary() -> None:
    agent_runner.enforce_step_limit(agent_runner.MAX_AGENT_STEPS)


def test_enforce_step_limit_raises_after_cap() -> None:
    with pytest.raises(RuntimeError, match=str(agent_runner.MAX_AGENT_STEPS)):
        agent_runner.enforce_step_limit(agent_runner.MAX_AGENT_STEPS + 1)


def test_o11y_trajectory_metadata_validates_against_harbor_model() -> None:
    trajectory = {
        "schema_version": "ATIF-v1.7",
        "session_id": "run-1",
        "trajectory_id": "trajectory-1",
        "agent": {
            "name": "o11y-bench",
            "version": "1.0.0",
            "model_name": "openai/gpt-5.4-mini",
            "tool_definitions": [],
        },
        "steps": [
            agent_runner.make_atif_step(1, "system", "system prompt"),
            agent_runner.make_atif_step(2, "user", "task prompt"),
        ],
        "final_metrics": {
            "extra": {
                "total_tool_calls": 0,
                "reasoning_effort": "off",
                "elapsed_seconds": 0.1,
            },
        },
    }

    parsed = Trajectory.model_validate(trajectory)

    assert parsed.schema_version == "ATIF-v1.7"
    assert parsed.trajectory_id == "trajectory-1"
    assert parsed.final_metrics is not None
    assert parsed.final_metrics.extra is not None
    assert parsed.final_metrics.extra["reasoning_effort"] == "off"


def test_build_litellm_kwargs_omits_temperature_when_unset() -> None:
    result = agent_runner.build_litellm_kwargs("openai/gpt-5.4-mini", [], "off", None)

    assert "temperature" not in result


def test_cache_control_injection_is_only_enabled_for_anthropic() -> None:
    anthropic = agent_runner.build_litellm_kwargs(
        "anthropic/claude-sonnet-4-6", [], "off", None
    )
    vertex = agent_runner.build_litellm_kwargs(
        "vertex_ai/gemini-4-argon", [], "low", None
    )

    assert "cache_control_injection_points" in anthropic
    assert "cache_control_injection_points" not in vertex


def test_build_litellm_kwargs_parses_explicit_zero_temperature() -> None:
    result = agent_runner.build_litellm_kwargs("openai/gpt-5.4-mini", [], "off", "0")

    assert result["temperature"] == 0.0


def test_build_litellm_kwargs_prefers_reasoning_effort_over_temperature() -> None:
    result = agent_runner.build_litellm_kwargs("openai/gpt-5.4-mini", [], "medium", "0")

    assert result["reasoning_effort"] == "medium"
    assert "temperature" not in result


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        # Claude 5.x thinks adaptively unless told otherwise; each family takes a different "off".
        ("anthropic/claude-haiku-5-5", {"thinking": {"type": "disabled"}}),
        ("anthropic/claude-sonnet-5-5", {"thinking": {"type": "between_tools"}}),
        ("openai/gpt-5.4-mini", {"reasoning_effort": "none"}),
        ("openai/gpt-5.4-2026-03-05", {"reasoning_effort": "none"}),
        ("openai/gpt-5.6-luna", {"reasoning_effort": "none"}),
        ("openai/gpt-6-sol", {"reasoning_effort": "none"}),
        # Models that do not think unless asked: "off" sends nothing.
        ("anthropic/claude-haiku-4-5-20251001", {}),
        ("anthropic/claude-sonnet-4-6", {}),
        ("openai/gpt-5.1-codex-mini", {}),
        ("google/gemini-3-flash-preview", {}),
    ],
)
def test_off_turns_thinking_off_the_way_each_model_accepts(
    model: str, expected: dict[str, object]
) -> None:
    result = agent_runner.build_litellm_kwargs(model, [], "off", None)

    settings = {k: v for k, v in result.items() if k in ("thinking", "reasoning_effort")}
    assert settings == expected


@pytest.mark.parametrize(
    "model",
    [
        "anthropic/claude-opus-5-5",
        "anthropic/claude-fable-5-1",
        "openai/gpt-6.1-sol",
        "vertex_ai/gemini-4-argon",
    ],
)
def test_off_fails_for_models_that_cannot_turn_thinking_off(model: str) -> None:
    with pytest.raises(ValueError, match="cannot turn thinking off"):
        agent_runner.build_litellm_kwargs(model, [], "off", None)


def test_default_effort_sends_no_thinking_settings_but_keeps_temperature() -> None:
    result = agent_runner.build_litellm_kwargs("anthropic/claude-haiku-5-5", [], "default", "0")

    assert "thinking" not in result and "reasoning_effort" not in result
    assert result["temperature"] == 0.0


def test_build_litellm_kwargs_omits_temperature_for_claude_opus_4_7() -> None:
    result = agent_runner.build_litellm_kwargs("anthropic/claude-opus-4-7", [], "off", "0")

    assert "temperature" not in result


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (_FakeError("rate limit", status_code=429), True),
        (_FakeError("Anthropic overloaded_error", status_code=529), True),
        (_FakeError("gateway failure", response_status_code=503), True),
        (_FakeError("Anthropic overloaded_error without status"), True),
        (_FakeError("job 503 completed without result"), False),
    ],
)
def test_is_retryable_upstream_error_handles_retryable_upstream_failures(
    error: Exception, expected: bool
) -> None:
    assert agent_runner.is_retryable_upstream_error(error) is expected
