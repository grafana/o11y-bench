import pytest

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
