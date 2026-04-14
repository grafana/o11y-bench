from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.models.agent.context import AgentContext

from agents.langchain_o11y_agent import LangChainO11yBenchAgent
from agents.o11y_agent import (
    VIEWER_COMMAND_STDOUT_PATH,
    O11yBenchAgent,
    build_runner_command,
    select_remote_mcp_url,
)


def test_select_remote_mcp_url_skips_loopback_and_default_stack_hosts() -> None:
    servers = [
        SimpleNamespace(url="http://localhost:8080/mcp"),
        SimpleNamespace(url="http://127.0.0.1:8080/mcp"),
        SimpleNamespace(url="http://o11y-stack:8080/mcp"),
        SimpleNamespace(url="http://o11y-stack-job123:8080/mcp"),
    ]

    assert select_remote_mcp_url(servers) == "http://o11y-stack-job123:8080/mcp"


def test_select_remote_mcp_url_returns_none_when_no_remote_url_exists() -> None:
    servers = [
        SimpleNamespace(url="http://localhost:8080/mcp"),
        SimpleNamespace(url="http://[::1]:8080/mcp"),
        SimpleNamespace(url=""),
        SimpleNamespace(url=None),
    ]

    assert select_remote_mcp_url(servers) is None


def test_build_runner_command_uses_bash_and_quotes_viewer_log_path() -> None:
    command = build_runner_command()

    assert command.startswith("bash -lc ")
    assert f'tee "{VIEWER_COMMAND_STDOUT_PATH}"' in command


class MockExecResult:
    def __init__(self, return_code: int, stdout: str = "", stderr: str = "") -> None:
        self.return_code = return_code
        self.stdout = stdout
        self.stderr = stderr


class MockEnvironment:
    def __init__(self, result: MockExecResult) -> None:
        self.result = result
        self.exec_calls: list[tuple[str, dict[str, str]]] = []
        self.upload_calls: list[tuple[Path, str]] = []

    async def upload_file(self, source_path: Path, target_path: str) -> None:
        self.upload_calls.append((source_path, target_path))

    async def exec(self, command: str, env: dict[str, str] | None = None) -> MockExecResult:
        self.exec_calls.append((command, env or {}))
        return self.result

    async def download_file(self, source_path: str, target_path: Path) -> None:
        raise FileNotFoundError(source_path)


@pytest.mark.anyio
async def test_run_raises_nonzero_agent_exit_with_viewer_compatible_command(
    tmp_path: Path,
) -> None:
    agent = O11yBenchAgent(logs_dir=tmp_path, model_name="anthropic/claude-opus-4-6")
    env: Any = MockEnvironment(MockExecResult(return_code=1))
    context = AgentContext()

    with pytest.raises(NonZeroAgentExitCodeError, match="Agent exited with code 1"):
        await agent.run("do the thing", env, context)

    assert env.exec_calls
    command, _ = env.exec_calls[0]
    assert VIEWER_COMMAND_STDOUT_PATH in command


@pytest.mark.anyio
async def test_run_passes_through_scenario_clock_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("O11Y_SCENARIO_TIME_ISO", "2026-04-04T10:05:14Z")
    agent = O11yBenchAgent(logs_dir=tmp_path, model_name="anthropic/claude-opus-4-6")
    env: Any = MockEnvironment(MockExecResult(return_code=1))
    context = AgentContext()

    with pytest.raises(NonZeroAgentExitCodeError):
        await agent.run("do the thing", env, context)

    assert env.exec_calls
    _, call_env = env.exec_calls[0]
    assert call_env["O11Y_SCENARIO_TIME_ISO"] == "2026-04-04T10:05:14Z"


@pytest.mark.anyio
async def test_langchain_agent_setup_uploads_langchain_runner(tmp_path: Path) -> None:
    agent = LangChainO11yBenchAgent(logs_dir=tmp_path, model_name="openai/gpt-5.4-nano")
    env: Any = MockEnvironment(MockExecResult(return_code=0))

    await agent.setup(env)

    assert env.upload_calls
    uploaded_names = {source.name for source, _target in env.upload_calls}
    assert "langchain_agent_runner.py" in uploaded_names
    assert "system_prompt.txt" in uploaded_names
    assert "task_prompt.txt" in uploaded_names
