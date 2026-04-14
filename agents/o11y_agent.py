"""Custom Harbor agent: o11y-bench system prompt + MCP tools via litellm.

The agent loop runs inside the container (like Claude Code) because the MCP
server is only reachable via Docker networking. Uses a uv inline script with
litellm for multi-provider LLM support and the mcp SDK for tool calls.

Usage:
  harbor run --yes -c job.yaml -p tasks/query-cpu-metrics
  harbor run --yes -c job.yaml -m anthropic/claude-sonnet-4-6
"""

import json
import os
import shlex
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from harbor.agents.base import BaseAgent
from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

RUNNER_SCRIPT = Path(__file__).parent / "agent_runner.py"
SYSTEM_PROMPT = Path(__file__).parent / "system_prompt.txt"
TASK_PROMPT = Path(__file__).parent / "task_prompt.txt"
VIEWER_COMMAND_STDOUT_PATH = "/logs/agent/command-0/stdout.txt"


def normalize_litellm_model_name(model_name: str) -> str:
    if model_name.startswith("google/"):
        return f"gemini/{model_name.split('/', 1)[1]}"
    return model_name


def select_remote_mcp_url(mcp_servers: list[Any]) -> str | None:
    """Prefer an explicitly configured non-loopback MCP URL over task defaults."""
    for server in mcp_servers:
        url = getattr(server, "url", None)
        if not isinstance(url, str) or not url:
            continue
        host = (urlparse(url).hostname or "").lower()
        if host in {"localhost", "127.0.0.1", "::1", "o11y-stack"}:
            continue
        return url
    return None


def build_runner_command() -> str:
    command = (
        "set -o pipefail; "
        "mkdir -p /logs/agent/command-0; "
        f'uv run /app/agent_runner.py 2>&1 | tee "{VIEWER_COMMAND_STDOUT_PATH}"'
    )
    return f"bash -lc {shlex.quote(command)}"


class O11yBenchAgent(BaseAgent):
    """Agent that runs the o11y-bench harness inside the container."""

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        reasoning_effort: str = "off",
        temperature: float = 0.0,
        extra_env: dict[str, str] | None = None,
        **kwargs: Any,
    ):
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self._extra_env = extra_env or {}

    @staticmethod
    def name() -> str:
        return "o11y-bench"

    def version(self) -> str:
        return "1.0.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        await environment.exec(command="mkdir -p /app/agents")
        await environment.upload_file(
            source_path=RUNNER_SCRIPT,
            target_path="/app/agent_runner.py",
        )
        await environment.upload_file(
            source_path=SYSTEM_PROMPT,
            target_path="/app/system_prompt.txt",
        )
        await environment.upload_file(
            source_path=TASK_PROMPT,
            target_path="/app/task_prompt.txt",
        )

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        requested_model = self.model_name or "anthropic/claude-sonnet-4-6"
        model = normalize_litellm_model_name(requested_model)

        mcp_url = select_remote_mcp_url(self.mcp_servers)

        instruction_path = self.logs_dir / "instruction.txt"
        instruction_path.write_text(instruction)
        await environment.upload_file(
            source_path=instruction_path,
            target_path="/app/instruction.txt",
        )

        env: dict[str, str] = {
            "MODEL": model,
            "REASONING_EFFORT": self.reasoning_effort,
            "TEMPERATURE": str(self.temperature),
            "PATH": "/usr/local/bin:/usr/local/sbin:/usr/bin:/usr/sbin:/bin:/sbin",
            # uv may run the runner from a cache dir; keep uploads under /app importable.
            "PYTHONPATH": "/app",
            "O11Y_SCENARIO_TIME_ISO": os.environ.get("O11Y_SCENARIO_TIME_ISO", ""),
        }
        for key in (
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "OPENAI_API_BASE",
            "OPENROUTER_API_KEY",
        ):
            val = os.environ.get(key)
            if val:
                env[key] = val

        gemini_api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if gemini_api_key:
            env["GEMINI_API_KEY"] = gemini_api_key
            env["GOOGLE_API_KEY"] = gemini_api_key
        if mcp_url:
            env["MCP_URL"] = mcp_url
        env.update(self._extra_env)

        self.logger.info(f"Running agent runner with model={model}")
        result = await environment.exec(command=build_runner_command(), env=env)

        try:
            await environment.download_file(
                source_path="/logs/agent/trajectory.json",
                target_path=self.logs_dir / "trajectory.json",
            )
            trajectory = json.loads((self.logs_dir / "trajectory.json").read_text())
            fm = trajectory.get("final_metrics") or {}
            context.n_input_tokens = fm.get("total_prompt_tokens")
            context.n_output_tokens = fm.get("total_completion_tokens")
            context.n_cache_tokens = fm.get("total_cached_tokens")
            context.cost_usd = fm.get("total_cost_usd")
        except Exception as e:
            self.logger.warning(f"Could not read trajectory metrics: {e}")

        context.metadata = {
            **(context.metadata or {}),
            "model": model,
            "requested_model": requested_model,
            "reasoning_effort": self.reasoning_effort,
        }

        if result.return_code != 0:
            message = f"Agent exited with code {result.return_code}"
            self.logger.error(message)
            raise NonZeroAgentExitCodeError(message)
