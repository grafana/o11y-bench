"""LangChain-based Harbor agent for o11y-bench."""

from pathlib import Path

from harbor.environments.base import BaseEnvironment

from .o11y_agent import (
    SYSTEM_PROMPT,
    TASK_PROMPT,
    O11yBenchAgent,
)

RUNNER_SCRIPT = Path(__file__).parent / "langchain_agent_runner.py"


class LangChainO11yBenchAgent(O11yBenchAgent):
    """Alternate o11y-bench agent implemented with a LangChain runner."""

    @staticmethod
    def name() -> str:
        return "o11y-bench-langchain"

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
