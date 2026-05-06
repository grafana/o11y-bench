"""OpenCode agent without MCP tools — uses gcx CLI instead.

Harbor passes MCP server configs from task.toml to every agent, including
built-in ones like opencode. Without this subclass, opencode would have both
MCP tools (from mcp-grafana) and gcx CLI access, which defeats the purpose of
benchmarking gcx-only interaction. Clearing mcp_servers ensures the agent can
only reach Grafana through gcx.
"""

from typing import Any

from harbor.agents.installed.opencode import OpenCode
from harbor.models.trajectories import Trajectory


class GcxOpenCodeAgent(OpenCode):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.mcp_servers = []

    def get_version_command(self) -> str | None:
        return ". ~/.nvm/nvm.sh; echo opencode $(opencode --version); echo $(gcx --version)"

    def _convert_events_to_trajectory(self, events: list[dict[str, Any]]) -> Trajectory | None:
        """Inject a synthetic step_finish when the agent was killed mid-step.

        The base class only records a turn when step_finish is seen, so
        timeouts silently drop the in-progress turn. Here, we force a
        step_finish if one doesn't exist, so we always write the trajectory,
        even in the event of a timeout on the job.
        """
        if events:
            has_open_step = False
            for event in events:
                etype = event.get("type")
                if etype == "step_start":
                    has_open_step = True
                elif etype == "step_finish":
                    has_open_step = False
            if has_open_step:
                events = [*events, {"type": "step_finish", "part": {}}]

        return super()._convert_events_to_trajectory(events)
