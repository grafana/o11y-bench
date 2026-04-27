"""OpenCode agent without MCP tools — uses gcx CLI instead.

Harbor passes MCP server configs from task.toml to every agent, including
built-in ones like opencode. Without this subclass, opencode would have both
MCP tools (from mcp-grafana) and gcx CLI access, which defeats the purpose of
benchmarking gcx-only interaction. Clearing mcp_servers ensures the agent can
only reach Grafana through gcx.
"""

from typing import Any

from harbor.agents.installed.opencode import OpenCode


class GcxOpenCodeAgent(OpenCode):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.mcp_servers = []
