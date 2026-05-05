# /// script
# dependencies = [
#   "langchain>=1.0.0",
#   "langchain-anthropic>=0.3.0",
#   "langchain-google-genai>=2.0.0",
#   "langchain-mcp-adapters>=0.1.0",
#   "langchain-openai>=0.3.0",
# ]
# ///
"""Barebones LangChain runner for a custom Harbor agent example."""

import asyncio
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient


def scenario_clock_iso() -> str:
    env_ts = os.environ.get("O11Y_SCENARIO_TIME_ISO", "").strip()
    if env_ts:
        return env_ts
    return datetime.now(UTC).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_system_prompt() -> str:
    return Path(__file__).with_name("system_prompt.txt").read_text().strip()


def _load_task_prompt_template() -> str:
    return Path(__file__).with_name("task_prompt.txt").read_text().strip()


SYSTEM_PROMPT = _load_system_prompt()
TASK_PROMPT_TEMPLATE = _load_task_prompt_template()


def normalize_langchain_model_name(model_name: str) -> str:
    provider, sep, bare = model_name.partition("/")
    if not sep:
        return model_name
    if provider == "google":
        provider = "google_genai"
    return f"{provider}:{bare}"


def message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part)
    return str(content)


def usage_counts(message: BaseMessage) -> tuple[int, int]:
    usage = getattr(message, "usage_metadata", None)
    if not isinstance(usage, dict):
        return 0, 0
    input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
    output_tokens = usage.get("output_tokens") or usage.get("completion_tokens") or 0
    return int(input_tokens), int(output_tokens)


def make_atif_step(
    step_id: int,
    source: str,
    message: str,
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    observation: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    step: dict[str, Any] = {
        "step_id": step_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "source": source,
        "message": message,
    }
    if tool_calls:
        step["tool_calls"] = tool_calls
    if observation:
        step["observation"] = observation
    if metrics:
        step["metrics"] = metrics
    return step


def convert_messages_to_steps(
    messages: list[BaseMessage],
    *,
    task_prompt: str,
    model_name: str,
    tool_defs: list[dict[str, Any]],
) -> dict[str, Any]:
    steps: list[dict[str, Any]] = [
        make_atif_step(1, "system", SYSTEM_PROMPT),
        make_atif_step(2, "user", task_prompt),
    ]
    step_id = 2
    tool_call_count = 0
    total_input_tokens = 0
    total_output_tokens = 0

    pending_ai: AIMessage | None = None
    pending_tool_results: list[dict[str, Any]] = []

    def flush_pending() -> None:
        nonlocal pending_ai, pending_tool_results, step_id, tool_call_count
        nonlocal total_input_tokens, total_output_tokens
        if pending_ai is None:
            return

        tool_calls_data = [
            {
                "tool_call_id": tool_call.get("id", ""),
                "function_name": tool_call.get("name", ""),
                "arguments": tool_call.get("args", {}),
            }
            for tool_call in pending_ai.tool_calls
        ]
        input_tokens, output_tokens = usage_counts(pending_ai)
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        step_id += 1
        steps.append(
            make_atif_step(
                step_id,
                "agent",
                message_text(pending_ai.content) or "(tool use)",
                tool_calls=tool_calls_data or None,
                observation={"results": pending_tool_results} if pending_tool_results else None,
                metrics={
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                },
            )
        )
        tool_call_count += len(tool_calls_data)
        pending_ai = None
        pending_tool_results = []

    for message in messages:
        if isinstance(message, SystemMessage | HumanMessage):
            continue
        if isinstance(message, AIMessage):
            flush_pending()
            pending_ai = message
            continue
        if isinstance(message, ToolMessage):
            pending_tool_results.append(
                {
                    "source_call_id": message.tool_call_id,
                    "content": message_text(message.content),
                }
            )

    flush_pending()
    session_id = str(uuid.uuid4())

    return {
        "schema_version": "ATIF-v1.7",
        "session_id": session_id,
        "trajectory_id": str(uuid.uuid4()),
        "agent": {
            "name": "o11y-bench-langchain",
            "version": "1.0.0",
            "model_name": model_name,
            "tool_definitions": tool_defs,
        },
        "steps": steps,
        "final_metrics": {
            "total_prompt_tokens": total_input_tokens,
            "total_completion_tokens": total_output_tokens,
            "total_cached_tokens": 0,
            "total_cost_usd": 0.0,
            "total_steps": len(steps),
            "extra": {
                "total_tool_calls": tool_call_count,
                "reasoning_effort": os.environ.get("REASONING_EFFORT", "off"),
            },
        },
    }


async def run_agent() -> None:
    requested_model = os.environ["MODEL"]
    model = normalize_langchain_model_name(requested_model)
    stack_host = os.environ.get("STACK_HOST", "127.0.0.1")
    mcp_url = os.environ.get("MCP_URL", f"http://{stack_host}:8080/mcp")
    statement = Path("/app/instruction.txt").read_text().strip()
    task_prompt = TASK_PROMPT_TEMPLATE.format(
        current_time=scenario_clock_iso(),
        statement=statement,
    )

    if os.environ.get("OPENAI_API_BASE") and not os.environ.get("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = os.environ["OPENAI_API_BASE"]

    client = MultiServerMCPClient(
        {
            "grafana": {
                "transport": "http",
                "url": mcp_url,
            }
        }
    )
    tools = await client.get_tools()
    agent = create_agent(
        model=init_chat_model(model),
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )
    result = await agent.ainvoke({"messages": [{"role": "user", "content": task_prompt}]})

    messages = [
        message for message in result.get("messages", []) if isinstance(message, BaseMessage)
    ]
    trajectory = convert_messages_to_steps(
        messages,
        task_prompt=task_prompt,
        model_name=model,
        tool_defs=[{"name": tool.name, "description": tool.description or ""} for tool in tools],
    )

    agent_dir = Path("/logs/agent")
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "trajectory.json").write_text(json.dumps(trajectory, indent=2))

    final_text = ""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            final_text = message_text(message.content)
            if final_text:
                break
    print(final_text)


if __name__ == "__main__":
    asyncio.run(run_agent())
