# /// script
# dependencies = [
#   "litellm==1.83.10",
#   "mcp>=1.9.0",
# ]
# ///
"""o11y-bench agent runner — executes inside the Harbor container.

Connects to mcp-grafana via streamable-http, runs an agent loop using litellm
(multi-provider: Anthropic, OpenAI, Google, etc.), and writes an ATIF trajectory.

Config via env vars: MODEL, MCP_URL, REASONING_EFFORT, TEMPERATURE, provider API keys.
For OpenAI-compatible endpoints (local or hosted), use ``openai/<id>`` + ``OPENAI_API_BASE``
(and ``OPENAI_API_KEY`` if required). See LiteLLM provider docs for other routes.
Timeout is handled by Harbor's agent timeout in task.toml.

Prompt caching:
- Anthropic: litellm's cache_control_injection_points (auto-injects cache breakpoints)
- OpenAI: automatic for 1024+ token prompts (no config needed)
- Gemini: server-side caching handled by Google (no config needed)
"""

import asyncio
import copy
import json
import os
import random
import re
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, cast


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
MAX_AGENT_STEPS = 50
# Same table as reporting/model_costs.py (task image avoids that import); keep aligned.
USD_PER_1M_TOKENS: Final[dict[str, tuple[float, float, float]]] = {
    "gpt-5.4-mini": (0.75, 0.075, 4.50),
    "gpt-5.4-mini-2026-03-17": (0.75, 0.075, 4.50),
    "gpt-5.4-nano": (0.20, 0.02, 1.25),
    "gpt-5.4-nano-2026-03-17": (0.20, 0.02, 1.25),
}


def normalize_model_name(model_name: str) -> str:
    if "/" in model_name:
        return model_name.rsplit("/", 1)[-1]
    return model_name


def estimate_cost_usd(
    model_name: str, n_input_tokens: int, n_cache_tokens: int, n_output_tokens: int
) -> float | None:
    rates = USD_PER_1M_TOKENS.get(normalize_model_name(model_name))
    if rates is None:
        return None

    input_rate, cached_input_rate, output_rate = rates
    uncached_input_tokens = max(0, n_input_tokens - n_cache_tokens)
    return (
        (uncached_input_tokens * input_rate)
        + (n_cache_tokens * cached_input_rate)
        + (n_output_tokens * output_rate)
    ) / 1_000_000


def relax_mcp_tool_input_schema_for_llm(schema: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy MCP JSON Schema so bare ``type: object`` nodes allow arbitrary keys.

    mcp-grafana exposes fields like ``dashboard`` as ``{"type": "object"}`` without
    ``properties``; some OpenAI-compatible models omit or refuse to fill those unless
    ``additionalProperties`` is set.
    """

    out = copy.deepcopy(schema)

    def walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        if node.get("type") == "object":
            props = node.get("properties")
            if not props:
                node.setdefault("additionalProperties", True)
            else:
                for child in props.values():
                    walk(child)
        elif node.get("type") == "array" and "items" in node:
            walk(node["items"])
        for key in ("anyOf", "oneOf", "allOf"):
            if key in node and isinstance(node[key], list):
                for item in node[key]:
                    walk(item)

    walk(out)
    return out


async def discover_tools(session: Any) -> list[dict[str, Any]]:
    """Convert MCP tools to OpenAI function-calling format for litellm."""
    result = await session.list_tools()
    tools: list[dict[str, Any]] = []
    for t in result.tools:
        params: Any = t.inputSchema
        if isinstance(params, dict):
            params = relax_mcp_tool_input_schema_for_llm(params)
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": params,
                },
            }
        )
    return tools


async def call_mcp_tool(session: Any, name: str, arguments: dict[str, Any]) -> str:
    """Call an MCP tool and return text content."""
    result = await session.call_tool(name, arguments)
    texts = [item.text for item in result.content if hasattr(item, "text")]
    return "\n".join(texts) if texts else json.dumps(str(result.content))


def make_atif_step(
    step_id: int,
    source: str,
    message: str | list[dict[str, Any]],
    *,
    reasoning_content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    observation: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an ATIF-v1.6 step dict."""
    step: dict[str, Any] = {
        "step_id": step_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "source": source,
        "message": message,
    }
    if reasoning_content is not None:
        step["reasoning_content"] = reasoning_content
    if tool_calls:
        step["tool_calls"] = tool_calls
    if observation:
        step["observation"] = observation
    if metrics:
        step["metrics"] = metrics
    return step


RETRYABLE_UPSTREAM_STATUS_CODES: Final[frozenset[int]] = frozenset({429, 503, 529})
RETRYABLE_UPSTREAM_TEXT_RE: Final[re.Pattern[str]] = re.compile(
    r"""
    rate_limit
    | rate \s+ limit
    | overloaded(?:_error)?
    | temporarily \s+ unavailable
    | \b(?:http(?: \s+ status)?|status(?: \s+ code)?|server \s+ error)
      (?:\s+|=|:)+['"]?(?:429|503|529)['"]?\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _retryable_status_codes(exc: Exception) -> tuple[int, ...]:
    response = getattr(exc, "response", None)
    codes = (
        getattr(exc, "status_code", None),
        getattr(response, "status_code", None) if response is not None else None,
    )
    return tuple(code for code in codes if isinstance(code, int))


def is_retryable_upstream_error(exc: Exception) -> bool:
    """Return True for transient, retryable upstream capacity or rate-limit errors."""
    if any(code in RETRYABLE_UPSTREAM_STATUS_CODES for code in _retryable_status_codes(exc)):
        return True

    text = str(exc).lower()
    return RETRYABLE_UPSTREAM_TEXT_RE.search(text) is not None


def retry_delay_seconds(exc: Exception, attempt: int) -> float:
    response = getattr(exc, "response", None)
    if response is not None:
        headers = getattr(response, "headers", {}) or {}
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if retry_after:
            try:
                return max(1.0, float(retry_after))
            except ValueError:
                pass

    return float(min(60.0, 5.0 * (2**attempt)) + random.uniform(0.0, 1.0))


def parse_tool_arguments(arguments: Any) -> dict[str, Any]:
    if not isinstance(arguments, str):
        return arguments if isinstance(arguments, dict) else {"_raw": arguments}
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return {"_raw": arguments}
    return parsed if isinstance(parsed, dict) else {"_raw": parsed}


def enforce_step_limit(step: int) -> None:
    if step > MAX_AGENT_STEPS:
        raise RuntimeError(f"Agent exceeded max step limit ({MAX_AGENT_STEPS})")


async def completion_with_retries(
    litellm_module: Any,
    messages: list[dict[str, Any]],
    litellm_kwargs: dict[str, Any],
    max_retries: int = 5,
) -> Any:
    for attempt in range(max_retries + 1):
        try:
            return await litellm_module.acompletion(messages=messages, **litellm_kwargs)
        except Exception as e:
            if not is_retryable_upstream_error(e) or attempt == max_retries:
                raise

            delay = retry_delay_seconds(e, attempt)
            print(f"retry({delay:.1f}s)", end=" ", flush=True)
            await asyncio.sleep(delay)

    raise RuntimeError("completion retries exhausted")


async def run_agent() -> None:
    import litellm
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    model = os.environ["MODEL"]
    stack_host = os.environ.get("STACK_HOST", "127.0.0.1")
    mcp_url = os.environ.get("MCP_URL", f"http://{stack_host}:8080/mcp")
    reasoning_effort = os.environ.get("REASONING_EFFORT", "off")
    temperature = float(os.environ.get("TEMPERATURE", "0.0"))

    statement = Path("/app/instruction.txt").read_text().strip()

    env_ts = scenario_clock_iso()
    task_prompt = TASK_PROMPT_TEMPLATE.format(
        current_time=env_ts,
        statement=statement,
    )

    litellm.suppress_debug_info = True

    agent_dir = Path("/logs/agent")
    agent_dir.mkdir(parents=True, exist_ok=True)

    session_id = str(uuid.uuid4())
    atif_steps: list[dict[str, Any]] = []
    tool_defs: list[dict[str, Any]] = []
    stats = {"input": 0, "output": 0, "cache": 0, "cost": 0.0}
    step_id = 0
    tool_call_count = 0
    start = time.time()

    def flush_trajectory() -> None:
        """Write trajectory to disk after each step so partial work survives kills."""
        trajectory = {
            "schema_version": "ATIF-v1.6",
            "session_id": session_id,
            "agent": {
                "name": "o11y-bench",
                "version": "1.0.0",
                "model_name": model,
                "tool_definitions": tool_defs,
            },
            "steps": atif_steps,
            "final_metrics": {
                "total_prompt_tokens": stats["input"],
                "total_completion_tokens": stats["output"],
                "total_cached_tokens": stats["cache"],
                "total_cost_usd": stats["cost"],
                "total_steps": step_id,
                "total_tool_calls": tool_call_count,
                "reasoning_effort": reasoning_effort,
                "elapsed_seconds": time.time() - start,
            },
        }
        (agent_dir / "trajectory.json").write_text(json.dumps(trajectory, indent=2))

    try:
        print(f"Connecting to MCP at {mcp_url}...")
        async with streamable_http_client(mcp_url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await discover_tools(session)
                tool_defs[:] = [t["function"] for t in tools]
                print(f"Discovered {len(tools)} tools, model={model}")

                step_id += 1
                atif_steps.append(make_atif_step(step_id, "system", SYSTEM_PROMPT))
                step_id += 1
                atif_steps.append(make_atif_step(step_id, "user", task_prompt))

                messages: list[dict[str, Any]] = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": task_prompt},
                ]

                litellm_kwargs: dict[str, Any] = {
                    "model": model,
                    "tools": tools,
                    "drop_params": True,
                    "cache_control_injection_points": [
                        {"location": "message", "role": "system"},
                    ],
                }
                if reasoning_effort != "off":
                    litellm_kwargs["reasoning_effort"] = reasoning_effort
                else:
                    litellm_kwargs["temperature"] = temperature

                step = 0
                while True:
                    step += 1
                    enforce_step_limit(step)
                    print(f"[{step}]", end=" ", flush=True)

                    resp = await completion_with_retries(litellm, messages, litellm_kwargs)

                    u = cast(Any, resp).usage
                    prompt_tokens = 0
                    completion_tokens = 0
                    cached_tokens = 0
                    if u:
                        prompt_tokens = getattr(u, "prompt_tokens", 0) or 0
                        completion_tokens = getattr(u, "completion_tokens", 0) or 0
                        stats["input"] += prompt_tokens
                        stats["output"] += completion_tokens
                        if hasattr(u, "prompt_tokens_details") and u.prompt_tokens_details:
                            cached_tokens = (
                                getattr(u.prompt_tokens_details, "cached_tokens", 0) or 0
                            )
                            stats["cache"] += cached_tokens
                    try:
                        step_cost = litellm.completion_cost(completion_response=resp) or 0.0
                    except Exception:
                        step_cost = 0.0
                    if step_cost <= 0:
                        estimated = estimate_cost_usd(
                            model,
                            int(prompt_tokens),
                            int(cached_tokens),
                            int(completion_tokens),
                        )
                        if estimated is not None:
                            step_cost = estimated
                    stats["cost"] += step_cost

                    msg = cast(Any, resp).choices[0].message
                    content = msg.content or ""
                    tool_calls = msg.tool_calls or []
                    reasoning = getattr(msg, "reasoning_content", None)

                    # Build ATIF tool_calls
                    atif_tool_calls = [
                        {
                            "tool_call_id": tc.id,
                            "function_name": tc.function.name,
                            "arguments": parse_tool_arguments(tc.function.arguments),
                        }
                        for tc in tool_calls
                    ]

                    if not tool_calls:
                        step_id += 1
                        atif_steps.append(
                            make_atif_step(
                                step_id,
                                "agent",
                                content,
                                reasoning_content=reasoning,
                                metrics={
                                    "prompt_tokens": prompt_tokens,
                                    "completion_tokens": completion_tokens,
                                    "cached_tokens": cached_tokens,
                                    "cost_usd": step_cost,
                                },
                            )
                        )
                        flush_trajectory()
                        print("done")
                        break

                    messages.append(msg.model_dump())
                    tool_call_count += len(tool_calls)

                    # Execute tool calls and collect observations
                    observation_results: list[dict[str, Any]] = []
                    for tc in tool_calls:
                        fn = tc.function.name or ""
                        fa = parse_tool_arguments(tc.function.arguments)
                        try:
                            out = await call_mcp_tool(session, fn, fa)
                            print(f"{fn}({len(out)})", end=" ", flush=True)
                        except Exception as e:
                            out = f"Error: {e}"
                            print(f"{fn}(ERR)", end=" ", flush=True)

                        observation_results.append({"source_call_id": tc.id, "content": out})
                        messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})

                    step_id += 1
                    atif_steps.append(
                        make_atif_step(
                            step_id,
                            "agent",
                            content,
                            reasoning_content=reasoning,
                            tool_calls=atif_tool_calls,
                            observation={"results": observation_results},
                            metrics={
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "cached_tokens": cached_tokens,
                                "cost_usd": step_cost,
                            },
                        )
                    )
                    flush_trajectory()
                    print()
    finally:
        elapsed = time.time() - start
        print(f"\n{elapsed:.1f}s | {stats['input']}in {stats['output']}out ${stats['cost']:.4f}")
        flush_trajectory()


if __name__ == "__main__":
    asyncio.run(run_agent())
