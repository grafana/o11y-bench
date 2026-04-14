from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from grading.models import Message, ToolCall, ToolResult, Transcript


def parse_claude_code_jsonl(log_path: Path) -> Transcript:
    """Parse Claude Code stream-json JSONL transcript."""
    messages: list[Message] = []
    for line in log_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        event_type = event.get("type", "")

        if event_type == "system":
            messages.append(Message(role="system", content=event.get("message", "")))

        elif event_type == "user":
            messages.append(Message(role="user", content=event.get("message", "")))

        elif event_type == "assistant":
            msg = event.get("message", {})
            content_blocks = msg.get("content", [])
            text_parts = []
            thinking_parts = []
            tool_calls = []

            for block in content_blocks:
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "thinking":
                    thinking_parts.append(block.get("thinking", ""))
                elif block.get("type") == "tool_use":
                    tool_calls.append(
                        ToolCall(
                            id=block.get("id", ""),
                            name=block.get("name", ""),
                            arguments=block.get("input", {}),
                        )
                    )

            messages.append(
                Message(
                    role="assistant",
                    content="\n".join(text_parts) if text_parts else None,
                    thinking_content="\n".join(thinking_parts) if thinking_parts else None,
                    tool_calls=tool_calls or None,
                )
            )

        elif event_type == "result":
            # Final result message from Claude Code
            result_msg = event.get("result", "")
            if result_msg:
                # Check if last message is assistant; if so, append content
                if messages and messages[-1].role == "assistant" and not messages[-1].content:
                    messages[-1].content = result_msg
                else:
                    messages.append(Message(role="assistant", content=result_msg))

        elif event_type == "tool_result":
            tool_result = ToolResult(
                tool_call_id=event.get("tool_use_id", ""),
                content=event.get("content", ""),
            )
            # Handle string or list content
            content = event.get("content", "")
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        text_parts.append(part)
                tool_result.content = "\n".join(text_parts)
            elif isinstance(content, str):
                tool_result.content = content
            else:
                tool_result.content = str(content)

            messages.append(
                Message(
                    role="tool",
                    tool_results=[tool_result],
                )
            )

    return Transcript(messages=messages)


def _parse_atif_steps(steps: list[dict[str, Any]]) -> list[Message]:
    """Convert ATIF steps into Message objects."""
    messages: list[Message] = []
    for step in steps:
        source = step.get("source", "")
        message_text = step.get("message", "")
        tool_calls_data = step.get("tool_calls", [])
        observation = step.get("observation")

        if source == "user":
            messages.append(Message(role="user", content=message_text or ""))
        elif source in ("agent", "system"):
            role = "assistant" if source == "agent" else "system"
            tool_calls = []
            for tc in tool_calls_data:
                tool_calls.append(
                    ToolCall(
                        id=tc.get("tool_call_id", tc.get("id", "")),
                        name=tc.get("function_name", tc.get("name", "")),
                        arguments=tc.get("arguments", {}),
                    )
                )
            # Only treat the message as content if it's not a placeholder like "(tool use)"
            content = message_text if message_text and message_text != "(tool use)" else None
            messages.append(
                Message(
                    role=role,
                    content=content,
                    tool_calls=tool_calls or None,
                    thinking_content=step.get("reasoning_content"),
                )
            )
            # Emit tool results from observation as separate tool messages
            if observation and "results" in observation:
                for result in observation["results"]:
                    messages.append(
                        Message(
                            role="tool",
                            tool_results=[
                                ToolResult(
                                    tool_call_id=result.get(
                                        "source_call_id", result.get("tool_call_id", "")
                                    ),
                                    content=str(result.get("content", "")),
                                )
                            ],
                        )
                    )
    return messages


def parse_atif_trajectory(logs_dir: Path) -> Transcript:
    """Parse ATIF (Agent Trajectory Interchange Format) trajectory."""
    messages: list[Message] = []

    # Look for trajectory files
    trajectory_files = sorted(logs_dir.glob("*.jsonl")) + sorted(logs_dir.glob("*.json"))

    for traj_file in trajectory_files:
        content = traj_file.read_text()

        # Try as a single JSON object with a "steps" array (ATIF trajectory.json)
        try:
            data = json.loads(content)
            if isinstance(data, dict) and "steps" in data:
                messages.extend(_parse_atif_steps(data["steps"]))
                continue
        except json.JSONDecodeError:
            pass

        # Fall back to JSONL (one entry per line)
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Handle JSONL where each line is an ATIF step
            if "source" in entry and entry.get("source") in ("agent", "user", "system"):
                messages.extend(_parse_atif_steps([entry]))
                continue

            role = entry.get("role", "")
            if role == "user":
                messages.append(Message(role="user", content=entry.get("content", "")))
            elif role == "assistant":
                content_val = entry.get("content", "")
                tool_calls = []
                if "tool_calls" in entry:
                    for tc in entry["tool_calls"]:
                        tool_calls.append(
                            ToolCall(
                                id=tc.get("id", ""),
                                name=tc.get("function", {}).get("name", tc.get("name", "")),
                                arguments=tc.get("function", {}).get(
                                    "arguments", tc.get("arguments", {})
                                ),
                            )
                        )
                messages.append(
                    Message(
                        role="assistant",
                        content=content_val if isinstance(content_val, str) else None,
                        tool_calls=tool_calls or None,
                    )
                )
            elif role == "tool":
                messages.append(
                    Message(
                        role="tool",
                        tool_results=[
                            ToolResult(
                                tool_call_id=entry.get("tool_call_id", ""),
                                content=str(entry.get("content", "")),
                            )
                        ],
                    )
                )

    return Transcript(messages=messages)


def parse_transcript(logs_dir: Path) -> Transcript:
    """Parse agent transcript from logs directory, auto-detecting format."""
    # Claude Code: look for claude-code.txt or stream.jsonl
    for name in ("claude-code.txt", "stream.jsonl", "transcript.jsonl"):
        candidate = logs_dir / name
        if candidate.exists() and candidate.stat().st_size > 0:
            return parse_claude_code_jsonl(candidate)

    # ATIF or other JSONL files
    jsonl_files = list(logs_dir.glob("*.jsonl"))
    if jsonl_files:
        return parse_atif_trajectory(logs_dir)

    # Fallback: try any text file
    for f in sorted(logs_dir.iterdir()):
        if f.is_file() and f.stat().st_size > 0:
            try:
                return parse_atif_trajectory(logs_dir)
            except Exception:
                pass

    return Transcript(messages=[])
