import re
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

from grading.models import ToolCall, Transcript

# Parsed from the first user turn: agent_runner injects Current time in <context>…</context>.
_SCENARIO_TIME_RE = re.compile(
    r"Current time:\s*(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)",
    re.IGNORECASE,
)

_TRACE_ID_HEX_32 = re.compile(r"(?<![0-9a-fA-F])([0-9a-fA-F]{32})(?![0-9a-fA-F])")
# Tempo search JSON often omits a leading zero (31 hex chars); get-trace uses 32.
_TRACE_ID_JSON = re.compile(
    r'"(?:traceID|traceId)"\s*:\s*"([0-9a-fA-F]{31,32})"',
    re.IGNORECASE,
)


def _first_user_message_text(transcript: Transcript) -> str | None:
    for msg in transcript.messages:
        if msg.role == "user" and msg.content:
            return str(msg.content).strip()
    return None


def prometheus_eval_time_unix(params: dict[str, Any], transcript: Transcript) -> float | None:
    raw = params.get("time_unix")
    if raw is not None:
        return float(raw)

    user_text = _first_user_message_text(transcript)
    if not user_text:
        return None

    match = _SCENARIO_TIME_RE.search(user_text)
    if not match:
        return None

    iso = match.group(1)
    dt = datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    return dt.timestamp()


def final_assistant_text(transcript: Transcript) -> str | None:
    for msg in reversed(transcript.messages):
        if msg.role == "assistant" and msg.content:
            return msg.content
    return None


def assistant_scope_note(assistant_scope: str) -> str:
    return "all assistant turns" if assistant_scope.strip().lower() == "all" else "final response"


def assistant_text_blobs(transcript: Transcript, assistant_scope: str) -> list[str]:
    scope = (assistant_scope or "final").strip().lower()
    if scope not in {"final", "all"}:
        scope = "final"

    if scope == "final":
        text = final_assistant_text(transcript)
        return [text] if text else []

    blobs: list[str] = []
    for msg in transcript.messages:
        if msg.role == "assistant" and msg.content:
            blobs.append(msg.content)
    return blobs


def iter_tool_calls(transcript: Transcript) -> Iterator[ToolCall]:
    for msg in transcript.messages:
        if msg.role == "assistant" and msg.tool_calls:
            yield from msg.tool_calls


def tool_call_id_to_name(transcript: Transcript) -> dict[str, str]:
    out: dict[str, str] = {}
    for tool_call in iter_tool_calls(transcript):
        out[tool_call.id] = tool_call.name
    return out


def as_name_set(raw: str | list[str] | tuple[str, ...] | set[str] | None) -> set[str]:
    if raw is None:
        return set()
    if isinstance(raw, str):
        return {raw}
    return {str(value) for value in raw}


def require_stack_url(url: str, env_name: str) -> str | None:
    if url:
        return None
    return f"{env_name} is not set."


def trace_ids_from_tool_content(content: str) -> set[str]:
    ids: set[str] = set()
    for regex in (_TRACE_ID_HEX_32, _TRACE_ID_JSON):
        for match in regex.finditer(content):
            raw = match.group(1).lower()
            ids.add(raw)
            if len(raw) == 31:
                ids.add(raw.zfill(32))
    return ids


def trace_id_variants_for_prefix_match(candidate_ids: set[str]) -> set[str]:
    out: set[str] = set()
    for trace_id in candidate_ids:
        lowered = trace_id.lower()
        out.add(lowered)
        stripped = lowered.lstrip("0")
        if stripped and stripped != lowered:
            out.add(stripped)
    return out


def response_cites_trace_id_prefix(
    text_blobs: list[str],
    candidate_ids: set[str],
    prefix_min_chars: int,
) -> tuple[bool, str | None]:
    expanded = trace_id_variants_for_prefix_match(candidate_ids)
    for response_text in text_blobs:
        haystack = response_text.lower()
        for trace_id in expanded:
            if len(trace_id) < prefix_min_chars:
                continue
            if not re.fullmatch(r"[0-9a-f]+", trace_id):
                continue
            for prefix_len in range(len(trace_id), prefix_min_chars - 1, -1):
                prefix = trace_id[:prefix_len]
                if prefix in haystack:
                    return True, prefix
    return False, None


def tempo_tool_matches_name(
    tool_name: str,
    allowed_names: set[str] | None,
    name_prefix: str,
) -> bool:
    if allowed_names is not None:
        return tool_name in allowed_names
    return tool_name.startswith(name_prefix)


def additional_trace_id_tool_names(params: dict[str, Any]) -> set[str]:
    return as_name_set(params.get("additional_tool_names"))
