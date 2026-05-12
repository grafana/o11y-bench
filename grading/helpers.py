import re
from typing import Any

from grading.env_context import synthetic_eval_time_unix
from grading.models import Transcript

_TRACE_ID_HEX_32 = re.compile(r"(?<![0-9a-fA-F])([0-9a-fA-F]{32})(?![0-9a-fA-F])")
# _TRACE_ID_HEX_32 catches most trace IDs, but only matches exactly 32 hex chars.
# Tempo search JSON sometimes omits a leading zero (31 chars), so this pattern
# looks for known key names to pick those up too, but only in the JSON format we
# know that trace commands use.
_TRACE_ID_JSON = re.compile(
    r'"trace[_]?id"\s*:\s*"([0-9a-fA-F]{31,32})"',
    re.IGNORECASE,
)


def prometheus_eval_time_unix(params: dict[str, Any], _transcript: Transcript) -> float | None:
    raw = params.get("time_unix")
    if raw is not None:
        return float(raw)
    return synthetic_eval_time_unix()


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
            # if the shortest prefix isn't in the text, no longer prefix can be
            # either. A bit of a fast-fail.
            if trace_id[:prefix_min_chars] not in haystack:
                continue
            for prefix_len in range(len(trace_id), prefix_min_chars - 1, -1):
                prefix = trace_id[:prefix_len]
                if prefix in haystack:
                    return True, prefix
    return False, None
