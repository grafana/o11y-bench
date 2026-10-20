import re
from typing import Any

from grading.env_context import synthetic_eval_time_unix
from grading.models import Transcript

# Tempo prints trace IDs with leading zeros dropped, so a shorter hex string (25+) is still a full ID.
_TRACE_ID_HEX = re.compile(r"(?<![0-9a-fA-F])([0-9a-fA-F]{25,32})(?![0-9a-fA-F])")
# The check and the facts look at the same leading IDs, so they agree on the example.
MAX_CITED_TRACES = 5


def prometheus_eval_time_unix(params: dict[str, Any], _transcript: Transcript) -> float | None:
    raw = params.get("time_unix")
    if raw is not None:
        return float(raw)
    return synthetic_eval_time_unix()


def require_stack_url(url: str, env_name: str) -> str | None:
    if url:
        return None
    return f"{env_name} is not set."


def cited_trace_ids(answer: str) -> list[str]:
    """The first full trace IDs the answer cites, in order, as 32 hex. A short prefix can't be read back."""
    ids = dict.fromkeys(m.lower().zfill(32) for m in _TRACE_ID_HEX.findall(answer))
    return list(ids)[:MAX_CITED_TRACES]
