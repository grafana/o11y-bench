"""Builders shared by the grading tests."""

import re
from collections.abc import Callable
from typing import Any
from urllib.parse import parse_qs, urlsplit

from grading.models import Message, Transcript


def answer_transcript(text: str) -> Transcript:
    return Transcript(messages=[Message("user", "q"), Message("assistant", text)])


def fake_tempo(
    stored: set[str], matching: set[str], trace: dict[str, Any] | None = None
) -> Callable[[str, float], dict[str, Any]]:
    """A Tempo `http_get_json`: a bare `trace:id` search finds a stored trace, one with `&&` finds
    a matching one, and `/api/v2/traces/<id>` returns `trace`."""

    def get(url: str, timeout_sec: float) -> dict[str, Any]:
        if "/api/v2/traces/" in url:
            return trace or {}
        query = parse_qs(urlsplit(url).query)["q"][0]
        found = re.search(r'trace:id = "(\w+)"', query)
        assert found and "/api/search?" in url
        hit = found.group(1) in (matching if "&&" in query else stored)
        # Tempo drops the leading zero of a trace ID in search results.
        return {"traces": [{"traceID": found.group(1).lstrip("0")}] if hit else []}

    return get
