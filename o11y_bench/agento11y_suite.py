"""Build and synchronize an Agent Observability suite from o11y-bench task specs."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

import yaml

from .config import ROOT

DEFAULT_SUITE_ID = "o11y-bench"
DEFAULT_SPECS_DIR = ROOT / "tasks-spec"


def build_test_suite(specs_dir: Path = DEFAULT_SPECS_DIR) -> Any:
    """Convert source-controlled task YAML files to the Agent Observability suite shape."""

    agento11y = _import_agento11y()
    cases: list[Any] = []
    seen: set[str] = set()
    for path in sorted(specs_dir.rglob("*.yaml")):
        payload = yaml.safe_load(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError(f"task spec must be a mapping: {path}")
        case_id = str(payload.get("id") or "").strip()
        if not case_id:
            raise ValueError(f"task spec is missing id: {path}")
        if case_id in seen:
            raise ValueError(f"duplicate task id {case_id!r}: {path}")
        seen.add(case_id)
        statement = str(payload.get("statement") or "").strip()
        if not statement:
            raise ValueError(f"task spec {case_id!r} is missing statement: {path}")

        expected = {
            key: payload[key]
            for key in ("checks", "rubric")
            if isinstance(payload.get(key), list) and payload[key]
        }
        try:
            relative_path = path.relative_to(ROOT).as_posix()
        except ValueError:
            relative_path = path.relative_to(specs_dir).as_posix()
        cases.append(
            agento11y.TestCase(
                test_case_id=case_id,
                name=case_id.replace("-", " ").title(),
                description=statement,
                tags=[str(tag) for tag in payload.get("tags", [])],
                category=str(payload.get("category") or ""),
                input=statement,
                expected=expected,
                metadata={
                    "source": "o11y-bench",
                    "source_path": relative_path,
                    "spec_format": "o11y-bench/task-spec-v1",
                },
            )
        )

    if not cases:
        raise ValueError(f"no task specs found under {specs_dir}")
    return agento11y.TestSuite(
        suite_id=os.getenv("AGENTO11Y_SUITE_ID", DEFAULT_SUITE_ID).strip() or DEFAULT_SUITE_ID,
        name="o11y-bench",
        description="Observability and SRE agent benchmark maintained in o11y-bench task YAML.",
        tags=["o11y-bench", "observability", "sre"],
        test_cases=cases,
    )


def sync_test_suite(
    *,
    specs_dir: Path = DEFAULT_SPECS_DIR,
    publish: bool = True,
    progress: Callable[[str], None] | None = None,
) -> Any:
    """Push the source suite to Agent Observability and optionally publish its draft."""

    agento11y = _import_agento11y()
    if progress is not None:
        progress(f"[1/2] Loading and validating test cases from {specs_dir}")
    suite = build_test_suite(specs_dir)
    action = "uploading and publishing" if publish else "uploading as a draft"
    if progress is not None:
        progress(f"[2/2] {action.capitalize()} {len(suite.cases)} test cases")
    return agento11y.TestSuitesClient().push_suite(
        suite,
        publish=publish,
        changelog="Synchronized from o11y-bench task specs",
        prune=True,
    )


def test_suite_url(suite_id: str) -> str:
    """Build the Grafana test-suite deep link from the configured control URL."""

    control_url = os.getenv("AGENTO11Y_CONTROL_ENDPOINT", "").strip()
    grafana_url = os.getenv("AGENTO11Y_GRAFANA_URL", "").strip()
    source = control_url or grafana_url
    if not source:
        return ""
    parsed = urlsplit(source)
    if not parsed.scheme or not parsed.netloc:
        return ""
    app_path = "/a/grafana-agento11y-app"
    if app_path in parsed.path:
        base_path = parsed.path.split(app_path, 1)[0] + app_path
    else:
        grafana_parsed = urlsplit(grafana_url) if grafana_url else parsed
        parsed = grafana_parsed
        base_path = app_path
    base = urlunsplit((parsed.scheme, parsed.netloc, base_path.rstrip("/"), "", ""))
    return f"{base}/experiments/test-suites/{quote(suite_id, safe='')}"


def _import_agento11y() -> Any:
    try:
        from agento11y import experiments
    except ImportError as exc:
        raise RuntimeError(
            "Agent Observability publishing requires agento11y>=0.12.0. "
            "Run `mise run agento11y:setup` to install the published SDK."
        ) from exc
    return experiments


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synchronize o11y-bench task specs to Agent Observability"
    )
    parser.add_argument("--path", type=Path, default=DEFAULT_SPECS_DIR)
    parser.add_argument("--no-publish", action="store_true")
    args = parser.parse_args()
    pushed = sync_test_suite(
        specs_dir=args.path,
        publish=not args.no_publish,
        progress=lambda message: print(message, flush=True),
    )
    state = "published" if pushed.published else "draft"
    print(
        f"Synchronized {len(pushed.suite.cases)} cases to "
        f"{pushed.suite_id}@{pushed.suite_version} ({state})"
    )
    url = test_suite_url(pushed.suite_id)
    if url:
        print(f"View the test suite here: {url}")


if __name__ == "__main__":
    main()
