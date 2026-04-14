import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from grading.env_context import (
    VerifierContext,
    fetch_grafana_dashboard_model,
    search_grafana_dashboard_uid,
)
from grading.helpers import require_stack_url
from grading.models import DashboardFact, DashboardSelector


@dataclass(frozen=True)
class DashboardSnapshot:
    model: dict[str, Any]
    uid: str
    panels: list[dict[str, Any]]
    variables: list[dict[str, Any]]
    annotations: list[dict[str, Any]]


def load_dashboard_snapshot(
    selector: DashboardSelector | DashboardFact,
    ctx: VerifierContext | None,
) -> tuple[DashboardSnapshot | None, str | None]:
    if ctx is None:
        return None, "GRAFANA_URL is not set."
    missing = require_stack_url(ctx.grafana_url, "GRAFANA_URL")
    if missing:
        return None, missing

    resolved_uid, err = resolve_dashboard_uid(ctx, selector)
    if err:
        return None, err

    dashboard, err = fetch_grafana_dashboard_model(ctx.grafana_url, resolved_uid, ctx.timeout_sec)
    if err or dashboard is None:
        return None, err or "Missing dashboard model."

    return (
        DashboardSnapshot(
            model=dashboard,
            uid=resolved_uid,
            panels=collect_dashboard_panels(dashboard.get("panels")),
            variables=dashboard_named_items(dashboard.get("templating"), "list"),
            annotations=dashboard_named_items(dashboard.get("annotations"), "list"),
        ),
        None,
    )


def resolve_dashboard_uid(
    ctx: VerifierContext,
    selector: DashboardSelector | DashboardFact,
) -> tuple[str, str | None]:
    if selector.uid:
        return selector.uid, None
    if selector.title:
        resolved, err = search_grafana_dashboard_uid(
            ctx.grafana_url,
            selector.title,
            ctx.timeout_sec,
        )
        if err or not resolved:
            return "", err or "Could not resolve dashboard uid from title."
        return resolved, None
    return "", "Dashboard validation requires uid or title."


def dashboard_named_items(container: Any, key: str) -> list[dict[str, Any]]:
    if not isinstance(container, dict):
        return []
    items = container.get(key)
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def collect_dashboard_panels(panels: Any) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    if not isinstance(panels, list):
        return collected
    for panel in panels:
        if not isinstance(panel, dict):
            continue
        collected.append(panel)
        collected.extend(collect_dashboard_panels(panel.get("panels")))
    return collected


def collect_named_values(items: list[dict[str, Any]], field: str) -> list[str]:
    values: list[str] = []
    for item in items:
        value = item.get(field)
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
    return values


def find_named_dashboard_item(
    items: list[dict[str, Any]],
    *,
    item_name: str,
    field: str,
) -> dict[str, Any] | None:
    normalized_name = normalize_token(item_name)
    for item in items:
        value = item.get(field)
        if isinstance(value, str) and normalize_token(value) == normalized_name:
            return item
    return None


def dashboard_datasource_type(item: dict[str, Any]) -> str:
    datasource = item.get("datasource")
    if isinstance(datasource, dict):
        kind = datasource.get("type")
        if isinstance(kind, str):
            return kind
    if isinstance(datasource, str):
        return datasource
    return ""


def collect_panel_query_texts(panel: dict[str, Any]) -> list[str]:
    targets = panel.get("targets")
    if not isinstance(targets, list):
        return []
    return collect_query_strings(targets, keys=("expr", "query", "rawSql"))


def collect_annotation_query_texts(annotation: dict[str, Any]) -> list[str]:
    return collect_query_strings((annotation, annotation.get("target")), keys=("expr", "query"))


def collect_variable_query_texts(variable: dict[str, Any]) -> list[str]:
    queries = collect_query_strings(
        (variable.get("query"), variable.get("definition"), variable),
        keys=("query", "definition", "expr"),
    )
    return [query for query in queries if query.strip()]


def collect_query_strings(sources: Any, *, keys: tuple[str, ...]) -> list[str]:
    queries: list[str] = []

    def visit(source: Any) -> None:
        if isinstance(source, dict):
            for key in keys:
                value = source.get(key)
                if isinstance(value, str) and value.strip():
                    queries.append(value.strip())
            for value in source.values():
                visit(value)
            return
        if isinstance(source, list | tuple):
            for value in source:
                visit(value)

    visit(sources)
    return queries


def parse_tag_keys(raw: Any) -> list[str]:
    if isinstance(raw, str):
        return [normalize_token(part) for part in raw.split(",") if part.strip()]
    if isinstance(raw, list):
        return [normalize_token(str(part)) for part in raw if str(part).strip()]
    return []


def split_response_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def line_mentions_all(line: str, *fragments: str) -> bool:
    normalized_line = normalize_token(line)
    return all(normalize_token(fragment) in normalized_line for fragment in fragments)


def strip_markdown_code_fences(text: str) -> str:
    text = re.sub(r"```[a-zA-Z0-9_-]*", "", text)
    return text.replace("```", "")


def normalize_token(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def normalize_query_text(text: str) -> str:
    return " ".join(text.split()).strip()


def normalize_query_value(text: str) -> str:
    normalized = normalize_token(text)
    if match := re.fullmatch(r"\$([a-zA-Z_][a-zA-Z0-9_]*)", normalized):
        return f"${match.group(1)}"
    if match := re.fullmatch(r"\$\{([a-zA-Z_][a-zA-Z0-9_]*)(?::[^}]+)?\}", normalized):
        return f"${match.group(1)}"
    return normalized


def select_panels_for_titles(
    snapshot: DashboardSnapshot,
    panel_titles: Iterable[str],
) -> tuple[list[dict[str, Any]], str | None]:
    requested_titles = [str(title) for title in panel_titles if str(title).strip()]
    if not requested_titles:
        return snapshot.panels, None

    selected: list[dict[str, Any]] = []
    missing_titles: list[str] = []
    for title in requested_titles:
        panel = find_named_dashboard_item(snapshot.panels, item_name=title, field="title")
        if panel is None:
            missing_titles.append(title)
            continue
        selected.append(panel)

    if missing_titles:
        unique_titles = ", ".join(sorted(set(missing_titles), key=normalize_token))
        return [], f"Requested panel_titles not found in dashboard: {unique_titles}"
    return selected, None
