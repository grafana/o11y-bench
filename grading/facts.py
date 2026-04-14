import json
import re
from dataclasses import dataclass
from typing import Any

from grading.dashboard_snapshot import (
    collect_annotation_query_texts,
    collect_panel_query_texts,
    collect_variable_query_texts,
    load_dashboard_snapshot,
    normalize_query_text,
)
from grading.env_context import (
    VerifierContext,
    default_tempo_search_window_sec,
    fetch_grafana_datasources_checked,
    fetch_loki_query_result,
    fetch_prometheus_query_result,
    fetch_tempo_search_traces,
    resolve_grafana_datasource,
)
from grading.models import (
    DashboardFact,
    DatasourceDetailFact,
    DatasourceInventoryFact,
    FactSpec,
    QueryFact,
)


@dataclass(frozen=True)
class FactResult:
    summary: str
    debug: dict[str, Any]


APPROXIMATE_CRITERION_RE = re.compile(
    r"\b(roughly|approx(?:imate|imately)?|about)\b", re.IGNORECASE
)


def resolve_fact(
    fact: FactSpec,
    ctx: VerifierContext,
    cache: dict[str, FactResult] | None = None,
) -> FactResult:
    cache_key = json.dumps(fact.model_dump(mode="json", exclude_none=True), sort_keys=True)
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    match fact:
        case QueryFact():
            result = resolve_query_fact(fact, ctx)
        case DashboardFact():
            result = resolve_dashboard_fact(fact, ctx)
        case DatasourceInventoryFact():
            result = resolve_datasource_inventory_fact(ctx)
        case DatasourceDetailFact():
            result = resolve_datasource_detail_fact(fact, ctx)

    if cache is not None:
        cache[cache_key] = result
    return result


def render_fact_summary_for_criterion(
    fact_result: FactResult,
    criterion_text: str,
) -> str:
    if not APPROXIMATE_CRITERION_RE.search(criterion_text):
        return fact_result.summary

    query_result = fact_result.debug.get("result")
    if not isinstance(query_result, dict):
        return fact_result.summary

    match query_result.get("shape"):
        case "scalar":
            return render_approximate_scalar_summary(
                fact_result.debug.get("query"),
                query_result.get("value"),
            )
        case _:
            return fact_result.summary


def resolve_query_fact(fact: QueryFact, ctx: VerifierContext) -> FactResult:
    match fact.backend:
        case "prometheus":
            result, err = fetch_prometheus_query_result(
                ctx.prometheus_url,
                fact.query,
                ctx.timeout_sec,
                time_unix=fact.time_unix,
            )
            if err or result is None:
                return FactResult(
                    summary=f"Could not load canonical Prometheus query result: {err or 'unknown error'}.",
                    debug={
                        "backend": fact.backend,
                        "query": fact.query,
                        "error": err or "unknown error",
                    },
                )
            return FactResult(
                summary=render_query_result_summary(fact.query, result),
                debug={"backend": fact.backend, "query": fact.query, "result": result},
            )
        case "loki":
            result, err = fetch_loki_query_result(
                ctx.loki_url,
                fact.query,
                ctx.timeout_sec,
                time_unix=fact.time_unix,
            )
            if err or result is None:
                return FactResult(
                    summary=f"Could not load canonical Loki query result: {err or 'unknown error'}.",
                    debug={
                        "backend": fact.backend,
                        "query": fact.query,
                        "error": err or "unknown error",
                    },
                )
            return FactResult(
                summary=render_query_result_summary(fact.query, result),
                debug={"backend": fact.backend, "query": fact.query, "result": result},
            )
        case "tempo":
            start_sec = fact.start_sec
            end_sec = fact.end_sec
            if start_sec is None or end_sec is None:
                start_sec, end_sec = default_tempo_search_window_sec(fact.lookback_hours or 24.0)
            traces, err = fetch_tempo_search_traces(
                ctx.tempo_url,
                fact.query,
                ctx.timeout_sec,
                limit=fact.limit or 20,
                start_sec=start_sec,
                end_sec=end_sec,
            )
            if err:
                return FactResult(
                    summary=f"Could not load canonical Tempo query result: {err}.",
                    debug={"backend": fact.backend, "query": fact.query, "error": err},
                )
            return FactResult(
                summary=render_tempo_result_summary(traces),
                debug={"backend": fact.backend, "query": fact.query, "result": traces},
            )


def resolve_dashboard_fact(fact: DashboardFact, ctx: VerifierContext) -> FactResult:
    snapshot, err = load_dashboard_snapshot(fact, ctx)
    if err or snapshot is None:
        return FactResult(
            summary=f"Could not load saved dashboard state: {err or 'unknown error'}.",
            debug={"resource": fact.resource, "error": err or "unknown error"},
        )

    debug = {
        "resource": fact.resource,
        "uid": snapshot.uid,
        "panels": [
            {
                "title": str(panel.get("title", "")),
                "type": str(panel.get("type", "")),
                "datasource_type": str(panel.get("datasource", {}).get("type", "")),
                "queries": [
                    normalize_query_text(query) for query in collect_panel_query_texts(panel)
                ],
            }
            for panel in snapshot.panels
        ],
        "variables": [
            {
                "name": str(variable.get("name", "")),
                "type": str(variable.get("type", "")),
                "include_all": bool(variable.get("includeAll", False)),
                "multi": bool(variable.get("multi", False)),
                "queries": [
                    normalize_query_text(query) for query in collect_variable_query_texts(variable)
                ],
            }
            for variable in snapshot.variables
        ],
        "annotations": [
            {
                "name": str(annotation.get("name", "")),
                "queries": [
                    normalize_query_text(query)
                    for query in collect_annotation_query_texts(annotation)
                ],
            }
            for annotation in snapshot.annotations
        ],
    }
    return FactResult(summary=render_dashboard_summary(debug), debug=debug)


def resolve_datasource_inventory_fact(ctx: VerifierContext) -> FactResult:
    datasources, err = fetch_grafana_datasources_checked(ctx)
    if err:
        return FactResult(
            summary=f"Could not load Grafana datasource inventory: {err}.",
            debug={"resource": "datasource_inventory", "error": err},
        )
    assert datasources is not None
    items = [
        {
            "name": str(item.get("name", "")),
            "type": str(item.get("type", "")),
            "url": str(item.get("url", "")),
            "access": str(item.get("access", "")),
        }
        for item in datasources
    ]
    debug = {
        "resource": "datasource_inventory",
        "items": items,
    }
    return FactResult(summary=render_datasource_inventory_summary(items), debug=debug)


def resolve_datasource_detail_fact(
    fact: DatasourceDetailFact,
    ctx: VerifierContext,
) -> FactResult:
    datasources, err = fetch_grafana_datasources_checked(ctx)
    if err:
        return FactResult(
            summary=f"Could not load Grafana datasource detail: {err}.",
            debug={"resource": fact.resource, "error": err},
        )
    assert datasources is not None

    datasource, err = resolve_grafana_datasource(
        datasources,
        name=fact.name,
        datasource_type=fact.type,
    )
    if err or datasource is None:
        ident = fact.name or fact.type or "unknown selector"
        return FactResult(
            summary=f"Grafana did not return a datasource matching {ident}.",
            debug={"resource": fact.resource, "selector": {"name": fact.name, "type": fact.type}},
        )
    detail = {
        "name": str(datasource.get("name", "")),
        "type": str(datasource.get("type", "")),
        "url": str(datasource.get("url", "")),
        "access": str(datasource.get("access", "")),
    }
    debug = {
        "resource": fact.resource,
        "item": detail,
    }
    return FactResult(summary=render_datasource_detail_summary(detail), debug=debug)


def render_query_result_summary(query: str, result: dict[str, Any]) -> str:
    match result.get("shape"):
        case "scalar":
            return f'The canonical query "{query}" returned {format_number(result.get("value"))}.'
        case "items":
            items = sorted(
                result.get("items") or [],
                key=lambda item: float(item.get("value", 0.0)),
                reverse=True,
            )
            if not items:
                return "The canonical query returned no results."
            rendered = [render_labeled_item(item) for item in items[:5]]
            if len(items) == 1:
                return f"The top canonical result was {rendered[0]}."
            remainder = ", ".join(rendered[1:])
            if remainder:
                return (
                    f"The top canonical result was {rendered[0]}. "
                    f"Other canonical results were {remainder}."
                )
            return f"The top canonical result was {rendered[0]}."
        case _:
            return "The canonical query returned an unsupported result shape."


def render_approximate_scalar_summary(query: Any, value: Any) -> str:
    query_text = str(query).strip() if isinstance(query, str) else ""
    prefix = (
        f'The canonical query "{query_text}" returned'
        if query_text
        else "The canonical query value was"
    )
    try:
        number = float(value)
    except TypeError, ValueError:
        return f"{prefix} {format_number(value)}."

    if number == 0:
        return f"{prefix} 0."

    tolerance = abs(number) * 0.05
    lower = number - tolerance
    upper = number + tolerance
    return (
        f"{prefix} {format_number(number)}; answers in the same rough range "
        f"({format_number(lower)} to {format_number(upper)}) are consistent with that fact."
    )


def render_tempo_result_summary(traces: list[dict[str, Any]]) -> str:
    if not traces:
        return "The canonical Tempo query returned no traces."

    rendered: list[str] = []
    for trace in traces[:3]:
        parts: list[str] = []
        trace_id = str(trace.get("traceID", "")).strip()
        root_service = str(trace.get("rootServiceName", "")).strip()
        root_name = str(trace.get("rootTraceName", "")).strip()
        duration_ms = trace.get("durationMs")
        if trace_id:
            parts.append(f"trace {trace_id}")
        if root_name:
            parts.append(root_name)
        if root_service:
            parts.append(f"rooted at {root_service}")
        if duration_ms is not None:
            parts.append(f"{format_number(duration_ms)}ms")
        service_stats = trace.get("serviceStats")
        if isinstance(service_stats, dict) and service_stats:
            services = ", ".join(sorted(str(name) for name in service_stats))
            parts.append(f"services {services}")
        rendered.append(" ".join(parts))

    if len(rendered) == 1:
        return f"The canonical Tempo query returned {rendered[0]}."
    return (
        f"The canonical Tempo query returned {len(traces)} traces, including {', '.join(rendered)}."
    )


def render_dashboard_summary(snapshot: dict[str, Any]) -> str:
    uid = snapshot.get("uid", "")
    panels = snapshot.get("panels") or []
    variables = snapshot.get("variables") or []
    annotations = snapshot.get("annotations") or []

    parts = [f"The saved dashboard {uid} has {len(panels)} panel{'s' if len(panels) != 1 else ''}."]
    if panels:
        panel_bits = []
        for panel in panels[:4]:
            title = str(panel.get("title", "")).strip() or "Untitled"
            panel_type = str(panel.get("type", "")).strip() or "unknown"
            datasource_type = str(panel.get("datasource_type", "")).strip()
            queries = [query for query in panel.get("queries") or [] if query]
            bit = f"{title} ({panel_type}"
            if datasource_type:
                bit += f", {datasource_type}"
            if queries:
                bit += f', query "{queries[0]}"'
            bit += ")"
            panel_bits.append(bit)
        parts.append("Panels: " + "; ".join(panel_bits) + ".")

    if variables:
        variable_bits = []
        for variable in variables[:3]:
            name = str(variable.get("name", "")).strip() or "unnamed"
            var_type = str(variable.get("type", "")).strip() or "unknown"
            query = next((query for query in variable.get("queries") or [] if query), "")
            flags = []
            if variable.get("include_all"):
                flags.append("include all")
            if variable.get("multi"):
                flags.append("multi")
            bit = f"{name} ({var_type}"
            if flags:
                bit += ", " + ", ".join(flags)
            if query:
                bit += f', query "{query}"'
            bit += ")"
            variable_bits.append(bit)
        parts.append("Variables: " + "; ".join(variable_bits) + ".")

    if annotations:
        annotation_bits = []
        for annotation in annotations[:2]:
            name = str(annotation.get("name", "")).strip() or "unnamed"
            query = next((query for query in annotation.get("queries") or [] if query), "")
            bit = name
            if query:
                bit += f' (query "{query}")'
            annotation_bits.append(bit)
        parts.append("Annotations: " + "; ".join(annotation_bits) + ".")

    return " ".join(parts)


def render_datasource_inventory_summary(items: list[dict[str, Any]]) -> str:
    if not items:
        return "Grafana has no configured datasources."
    rendered = [
        f"{str(item.get('name', '')).strip() or 'unnamed'} ({str(item.get('type', '')).strip() or 'unknown'})"
        for item in items[:5]
    ]
    return f"Grafana has datasources {', '.join(rendered)}."


def render_datasource_detail_summary(item: dict[str, Any]) -> str:
    name = str(item.get("name", "")).strip() or "unnamed"
    kind = str(item.get("type", "")).strip() or "unknown"
    url = str(item.get("url", "")).strip()
    access = str(item.get("access", "")).strip()
    parts = [f"Grafana datasource {name} is type {kind}."]
    if url:
        parts.append(f"URL: {url}.")
    if access:
        parts.append(f"Access mode: {access}.")
    return " ".join(parts)


def render_labeled_item(item: dict[str, Any]) -> str:
    labels = item.get("labels") or {}
    label_text = (
        ", ".join(f"{key}={value}" for key, value in sorted(labels.items()))
        if isinstance(labels, dict) and labels
        else "unlabeled result"
    )
    return f"{label_text} with value {format_number(item.get('value'))}"


def format_number(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return str(value)
    try:
        number = float(value)
    except TypeError, ValueError:
        return str(value)
    if number == 0:
        return "0"
    if abs(number) >= 1000:
        return f"{number:.0f}"
    if abs(number) >= 100:
        return f"{number:.1f}".rstrip("0").rstrip(".")
    if abs(number) >= 1:
        return f"{number:.4f}".rstrip("0").rstrip(".")
    return f"{number:.6f}".rstrip("0").rstrip(".")
