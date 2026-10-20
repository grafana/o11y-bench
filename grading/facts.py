import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from grading.dashboard_diff import diff_dashboards, render_compact_diff, seed_dashboard
from grading.dashboard_snapshot import (
    collect_annotation_query_texts,
    collect_panel_query_texts,
    collect_variable_query_texts,
    dashboard_datasource_type,
    load_dashboard_snapshot,
    normalize_query_text,
)
from grading.env_context import (
    VerifierContext,
    default_tempo_search_window_sec,
    fetch_grafana_api,
    fetch_grafana_datasources_checked,
    fetch_loki_query_result,
    fetch_prometheus_query_result,
    fetch_tempo_metrics_instant,
    fetch_tempo_search_traces,
    fetch_tempo_trace,
    find_trace_example,
    resolve_grafana_datasource,
)
from grading.helpers import cited_trace_ids
from grading.models import (
    CitedTraceFact,
    CitedTraceParams,
    DashboardDiffFact,
    DashboardFact,
    DatasourceDetailFact,
    DatasourceInventoryFact,
    FactSpec,
    GrafanaApiFact,
    QueryFact,
)


@dataclass(frozen=True)
class FactResult:
    summary: str
    debug: dict[str, Any]
    # Set when the truth could not be read. The criterion is then a grading error, not a fail.
    error: str | None = None
    # Set when the criterion fails without a judge call: the summary says why.
    fails: bool = False


# A fixed set of intro phrases, so the judge sees a consistent statement of what the data is.
PREFIX_RANKING = "Ranking from the snapshot window:"
PREFIX_MATCHING_METRICS = "Matching metrics exist in the snapshot window:"
PREFIX_MATCHING_TRACES = "Matching traces exist in the snapshot window:"
PREFIX_GROUPED_COUNTS = "Grouped counts from the snapshot window:"
PREFIX_MATCHING_LOG_TELEMETRY = "Matching log telemetry exists in the snapshot window:"
PREFIX_GRAFANA_API = "Grafana API response from {path}:"
PREFIX_EXAMPLE_TRACE = "The example trace the final response cites, read back from Tempo:"

MAX_RENDERED_RESULTS = 50
DEFAULT_TEMPO_LIMIT = 20
MAX_RENDERED_SPANS = 3
EMPTY_TRUTH = "The query returned no results: nothing matches."
# A Grafana API read longer than this is a truth error, not a cut sample: a wrong claim about
# an item past the cut could not be caught. Point the fact at a narrower path instead.
MAX_GRAFANA_API_CHARS = 8000
# A TraceQL metrics query returns a number per series, not a page of traces.
TEMPO_METRICS = re.compile(
    r"\|\s*(rate|count_over_time|min_over_time|max_over_time|avg_over_time|sum_over_time"
    r"|quantile_over_time|histogram_over_time)\s*\("
)
NOT_RECORDED = "not_recorded: this fact is missing from the saved truth.json"
MAX_TRACE_SPANS = 40


def fact_cache_key(
    fact: FactSpec, answer: str = "", example: CitedTraceParams | None = None
) -> str:
    key = fact.model_dump(mode="json", exclude_defaults=True)
    if isinstance(fact, CitedTraceFact):
        # The truth depends on which traces the answer cites and what the example must match.
        key["trace_ids"] = cited_trace_ids(answer)
        if example:
            key["example"] = example.model_dump(mode="json")
    return json.dumps(key, sort_keys=True)


def resolve_fact(
    fact: FactSpec,
    ctx: VerifierContext | None,
    cache: dict[str, FactResult] | None = None,
    *,
    offline: bool = False,
    seeds: Sequence[dict[str, Any]] = (),
    answer: str = "",
    example: CitedTraceParams | None = None,
) -> FactResult:
    """Resolve a fact once per grading pass. Offline, only facts already in the cache exist.

    `seeds` are the task's `setup_dashboards`: the before state of a dashboard diff.
    `answer` is the final response, whose cited trace IDs a cited-trace fact reads back, and
    `example` the task's cited_trace check, which says what that trace must match.
    """
    cache_key = fact_cache_key(fact, answer, example)
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    if offline or ctx is None:
        return FactResult(summary="", debug={}, error=NOT_RECORDED if offline else "no stack")

    match fact:
        case QueryFact():
            result = resolve_query_fact(fact, ctx)
        case DashboardFact():
            result = resolve_dashboard_fact(fact, ctx)
        case DashboardDiffFact():
            result = resolve_dashboard_diff_fact(fact, ctx, seeds)
        case DatasourceInventoryFact():
            result = resolve_datasource_inventory_fact(ctx)
        case DatasourceDetailFact():
            result = resolve_datasource_detail_fact(fact, ctx)
        case GrafanaApiFact():
            result = resolve_grafana_api_fact(fact, ctx)
        case CitedTraceFact():
            result = resolve_cited_trace_fact(cited_trace_ids(answer), example, ctx)

    if cache is not None:
        cache[cache_key] = result
    return result


def provenance(fact: FactSpec) -> str:
    match fact:
        case QueryFact():
            return f"source={fact.backend}; query={fact.query}"
        case DashboardFact():
            return f"source=grafana; path=/api/dashboards/uid/{fact.uid or fact.title}"
        case GrafanaApiFact():
            return f"source=grafana; path={fact.path}"
        case DashboardDiffFact():
            return (
                f"source=grafana; path=/api/dashboards/uid/{fact.uid or fact.title}, "
                "compared with the seeded dashboard"
            )
        case CitedTraceFact():
            return "source=tempo; path=/api/v2/traces/<id> of the example trace, derived facts computed"
        case _:
            return "source=grafana; path=/api/datasources"


def resolve_cited_trace_fact(
    trace_ids: list[str], example: CitedTraceParams | None, ctx: VerifierContext
) -> FactResult:
    """The task's example trace: what it must be, what it shows, and its span tree."""
    debug: dict[str, Any] = {"trace_ids": trace_ids}
    if example is None:
        return FactResult(summary="", debug=debug, error="no cited_trace check defines the example")
    found, misses, err = find_trace_example(
        ctx.tempo_url, trace_ids, example.match, example.lookback_hours, ctx.timeout_sec
    )
    if err:
        return FactResult(summary="", debug=debug, error=f"transport: {err}")
    if found is None:
        why = (
            f"cites no trace that is in Tempo and matches {example.match}: " + "; ".join(misses)
            if trace_ids
            else "cites no full trace ID"
        )
        return FactResult(
            summary=f"The final response {why}, so a claim about its example trace earns nothing.",
            debug=debug,
            fails=True,
        )
    spans, err = fetch_tempo_trace(ctx.tempo_url, found, ctx.timeout_sec)
    if err:
        return FactResult(summary="", debug=debug, error=f"transport: {err}")
    if not spans:
        return FactResult(
            summary="",
            debug=debug,
            error=f"transport: {found} matched in search but Tempo returned no spans for it",
        )
    debug["example"] = found
    lines = [f"{PREFIX_EXAMPLE_TRACE} {found}, the first cited ID that matches {example.match}."]
    lines += [f"- {line}" for line in derive_trace_facts(spans)]
    if misses:
        lines.append("- cited before it: " + "; ".join(misses))
    lines.append(render_trace_tree(found, spans))
    return FactResult(summary="\n".join(lines), debug=debug)


def derive_trace_facts(spans: list[dict[str, Any]]) -> list[str]:
    """What a trace shows, computed so the judge compares names instead of doing arithmetic."""
    by_id = {span["span_id"]: span for span in spans}

    def depth(span: dict[str, Any]) -> int:
        level, parent = 0, span["parent_id"]
        while parent in by_id and level < len(spans):
            level, parent = level + 1, by_id[parent]["parent_id"]
        return level

    def own_ns(span: dict[str, Any]) -> int:
        """Duration minus the time its child spans cover."""
        covered, cursor = 0, span["start_ns"]
        kids = sorted(
            (max(c["start_ns"], span["start_ns"]), min(c["end_ns"], span["end_ns"]))
            for c in spans
            if c["parent_id"] == span["span_id"]
        )
        for start, end in kids:
            start = max(start, cursor)
            if end > start:
                covered, cursor = covered + end - start, end
        return int(span["end_ns"] - span["start_ns"] - covered)

    def label(span: dict[str, Any]) -> str:
        return f"{span['service']} `{span['name']}`"

    def ms(ns: int) -> str:
        return format_number(ns / 1e6)

    root = min((s for s in spans if s["parent_id"] not in by_id), key=lambda s: s["start_ns"])
    total = max(s["end_ns"] for s in spans) - min(s["start_ns"] for s in spans)
    ordered = sorted(spans, key=lambda s: (depth(s), s["start_ns"]))
    errors = [s for s in spans if s["error"]]
    origin = max(errors, key=lambda s: (depth(s), s["start_ns"])) if errors else None
    busiest = max(spans, key=own_ns)
    return [
        f"root span: {label(root)}, {ms(total)}ms",
        "services, outermost first: " + ", ".join(dict.fromkeys(s["service"] for s in ordered)),
        "spans with error status: " + (", ".join(label(s) for s in errors) or "none"),
        "error origin (the deepest span with error status): "
        + (label(origin) if origin else "none, no span has error status"),
        f"most own time (duration minus its child spans): {label(busiest)}, "
        f"{ms(own_ns(busiest))} of {ms(total)}ms",
    ]


def render_trace_tree(trace_id: str, spans: list[dict[str, Any]]) -> str:
    start = min(span["start_ns"] for span in spans)
    end = max(span["end_ns"] for span in spans)
    children: dict[str, list[dict[str, Any]]] = {}
    ids = {span["span_id"] for span in spans}
    for span in sorted(spans, key=lambda item: item["start_ns"]):
        parent = span["parent_id"] if span["parent_id"] in ids else ""
        children.setdefault(parent, []).append(span)
    lines = [f"trace {trace_id}: {len(spans)} spans, {format_number((end - start) / 1e6)}ms"]

    def walk(parent: str, depth: int) -> None:
        for span in children.get(parent, []):
            if len(lines) > MAX_TRACE_SPANS:
                return
            parts = [
                span["service"],
                span["name"],
                span["kind"],
                f"{format_number((span['end_ns'] - span['start_ns']) / 1e6)}ms",
                f"starts +{format_number((span['start_ns'] - start) / 1e6)}ms",
            ]
            if span["http_status"] is not None:
                parts.append(f"http {span['http_status']}")
            if span["error"]:
                parts.append("status error")
            lines.append("  " * (depth + 1) + " | ".join(str(part) for part in parts if part))
            walk(span["span_id"], depth + 1)

    walk("", 0)
    if len(spans) > MAX_TRACE_SPANS:
        lines.append(f"  ({MAX_TRACE_SPANS} of {len(spans)} spans shown)")
    return "\n".join(lines)


def resolve_query_fact(fact: QueryFact, ctx: VerifierContext) -> FactResult:
    debug: dict[str, Any] = {"backend": fact.backend, "query": fact.query}
    start_sec, end_sec = fact.start_sec, fact.end_sec
    if fact.backend == "tempo" and (start_sec is None or end_sec is None):
        start_sec, end_sec = default_tempo_search_window_sec(fact.lookback_hours or 24.0)
    match fact.backend:
        case "prometheus":
            result, err = fetch_prometheus_query_result(
                ctx.prometheus_url, fact.query, ctx.timeout_sec, time_unix=fact.time_unix
            )
        case "loki":
            result, err = fetch_loki_query_result(
                ctx.loki_url, fact.query, ctx.timeout_sec, time_unix=fact.time_unix
            )
        case "tempo" if TEMPO_METRICS.search(fact.query):
            assert start_sec is not None and end_sec is not None
            items, err = fetch_tempo_metrics_instant(
                ctx.tempo_url, fact.query, ctx.timeout_sec, start_sec, end_sec
            )
            result = None if err else {"shape": "items", "items": items}
        case "tempo":
            traces, err = fetch_tempo_search_traces(
                ctx.tempo_url,
                fact.query,
                ctx.timeout_sec,
                limit=fact.limit or DEFAULT_TEMPO_LIMIT,
                start_sec=start_sec,
                end_sec=end_sec,
            )
            result = None if err else {"shape": "traces", "traces": traces}

    if err or result is None:
        return FactResult(summary="", debug=debug, error=f"transport: {err or 'unknown error'}")
    debug["result"] = result
    if query_result_is_empty(result):
        if not fact.allow_empty:
            return FactResult(
                summary="",
                debug=debug,
                error="empty_evidence: the query returned no results; "
                "set allow_empty if nothing matching is the truth",
            )
        return FactResult(summary=EMPTY_TRUTH, debug=debug)
    return FactResult(summary=render_query_result(fact, result), debug=debug)


def query_result_is_empty(result: dict[str, Any]) -> bool:
    match result.get("shape"):
        case "items":
            return not result.get("items")
        case "traces":
            return not result.get("traces")
        case _:
            return bool(result.get("empty"))


def intent_prefix(backend: str, query: str) -> str:
    q = query.lower()
    ranking = "topk(" in q or "bottomk(" in q
    if backend == "tempo":
        if TEMPO_METRICS.search(query):
            return PREFIX_RANKING if ranking else PREFIX_GROUPED_COUNTS
        return PREFIX_MATCHING_TRACES
    if backend == "loki":
        if ranking:
            return PREFIX_RANKING
        if any(m in q for m in ("sum(", "count(", "avg(", "rate(", "_over_time(")):
            return PREFIX_GROUPED_COUNTS
        return PREFIX_MATCHING_LOG_TELEMETRY
    return PREFIX_RANKING if ranking else PREFIX_MATCHING_METRICS


def render_query_result(fact: QueryFact, result: dict[str, Any]) -> str:
    prefix = intent_prefix(fact.backend, fact.query)
    match result.get("shape"):
        case "scalar":
            rows = [format_number(result.get("value"))]
        case "items":
            items = sorted(
                result.get("items") or [],
                key=lambda item: float(item.get("value", 0.0)),
                reverse=True,
            )
            rows = [render_labeled_item(item) for item in items]
        case "traces":
            rows = [render_trace(trace) for trace in result.get("traces") or []]
        case shape:
            raise ValueError(f"Unsupported fact result shape {shape!r}.")
    body = render_rows(rows)
    limit = fact.limit or DEFAULT_TEMPO_LIMIT
    if result.get("shape") == "traces" and len(rows) >= limit:
        body += f" (the query stops at {limit} results, so more may match)"
    return f"{prefix} {body}"


def render_rows(rows: list[str]) -> str:
    noun = "result" if len(rows) == 1 else "results"
    body = f"Query returned {len(rows)} {noun}: " + "; ".join(rows[:MAX_RENDERED_RESULTS])
    if len(rows) > MAX_RENDERED_RESULTS:
        body += f" ... ({MAX_RENDERED_RESULTS} of {len(rows)} results shown)"
    return body


def resolve_grafana_api_fact(fact: GrafanaApiFact, ctx: VerifierContext) -> FactResult:
    debug: dict[str, Any] = {"path": fact.path}
    payload, err, status = fetch_grafana_api(ctx.grafana_url, fact.path, ctx.timeout_sec)
    if err:
        if status == 404 and fact.allow_empty:
            return FactResult(summary=EMPTY_TRUTH, debug=debug)
        return FactResult(summary="", debug=debug, error=f"transport: {err}")
    debug["result"] = payload
    if not payload:
        if fact.allow_empty:
            return FactResult(summary=EMPTY_TRUTH, debug=debug)
        return FactResult(
            summary="",
            debug=debug,
            error=f"empty_evidence: Grafana API GET {fact.path} returned nothing; "
            "set allow_empty if nothing there is the truth",
        )
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    if len(body) > MAX_GRAFANA_API_CHARS:
        return FactResult(
            summary="",
            debug=debug,
            error=f"too_large: Grafana API GET {fact.path} rendered {len(body)} chars, over "
            f"{MAX_GRAFANA_API_CHARS}; the judge would see only part of it",
        )
    return FactResult(summary=f"{PREFIX_GRAFANA_API.format(path=fact.path)} {body}", debug=debug)


def resolve_dashboard_fact(fact: DashboardFact, ctx: VerifierContext) -> FactResult:
    snapshot, err = load_dashboard_snapshot(fact, ctx)
    if err or snapshot is None:
        return FactResult(
            summary="",
            debug={"resource": fact.resource},
            error=f"transport: could not load saved dashboard state: {err or 'unknown error'}",
        )

    debug = {
        "resource": fact.resource,
        "uid": snapshot.uid,
        "panels": [
            {
                "title": str(panel.get("title", "")),
                "type": str(panel.get("type", "")),
                "datasource_type": dashboard_datasource_type(panel),
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
    summary = PREFIX_GRAFANA_API.format(path=f"/api/dashboards/uid/{snapshot.uid}")
    return FactResult(summary=f"{summary} {render_dashboard_summary(debug)}", debug=debug)


def resolve_dashboard_diff_fact(
    fact: DashboardDiffFact,
    ctx: VerifierContext,
    seeds: Sequence[dict[str, Any]],
) -> FactResult:
    debug: dict[str, Any] = {"resource": fact.resource}
    seed = next(
        (
            s
            for s in seeds
            if (fact.uid and s.get("uid") == fact.uid)
            or (not fact.uid and fact.title and s.get("title") == fact.title)
        ),
        None,
    )
    if seed is None:
        # An authoring mistake: nothing was seeded to compare against.
        return FactResult(
            summary="",
            debug=debug,
            error="empty_evidence: the task seeds no dashboard "
            f"{fact.uid or fact.title!r} (setup_dashboards)",
        )
    snapshot, err = load_dashboard_snapshot(fact, ctx)
    if err or snapshot is None:
        return FactResult(
            summary="",
            debug=debug,
            error=f"transport: could not load saved dashboard state: {err or 'unknown error'}",
        )
    diff = diff_dashboards(seed_dashboard(seed), snapshot.model)
    debug.update({"uid": snapshot.uid, "diff": asdict(diff)})
    return FactResult(summary=render_compact_diff(snapshot.uid, diff), debug=debug)


def resolve_datasource_inventory_fact(ctx: VerifierContext) -> FactResult:
    datasources, err = fetch_grafana_datasources_checked(ctx)
    if err:
        return FactResult(
            summary="",
            debug={"resource": "datasource_inventory"},
            error=f"transport: could not load Grafana datasource inventory: {err}",
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
    summary = PREFIX_GRAFANA_API.format(path="/api/datasources")
    return FactResult(
        summary=f"{summary} {render_datasource_inventory_summary(items)}", debug=debug
    )


def resolve_datasource_detail_fact(
    fact: DatasourceDetailFact,
    ctx: VerifierContext,
) -> FactResult:
    datasources, err = fetch_grafana_datasources_checked(ctx)
    if err:
        return FactResult(
            summary="",
            debug={"resource": fact.resource},
            error=f"transport: could not load Grafana datasource detail: {err}",
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
            summary="",
            debug={"resource": fact.resource, "selector": {"name": fact.name, "type": fact.type}},
            error=f"empty_evidence: Grafana did not return a datasource matching {ident}",
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
    summary = PREFIX_GRAFANA_API.format(path="/api/datasources")
    return FactResult(summary=f"{summary} {render_datasource_detail_summary(detail)}", debug=debug)


def render_dashboard_summary(snapshot: dict[str, Any]) -> str:
    uid = snapshot.get("uid", "")
    panels = snapshot.get("panels") or []
    variables = snapshot.get("variables") or []
    annotations = snapshot.get("annotations") or []

    parts = [f"The saved dashboard {uid} has {len(panels)} panel{'s' if len(panels) != 1 else ''}."]
    if panels:
        panel_bits = []
        for panel in panels:
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
        for variable in variables:
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
        for annotation in annotations:
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
        for item in items
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
    value = format_number(item.get("value"))
    if not isinstance(labels, dict) or not labels:
        return value
    label_text = ", ".join(f'{key}="{val}"' for key, val in sorted(labels.items()))
    return f"{{{label_text}}} {value}"


def render_trace(trace: dict[str, Any]) -> str:
    parts = [f"trace {str(trace.get('traceID', '')).strip()}"]
    if root_name := str(trace.get("rootTraceName", "")).strip():
        parts.append(root_name)
    if root_service := str(trace.get("rootServiceName", "")).strip():
        parts.append(f"rooted at {root_service}")
    if (duration_ms := trace.get("durationMs")) is not None:
        parts.append(f"{format_number(duration_ms)}ms")
    service_stats = trace.get("serviceStats")
    if isinstance(service_stats, dict) and service_stats:
        parts.append("services " + ", ".join(sorted(str(name) for name in service_stats)))
    spans = [
        span
        for span_set in trace.get("spanSets") or [trace.get("spanSet") or {}]
        if isinstance(span_set, dict)
        for span in span_set.get("spans") or []
        if isinstance(span, dict)
    ]
    if spans:
        shown = [render_span(span) for span in spans[:MAX_RENDERED_SPANS]]
        more = (
            f" (+{len(spans) - MAX_RENDERED_SPANS} more)" if len(spans) > MAX_RENDERED_SPANS else ""
        )
        parts.append("matched spans [" + "; ".join(shown) + "]" + more)
    return " ".join(parts)


def render_span(span: dict[str, Any]) -> str:
    """A matched span: name, duration and the attributes the query selected."""
    parts = [str(span.get("name") or span.get("spanID") or "span").strip()]
    try:
        parts.append(f"{format_number(int(span['durationNanos']) / 1e6)}ms")
    except KeyError, TypeError, ValueError:
        pass
    for attribute in span.get("attributes") or []:
        if not isinstance(attribute, dict):
            continue
        value = attribute.get("value")
        if isinstance(value, dict) and value:
            value = next(iter(value.values()))
        parts.append(f"{attribute.get('key')}={value}")
    return " ".join(parts)


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


def dump_truth(cache: dict[str, FactResult]) -> list[dict[str, Any]]:
    """Every fact resolved in a grading pass, saved as verifier/truth.json for regrades."""
    return [
        {"key": key, "summary": r.summary, "error": r.error, "fails": r.fails, "debug": r.debug}
        for key, r in sorted(cache.items())
    ]


def load_truth(entries: list[dict[str, Any]]) -> dict[str, FactResult]:
    return {
        str(entry["key"]): FactResult(
            summary=str(entry.get("summary") or ""),
            debug=dict(entry.get("debug") or {}),
            error=entry.get("error"),
            fails=bool(entry.get("fails")),
        )
        for entry in entries
    }
