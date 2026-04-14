"""Verifier HTTP context: Grafana, Prometheus, Loki, Tempo (same stack as the agent)."""

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast
from urllib.parse import urlencode


@dataclass(frozen=True)
class VerifierContext:
    grafana_url: str = "http://127.0.0.1:3000"
    prometheus_url: str = "http://127.0.0.1:9090"
    loki_url: str = "http://127.0.0.1:3100"
    tempo_url: str = "http://127.0.0.1:3200"
    timeout_sec: float = 15.0


def load_verifier_context_from_env() -> VerifierContext:
    return VerifierContext(
        grafana_url=os.environ.get("GRAFANA_URL", VerifierContext.grafana_url).strip().rstrip("/"),
        prometheus_url=os.environ.get("PROMETHEUS_URL", VerifierContext.prometheus_url)
        .strip()
        .rstrip("/"),
        loki_url=os.environ.get("LOKI_URL", VerifierContext.loki_url).strip().rstrip("/"),
        tempo_url=os.environ.get("TEMPO_URL", VerifierContext.tempo_url).strip().rstrip("/"),
        timeout_sec=float(os.environ.get("VERIFIER_HTTP_TIMEOUT_SEC", "15")),
    )


def http_get_json(url: str, timeout_sec: float) -> dict[str, Any] | list[Any]:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        raw = resp.read().decode()
    payload = json.loads(raw)
    return cast(dict[str, Any] | list[Any], payload)


def synthetic_eval_time_unix() -> float:
    raw = os.environ.get("O11Y_SCENARIO_TIME_ISO", "").strip()
    if raw:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(UTC).timestamp()
    return datetime.now(UTC).timestamp()


def loki_eval_time_ns(time_unix: float | None = None) -> int:
    return int((synthetic_eval_time_unix() if time_unix is None else time_unix) * 1_000_000_000)


def instant_vector_first_scalar(payload: dict[str, Any]) -> float | None:
    """Parse first scalar from Prometheus- or Loki-style instant query JSON."""
    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    rtype = data.get("resultType")
    result = data.get("result")
    if rtype == "scalar" and isinstance(result, list) and len(result) >= 2:
        try:
            return float(str(result[1]))
        except TypeError, ValueError:
            return None
    if rtype == "vector" and isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict):
            val = first.get("value")
            if isinstance(val, list) and len(val) >= 2:
                try:
                    return float(str(val[1]))
                except TypeError, ValueError:
                    return None
    return None


def _instant_query_payload(
    base_url: str,
    path: str,
    query: str,
    timeout_sec: float,
    *,
    time_unix: float | None = None,
) -> tuple[dict[str, Any] | None, str]:
    if not base_url:
        return None, "Base URL is not set."
    effective_time_unix = synthetic_eval_time_unix() if time_unix is None else time_unix
    params: dict[str, str] = {"query": query}
    params["time"] = str(effective_time_unix)
    url = f"{base_url}{path}?{urlencode(params)}"
    try:
        payload = http_get_json(url, timeout_sec)
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return None, f"Request failed: {exc}"
    if not isinstance(payload, dict):
        return None, "Query returned non-object JSON."
    return payload, ""


def _normalize_instant_query_payload(
    payload: dict[str, Any],
    *,
    empty_vector_is_zero: bool = False,
) -> tuple[dict[str, Any] | None, str]:
    if payload.get("status") != "success":
        return None, f"Query not successful: {payload.get('error', payload)}"

    data = payload.get("data")
    if not isinstance(data, dict):
        return None, "Query response missing data object."

    result_type = data.get("resultType")
    result = data.get("result")
    if result_type == "scalar" and isinstance(result, list) and len(result) >= 2:
        try:
            return {"shape": "scalar", "value": float(str(result[1]))}, ""
        except TypeError, ValueError:
            return None, "Could not parse scalar result."

    if result_type == "vector" and isinstance(result, list):
        if empty_vector_is_zero and len(result) == 0:
            return {"shape": "scalar", "value": 0.0}, ""
        items: list[dict[str, Any]] = []
        for item in result:
            if not isinstance(item, dict):
                continue
            metric = item.get("metric")
            if not isinstance(metric, dict):
                metric = {}
            value = item.get("value")
            if not isinstance(value, list) or len(value) < 2:
                continue
            try:
                sample = float(str(value[1]))
            except TypeError, ValueError:
                continue
            items.append(
                {
                    "labels": {str(k): str(v) for k, v in metric.items()},
                    "value": sample,
                }
            )
        return {"shape": "items", "items": items}, ""

    return None, f"Unsupported instant query result type: {result_type!r}."


def _normalize_loki_range_payload(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    if payload.get("status") != "success":
        return None, f"Query not successful: {payload.get('error', payload)}"

    data = payload.get("data")
    if not isinstance(data, dict):
        return None, "Query response missing data object."

    if data.get("resultType") != "matrix":
        return None, f"Unsupported Loki query_range result type: {data.get('resultType')!r}."

    result = data.get("result")
    if not isinstance(result, list):
        return None, "Loki matrix result missing."
    if not result:
        return {"shape": "scalar", "value": 0.0}, ""

    items: list[dict[str, Any]] = []
    for series in result:
        if not isinstance(series, dict):
            continue
        metric = series.get("metric")
        if not isinstance(metric, dict):
            metric = {}
        values = series.get("values")
        if not isinstance(values, list) or not values:
            continue
        last = values[-1]
        if not isinstance(last, list) or len(last) < 2:
            continue
        try:
            sample = float(str(last[1]))
        except TypeError, ValueError:
            continue
        items.append(
            {
                "labels": {str(k): str(v) for k, v in metric.items()},
                "value": sample,
            }
        )

    if not items:
        return {"shape": "scalar", "value": 0.0}, ""
    if len(items) == 1 and not items[0]["labels"]:
        return {"shape": "scalar", "value": items[0]["value"]}, ""
    return {"shape": "items", "items": items}, ""


def fetch_prometheus_query_result(
    base_url: str,
    expr: str,
    timeout_sec: float,
    *,
    time_unix: float | None = None,
) -> tuple[dict[str, Any] | None, str]:
    payload, err = _instant_query_payload(
        base_url,
        "/api/v1/query",
        expr,
        timeout_sec,
        time_unix=time_unix,
    )
    if err or payload is None:
        return None, f"Prometheus {err or 'request failed.'}"
    return _normalize_instant_query_payload(payload)


def fetch_loki_query_result(
    base_url: str,
    logql: str,
    timeout_sec: float,
    *,
    time_unix: float | None = None,
) -> tuple[dict[str, Any] | None, str]:
    if not base_url:
        return None, "LOKI_URL is not set."
    eval_time_ns = loki_eval_time_ns(time_unix)
    # Loki query_range is more reliable than instant queries for metric-style LogQL,
    # but a zero-width [start=end] window can collapse to an empty matrix.
    params = {
        "query": logql,
        "start": str(eval_time_ns - 1_000_000_000),
        "end": str(eval_time_ns),
        "step": "1",
    }
    url = f"{base_url}/loki/api/v1/query_range?{urlencode(params)}"
    try:
        payload = http_get_json(url, timeout_sec)
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return None, f"Loki request failed: {exc}"
    if not isinstance(payload, dict):
        return None, "Loki returned non-object JSON."
    return _normalize_loki_range_payload(payload)


def fetch_prometheus_vector(
    base_url: str, expr: str, timeout_sec: float, time_unix: float | None = None
) -> tuple[list[tuple[dict[str, str], float]], str]:
    """Instant query; return (labelset, sample_value) for each vector element."""
    if not base_url:
        return [], "PROMETHEUS_URL is not set."
    effective_time_unix = synthetic_eval_time_unix() if time_unix is None else time_unix
    params: dict[str, str] = {"query": expr}
    params["time"] = str(effective_time_unix)
    url = f"{base_url}/api/v1/query?{urlencode(params)}"
    try:
        payload = http_get_json(url, timeout_sec)
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return [], f"Prometheus request failed: {exc}"
    if not isinstance(payload, dict):
        return [], "Prometheus returned non-object JSON."
    if payload.get("status") != "success":
        return [], f"Prometheus query not successful: {payload.get('error', payload)}"
    data = payload.get("data")
    if not isinstance(data, dict) or data.get("resultType") != "vector":
        return [], "Prometheus result is not an instant vector."
    result = data.get("result")
    if not isinstance(result, list):
        return [], "Prometheus vector result missing."
    out: list[tuple[dict[str, str], float]] = []
    for item in result:
        if not isinstance(item, dict):
            continue
        metric = item.get("metric")
        if not isinstance(metric, dict):
            metric = {}
        mlabels = {str(k): str(v) for k, v in metric.items()}
        val = item.get("value")
        if not isinstance(val, list) or len(val) < 2:
            continue
        try:
            sample = float(str(val[1]))
        except TypeError, ValueError:
            continue
        out.append((mlabels, sample))
    return out, ""


def fetch_prometheus_label_values(
    base_url: str,
    label_name: str,
    timeout_sec: float,
    *,
    match_expr: str | None = None,
    time_unix: float | None = None,
) -> tuple[list[str], str]:
    """Return distinct Prometheus label values, optionally scoped by match[] expression."""
    if not base_url:
        return [], "PROMETHEUS_URL is not set."
    params: list[tuple[str, str]] = []
    if match_expr:
        params.append(("match[]", match_expr))
    effective_time_unix = synthetic_eval_time_unix() if time_unix is None else time_unix
    ts = str(effective_time_unix)
    params.append(("start", ts))
    params.append(("end", ts))
    url = f"{base_url}/api/v1/label/{label_name}/values?{urlencode(params)}"
    try:
        payload = http_get_json(url, timeout_sec)
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return [], f"Prometheus label values request failed: {exc}"
    if not isinstance(payload, dict):
        return [], "Prometheus returned non-object JSON."
    if payload.get("status") != "success":
        return [], f"Prometheus label values not successful: {payload.get('error', payload)}"
    data = payload.get("data")
    if not isinstance(data, list):
        return [], "Prometheus label values response missing list data."
    values = [str(value) for value in data if isinstance(value, str | int | float)]
    return values, ""


def fetch_prometheus_instant(
    base_url: str, expr: str, timeout_sec: float, time_unix: float | None = None
) -> tuple[float | None, str]:
    """Return (scalar_value, error_message). error_message empty on success."""
    if not base_url:
        return None, "PROMETHEUS_URL is not set."
    effective_time_unix = synthetic_eval_time_unix() if time_unix is None else time_unix
    params: dict[str, str] = {"query": expr}
    params["time"] = str(effective_time_unix)
    url = f"{base_url}/api/v1/query?{urlencode(params)}"
    try:
        payload = http_get_json(url, timeout_sec)
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return None, f"Prometheus request failed: {exc}"
    if not isinstance(payload, dict):
        return None, "Prometheus returned non-object JSON."
    status = payload.get("status")
    if status != "success":
        return None, f"Prometheus query not successful: {payload.get('error', payload)}"
    value = instant_vector_first_scalar(payload)
    if value is None:
        return None, "Could not parse scalar from Prometheus response."
    return value, ""


def fetch_loki_instant(
    base_url: str, logql: str, timeout_sec: float, time_unix: float | None = None
) -> tuple[float | None, str]:
    """Run LogQL as an instant metric query; parse first series value."""
    if not base_url:
        return None, "LOKI_URL is not set."
    params = {"query": logql, "time": str(loki_eval_time_ns(time_unix))}
    url = f"{base_url}/loki/api/v1/query?{urlencode(params)}"
    try:
        payload = http_get_json(url, timeout_sec)
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return None, f"Loki request failed: {exc}"
    if not isinstance(payload, dict):
        return None, "Loki returned non-object JSON."
    status = payload.get("status")
    if status != "success":
        return None, f"Loki query not successful: {payload.get('error', payload)}"
    value = instant_vector_first_scalar(payload)
    if value is None:
        # Metric-style LogQL (e.g. sum(count_over_time(...))) often yields an empty instant
        # vector when the count is zero, rather than a one-element vector of 0.
        data = payload.get("data")
        if isinstance(data, dict) and data.get("resultType") == "vector":
            result = data.get("result")
            if isinstance(result, list) and len(result) == 0:
                return 0.0, ""
        return None, "Could not parse scalar from Loki response (use a metric-style LogQL)."
    return value, ""


def fetch_loki_streams(
    base_url: str,
    logql: str,
    timeout_sec: float,
    *,
    start_unix: float,
    end_unix: float,
    limit: int = 200,
) -> tuple[list[dict[str, Any]], str]:
    """Run LogQL as a log query and return stream records."""
    if not base_url:
        return [], "LOKI_URL is not set."

    start_ns = int(start_unix * 1_000_000_000)
    end_ns = int(end_unix * 1_000_000_000)
    params = {
        "query": logql,
        "start": str(start_ns),
        "end": str(end_ns),
        "limit": str(limit),
        "direction": "backward",
    }
    url = f"{base_url}/loki/api/v1/query_range?{urlencode(params)}"
    try:
        payload = http_get_json(url, timeout_sec)
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return [], f"Loki query_range failed: {exc}"
    if not isinstance(payload, dict):
        return [], "Loki returned non-object JSON."
    status = payload.get("status")
    if status != "success":
        return [], f"Loki query_range not successful: {payload.get('error', payload)}"
    data = payload.get("data")
    if not isinstance(data, dict) or data.get("resultType") != "streams":
        return [], "Loki result is not a streams response."
    result = data.get("result")
    if not isinstance(result, list):
        return [], "Loki streams result missing."

    records: list[dict[str, Any]] = []
    for stream_item in result:
        if not isinstance(stream_item, dict):
            continue
        stream = stream_item.get("stream")
        values = stream_item.get("values")
        if not isinstance(stream, dict) or not isinstance(values, list):
            continue
        labels = {str(key): str(value) for key, value in stream.items()}
        for value in values:
            if not isinstance(value, list) or len(value) < 2:
                continue
            ts_raw, line_raw = value[0], value[1]
            try:
                timestamp_ns = int(str(ts_raw))
            except TypeError, ValueError:
                continue
            records.append(
                {
                    "stream": labels,
                    "timestamp_ns": timestamp_ns,
                    "line": str(line_raw),
                }
            )
    return records, ""


def fetch_tempo_search_traces(
    base_url: str,
    search_q: str,
    timeout_sec: float,
    limit: int = 20,
    start_sec: int | None = None,
    end_sec: int | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Return Tempo trace search records from GET /api/search.

    Tempo 2.x expects ``start`` and ``end`` as **Unix epoch seconds** (not nanoseconds).
    """
    if not base_url:
        return [], "TEMPO_URL is not set."
    params: dict[str, str] = {"q": search_q, "limit": str(limit)}
    if start_sec is not None:
        params["start"] = str(int(start_sec))
    if end_sec is not None:
        params["end"] = str(int(end_sec))
    url = f"{base_url}/api/search?{urlencode(params)}"
    try:
        payload = http_get_json(url, timeout_sec)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")[:500] if exc.fp else ""
        return [], f"Tempo search HTTP {exc.code}: {body or exc.reason}"
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return [], f"Tempo search failed: {exc}"
    if not isinstance(payload, dict):
        return [], "Tempo search returned non-object JSON."
    traces = payload.get("traces")
    if not isinstance(traces, list):
        return [], "Tempo search response missing traces list."
    return [item for item in traces if isinstance(item, dict)], ""


def fetch_tempo_attribute_values(
    base_url: str,
    attribute: str,
    timeout_sec: float,
    query: str = "",
    start_sec: int | None = None,
    end_sec: int | None = None,
) -> tuple[list[str], str]:
    """Return Tempo attribute values from GET /api/v2/search/tag/<attribute>/values."""
    if not base_url:
        return [], "TEMPO_URL is not set."
    params: dict[str, str] = {"q": query}
    if start_sec is not None:
        params["start"] = str(int(start_sec))
    if end_sec is not None:
        params["end"] = str(int(end_sec))
    quoted_attribute = urllib.parse.quote(attribute, safe="")
    url = f"{base_url}/api/v2/search/tag/{quoted_attribute}/values?{urlencode(params)}"
    try:
        payload = http_get_json(url, timeout_sec)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")[:500] if exc.fp else ""
        return [], f"Tempo attribute values HTTP {exc.code}: {body or exc.reason}"
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return [], f"Tempo attribute values failed: {exc}"

    match payload:
        case list() as items:
            values = items
        case {"tagValues": list() as items}:
            values = items
        case {"values": list() as items}:
            values = items
        case _:
            return [], "Tempo attribute values response did not include a values list."
    normalized: list[str] = []
    for value in values:
        match value:
            case {"value": {"string": str() as raw_value}}:
                text = raw_value.strip()
            case {"value": raw_value}:
                text = str(raw_value).strip()
            case dict():
                continue
            case _:
                text = str(value).strip()
        if text:
            normalized.append(text)
    return normalized, ""


def fetch_tempo_search_trace_ids(
    base_url: str,
    search_q: str,
    timeout_sec: float,
    limit: int = 20,
    start_sec: int | None = None,
    end_sec: int | None = None,
) -> tuple[list[str], str]:
    traces, err = fetch_tempo_search_traces(
        base_url,
        search_q,
        timeout_sec,
        limit=limit,
        start_sec=start_sec,
        end_sec=end_sec,
    )
    if err:
        return [], err
    ids: list[str] = []
    for item in traces:
        tid = item.get("traceID") or item.get("traceId")
        if tid:
            ids.append(str(tid))
    return ids, ""


def default_tempo_search_window_sec(lookback_hours: float = 24.0) -> tuple[int, int]:
    """(start_sec, end_sec) ending at the synthetic eval time, for Tempo APIs."""
    end_sec = int(synthetic_eval_time_unix())
    start_sec = end_sec - int(lookback_hours * 3600)
    return start_sec, end_sec


def fetch_grafana_datasources(
    base_url: str, timeout_sec: float
) -> tuple[list[dict[str, Any]], str]:
    if not base_url:
        return [], "GRAFANA_URL is not set."
    url = f"{base_url}/api/datasources"
    try:
        payload = http_get_json(url, timeout_sec)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")[:300] if exc.fp else ""
        return [], f"Grafana datasources HTTP {exc.code}: {body or exc.reason}"
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return [], f"Grafana datasources request failed: {exc}"
    if not isinstance(payload, list):
        return [], "Grafana datasources response is not a list."
    return [x for x in payload if isinstance(x, dict)], ""


def fetch_grafana_datasources_checked(
    ctx: VerifierContext | None,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    if ctx is None:
        return None, "GRAFANA_URL is not set."
    if not ctx.grafana_url:
        return None, "GRAFANA_URL is not set."
    datasources, err = fetch_grafana_datasources(ctx.grafana_url, ctx.timeout_sec)
    if err:
        return None, err
    return datasources, None


def resolve_grafana_datasource(
    datasources: list[dict[str, Any]],
    *,
    name: str | None = None,
    datasource_type: str | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    if name:
        for item in datasources:
            if str(item.get("name", "")) == name:
                return item, None
        return None, f"Grafana datasource named {name!r} was not found."

    if datasource_type:
        want = datasource_type.lower()
        for item in datasources:
            if str(item.get("type", "")).strip().lower() == want:
                return item, None
        return None, f"Grafana datasource type {datasource_type!r} was not found."

    return None, "Datasource validation requires a name or type selector."


def fetch_grafana_dashboard_model(
    base_url: str, uid: str, timeout_sec: float
) -> tuple[dict[str, Any] | None, str]:
    if not base_url:
        return None, "GRAFANA_URL is not set."
    url = f"{base_url}/api/dashboards/uid/{uid}"
    try:
        payload = http_get_json(url, timeout_sec)
    except urllib.error.HTTPError as exc:
        return None, f"Grafana dashboard HTTP {exc.code} for uid={uid}."
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return None, f"Grafana request failed: {exc}"
    if not isinstance(payload, dict):
        return None, "Grafana returned non-object JSON."
    dash = payload.get("dashboard")
    if not isinstance(dash, dict):
        return None, "Grafana response missing dashboard object."
    return dash, ""


def search_grafana_dashboard_uid(
    base_url: str, title: str, timeout_sec: float
) -> tuple[str | None, str]:
    """Resolve a dashboard uid by title search (best-effort)."""
    if not base_url:
        return None, "GRAFANA_URL is not set."
    params = {"type": "dash-db", "query": title}
    url = f"{base_url}/api/search?{urlencode(params)}"
    try:
        payload = http_get_json(url, timeout_sec)
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return None, f"Grafana search failed: {exc}"
    if not isinstance(payload, list) or not payload:
        return None, f"No dashboards found for search query={title!r}."
    want = title.lower()
    for item in payload:
        if not isinstance(item, dict):
            continue
        t = str(item.get("title", "")).lower()
        if t == want:
            uid = item.get("uid")
            return (str(uid) if uid else None), ""
    for item in payload:
        if not isinstance(item, dict):
            continue
        t = str(item.get("title", "")).lower()
        if want in t or t in want:
            uid = item.get("uid")
            return (str(uid) if uid else None), ""
    uid = payload[0].get("uid") if isinstance(payload[0], dict) else None
    return (str(uid) if uid else None), ""
