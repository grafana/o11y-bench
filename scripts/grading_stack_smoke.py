#!/usr/bin/env python3
"""
Live HTTP smoke test for grading/env_context.py (Prometheus, Loki, Tempo, Grafana).

Use after the o11y-stack container is up with ports published, e.g.:

  docker build -t o11y-bench-stack ./docker
  docker run --rm -p 3000:3000 -p 9090:9090 -p 3100:3100 -p 3200:3200 -p 8080:8080 o11y-bench-stack

Then:

  uv run python scripts/grading_stack_smoke.py

Set GRAFANA_URL, PROMETHEUS_URL, LOKI_URL, TEMPO_URL if not using localhost defaults.
Exits 0 on success, 1 on any required failure.

Prometheus instant ratio check is **optional**: if the query returns no scalar (empty TSDB, wrong
time range, or parse failure), the script prints SKIP and continues—Loki + Tempo + Grafana health
remain required.
"""

import argparse
import os
import urllib.error

from grading.env_context import (
    default_tempo_search_window_sec,
    fetch_grafana_dashboard_model,
    fetch_loki_instant,
    fetch_prometheus_instant,
    fetch_tempo_search_trace_ids,
    http_get_json,
)


def _ok(name: str, detail: str) -> None:
    print(f"OK  {name}: {detail}")


def _fail(errors: list[str], name: str, detail: str) -> None:
    errors.append(f"{name}: {detail}")
    print(f"FAIL {name}: {detail}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke test env-backed grading checks against live stack"
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout seconds")
    args = parser.parse_args()
    timeout = args.timeout

    g = os.environ.get("GRAFANA_URL", "http://127.0.0.1:3000").strip().rstrip("/")
    p = os.environ.get("PROMETHEUS_URL", "http://127.0.0.1:9090").strip().rstrip("/")
    loki = os.environ.get("LOKI_URL", "http://127.0.0.1:3100").strip().rstrip("/")
    t = os.environ.get("TEMPO_URL", "http://127.0.0.1:3200").strip().rstrip("/")

    errors: list[str] = []

    # --- Prometheus (promql-error-rate canonical: *-service jobs emit http_requests_total) ---
    expr = (
        'sum(rate(http_requests_total{status=~"5..",job=~".+-service"}[1h])) '
        '/ sum(rate(http_requests_total{job=~".+-service"}[1h]))'
    )
    val, err = fetch_prometheus_instant(p, expr, timeout)
    if err or val is None:
        print(
            f"SKIP prometheus_instant: {err or 'no scalar'} "
            "(optional when Prometheus has no data for this query)"
        )
    elif val < 0 or val > 1.001:
        _fail(errors, "prometheus_instant", f"unexpected value {val!r} (expected ratio in [0,1])")
    else:
        _ok("prometheus_instant", f"5xx share ratio ≈ {val:.6f}")

    # --- Loki (logql-metrics-query canonical) ---
    logql = 'sum(count_over_time({level="error"}[6h]))'
    lv, lerr = fetch_loki_instant(loki, logql, timeout)
    if lerr:
        _fail(errors, "loki_instant", lerr)
    elif lv is None or lv < 0:
        _fail(errors, "loki_instant", f"unexpected value {lv!r}")
    else:
        _ok("loki_instant", f"error-ish log lines (6h) ≈ {lv:.0f}")

    # --- Loki (logql-top-5xx-endpoint: top path by 5xx count) ---
    logql_top = (
        'topk(1, sum by (path) (count_over_time({job=~".+"} | json | path=~".+" | '
        '__error__="" | status >= 500 [6h])))'
    )
    topv, toperr = fetch_loki_instant(loki, logql_top, timeout)
    if toperr:
        _fail(errors, "loki_top_path_5xx", toperr)
    elif topv is None or topv < 0:
        _fail(errors, "loki_top_path_5xx", f"unexpected value {topv!r}")
    else:
        _ok("loki_top_path_5xx", f"worst-path 5xx count (6h) ≈ {topv:.0f}")

    # --- Tempo search (same family as traceql-find-service-traces) ---
    start_sec, end_sec = default_tempo_search_window_sec(48.0)
    tempo_queries = [
        '{ resource.service.name = "order-service" }',
        "{ resource.service.name = `order-service` }",
    ]
    ids: list[str] = []
    last_terr = ""
    for q in tempo_queries:
        ids, last_terr = fetch_tempo_search_trace_ids(
            t, q, timeout, limit=30, start_sec=start_sec, end_sec=end_sec
        )
        if ids:
            _ok("tempo_search", f"{len(ids)} trace(s) for query {q[:50]}…, e.g. {ids[0][:16]}…")
            break
    if not ids:
        detail = last_terr or "no traces for any fallback TraceQL query"
        _fail(errors, "tempo_search", detail)

    # --- Grafana health (required) ---
    try:
        health = http_get_json(f"{g}/api/health", timeout)
        if isinstance(health, dict) and health.get("database") == "ok":
            _ok("grafana_health", "api/health database ok")
        else:
            _ok("grafana_health", str(health)[:120])
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        _fail(errors, "grafana_health", str(exc))

    # --- Grafana dashboard uid (optional: only provisioned when Harbor mounts setup.json) ---
    dash, gerr = fetch_grafana_dashboard_model(g, "service-overview", timeout)
    if gerr:
        print(f"SKIP grafana_dashboard_uid: {gerr} (expected without task setup.json)")
    else:
        blob = str(dash).lower()
        if "panel" not in blob and "panels" not in blob:
            _fail(errors, "grafana_dashboard_uid", "dashboard JSON missing panels")
        else:
            _ok("grafana_dashboard_uid", "service-overview model fetched")

    if errors:
        print("\nSmoke test failed:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("\nAll required smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
