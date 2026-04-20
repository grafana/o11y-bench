#!/usr/bin/env python3
"""
Generate realistic telemetry data for o11y-bench.
Generates correlated metrics, logs, and traces over ``HOURS_OF_HISTORY`` (see below).

Key properties:
- Deterministic enough for debugging: seeded RNG with a runtime-selected data end time
- Correlated: each simulated request shares a trace_id across logs, traces, and metrics
- Structured: logs are JSON format with extractable fields
- Histograms: standard Prometheus histogram buckets for latency percentiles
- Incidents: deliberate anomaly windows for RCA/incident problems
"""

import bisect
import json
import os
import random
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta

# Configuration
LOKI_URL = "http://localhost:3100"
TEMPO_URL = "http://localhost:4318"  # OTLP HTTP endpoint
TEMPO_QUERY_URL = "http://localhost:3200"
PROMETHEUS_DATA_DIR = "/prometheus"
# Empty OTLP payload is enough to verify the HTTP ingest path is up.
_OTLP_HEALTH_BODY = b'{"resourceSpans":[]}'

HOURS_OF_HISTORY = 24  # 1 day (enough for trend analysis; incidents are 3-6h before end)
METRICS_INTERVAL = 30  # seconds between metric samples

# Services to simulate
SERVICES = [
    {"name": "webapp", "instance": "webapp:8080"},
    {"name": "api-gateway", "instance": "api-gateway:8081"},
    {"name": "user-service", "instance": "user-service:8082"},
    {"name": "order-service", "instance": "order-service:8083"},
    {"name": "payment-service", "instance": "payment-service:8084"},
]
SERVICE_INSTANCES = {svc["name"]: svc["instance"] for svc in SERVICES}

# Traffic pattern by hour (0-23)
HOURLY_TRAFFIC = [
    0.15,
    0.10,
    0.08,
    0.08,
    0.10,
    0.20,  # 00-05: night
    0.40,
    0.70,
    0.90,
    1.00,
    1.00,
    0.95,  # 06-11: morning ramp
    0.90,
    0.95,
    1.00,
    1.00,
    0.95,
    0.90,  # 12-17: afternoon
    0.80,
    0.70,
    0.55,
    0.40,
    0.30,
    0.20,  # 18-23: evening decline
]

HTTP_ENDPOINTS = [
    {"method": "GET", "path": "/api/users", "service": "user-service"},
    {"method": "GET", "path": "/api/products", "service": "order-service"},
    {"method": "POST", "path": "/api/orders", "service": "order-service"},
    {"method": "POST", "path": "/api/payments", "service": "payment-service"},
    {"method": "GET", "path": "/api/cart", "service": "order-service"},
]

SERVICE_DEPENDENCIES = {
    "webapp": ["api-gateway"],
    "api-gateway": ["user-service", "order-service", "payment-service"],
    "user-service": [],
    "order-service": ["user-service", "payment-service"],
    "payment-service": [],
}

# Standard Prometheus histogram buckets
HISTOGRAM_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def rand_hex(n: int) -> str:
    """Generate a deterministic hex string of length n using the seeded RNG."""
    return "".join(f"{random.randint(0, 255):02x}" for _ in range(n // 2 + 1))[:n]


def format_log_timestamp(ts: datetime, *, randomize_millis: bool = False) -> str:
    millis = random.randint(0, 999) if randomize_millis else ts.microsecond // 1000
    return ts.strftime("%Y-%m-%dT%H:%M:%S.") + f"{millis:03d}Z"


def service_http_url(service_name: str, path: str) -> str:
    return f"http://{SERVICE_INSTANCES[service_name]}{path}"


def build_json_log(
    service: str,
    ts: datetime,
    level: str,
    fields: dict[str, object],
    *,
    randomize_millis: bool = False,
) -> dict:
    log_entry = {
        "timestamp": format_log_timestamp(ts, randomize_millis=randomize_millis),
        "level": level,
        "service": service,
        **fields,
    }
    return {
        "labels": {"job": service, "service": service, "level": level},
        "ts_ns": int(ts.timestamp() * 1e9),
        "line": json.dumps(log_entry),
    }


# -- Incident Patterns -------------------------------------------------------


class IncidentConfig:
    """Defines incident windows relative to data end time."""

    def __init__(self, end_time: datetime) -> None:
        self.end_time = end_time

        # Error spike: payment-service error rate jumps, cascading to order-service
        # Starts ~3 hours before data end, lasts 30 minutes
        self.error_spike_start = end_time - timedelta(hours=3)
        self.error_spike_end = self.error_spike_start + timedelta(minutes=30)

        # Latency degradation: order-service p95 latency increases 5x
        # Starts ~6 hours before data end, lasts 45 minutes
        self.latency_start = end_time - timedelta(hours=6)
        self.latency_end = self.latency_start + timedelta(minutes=45)

        # Cache refresh lag: user-service gets stuck refreshing auth/profile cache.
        # Starts ~9 hours before data end, lasts 40 minutes.
        self.cache_refresh_start = end_time - timedelta(hours=9)
        self.cache_refresh_end = self.cache_refresh_start + timedelta(minutes=40)

        # Deployment event: just before error spike starts
        self.deployment_time = self.error_spike_start - timedelta(minutes=2)
        self.cache_deployment_time = self.cache_refresh_start - timedelta(minutes=3)

    def get_error_rate(self, ts: datetime, service: str) -> float:
        """Return error rate for a service at a given time."""
        if self.error_spike_start <= ts <= self.error_spike_end:
            if service == "payment-service":
                return 0.70  # 70% errors — payment-service owns the spike with margin
            if service == "order-service":
                return 0.15  # 15% cascading errors
            if service == "api-gateway":
                return 0.08  # Some cascading
        return 0.02  # Normal 2% error rate

    def get_latency_multiplier(self, ts: datetime, service: str) -> float:
        """Return latency multiplier for a service at a given time."""
        if self.cache_refresh_start <= ts <= self.cache_refresh_end and service == "user-service":
            return 4.0  # auth/profile cache refresh stalls user-service requests
        if self.latency_start <= ts <= self.latency_end:
            if service == "order-service":
                return 5.0  # 5x latency
            if service in ("api-gateway", "webapp"):
                return 2.0  # Upstream services affected
        return 1.0

    def get_slow_probability(self, ts: datetime, path: str) -> float:
        """Return probability that a request at this timestamp/path is flagged slow.

        Slow requests are concentrated in two latency incidents:
        - Order-service latency incident (~6h before end): /api/orders dominates.
        - User-service cache-refresh incident (~9h before end): /api/users slow.

        A small baseline (1%) keeps healthy periods realistic without letting
        background noise outweigh the incident spike in a 6h query window. This
        gives the canonical queries a deterministic, unambiguous dominant slow
        path regardless of which wall-clock end-time the 6h window sits at.
        """
        if self.latency_start <= ts <= self.latency_end:
            if path == "/api/orders":
                return 0.60
            if path in ("/api/products", "/api/cart"):
                return 0.10
        if self.cache_refresh_start <= ts <= self.cache_refresh_end and path == "/api/users":
            return 0.40
        return 0.01  # small baseline slow rate for realism, not enough to dominate

    def get_cache_refresh_lag(self, ts: datetime, service: str) -> int:
        """Synthetic cache refresh lag around the user-service incident."""
        if service != "user-service":
            return 0

        warmup_start = self.cache_refresh_start - timedelta(minutes=8)
        recovery_end = self.cache_refresh_end + timedelta(minutes=25)
        if ts < warmup_start or ts > recovery_end:
            return 0

        if ts <= self.cache_refresh_start:
            warmup_minutes = (ts - warmup_start).total_seconds() / 60.0
            return max(0, round(20 + warmup_minutes * 18))

        if ts <= self.cache_refresh_end:
            incident_progress = (ts - self.cache_refresh_start).total_seconds() / 60.0
            return round(170 + incident_progress * 9)

        cooldown_progress = (ts - self.cache_refresh_end).total_seconds() / 60.0
        return max(0, round(520 - cooldown_progress * 18))

    def get_retry_queue_depth(self, ts: datetime, service: str) -> int:
        """Synthetic retry/backlog depth around the payment-led error incident."""
        if service not in {"payment-service", "order-service"}:
            return 0

        warmup_start = self.error_spike_start - timedelta(minutes=5)
        recovery_end = self.error_spike_end + timedelta(minutes=20)
        if ts < warmup_start or ts > recovery_end:
            return 0

        if ts <= self.error_spike_start:
            warmup_minutes = (ts - warmup_start).total_seconds() / 60.0
            if service == "payment-service":
                return round(5 + warmup_minutes * 3)
            return round(1 + warmup_minutes * 0.8)

        if ts <= self.error_spike_end:
            incident_progress = (ts - self.error_spike_start).total_seconds() / 60.0
            if service == "payment-service":
                return round(20 + incident_progress * 2.7)
            return round(6 + incident_progress * 0.6)

        cooldown_progress = (ts - self.error_spike_end).total_seconds() / 60.0
        if service == "payment-service":
            return max(0, round(100 - cooldown_progress * 4.5))
        return max(0, round(24 - cooldown_progress * 1.0))

    def is_deployment_time(self, ts: datetime) -> bool:
        """Check if this timestamp should have deployment log entries."""
        return abs((ts - self.deployment_time).total_seconds()) < 60

    def is_cache_deployment_time(self, ts: datetime) -> bool:
        """Check if this timestamp should have cache incident deployment entries."""
        return abs((ts - self.cache_deployment_time).total_seconds()) < 60


def get_traffic_multiplier(ts: datetime) -> float:
    base = HOURLY_TRAFFIC[ts.hour]
    if ts.weekday() >= 5:
        base *= 0.6
    return base * random.uniform(0.9, 1.1)


# -- Trace Generation --------------------------------------------------------


def create_otlp_span(
    trace_id: str,
    span_id: str,
    parent_span_id: str | None,
    service_name: str,
    operation_name: str,
    start_time_ns: int,
    duration_ns: int,
    http_method: str | None = None,
    http_url: str | None = None,
    http_route: str | None = None,
    http_status_code: int | None = None,
    error: bool = False,
    span_kind: int = 2,  # SPAN_KIND_SERVER
) -> dict:
    attributes = [{"key": "service.name", "value": {"stringValue": service_name}}]

    if http_method:
        attributes.append({"key": "http.method", "value": {"stringValue": http_method}})
    if http_route:
        attributes.append({"key": "http.route", "value": {"stringValue": http_route}})
    if http_url:
        attributes.append({"key": "http.url", "value": {"stringValue": http_url}})
    if http_status_code:
        attributes.append({"key": "http.status_code", "value": {"intValue": str(http_status_code)}})

    span = {
        "traceId": trace_id,
        "spanId": span_id,
        "name": operation_name,
        "kind": span_kind,
        "startTimeUnixNano": str(start_time_ns),
        "endTimeUnixNano": str(start_time_ns + duration_ns),
        "attributes": attributes,
        "status": {"code": 2 if error else 0},
    }

    if parent_span_id:
        span["parentSpanId"] = parent_span_id

    return span


def generate_request_trace(
    trace_id: str,
    endpoint: dict,
    ts: datetime,
    is_error: bool,
    base_duration_ms: float,
    cache_refresh_active: bool = False,
) -> list[dict]:
    """Generate trace spans for a single request.

    Error propagation mirrors real distributed systems: the error originates
    at the target service (or its dependency) and upstream spans carry the
    HTTP error status code but do NOT have span status=ERROR. This lets
    agents trace the failure to its origin by inspecting span status.
    """
    http_status = 500 if is_error else 200
    start_time_ns = int(ts.timestamp() * 1e9)
    spans = []

    root_span_id = rand_hex(16)
    root_duration_ns = int(base_duration_ms * 1e6 * random.uniform(1.0, 1.2))

    # Upstream spans: carry the HTTP status code but NOT span error status.
    # They observed the failure as a downstream response, not the origin.
    spans.append(
        create_otlp_span(
            trace_id=trace_id,
            span_id=root_span_id,
            parent_span_id=None,
            service_name="webapp",
            operation_name=f"{endpoint['method']} {endpoint['path']}",
            start_time_ns=start_time_ns,
            duration_ns=root_duration_ns,
            http_method=endpoint["method"],
            http_url=service_http_url("webapp", endpoint["path"]),
            http_route=endpoint["path"],
            http_status_code=http_status,
            error=False,
        )
    )

    gateway_span_id = rand_hex(16)
    gateway_start = start_time_ns + int(2e6)
    gateway_duration_ns = int(root_duration_ns * 0.9)

    spans.append(
        create_otlp_span(
            trace_id=trace_id,
            span_id=gateway_span_id,
            parent_span_id=root_span_id,
            service_name="api-gateway",
            operation_name=f"route {endpoint['path']}",
            start_time_ns=gateway_start,
            duration_ns=gateway_duration_ns,
            http_method=endpoint["method"],
            http_url=service_http_url("api-gateway", endpoint["path"]),
            http_route=endpoint["path"],
            http_status_code=http_status,
            error=False,
        )
    )

    target_service = endpoint["service"]
    target_span_id = rand_hex(16)
    target_start = gateway_start + int(1e6)
    target_duration_ns = int(gateway_duration_ns * 0.8)

    # Target service span: this is where the error originates.
    spans.append(
        create_otlp_span(
            trace_id=trace_id,
            span_id=target_span_id,
            parent_span_id=gateway_span_id,
            service_name=target_service,
            operation_name=f"handle {endpoint['method']} {endpoint['path']}",
            start_time_ns=target_start,
            duration_ns=target_duration_ns,
            http_method=endpoint["method"],
            http_url=service_http_url(target_service, endpoint["path"]),
            http_route=endpoint["path"],
            http_status_code=http_status,
            error=is_error,
        )
    )

    if (
        cache_refresh_active
        and target_service == "user-service"
        and endpoint["path"] == "/api/users"
    ):
        refresh_duration_ns = int(target_duration_ns * 0.6)
        spans.append(
            create_otlp_span(
                trace_id=trace_id,
                span_id=rand_hex(16),
                parent_span_id=target_span_id,
                service_name="user-service",
                operation_name="refresh auth cache",
                start_time_ns=target_start + int(target_duration_ns * 0.15),
                duration_ns=refresh_duration_ns,
                error=False,
                span_kind=1,
            )
        )

    deps = SERVICE_DEPENDENCIES.get(target_service, [])
    if deps and random.random() < 0.7:
        dep_service = random.choice(deps)
        # Dependency span also carries error status (the deepest origin).
        spans.append(
            create_otlp_span(
                trace_id=trace_id,
                span_id=rand_hex(16),
                parent_span_id=target_span_id,
                service_name=dep_service,
                operation_name=f"call {dep_service}",
                start_time_ns=target_start + int(target_duration_ns * 0.2),
                duration_ns=int(target_duration_ns * 0.5),
                error=is_error,
                span_kind=3,  # SPAN_KIND_CLIENT
            )
        )

    return spans


# -- Log Generation (JSON format) --------------------------------------------


def generate_request_log(
    trace_id: str,
    service: str,
    endpoint: dict,
    ts: datetime,
    is_error: bool,
    duration_ms: float,
) -> dict:
    """Generate a structured JSON log entry for a request."""
    status = 500 + random.choice([0, 2, 3]) if is_error else 200
    message = (
        random.choice(
            [
                "request failed",
                "internal server error",
                "upstream service error",
                "database connection timeout",
            ]
        )
        if is_error
        else "request completed"
    )
    return build_json_log(
        service,
        ts,
        "error" if is_error else "info",
        {
            "method": endpoint["method"],
            "path": endpoint["path"],
            "status": status,
            "duration_ms": round(duration_ms, 1),
            "trace_id": trace_id,
            "message": message,
        },
        randomize_millis=True,
    )


def generate_deployment_log(service: str, ts: datetime) -> dict:
    """Generate a deployment event log entry."""
    return build_json_log(
        service,
        ts,
        "info",
        {
            "message": f"deployment started: {service} v2.4.1 -> v2.5.0",
            "event": "deployment",
            "version": "v2.5.0",
        },
    )


def generate_warning_log(service: str, ts: datetime) -> dict:
    """Generate a warning log entry (slow query, high memory, retry)."""
    templates = [
        {
            "message": "slow database query",
            "table": random.choice(["users", "orders", "products", "sessions"]),
            "query_ms": random.randint(500, 3000),
        },
        {"message": f"high memory usage: {random.randint(75, 95)}%"},
        {"message": f"retry attempt {random.randint(1, 3)} for upstream call"},
    ]
    tmpl = random.choice(templates)
    log_entry = {"message": tmpl["message"]}
    if "table" in tmpl:
        log_entry["table"] = tmpl["table"]
        log_entry["query_ms"] = tmpl["query_ms"]
    return build_json_log(service, ts, "warn", log_entry, randomize_millis=True)


def generate_retry_backlog_log(service: str, ts: datetime, queue_depth: int) -> dict:
    return build_json_log(
        service,
        ts,
        "warn",
        {
            "queue": "payment-retries",
            "queue_depth": queue_depth,
            "message": "retry backlog elevated",
        },
    )


def generate_cache_refresh_log(service: str, ts: datetime, lag_seconds: int) -> dict:
    stale_keys = max(20, round(lag_seconds * random.uniform(1.3, 1.8)))
    return build_json_log(
        service,
        ts,
        "warn",
        {
            "job_name": "auth-cache-refresh",
            "lag_seconds": lag_seconds,
            "stale_keys": stale_keys,
            "message": "cache refresh lag elevated",
        },
    )


# -- Metrics Accumulation ----------------------------------------------------


class ServiceMetrics:
    """Accumulates metrics for a single service."""

    def __init__(self) -> None:
        self.cpu_seconds = 0.0
        self.requests_by_status: dict[int, int] = {}
        self.duration_sum = 0.0
        self.duration_count = 0
        # Histogram bucket counts (cumulative will be computed at write time)
        self.bucket_counts = [0] * (len(HISTOGRAM_BUCKETS) + 1)  # +1 for +Inf

    def record_request(self, status: int, duration_s: float) -> None:
        self.requests_by_status[status] = self.requests_by_status.get(status, 0) + 1
        self.duration_sum += duration_s
        self.duration_count += 1

        # Place duration into histogram bucket
        idx = bisect.bisect_left(HISTOGRAM_BUCKETS, duration_s)
        # idx is the first bucket >= duration_s; we want the bucket that contains it
        # bisect_left gives index where duration_s would be inserted
        # All buckets at idx and above should be incremented (cumulative)
        # But we track raw counts per bucket, compute cumulative at write time
        if idx < len(HISTOGRAM_BUCKETS):
            self.bucket_counts[idx] += 1
        else:
            self.bucket_counts[-1] += 1  # +Inf bucket


# -- Unified Event Generator -------------------------------------------------


def push_traces_batch(spans: list[dict], retries: int = 3) -> bool:
    if not spans:
        return True

    service_spans: dict[str, list] = {}
    for span in spans:
        service = next(
            (
                attr["value"]["stringValue"]
                for attr in span.get("attributes", [])
                if attr.get("key") == "service.name"
            ),
            "unknown",
        )
        service_spans.setdefault(service, []).append(span)

    resource_spans = [
        {
            "resource": {
                "attributes": [{"key": "service.name", "value": {"stringValue": service}}]
            },
            "scopeSpans": [{"scope": {"name": "o11y-bench"}, "spans": svc_spans}],
        }
        for service, svc_spans in service_spans.items()
    ]

    payload = json.dumps({"resourceSpans": resource_spans}).encode()

    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                f"{TEMPO_URL}/v1/traces",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=10)
            return True
        except urllib.error.HTTPError as exc:
            if exc.code == 503 and attempt < retries - 1:
                time.sleep(0.5)
                continue
            return False
        except Exception:
            return False
    return False


def push_logs_batch(logs: list[dict]) -> bool:
    if not logs:
        return True

    streams: dict[str, dict] = {}
    for entry in logs:
        key = json.dumps(entry["labels"], sort_keys=True)
        if key not in streams:
            streams[key] = {"stream": entry["labels"], "values": []}
        streams[key]["values"].append([str(entry["ts_ns"]), entry["line"]])

    # Loki requires entries per stream to be in timestamp order
    for stream in streams.values():
        stream["values"].sort(key=lambda v: int(v[0]))

    try:
        req = urllib.request.Request(
            f"{LOKI_URL}/loki/api/v1/push",
            data=json.dumps({"streams": list(streams.values())}).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=10)
        return True
    except Exception as exc:
        log(f"  Loki error: {exc}")
        return False


def wait_for_tempo(max_attempts: int = 10) -> None:
    url = f"{TEMPO_URL}/v1/traces"
    headers = {"Content-Type": "application/json"}
    for attempt in range(max_attempts):
        try:
            urllib.request.urlopen(
                urllib.request.Request(url, data=_OTLP_HEALTH_BODY, headers=headers),
                timeout=5,
            )
            return
        except Exception as exc:
            if attempt == max_attempts - 1:
                log(f"  Tempo OTLP not ready after {max_attempts} attempts: {exc}")
                raise RuntimeError("Tempo OTLP never became ready") from exc
            time.sleep(1)


def flush_tempo(max_attempts: int = 5) -> None:
    req = urllib.request.Request(f"{TEMPO_QUERY_URL}/flush", data=b"", method="POST")
    for attempt in range(max_attempts):
        try:
            urllib.request.urlopen(req, timeout=10)
            return
        except Exception as exc:
            if attempt == max_attempts - 1:
                log(f"  Tempo flush failed after {max_attempts} attempts: {exc}")
                raise RuntimeError("Tempo flush never succeeded") from exc
            time.sleep(1)


def fetch_tempo_search_traces(
    query: str,
    start_sec: int,
    end_sec: int,
    *,
    limit: int = 20,
) -> tuple[list[dict], str]:
    params = urllib.parse.urlencode(
        {"q": query, "start": start_sec, "end": end_sec, "limit": limit}
    )
    url = f"{TEMPO_QUERY_URL}/api/search?{params}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            payload = json.load(resp)
    except Exception as exc:
        return [], f"Tempo search failed: {exc}"
    traces = payload.get("traces")
    if not isinstance(traces, list):
        return [], "Tempo search response missing traces list."
    return [item for item in traces if isinstance(item, dict)], ""


def _probe_detail(name: str, traces: list[dict], err: str) -> str:
    if traces:
        return f"{name}={len(traces)}"
    return f"{name}={err or 'empty'}"


def wait_for_tempo_searchable(
    end_time: datetime, max_attempts: int = 30, poll_interval: float = 1.0
) -> None:
    # Probes are tracked independently: broad and error searches may become ready
    # on different polls (errors are sparser), and requiring both on the same tick
    # caused spurious failures. Poll interval matches Tempo's blocklist_poll.
    start_sec = int((end_time - timedelta(hours=24)).timestamp())
    end_sec = int(end_time.timestamp())
    broad_traces: list[dict] = []
    broad_err = ""
    error_traces: list[dict] = []
    error_err = ""

    for attempt in range(max_attempts):
        attempt_num = attempt + 1
        if not broad_traces:
            broad_traces, broad_err = fetch_tempo_search_traces("", start_sec, end_sec, limit=5)
        if not error_traces:
            error_traces, error_err = fetch_tempo_search_traces(
                "{ span:status = error }", start_sec, end_sec, limit=5
            )

        if broad_traces and error_traces:
            log(
                "  Tempo search ready "
                f"(attempt {attempt_num}/{max_attempts}, "
                f"broad={len(broad_traces)}, error={len(error_traces)})"
            )
            return

        if attempt == 0 or attempt_num % 5 == 0:
            details = [
                _probe_detail("broad", broad_traces, broad_err),
                _probe_detail("error", error_traces, error_err),
            ]
            log(
                "  Waiting for Tempo search "
                f"(attempt {attempt_num}/{max_attempts}): {'; '.join(details)}"
            )
        time.sleep(poll_interval)

    details = [
        _probe_detail("broad", broad_traces, broad_err),
        _probe_detail("error", error_traces, error_err),
    ]
    log(f"  Tempo search never became ready after {max_attempts} attempts: {'; '.join(details)}")
    raise RuntimeError("Tempo search never became ready")


def data_end_time_utc() -> datetime:
    raw = os.environ.get("O11Y_SCENARIO_TIME_ISO", "").strip()
    if not raw:
        return datetime.now(UTC).replace(microsecond=0)
    return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(UTC)


def generate_all_data() -> dict[str, ServiceMetrics]:
    """Generate correlated metrics, logs, and traces in a single pass."""
    log("Generating correlated telemetry data...")

    end_time = data_end_time_utc()
    start_time = end_time - timedelta(hours=HOURS_OF_HISTORY)
    incidents = IncidentConfig(end_time)
    # Tempo is required for the benchmark. If OTLP never comes up, fail fast.
    wait_for_tempo()

    # State
    service_metrics = {s["name"]: ServiceMetrics() for s in SERVICES}
    trace_batch: list[dict] = []
    log_batch: list[dict] = []
    metrics_file = "/tmp/metrics.txt"

    total_requests = 0
    total_traces = 0
    total_logs = 0
    total_samples = 0

    current = start_time
    steps_done = 0
    total_steps = int((end_time - start_time).total_seconds() / METRICS_INTERVAL)
    last_progress = -1
    last_minute = None

    # Pre-compute label strings for each service (avoid repeated f-string formatting)
    svc_labels = [
        (svc["name"], svc["instance"], f'job="{svc["name"]}",instance="{svc["instance"]}"')
        for svc in SERVICES
    ]

    with open(metrics_file, "w", buffering=1024 * 1024) as mf:
        # Write OpenMetrics TYPE headers (required for promtool to
        # correctly interpret counters and histograms)
        mf.write(
            "# TYPE up gauge\n"
            "# TYPE process_cpu_seconds_total counter\n"
            "# TYPE process_resident_memory_bytes gauge\n"
            "# TYPE service_retry_queue_depth gauge\n"
            "# TYPE service_cache_refresh_lag_seconds gauge\n"
            "# TYPE http_requests_total counter\n"
            "# TYPE http_request_duration_seconds histogram\n"
            "# TYPE go_goroutines gauge\n"
        )

        # Buffer for batched writes
        buf: list[str] = []
        buf_size = 0

        while current < end_time:
            traffic = get_traffic_multiplier(current)
            ts_epoch = current.timestamp()
            current_minute = int(ts_epoch) // 60

            # Generate requests/logs/traces once per minute
            if current_minute != last_minute:
                last_minute = current_minute
                num_requests = max(1, int(traffic * 6))

                for _ in range(num_requests):
                    req_ts = current + timedelta(seconds=random.uniform(0, 59))
                    trace_id = rand_hex(32)
                    endpoint = random.choice(HTTP_ENDPOINTS)
                    target_service = endpoint["service"]

                    error_rate = incidents.get_error_rate(req_ts, target_service)
                    latency_mult = incidents.get_latency_multiplier(req_ts, target_service)
                    slow_prob = incidents.get_slow_probability(req_ts, endpoint["path"])
                    is_error = random.random() < error_rate
                    is_slow = random.random() < slow_prob

                    base_duration_ms = (
                        random.randint(500, 3000) if is_slow else random.randint(10, 50)
                    )
                    base_duration_ms *= latency_mult
                    duration_s = base_duration_ms / 1000.0

                    spans = generate_request_trace(
                        trace_id,
                        endpoint,
                        req_ts,
                        is_error,
                        base_duration_ms,
                        cache_refresh_active=(
                            incidents.cache_refresh_start <= req_ts <= incidents.cache_refresh_end
                        ),
                    )
                    trace_batch.extend(spans)
                    total_traces += 1

                    log_entry = generate_request_log(
                        trace_id,
                        target_service,
                        endpoint,
                        req_ts,
                        is_error,
                        base_duration_ms,
                    )
                    log_batch.append(log_entry)
                    total_logs += 1

                    # Edge log so Loki carries the same trace_id as the trace root (webapp span).
                    webapp_log = generate_request_log(
                        trace_id,
                        "webapp",
                        endpoint,
                        req_ts,
                        is_error,
                        base_duration_ms * 0.4,
                    )
                    log_batch.append(webapp_log)
                    total_logs += 1

                    if random.random() < 0.3:
                        upstream_log = generate_request_log(
                            trace_id,
                            "api-gateway",
                            endpoint,
                            req_ts,
                            is_error,
                            base_duration_ms * 1.1,
                        )
                        log_batch.append(upstream_log)
                        total_logs += 1

                    status = 500 if is_error else 200
                    service_metrics[target_service].record_request(status, duration_s)

                    total_requests += 1

                # Warning logs (once per minute)
                for svc in SERVICES:
                    if random.random() < 0.05 * traffic:
                        warn_ts = current + timedelta(seconds=random.uniform(0, 59))
                        warn_log = generate_warning_log(svc["name"], warn_ts)
                        log_batch.append(warn_log)
                        total_logs += 1

                for backlog_service in ("payment-service", "order-service"):
                    backlog_depth = incidents.get_retry_queue_depth(current, backlog_service)
                    if backlog_depth >= 8:
                        backlog_ts = current + timedelta(seconds=random.uniform(0, 59))
                        backlog_log = generate_retry_backlog_log(
                            backlog_service, backlog_ts, backlog_depth
                        )
                        log_batch.append(backlog_log)
                        total_logs += 1

                cache_refresh_lag = incidents.get_cache_refresh_lag(current, "user-service")
                if cache_refresh_lag >= 90:
                    cache_ts = current + timedelta(seconds=random.uniform(0, 59))
                    cache_log = generate_cache_refresh_log(
                        "user-service", cache_ts, cache_refresh_lag
                    )
                    log_batch.append(cache_log)
                    total_logs += 1

                # Deployment logs at incident time
                if incidents.is_deployment_time(current):
                    for svc_name in ["payment-service", "order-service"]:
                        dep_log = generate_deployment_log(svc_name, current)
                        log_batch.append(dep_log)
                        total_logs += 1
                if incidents.is_cache_deployment_time(current):
                    dep_log = generate_deployment_log("user-service", current)
                    log_batch.append(dep_log)
                    total_logs += 1

            # Write metrics snapshot every METRICS_INTERVAL seconds
            ts = f" {ts_epoch}\n"
            for job, _instance, labels in svc_labels:
                sm = service_metrics[job]
                samples_before = len(buf)

                # Keep scrape targets "up" stable; flaky up gauges break PromQL oracles
                # (e.g. sum(up)) and confuse health-check tasks without adding realism.
                up = 1
                buf.append(f"up{{{labels}}} {up}{ts}")

                sm.cpu_seconds += 0.05 * traffic * random.uniform(0.8, 1.2)
                buf.append(f"process_cpu_seconds_total{{{labels}}} {sm.cpu_seconds:.3f}{ts}")

                memory = int(100_000_000 * (0.8 + 0.4 * traffic) * random.uniform(0.95, 1.05))
                buf.append(f"process_resident_memory_bytes{{{labels}}} {memory}{ts}")

                retry_depth = incidents.get_retry_queue_depth(current, job)
                buf.append(f"service_retry_queue_depth{{{labels}}} {retry_depth}{ts}")

                cache_refresh_lag = incidents.get_cache_refresh_lag(current, job)
                buf.append(f"service_cache_refresh_lag_seconds{{{labels}}} {cache_refresh_lag}{ts}")

                for status_code, count in sorted(sm.requests_by_status.items()):
                    buf.append(
                        f'http_requests_total{{{labels},status="{status_code}"}} {count}{ts}'
                    )

                buf.append(
                    f"http_request_duration_seconds_sum{{{labels}}} {sm.duration_sum:.3f}{ts}"
                )
                buf.append(
                    f"http_request_duration_seconds_count{{{labels}}} {sm.duration_count}{ts}"
                )

                cumulative = 0
                for i, le in enumerate(HISTOGRAM_BUCKETS):
                    cumulative += sm.bucket_counts[i]
                    buf.append(
                        "http_request_duration_seconds_bucket"
                        f'{{{labels},le="{le}"}} {cumulative}{ts}'
                    )
                cumulative += sm.bucket_counts[-1]
                buf.append(
                    f'http_request_duration_seconds_bucket{{{labels},le="+Inf"}} {cumulative}{ts}'
                )

                goroutines = int(30 + 50 * traffic * random.uniform(0.9, 1.1))
                buf.append(f"go_goroutines{{{labels}}} {goroutines}{ts}")

                total_samples += len(buf) - samples_before

            buf_size += 1
            # Flush buffer every 500 steps (~50KB)
            if buf_size >= 500:
                mf.write("".join(buf))
                buf.clear()
                buf_size = 0

            # Flush batches periodically (small batches help Tempo process spans faster)
            if len(trace_batch) >= 200:
                push_traces_batch(trace_batch)
                trace_batch = []

            if len(log_batch) >= 5000:
                push_logs_batch(log_batch)
                log_batch = []

            steps_done += 1
            progress = int(steps_done / total_steps * 100)
            if progress >= last_progress + 20:
                log(
                    "  Progress: "
                    f"{progress}% ({total_requests:,} requests, {total_traces:,} traces, "
                    f"{total_logs:,} logs)"
                )
                last_progress = progress

            current += timedelta(seconds=METRICS_INTERVAL)

        if buf:
            mf.write("".join(buf))
        mf.write("# EOF\n")

    # Flush remaining batches
    if trace_batch:
        push_traces_batch(trace_batch)
    if log_batch:
        push_logs_batch(log_batch)

    log("  Flushing Tempo blocks...")
    flush_tempo()
    log("  Waiting for Tempo search to become ready...")
    wait_for_tempo_searchable(end_time)

    log(f"  Requests: {total_requests:,}")
    log(f"  Traces: {total_traces:,}")
    log(f"  Logs: {total_logs:,}")
    log(f"  Metric samples: {total_samples:,}")

    # Import metrics to Prometheus TSDB
    log("  Importing metrics to Prometheus TSDB...")
    try:
        result = subprocess.run(
            [
                "promtool",
                "tsdb",
                "create-blocks-from",
                "openmetrics",
                metrics_file,
                PROMETHEUS_DATA_DIR,
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode == 0:
            log("  Metrics import complete")
        else:
            stderr = result.stderr[:500] if result.stderr else "no stderr"
            log(f"  promtool failed (rc={result.returncode}): {stderr}")
            raise SystemExit(1) from None
    except subprocess.TimeoutExpired as exc:
        log("  promtool timed out")
        raise SystemExit(1) from exc
    except SystemExit:
        raise
    except Exception as exc:
        log(f"  promtool error: {exc}")
        raise SystemExit(1) from exc

    try:
        os.remove(metrics_file)
    except Exception:
        pass

    env_ts = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    with open("/tmp/env_timestamp", "w") as f:
        f.write(env_ts)
    log(f"  Environment timestamp: {env_ts}")

    return service_metrics


# -- Main --------------------------------------------------------------------


def main() -> None:
    random.seed(42)

    log("=" * 50)
    log("Telemetry Data Generator (deterministic, correlated)")
    log(f"History: {HOURS_OF_HISTORY}h ({HOURS_OF_HISTORY // 24}d), interval: {METRICS_INTERVAL}s")
    log("=" * 50)

    start = time.time()
    generate_all_data()
    elapsed = time.time() - start

    log(f"Data generation complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
