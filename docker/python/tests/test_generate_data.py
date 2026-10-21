import json
import random
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from o11y_stack import generate_data


@contextmanager
def seeded(seed: int) -> Iterator[None]:
    """Seed the generator's global RNG for one block, then put the caller's state back."""
    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)


def test_incident_config_scopes_incidents_to_the_right_services():
    end_time = datetime(2026, 3, 30, 12, 0, tzinfo=UTC)
    incidents = generate_data.IncidentConfig(end_time)

    cache_mid = incidents.cache_refresh_start + timedelta(minutes=5)
    error_mid = incidents.error_spike_start + timedelta(minutes=10)

    assert incidents.get_cache_refresh_lag(cache_mid, "user-service") > 0
    assert incidents.get_cache_refresh_lag(cache_mid, "order-service") == 0
    assert incidents.get_cache_refresh_lag(cache_mid, "payment-service") == 0
    assert incidents.get_latency_multiplier(cache_mid, "user-service") > 1.0
    assert incidents.get_latency_multiplier(cache_mid, "order-service") == 1.0
    assert incidents.get_retry_queue_depth(error_mid, "payment-service") > 0
    assert incidents.get_retry_queue_depth(error_mid, "order-service") > 0
    assert incidents.get_retry_queue_depth(error_mid, "user-service") == 0


def test_a_slow_checkout_keeps_its_time_on_order_service_not_the_payment_call():
    # During the order-service latency incident the handler is slow; the call it makes is not.
    with seeded(1):
        traces = [
            generate_data.generate_request_trace(
                trace_id="b" * 32,
                endpoint={"method": "POST", "path": "/api/orders", "service": "order-service"},
                ts=datetime(2026, 3, 30, 12, 0, tzinfo=UTC),
                is_error=False,
                base_duration_ms=8000,
            )
            for _ in range(20)
        ]

    def duration_ms(span: dict) -> float:
        return (int(span["endTimeUnixNano"]) - int(span["startTimeUnixNano"])) / 1e6

    calls = [s for spans in traces for s in spans if s["name"].startswith("call ")]
    handlers = [s for spans in traces for s in spans if s["name"] == "handle POST /api/orders"]
    assert calls, "no dependency calls generated"
    assert max(duration_ms(s) for s in calls) <= 25
    assert min(duration_ms(s) for s in handlers) > 5000


def test_generate_request_trace_preserves_causal_chain_and_cache_child_span():
    spans = generate_data.generate_request_trace(
        trace_id="a" * 32,
        endpoint={"method": "GET", "path": "/api/users", "service": "user-service"},
        ts=datetime(2026, 3, 30, 12, 0, tzinfo=UTC),
        is_error=True,
        base_duration_ms=1200,
        cache_refresh_active=True,
    )

    def _span(service_name: str, *, has_http_route: bool) -> dict:
        for span in spans:
            attributes = span["attributes"]
            if not any(
                attr["key"] == "service.name" and attr["value"]["stringValue"] == service_name
                for attr in attributes
            ):
                continue
            has_route = any(attr["key"] == "http.route" for attr in attributes)
            if has_route == has_http_route:
                return span
        raise AssertionError(f"missing span for {service_name!r}")

    webapp_span = _span("webapp", has_http_route=True)
    gateway_span = _span("api-gateway", has_http_route=True)
    target_span = _span("user-service", has_http_route=True)
    cache_span = _span("user-service", has_http_route=False)

    assert webapp_span.get("parentSpanId") is None
    assert gateway_span["parentSpanId"] == webapp_span["spanId"]
    assert target_span["parentSpanId"] == gateway_span["spanId"]
    assert cache_span["parentSpanId"] == target_span["spanId"]
    assert webapp_span["status"]["code"] == 0
    assert gateway_span["status"]["code"] == 0
    assert target_span["status"]["code"] == 2
    assert any(
        attr["key"] == "http.status_code" and attr["value"]["intValue"] == "500"
        for attr in target_span["attributes"]
    )
    assert all(attr["key"] != "http.route" for attr in cache_span["attributes"])


def test_log_builders_emit_core_schema():
    cases = [
        (
            generate_data.generate_request_log,
            (
                "trace-1",
                "user-service",
                {"method": "GET", "path": "/api/users"},
                datetime(2026, 3, 30, 12, 0, 5, 456000, tzinfo=UTC),
                False,
                42.7,
            ),
            "info",
            {"method", "path", "status", "duration_ms", "trace_id"},
        ),
        (
            generate_data.generate_deployment_log,
            ("payment-service", datetime(2026, 3, 30, 12, 0, 5, tzinfo=UTC)),
            "info",
            {"event", "version"},
        ),
        (
            generate_data.generate_retry_backlog_log,
            ("payment-service", datetime(2026, 3, 30, 12, 0, 5, tzinfo=UTC), 17),
            "warn",
            {"queue", "queue_depth"},
        ),
        (
            generate_data.generate_cache_refresh_log,
            ("user-service", datetime(2026, 3, 30, 12, 0, 5, tzinfo=UTC), 45),
            "warn",
            {"job_name", "lag_seconds", "stale_keys"},
        ),
    ]

    for factory, args, expected_level, expected_fields in cases:
        payload = json.loads(factory(*args)["line"])
        service_name = args[1] if factory is generate_data.generate_request_log else args[0]

        assert payload["level"] == expected_level
        assert payload["service"] == service_name
        assert payload["timestamp"].endswith("Z")
        assert datetime.fromisoformat(payload["timestamp"].replace("Z", "+00:00")).tzinfo == UTC
        assert expected_fields.issubset(payload)

        if factory is generate_data.generate_request_log:
            assert payload["method"] == "GET"
            assert payload["path"] == "/api/users"
            assert payload["status"] == 200
            assert payload["trace_id"] == "trace-1"
        elif factory is generate_data.generate_deployment_log:
            assert payload["event"] == "deployment"
            assert payload["version"] == "v2.5.0"
        elif factory is generate_data.generate_retry_backlog_log:
            assert payload["queue_depth"] == 17
        else:
            assert payload["job_name"] == "auth-cache-refresh"
            assert payload["lag_seconds"] == 45
            assert payload["stale_keys"] > payload["lag_seconds"]


def test_service_metrics_accumulates_counts_and_histogram_buckets():
    metrics = generate_data.ServiceMetrics()

    metrics.record_request(200, 0.004)
    metrics.record_request(500, 0.01)
    metrics.record_request(500, 12.0)

    assert metrics.requests_by_status == {200: 1, 500: 2}
    assert metrics.duration_count == 3
    assert metrics.duration_sum == pytest.approx(12.014)
    assert sum(metrics.bucket_counts[:-1]) == 2
    assert metrics.bucket_counts[-1] == 1


def test_data_end_time_is_the_override_floored_to_the_minute(monkeypatch: pytest.MonkeyPatch):
    # Whole minutes, so subquery steps meet the same samples whatever the clock's seconds.
    monkeypatch.setenv("O11Y_SCENARIO_TIME_ISO", "2026-04-04T10:05:14Z")

    assert generate_data.data_end_time_utc() == datetime(2026, 4, 4, 10, 5, tzinfo=UTC)


@pytest.fixture(scope="module")
def generate(
    tmp_path_factory: pytest.TempPathFactory,
) -> Callable[[str], tuple[dict, list[dict]]]:
    """`generate(clock)` runs the whole generator at that scenario time: (metrics, pushed logs).

    Only the network and promtool are faked, and only for the duration of a run.
    """
    out_dir = tmp_path_factory.mktemp("generated")
    runs: dict[str, tuple[dict, list[dict]]] = {}

    def run(clock: str) -> tuple[dict, list[dict]]:
        if clock in runs:
            return runs[clock]
        logs: list[dict] = []
        with pytest.MonkeyPatch.context() as mp:
            for name in ("wait_for_loki_complete", "wait_for_tempo_searchable"):
                mp.setattr(generate_data, name, lambda *args, **kwargs: None)
            mp.setattr(generate_data, "push_traces_batch", lambda spans: True)
            mp.setattr(generate_data, "push_logs_batch", lambda batch: logs.extend(batch))
            mp.setattr(
                generate_data,
                "open",
                lambda path, *a, **kw: open(out_dir / Path(path).name, *a, **kw),
                raising=False,
            )
            mp.setenv("O11Y_SCENARIO_TIME_ISO", clock)
            with seeded(42):
                metrics = generate_data.generate_all_data()
        runs[clock] = ({name: vars(m) for name, m in metrics.items()}, logs)
        return runs[clock]

    return run


WEEKDAY_NIGHT = "2026-10-19T23:01:50Z"
SATURDAY_NOON = "2026-10-24T12:30:05Z"


def test_generated_data_relative_to_the_clock_does_not_depend_on_the_clock(generate) -> None:
    # A Monday-night job and a Saturday-noon regrade must generate the same telemetry, so the
    # truths a regrade reads match what the agent saw.
    weekday_metrics, weekday_logs = generate(WEEKDAY_NIGHT)
    saturday_metrics, saturday_logs = generate(SATURDAY_NOON)

    assert weekday_logs
    assert saturday_metrics == weekday_metrics
    assert len(saturday_logs) == len(weekday_logs)


def _payloads(logs: list[dict]) -> list[tuple[int, dict]]:
    return [(log["ts_ns"], json.loads(log["line"])) for log in logs]


def test_each_rollout_is_logged_once(generate) -> None:
    # A second identical line reads as a second deployment.
    _, logs = generate(WEEKDAY_NIGHT)

    rollouts = sorted(p["service"] for _, p in _payloads(logs) if p.get("event") == "deployment")

    assert rollouts == ["order-service", "payment-service", "user-service"]


def test_the_user_service_rollout_precedes_the_first_cache_refresh_warning(generate) -> None:
    # The rollout is the cache incident's trigger.
    _, logs = generate(WEEKDAY_NIGHT)
    payloads = _payloads(logs)

    rollout = min(
        ts
        for ts, p in payloads
        if p.get("event") == "deployment" and p["service"] == "user-service"
    )
    first_warning = min(ts for ts, p in payloads if p.get("job_name") == "auth-cache-refresh")

    assert rollout < first_warning


@pytest.mark.parametrize("late", ["broad", "error", "metrics"])
@pytest.mark.parametrize(("max_attempts", "ready"), [(2, False), (3, True)])
def test_tempo_is_searchable_only_once_every_probe_answers(
    monkeypatch: pytest.MonkeyPatch, late: str, max_attempts: int, ready: bool
) -> None:
    # Each probe answers from the first poll except `late`, which answers from the third.
    end_time = datetime(2026, 4, 19, 16, 0, tzinfo=UTC)
    polls_waited: list[None] = []

    def answers(probe: str) -> list[dict]:
        polls_needed = 2 if probe == late else 0
        return [{"traceID": "abc", "value": 0.5}] if len(polls_waited) >= polls_needed else []

    monkeypatch.setattr(
        generate_data,
        "fetch_tempo_search_traces",
        lambda query, _start, _end, *, limit=20: (answers("error" if query else "broad"), ""),
    )
    monkeypatch.setattr(
        generate_data, "fetch_tempo_metrics_series", lambda *_: (answers("metrics"), "")
    )
    monkeypatch.setattr(generate_data.time, "sleep", lambda _seconds: polls_waited.append(None))

    if ready:
        generate_data.wait_for_tempo_searchable(end_time, max_attempts=max_attempts)
    else:
        with pytest.raises(RuntimeError):
            generate_data.wait_for_tempo_searchable(end_time, max_attempts=max_attempts)


def test_wait_for_tempo_searchable_raises_when_search_stays_empty(
    monkeypatch: pytest.MonkeyPatch,
):
    end_time = datetime(2026, 4, 19, 16, 0, tzinfo=UTC)

    monkeypatch.setattr(
        generate_data,
        "fetch_tempo_search_traces",
        lambda _query, _start, _end, *, limit=20: ([], ""),
    )
    monkeypatch.setattr(
        generate_data, "fetch_tempo_metrics_series", lambda _query, _start, _end: ([], "")
    )
    monkeypatch.setattr(generate_data.time, "sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="Tempo search never became ready"):
        generate_data.wait_for_tempo_searchable(end_time, max_attempts=2)


def test_wait_for_tempo_searchable_waits_until_every_trace_is_indexed(
    monkeypatch: pytest.MonkeyPatch,
):
    # Tempo has no flush call: search answers from the first block while later ones are still
    # being cut, so a partial dataset must not count as ready.
    end_time = datetime(2026, 4, 19, 16, 0, tzinfo=UTC)
    indexed = iter([10, 40, 50])

    monkeypatch.setattr(
        generate_data,
        "fetch_tempo_search_traces",
        lambda _query, _start, _end, *, limit=20: ([{"traceID": "abc"}], ""),
    )
    monkeypatch.setattr(
        generate_data, "fetch_tempo_metrics_series", lambda _q, _start, _end: ([{"value": 1}], "")
    )
    monkeypatch.setattr(
        generate_data, "fetch_tempo_root_span_count", lambda _start, _end: (next(indexed), "")
    )
    monkeypatch.setattr(generate_data.time, "sleep", lambda _seconds: None)

    generate_data.wait_for_tempo_searchable(end_time, max_attempts=3, expected_traces=50)

    with pytest.raises(RuntimeError, match="never became ready"):
        monkeypatch.setattr(generate_data, "fetch_tempo_root_span_count", lambda _s, _e: (49, ""))
        generate_data.wait_for_tempo_searchable(end_time, max_attempts=2, expected_traces=50)


def test_wait_for_loki_complete_waits_for_every_line(monkeypatch: pytest.MonkeyPatch):
    end_time = datetime(2026, 4, 19, 16, 0, tzinfo=UTC)
    counts = iter([0, 40, 50])

    monkeypatch.setattr(generate_data.urllib.request, "urlopen", lambda *_a, **_kw: None)
    monkeypatch.setattr(generate_data, "fetch_loki_line_count", lambda _end: (next(counts), ""))
    monkeypatch.setattr(generate_data.time, "sleep", lambda _seconds: None)

    generate_data.wait_for_loki_complete(end_time, 50, max_attempts=3)

    monkeypatch.setattr(generate_data, "fetch_loki_line_count", lambda _end: (49, ""))
    with pytest.raises(RuntimeError, match="Loki never became ready"):
        generate_data.wait_for_loki_complete(end_time, 50, max_attempts=2)
