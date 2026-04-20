import json
from datetime import UTC, datetime, timedelta

import pytest
from o11y_stack import generate_data


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


def test_data_end_time_honors_explicit_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("O11Y_SCENARIO_TIME_ISO", "2026-04-04T10:05:14Z")

    assert generate_data.data_end_time_utc() == datetime(2026, 4, 4, 10, 5, 14, tzinfo=UTC)


def test_wait_for_tempo_searchable_requires_trace_by_id_and_search(monkeypatch: pytest.MonkeyPatch):
    end_time = datetime(2026, 4, 19, 16, 0, tzinfo=UTC)
    attempts = {"search": 0, "errors": 0}

    def fake_search(
        query: str, start_sec: int, end_sec: int, *, limit: int = 20
    ) -> tuple[list[dict], str]:
        del start_sec, end_sec, limit
        if not query:
            attempts["search"] += 1
            return ([{"traceID": "abc"}], "") if attempts["search"] >= 2 else ([], "")
        attempts["errors"] += 1
        return ([{"traceID": "def"}], "") if attempts["errors"] >= 2 else ([], "")

    monkeypatch.setattr(generate_data, "fetch_tempo_search_traces", fake_search)
    monkeypatch.setattr(generate_data.time, "sleep", lambda _seconds: None)

    generate_data.wait_for_tempo_searchable(end_time, max_attempts=3)


def test_wait_for_tempo_searchable_raises_when_search_stays_empty(
    monkeypatch: pytest.MonkeyPatch,
):
    end_time = datetime(2026, 4, 19, 16, 0, tzinfo=UTC)

    monkeypatch.setattr(
        generate_data,
        "fetch_tempo_search_traces",
        lambda _query, _start, _end, *, limit=20: ([], ""),
    )
    monkeypatch.setattr(generate_data.time, "sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="Tempo search never became ready"):
        generate_data.wait_for_tempo_searchable(end_time, max_attempts=2)
