from unittest.mock import patch

import pytest

from grading import env_context
from grading.checks import run_checks
from grading.env_context import VerifierContext
from grading.helpers import MAX_CITED_TRACES
from grading.models import CheckItem, Message, Transcript

from .helpers import answer_transcript, fake_tempo


def test_load_verifier_context_defaults_to_localhost(monkeypatch) -> None:
    monkeypatch.delenv("GRAFANA_URL", raising=False)
    monkeypatch.delenv("PROMETHEUS_URL", raising=False)
    monkeypatch.delenv("LOKI_URL", raising=False)
    monkeypatch.delenv("TEMPO_URL", raising=False)

    ctx = env_context.load_verifier_context_from_env()

    assert ctx.grafana_url == "http://127.0.0.1:3000"
    assert ctx.prometheus_url == "http://127.0.0.1:9090"
    assert ctx.loki_url == "http://127.0.0.1:3100"
    assert ctx.tempo_url == "http://127.0.0.1:3200"


def test_fetch_loki_query_result_uses_range_query_with_nanosecond_eval_time(
    monkeypatch,
) -> None:
    monkeypatch.setenv("O11Y_SCENARIO_TIME_ISO", "2026-04-04T10:05:14Z")
    captured: dict[str, str] = {}

    def fake_http_get_json(url: str, timeout_sec: float) -> dict[str, object]:
        captured["url"] = url
        return {
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [{"metric": {}, "values": [[1742486400, "91"]]}],
            },
        }

    with patch("grading.env_context.http_get_json", side_effect=fake_http_get_json):
        result, err = env_context.fetch_loki_query_result(
            "http://loki",
            'sum(count_over_time({service=~".+"}[24h]))',
            15.0,
        )

    assert err == ""
    assert result == {"shape": "scalar", "value": 91.0}
    assert "/loki/api/v1/query_range?" in captured["url"]
    expected_ns = int(env_context.synthetic_eval_time_unix() * 1_000_000_000)
    assert f"start={expected_ns - 1_000_000_000}" in captured["url"]
    assert f"end={expected_ns}" in captured["url"]
    assert "step=1" in captured["url"]


def test_fetch_tempo_attribute_values_normalizes_v2_response_objects() -> None:
    def fake_http_get_json(url: str, timeout_sec: float) -> dict[str, object]:
        assert "/api/v2/search/tag/resource.service.name/values?" in url
        return {
            "tagValues": [
                {"type": "string", "value": "api-gateway"},
                {"type": "string", "value": "order-service"},
            ]
        }

    with patch("grading.env_context.http_get_json", side_effect=fake_http_get_json):
        values, err = env_context.fetch_tempo_attribute_values(
            "http://tempo",
            "resource.service.name",
            15.0,
            start_sec=1,
            end_sec=2,
        )

    assert err == ""
    assert values == ["api-gateway", "order-service"]


TRACE_ID = "0595f4e097fcacc5724644e76d76ad60"
FAILING_ORDER = '{ span.http.route = "/api/orders" && status = error }'


@pytest.mark.parametrize(
    ("answer", "stored", "matching", "score", "why"),
    [
        (f"Failing trace {TRACE_ID}: order-service 500.", {TRACE_ID}, {TRACE_ID}, 1.0, "matches"),
        (f"Trace {TRACE_ID} looks fine.", {TRACE_ID}, set(), 0.0, "does not match"),
        (f"Trace {TRACE_ID} is the one.", set(), set(), 0.0, "not in Tempo"),
        (f"Trace {TRACE_ID[:16]} is the one.", {TRACE_ID}, {TRACE_ID}, 0.0, "no full trace ID"),
        # As Tempo prints it, leading zero dropped.
        (f"Trace {TRACE_ID[1:]} failed.", {TRACE_ID}, {TRACE_ID}, 1.0, "matches"),
    ],
)
def test_cited_trace_is_read_back_from_tempo_and_must_match(
    answer: str, stored: set[str], matching: set[str], score: float, why: str
) -> None:
    check = CheckItem.model_validate(
        {
            "name": "cited trace",
            "weight": 1.0,
            "type": "state",
            "params": {"mode": "cited_trace", "match": FAILING_ORDER},
        }
    )
    tempo = fake_tempo(stored, matching)
    with patch("grading.env_context.http_get_json", side_effect=tempo):
        scores, explanations = run_checks(
            [check], answer_transcript(answer), VerifierContext(tempo_url="http://tempo")
        )

    assert scores["cited trace"] == score
    assert why in explanations["cited trace"]


def test_state_datasource_detail_requires_configured_detail() -> None:
    with patch(
        "grading.checks.fetch_grafana_datasources_checked",
        return_value=(
            [
                {
                    "name": "Prometheus",
                    "type": "prometheus",
                    "url": "http://prom:9090",
                    "access": "proxy",
                }
            ],
            None,
        ),
    ):
        scores, explanations = run_checks(
            [
                CheckItem.model_validate(
                    {
                        "name": "datasource detail",
                        "weight": 1.0,
                        "type": "state",
                        "params": {
                            "mode": "datasource_detail",
                            "name": "Prometheus",
                            "require_url": True,
                            "require_access": True,
                        },
                    }
                )
            ],
            Transcript(messages=[]),
            VerifierContext(grafana_url="http://grafana"),
        )

    assert scores["datasource detail"] == 1.0
    assert "matches the expected state" in explanations["datasource detail"]


def test_state_tempo_trace_service_inventory_uses_attribute_values() -> None:
    with patch(
        "grading.checks.fetch_tempo_attribute_values",
        return_value=(
            [
                "api-gateway",
                "order-service",
                "payment-service",
                "user-service",
                "webapp",
            ],
            "",
        ),
    ):
        scores, explanations = run_checks(
            [
                CheckItem.model_validate(
                    {
                        "name": "tempo inventory",
                        "weight": 1.0,
                        "type": "state",
                        "params": {
                            "mode": "tempo_trace_service_inventory",
                            "lookback_hours": 24,
                            "count": 5,
                            "services_all": [
                                "webapp",
                                "api-gateway",
                                "order-service",
                                "payment-service",
                                "user-service",
                            ],
                        },
                    }
                )
            ],
            Transcript(messages=[]),
            VerifierContext(tempo_url="http://tempo"),
        )

    assert scores["tempo inventory"] == 1.0
    assert "includes 5 services" in explanations["tempo inventory"]


def test_state_dashboard_state_validates_saved_structure() -> None:
    dash = {
        "uid": "svc-overview",
        "templating": {
            "list": [
                {
                    "name": "service",
                    "type": "query",
                    "includeAll": True,
                    "multi": True,
                    "query": {"query": "label_values(http_requests_total, job)"},
                }
            ]
        },
        "annotations": {"list": []},
        "panels": [
            {
                "title": "Request Rate",
                "type": "timeseries",
                "datasource": {"type": "prometheus"},
                "targets": [
                    {"expr": 'sum(rate(http_requests_total{job=~"${service:regex}"}[5m]))'}
                ],
            },
            {
                "title": "Recent Errors",
                "type": "logs",
                "datasource": {"type": "loki"},
                "targets": [{"expr": '{job=~"${service:regex}"} | json | status >= 500'}],
            },
        ],
    }
    with patch(
        "grading.dashboard_snapshot.fetch_grafana_dashboard_model",
        return_value=(dash, ""),
    ):
        scores, explanations = run_checks(
            [
                CheckItem.model_validate(
                    {
                        "name": "dashboard state",
                        "weight": 1.0,
                        "type": "state",
                        "params": {
                            "mode": "dashboard_state",
                            "uid": "svc-overview",
                            "variable_count": 1,
                            "variables": [{"name": "service", "include_all": True, "multi": True}],
                            "panels": [
                                {
                                    "title": "Request Rate",
                                    "type": "timeseries",
                                    "datasource_type": "prometheus",
                                },
                                {
                                    "title": "Recent Errors",
                                    "type": "logs",
                                    "datasource_type": "loki",
                                },
                            ],
                        },
                    }
                )
            ],
            Transcript(messages=[]),
            VerifierContext(grafana_url="http://grafana"),
        )

    assert scores["dashboard state"] == 1.0
    assert "satisfies" in explanations["dashboard state"].lower()


def test_state_dashboard_execute_case_supports_all_binding() -> None:
    ctx = VerifierContext(grafana_url="http://grafana", prometheus_url="http://prometheus")
    dash = {
        "uid": "service-overview",
        "templating": {
            "list": [
                {
                    "name": "service",
                    "type": "query",
                    "includeAll": True,
                    "multi": True,
                    "query": {"query": "label_values(http_requests_total, job)"},
                }
            ]
        },
        "annotations": {"list": []},
        "panels": [
            {
                "title": "Request Rate",
                "type": "timeseries",
                "datasource": {"type": "prometheus"},
                "targets": [
                    {"expr": 'sum(rate(http_requests_total{job=~"${service:regex}"}[1m])) by (job)'}
                ],
            }
        ],
    }

    with (
        patch("grading.dashboard_snapshot.fetch_grafana_dashboard_model", return_value=(dash, "")),
        patch(
            "grading.dashboard_queries.fetch_prometheus_label_values",
            return_value=(["user-service", "order-service", "payment-service"], ""),
        ),
        patch(
            "grading.dashboard_queries.fetch_prometheus_vector",
            return_value=(
                [
                    ({"job": "user-service"}, 1.0),
                    ({"job": "order-service"}, 2.0),
                    ({"job": "payment-service"}, 3.0),
                ],
                "",
            ),
        ),
    ):
        scores, explanations = run_checks(
            [
                CheckItem.model_validate(
                    {
                        "name": "dashboard state",
                        "weight": 1.0,
                        "type": "state",
                        "params": {
                            "mode": "dashboard_state",
                            "uid": "service-overview",
                            "panels": [
                                {
                                    "title": "Request Rate",
                                    "execute_cases": [
                                        {
                                            "result_kind": "prometheus_vector",
                                            "bindings": {"service": "__all__"},
                                            "series_count_min": 3,
                                            "distinct_label_values": [
                                                {
                                                    "name": "job",
                                                    "values_exact": [
                                                        "user-service",
                                                        "order-service",
                                                        "payment-service",
                                                    ],
                                                }
                                            ],
                                        }
                                    ],
                                }
                            ],
                        },
                    }
                )
            ],
            Transcript(
                messages=[Message(role="user", content="Current time: 2025-03-20T16:00:00Z")]
            ),
            ctx,
        )

    assert scores["dashboard state"] == 1.0
    assert "satisfies" in explanations["dashboard state"].lower()


def test_a_trace_cited_past_the_limit_is_not_the_example() -> None:
    ids = [f"{i:032x}" for i in range(1, MAX_CITED_TRACES + 2)]
    check = CheckItem.model_validate(
        {
            "name": "cited trace",
            "weight": 1.0,
            "type": "state",
            "params": {"mode": "cited_trace", "match": FAILING_ORDER},
        }
    )
    # Only the last ID cited is a failing trace; the example comes from the first few IDs.
    tempo = fake_tempo(set(ids), {ids[-1]})
    with patch("grading.env_context.http_get_json", side_effect=tempo):
        scores, _ = run_checks(
            [check], answer_transcript("Traces " + ", ".join(ids)), VerifierContext(tempo_url="t")
        )

    assert scores["cited trace"] == 0.0
