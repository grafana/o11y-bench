from unittest.mock import patch

from grading.env_context import VerifierContext
from grading.facts import resolve_fact
from grading.judge import build_evaluation_prompt, build_judge_criteria
from grading.models import JudgeCriterion, Problem


def test_build_evaluation_prompt_renders_clean_criteria_only() -> None:
    prompt_text = (
        "The final response is accurate.\n"
        'Source of truth: The canonical query "sum(rate(x[5m]))" returned 0.0556.'
    )
    prompt = build_evaluation_prompt(
        [
            JudgeCriterion(
                criterion="The final response is accurate.",
                prompt_text=prompt_text,
                weight=1.0,
            )
        ]
    )

    assert "{criteria_text}" not in prompt
    assert 'Source of truth: The canonical query "sum(rate(x[5m]))" returned 0.0556.' in prompt


def test_build_judge_criteria_inlines_fact_summary() -> None:
    problem = Problem(
        id="promql-highest-backend-error-ratio",
        category="prometheus_query",
        statement="x",
        rubric=[
            {
                "criterion": (
                    "The final response identifies the backend with the highest 5xx share."
                ),
                "weight": 1.0,
                "fact": {
                    "kind": "query",
                    "backend": "prometheus",
                    "query": "ignored",
                },
            }
        ],
    )

    with patch(
        "grading.facts.fetch_prometheus_query_result",
        return_value=(
            {
                "shape": "items",
                "items": [
                    {"labels": {"job": "payment-service"}, "value": 0.0555556},
                    {"labels": {"job": "order-service"}, "value": 0.0401786},
                ],
            },
            "",
        ),
    ):
        criteria = build_judge_criteria(problem, VerifierContext(prometheus_url="http://prom:9090"))

    expected_prompt_text = (
        "The final response identifies the backend with the highest 5xx share.\n"
        "Source of truth: The top canonical result was "
        "job=payment-service with value 0.055556. "
        "Other canonical results were job=order-service with value 0.040179."
    )
    assert criteria == [
        JudgeCriterion(
            criterion="The final response identifies the backend with the highest 5xx share.",
            prompt_text=expected_prompt_text,
            weight=1.0,
        )
    ]


def test_build_judge_criteria_sorts_item_facts_by_value_descending() -> None:
    problem = Problem(
        id="promql-highest-backend-error-ratio",
        category="prometheus_query",
        statement="x",
        rubric=[
            {
                "criterion": (
                    "The final response identifies the backend with the highest 5xx share."
                ),
                "weight": 1.0,
                "fact": {
                    "kind": "query",
                    "backend": "prometheus",
                    "query": "ignored",
                },
            }
        ],
    )

    with patch(
        "grading.facts.fetch_prometheus_query_result",
        return_value=(
            {
                "shape": "items",
                "items": [
                    {"labels": {"job": "order-service"}, "value": 0.0382775},
                    {"labels": {"job": "payment-service"}, "value": 0.0694444},
                    {"labels": {"job": "user-service"}, "value": 0.0128205},
                ],
            },
            "",
        ),
    ):
        criteria = build_judge_criteria(problem, VerifierContext(prometheus_url="http://prom:9090"))

    assert (
        "The top canonical result was job=payment-service with value 0.069444."
        in criteria[0].prompt_text
    )


def test_build_judge_criteria_uses_approximate_scalar_summary_for_rough_criteria() -> None:
    problem = Problem(
        id="promql-cache-lag-vs-user-latency",
        category="prometheus_query",
        statement="x",
        rubric=[
            {
                "criterion": (
                    "The final response states the peak user-service p95 latency "
                    "roughly accurately."
                ),
                "weight": 1.0,
                "fact": {
                    "kind": "query",
                    "backend": "prometheus",
                    "query": "ignored",
                },
            }
        ],
    )

    with patch(
        "grading.facts.fetch_prometheus_query_result",
        return_value=({"shape": "scalar", "value": 9.75}, ""),
    ):
        criteria = build_judge_criteria(problem, VerifierContext(prometheus_url="http://prom:9090"))

    assert 'The canonical query "ignored" returned 9.75;' in criteria[0].prompt_text
    assert "(9.2625 to 10.2375)" in criteria[0].prompt_text


def test_resolve_fact_caches_identical_queries() -> None:
    cache: dict[str, object] = {}
    problem = Problem(
        id="x",
        category="x",
        statement="x",
        rubric=[
            {
                "criterion": "x",
                "weight": 1.0,
                "fact": {"kind": "query", "backend": "prometheus", "query": "ignored"},
            }
        ],
    )
    spec = problem.rubric[0].fact
    ctx = VerifierContext(prometheus_url="http://prom:9090")

    with patch(
        "grading.facts.fetch_prometheus_query_result",
        return_value=({"shape": "scalar", "value": 42.0}, ""),
    ) as mocked_fetch:
        assert spec is not None
        first = resolve_fact(spec, ctx, cache)
        second = resolve_fact(spec, ctx, cache)

    assert first.summary == 'The canonical query "ignored" returned 42.'
    assert second.summary == 'The canonical query "ignored" returned 42.'
    assert mocked_fetch.call_count == 1


def test_resolve_dashboard_fact_renders_panels_variables_and_queries() -> None:
    problem = Problem(
        id="search-dashboards",
        category="grafana_api",
        statement="x",
        rubric=[
            {
                "criterion": (
                    "The final response accurately reports the saved panel count and panel types."
                ),
                "weight": 1.0,
                "fact": {
                    "kind": "resource",
                    "resource": "dashboard",
                    "uid": "svc-overview",
                },
            }
        ],
    )

    dashboard = {
        "uid": "svc-overview",
        "panels": [
            {
                "title": "Request Rate",
                "type": "timeseries",
                "datasource": {"type": "prometheus"},
                "targets": [{"expr": "rate(http_requests_total[5m])"}],
            },
            {
                "title": "Error Rate",
                "type": "stat",
                "datasource": {"type": "prometheus"},
                "targets": [{"expr": 'sum(rate(http_requests_total{status=~"5.."}[5m]))'}],
            },
        ],
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
    }
    with patch(
        "grading.dashboard_snapshot.fetch_grafana_dashboard_model",
        return_value=(dashboard, ""),
    ):
        criteria = build_judge_criteria(problem, VerifierContext(grafana_url="http://grafana"))

    prompt_text = criteria[0].prompt_text
    assert "Source of truth:" in prompt_text
    assert "The saved dashboard svc-overview has 2 panels." in prompt_text
    assert "Request Rate (timeseries, prometheus, query" in prompt_text
    assert "Variables: service (query, include all, multi, query" in prompt_text


def test_resolve_datasource_detail_fact_prefers_name_when_name_and_type_are_both_set() -> None:
    problem = Problem(
        id="get-datasource-details",
        category="grafana_api",
        statement="x",
        rubric=[
            {
                "criterion": (
                    "The final response reports the selected datasource detail accurately."
                ),
                "weight": 1.0,
                "fact": {
                    "kind": "resource",
                    "resource": "datasource_detail",
                    "name": "Prometheus",
                    "type": "prometheus",
                },
            }
        ],
    )

    with patch(
        "grading.facts.fetch_grafana_datasources_checked",
        return_value=(
            [
                {
                    "name": "Mimir",
                    "type": "prometheus",
                    "url": "http://mimir:9009",
                    "access": "proxy",
                },
                {
                    "name": "Prometheus",
                    "type": "prometheus",
                    "url": "http://prom:9090",
                    "access": "proxy",
                },
            ],
            None,
        ),
    ):
        criteria = build_judge_criteria(problem, VerifierContext(grafana_url="http://grafana"))

    prompt_text = criteria[0].prompt_text
    assert "Grafana datasource Prometheus is type prometheus." in prompt_text
    assert "URL: http://prom:9090." in prompt_text
