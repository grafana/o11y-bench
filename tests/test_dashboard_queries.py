from unittest.mock import patch

import pytest

from grading.dashboard_queries import (
    normalize_promql_macros,
    validate_dashboard_execute_case,
    validate_prometheus_vector_execute_case,
)
from grading.env_context import VerifierContext
from grading.models import ExecuteCase

CANONICAL = 'service_retry_queue_depth{job=~"payment-service|order-service"}'
RAW = {
    "payment-service": (
        {"__name__": "service_retry_queue_depth", "job": "payment-service", "instance": "p:8080"},
        37.0,
    ),
    "order-service": (
        {"__name__": "service_retry_queue_depth", "job": "order-service", "instance": "o:8080"},
        9.0,
    ),
}


def _check(panel_series: dict[str, list], canonical: list | None = None) -> tuple[bool, str]:
    """Run the panel's queries and the canonical against fake Prometheus results."""

    def fetch(url, query, timeout, time_unix=None):
        if query == CANONICAL:
            return (list(RAW.values()) if canonical is None else canonical), ""
        return panel_series[query], ""

    case = ExecuteCase(result_kind="prometheus_vector", canonical_query=CANONICAL)
    with patch("grading.dashboard_queries.fetch_prometheus_vector", side_effect=fetch):
        return validate_prometheus_vector_execute_case(
            list(panel_series),
            case,
            ctx=VerifierContext(),
            eval_end_unix=0.0,
            bindings={},
        )


@pytest.mark.parametrize(
    "panel_series",
    [
        # Haiku 5.5: one target per job, aggregated `by (job)`, which keeps the compared label.
        {
            'max by (job) (service_retry_queue_depth{job="payment-service"})': [
                ({"job": "payment-service"}, 37.0)
            ],
            'max by (job) (service_retry_queue_depth{job="order-service"})': [
                ({"job": "order-service"}, 9.0)
            ],
        },
        # Sonnet 5.5: one unlabelled target per service, named by legend.
        {
            'max(service_retry_queue_depth{job="order-service"})': [({}, 9.0)],
            'max(service_retry_queue_depth{job="payment-service"})': [({}, 37.0)],
        },
    ],
    ids=["labelled-by-job", "unlabelled-per-service"],
)
def test_a_panel_that_shows_the_canonical_series_matches(panel_series) -> None:
    ok, detail = _check(panel_series)
    assert ok, detail


@pytest.mark.parametrize(
    "panel_series",
    [
        # Summed into one series: the per-job comparison is gone.
        {"sum(service_retry_queue_depth)": [({}, 46.0)]},
        # Right series, wrong values.
        {
            'service_retry_queue_depth{job=~"payment-service|order-service"}[5m] offset 1h': [
                ({"job": "payment-service"}, 5.0),
                ({"job": "order-service"}, 9.0),
            ]
        },
        # Only one of the two jobs.
        {'service_retry_queue_depth{job="payment-service"}': [RAW["payment-service"]]},
        # Unlabelled targets, one of them showing the wrong service's value.
        {
            'max(service_retry_queue_depth{job="cache-service"})': [({}, 3.0)],
            'max(service_retry_queue_depth{job="payment-service"})': [({}, 37.0)],
        },
        # Labelled by something else: the values alone do not make a match.
        {
            "sum by (pod) (service_retry_queue_depth)": [
                ({"pod": "a"}, 37.0),
                ({"pod": "b"}, 9.0),
            ]
        },
    ],
    ids=["collapsed", "wrong-values", "missing-job", "unlabelled-wrong-values", "other-label"],
)
def test_a_panel_that_shows_different_series_does_not_match(panel_series) -> None:
    ok, detail = _check(panel_series)
    assert not ok
    assert detail == "vector result does not match canonical series"


def test_a_panel_is_read_back_at_the_offset_where_the_services_differ() -> None:
    # Backlogs are 0 at the end of the window, so only an earlier moment shows them.
    end = 10_000.0

    def fetch(url, query, timeout, time_unix=None):
        return [({"job": "payment-service"}, 0.0 if time_unix == end else 37.0)], ""

    verdicts = []
    for offset_sec in (None, 600):
        case = ExecuteCase(result_kind="prometheus_vector", min_value=1, eval_offset_sec=offset_sec)
        with patch("grading.dashboard_queries.fetch_prometheus_vector", side_effect=fetch):
            ok, _ = validate_dashboard_execute_case(
                {"datasource": {"type": "prometheus"}},
                case,
                item_kind="Panel",
                query_texts=["max by (job) (x)"],
                ctx=VerifierContext(),
                dashboard_variables=[],
                eval_end_unix=end,
            )
        verdicts.append(ok)
    assert verdicts == [False, True]


@pytest.mark.parametrize("macro", ["$__rate_interval", "${__rate_interval}"])
def test_the_rate_interval_covers_the_scrape_interval_of_the_data(macro: str) -> None:
    # Samples are 30s apart and the last is 30s before the clock, so a 1m window holds one.
    assert "[2m]" in normalize_promql_macros(f"rate(x[{macro}])")


def test_a_quantile_with_no_requests_to_rank_matches_the_canonical() -> None:
    # Low traffic: no user-service requests in the last 5m, so both quantiles are NaN.
    nan = float("nan")
    ok, detail = _check({"histogram_quantile(0.95, x)": [({}, nan)]}, canonical=[({}, nan)])
    assert ok, detail
    ok, _ = _check({"histogram_quantile(0.95, x)": [({}, 0.25)]}, canonical=[({}, nan)])
    assert not ok
