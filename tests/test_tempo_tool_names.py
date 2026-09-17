import pytest

from grading.helpers import tempo_tool_matches_name

# mcp-grafana has exposed Tempo under two naming conventions: the datasource-proxy
# tools were prefixed, the inlined native tools infix the datasource name. Grading
# must recognise both, or a transcript from one of them harvests zero trace IDs
# and a correct answer scores 0.
PROXIED = [
    "tempo_traceql-search",
    "tempo_get-trace",
    "tempo_get-attribute-names",
    "tempo_get-attribute-values",
    "tempo_traceql-metrics-instant",
    "tempo_traceql-metrics-range",
    "tempo_docs-traceql",
]
NATIVE = [
    "search_tempo_traces",
    "get_tempo_trace",
    "query_tempo_metrics",
    "list_tempo_attribute_names",
    "list_tempo_attribute_values",
    "diff_tempo_traces",
    "get_tempo_traceql_docs",
]
NON_TEMPO = [
    "query_prometheus",
    "query_loki_logs",
    "search_dashboards",
    "get_datasource",
    "grafana_api_request",
    "list_prometheus_metric_names",
    "get_panel_image",
]


@pytest.mark.parametrize("name", PROXIED + NATIVE)
def test_matches_both_naming_conventions(name: str) -> None:
    assert tempo_tool_matches_name(name, None, "tempo_") is True


@pytest.mark.parametrize("name", NON_TEMPO)
def test_does_not_match_unrelated_tools(name: str) -> None:
    assert tempo_tool_matches_name(name, None, "tempo_") is False


def test_explicit_allowlist_stays_exact() -> None:
    # An allowlist in a task spec is a deliberate narrowing and must not be
    # widened by the convention matching above.
    assert tempo_tool_matches_name("search_tempo_traces", {"get_tempo_trace"}, "tempo_") is False
    assert tempo_tool_matches_name("get_tempo_trace", {"get_tempo_trace"}, "tempo_") is True
