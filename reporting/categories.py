CATEGORY_LABELS: dict[str, str] = {
    "prometheus_query": "PromQL",
    "loki_query": "LogQL",
    "tempo_query": "TraceQL",
    "dashboarding": "Dashboarding",
    "grafana_api": "Grafana API",
    "investigation": "Investigation",
}


def category_label(category: str) -> str:
    return CATEGORY_LABELS.get(category, category.replace("_", " ").title())
