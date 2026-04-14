from grading.dashboard_state import validate_dashboard_state
from grading.env_context import (
    VerifierContext,
    default_tempo_search_window_sec,
    fetch_grafana_datasources_checked,
    fetch_tempo_attribute_values,
    resolve_grafana_datasource,
)
from grading.helpers import (
    additional_trace_id_tool_names,
    as_name_set,
    assistant_scope_note,
    assistant_text_blobs,
    require_stack_url,
    response_cites_trace_id_prefix,
    tempo_tool_matches_name,
    tool_call_id_to_name,
    trace_ids_from_tool_content,
)
from grading.models import (
    CheckItem,
    DashboardStateParams,
    DatasourceDetailStateParams,
    DatasourceInventoryStateParams,
    TempoTraceServiceInventoryStateParams,
    ToolTraceIdGroundingParams,
    Transcript,
)


def run_checks(
    checks: list[CheckItem],
    transcript: Transcript,
    ctx: VerifierContext | None = None,
) -> tuple[dict[str, float], dict[str, str]]:
    subscores: dict[str, float] = {}
    explanations: dict[str, str] = {}

    for check in checks:
        score, explanation = run_check(check, transcript, ctx)
        subscores[check.name] = score
        explanations[check.name] = explanation

    return subscores, explanations


def run_check(
    check: CheckItem,
    transcript: Transcript,
    ctx: VerifierContext | None,
) -> tuple[float, str]:
    match check.params:
        case ToolTraceIdGroundingParams():
            return validate_tool_trace_id_grounding(check.params, transcript)
        case DashboardStateParams():
            return validate_dashboard_state(check.params, transcript, ctx)
        case DatasourceInventoryStateParams():
            return validate_datasource_inventory(check.params, ctx)
        case DatasourceDetailStateParams():
            return validate_datasource_detail(check.params, ctx)
        case TempoTraceServiceInventoryStateParams():
            return validate_tempo_trace_service_inventory(check.params, ctx)
        case _:
            return 0.0, f"Unsupported check params mode {check.params.mode!r}."


def validate_tool_trace_id_grounding(
    params: ToolTraceIdGroundingParams,
    transcript: Transcript,
) -> tuple[float, str]:
    allowed_names = as_name_set(params.tool_name) if params.tool_name else None
    extra_tool_names = additional_trace_id_tool_names(
        {"additional_tool_names": params.additional_tool_names}
    )

    call_id_to_name = tool_call_id_to_name(transcript)
    found_ids: set[str] = set()
    for msg in transcript.messages:
        if msg.role != "tool" or not msg.tool_results:
            continue
        for tool_result in msg.tool_results:
            name = call_id_to_name.get(tool_result.tool_call_id, "")
            if name not in extra_tool_names and not tempo_tool_matches_name(
                name,
                allowed_names if allowed_names else None,
                params.tool_name_prefix,
            ):
                continue
            found_ids.update(trace_ids_from_tool_content(tool_result.content))

    if not found_ids:
        scope = (
            f" ({sorted(allowed_names)})"
            if allowed_names
            else f" (prefix {params.tool_name_prefix!r})"
        )
        if extra_tool_names:
            scope += f" + {sorted(extra_tool_names)}"
        return 0.0, f"No hex trace IDs found in tool results{scope}."

    text_blobs = assistant_text_blobs(transcript, params.assistant_scope)
    if not text_blobs:
        return 0.0, "No agent response found."

    matched, prefix = response_cites_trace_id_prefix(
        text_blobs,
        found_ids,
        params.prefix_min_chars,
    )
    if matched and prefix:
        scope_note = assistant_scope_note(params.assistant_scope)
        return (
            1.0,
            f"Response cites trace ID prefix grounded in tool results ({prefix}, {scope_note}).",
        )
    return 0.0, "Response does not cite any trace ID prefix that appeared in tool results."


def validate_datasource_inventory(
    params: DatasourceInventoryStateParams,
    ctx: VerifierContext | None,
) -> tuple[float, str]:
    datasources, err = fetch_grafana_datasources_checked(ctx)
    if err:
        return 0.0, err
    assert datasources is not None

    have_types = {str(item.get("type", "")).lower() for item in datasources}
    missing_types = [value for value in params.types if value.lower() not in have_types]
    if missing_types:
        return 0.0, f"Grafana is missing datasource types {missing_types}."

    have_names = {str(item.get("name", "")) for item in datasources}
    missing_names = [value for value in params.names if value not in have_names]
    if missing_names:
        return 0.0, f"Grafana is missing datasource names {missing_names}."

    return 1.0, f"Grafana has the expected datasource inventory ({len(datasources)} datasources)."


def validate_datasource_detail(
    params: DatasourceDetailStateParams,
    ctx: VerifierContext | None,
) -> tuple[float, str]:
    datasources, err = fetch_grafana_datasources_checked(ctx)
    if err:
        return 0.0, err
    assert datasources is not None

    datasource, err = resolve_grafana_datasource(
        datasources,
        name=params.name,
        datasource_type=params.type,
    )
    if err:
        return 0.0, err
    assert datasource is not None

    if params.type and str(datasource.get("type", "")).lower() != params.type.lower():
        return 0.0, f"Datasource has type {datasource.get('type')!r}, expected {params.type!r}."
    if params.require_url and not str(datasource.get("url", "")).strip():
        return 0.0, "Datasource does not have a URL configured."
    if params.require_access and not str(datasource.get("access", "")).strip():
        return 0.0, "Datasource does not have an access mode configured."
    return 1.0, "Datasource detail exists and matches the expected state."


def validate_tempo_trace_service_inventory(
    params: TempoTraceServiceInventoryStateParams,
    ctx: VerifierContext | None,
) -> tuple[float, str]:
    if ctx is None:
        return 0.0, "TEMPO_URL is not set."
    missing = require_stack_url(ctx.tempo_url, "TEMPO_URL")
    if missing:
        return 0.0, missing

    start_sec, end_sec = default_tempo_search_window_sec(params.lookback_hours)
    services, err = fetch_tempo_attribute_values(
        ctx.tempo_url,
        "resource.service.name",
        ctx.timeout_sec,
        query="",
        start_sec=start_sec,
        end_sec=end_sec,
    )
    if err:
        return 0.0, err
    discovered = set(services)
    if not discovered:
        return 0.0, "Tempo attribute values returned no services."

    missing_services = sorted(set(params.services_all) - discovered)
    if missing_services:
        return 0.0, f"Tempo trace coverage missing expected services {missing_services}."
    if params.count is not None and len(discovered) != params.count:
        return (
            0.0,
            f"Tempo trace coverage found {len(discovered)} services, expected {params.count}.",
        )
    return 1.0, f"Tempo trace coverage includes {len(discovered)} services: {sorted(discovered)}."
