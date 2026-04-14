from collections.abc import Callable
from typing import Any

from grading.dashboard_queries import validate_dashboard_execute_case, validate_query_semantics
from grading.dashboard_snapshot import (
    DashboardSnapshot,
    collect_annotation_query_texts,
    collect_named_values,
    collect_panel_query_texts,
    collect_variable_query_texts,
    dashboard_datasource_type,
    find_named_dashboard_item,
    line_mentions_all,
    load_dashboard_snapshot,
    normalize_query_text,
    normalize_token,
    parse_tag_keys,
    select_panels_for_titles,
    split_response_lines,
    strip_markdown_code_fences,
)
from grading.env_context import VerifierContext
from grading.helpers import prometheus_eval_time_unix
from grading.models import DashboardItemExpectation, DashboardStateParams, Transcript


def validate_dashboard_state(
    params: DashboardStateParams,
    transcript: Transcript,
    ctx: VerifierContext | None,
) -> tuple[float, str]:
    snapshot, err = load_dashboard_snapshot(params, ctx)
    if err:
        return 0.0, err
    assert snapshot is not None

    if params.panel_count is not None and len(snapshot.panels) != params.panel_count:
        return 0.0, f"Dashboard has {len(snapshot.panels)} panels, expected {params.panel_count}."
    if params.variable_count is not None and len(snapshot.variables) != params.variable_count:
        return (
            0.0,
            f"Dashboard has {len(snapshot.variables)} variables, expected {params.variable_count}.",
        )
    if params.annotation_count is not None and len(snapshot.annotations) != params.annotation_count:
        return (
            0.0,
            f"Dashboard has {len(snapshot.annotations)} annotations, expected {params.annotation_count}.",
        )

    for expectations, items, name_field, item_kind, query_texts_getter in (
        (params.variables, snapshot.variables, "name", "Variable", collect_variable_query_texts),
        (params.panels, snapshot.panels, "title", "Panel", collect_panel_query_texts),
        (
            params.annotations,
            snapshot.annotations,
            "name",
            "Annotation",
            collect_annotation_query_texts,
        ),
    ):
        for expectation in expectations:
            ok, explanation = validate_named_dashboard_item_expectation(
                items,
                expectation,
                name_field=name_field,
                item_kind=item_kind,
                query_texts_getter=query_texts_getter,
                transcript=transcript,
                ctx=ctx,
                dashboard_variables=snapshot.variables,
            )
            if not ok:
                return 0.0, explanation

    return 1.0, (
        f"Dashboard uid={snapshot.uid} satisfies panel/variable/annotation state checks "
        f"({len(params.panels)} panels, {len(params.variables)} variables, {len(params.annotations)} annotations)."
    )


def validate_named_dashboard_item_expectation(
    items: list[dict[str, Any]],
    expectation: DashboardItemExpectation,
    *,
    name_field: str,
    item_kind: str,
    query_texts_getter: Callable[[dict[str, Any]], list[str]],
    transcript: Transcript,
    ctx: VerifierContext | None,
    dashboard_variables: list[dict[str, Any]],
) -> tuple[bool, str]:
    item_name = expectation.title if name_field == "title" else expectation.name
    if item_name:
        item = find_named_dashboard_item(items, item_name=item_name, field=name_field)
        if item is None:
            available = collect_named_values(items, name_field)
            return False, (
                f"Dashboard missing {item_kind.lower()} {name_field} {item_name!r} ({name_field}s: {available})."
            )
        label = f"{item_kind} {item_name!r}"
    else:
        required_matches = max(1, expectation.match_count or 1)
        matches = [
            item
            for item in items
            if validate_dashboard_item_common(
                item,
                expectation,
                item_kind=item_kind,
                query_texts=query_texts_getter(item),
                transcript=transcript,
                ctx=ctx,
                dashboard_variables=dashboard_variables,
            )[0]
        ]
        if len(matches) < required_matches:
            return (
                False,
                f"Dashboard has {len(matches)} {item_kind.lower()} matches; expected at least {required_matches}.",
            )
        item = matches[0]
        label = f"{item_kind} matching structural expectation"

    return validate_dashboard_item_common(
        item,
        expectation,
        item_kind=label,
        query_texts=query_texts_getter(item),
        transcript=transcript,
        ctx=ctx,
        dashboard_variables=dashboard_variables,
    )


def validate_dashboard_item_common(
    item: dict[str, Any],
    expectation: DashboardItemExpectation,
    *,
    item_kind: str,
    query_texts: list[str],
    transcript: Transcript,
    ctx: VerifierContext | None,
    dashboard_variables: list[dict[str, Any]],
) -> tuple[bool, str]:
    if expectation.type and normalize_token(str(item.get("type", ""))) != normalize_token(
        expectation.type
    ):
        return False, f"{item_kind} has type {item.get('type')!r}, expected {expectation.type!r}."

    if expectation.type_any_of:
        actual_type = str(item.get("type", "")).strip()
        if not any(
            normalize_token(actual_type) == normalize_token(allowed)
            for allowed in expectation.type_any_of
        ):
            return (
                False,
                f"{item_kind} has type {actual_type!r}, expected one of {expectation.type_any_of!r}.",
            )

    if expectation.datasource_type:
        actual_ds_type = dashboard_datasource_type(item)
        if normalize_token(actual_ds_type) != normalize_token(expectation.datasource_type):
            return (
                False,
                f"{item_kind} has datasource type {actual_ds_type!r}, expected {expectation.datasource_type!r}.",
            )

    if expectation.enabled is not None:
        actual_enabled = bool(item.get("enable", item.get("enabled", False)))
        if actual_enabled != expectation.enabled:
            return (
                False,
                f"{item_kind} enabled={actual_enabled!r}, expected {expectation.enabled!r}.",
            )

    if expectation.include_all is not None:
        actual_include_all = bool(item.get("includeAll", False))
        if actual_include_all != expectation.include_all:
            return False, (
                f"{item_kind} includeAll={actual_include_all!r}, expected {expectation.include_all!r}."
            )

    if expectation.multi is not None:
        actual_multi = bool(item.get("multi", False))
        if actual_multi != expectation.multi:
            return False, f"{item_kind} multi={actual_multi!r}, expected {expectation.multi!r}."

    if expectation.tag_keys_all:
        actual_tag_keys = set(parse_tag_keys(item.get("tagKeys")))
        missing = [
            tag for tag in expectation.tag_keys_all if normalize_token(tag) not in actual_tag_keys
        ]
        if missing:
            return False, f"{item_kind} is missing tag keys {missing!r}."

    for field_name, fragments in (
        ("titleFormat", expectation.title_format_contains),
        ("textFormat", expectation.text_format_contains),
    ):
        actual_value = str(item.get(field_name, ""))
        for fragment in fragments:
            if normalize_token(fragment) not in normalize_token(actual_value):
                return False, f"{item_kind} field {field_name!r} is missing {fragment!r}."

    if expectation.query is not None:
        query_candidates = [query for query in query_texts if query.strip()]
        if not query_candidates:
            return False, f"{item_kind} has no saved query text to validate."
        query_failures: list[str] = []
        for query_text in query_candidates:
            ok, explanation = validate_query_semantics(query_text, expectation.query)
            if ok:
                break
            query_failures.append(explanation)
        else:
            return False, (
                f"{item_kind} query does not satisfy the required semantics: {query_failures[0]}"
            )

    for expected_query in expectation.queries_all:
        query_candidates = [query for query in query_texts if query.strip()]
        if not query_candidates:
            return False, f"{item_kind} has no saved query text to validate."
        required_query_failures: list[str] = []
        for query_text in query_candidates:
            ok, explanation = validate_query_semantics(query_text, expected_query)
            if ok:
                break
            required_query_failures.append(explanation)
        else:
            return False, (
                f"{item_kind} is missing one required saved query: {required_query_failures[0]}"
            )

    eval_end_unix = prometheus_eval_time_unix({}, transcript)
    for index, execute_case in enumerate(expectation.execute_cases, start=1):
        if eval_end_unix is None:
            return False, "execute_cases require transcript scenario time."
        ok, explanation = validate_dashboard_execute_case(
            item,
            execute_case,
            item_kind=item_kind,
            query_texts=query_texts,
            ctx=ctx,
            dashboard_variables=dashboard_variables,
            eval_end_unix=eval_end_unix,
        )
        if not ok:
            return False, f"{item_kind} execute_case #{index} failed: {explanation}"

    return True, ""


def summarize_dashboard_panels(
    snapshot: DashboardSnapshot,
    panel_titles: list[str],
    response_text: str,
) -> tuple[float, str]:
    panels, error = select_panels_for_titles(snapshot, panel_titles)
    if error:
        return 0.0, error
    if not panels:
        return 0.0, "Dashboard has no panels to cite."

    summary_lines = split_response_lines(response_text)
    for panel in panels:
        title = str(panel.get("title", "")).strip()
        panel_type = str(panel.get("type", "")).strip()
        if not title or not panel_type:
            return 0.0, "Dashboard panel summary requires titled panels with types."
        if not any(line_mentions_all(line, title, panel_type) for line in summary_lines):
            return 0.0, f"Response does not pair panel {title!r} with type {panel_type!r}."
    return 1.0, f"Response cites {len(panels)} dashboard panels with matching titles and types."


def response_cites_saved_queries(
    snapshot: DashboardSnapshot, response_text: str
) -> tuple[float, str]:
    normalized_response = normalize_query_text(strip_markdown_code_fences(response_text)).lower()
    for panel in snapshot.panels:
        title = str(panel.get("title", "")).strip()
        if title and normalize_token(title) not in normalize_token(response_text):
            return 0.0, f"Response does not mention panel title {title!r}."

        queries = [
            normalize_query_text(query).lower() for query in collect_panel_query_texts(panel)
        ]
        queries = [query for query in queries if query]
        if not queries:
            return 0.0, f"Panel {title!r} has no saved query text to validate."
        if not any(query in normalized_response for query in queries):
            return 0.0, f"Response does not cite the saved query for panel {title!r}."
    return 1.0, f"Response cites saved queries for {len(snapshot.panels)} dashboard panels."
