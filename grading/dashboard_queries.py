import json
import re
from collections.abc import Callable
from typing import Any

from grading.dashboard_snapshot import (
    collect_variable_query_texts,
    dashboard_datasource_type,
    find_named_dashboard_item,
    normalize_query_text,
    normalize_query_value,
    normalize_token,
)
from grading.env_context import (
    VerifierContext,
    fetch_loki_instant,
    fetch_loki_streams,
    fetch_prometheus_instant,
    fetch_prometheus_label_values,
    fetch_prometheus_vector,
)
from grading.models import (
    DistinctValueExpectation,
    ExecuteCase,
    FieldMatcherExpectation,
    LineFilterExpectation,
    NameMatcherExpectation,
    QueryExpectation,
)


class RawBindingValue:
    def __init__(self, text: str):
        self.text = text


type BindingValue = str | list[str] | RawBindingValue
type SeriesSample = tuple[dict[str, str], float]


VARIABLE_REFERENCE_RE = re.compile(
    r"\$(?P<name>[A-Za-z_][A-Za-z0-9_]*)|\$\{(?P<name_braced>[A-Za-z_][A-Za-z0-9_]*)(?::(?P<modifier>[^}]+))?\}"
)
REGEX_META_CHARS = frozenset(".+*?()|[]{}^$\\")


def validate_dashboard_execute_case(
    item: dict[str, Any],
    execute_case: ExecuteCase,
    *,
    item_kind: str,
    query_texts: list[str],
    ctx: VerifierContext | None,
    dashboard_variables: list[dict[str, Any]],
    eval_end_unix: float,
) -> tuple[bool, str]:
    if ctx is None:
        return False, "execute_cases require verifier HTTP context."

    datasource_type = dashboard_datasource_type(item).lower()
    try:
        bindings = resolve_execute_case_bindings(
            execute_case.bindings,
            dashboard_variables=dashboard_variables,
            ctx=ctx,
            eval_end_unix=eval_end_unix,
        )
    except ValueError as exc:
        return False, str(exc)

    lookback_sec = execute_case.lookback_sec or 24 * 60 * 60
    expanded_queries = [
        expand_grafana_query_text(query_text, bindings=bindings, lookback_sec=lookback_sec)
        for query_text in query_texts
        if query_text.strip()
    ]
    if not expanded_queries:
        return False, f"{item_kind} has no saved query text to execute."

    match execute_case.result_kind:
        case "prometheus_scalar":
            if datasource_type != "prometheus":
                return False, f"{item_kind} is {datasource_type!r}, expected Prometheus."
            return validate_prometheus_scalar_execute_case(
                expanded_queries,
                execute_case,
                ctx=ctx,
                eval_end_unix=eval_end_unix,
                bindings=bindings,
            )
        case "prometheus_vector":
            if datasource_type != "prometheus":
                return False, f"{item_kind} is {datasource_type!r}, expected Prometheus."
            return validate_prometheus_vector_execute_case(
                expanded_queries,
                execute_case,
                ctx=ctx,
                eval_end_unix=eval_end_unix,
                bindings=bindings,
            )
        case "loki_metric":
            if datasource_type != "loki":
                return False, f"{item_kind} is {datasource_type!r}, expected Loki."
            return validate_loki_metric_execute_case(
                expanded_queries,
                execute_case,
                ctx=ctx,
                eval_end_unix=eval_end_unix,
                bindings=bindings,
            )
        case "loki_streams":
            if datasource_type != "loki":
                return False, f"{item_kind} is {datasource_type!r}, expected Loki."
            return validate_loki_streams_execute_case(
                expanded_queries,
                execute_case,
                ctx=ctx,
                eval_end_unix=eval_end_unix,
                lookback_sec=lookback_sec,
            )
        case _:
            return False, f"unsupported execute_case result_kind {execute_case.result_kind!r}"


def validate_prometheus_scalar_execute_case(
    expanded_queries: list[str],
    execute_case: ExecuteCase,
    *,
    ctx: VerifierContext,
    eval_end_unix: float,
    bindings: dict[str, BindingValue],
) -> tuple[bool, str]:
    values, failure = collect_scalar_query_values(
        expanded_queries,
        fetch_value=fetch_prometheus_instant,
        base_url=ctx.prometheus_url,
        timeout_sec=ctx.timeout_sec,
        eval_end_unix=eval_end_unix,
        missing_value_error="query returned no scalar value",
    )
    if not values:
        return False, failure

    ok, explanation = validate_scalar_thresholds(values, execute_case)
    if not ok:
        return False, explanation

    ok, explanation = validate_canonical_scalar_result(
        values,
        execute_case,
        fetch_value=fetch_prometheus_instant,
        base_url=ctx.prometheus_url,
        timeout_sec=ctx.timeout_sec,
        eval_end_unix=eval_end_unix,
        bindings=bindings,
        missing_value_error="canonical query returned no scalar value",
        result_label="scalar result",
    )
    if not ok:
        return False, explanation
    return True, f"scalar query executed successfully ({values[0]:g})"


def validate_prometheus_vector_execute_case(
    expanded_queries: list[str],
    execute_case: ExecuteCase,
    *,
    ctx: VerifierContext,
    eval_end_unix: float,
    bindings: dict[str, BindingValue],
) -> tuple[bool, str]:
    combined_series: list[SeriesSample] = []
    failures: list[str] = []
    for query in expanded_queries:
        series, err = fetch_prometheus_vector(
            ctx.prometheus_url,
            query,
            ctx.timeout_sec,
            time_unix=eval_end_unix,
        )
        if err:
            failures.append(err)
            continue
        combined_series.extend(series)

    if execute_case.non_empty is not False and not combined_series:
        detail = failures[0] if failures else "no vector series returned"
        return False, detail

    if execute_case.canonical_query:
        expanded_canonical = expand_execute_case_query(
            execute_case.canonical_query,
            execute_case,
            bindings,
        )
        canonical_series, err = fetch_prometheus_vector(
            ctx.prometheus_url,
            expanded_canonical,
            ctx.timeout_sec,
            time_unix=eval_end_unix,
        )
        if err:
            return False, err
        tolerance = execute_case.numeric_tolerance or 0.08
        if not vector_series_match(combined_series, canonical_series, tolerance):
            return False, "vector result does not match canonical series"

    return validate_series_expectations(combined_series, execute_case)


def validate_loki_metric_execute_case(
    expanded_queries: list[str],
    execute_case: ExecuteCase,
    *,
    ctx: VerifierContext,
    eval_end_unix: float,
    bindings: dict[str, BindingValue],
) -> tuple[bool, str]:
    values, failure = collect_scalar_query_values(
        expanded_queries,
        fetch_value=fetch_loki_instant,
        base_url=ctx.loki_url,
        timeout_sec=ctx.timeout_sec,
        eval_end_unix=eval_end_unix,
        missing_value_error="query returned no Loki metric value",
    )
    if not values:
        return False, failure

    ok, explanation = validate_scalar_thresholds(values, execute_case)
    if not ok:
        return False, explanation

    ok, explanation = validate_canonical_scalar_result(
        values,
        execute_case,
        fetch_value=fetch_loki_instant,
        base_url=ctx.loki_url,
        timeout_sec=ctx.timeout_sec,
        eval_end_unix=eval_end_unix,
        bindings=bindings,
        missing_value_error="canonical Loki query returned no value",
        result_label="Loki metric result",
    )
    if not ok:
        return False, explanation
    return True, f"Loki metric query executed successfully ({values[0]:g})"


def validate_loki_streams_execute_case(
    expanded_queries: list[str],
    execute_case: ExecuteCase,
    *,
    ctx: VerifierContext,
    eval_end_unix: float,
    lookback_sec: int,
) -> tuple[bool, str]:
    start_unix = eval_end_unix - float(lookback_sec)
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    limit = execute_case.limit or 200

    for query in expanded_queries:
        query_records, err = fetch_loki_streams(
            ctx.loki_url,
            query,
            ctx.timeout_sec,
            start_unix=start_unix,
            end_unix=eval_end_unix,
            limit=limit,
        )
        if err:
            failures.append(err)
            continue
        for record in query_records:
            records.append({**record, "json": parse_json_line(record.get("line", ""))})

    if execute_case.non_empty is not False and not records:
        detail = failures[0] if failures else "no Loki log records returned"
        return False, detail

    for label_matcher in execute_case.series_label_matchers:
        if not any(
            generic_matcher_matches(name, "=", value, label_matcher)
            for record in records
            for name, value in record.get("stream", {}).items()
        ):
            return (
                False,
                f"missing stream-label match {label_matcher.model_dump(exclude_none=True)!r}",
            )

    for field_matcher in execute_case.json_field_matchers:
        if not any(
            generic_matcher_matches(name, "=", value, field_matcher)
            for record in records
            for name, value in flattened_json_scalar_items(record.get("json"))
        ):
            return (
                False,
                f"missing JSON field match {field_matcher.model_dump(exclude_none=True)!r}",
            )

    if execute_case.required_json_fields:
        if not any(
            isinstance(record.get("json"), dict)
            and all(field in record["json"] for field in execute_case.required_json_fields)
            for record in records
        ):
            return (
                False,
                f"no JSON log record contains all required fields {execute_case.required_json_fields!r}",
            )

    ok, explanation = validate_distinct_value_expectations(
        records,
        execute_case.distinct_label_values,
    )
    if not ok:
        return False, explanation
    return True, f"Loki log query returned {len(records)} records"


def validate_scalar_thresholds(
    values: list[float],
    execute_case: ExecuteCase,
) -> tuple[bool, str]:
    if execute_case.min_value is not None and max(values) < execute_case.min_value:
        return False, f"scalar result {values!r} is below min_value {execute_case.min_value:g}"
    if execute_case.max_value is not None and min(values) > execute_case.max_value:
        return False, f"scalar result {values!r} is above max_value {execute_case.max_value:g}"
    return True, ""


def collect_scalar_query_values(
    expanded_queries: list[str],
    *,
    fetch_value: Callable[..., tuple[float | None, str]],
    base_url: str,
    timeout_sec: float,
    eval_end_unix: float,
    missing_value_error: str,
) -> tuple[list[float], str]:
    values: list[float] = []
    failures: list[str] = []
    for query in expanded_queries:
        value, err = fetch_value(base_url, query, timeout_sec, time_unix=eval_end_unix)
        if err:
            failures.append(err)
        elif value is None:
            failures.append(missing_value_error)
        else:
            values.append(value)
    return values, failures[0] if failures else missing_value_error


def validate_canonical_scalar_result(
    values: list[float],
    execute_case: ExecuteCase,
    *,
    fetch_value: Callable[..., tuple[float | None, str]],
    base_url: str,
    timeout_sec: float,
    eval_end_unix: float,
    bindings: dict[str, BindingValue],
    missing_value_error: str,
    result_label: str,
) -> tuple[bool, str]:
    if not execute_case.canonical_query:
        return True, ""

    expanded_canonical = expand_execute_case_query(
        execute_case.canonical_query,
        execute_case,
        bindings,
    )
    ref_value, err = fetch_value(base_url, expanded_canonical, timeout_sec, time_unix=eval_end_unix)
    if err or ref_value is None:
        return False, err or missing_value_error

    tolerance = execute_case.numeric_tolerance or 0.08
    if any(scalar_matches(actual, ref_value, tolerance) for actual in values):
        return True, ""
    return (
        False,
        f"{result_label} {values!r} does not match canonical value {ref_value:g} within tolerance {tolerance:.0%}",
    )


def validate_series_expectations(
    combined_series: list[SeriesSample],
    execute_case: ExecuteCase,
) -> tuple[bool, str]:
    if (
        execute_case.series_count_exact is not None
        and len(combined_series) != execute_case.series_count_exact
    ):
        return False, (
            f"vector query returned {len(combined_series)} series, expected exactly {execute_case.series_count_exact}"
        )
    if (
        execute_case.series_count_min is not None
        and len(combined_series) < execute_case.series_count_min
    ):
        return False, (
            f"vector query returned {len(combined_series)} series, below series_count_min {execute_case.series_count_min}"
        )
    if (
        execute_case.series_count_max is not None
        and len(combined_series) > execute_case.series_count_max
    ):
        return False, (
            f"vector query returned {len(combined_series)} series, above series_count_max {execute_case.series_count_max}"
        )

    series_values = [value for _, value in combined_series]
    ok, explanation = validate_scalar_thresholds(series_values, execute_case)
    if not ok:
        return False, explanation

    for matcher in execute_case.series_label_matchers:
        if not any(
            generic_matcher_matches(name, "=", value, matcher)
            for labels, _ in combined_series
            for name, value in labels.items()
        ):
            return False, f"missing series-label match {matcher.model_dump(exclude_none=True)!r}"

    ok, explanation = validate_distinct_value_expectations(
        combined_series,
        execute_case.distinct_label_values,
    )
    if not ok:
        return False, explanation
    return True, f"vector query returned {len(combined_series)} series"


def validate_distinct_value_expectations(
    source_items: list[Any],
    expectations: list[DistinctValueExpectation],
) -> tuple[bool, str]:
    for expectation in expectations:
        values = collect_distinct_values(source_items, expectation)
        missing = [value for value in expectation.values_all if value not in values]
        if missing:
            return (
                False,
                f"distinct values for {expectation.model_dump(exclude_none=True)!r} are missing {missing!r}",
            )
        if expectation.values_exact is not None:
            exact = set(expectation.values_exact)
            if values != exact:
                return False, f"distinct values {sorted(values)!r} did not equal {sorted(exact)!r}"
    return True, ""


def collect_distinct_values(
    source_items: list[Any],
    expectation: DistinctValueExpectation,
) -> set[str]:
    values: set[str] = set()
    if expectation.source == "json":
        for record in source_items:
            parsed = record.get("json") if isinstance(record, dict) else None
            if isinstance(parsed, dict) and expectation.name in parsed:
                values.add(str(parsed[expectation.name]))
        return values

    if expectation.source == "stream":
        for record in source_items:
            if isinstance(record, dict):
                stream = record.get("stream")
                if isinstance(stream, dict) and expectation.name in stream:
                    values.add(str(stream[expectation.name]))
        return values

    for item in source_items:
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], dict):
            labels = item[0]
            if expectation.name in labels:
                values.add(str(labels[expectation.name]))
    return values


def flattened_json_scalar_items(raw: Any) -> list[tuple[str, str]]:
    if not isinstance(raw, dict):
        return []
    items: list[tuple[str, str]] = []
    for key, value in raw.items():
        if isinstance(value, dict):
            items.extend(flattened_json_scalar_items(value))
            continue
        if isinstance(value, list):
            for entry in value:
                if isinstance(entry, str | int | float | bool):
                    items.append((str(key), str(entry)))
            continue
        if isinstance(value, str | int | float | bool):
            items.append((str(key), str(value)))
    return items


def parse_json_line(raw: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(raw)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def resolve_execute_case_bindings(
    raw: dict[str, str | list[str]],
    *,
    dashboard_variables: list[dict[str, Any]],
    ctx: VerifierContext,
    eval_end_unix: float,
) -> dict[str, BindingValue]:
    bindings: dict[str, BindingValue] = {}
    for key, value in raw.items():
        if isinstance(value, list):
            bindings[key] = [str(item) for item in value]
        elif value == "__all__":
            bindings[key] = resolve_all_binding_value(
                key,
                dashboard_variables=dashboard_variables,
                ctx=ctx,
                eval_end_unix=eval_end_unix,
            )
        else:
            bindings[key] = value
    return bindings


def resolve_all_binding_value(
    variable_name: str,
    *,
    dashboard_variables: list[dict[str, Any]],
    ctx: VerifierContext,
    eval_end_unix: float,
) -> BindingValue:
    variable = find_named_dashboard_item(
        dashboard_variables,
        item_name=variable_name,
        field="name",
    )
    if variable is None:
        raise ValueError(
            f"Execute-case binding requested '__all__' for missing dashboard variable {variable_name!r}."
        )

    all_value = variable.get("allValue")
    if isinstance(all_value, str) and all_value.strip():
        return RawBindingValue(all_value.strip())

    options = resolve_variable_option_values(variable, ctx=ctx, eval_end_unix=eval_end_unix)
    return options or "__all__"


def resolve_variable_option_values(
    variable: dict[str, Any],
    *,
    ctx: VerifierContext,
    eval_end_unix: float,
) -> list[str]:
    for query_text in collect_variable_query_texts(variable):
        parsed = parse_grafana_label_values_query(query_text)
        if parsed is None:
            continue
        source_expr, label_name = parsed
        option_series, err = fetch_prometheus_vector(
            ctx.prometheus_url,
            f"count by ({label_name}) ({source_expr})",
            ctx.timeout_sec,
            time_unix=eval_end_unix,
        )
        if not err and option_series:
            values = sorted(
                {
                    str(labels[label_name])
                    for labels, _ in option_series
                    if label_name in labels and str(labels[label_name]).strip()
                }
            )
            if values:
                return values

        values, err = fetch_prometheus_label_values(
            ctx.prometheus_url,
            label_name,
            ctx.timeout_sec,
            match_expr=label_values_source_to_match_expr(source_expr),
            time_unix=eval_end_unix,
        )
        if not err and values:
            return values
    return []


def expand_grafana_query_text(
    query_text: str,
    *,
    bindings: dict[str, BindingValue],
    lookback_sec: int,
) -> str:
    range_text = seconds_to_promql_duration(lookback_sec)
    expanded = query_text.replace("${__range}", range_text).replace("$__range", range_text)
    expanded = normalize_promql_macros(expanded)

    def replace(match: re.Match[str]) -> str:
        var_name = match.group("name") or match.group("name_braced") or ""
        modifier = match.group("modifier")
        value = bindings.get(var_name)
        return render_binding_value(value, modifier)

    return VARIABLE_REFERENCE_RE.sub(replace, expanded)


def render_binding_value(value: BindingValue | None, modifier: str | None) -> str:
    if modifier in {"regex", "regex-safe"}:
        if isinstance(value, RawBindingValue):
            return value.text
        if value is None or value == "__all__":
            return ".+"
        if isinstance(value, list):
            if not value:
                return ".+"
            return "|".join(render_regex_binding_part(str(item)) for item in value)
        return render_regex_binding_part(str(value))

    if isinstance(value, RawBindingValue):
        return value.text
    if value is None or value == "__all__":
        return "All"
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)


def render_regex_binding_part(raw: str) -> str:
    escaped_regex = "".join(f"\\{char}" if char in REGEX_META_CHARS else char for char in raw)
    return escaped_regex.replace("\\", "\\\\").replace('"', '\\"')


def scalar_matches(actual: float, expected: float, tolerance: float) -> bool:
    if expected == 0:
        return actual == 0
    return abs(actual - expected) / abs(expected) <= tolerance


def vector_series_match(
    actual: list[SeriesSample],
    expected: list[SeriesSample],
    tolerance: float,
) -> bool:
    actual_map = {series_label_key(labels): value for labels, value in actual}
    expected_map = {series_label_key(labels): value for labels, value in expected}
    if actual_map.keys() != expected_map.keys():
        return False
    return all(scalar_matches(actual_map[key], expected_map[key], tolerance) for key in actual_map)


def series_label_key(labels: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(key), str(value)) for key, value in labels.items()))


def seconds_to_promql_duration(total_seconds: int) -> str:
    if total_seconds <= 0:
        return "1m"
    units = (
        ("w", 7 * 24 * 60 * 60),
        ("d", 24 * 60 * 60),
        ("h", 60 * 60),
        ("m", 60),
        ("s", 1),
    )
    remainder = int(total_seconds)
    parts: list[str] = []
    for suffix, factor in units:
        amount, remainder = divmod(remainder, factor)
        if amount:
            parts.append(f"{amount}{suffix}")
    return "".join(parts) or "1m"


def expand_execute_case_query(
    query_text: str,
    execute_case: ExecuteCase,
    bindings: dict[str, BindingValue],
) -> str:
    return expand_grafana_query_text(
        query_text,
        bindings=bindings,
        lookback_sec=execute_case.lookback_sec or 24 * 60 * 60,
    )


def validate_query_semantics(query_text: str, expectation: QueryExpectation) -> tuple[bool, str]:
    if expectation.any_of:
        base_data = expectation.model_dump(exclude={"any_of"}, exclude_none=True)
        base = QueryExpectation.model_validate(base_data)
        base_ok, base_explanation = _validate_query_semantics(query_text, base)
        if not base_ok:
            return False, base_explanation
        failures: list[str] = []
        for alternative in expectation.any_of:
            merged = QueryExpectation.model_validate(
                {**base_data, **alternative.model_dump(exclude_none=True)}
            )
            ok, explanation = _validate_query_semantics(query_text, merged)
            if ok:
                return True, ""
            failures.append(explanation)
        detail = failures[0] if failures else "no alternative matched"
        return False, f"none of the allowed query alternatives matched ({detail})"
    return _validate_query_semantics(query_text, expectation)


def _validate_query_semantics(query_text: str, expectation: QueryExpectation) -> tuple[bool, str]:
    match expectation.language:
        case "promql":
            return validate_promql_query_semantics(query_text, expectation)
        case "logql":
            return validate_logql_query_semantics(query_text, expectation)
        case "traceql":
            return validate_traceql_query_semantics(query_text, expectation)
        case "grafana_label_values":
            return validate_grafana_label_values_query_semantics(query_text, expectation)
        case _:
            return False, f"unknown query language {expectation.language!r}"


def validate_promql_query_semantics(
    query_text: str,
    expectation: QueryExpectation,
) -> tuple[bool, str]:
    try:
        import promql_parser
    except ImportError:
        return False, "promql-parser package not installed."

    try:
        ast = promql_parser.parse(normalize_promql_macros(query_text))
    except Exception as exc:
        return False, f"invalid PromQL: {exc}"

    metrics: set[str] = set()
    functions: set[str] = set()
    matchers: list[tuple[str, str, str]] = []

    def visit(node: Any) -> None:
        if node is None:
            return
        if isinstance(node, promql_parser.Call):
            functions.add(str(node.func.name))
            for arg in node.args:
                visit(arg)
            return
        if isinstance(node, promql_parser.MatrixSelector):
            visit(node.vector_selector)
            return
        if isinstance(node, promql_parser.VectorSelector):
            if node.name:
                metrics.add(str(node.name))
            for matcher in node.matchers.matchers:
                matchers.append(
                    (str(matcher.name), promql_match_op(str(matcher.op)), str(matcher.value))
                )
            return
        for attr in ("expr", "lhs", "rhs", "param", "vector_selector"):
            if hasattr(node, attr):
                visit(getattr(node, attr))
        if hasattr(node, "args") and isinstance(node.args, list):
            for arg in node.args:
                visit(arg)

    visit(ast)

    missing_metrics = [metric for metric in expectation.metric_names if metric not in metrics]
    if missing_metrics:
        return False, f"missing metric names {missing_metrics!r}"

    missing_functions = [func for func in expectation.functions if func not in functions]
    if missing_functions:
        return False, f"missing functions {missing_functions!r}"

    for matcher_expectation in expectation.label_matchers:
        ok = any(
            promql_matcher_matches(name, op, value, matcher_expectation)
            for name, op, value in matchers
        )
        if not ok:
            return (
                False,
                f"missing label matcher {matcher_expectation.model_dump(exclude_none=True)!r}",
            )

    return validate_generic_query_references(query_text, expectation)


def validate_logql_query_semantics(
    query_text: str,
    expectation: QueryExpectation,
) -> tuple[bool, str]:
    label_matchers = extract_label_matchers(query_text)
    for matcher_expectation in expectation.label_matchers:
        ok = any(
            generic_matcher_matches(name, op, value, matcher_expectation)
            for name, op, value in label_matchers
        )
        if not ok:
            return (
                False,
                f"missing stream selector matcher {matcher_expectation.model_dump(exclude_none=True)!r}",
            )

    normalized = normalize_query_text(query_text)
    for stage in expectation.pipeline_stages:
        if re.search(rf"\|\s*{re.escape(stage)}\b", normalized, re.IGNORECASE) is None:
            return False, f"missing pipeline stage {stage!r}"

    extracted_filters = extract_field_filters(query_text)
    for field_filter_expectation in expectation.field_filters:
        ok = any(
            generic_matcher_matches(name, op, value, field_filter_expectation)
            for name, op, value in extracted_filters
        )
        if not ok:
            return False, (
                f"missing field filter {field_filter_expectation.model_dump(exclude_none=True)!r}"
            )

    line_filters = extract_line_filters(query_text)
    for line_filter_expectation in expectation.line_filters:
        ok = any(
            line_filter_matches(op, value, line_filter_expectation) for op, value in line_filters
        )
        if not ok:
            return False, (
                f"missing line filter {line_filter_expectation.model_dump(exclude_none=True)!r}"
            )

    return validate_generic_query_references(query_text, expectation)


def validate_traceql_query_semantics(
    query_text: str,
    expectation: QueryExpectation,
) -> tuple[bool, str]:
    extracted_filters = extract_field_filters(query_text)
    for field_filter_expectation in expectation.field_filters:
        ok = any(
            generic_matcher_matches(name, op, value, field_filter_expectation)
            for name, op, value in extracted_filters
        )
        if not ok:
            return False, (
                f"missing trace filter {field_filter_expectation.model_dump(exclude_none=True)!r}"
            )
    return validate_generic_query_references(query_text, expectation)


def validate_grafana_label_values_query_semantics(
    query_text: str,
    expectation: QueryExpectation,
) -> tuple[bool, str]:
    metric_name = expectation.metric_name or ""
    label_name = expectation.label_name or ""
    parsed = parse_grafana_label_values_query(query_text)
    if parsed is None:
        return (
            False,
            f"expected label_values({metric_name}, {label_name}) style query, got {query_text!r}",
        )
    source_expr, actual_label = parsed
    if normalize_query_value(actual_label) != normalize_query_value(label_name):
        return (
            False,
            f"expected label_values source for label {label_name!r}, got label {actual_label!r}",
        )
    if not grafana_label_values_source_matches(source_expr, metric_name):
        return (
            False,
            f"expected label_values source derived from {metric_name!r}, got {source_expr!r}",
        )
    return True, ""


def parse_grafana_label_values_query(query_text: str) -> tuple[str, str] | None:
    normalized = normalize_query_text(query_text)
    match = re.match(
        r"^label_values\(\s*(.+)\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)$",
        normalized,
        re.IGNORECASE,
    )
    if match is None:
        return None
    return match.group(1), match.group(2)


def label_values_source_to_match_expr(source_expr: str) -> str:
    normalized = normalize_query_text(source_expr)
    if re.fullmatch(r"[A-Za-z_:][A-Za-z0-9_:]*", normalized):
        return f'{{__name__="{normalized}"}}'
    return normalized


def normalize_promql_macros(query_text: str) -> str:
    return (
        query_text.replace("$__range", "5m")
        .replace("${__range}", "5m")
        .replace("$__interval", "1m")
        .replace("${__interval}", "1m")
        .replace("$__rate_interval", "1m")
        .replace("${__rate_interval}", "1m")
    )


def grafana_label_values_source_matches(source_expr: str, metric_name: str) -> bool:
    normalized_source = normalize_query_value(source_expr)
    normalized_metric = normalize_query_value(metric_name)
    if normalized_source == normalized_metric or normalized_metric in normalized_source:
        return True

    metric_hints = {normalized_metric}
    for suffix in ("_total", "_bucket", "_sum", "_count"):
        if normalized_metric.endswith(suffix):
            base = normalized_metric[: -len(suffix)]
            metric_hints.add(base)
            if base.endswith("s"):
                metric_hints.add(base[:-1])

    return any(hint and hint in normalized_source for hint in metric_hints)


def validate_generic_query_references(
    query_text: str,
    expectation: QueryExpectation,
) -> tuple[bool, str]:
    normalized = normalize_query_text(query_text).lower()
    for reference in expectation.references_all:
        if normalize_query_text(reference).lower() not in normalized:
            return False, f"missing reference {reference!r}"
    for group in expectation.references_any_of_groups:
        if not any(normalize_query_text(option).lower() in normalized for option in group):
            return False, f"missing any allowed reference from {group!r}"
    return True, ""


def promql_matcher_matches(
    name: str,
    op: str,
    value: str,
    expectation: NameMatcherExpectation,
) -> bool:
    if normalize_token(name) != normalize_token(expectation.name):
        return False
    return matcher_constraints_match(op, value, expectation)


def generic_matcher_matches(
    name: str,
    op: str,
    value: str,
    expectation: NameMatcherExpectation | FieldMatcherExpectation,
) -> bool:
    expected_name = (
        expectation.name if isinstance(expectation, NameMatcherExpectation) else expectation.field
    )
    if normalize_token(name) != normalize_token(expected_name):
        return False
    return matcher_constraints_match(op, value, expectation)


def matcher_constraints_match(
    op: str,
    value: str,
    expectation: NameMatcherExpectation | FieldMatcherExpectation | LineFilterExpectation,
) -> bool:
    if expectation.op is not None and expectation.op != op:
        return False
    if expectation.op_any_of and op not in set(expectation.op_any_of):
        return False

    normalized_value = normalize_query_value(value)
    if (
        expectation.value is not None
        and normalize_query_value(expectation.value) != normalized_value
    ):
        return False
    if expectation.value_any_of and normalized_value not in {
        normalize_query_value(candidate) for candidate in expectation.value_any_of
    }:
        return False
    if any(
        normalize_token(fragment) not in normalized_value for fragment in expectation.value_all_of
    ):
        return False
    return True


def line_filter_matches(
    op: str,
    value: str,
    expectation: LineFilterExpectation,
) -> bool:
    return matcher_constraints_match(op, value, expectation)


def extract_label_matchers(query_text: str) -> list[tuple[str, str, str]]:
    matches: list[tuple[str, str, str]] = []
    matcher_re = re.compile(
        r"([A-Za-z_][A-Za-z0-9_.:-]*)\s*(=~|!~|!=|=)\s*(\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'|`(?:[^`\\]|\\.)*`)"
    )
    for matcher in matcher_re.finditer(query_text):
        matches.append((matcher.group(1), matcher.group(2), strip_quotes(matcher.group(3))))
    return matches


def extract_field_filters(query_text: str) -> list[tuple[str, str, str]]:
    matcher_re = re.compile(
        r"\b([A-Za-z_][A-Za-z0-9_.:-]*)\s*(=~|!~|!=|=|>=|<=|>|<)\s*(\"[^\"]*\"|'[^']*'|`[^`]*`|[A-Za-z0-9_./:-]+)"
    )
    matches: list[tuple[str, str, str]] = []
    for matcher in matcher_re.finditer(query_text):
        matches.append((matcher.group(1), matcher.group(2), strip_quotes(matcher.group(3))))
    return matches


def extract_line_filters(query_text: str) -> list[tuple[str, str]]:
    matcher_re = re.compile(
        r"\|\s*(=|~|!=|!~)\s*(\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'|`(?:[^`\\]|\\.)*`)"
    )
    matches: list[tuple[str, str]] = []
    for matcher in matcher_re.finditer(query_text):
        op = "=~" if matcher.group(1) == "~" else matcher.group(1)
        matches.append((op, strip_quotes(matcher.group(2))))
    return matches


def strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'", "`"}:
        return value[1:-1]
    return value


def promql_match_op(raw: str) -> str:
    return {
        "MatchOp.Equal": "=",
        "MatchOp.NotEqual": "!=",
        "MatchOp.Re": "=~",
        "MatchOp.NotRe": "!~",
    }.get(raw, raw)
