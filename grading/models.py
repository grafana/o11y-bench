import json
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DashboardSelector(SpecModel):
    uid: str | None = None
    title: str | None = None

    @model_validator(mode="after")
    def validate_selector(self) -> Self:
        if self.uid or self.title:
            return self
        raise ValueError("Dashboard specs require a uid or title.")


class DatasourceSelector(SpecModel):
    name: str | None = None
    type: str | None = None

    @model_validator(mode="after")
    def validate_selector(self) -> Self:
        if self.name or self.type:
            return self
        raise ValueError("Datasource specs require a name or type selector.")


class QueryFact(SpecModel):
    kind: Literal["query"]
    backend: Literal["prometheus", "loki", "tempo"]
    query: str
    # An empty result is the intended truth ("nothing matches") instead of a grading error.
    allow_empty: bool = False
    time_unix: float | None = None
    limit: int | None = None
    lookback_hours: float | None = None
    start_sec: int | None = None
    end_sec: int | None = None

    @model_validator(mode="after")
    def validate_query(self) -> Self:
        window = (self.limit, self.lookback_hours, self.start_sec, self.end_sec)
        if self.backend != "tempo" and any(v is not None for v in window):
            raise ValueError("limit, lookback_hours, start_sec and end_sec apply to tempo facts.")
        if self.backend == "tempo" and self.time_unix is not None:
            raise ValueError("time_unix applies to prometheus and loki facts.")
        if self.backend != "prometheus":
            return self

        try:
            import promql_parser
        except ImportError as exc:
            raise ValueError("promql-parser package not installed.") from exc

        expr = self.query.strip()
        if not expr:
            raise ValueError("Prometheus fact queries require a non-empty query.")
        try:
            promql_parser.parse(expr)
        except Exception as exc:
            raise ValueError(f"Invalid PromQL fact query: {exc}") from exc
        return self


class DashboardFact(DashboardSelector):
    kind: Literal["resource"]
    resource: Literal["dashboard"]


class DashboardDiffFact(DashboardSelector):
    """The saved dashboard compared with the one the task seeded, as a compact structural diff."""

    kind: Literal["resource"]
    resource: Literal["dashboard_diff"]


class DatasourceInventoryFact(SpecModel):
    kind: Literal["resource"]
    resource: Literal["datasource_inventory"]


class DatasourceDetailFact(DatasourceSelector):
    kind: Literal["resource"]
    resource: Literal["datasource_detail"]


type ResourceFact = Annotated[
    DashboardFact | DashboardDiffFact | DatasourceInventoryFact | DatasourceDetailFact,
    Field(discriminator="resource"),
]


class GrafanaApiFact(SpecModel):
    """Any Grafana API read as the truth, e.g. /api/folders or /api/v1/provisioning/alert-rules."""

    kind: Literal["grafana"]
    path: str
    # A 404 or an empty response is the intended truth ("nothing there").
    allow_empty: bool = False

    @model_validator(mode="after")
    def validate_path(self) -> Self:
        if not self.path.startswith(("/api/", "/apis/")):
            raise ValueError(
                f"Grafana API fact path must start with /api/ or /apis/: {self.path!r}"
            )
        return self


class CitedTraceFact(SpecModel):
    """The task's example trace: the first one the final response cites that is in Tempo and
    matches the task's cited_trace check, as derived facts and a span tree. A criterion on it
    fails without a judge call when the response cites no such trace."""

    kind: Literal["cited_trace"]


type FactSpec = Annotated[
    QueryFact | ResourceFact | GrafanaApiFact | CitedTraceFact, Field(discriminator="kind")
]


class MatcherExpectation(SpecModel):
    op: str | None = None
    op_any_of: list[str] = Field(default_factory=list)
    value: str | None = None
    value_any_of: list[str] = Field(default_factory=list)
    value_all_of: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_values(self) -> Self:
        if self.value is not None or self.value_any_of or self.value_all_of:
            return self
        raise ValueError("Matcher expectations require value, value_any_of, or value_all_of.")


class NameMatcherExpectation(MatcherExpectation):
    name: str


class FieldMatcherExpectation(MatcherExpectation):
    field: str


class LineFilterExpectation(MatcherExpectation):
    pass


class QueryExpectation(SpecModel):
    language: Literal["promql", "logql", "traceql", "grafana_label_values"] | None = None
    any_of: list[QueryExpectation] = Field(default_factory=list)
    metric_names: list[str] = Field(default_factory=list)
    functions: list[str] = Field(default_factory=list)
    pipeline_stages: list[str] = Field(default_factory=list)
    references_all: list[str] = Field(default_factory=list)
    references_any_of_groups: list[list[str]] = Field(default_factory=list)
    tag_keys_all: list[str] = Field(default_factory=list)
    metric_name: str | None = None
    label_name: str | None = None
    label_matchers: list[NameMatcherExpectation] = Field(default_factory=list)
    field_filters: list[FieldMatcherExpectation] = Field(default_factory=list)
    line_filters: list[LineFilterExpectation] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_query(self) -> Self:
        if self.language is None and not self.any_of:
            raise ValueError("Query expectations require a language.")
        if self.language == "grafana_label_values" and (
            not self.metric_name or not self.label_name
        ):
            raise ValueError(
                "grafana_label_values expectations require metric_name and label_name."
            )
        return self


class DistinctValueExpectation(SpecModel):
    name: str
    source: Literal["label", "stream", "json"] = "label"
    values_all: list[str] = Field(default_factory=list)
    values_exact: list[str] | None = None


class ExecuteCase(SpecModel):
    result_kind: Literal[
        "prometheus_scalar",
        "prometheus_vector",
        "loki_metric",
        "loki_streams",
    ]
    bindings: dict[str, str | list[str]] = Field(default_factory=dict)
    lookback_sec: int | None = None
    # Evaluate this many seconds before the scenario clock, for a series that is flat at "now".
    eval_offset_sec: int | None = None
    limit: int | None = None
    series_count_exact: int | None = None
    series_count_min: int | None = None
    series_count_max: int | None = None
    non_empty: bool | None = None
    numeric_tolerance: float | None = None
    min_value: float | None = None
    max_value: float | None = None
    canonical_query: str | None = None
    required_json_fields: list[str] = Field(default_factory=list)
    series_label_matchers: list[NameMatcherExpectation] = Field(default_factory=list)
    json_field_matchers: list[FieldMatcherExpectation] = Field(default_factory=list)
    distinct_label_values: list[DistinctValueExpectation] = Field(default_factory=list)


class DashboardItemExpectation(SpecModel):
    title: str | None = None
    name: str | None = None
    type: str | None = None
    type_any_of: list[str] = Field(default_factory=list)
    datasource_type: str | None = None
    match_count: int | None = None
    enabled: bool | None = None
    include_all: bool | None = None
    multi: bool | None = None
    tag_keys_all: list[str] = Field(default_factory=list)
    title_format_contains: list[str] = Field(default_factory=list)
    text_format_contains: list[str] = Field(default_factory=list)
    query: QueryExpectation | None = None
    queries_all: list[QueryExpectation] = Field(default_factory=list)
    execute_cases: list[ExecuteCase] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_expectation(self) -> Self:
        if self.model_fields_set:
            return self
        raise ValueError("Dashboard item expectations must include at least one constraint.")


class CitedTraceParams(SpecModel):
    """A trace the final response cites by its full ID, read back from Tempo: it must exist
    there and match `match`, a TraceQL spanset filter such as
    `{ span.http.route = "/api/orders" && status = error }`."""

    mode: Literal["cited_trace"]
    match: str
    lookback_hours: float = 24.0

    @model_validator(mode="after")
    def validate_match(self) -> Self:
        if self.match.strip().startswith("{"):
            return self
        raise ValueError(f"cited_trace match must be a TraceQL spanset filter: {self.match!r}")


class DashboardStateParams(DashboardSelector):
    mode: Literal["dashboard_state"]
    panel_count: int | None = None
    variable_count: int | None = None
    annotation_count: int | None = None
    panels: list[DashboardItemExpectation] = Field(default_factory=list)
    variables: list[DashboardItemExpectation] = Field(default_factory=list)
    annotations: list[DashboardItemExpectation] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_expectations(self) -> Self:
        if any(
            value is not None and value != []
            for value in (
                self.panel_count,
                self.variable_count,
                self.annotation_count,
                self.panels,
                self.variables,
                self.annotations,
            )
        ):
            return self
        raise ValueError(
            "dashboard_state requires at least one panel, variable, annotation, or count expectation."
        )


class DatasourceInventoryStateParams(SpecModel):
    mode: Literal["datasource_inventory"]
    types: list[str] = Field(default_factory=list)
    names: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_expectations(self) -> Self:
        if self.types or self.names:
            return self
        raise ValueError("datasource_inventory requires at least one expected type or name.")


class DatasourceDetailStateParams(DatasourceSelector):
    mode: Literal["datasource_detail"]
    require_url: bool = False
    require_access: bool = False


class TempoTraceServiceInventoryStateParams(SpecModel):
    mode: Literal["tempo_trace_service_inventory"]
    lookback_hours: float = 24.0
    count: int | None = None
    services_all: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_expectations(self) -> Self:
        if self.count is not None or self.services_all:
            return self
        raise ValueError(
            "tempo_trace_service_inventory requires an expected count or expected services."
        )


type CheckParams = Annotated[
    CitedTraceParams
    | DashboardStateParams
    | DatasourceInventoryStateParams
    | DatasourceDetailStateParams
    | TempoTraceServiceInventoryStateParams,
    Field(discriminator="mode"),
]


class RubricItem(SpecModel):
    criterion: str
    weight: float = Field(gt=0)
    fact: FactSpec | None = None


class CheckItem(SpecModel):
    name: str
    weight: float = Field(gt=0)
    # Every check reads state back from the stack after the agent finished.
    type: Literal["state"]
    params: CheckParams


@dataclass(frozen=True)
class JudgeCriterion:
    criterion: str
    weight: float
    prompt_text: str
    # The truth for this criterion could not be read; it is not sent to the judge.
    error: str | None = None
    # Backed by a fact the answer must state, so the judge's quote must be in the answer.
    answer_backed: bool = False
    # Fails without a judge call, for this reason: the example it is about does not exist.
    fails: str = ""


class Problem(SpecModel):
    id: str
    category: str
    tags: list[str] = Field(default_factory=list)
    statement: str
    rubric: list[RubricItem] = Field(default_factory=list)
    checks: list[CheckItem] = Field(default_factory=list)
    setup_dashboards: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def trace_example(self) -> CitedTraceParams | None:
        """What the task's example trace must match, from its cited_trace check."""
        return next((c.params for c in self.checks if isinstance(c.params, CitedTraceParams)), None)

    @model_validator(mode="after")
    def validate_trace_example(self) -> Self:
        cited = [c for c in self.checks if isinstance(c.params, CitedTraceParams)]
        if len(cited) > 1:
            raise ValueError("A task has at most one cited_trace check: it defines the example.")
        if not cited and any(isinstance(item.fact, CitedTraceFact) for item in self.rubric):
            raise ValueError("A cited_trace fact needs a cited_trace check to say what it matches.")
        return self

    @model_validator(mode="after")
    def validate_item_names_are_unique(self) -> Self:
        """Scores and weights are keyed by name, so a repeated name would merge two items."""
        names = [c.name for c in self.checks] + [r.criterion for r in self.rubric]
        repeated = sorted({n for n in names if names.count(n) > 1})
        if repeated:
            raise ValueError(f"Check and criterion names must be unique: {repeated}")
        return self


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass
class Message:
    role: str
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_results: list[ToolResult] | None = None
    thinking_content: str | None = None


# Judge transcript budgets, in characters.
AGENT_STEP_MAX_CHARS = 200_000
MIDDLE_STEP_MAX_CHARS = 500
MIDDLE_STEPS_TOTAL_CHARS = 4096
TOOL_CALL_MAX_CHARS = 2400
TOOL_RESULT_MAX_CHARS = 400
TOOL_RESULTS_TOTAL_CHARS = 6000
TOOL_ERRORS_TOTAL_CHARS = 2000
CLOSING_NOTE_MAX_CHARS = 200
FINAL_RESPONSE_LABEL = "Agent (final response):"


@dataclass
class Transcript:
    messages: list[Message] = field(default_factory=list)

    def final_answer_indexes(self) -> list[int]:
        """Message indexes whose text is the final response, oldest first.

        The final response is the last assistant message with text after the last user
        message. If that message also calls tools it is a planning turn, so there is no
        answer. A short closing note gets the nearest earlier long message prepended.
        """
        last_user = max(
            (i for i, msg in enumerate(self.messages) if msg.role == "user"), default=-1
        )
        texts = [
            i
            for i, msg in enumerate(self.messages)
            if i > last_user and msg.role == "assistant" and (msg.content or "").strip()
        ]
        if not texts or self.messages[texts[-1]].tool_calls:
            return []
        final = texts[-1]
        if len(self.messages[final].content or "") >= CLOSING_NOTE_MAX_CHARS:
            return [final]
        earlier = next(
            (
                i
                for i in reversed(texts[:-1])
                if len(self.messages[i].content or "") >= CLOSING_NOTE_MAX_CHARS
            ),
            None,
        )
        return [final] if earlier is None else [earlier, final]

    def final_answer(self) -> str:
        return "\n\n".join(
            (self.messages[i].content or "").strip() for i in self.final_answer_indexes()
        )

    def to_text(self) -> str:
        """Render the transcript the way the judge reads it.

        One line per step, tool calls inline with a ✓/✗ marker, a short excerpt of each
        tool result after its step, and the final response labelled. The first, last and
        final-response steps are shown whole; other agent text shares a small budget.
        """
        final = set(self.final_answer_indexes())
        agent_steps = [i for i, msg in enumerate(self.messages) if msg.role == "assistant"]
        whole = final | set(agent_steps[:1]) | set(agent_steps[-1:])
        results: dict[str, ToolResult] = {
            result.tool_call_id: result
            for msg in self.messages
            if msg.role == "tool"
            for result in msg.tool_results or []
        }
        shown_results = _budgeted_tool_results(self.messages, results)

        lines: list[str] = []
        middle_left = MIDDLE_STEPS_TOTAL_CHARS
        middle_truncated = False
        for index, msg in enumerate(self.messages):
            if msg.role == "user":
                lines.append(f"User: {collapse_whitespace(msg.content or '')}")
                continue
            if msg.role != "assistant":
                continue
            text = collapse_whitespace(msg.content or "")
            if index in whole:
                text = text[:AGENT_STEP_MAX_CHARS]
            elif middle_truncated:
                text = ""
            elif text:
                take = min(len(text), MIDDLE_STEP_MAX_CHARS, middle_left)
                if take < len(text):
                    middle_truncated = take < MIDDLE_STEP_MAX_CHARS
                    text = text[:take] + "...(truncated)"
                middle_left -= take
            label = FINAL_RESPONSE_LABEL if index in final else "Agent:"
            line = f"{label} {text}".rstrip()
            if msg.tool_calls:
                calls = ", ".join(
                    format_tool_call(call)
                    + (" ✗" if call.id in results and results[call.id].is_error else " ✓")
                    for call in msg.tool_calls
                )
                line += f" | Tools: {calls}"
            lines.append(line)
            for call in msg.tool_calls or []:
                if call.id in shown_results:
                    lines.append(shown_results[call.id])
        return "\n".join(lines)


def _budgeted_tool_results(
    messages: list[Message], results: dict[str, ToolResult]
) -> dict[str, str]:
    """Pick tool-result excerpt lines, newest first, within separate result/error budgets."""
    calls = [call for msg in messages for call in msg.tool_calls or [] if call.id in results]
    budgets = {False: TOOL_RESULTS_TOTAL_CHARS, True: TOOL_ERRORS_TOTAL_CHARS}
    shown: dict[str, str] = {}
    for call in reversed(calls):
        result = results[call.id]
        kind = "error" if result.is_error else "result"
        line = f"  ↳ {call.name} {kind}: {excerpt(result.content, TOOL_RESULT_MAX_CHARS)}"
        if len(line) <= budgets[result.is_error]:
            budgets[result.is_error] -= len(line)
            shown[call.id] = line
    return shown


def excerpt(text: str, limit: int) -> str:
    text = collapse_whitespace(text)
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    return f"{text[:head]} …[{len(text) - limit} characters omitted]… {text[-tail:]}"


def format_tool_call(call: ToolCall) -> str:
    """Render `name(k=v, ...)`, sharing the character cap fairly across argument values."""
    values = {
        key: value if isinstance(value, str) else json.dumps(value, separators=(",", ":"))
        for key, value in sorted(call.arguments.items())
    }
    budget = TOOL_CALL_MAX_CHARS - len(call.name) - 2 - sum(len(k) + 3 for k in values)
    shares: dict[str, int] = {}
    remaining = sorted(values, key=lambda key: len(values[key]))
    while remaining:
        share = max(budget, 0) // len(remaining)
        key = remaining.pop(0)
        shares[key] = min(len(values[key]), share)
        budget -= shares[key]
    rendered = ", ".join(
        f"{key}={value if len(value) <= shares[key] else value[: max(shares[key] - 1, 0)] + '…'}"
        for key, value in values.items()
    )
    return f"{call.name}({rendered})"


def collapse_whitespace(text: str) -> str:
    return " ".join(text.split())
