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
    time_unix: float | None = None
    limit: int | None = None
    lookback_hours: float | None = None
    start_sec: int | None = None
    end_sec: int | None = None

    @model_validator(mode="after")
    def validate_query(self) -> Self:
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


class DatasourceInventoryFact(SpecModel):
    kind: Literal["resource"]
    resource: Literal["datasource_inventory"]


class DatasourceDetailFact(DatasourceSelector):
    kind: Literal["resource"]
    resource: Literal["datasource_detail"]


type ResourceFact = Annotated[
    DashboardFact | DatasourceInventoryFact | DatasourceDetailFact,
    Field(discriminator="resource"),
]
type FactSpec = Annotated[QueryFact | ResourceFact, Field(discriminator="kind")]


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


QueryExpectation.model_rebuild()


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
        if any(
            value
            for value in (
                self.title,
                self.name,
                self.type,
                self.type_any_of,
                self.datasource_type,
                self.match_count,
                self.enabled,
                self.include_all,
                self.multi,
                self.tag_keys_all,
                self.title_format_contains,
                self.text_format_contains,
                self.query,
                self.queries_all,
                self.execute_cases,
            )
        ):
            return self
        raise ValueError("Dashboard item expectations must include at least one constraint.")


class ToolTraceIdGroundingParams(SpecModel):
    mode: Literal["tool_trace_id"]
    prefix_min_chars: int = 8
    assistant_scope: Literal["final", "all"] = "final"
    tool_name: str | list[str] | None = None
    tool_name_prefix: str = "tempo_"
    additional_tool_names: str | list[str] | None = None


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
    ToolTraceIdGroundingParams
    | DashboardStateParams
    | DatasourceInventoryStateParams
    | DatasourceDetailStateParams
    | TempoTraceServiceInventoryStateParams,
    Field(discriminator="mode"),
]


class RubricItem(SpecModel):
    criterion: str
    weight: float
    fact: FactSpec | None = None


class CheckItem(SpecModel):
    name: str
    weight: float
    type: Literal["grounding", "state"]
    params: CheckParams

    @model_validator(mode="after")
    def validate_type_matches_params(self) -> Self:
        expected_type = check_type_for_params(self.params)
        if self.type != expected_type:
            raise ValueError(
                f"check type {self.type!r} does not match params mode {self.params.mode!r}"
            )
        return self


@dataclass(frozen=True)
class JudgeCriterion:
    criterion: str
    weight: float
    prompt_text: str


class Problem(SpecModel):
    id: str
    category: str
    tags: list[str] = Field(default_factory=list)
    statement: str
    rubric: list[RubricItem] = Field(default_factory=list)
    checks: list[CheckItem] = Field(default_factory=list)
    setup_dashboards: list[dict[str, Any]] = Field(default_factory=list)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    tool_call_id: str
    content: str


@dataclass
class Message:
    role: str
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_results: list[ToolResult] | None = None
    thinking_content: str | None = None


@dataclass
class Transcript:
    messages: list[Message] = field(default_factory=list)

    def _render_text(
        self,
        *,
        thinking_chars: int,
        assistant_chars: int,
        tool_result_chars: int,
        tool_arg_chars: int,
    ) -> str:
        lines: list[str] = []
        for msg in self.messages:
            match msg.role:
                case "system":
                    lines.append(f"[System]: {msg.content}")
                case "user":
                    lines.append(f"[User]: {msg.content}")
                case "assistant":
                    if msg.thinking_content:
                        thinking_preview = collapse_whitespace(msg.thinking_content)[
                            :thinking_chars
                        ]
                        if thinking_preview:
                            lines.append(f"[Assistant Thinking]: {thinking_preview}")
                    if msg.content:
                        content = collapse_whitespace(msg.content)[:assistant_chars]
                        lines.append(f"[Assistant]: {content}")
                    if msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            args_text = json.dumps(
                                tool_call.arguments,
                                separators=(",", ":"),
                                sort_keys=True,
                            )
                            if len(args_text) > tool_arg_chars:
                                args_text = args_text[:tool_arg_chars] + "..."
                            lines.append(f"[Assistant Tool Call]: {tool_call.name}({args_text})")
                case "tool":
                    if msg.tool_results:
                        for tool_result in msg.tool_results:
                            content_preview = collapse_whitespace(tool_result.content)[
                                :tool_result_chars
                            ]
                            lines.append(
                                f"[Tool Result ({tool_result.tool_call_id})]: {content_preview}"
                            )
        return "\n".join(lines)

    def to_text(self, *, max_chars: int | None = None) -> str:
        if max_chars is None:
            return self._render_text(
                thinking_chars=2000,
                assistant_chars=1_000_000,
                tool_result_chars=2000,
                tool_arg_chars=1_000_000,
            )

        render_attempts = [
            (800, 4000, 1200, 300),
            (400, 2500, 600, 200),
            (200, 1200, 300, 120),
        ]
        for thinking_chars, assistant_chars, tool_result_chars, tool_arg_chars in render_attempts:
            rendered = self._render_text(
                thinking_chars=thinking_chars,
                assistant_chars=assistant_chars,
                tool_result_chars=tool_result_chars,
                tool_arg_chars=tool_arg_chars,
            )
            if len(rendered) <= max_chars:
                return rendered

        rendered = self._render_text(
            thinking_chars=24,
            assistant_chars=120,
            tool_result_chars=48,
            tool_arg_chars=24,
        )
        if len(rendered) <= max_chars:
            return rendered

        marker = "\n\n[Transcript truncated for judge context budget]\n\n"
        available = max_chars - len(marker)
        if available <= 0:
            return marker[:max_chars]

        head_chars = (available * 3) // 4
        tail_chars = available - head_chars
        return rendered[:head_chars] + marker + rendered[-tail_chars:]


def collapse_whitespace(text: str) -> str:
    return " ".join(text.split())


def check_type_for_params(params: CheckParams) -> Literal["grounding", "state"]:
    match params:
        case ToolTraceIdGroundingParams():
            return "grounding"
        case (
            DashboardStateParams()
            | DatasourceInventoryStateParams()
            | DatasourceDetailStateParams()
            | TempoTraceServiceInventoryStateParams()
        ):
            return "state"
