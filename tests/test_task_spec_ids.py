"""Task id discovery and task-spec validation."""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from grading.models import CheckItem, DashboardItemExpectation, Problem, RubricItem
from scripts.sync_tasks import task_spec_ids

ROOT = Path(__file__).resolve().parents[1]


def test_task_spec_ids_are_sorted_unique_and_match_cli_output() -> None:
    specs = ROOT / "tasks-spec"
    yaml_count = len(list(specs.rglob("*.yaml")))
    ids = task_spec_ids(specs)

    assert len(ids) == yaml_count
    assert len(ids) == len(set(ids)), "duplicate task id in specs"
    assert ids == sorted(ids)
    result = subprocess.run(
        [sys.executable, "-m", "scripts.sync_tasks", "--list-ids"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stderr == ""

    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert lines == ids


def test_all_task_specs_load_as_problem_models() -> None:
    spec_files = sorted((ROOT / "tasks-spec").rglob("*.yaml"))
    assert spec_files
    for spec_file in spec_files:
        data = yaml.safe_load(spec_file.read_text())
        Problem.model_validate(data)


def test_check_model_rejects_unknown_type() -> None:
    with pytest.raises(ValidationError) as excinfo:
        CheckItem.model_validate(
            {
                "name": "bad check",
                "weight": 1.0,
                "type": "not_a_check",
                "params": {},
            }
        )
    message = str(excinfo.value)
    assert "state" in message


def test_rubric_prometheus_fact_rejects_invalid_promql() -> None:
    with pytest.raises(ValidationError) as excinfo:
        RubricItem.model_validate(
            {
                "criterion": "Bad fact",
                "weight": 1.0,
                "fact": {
                    "kind": "query",
                    "backend": "prometheus",
                    "query": "sum(",
                },
            }
        )
    assert "Invalid PromQL fact query" in str(excinfo.value)


def test_rubric_item_rejects_unknown_fact_kind() -> None:
    with pytest.raises(ValidationError) as excinfo:
        RubricItem.model_validate(
            {
                "criterion": "Bad fact",
                "weight": 1.0,
                "fact": {"kind": "mystery"},
            }
        )
    message = str(excinfo.value)
    assert "query" in message
    assert "resource" in message


def test_a_cited_trace_fact_needs_the_one_check_that_defines_the_example() -> None:
    check = {
        "name": "cites a failing trace",
        "weight": 1,
        "type": "state",
        "params": {"mode": "cited_trace", "match": "{ status = error }"},
    }
    fact = {"criterion": "Places the error origin.", "weight": 1, "fact": {"kind": "cited_trace"}}
    base = {"id": "demo", "category": "tempo_query", "statement": "x"}
    with pytest.raises(ValidationError, match="needs a cited_trace check"):
        Problem.model_validate({**base, "rubric": [fact]})
    with pytest.raises(ValidationError, match="at most one cited_trace check"):
        Problem.model_validate({**base, "checks": [check, {**check, "name": "again"}]})
    assert Problem.model_validate({**base, "checks": [check], "rubric": [fact]}).trace_example


def _query(backend: str, **extra: object) -> dict:
    return {"kind": "query", "backend": backend, "query": "{}", **extra}


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        pytest.param(
            {"rubric": [{"criterion": "c", "weight": 0}]}, "greater than 0", id="zero-weight"
        ),
        pytest.param(
            {"rubric": [{"criterion": "c", "weight": 1}, {"criterion": "c", "weight": 1}]},
            "must be unique",
            id="repeated-name",
        ),
        pytest.param(
            {"rubric": [{"criterion": "c", "weight": 1, "fact": _query("tempo", time_unix=1)}]},
            "time_unix applies to prometheus and loki",
            id="time-on-tempo",
        ),
        pytest.param(
            {"rubric": [{"criterion": "c", "weight": 1, "fact": _query("loki", limit=3)}]},
            "apply to tempo facts",
            id="limit-on-loki",
        ),
    ],
)
def test_a_spec_that_would_silently_misgrade_cannot_be_written(
    overrides: dict, message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        Problem.model_validate({"id": "demo", "category": "x", "statement": "x", **overrides})


def test_a_dashboard_expectation_may_require_a_falsy_value() -> None:
    DashboardItemExpectation.model_validate({"name": "service", "multi": False})
    with pytest.raises(ValidationError, match="at least one constraint"):
        DashboardItemExpectation.model_validate({})
