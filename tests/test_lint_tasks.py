from pathlib import Path

import pytest

from grading.models import Problem
from scripts.lint_tasks import lint_problem, lint_specs
from scripts.sync_tasks import DEFAULT_SPECS_ROOT


def _rules(**spec) -> list[str]:
    problem = Problem(id="t", statement=spec.pop("statement", "q"), **spec)
    return [finding.rule for finding in lint_problem(problem, Path("t.yaml"))]


FACT = {"kind": "query", "backend": "prometheus", "query": "up"}


def test_task_specs_are_lint_clean() -> None:
    assert [str(finding) for finding in lint_specs(DEFAULT_SPECS_ROOT)] == []


# A sibling criterion backed by a fact keeps the task itself from tripping no-outcome-check, and
# is what makes an unbacked criterion about provenance a finding.
SIBLING = {"criterion": "Reports the error count.", "weight": 1, "fact": FACT}


@pytest.mark.parametrize(
    ("criterion", "statement", "backed", "rules"),
    [
        ("Names the slowest service.", "q", False, []),
        ("The transcript shows a rate() query.", "q", True, ["method-criterion"]),
        ("The response is helpful.", "q", False, ["vague-outcome"]),
        ("Quotes the tool output verbatim.", "q", False, ["tool-output-criterion"]),
        ("Does not invent a service that is absent.", "q", False, ["provenance-criterion"]),
        ("Does not invent a service that is absent.", "q", True, []),
        ("States the share within 5% of the truth.", "q", True, ["precision-rule"]),
        ("States the share within 5% of the truth.", "Roughly what share?", True, []),
        ("States the share roughly accurately.", "Roughly?", True, ["precision-rule"]),
    ],
    ids=[
        "clean",
        "method",
        "vague",
        "tool-output",
        "provenance",
        "provenance-backed-by-a-fact",
        "tolerance-without-a-rough-ask",
        "tolerance-for-a-rough-ask",
        "vague-tolerance-even-for-a-rough-ask",
    ],
)
def test_criteria_rules(criterion: str, statement: str, backed: bool, rules: list[str]) -> None:
    item = {"criterion": criterion, "weight": 1, **({"fact": FACT} if backed else {})}

    assert _rules(category="prometheus_query", statement=statement, rubric=[item, SIBLING]) == rules


def test_result_tasks_need_a_fact_or_saved_state() -> None:
    rubric = [{"criterion": "Names the slowest service.", "weight": 1}]
    assert _rules(category="tempo_query", rubric=rubric) == ["no-outcome-check"]
    assert _rules(category="dashboarding", rubric=rubric) == []


def test_dashboard_diff_needs_a_seeded_dashboard() -> None:
    rubric = [
        {
            "criterion": "Only the target panel changed.",
            "weight": 1,
            "fact": {"kind": "resource", "resource": "dashboard_diff", "uid": "svc"},
        }
    ]
    assert _rules(category="dashboarding", rubric=rubric) == ["diff-without-seed"]
    seeded = [{"uid": "svc", "title": "Svc", "panels": []}]
    assert _rules(category="dashboarding", rubric=rubric, setup_dashboards=seeded) == []


def test_flags_a_criterion_left_with_two_weights(tmp_path: Path) -> None:
    # Deleting a criterion line but not its weight attaches the weight to the item above.
    spec = tmp_path / "t.yaml"
    spec.write_text(
        "id: t\ncategory: dashboarding\nstatement: q\nrubric:\n"
        "- criterion: Names the slowest service.\n  weight: 30\n  weight: 10\n"
    )
    assert [(f.rule, f.detail) for f in lint_specs(tmp_path)] == [
        ("duplicate-key", "weight (line 5)")
    ]
