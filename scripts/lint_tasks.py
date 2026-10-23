#!/usr/bin/env python3
"""Lint task specs for checks that grade the method instead of the outcome.

No model key needed.
"""

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from grading.models import DashboardDiffFact, Problem
from scripts.sync_tasks import DEFAULT_SPECS_ROOT, iter_spec_files

PRECISION = re.compile(
    r"(?i)\bsignificant (figures|digits)\b|\bsig\.? ?figs?\b|\brounded (up |down )?to\b"
    r"|\bdecimal (places?|digits?)\b|±|\+/-|\bplus or minus\b"
    r"|\bwithin \d+(\.\d+)? ?(%|percent|percentage points?)(?! of each other)|\broughly accurate"
)
VAGUE = re.compile(r"(?i)completes the requested outcome|\bis helpful\b|helpful suggestions?")
METHOD = re.compile(
    r"(?i)\b(calls?|called|invokes?|invoked|uses?|used|runs?)\s+`[a-z]+_[a-z_]+`"
    r"|visible in the (transcript|tool)|\btool (call|result)s?\b|\bthe transcript\b"
)
TOOL_OUTPUT = re.compile(
    r"(?i)\btool (output|response)s?\b|\bthe (query|search|tool) (output|result|response)s?\b"
    r"|\b(returned|surfaced|produced) by (a |an |the |any )?"
    r"(tool|query|search|log|trace|metric|prometheus|loki|tempo)\b|\bif a tool returns\b"
)
PROVENANCE = re.compile(
    r"(?i)\b(invent(s|ed|ing)?|fabricat\w*|hallucinat\w*|made[- ]up"
    r"|ground(ed|s|ing)? (in|on|solely)"
    r"|(backed|supported) by (a |an |the |any )?(tool|query|telemetry|data|evidence|transcript|real)"
    r"|absent from (the )?(telemetry|data|results?|tool))\b"
)
# Categories whose tasks ask for a result that a fact or saved state can back.
OUTCOME_CATEGORIES = {"prometheus_query", "loki_query", "tempo_query", "investigation"}

HINTS = {
    "precision-rule": "the judge applies one number rule; leave rounding and tolerance out",
    "vague-outcome": "name the entity, number or setting, backed with a fact",
    "method-criterion": "grade the result instead",
    "tool-output-criterion": "state the check as a fact",
    "provenance-criterion": "attach the facts the task already reads",
    "no-outcome-check": "back at least one criterion with a fact",
    "duplicate-key": "YAML keeps the last value; remove the stray key",
    "diff-without-seed": "a dashboard_diff compares with a dashboard the task seeds in setup_dashboards",
}


@dataclass(frozen=True)
class Finding:
    path: Path
    rule: str
    detail: str

    def __str__(self) -> str:
        path = (
            self.path.relative_to(Path.cwd()) if self.path.is_relative_to(Path.cwd()) else self.path
        )
        return f"{path}: {self.rule}: {self.detail} ({HINTS[self.rule]})"


def lint_problem(problem: Problem, path: Path) -> list[Finding]:
    findings: list[Finding] = []
    has_facts = any(item.fact is not None for item in problem.rubric)

    for item in problem.rubric:
        text = item.criterion
        if PRECISION.search(text):
            findings.append(Finding(path, "precision-rule", text))
        for rule, pattern in (
            ("vague-outcome", VAGUE),
            ("method-criterion", METHOD),
            ("tool-output-criterion", TOOL_OUTPUT),
        ):
            if pattern.search(text):
                findings.append(Finding(path, rule, text))
                break
        else:
            if item.fact is None and has_facts and PROVENANCE.search(text):
                findings.append(Finding(path, "provenance-criterion", text))

    seeded = {(d.get("uid"), d.get("title")) for d in problem.setup_dashboards}
    for item in problem.rubric:
        fact = item.fact
        if isinstance(fact, DashboardDiffFact) and not any(
            (fact.uid and fact.uid == uid) or (not fact.uid and fact.title == title)
            for uid, title in seeded
        ):
            findings.append(Finding(path, "diff-without-seed", fact.uid or fact.title or ""))

    if (
        problem.category in OUTCOME_CATEGORIES
        and problem.rubric
        and not has_facts
        and not problem.checks
    ):
        findings.append(Finding(path, "no-outcome-check", problem.id))
    return findings


def duplicate_keys(text: str) -> list[str]:
    """Keys repeated in one mapping; yaml.safe_load keeps the last one silently."""
    found: list[str] = []

    def walk(node: yaml.Node) -> None:
        if isinstance(node, yaml.MappingNode):
            keys = [key.value for key, _ in node.value]
            found.extend(
                f"{key} (line {node.start_mark.line + 1})"
                for key in sorted(set(keys))
                if keys.count(key) > 1
            )
        if isinstance(node, yaml.MappingNode | yaml.SequenceNode):
            for child in node.value:
                for item in child if isinstance(child, tuple) else (child,):
                    walk(item)

    walk(yaml.compose(text))
    return found


def lint_specs(specs_path: Path) -> list[Finding]:
    findings: list[Finding] = []
    for path in iter_spec_files(specs_path):
        text = path.read_text()
        findings.extend(Finding(path, "duplicate-key", key) for key in duplicate_keys(text))
        problem = Problem.model_validate(yaml.safe_load(text))
        findings.extend(lint_problem(problem, path))
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description="Lint o11y-bench task specs")
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_SPECS_ROOT)
    args = parser.parse_args()
    findings = lint_specs(args.path)
    for finding in findings:
        print(finding)
    print(f"{len(findings)} finding(s)", file=sys.stderr)
    raise SystemExit(1 if findings else 0)


if __name__ == "__main__":
    main()
