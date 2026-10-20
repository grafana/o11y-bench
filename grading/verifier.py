"""Standalone verifier for o11y-bench Harbor tasks."""

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from grading.checks import run_checks
from grading.env_context import VerifierContext, load_verifier_context_from_env
from grading.facts import FactResult, dump_truth, load_truth
from grading.judge import (
    Evaluation,
    JudgeError,
    build_judge_criteria,
    evaluate_with_llm,
    grading_model,
)
from grading.models import CheckItem, JudgeCriterion, Problem, Transcript
from grading.scoring import calculate_score, normalize_weights
from grading.transcript_parser import parse_transcript

# Why an item passed or failed.
OUTCOME_PASS = "pass"
OUTCOME_WRONG = "wrong"
OUTCOME_NOT_STATED = "not_stated"
OUTCOME_NO_ANSWER = "no_answer"
OUTCOME_UNGROUNDED = "ungrounded"
OUTCOME_ERROR_TRUTH = "error/truth"
OUTCOME_ERROR_JUDGE = "error/judge"
OUTCOME_ERROR_CHECK = "error/check"

type Status = Literal["passed", "failed", "error"]


@dataclass
class ItemResult:
    name: str
    weight: float
    score: float
    outcome: str
    explanation: str
    quote: str = ""

    @property
    def passed(self) -> bool:
        return self.score >= 1.0


@dataclass
class GradeResult:
    score: float
    status: Status
    model: str
    checks: list[ItemResult] = field(default_factory=list)
    rubric: list[ItemResult] = field(default_factory=list)
    truth: dict[str, FactResult] = field(default_factory=dict)

    @property
    def items(self) -> list[ItemResult]:
        return self.checks + self.rubric

    def details(self) -> dict[str, Any]:
        """grading_details.json: summary keys, then per item `<name>`, `outcome:`, `explanation:`, `quote:`."""
        details: dict[str, Any] = {
            "score": self.score,
            "status": self.status,
            "grading_model": self.model,
            "checks_passed": int(all(item.passed for item in self.checks)),
            "rubric_passed": int(bool(self.items) and all(item.passed for item in self.items)),
        }
        for item in self.items:
            details[item.name] = item.score
        for item in self.items:
            details[f"outcome:{item.name}"] = item.outcome
            details[f"explanation:{item.name}"] = item.explanation
            if item.quote:
                details[f"quote:{item.name}"] = item.quote
        return details


QUOTE_NOISE = re.compile("[*`_\"\u201c\u201d'\u2018\u2019]")


def _normalize_quote_text(text: str) -> str:
    return " ".join(QUOTE_NOISE.sub("", text).split()).lower()


def quote_in_answer(quote: str, answer: str) -> bool:
    q = _normalize_quote_text(quote)
    return bool(q) and q in _normalize_quote_text(answer)


def criterion_outcome(criterion: JudgeCriterion, evaluation: Evaluation | None, answer: str) -> str:
    if criterion.error:
        return OUTCOME_ERROR_TRUTH
    if criterion.fails:
        return OUTCOME_WRONG
    if evaluation is None:
        return OUTCOME_ERROR_JUDGE
    passed = evaluation.score >= 1.0
    if not answer.strip() and not passed:
        return OUTCOME_NO_ANSWER
    if criterion.answer_backed:
        quote = evaluation.quote.strip()
        # A quote the answer doesn't contain is flagged even on a pass; the score stands.
        if quote and not quote_in_answer(quote, answer):
            return OUTCOME_UNGROUNDED
        if not quote and not passed:
            return OUTCOME_NOT_STATED
    return OUTCOME_PASS if passed else OUTCOME_WRONG


def grade_checks(
    checks: list[CheckItem],
    transcript: Transcript,
    ctx: VerifierContext | None,
    previous: dict[str, Any] | None,
) -> list[ItemResult]:
    """Run checks. With `previous` (an offline regrade), checks reuse their saved result."""
    live = checks if previous is None else []
    scores, explanations = run_checks(live, transcript, ctx)
    results: list[ItemResult] = []
    for check in checks:
        if check.name in scores:
            score, explanation = scores[check.name], explanations[check.name]
            outcome = OUTCOME_PASS if score >= 1.0 else OUTCOME_WRONG
        elif previous is not None and isinstance(previous.get(check.name), int | float):
            score = float(previous[check.name])
            explanation = str(previous.get(f"explanation:{check.name}", ""))
            outcome = OUTCOME_PASS if score >= 1.0 else OUTCOME_WRONG
        else:
            score, explanation = 0.0, "No saved state-check result to reuse offline."
            outcome = OUTCOME_ERROR_CHECK
        results.append(ItemResult(check.name, check.weight, score, outcome, explanation))
    return results


def grade(
    problem: Problem,
    transcript: Transcript,
    model: str,
    *,
    ctx: VerifierContext | None = None,
    truth: dict[str, FactResult] | None = None,
    previous: dict[str, Any] | None = None,
) -> GradeResult:
    """Grade a transcript.

    Live (default): reads the stack from the environment. Offline (`previous` given): facts
    come only from `truth` (a saved truth.json) and state checks reuse `previous` results.
    """
    offline = previous is not None
    if ctx is None and not offline:
        ctx = load_verifier_context_from_env()
    cache: dict[str, FactResult] = dict(truth or {})

    checks = grade_checks(problem.checks, transcript, ctx, previous)

    answer = transcript.final_answer()
    criteria = build_judge_criteria(problem, ctx, cache, offline=offline, answer=answer)
    judged = {index: c for index, c in enumerate(criteria) if not c.error and not c.fails}
    evaluations: dict[int, Evaluation] = {}
    judge_error = ""
    if judged:
        try:
            evaluations = evaluate_with_llm(transcript, model, judged)
        except JudgeError as exc:
            judge_error = str(exc)

    rubric: list[ItemResult] = []
    for index, criterion in enumerate(criteria):
        evaluation = evaluations.get(index)
        outcome = criterion_outcome(criterion, evaluation, answer)
        if criterion.error:
            explanation = f"Could not read the truth: {criterion.error}"
        elif criterion.fails:
            explanation = criterion.fails
        elif evaluation is None:
            explanation = judge_error or "The judge returned no evaluation for this criterion."
        else:
            explanation = evaluation.explanation
        rubric.append(
            ItemResult(
                name=criterion.criterion,
                weight=criterion.weight,
                score=evaluation.score if evaluation else 0.0,
                outcome=outcome,
                explanation=explanation,
                quote=evaluation.quote if evaluation else "",
            )
        )

    items = checks + rubric
    weights = normalize_weights({item.name: item.weight for item in items})
    score = calculate_score({item.name: item.score for item in items}, weights)
    status: Status
    if judge_error or any(item.outcome.startswith("error/") for item in items):
        status = "error"
    elif items and all(item.passed for item in items):
        status = "passed"
    else:
        status = "failed"
    return GradeResult(score, status, model, checks, rubric, cache)


def write_outputs(output_dir: Path, result: GradeResult) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "reward.txt").write_text(str(result.score))
    (output_dir / "grading_details.json").write_text(json.dumps(result.details(), indent=2))
    (output_dir / "truth.json").write_text(json.dumps(dump_truth(result.truth), indent=2))


def read_truth(output_dir: Path) -> dict[str, FactResult] | None:
    path = output_dir / "truth.json"
    if not path.exists():
        return None
    return load_truth(json.loads(path.read_text()))


# ── CLI Entry Point ─────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="o11y-bench verifier for Harbor")
    parser.add_argument("--problem", required=True, help="Path to problem.yaml")
    parser.add_argument("--logs", required=True, help="Path to agent logs directory")
    parser.add_argument("--output", required=True, help="Path to verifier output directory")
    args = parser.parse_args()

    problem = Problem(**yaml.safe_load(Path(args.problem).read_text()))

    print(f"Parsing transcript from {args.logs}...")
    transcript = parse_transcript(Path(args.logs))
    print(f"  Found {len(transcript.messages)} messages")

    model = grading_model()
    print(f"Grading with model={model}...")
    result = grade(problem, transcript, model)
    print(f"  Score: {result.score:.2f} ({result.status})")

    write_outputs(Path(args.output), result)
    print(f"  Wrote {args.output}")


if __name__ == "__main__":
    main()
