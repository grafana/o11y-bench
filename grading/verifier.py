# /// script
# dependencies = [
#   "anthropic>=0.75.0",
#   "pyyaml>=6.0",
#   "promql-parser>=0.1.0",
#   "pydantic>=2.0",
# ]
# ///
"""Standalone verifier for o11y-bench Harbor tasks."""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml

from grading.checks import run_checks
from grading.env_context import load_verifier_context_from_env
from grading.judge import build_judge_criteria, evaluate_with_llm
from grading.models import Problem, Transcript
from grading.scoring import calculate_score, normalize_weights
from grading.transcript_parser import parse_transcript


def grade(
    problem: Problem,
    transcript: Transcript,
    model: str,
) -> tuple[float, dict[str, float | int], bool, dict[str, str]]:
    """Grade a transcript. Returns (score, rewards_dict, checks_passed)."""
    all_subscores: dict[str, float] = {}
    all_explanations: dict[str, str] = {}
    raw_weights: dict[str, float] = {}

    # 1. Deterministic checks (free, instant)
    checks_passed = True
    ctx = load_verifier_context_from_env()
    if problem.checks:
        v_scores, v_explanations = run_checks(problem.checks, transcript, ctx)
        for v in problem.checks:
            all_subscores[v.name] = v_scores[v.name]
            all_explanations[v.name] = v_explanations[v.name]
            raw_weights[v.name] = v.weight
            if v_scores[v.name] < 1.0:
                checks_passed = False

    # 2. LLM rubric
    judge_criteria = build_judge_criteria(problem, ctx)
    if judge_criteria:
        llm_subscores, llm_explanations = evaluate_with_llm(transcript, model, judge_criteria)
        for criterion in llm_subscores:
            all_subscores[criterion] = llm_subscores[criterion]
            all_explanations[criterion] = llm_explanations[criterion]
        for r in problem.rubric:
            raw_weights[r.criterion] = r.weight

    rubric_passed = (
        1 if all_subscores and all(score >= 1.0 for score in all_subscores.values()) else 0
    )

    # 3. Normalize weights and compute score
    all_weights = normalize_weights(raw_weights)
    score = calculate_score(all_subscores, all_weights)

    # Build rewards dict for Harbor
    rewards: dict[str, float | int] = {
        "score": score,
        "checks_passed": 1 if checks_passed else 0,
        "rubric_passed": rubric_passed,
    }
    # Add per-criterion subscores
    for criterion, subscore in all_subscores.items():
        rewards[criterion] = subscore

    return score, rewards, checks_passed, all_explanations


# ── CLI Entry Point ─────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="o11y-bench verifier for Harbor")
    parser.add_argument("--problem", required=True, help="Path to problem.yaml")
    parser.add_argument("--logs", required=True, help="Path to agent logs directory")
    parser.add_argument("--output", required=True, help="Path to verifier output directory")
    args = parser.parse_args()

    problem_path = Path(args.problem)
    logs_dir = Path(args.logs)
    output_dir = Path(args.output)

    # Load problem definition
    with open(problem_path) as f:
        data = yaml.safe_load(f)
    problem = Problem(**data)

    # Parse transcript
    print(f"Parsing transcript from {logs_dir}...")
    transcript = parse_transcript(logs_dir)
    print(f"  Found {len(transcript.messages)} messages")

    # Grade
    model = os.getenv("GRADING_MODEL", "claude-haiku-4-5-20251001")
    print(f"Grading with model={model}...")

    score, rewards, checks_passed, explanations = grade(problem, transcript, model)

    print(f"  Score: {score:.2f}")
    print(f"  Checks passed: {checks_passed}")

    # Write reward.txt — single float, standard Harbor format
    output_dir.mkdir(parents=True, exist_ok=True)
    reward_txt = output_dir / "reward.txt"
    reward_txt.write_text(str(score))
    print(f"  Wrote {reward_txt}")

    # Write scores + explanations to grading_details.json (flat: "explanation:<criterion>" keys)
    details: dict[str, Any] = dict(rewards)
    for criterion, explanation in explanations.items():
        details[f"explanation:{criterion}"] = explanation
    details_path = output_dir / "grading_details.json"
    details_path.write_text(json.dumps(details, indent=2))
    print(f"  Wrote {details_path}")


if __name__ == "__main__":
    main()
