import os
import re
from pathlib import Path

from grading.env_context import VerifierContext
from grading.facts import FactResult, render_fact_summary_for_criterion, resolve_fact
from grading.models import JudgeCriterion, Problem, Transcript

GRADER_PROMPT_TEMPLATE = Path(__file__).with_name("grader_prompt.txt").read_text().strip()
JUDGE_TRANSCRIPT_CHAR_BUDGETS = (180_000, 120_000, 80_000)


def build_evaluation_prompt(criteria: list[JudgeCriterion]) -> str:
    criteria_text = "\n".join(
        f'<criterion id="{index}">{criterion.prompt_text}</criterion>'
        for index, criterion in enumerate(criteria)
    )
    return GRADER_PROMPT_TEMPLATE.format(criteria_text=criteria_text)


def build_judge_criteria(
    problem: Problem,
    ctx: VerifierContext | None,
) -> list[JudgeCriterion]:
    if ctx is None:
        return [
            JudgeCriterion(
                criterion=item.criterion,
                weight=item.weight,
                prompt_text=item.criterion,
            )
            for item in problem.rubric
        ]

    cache: dict[str, FactResult] = {}
    criteria: list[JudgeCriterion] = []
    for item in problem.rubric:
        prompt_text = item.criterion
        if item.fact is not None:
            fact_result = resolve_fact(item.fact, ctx, cache)
            summary = render_fact_summary_for_criterion(fact_result, item.criterion)
            prompt_text = f"{prompt_text}\nSource of truth: {summary}"
        criteria.append(
            JudgeCriterion(
                criterion=item.criterion,
                weight=item.weight,
                prompt_text=prompt_text,
            )
        )
    return criteria


def parse_evaluation_response(
    response_text: str,
    criteria: list[JudgeCriterion],
) -> tuple[dict[str, float], dict[str, str]]:
    pattern = (
        r'<evaluation id="(\d+)">\s*<answer>(YES|NO)</answer>'
        r"\s*<explanation>(.*?)</explanation>\s*</evaluation>"
    )
    evaluations = re.findall(pattern, response_text, re.DOTALL | re.IGNORECASE)
    eval_map = {
        int(eval_id): (answer.upper(), explanation.strip())
        for eval_id, answer, explanation in evaluations
    }

    subscores: dict[str, float] = {}
    explanations: dict[str, str] = {}
    for index, criterion in enumerate(criteria):
        answer, explanation = eval_map.get(index, ("NO", "No evaluation found in response"))
        subscores[criterion.criterion] = 1.0 if answer == "YES" else 0.0
        explanations[criterion.criterion] = explanation
    return subscores, explanations


def evaluate_with_llm(
    transcript: Transcript,
    model: str,
    criteria: list[JudgeCriterion],
) -> tuple[dict[str, float], dict[str, str]]:
    from anthropic import Anthropic, BadRequestError

    evaluation_prompt = build_evaluation_prompt(criteria)
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    last_error: BadRequestError | None = None

    for budget in JUDGE_TRANSCRIPT_CHAR_BUDGETS:
        transcript_text = transcript.to_text(max_chars=budget)
        try:
            response = client.messages.create(
                model=model,
                max_tokens=8000,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": f"<transcript>\n{transcript_text}\n</transcript>\n\n{evaluation_prompt}",
                    }
                ],
            )
            response_text = getattr(response.content[0], "text", "") if response.content else ""
            return parse_evaluation_response(response_text, criteria)
        except BadRequestError as exc:
            if "prompt is too long" not in str(exc):
                raise
            last_error = exc

    if last_error is not None:
        raise last_error
    raise RuntimeError("LLM evaluation failed without a captured error")
