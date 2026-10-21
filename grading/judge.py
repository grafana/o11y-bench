import html
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from grading.env_context import VerifierContext
from grading.facts import FactResult, provenance, resolve_fact
from grading.models import (
    DashboardDiffFact,
    DashboardFact,
    GrafanaApiFact,
    JudgeCriterion,
    Problem,
    Transcript,
)

JUDGE_SYSTEM_PROMPT = Path(__file__).with_name("judge_system.txt").read_text().strip()
JUDGE_REQUEST_TEMPLATE = Path(__file__).with_name("judge_request.txt").read_text().strip()

# Default judge. Format: model[@effort].
DEFAULT_GRADING_MODEL = "claude-haiku-5-5@high"
JUDGE_MAX_ATTEMPTS = 3
# A hung judge call must not hold a regrade worker for the SDK's ten-minute default.
JUDGE_TIMEOUT_SEC = 300.0
JUDGE_MAX_TOKENS = 8192
REASONING_HEADROOM = {"low": 4096, "medium": 8192, "high": 16384, "xhigh": 32768, "max": 32768}
# A fact rendered at least this long is shown once; later criteria point back to it.
SHARED_TRUTH_MIN_CHARS = 300


class JudgeError(Exception):
    """The judge could not grade: an API failure or no parseable evaluation."""


@dataclass(frozen=True)
class Evaluation:
    score: float
    quote: str
    explanation: str


def grading_model() -> str:
    return os.getenv("GRADING_MODEL") or DEFAULT_GRADING_MODEL


def split_model(spec: str) -> tuple[str, str | None]:
    model, _, effort = spec.partition("@")
    return model, (effort or None)


def build_judge_criteria(
    problem: Problem,
    ctx: VerifierContext | None,
    cache: dict[str, FactResult] | None = None,
    *,
    offline: bool = False,
    answer: str = "",
) -> list[JudgeCriterion]:
    cache = {} if cache is None else cache
    shown: dict[str, int] = {}
    criteria: list[JudgeCriterion] = []
    for index, item in enumerate(problem.rubric):
        if item.fact is None:
            criteria.append(JudgeCriterion(item.criterion, item.weight, item.criterion))
            continue
        fact = resolve_fact(
            item.fact,
            ctx,
            cache,
            offline=offline,
            seeds=problem.setup_dashboards,
            answer=answer,
            example=problem.trace_example,
        )
        # Saved dashboard state is judged from the state, not from what the answer says.
        answer_backed = not isinstance(item.fact, DashboardFact | DashboardDiffFact) and not (
            isinstance(item.fact, GrafanaApiFact) and "/dashboards/" in item.fact.path
        )
        if fact.error:
            criteria.append(JudgeCriterion(item.criterion, item.weight, "", error=fact.error))
            continue
        if fact.fails:
            criteria.append(JudgeCriterion(item.criterion, item.weight, "", fails=fact.summary))
            continue
        truth = f"{fact.summary} ({provenance(item.fact)})"
        if len(truth) >= SHARED_TRUTH_MIN_CHARS and truth in shown:
            truth = f"The same data as the Source of truth of criterion {shown[truth]}."
        else:
            shown.setdefault(truth, index)
        criteria.append(
            JudgeCriterion(
                item.criterion,
                item.weight,
                f"{item.criterion}\nSource of truth:\n{truth}",
                answer_backed=answer_backed,
            )
        )
    return criteria


def build_request(transcript_text: str, criteria: dict[int, JudgeCriterion]) -> str:
    # The transcript and the truth in each criterion are untrusted: unescaped, they could close a
    # block or carry an <evaluation> the judge quotes back and the parser reads as the judge's.
    # _tag unescapes. A criterion only needs its "<" neutralised, which leaves the "&" and ">" of
    # queries and URLs in its truth as they are.
    criteria_xml = "\n".join(
        f'<criterion id="{index}">{criterion.prompt_text.replace("<", "&lt;")}</criterion>'
        for index, criterion in criteria.items()
    )
    return JUDGE_REQUEST_TEMPLATE.replace("{criteria}", criteria_xml).replace(
        "{transcript}", html.escape(transcript_text, quote=False)
    )


EVALUATION_RE = re.compile(r"(?is)<evaluation\b([^>]*)>(.*?)</evaluation>")
ID_RE = re.compile(r"""(?i)criterion_id\s*=\s*["']?(\d+)""")


def _tag(body: str, name: str) -> str | None:
    match = re.search(rf"(?is)<{name}>(.*?)</{name}>", body)
    return html.unescape(match.group(1)).strip() if match else None


def parse_evaluations(response_text: str) -> dict[int, Evaluation]:
    """Parse `<evaluation criterion_id=N>` blocks; malformed blocks are dropped."""
    evaluations: dict[int, Evaluation] = {}
    for attrs, body in EVALUATION_RE.findall(response_text):
        id_match = ID_RE.search(attrs)
        if not id_match:
            continue
        try:
            score = float(_tag(body, "score") or "")
        except ValueError:
            continue
        if score not in (0.0, 0.5, 1.0):
            continue
        evaluations[int(id_match.group(1))] = Evaluation(
            score=score,
            quote=_tag(body, "quote") or "",
            explanation=_tag(body, "explanation") or "",
        )
    return evaluations


def evaluate_with_llm(
    transcript: Transcript,
    model_spec: str,
    criteria: dict[int, JudgeCriterion],
) -> dict[int, Evaluation]:
    """One judge call for every criterion. Raises JudgeError when nothing can be graded."""
    from anthropic import Anthropic, APIError, BadRequestError

    model, effort = split_model(model_spec)
    request = build_request(transcript.to_text(), criteria)
    params: dict[str, object] = {
        "model": model,
        "max_tokens": JUDGE_MAX_TOKENS + REASONING_HEADROOM.get(effort or "", 0),
        "system": [
            {"type": "text", "text": JUDGE_SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}
        ],
    }
    if effort == "none":
        params["thinking"] = {"type": "disabled"}
    elif effort:
        params["output_config"] = {"effort": effort}

    client = Anthropic(timeout=JUDGE_TIMEOUT_SEC)
    content = request
    last_error = ""
    for attempt in range(1, JUDGE_MAX_ATTEMPTS + 1):
        try:
            with client.messages.stream(
                messages=[{"role": "user", "content": content}],
                **params,  # type: ignore[arg-type]
            ) as stream:
                response = stream.get_final_message()
        except BadRequestError as exc:
            raise JudgeError(f"judge request rejected: {exc}") from exc
        except APIError as exc:
            last_error = f"judge API error: {exc}"
        else:
            text = "".join(block.text for block in response.content if block.type == "text")
            evaluations = parse_evaluations(text)
            if evaluations:
                return evaluations
            last_error = "judge returned no parseable <evaluation> blocks"
            content = (
                f"{request}\n\nRetry instruction: your previous reply could not be parsed. "
                "Return ONLY complete <evaluation criterion_id=...> XML blocks, one per "
                f"criterion. Start of the previous reply: {text[:300]}"
            )
        if attempt < JUDGE_MAX_ATTEMPTS:
            time.sleep(0.25 * attempt)
    raise JudgeError(last_error)
