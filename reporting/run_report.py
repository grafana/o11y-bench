#!/usr/bin/env python3
# /// script
# dependencies = []
# ///
"""Generate a per-run HTML report for a Harbor benchmark job.

Reads a single Harbor job directory and produces a self-contained HTML report
with per-problem details, transcripts, and grading criteria.

Usage:
    uv run python -m reporting.run_report --job-dir jobs/2026-03-18__08-06-52 --output run_report.html
    uv run python -m reporting.run_report --job-dir jobs/2026-03-18__08-06-52  # writes run_report.html in job dir
"""

import argparse
import base64
import gzip
import json
import os
import sys
from pathlib import Path
from typing import Any, TypedDict

import markdown

from .categories import category_label
from .report_data import (
    agent_result_metrics,
    agent_seconds,
    classify_trial_artifact,
    count_tool_calls,
    escape_html,
    format_cost,
    format_duration,
    grading_counts_as_pass,
    is_invalid_infra_trial,
    load_task_categories,
    resolve_tasks_dir,
    score_bar_class,
    score_color_class,
    trial_model_display,
)
from .report_paths import run_report_output_path

JsonDict = dict[str, Any]


class CategoryStats(TypedDict):
    total: int
    passed: int
    consistent: int
    scores: list[float]


def compress_text(text: str) -> str:
    return base64.b64encode(gzip.compress(text.encode())).decode()


def status_badge_html(full_passed: bool, *, has_error: bool = False) -> str:
    if has_error:
        return '<span class="px-2 py-0.5 rounded text-xs font-medium bg-red-100 text-red-800">error</span>'
    if full_passed:
        return '<span class="px-2 py-0.5 rounded text-xs font-medium bg-green-100 text-green-800">pass</span>'
    return (
        '<span class="px-2 py-0.5 rounded text-xs font-medium bg-red-100 text-red-800">fail</span>'
    )


def invalid_badge_html() -> str:
    return '<span class="px-2 py-0.5 rounded text-xs font-medium bg-amber-100 text-amber-800">invalid</span>'


def pass_label_html(num_passed: int, total: int) -> str:
    if num_passed == total:
        color = "text-green-600"
    elif num_passed > 0:
        color = "text-yellow-600"
    else:
        color = "text-red-600"
    return f'<span class="{color} font-medium">{num_passed}/{total} passed</span>'


def collapse_whitespace(text: str) -> str:
    return " ".join(text.split())


def _tool_result_plaintext(content: object) -> str:
    if isinstance(content, list):
        return " ".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    return str(content)


# ---------------------------------------------------------------------------
# Transcript rendering (Harbor JSONL format)
# ---------------------------------------------------------------------------


def render_transcript(messages: list[JsonDict]) -> str:
    """Render Harbor-format transcript JSONL messages to HTML.

    Harbor message format:
      {"type": "system", "message": "..."}
      {"type": "user", "message": "..."}
      {"type": "assistant", "message": {"content": [{"type": "text", ...}, {"type": "tool_use", ...}]}}
      {"type": "tool_result", "tool_use_id": "...", "content": "..."}
    """
    parts: list[str] = []
    in_agent_section = False

    tc_to_result: dict[str, str] = {}
    for msg in messages:
        if msg.get("type") == "tool_result":
            tid = msg.get("tool_use_id", "")
            tc_to_result[tid] = _tool_result_plaintext(msg.get("content", ""))

    for msg in messages:
        msg_type = msg.get("type")

        if msg_type == "system":
            continue
        if msg_type == "tool_result":
            continue

        if msg_type == "user":
            raw = msg.get("message", "")
            if isinstance(raw, dict):
                raw = " ".join(
                    c.get("text", "")
                    for c in raw.get("content", [])
                    if isinstance(c, dict) and c.get("type") == "text"
                )
            parts.append(
                '<div class="border-b border-gray-200 pb-2 mb-2">'
                '<div class="text-[10px] font-semibold text-gray-400 tracking-wider mb-1.5">SCENARIO INPUT</div>'
                '<div class="pl-3 border-l-2 border-blue-300">'
                f'<span class="font-medium text-blue-700 text-xs">User:</span> '
                f'<span class="text-gray-700 text-xs">{escape_html(str(raw))}</span>'
                "</div></div>"
            )
            in_agent_section = False
            continue

        if msg_type == "assistant":
            if not in_agent_section:
                parts.append(
                    '<div class="text-[10px] font-semibold text-gray-400 tracking-wider mb-1.5">AGENT EXECUTION</div>'
                )
                in_agent_section = True

            raw = msg.get("message", {})
            content_blocks = []
            if isinstance(raw, dict):
                content_blocks = raw.get("content", [])
            elif isinstance(raw, str):
                content_blocks = [{"type": "text", "text": raw}]

            agent_parts = ['<div class="pl-3 border-l-2 border-green-300 mb-2">']
            agent_parts.append('<span class="font-medium text-green-700 text-xs">Agent:</span>')

            for block in content_blocks:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")

                if btype == "thinking":
                    thinking = block.get("thinking", "")
                    collapsed = collapse_whitespace(thinking)
                    if len(collapsed) > 300:
                        collapsed = collapsed[:300] + "..."
                    agent_parts.append(
                        f'<details class="mt-1">'
                        f'<summary class="text-[10px] text-purple-400 cursor-pointer hover:text-purple-600">'
                        f"Thinking ({len(thinking):,} chars)</summary>"
                        f'<div class="mt-1 px-2 py-1 bg-purple-50 rounded text-[11px] text-purple-600">'
                        f"{escape_html(collapsed)}</div></details>"
                    )

                elif btype == "text":
                    text = block.get("text", "")
                    if text.strip():
                        body = markdown.markdown(
                            text,
                            extensions=("tables", "fenced_code", "sane_lists"),
                            output_format="html",
                        )
                        agent_parts.append(
                            '<div class="agent-markdown mt-1 text-gray-700 text-xs leading-relaxed">'
                            f"{body}</div>"
                        )

                elif btype == "tool_use":
                    tc_name = block.get("name", "")
                    tc_id = block.get("id", "")
                    tc_input = block.get("input", {})
                    args_compact = json.dumps(tc_input, separators=(",", ":"))
                    if len(args_compact) > 150:
                        args_compact = args_compact[:150] + "..."

                    result_content = tc_to_result.get(tc_id, "")
                    result_html = ""
                    if result_content:
                        preview = result_content[:2000]
                        truncated = " ..." if len(result_content) > 2000 else ""
                        result_html = (
                            f'<details class="mt-0.5">'
                            f'<summary class="text-[10px] text-gray-400 cursor-pointer hover:text-gray-500">'
                            f"Result ({len(result_content):,} chars)</summary>"
                            f'<pre class="text-[10px] text-gray-400 mt-0.5 whitespace-pre-wrap">'
                            f"{escape_html(preview)}{truncated}</pre></details>"
                        )

                    agent_parts.append(
                        f'<div class="mt-1 px-2 py-1 bg-gray-100 rounded text-[11px] text-gray-500 font-mono">'
                        f"{escape_html(tc_name)}({escape_html(args_compact)})"
                        f"{result_html}"
                        f"</div>"
                    )

            agent_parts.append("</div>")
            parts.append("\n".join(agent_parts))

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Criteria rendering (from grading_details.json)
# ---------------------------------------------------------------------------


def render_criteria(grading: JsonDict) -> str:
    """Render grading criteria as checkmark list with explanations."""
    items: list[str] = []
    for key, val in grading.items():
        if key in (
            "score",
            "checks_passed",
            "rubric_passed",
            "validators_passed",
            "criteria_passed",
        ) or key.startswith("explanation:"):
            continue
        score = float(val)
        icon = (
            '<span class="text-green-600">&#10003;</span>'
            if score >= 1.0
            else '<span class="text-yellow-600">&#9679;</span>'
            if score > 0
            else '<span class="text-red-600">&#10007;</span>'
        )
        explanation = grading.get(f"explanation:{key}", "")
        explanation_html = (
            f'<div class="text-gray-400 mt-0.5">{escape_html(explanation)}</div>'
            if explanation
            else ""
        )
        items.append(
            f'<div class="flex items-start gap-2 text-xs">'
            f"{icon}"
            f'<div class="flex-1"><span class="text-gray-700">{escape_html(key)}</span>{explanation_html}</div>'
            f'<span class="text-gray-300 text-[10px] shrink-0">{score:.0%}</span>'
            f"</div>"
        )
    return '<div class="space-y-1.5">' + "\n".join(items) + "</div>"


def _load_atif_trajectory(path: Path) -> list[JsonDict]:
    """Convert an ATIF trajectory.json into the transcript format the report expects."""
    data = json.loads(path.read_text())
    steps = data.get("steps", [])
    messages: list[JsonDict] = []
    for step in steps:
        source = step.get("source", "")
        message_text = step.get("message", "")
        tool_calls = step.get("tool_calls", [])
        observation = step.get("observation")
        reasoning = step.get("reasoning_content")

        if source == "user":
            messages.append({"type": "user", "message": message_text or ""})
        elif source in ("agent", "system"):
            if source == "system":
                messages.append({"type": "system", "message": message_text or ""})
                continue
            content_blocks: list[JsonDict] = []
            if reasoning:
                content_blocks.append({"type": "thinking", "thinking": reasoning})
            if message_text and message_text != "(tool use)":
                content_blocks.append({"type": "text", "text": message_text})
            for tc in tool_calls:
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.get("tool_call_id", tc.get("id", "")),
                        "name": tc.get("function_name", tc.get("name", "")),
                        "input": tc.get("arguments", {}),
                    }
                )
            messages.append({"type": "assistant", "message": {"content": content_blocks}})
            if observation and "results" in observation:
                for result in observation["results"]:
                    messages.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": result.get(
                                "source_call_id", result.get("tool_call_id", "")
                            ),
                            "content": result.get("content", ""),
                        }
                    )
    return messages


def _prepend_instruction(messages: list[JsonDict], trial_dir: Path) -> list[JsonDict]:
    """Prepend the task instruction as a user message if the transcript lacks one."""
    if any(m.get("type") == "user" for m in messages):
        return messages
    config_path = trial_dir / "config.json"
    if not config_path.exists():
        return messages
    try:
        config = json.loads(config_path.read_text())
        task_path = Path(config.get("task", {}).get("path", ""))
        instruction_path = task_path / "instruction.md"
        if instruction_path.exists():
            user_msg = {"type": "user", "message": instruction_path.read_text().strip()}
            return [user_msg, *messages]
    except Exception:
        pass
    return messages


def load_transcript(trial_dir: Path) -> list[JsonDict]:
    """Load transcript from ATIF trajectory.json."""
    trajectory = trial_dir / "agent" / "trajectory.json"
    if trajectory.exists():
        try:
            return _prepend_instruction(_load_atif_trajectory(trajectory), trial_dir)
        except Exception:
            pass
    return []


def load_trials(job_dir: Path) -> list[JsonDict]:
    """Load all trials from a job directory."""
    trials: list[JsonDict] = []
    result_paths: list[Path] = []
    for root, dirs, files in os.walk(job_dir, followlinks=True):
        dirs[:] = [name for name in dirs if not name.startswith(".")]
        for name in files:
            if name == "result.json":
                result_paths.append(Path(root) / name)
    for result_path in sorted(result_paths):
        # Skip any top-level job-level result.json (depth < 2 relative to job_dir)
        rel = result_path.relative_to(job_dir)
        if len(rel.parts) < 2:
            continue
        trial_dir = result_path.parent
        try:
            result = json.loads(result_path.read_text())
        except Exception:
            continue
        result["__result_path"] = str(result_path.resolve())
        if classify_trial_artifact(trial_dir, result) != "complete":
            continue

        grading_path = trial_dir / "verifier" / "grading_details.json"
        if not grading_path.exists():
            continue  # trial failed before verifier ran
        trials.append(
            {
                "result": result,
                "transcript": load_transcript(trial_dir),
                "grading": json.loads(grading_path.read_text()),
                "result_path": result_path,
            }
        )
    return trials


# ---------------------------------------------------------------------------
# HTML rendering for a single shot/trial
# ---------------------------------------------------------------------------


def render_trial_detail(trial: JsonDict) -> str:
    result = trial["result"]
    grading = trial["grading"]
    transcript = trial["transcript"]

    score = grading.get("score", 0.0)
    full_passed = grading_counts_as_pass(result, grading)
    has_error = result.get("exception_info") is not None

    cost_usd, n_in, n_cache, n_out = agent_result_metrics(result)
    token_str = f"in={n_in:,} cache={n_cache:,} out={n_out:,}"

    duration_s = agent_seconds(result)

    criteria_html = render_criteria(grading)
    transcript_html = render_transcript(transcript)

    error_html = ""
    if has_error:
        exc = result.get("exception_info", {})
        err_text = exc.get("message", str(exc)) if isinstance(exc, dict) else str(exc)
        error_html = (
            '<div class="mt-3 p-2 bg-red-50 rounded text-xs">'
            f'<pre class="text-red-700 whitespace-pre-wrap">{escape_html(err_text)}</pre>'
            "</div>"
        )

    # Count tool calls in transcript
    tool_count = count_tool_calls(transcript)

    return f"""
            {error_html}
            <div class="flex items-center justify-between mt-3 mb-3">
              <div class="flex items-center gap-2">
                {status_badge_html(full_passed, has_error=has_error)}
                <span class="text-xs text-gray-500">{tool_count} tool calls · {format_duration(duration_s)} · {format_cost(cost_usd)}</span>
              </div>
              <span class="text-sm font-semibold {score_color_class(score)}">{score:.0%}</span>
            </div>
            <div class="mb-3">
              <div class="text-xs font-medium text-gray-600 mb-1.5">Assertions:</div>
              {criteria_html}
            </div>
            <details class="mt-3" open>
              <summary class="text-xs text-gray-500 cursor-pointer hover:text-gray-700">
                Conversation
              </summary>
              <div class="mt-2 p-3 bg-gray-50 rounded max-h-[800px] overflow-y-auto">
                {transcript_html}
              </div>
            </details>
            <div class="mt-3 pt-2 border-t border-gray-100 text-xs text-gray-400">
              Tokens: {token_str}
            </div>"""


# ---------------------------------------------------------------------------
# Main report generation
# ---------------------------------------------------------------------------


def generate_report(job_dir: Path, tasks_dir: Path | None = None) -> str:
    resolved_tasks_dir = resolve_tasks_dir(job_dir, tasks_dir)
    categories: dict[str, str] = {}
    if resolved_tasks_dir:
        categories = load_task_categories(resolved_tasks_dir)

    trials = load_trials(job_dir)
    if not trials:
        print(f"No trials found in {job_dir}", file=sys.stderr)
        sys.exit(1)

    first_result = trials[0]["result"]
    model_display = trial_model_display(first_result, trials[0]["result_path"].resolve())
    job_id = first_result.get("config", {}).get("job_id", job_dir.name)
    job_timestamp = job_dir.name

    # Group trials by task_name (for multi-shot)
    tasks: dict[str, list[JsonDict]] = {}
    for trial in trials:
        task_name = trial["result"].get("task_name", "unknown")
        tasks.setdefault(task_name, []).append(trial)
    valid_trials = [t for t in trials if not is_invalid_infra_trial(t["result"])]

    # Compute summary stats
    total_tasks = sum(
        1
        for shot_list in tasks.values()
        if any(not is_invalid_infra_trial(t["result"]) for t in shot_list)
    )
    trial_metrics = [agent_result_metrics(t["result"]) for t in valid_trials]
    total_cost = sum(m[0] for m in trial_metrics)
    total_tokens_in = sum(m[1] for m in trial_metrics)
    total_tokens_cache = sum(m[2] for m in trial_metrics)
    total_tokens_out = sum(m[3] for m in trial_metrics)
    total_tokens = total_tokens_in + total_tokens_out
    total_tool_calls = sum(count_tool_calls(t.get("transcript") or []) for t in valid_trials)
    total_agent_secs = sum(agent_seconds(t["result"]) for t in valid_trials)

    # pass^k: task passes only if EVERY shot has a full criteria pass
    tasks_consistent = sum(
        1
        for shot_list in tasks.values()
        if (valid_shots := [t for t in shot_list if not is_invalid_infra_trial(t["result"])])
        and all(grading_counts_as_pass(t["result"], t["grading"]) for t in valid_shots)
    )

    # pass@k: task passes if ANY shot has a full criteria pass
    tasks_passed = sum(
        1
        for shot_list in tasks.values()
        if any(
            grading_counts_as_pass(t["result"], t["grading"])
            for t in shot_list
            if not is_invalid_infra_trial(t["result"])
        )
    )
    pass_hat_rate = tasks_consistent / total_tasks if total_tasks else 0.0
    pass_rate = tasks_passed / total_tasks if total_tasks else 0.0
    mean_score = (
        sum(t["grading"].get("score", 0.0) for t in valid_trials) / len(valid_trials)
        if valid_trials
        else 0.0
    )

    # Category breakdown
    cat_stats: dict[str, CategoryStats] = {}
    for task_name, shot_list in tasks.items():
        valid_shots = [t for t in shot_list if not is_invalid_infra_trial(t["result"])]
        if not valid_shots:
            continue
        cat = categories.get(task_name, "unknown")
        if cat not in cat_stats:
            cat_stats[cat] = {"total": 0, "passed": 0, "consistent": 0, "scores": []}
        cat_stats[cat]["total"] += 1
        if all(grading_counts_as_pass(t["result"], t["grading"]) for t in valid_shots):
            cat_stats[cat]["consistent"] += 1
        if any(grading_counts_as_pass(t["result"], t["grading"]) for t in valid_shots):
            cat_stats[cat]["passed"] += 1
        for t in valid_shots:
            cat_stats[cat]["scores"].append(t["grading"].get("score", 0.0))

    # Build per-problem HTML sections
    category_problems: dict[str, list[str]] = {}
    compressed_details: dict[str, str] = {}

    for task_name, shot_list in sorted(tasks.items()):
        cat = categories.get(task_name, "unknown")
        multi_trial = len(shot_list) > 1
        valid_shots = [t for t in shot_list if not is_invalid_infra_trial(t["result"])]
        invalid_only = bool(shot_list) and not valid_shots

        shots_passed = sum(
            1 for t in valid_shots if grading_counts_as_pass(t["result"], t["grading"])
        )
        task_consistent = bool(valid_shots) and shots_passed == len(valid_shots)
        task_score = max((t["grading"].get("score", 0.0) for t in valid_shots), default=0.0)

        if multi_trial:
            shot_htmls = []
            for i, trial in enumerate(shot_list):
                r = trial["result"]
                g = trial["grading"]
                dur_s = agent_seconds(r)
                shot_cost = agent_result_metrics(r)[0]
                has_err = r.get("exception_info") is not None
                vp = grading_counts_as_pass(r, g)
                detail_html = render_trial_detail(trial)
                shot_htmls.append(f"""
              <details class="bg-gray-50 rounded border border-gray-100 mb-1">
                <summary class="flex items-center justify-between p-2 cursor-pointer hover:bg-gray-100">
                  <div class="flex items-center gap-2 min-w-0 flex-1">
                    <span class="text-xs text-gray-700">Trial {i + 1}</span>
                  </div>
                  <div class="flex items-center gap-3 text-xs text-gray-400 shrink-0 ml-4">
                    {status_badge_html(vp, has_error=has_err)}
                    <span>{format_duration(dur_s)}</span>
                    <span>{format_cost(shot_cost)}</span>
                  </div>
                </summary>
                <div class="px-3 pb-3 border-t border-gray-100">
                  {detail_html}
                </div>
              </details>""")

            inner_html = f"""
            <div class="flex items-center justify-between mt-3 mb-3">
                <div class="flex items-center gap-2">
                {invalid_badge_html() if invalid_only else pass_label_html(shots_passed, len(valid_shots))}
                <span class="text-xs text-gray-500">{"excluded from aggregates due to infra failure" if invalid_only else f"pass^{len(valid_shots)}: {'yes' if task_consistent else 'no'} · pass@{len(valid_shots)}"}</span>
                </div>
              <span class="text-sm font-semibold {score_color_class(task_score)}">Best Score {task_score:.0%}</span>
             </div>
             {"".join(shot_htmls)}"""
            compressed_details[task_name] = compress_text(inner_html)

            summary_right = (
                f"{invalid_badge_html() if invalid_only else pass_label_html(shots_passed, len(valid_shots))}"
                f'<span class="font-semibold {score_color_class(task_score)}">{task_score:.0%}</span>'
                f"<span>{format_cost(sum(agent_result_metrics(t['result'])[0] for t in shot_list))}</span>"
            )
        else:
            trial = shot_list[0]
            r = trial["result"]
            g = trial["grading"]
            vp = grading_counts_as_pass(r, g)
            detail_html = render_trial_detail(trial)
            compressed_details[task_name] = compress_text(detail_html)

            shot_cost = agent_result_metrics(r)[0]
            has_err = r.get("exception_info") is not None
            trial_score = g.get("score", 0.0)
            score_color = score_color_class(trial_score)
            summary_right = (
                f"{invalid_badge_html() if invalid_only else status_badge_html(vp, has_error=has_err)}"
                f'<span class="font-semibold {score_color}">{trial_score:.0%}</span>'
                f"<span>{format_cost(shot_cost)}</span>"
            )

        open_tag = (
            f'\n        <details class="bg-white rounded border border-gray-200 mb-2" data-problem="{escape_html(task_name)}">'
            f'\n          <summary class="flex items-center justify-between p-3 cursor-pointer hover:bg-gray-50">'
            f'\n            <div class="flex items-center gap-2 min-w-0 flex-1">'
            f'\n              <span class="font-medium text-sm text-gray-900">{escape_html(task_name)}</span>'
            f'\n              <span class="text-[10px] text-gray-400">{escape_html(cat)}</span>'
            f"\n            </div>"
            f'\n            <div class="flex items-center gap-3 text-xs text-gray-400 shrink-0 ml-4">'
            f"\n              {summary_right}"
            f"\n            </div>"
            f"\n          </summary>"
            f'\n          <div class="px-3 pb-3 border-t border-gray-100" data-detail></div>'
        )
        close_tag = "\n        </details>"
        category_problems.setdefault(cat, []).append(open_tag + close_tag)

    compressed_json = json.dumps(compressed_details)

    # Category rows for table
    cat_rows: list[str] = []
    for cat in sorted(cat_stats):
        cs = cat_stats[cat]
        total = cs["total"]
        passed = cs["passed"]
        mean = sum(cs["scores"]) / len(cs["scores"]) if cs["scores"] else 0.0
        pass_hat = cs["consistent"] / total if total else 0.0
        bar_pct = pass_hat * 100
        cat_rows.append(
            f'<tr class="border-b border-gray-100">'
            f'<td class="py-2 px-3 text-sm">{escape_html(category_label(cat))}</td>'
            f'<td class="py-2 px-3 text-sm text-right {score_color_class(pass_hat)} font-semibold">{cs["consistent"]}/{total}</td>'
            f'<td class="py-2 px-3 text-sm text-right text-gray-500">{passed}/{total}</td>'
            f'<td class="py-2 px-3 text-sm text-right {score_color_class(mean)} font-semibold">{mean:.0%}</td>'
            f'<td class="py-2 px-3 w-24">'
            f'<div class="w-full bg-gray-100 rounded-full h-1.5">'
            f'<div class="h-1.5 rounded-full {score_bar_class(pass_hat)}" style="width:{bar_pct:.0f}%"></div>'
            f"</div></td></tr>"
        )

    # Category problem sections
    cat_sections: list[str] = []
    cat_pass_counts = {
        cat: (
            sum(
                1
                for task_name, sl in tasks.items()
                if categories.get(task_name, "unknown") == cat
                and any(
                    grading_counts_as_pass(t["result"], t["grading"])
                    for t in sl
                    if not is_invalid_infra_trial(t["result"])
                )
            ),
            sum(
                1
                for task_name, sl in tasks.items()
                if categories.get(task_name, "unknown") == cat
                and any(not is_invalid_infra_trial(t["result"]) for t in sl)
            ),
        )
        for cat in sorted(category_problems)
    }
    for cat in sorted(category_problems):
        passed_c, total_c = cat_pass_counts.get(cat, (0, 0))
        label = category_label(cat)
        cat_sections.append(
            f'<div class="mt-6 mb-2">'
            f'<h3 class="text-sm font-semibold text-gray-600">{escape_html(label)}'
            f'<span class="ml-2 text-xs font-normal text-gray-400">'
            f"{passed_c}/{total_c} passed</span></h3></div>"
        )
        cat_sections.extend(category_problems[cat])

    # Summary stats
    pass_hat_rate_pct = pass_hat_rate * 100
    pass_rate_pct = pass_rate * 100
    mean_score_pct = mean_score * 100
    shots_per_task = (
        max(
            (
                sum(1 for t in shot_list if not is_invalid_infra_trial(t["result"]))
                for shot_list in tasks.values()
            ),
            default=1,
        )
        if total_tasks
        else 1
    )
    k_label = f"pass^{shots_per_task}" if shots_per_task > 1 else "exact pass rate"
    pass_at_k_label = f"pass@{shots_per_task}" if shots_per_task > 1 else "pass@1"
    steps_per_trial = f"{total_tool_calls / len(valid_trials):.1f}" if valid_trials else "0"
    tokens_in_per_trial = total_tokens_in / len(valid_trials) if valid_trials else 0.0
    tokens_out_per_trial = total_tokens_out / len(valid_trials) if valid_trials else 0.0
    tokens_cache_per_trial = total_tokens_cache / len(valid_trials) if valid_trials else 0.0
    total_tokens_per_trial = total_tokens / len(valid_trials) if valid_trials else 0.0

    template_path = Path(__file__).parent / "report_template.html"
    template = template_path.read_text()
    return template.format(
        model_display=escape_html(model_display),
        job_id=escape_html(job_id),
        job_timestamp=escape_html(job_timestamp),
        shots_per_task=shots_per_task,
        k_label_title=escape_html(k_label.title()),
        pass_hat_rate_color=score_color_class(pass_hat_rate),
        pass_hat_rate_pct=pass_hat_rate_pct,
        pass_hat_rate_bar=score_bar_class(pass_hat_rate),
        pass_hat_rate_bar_pct=min(pass_hat_rate_pct, 100),
        tasks_consistent=tasks_consistent,
        pass_at_k_label=escape_html(pass_at_k_label.title()),
        pass_rate_color=score_color_class(pass_rate),
        pass_rate_pct=pass_rate_pct,
        pass_rate_bar=score_bar_class(pass_rate),
        pass_rate_bar_pct=min(pass_rate_pct, 100),
        tasks_passed=tasks_passed,
        total_tasks=total_tasks,
        mean_score_color=score_color_class(mean_score),
        mean_score_pct=mean_score_pct,
        mean_score_bar=score_bar_class(mean_score),
        mean_score_bar_pct=min(mean_score_pct, 100),
        n_trials=len(trials),
        total_cost_fmt=format_cost(total_cost),
        total_agent_time_fmt=format_duration(total_agent_secs),
        total_tokens_in=f"{total_tokens_in:,}",
        total_tokens_out=f"{total_tokens_out:,}",
        total_tokens_cache=f"{total_tokens_cache:,}",
        tokens_in_per_trial=f"{tokens_in_per_trial:,.0f}",
        tokens_out_per_trial=f"{tokens_out_per_trial:,.0f}",
        tokens_cache_per_trial=f"{tokens_cache_per_trial:,.0f}",
        total_tokens_per_trial=f"{total_tokens_per_trial:,.0f}",
        steps_per_trial=steps_per_trial,
        cat_rows="".join(cat_rows),
        cat_sections="".join(cat_sections),
        compressed_json=compressed_json,
    )


def write_report(job_dir: Path, tasks_dir: Path | None = None, output: Path | None = None) -> Path:
    report_path = run_report_output_path(job_dir, output)
    report_html = generate_report(job_dir, tasks_dir=tasks_dir)
    report_path.write_text(report_html)
    return report_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate per-run HTML report for a Harbor job")
    parser.add_argument(
        "--job-dir",
        required=True,
        type=Path,
        help="Harbor job directory (e.g. jobs/2026-03-18__08-06-52)",
    )
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        default=None,
        help="Harbor tasks directory for category metadata (auto-detected if omitted)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML file (default: <job-dir>/run_report.html)",
    )
    args = parser.parse_args()

    job_dir = args.job_dir.resolve()
    if not job_dir.exists():
        print(f"Job directory not found: {job_dir}", file=sys.stderr)
        sys.exit(1)

    output = write_report(job_dir, tasks_dir=args.tasks_dir, output=args.output)
    print(f"Report written to {output}", file=sys.stderr)


if __name__ == "__main__":
    main()
