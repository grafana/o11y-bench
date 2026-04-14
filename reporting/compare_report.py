#!/usr/bin/env python3
# /// script
# dependencies = []
# ///
"""Generate a side-by-side comparison HTML report for multiple Harbor benchmark runs.

Usage:
    uv run python -m reporting.compare_report --job-dir jobs/run-a --job-dir jobs/run-b --output compare.html
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from .categories import category_label
from .report_data import (
    escape_html,
    format_cost,
    format_duration,
    grading_counts_as_pass,
    is_invalid_infra_trial,
    load_task_categories,
    resolve_tasks_dir,
    score_color_class,
    trial_model_display,
    trial_to_row,
)
from .run_report import load_trials
from .summary import summarize_trials

JsonDict = dict[str, Any]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_job(job_dir: Path, tasks_dir: Path | None) -> JsonDict:
    """Load all data for one job and return a summary dict."""
    trials = load_trials(job_dir)
    if not trials:
        print(f"No trials found in {job_dir}", file=sys.stderr)
        sys.exit(1)

    resolved_tasks_dir = resolve_tasks_dir(job_dir, tasks_dir)
    categories: dict[str, str] = {}
    if resolved_tasks_dir:
        categories = load_task_categories(resolved_tasks_dir)

    first_result = trials[0]["result"]
    model_display = trial_model_display(first_result, trials[0]["result_path"].resolve())

    tasks: dict[str, list[JsonDict]] = {}
    for trial in trials:
        task_name = trial["result"].get("task_name", "unknown")
        tasks.setdefault(task_name, []).append(trial)

    summary = summarize_trials(
        [trial_to_row(trial["result"], trial.get("transcript")) for trial in trials]
    )

    tasks_passed = sum(
        1
        for shot_list in tasks.values()
        if any(
            grading_counts_as_pass(t["result"], t["grading"])
            for t in shot_list
            if not is_invalid_infra_trial(t["result"])
        )
    )
    tasks_consistent = sum(
        1
        for shot_list in tasks.values()
        if (
            valid_shots := [
                trial for trial in shot_list if not is_invalid_infra_trial(trial["result"])
            ]
        )
        and all(grading_counts_as_pass(trial["result"], trial["grading"]) for trial in valid_shots)
    )
    total_tasks = summary["n_tasks"]
    pass_rate = tasks_passed / total_tasks if total_tasks else 0.0
    pass_hat_rate = tasks_consistent / total_tasks if total_tasks else 0.0
    per_task = summary["per_task"]
    task_scores: dict[str, float] = {
        task_name: stats["best_score"] for task_name, stats in per_task.items()
    }
    task_passed: dict[str, bool] = {
        task_name: stats["passed"] for task_name, stats in per_task.items()
    }
    task_consistent: dict[str, bool] = {
        task_name: stats["consistent"] for task_name, stats in per_task.items()
    }
    task_cost: dict[str, float] = {
        task_name: stats["cost_usd"] for task_name, stats in per_task.items()
    }

    return {
        "job_dir": job_dir,
        "model_display": model_display,
        "trials": trials,
        "tasks": tasks,
        "categories": categories,
        "total_cost": summary["total_cost_usd"],
        "total_tokens_in": summary["total_tokens_in"],
        "total_tokens_out": summary["total_tokens_out"],
        "total_tokens_cache": summary["total_tokens_cache"],
        "tokens_per_trial": (
            (summary["total_tokens_in"] + summary["total_tokens_out"]) / summary["n_valid_trials"]
            if summary["n_valid_trials"]
            else 0.0
        ),
        "input_tokens_per_trial": (
            summary["total_tokens_in"] / summary["n_valid_trials"]
            if summary["n_valid_trials"]
            else 0.0
        ),
        "output_tokens_per_trial": (
            summary["total_tokens_out"] / summary["n_valid_trials"]
            if summary["n_valid_trials"]
            else 0.0
        ),
        "cache_tokens_per_trial": (
            summary["total_tokens_cache"] / summary["n_valid_trials"]
            if summary["n_valid_trials"]
            else 0.0
        ),
        "total_tool_calls": summary["total_tool_calls"],
        "total_agent_secs": summary["total_agent_secs"],
        "tasks_consistent": tasks_consistent,
        "tasks_passed": tasks_passed,
        "total_tasks": total_tasks,
        "pass_hat_rate": pass_hat_rate,
        "pass_rate": pass_rate,
        "mean_score": summary["mean_score"],
        "shots_per_task": summary["shots_per_task"],
        "task_scores": task_scores,
        "task_consistent": task_consistent,
        "task_passed": task_passed,
        "task_cost": task_cost,
        "steps_per_trial": f"{summary['steps_per_trial']:.1f}",
    }


def relative_href(from_file: Path, to_path: Path) -> str:
    return os.path.relpath(to_path, start=from_file.parent)


def winner_class(vals: list[float], idx: int, higher_is_better: bool = True) -> str:
    """Return CSS class to highlight the best value."""
    if not vals or len(set(vals)) == 1:
        return ""
    best = max(vals) if higher_is_better else min(vals)
    return "font-bold text-green-700" if vals[idx] == best else "text-gray-500"


def sort_jobs(jobs: list[JsonDict], sort_by: str) -> list[JsonDict]:
    if sort_by == "input":
        return jobs

    if sort_by == "name":
        return sorted(jobs, key=lambda j: j["model_display"])

    metric_key = {
        "pass-hat": "pass_hat_rate",
        "pass-at-k": "pass_rate",
        "mean": "mean_score",
    }[sort_by]
    return sorted(
        jobs,
        key=lambda j: (
            -j[metric_key],
            -j["pass_rate"],
            -j["mean_score"],
            j["model_display"],
        ),
    )


def generate_comparison(
    job_dirs: list[Path], tasks_dir: Path | None, output_path: Path, sort_by: str = "input"
) -> str:
    jobs = [load_job(d, tasks_dir) for d in job_dirs]
    jobs = sort_jobs(jobs, sort_by)
    n = len(jobs)

    # Build header columns
    header_parts = []
    for j in jobs:
        report_path = j["job_dir"] / "run_report.html"
        report_link = ""
        if report_path.exists():
            href = relative_href(output_path, report_path)
            report_link = (
                f'<div class="mt-1 text-[10px] font-medium">'
                f'<a href="{escape_html(href)}" class="text-blue-600 hover:text-blue-800 underline">'
                f"Open report</a></div>"
            )
        header_parts.append(
            f'<th class="py-2 px-3 text-sm font-semibold text-center border-b border-gray-200 max-w-[160px]">'
            f"{escape_html(j['model_display'])}"
            f'<div class="text-[10px] font-normal text-gray-400">{escape_html(j["job_dir"].name)}</div>'
            f"{report_link}"
            f"</th>"
        )
    header_cols = "".join(header_parts)

    # Summary rows
    def summary_row(
        label: str,
        values: list[str],
        winner_vals: list[float] | None = None,
        higher_is_better: bool = True,
    ) -> str:
        cells = ""
        for idx2, v in enumerate(values):
            cls = ""
            if winner_vals:
                cls = winner_class(winner_vals, idx2, higher_is_better)
            cells += f'<td class="py-2 px-3 text-sm text-center {cls}">{v}</td>'
        return f'<tr class="border-b border-gray-100"><td class="py-2 px-3 text-sm text-gray-600">{escape_html(label)}</td>{cells}</tr>'

    pass_hat_rates = [j["pass_hat_rate"] for j in jobs]
    pass_rates = [j["pass_rate"] for j in jobs]
    mean_scores = [j["mean_score"] for j in jobs]
    total_costs = [j["total_cost"] for j in jobs]
    total_secs = [j["total_agent_secs"] for j in jobs]
    steps = [float(j["steps_per_trial"]) for j in jobs]
    tokens_per_trial = [j["tokens_per_trial"] for j in jobs]

    summary_rows = "".join(
        [
            summary_row(
                "Pass^k",
                [
                    f"{j['pass_hat_rate'] * 100:.1f}% ({j['tasks_consistent']}/{j['total_tasks']})"
                    for j in jobs
                ],
                pass_hat_rates,
            ),
            summary_row(
                "Pass@k",
                [
                    f"{j['pass_rate'] * 100:.1f}% ({j['tasks_passed']}/{j['total_tasks']})"
                    for j in jobs
                ],
                pass_rates,
            ),
            summary_row("Mean Score", [f"{j['mean_score'] * 100:.1f}%" for j in jobs], mean_scores),
            summary_row(
                "Total Cost",
                [format_cost(j["total_cost"]) for j in jobs],
                total_costs,
                higher_is_better=False,
            ),
            summary_row(
                "Agent Time",
                [format_duration(j["total_agent_secs"]) for j in jobs],
                total_secs,
                higher_is_better=False,
            ),
            summary_row(
                "Steps/Trial", [j["steps_per_trial"] for j in jobs], steps, higher_is_better=False
            ),
            summary_row(
                "Tokens In",
                [f"{j['total_tokens_in']:,}" for j in jobs],
                [j["total_tokens_in"] for j in jobs],
                higher_is_better=False,
            ),
            summary_row(
                "Tokens Out",
                [f"{j['total_tokens_out']:,}" for j in jobs],
                [j["total_tokens_out"] for j in jobs],
                higher_is_better=False,
            ),
            summary_row(
                "Cache Tokens",
                [f"{j['total_tokens_cache']:,}" for j in jobs],
                [j["total_tokens_cache"] for j in jobs],
                higher_is_better=True,
            ),
            summary_row(
                "Tokens/Trial",
                [
                    f"{j['tokens_per_trial']:,.0f}"
                    f' <span class="text-[10px] text-gray-400">'
                    f"({j['input_tokens_per_trial']:,.0f} in / {j['output_tokens_per_trial']:,.0f} out"
                    f" / {j['cache_tokens_per_trial']:,.0f} cache)</span>"
                    for j in jobs
                ],
                tokens_per_trial,
                higher_is_better=False,
            ),
        ]
    )

    # All tasks across all runs
    all_tasks = sorted(set(t for j in jobs for t in j["tasks"]))

    task_rows = ""
    for task_name in all_tasks:
        cat = jobs[0]["categories"].get(task_name, "")
        cat_display = category_label(cat) if cat else ""
        cells = ""
        for i, j in enumerate(jobs):
            score = j["task_scores"].get(task_name)
            consistent = j["task_consistent"].get(task_name)
            passed = j["task_passed"].get(task_name)
            cost = j["task_cost"].get(task_name)
            if score is None:
                cells += '<td class="py-2 px-3 text-center text-gray-300 text-xs">—</td>'
            else:
                badge = (
                    '<span class="px-1.5 py-0.5 rounded text-[10px] font-medium bg-green-100 text-green-800">pass</span>'
                    if consistent
                    else '<span class="px-1.5 py-0.5 rounded text-[10px] font-medium bg-yellow-100 text-yellow-800">partial</span>'
                    if passed
                    else '<span class="px-1.5 py-0.5 rounded text-[10px] font-medium bg-red-100 text-red-800">fail</span>'
                )
                cells += (
                    f'<td class="py-2 px-3 text-center">'
                    f'<div class="flex items-center justify-center gap-1.5">'
                    f"{badge}"
                    f'<span class="text-xs font-semibold {score_color_class(score)}">{score:.0%}</span>'
                    f'<span class="text-[10px] text-gray-400">{format_cost(cost or 0)}</span>'
                    f"</div></td>"
                )
        task_rows += (
            f'<tr class="border-b border-gray-100 hover:bg-gray-50">'
            f'<td class="py-2 px-3 text-sm text-gray-900">{escape_html(task_name)}'
            f'<span class="ml-1.5 text-[10px] text-gray-400">{escape_html(cat_display)}</span></td>'
            f"{cells}</tr>"
        )

    template_path = Path(__file__).parent / "compare_template.html"
    template = template_path.read_text()
    return template.format(
        n_runs=n,
        n_tasks=len(all_tasks),
        header_cols=header_cols,
        summary_rows=summary_rows,
        task_rows=task_rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate comparison HTML report for multiple Harbor jobs"
    )
    parser.add_argument(
        "--job-dir",
        required=True,
        action="append",
        type=Path,
        dest="job_dirs",
        help="Harbor job directory (repeat for each run)",
    )
    parser.add_argument("--tasks-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("compare.html"))
    parser.add_argument(
        "--sort-by",
        choices=["input", "pass-hat", "pass-at-k", "mean", "name"],
        default="input",
    )
    args = parser.parse_args()

    tasks_dir = resolve_tasks_dir(args.job_dirs[0].resolve(), args.tasks_dir)

    html_out = generate_comparison(
        [d.resolve() for d in args.job_dirs],
        tasks_dir,
        args.output,
        sort_by=args.sort_by,
    )
    args.output.write_text(html_out)
    print(f"Comparison report written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
