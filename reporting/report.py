#!/usr/bin/env python3
"""Generate an HTML leaderboard from Harbor job result directories.

Reads all jobs/<timestamp>/*/result.json trial files, aggregates pass^k,
pass@k, and mean scores per model, and writes a self-contained HTML report.

Usage:
    uv run python -m reporting.report                        # reads latest jobs/full-suite-*/, writes jobs/<suite>/comparison.html
    uv run python -m reporting.report --jobs-dir jobs --output leaderboard.html
    uv run python -m reporting.report --jobs-dir jobs/2026-03-18__07-51-56  # single job
"""

# /// script
# dependencies = ["pyyaml>=6.0"]
# ///

import argparse
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import run_report as run_report_module
from .categories import category_label
from .report_data import (
    HIGH_SCORE_THRESHOLD,
    MEDIUM_SCORE_THRESHOLD,
    escape_html,
    format_compact_count,
    job_expected_trial_count,
    job_n_attempts,
    load_task_categories,
    load_trials,
    pretty_variant,
    trial_to_row,
    variant_key,
)
from .report_paths import (
    latest_suite_dir,
    normalize_cli_path,
    run_report_output_path,
    suite_report_output_path,
)
from .summary import TaskSummary, summarize_trials

JsonDict = dict[str, Any]


def _job_dir_from_trial(trial: JsonDict) -> Path | None:
    result_path = trial.get("__result_path")
    if not isinstance(result_path, str) or not result_path:
        return None
    return Path(result_path).parent.parent


# ── Aggregation ───────────────────────────────────────────────────────────────


def _category_stats(
    per_task: dict[str, TaskSummary], categories: dict[str, str]
) -> dict[str, JsonDict]:
    by_cat: dict[str, JsonDict] = defaultdict(
        lambda: {"n": 0, "passed": 0, "consistent": 0, "scores": []}
    )
    for task, stats in per_task.items():
        cat = categories.get(task, "unknown")
        by_cat[cat]["n"] += 1
        if stats["passed"]:
            by_cat[cat]["passed"] += 1
        if stats["consistent"]:
            by_cat[cat]["consistent"] += 1
        by_cat[cat]["scores"].append(stats["mean_score"])

    return {
        cat: {
            "pass_hat_rate": stats["consistent"] / stats["n"] if stats["n"] else 0.0,
            "pass_rate": stats["passed"] / stats["n"] if stats["n"] else 0.0,
            "mean_score": sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else 0.0,
            "n": stats["n"],
            "n_consistent": stats["consistent"],
            "n_passed": stats["passed"],
        }
        for cat, stats in sorted(by_cat.items())
    }


def aggregate(trials: list[JsonDict], categories: dict[str, str]) -> JsonDict:
    by_model: dict[tuple[str, str], list[JsonDict]] = defaultdict(list)
    for trial in trials:
        by_model[variant_key(trial)].append(trial)

    observed_tasks = {trial["task_name"] for trial in trials}
    all_tasks = sorted(categories) if categories else sorted(observed_tasks)
    models: list[JsonDict] = []
    expected_trials_cache: dict[Path, int | None] = {}
    n_attempts_cache: dict[Path, int | None] = {}

    for (model_name, reasoning_effort), model_trials in sorted(by_model.items()):
        summary = summarize_trials([trial_to_row(trial) for trial in model_trials])
        per_task = dict(summary["per_task"])
        shots_per_task = summary["shots_per_task"] or 1

        # Resolve the job_dir once for this model variant.
        model_job_dir: Path | None = None
        for trial in model_trials:
            model_job_dir = _job_dir_from_trial(trial)
            if model_job_dir is not None:
                break

        configured_attempts: int | None = None
        expected_trials: int | None = None
        if model_job_dir is not None:
            configured_attempts = n_attempts_cache.setdefault(
                model_job_dir, job_n_attempts(model_job_dir)
            )
            expected_trials = expected_trials_cache.setdefault(
                model_job_dir, job_expected_trial_count(model_job_dir)
            )

        for task_name in all_tasks:
            per_task.setdefault(
                task_name,
                {
                    "scores": [None] * shots_per_task,
                    "passed": False,
                    "consistent": False,
                    "mean_score": 0.0,
                    "best_score": 0.0,
                    "cost_usd": 0.0,
                },
            )
        n_tasks = len(all_tasks)
        n_consistent = sum(1 for stats in per_task.values() if stats["consistent"])
        n_passed = sum(1 for stats in per_task.values() if stats["passed"])
        n_valid_trials = summary["n_valid_trials"]
        expected_trials = expected_trials or (n_tasks * (configured_attempts or shots_per_task))
        report_links = sorted(
            {
                report_path.resolve().as_uri()
                for trial in model_trials
                for job_dir in [_job_dir_from_trial(trial)]
                if job_dir is not None
                for report_path in [run_report_output_path(job_dir)]
                if report_path.exists()
            }
        )

        models.append(
            {
                "name": model_name,
                "reasoning_effort": reasoning_effort,
                "label": pretty_variant(model_name, reasoning_effort),
                "n_trials": len(model_trials),
                "n_valid_trials": n_valid_trials,
                "expected_trials": expected_trials,
                "n_tasks": n_tasks,
                "n_consistent": n_consistent,
                "n_passed": n_passed,
                "pass_hat_rate": (n_consistent / n_tasks) if n_tasks else 0.0,
                "pass_rate": (n_passed / n_tasks) if n_tasks else 0.0,
                "mean_score": summary["mean_score"],
                "shots_per_task": shots_per_task,
                "total_cost_usd": summary["total_cost_usd"],
                "agent_mins": summary["total_agent_secs"] / 60,
                "total_tokens": summary["total_tokens_in"] + summary["total_tokens_out"],
                "tokens_per_trial": (
                    (summary["total_tokens_in"] + summary["total_tokens_out"]) / n_valid_trials
                    if n_valid_trials
                    else 0.0
                ),
                "input_tokens_per_trial": (
                    summary["total_tokens_in"] / n_valid_trials if n_valid_trials else 0.0
                ),
                "output_tokens_per_trial": (
                    summary["total_tokens_out"] / n_valid_trials if n_valid_trials else 0.0
                ),
                "cache_tokens_per_trial": (
                    summary["total_tokens_cache"] / n_valid_trials if n_valid_trials else 0.0
                ),
                "by_category": _category_stats(per_task, categories),
                "per_task": per_task,
                "report_links": report_links,
            }
        )

    models.sort(
        key=lambda model: (-model["pass_hat_rate"], -model["pass_rate"], -model["mean_score"])
    )
    return {"models": models, "all_tasks": all_tasks, "categories": categories}


def resolve_jobs_dir(jobs_dir_arg: str | None, *, quiet: bool = False) -> Path:
    if jobs_dir_arg is None:
        jobs_dir = latest_suite_dir(Path("jobs"))
        if jobs_dir is None:
            print("Error: no jobs/full-suite-* directory found under 'jobs'")
            raise SystemExit(1)
        return jobs_dir

    requested = normalize_cli_path(jobs_dir_arg)
    if not requested.exists():
        print(f"Error: jobs dir '{requested}' not found")
        raise SystemExit(1)

    if requested.is_dir() and not (requested / "result.json").exists():
        suite_dir = latest_suite_dir(requested)
        if suite_dir is not None:
            if not quiet:
                print(f"Resolved report source to latest suite: {suite_dir}")
            return suite_dir

    return requested


# ── HTML helpers ──────────────────────────────────────────────────────────────


def score_color(score: float) -> str:
    if score >= HIGH_SCORE_THRESHOLD:
        return "#16a34a"
    if score >= MEDIUM_SCORE_THRESHOLD:
        return "#ca8a04"
    return "#dc2626"


def pct(score: float) -> str:
    return f"{score * 100:.1f}%"


def primary_pass_label(shots_per_task: int) -> str:
    return "Exact Pass Rate" if shots_per_task <= 1 else f"Pass^{shots_per_task}"


def secondary_pass_label(shots_per_task: int) -> str:
    return "Pass@1" if shots_per_task <= 1 else f"Pass@{shots_per_task}"


def bar(score: float, width: int = 60) -> str:
    filled = int(score * width)
    color = score_color(score)
    return (
        f'<div style="display:inline-block;width:{width}px;height:8px;'
        f'background:#e5e7eb;border-radius:3px;overflow:hidden;vertical-align:middle;margin-left:6px">'
        f'<div style="width:{filled}px;height:100%;background:{color}"></div></div>'
    )


def format_trials(valid_trials: int, expected_trials: int) -> str:
    return f"{valid_trials}/{expected_trials}"


def trial_count_color(valid_trials: int, expected_trials: int) -> str:
    return "#6b7280" if valid_trials == expected_trials else "#dc2626"


# ── HTML rendering ────────────────────────────────────────────────────────────


def render_html(data: JsonDict) -> str:
    models = data["models"]
    all_tasks = data["all_tasks"]
    categories = data["categories"]
    jobs_dir = Path(data["jobs_dir"])
    expected_shots_per_task = max((model["shots_per_task"] for model in models), default=1)
    expected_trials_per_variant = len(all_tasks) * expected_shots_per_task
    incomplete_models = [
        model for model in models if model["n_valid_trials"] != model["expected_trials"]
    ]

    # Determine categories present (excluding unknown, sorted by label)
    all_cats = sorted(
        {categories.get(t, "") for t in all_tasks} - {"", "unknown"},
        key=category_label,
    )

    if not models:
        return "<html><body><p>No completed trials found.</p></body></html>"

    # ── Category Leaders ──
    cat_leaders_html = ""
    if all_cats:
        leaders = []
        for cat in all_cats:
            best_for_cat = max(
                (m for m in models if cat in m["by_category"]),
                key=lambda m: (
                    m["by_category"][cat]["pass_hat_rate"],
                    m["by_category"][cat]["pass_rate"],
                    m["by_category"][cat]["mean_score"],
                ),
                default=None,
            )
            if best_for_cat:
                stats = best_for_cat["by_category"][cat]
                color = score_color(stats["pass_hat_rate"])
                leaders.append(f"""
                <div style="background:white;border:1px solid #e5e7eb;border-radius:8px;padding:16px;min-width:160px;flex:1">
                  <div style="font-size:0.75em;color:#6b7280;text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px">{escape_html(category_label(cat))}</div>
                  <div style="font-weight:700;font-size:1em;margin-bottom:6px">{escape_html(best_for_cat["label"])}</div>
                  <div style="font-size:1.6em;font-weight:800;color:{color}">{pct(stats["pass_hat_rate"])}</div>
                  <div style="font-size:0.75em;color:#9ca3af">{stats["n_consistent"]}/{stats["n"]} tasks</div>
                </div>""")
        cat_leaders_html = f"""
        <div style="margin-bottom:24px">
          <h2 style="font-size:1em;font-weight:600;color:#374151;margin-bottom:12px;text-transform:uppercase;letter-spacing:.05em">Category Leaders</h2>
          <div style="display:flex;flex-wrap:wrap;gap:10px">{"".join(leaders)}</div>
        </div>"""

    # ── Leaderboard table ──
    cat_headers = "".join(
        f'<th style="padding:8px 10px;text-align:center;font-weight:600;color:#6b7280;font-size:0.78em;white-space:nowrap">{escape_html(category_label(c))}</th>'
        for c in all_cats
    )

    leaderboard_rows = []
    for rank, m in enumerate(models, 1):
        shots_per_task = m["shots_per_task"]
        report_links_html = ""
        if m["report_links"]:
            report_links_html = (
                '<div style="margin-top:4px;font-size:0.75em">'
                + " ".join(
                    f'<a href="{escape_html(link)}" style="color:#2563eb;text-decoration:none">report{"" if len(m["report_links"]) == 1 else f" {index}"}</a>'
                    for index, link in enumerate(m["report_links"], 1)
                )
                + "</div>"
            )
        cat_cells = "".join(
            f'<td style="text-align:center;padding:6px 10px;font-size:0.85em;color:{score_color(m["by_category"].get(c, {}).get("pass_hat_rate", 0))}">'
            f"{pct(m['by_category'].get(c, {}).get('pass_hat_rate', 0))}</td>"
            for c in all_cats
        )
        agent_time = f"{m['agent_mins']:.0f}m" if m["agent_mins"] >= 1 else "<1m"
        leaderboard_rows.append(f"""
        <tr style="border-bottom:1px solid #f3f4f6">
          <td style="padding:8px 10px;color:#9ca3af;font-size:0.85em">{rank}</td>
          <td style="padding:8px 10px;font-weight:600">{escape_html(m["label"])}{report_links_html}</td>
          <td style="padding:8px 10px;white-space:nowrap">
            <span style="font-weight:700;color:{score_color(m["pass_hat_rate"])}">{pct(m["pass_hat_rate"])}</span>
            <span style="color:#9ca3af;font-size:0.8em;margin-left:4px">{m["n_consistent"]}/{m["n_tasks"]}</span>
            {bar(m["pass_hat_rate"])}
            <div style="color:#9ca3af;font-size:0.75em;margin-top:3px">{escape_html(secondary_pass_label(shots_per_task))}: {pct(m["pass_rate"])} ({m["n_passed"]}/{m["n_tasks"]})</div>
          </td>
          <td style="padding:8px 10px;text-align:center;color:{score_color(m["mean_score"])};font-size:0.9em">{pct(m["mean_score"])}</td>
          <td style="padding:8px 10px;text-align:right;color:#6b7280;font-size:0.85em">{format_compact_count(m["total_tokens"])}</td>
          <td style="padding:8px 10px;text-align:right;color:#6b7280;font-size:0.85em">${m["total_cost_usd"]:.2f}</td>
          <td style="padding:8px 10px;text-align:right;color:#6b7280;font-size:0.85em">{agent_time}</td>
          <td style="padding:8px 10px;text-align:center;color:{trial_count_color(m["n_valid_trials"], m["expected_trials"])};font-size:0.85em">{format_trials(m["n_valid_trials"], m["expected_trials"])}</td>
          {cat_cells}
        </tr>""")

    leaderboard_html = f"""
    <table style="width:100%;border-collapse:collapse;font-size:0.9em">
      <thead>
        <tr style="border-bottom:2px solid #e5e7eb;background:#f9fafb">
          <th style="padding:8px 10px;text-align:left;color:#9ca3af;font-weight:500;font-size:0.8em">#</th>
          <th style="padding:8px 10px;text-align:left;font-weight:600;color:#374151">Model</th>
          <th style="padding:8px 10px;text-align:left;font-weight:600;color:#374151">Pass rate</th>
          <th style="padding:8px 10px;text-align:center;font-weight:600;color:#374151">Mean Score</th>
          <th style="padding:8px 10px;text-align:right;font-weight:600;color:#374151">Tokens</th>
          <th style="padding:8px 10px;text-align:right;font-weight:600;color:#374151">Cost</th>
          <th style="padding:8px 10px;text-align:right;font-weight:600;color:#374151">Agent Time</th>
          <th style="padding:8px 10px;text-align:center;font-weight:600;color:#374151">Trials</th>
          {cat_headers}
        </tr>
      </thead>
      <tbody>{"".join(leaderboard_rows)}</tbody>
    </table>"""

    # ── Per-task breakdown (collapsible per model) ──
    detail_sections = []
    for m in models:
        rows = []
        for task in sorted(m["per_task"]):
            stats = m["per_task"][task]
            cat = categories.get(task, "")
            scores_str = " / ".join(
                f'<span style="color:{score_color(s) if s is not None else "#d1d5db"}">'
                f"{'—' if s is None else pct(s)}</span>"
                for s in stats["scores"]
            )
            if stats["consistent"]:
                badge = '<span style="background:#dcfce7;color:#15803d;padding:1px 7px;border-radius:10px;font-size:0.75em;font-weight:600">PASS</span>'
            elif stats["passed"]:
                badge = '<span style="background:#fef3c7;color:#b45309;padding:1px 7px;border-radius:10px;font-size:0.75em;font-weight:600">PARTIAL</span>'
            else:
                badge = '<span style="background:#fee2e2;color:#b91c1c;padding:1px 7px;border-radius:10px;font-size:0.75em;font-weight:600">FAIL</span>'
            rows.append(f"""
            <tr style="border-bottom:1px solid #f9fafb">
              <td style="padding:5px 10px;font-family:monospace;font-size:0.82em;color:#374151">{escape_html(task)}</td>
              <td style="padding:5px 10px;color:#9ca3af;font-size:0.8em">{escape_html(category_label(cat))}</td>
              <td style="padding:5px 10px;text-align:center">{badge}</td>
              <td style="padding:5px 10px;text-align:center;font-size:0.85em">{scores_str}</td>
              <td style="padding:5px 10px;text-align:center;color:{score_color(stats["mean_score"])};font-size:0.85em">{pct(stats["mean_score"])}</td>
            </tr>""")

        detail_sections.append(f"""
        <details style="margin-bottom:10px;border:1px solid #e5e7eb;border-radius:8px;overflow:hidden">
          <summary style="padding:10px 16px;background:#f9fafb;cursor:pointer;font-weight:600;list-style:none;display:flex;align-items:center;gap:10px;user-select:none">
            <span style="color:#9ca3af;font-size:0.8em">▶</span>
            <span>{escape_html(m["label"])}</span>
            <span style="color:{score_color(m["pass_hat_rate"])};font-weight:700">{pct(m["pass_hat_rate"])} {escape_html(primary_pass_label(m["shots_per_task"]))}</span>
            <span style="color:#9ca3af;font-size:0.82em">{m["n_consistent"]}/{m["n_tasks"]} tasks</span>
          </summary>
          <table style="width:100%;border-collapse:collapse;font-size:0.88em">
            <thead>
              <tr style="background:#fafafa;border-bottom:1px solid #e5e7eb">
                <th style="padding:6px 10px;text-align:left;font-weight:600;color:#6b7280">Task</th>
                <th style="padding:6px 10px;text-align:left;font-weight:600;color:#6b7280">Category</th>
                <th style="padding:6px 10px;text-align:center;font-weight:600;color:#6b7280">Result</th>
                <th style="padding:6px 10px;text-align:center;font-weight:600;color:#6b7280">Shots</th>
                <th style="padding:6px 10px;text-align:center;font-weight:600;color:#6b7280">Mean</th>
              </tr>
            </thead>
            <tbody>{"".join(rows)}</tbody>
          </table>
        </details>""")

    generated = datetime.now(UTC).strftime("%Y-%m-%dT%H:%MZ")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>o11y-bench Leaderboard</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f8fafc; color: #111827; line-height: 1.4; }}
    .container {{ max-width: 1400px; margin: 0 auto; padding: 32px 24px; }}
    h1 {{ font-size: 1.5em; font-weight: 700; margin-bottom: 2px; }}
    h2 {{ font-size: 1em; font-weight: 600; margin: 24px 0 12px; color: #374151; }}
    .card {{ background: white; border: 1px solid #e5e7eb; border-radius: 10px; padding: 20px; margin-bottom: 20px; box-shadow: 0 1px 2px rgba(0,0,0,.04); overflow-x: auto; }}
    .meta {{ color: #6b7280; font-size: 0.82em; margin-bottom: 20px; }}
    details > summary::-webkit-details-marker {{ display: none; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>o11y-bench Leaderboard</h1>
    <p class="meta">{len(models)} model(s) &middot; {len(all_tasks)} tasks &middot; Source {escape_html(jobs_dir.name)} &middot; Generated {generated}</p>
    <p class="meta">Expected full run per variant: {len(all_tasks)} tasks &times; {expected_shots_per_task} trials = {expected_trials_per_variant} trials. Trials shows valid/expected, excluding infra-invalid trials but keeping `AgentTimeoutError` attempts as normal failures.</p>
    {"" if not incomplete_models else f'<p class="meta" style="color:#b45309">Warning: {len(incomplete_models)} variant(s) have fewer valid trials than expected. Check the Trials column.</p>'}

    {cat_leaders_html}

    <div class="card">
      <div style="display:flex;align-items:baseline;gap:8px;margin-bottom:4px">
        <h2 style="margin:0">Leaderboard</h2>
        <span style="font-size:0.78em;color:#9ca3af">ranked by pass^k, then pass@k, then mean score</span>
      </div>
      <p style="font-size:0.78em;color:#9ca3af;margin-bottom:14px">Pass^k: a task passes only if <em>every</em> shot scores 1.0. Pass@k: a task passes if <em>any</em> shot scores 1.0. Tokens are total input + output across valid trials.</p>
      {leaderboard_html}
    </div>

    <h2 style="margin-top:8px">Per-Task Breakdown</h2>
    {"".join(detail_sections)}
  </div>
</body>
</html>"""


def _job_dirs_for_report_refresh(jobs_dir: Path) -> list[Path]:
    if (jobs_dir / "config.json").exists():
        return [jobs_dir]
    return sorted(
        path for path in jobs_dir.iterdir() if path.is_dir() and (path / "config.json").exists()
    )


def _report_needs_refresh(job_dir: Path, report_path: Path) -> bool:
    if not report_path.exists():
        return True

    report_mtime = report_path.stat().st_mtime_ns
    input_mtimes = [job_dir.stat().st_mtime_ns]
    config_path = job_dir / "config.json"
    if config_path.exists():
        input_mtimes.append(config_path.stat().st_mtime_ns)
    for result_path in job_dir.rglob("result.json"):
        input_mtimes.append(result_path.stat().st_mtime_ns)
    return max(input_mtimes) > report_mtime


def _job_has_completed_trials(job_dir: Path) -> bool:
    return any(
        trial_dir.is_dir()
        and "__" in trial_dir.name
        and (trial_dir / "result.json").exists()
        and (trial_dir / "verifier" / "grading_details.json").exists()
        for trial_dir in job_dir.iterdir()
    )


def refresh_run_reports(
    jobs_dir: Path, tasks_dir: Path | None = None, *, quiet: bool = False
) -> tuple[int, int]:
    refreshed = 0
    skipped = 0
    job_dirs = _job_dirs_for_report_refresh(jobs_dir)
    total = len(job_dirs)
    for index, job_dir in enumerate(job_dirs, 1):
        report_path = run_report_output_path(job_dir)
        if not _job_has_completed_trials(job_dir):
            skipped += 1
            if not quiet:
                print(
                    f"[{index}/{total}] Skipping run report with no completed trials yet: {job_dir.name}"
                )
            continue
        if not _report_needs_refresh(job_dir, report_path):
            skipped += 1
            if not quiet:
                print(f"[{index}/{total}] Run report up to date: {job_dir.name}")
            continue

        if not quiet:
            print(f"[{index}/{total}] Refreshing run report: {job_dir.name}")
        run_report_module.write_report(
            job_dir,
            tasks_dir=tasks_dir,
            output=report_path,
        )
        refreshed += 1
    return refreshed, skipped


def write_report(
    jobs_dir: Path,
    tasks_dir: Path | None = None,
    output: Path | None = None,
    *,
    quiet: bool = False,
) -> Path:
    refreshed_reports, skipped_reports = refresh_run_reports(
        jobs_dir, tasks_dir=tasks_dir, quiet=quiet
    )
    categories = load_task_categories(tasks_dir) if tasks_dir and tasks_dir.exists() else {}
    trials = load_trials(jobs_dir)
    if not trials:
        print(f"No completed trials found in {jobs_dir}")
        raise SystemExit(1)

    # Check for mixed task checksums across models (indicates version mismatch).
    task_checksums: dict[str, set[str]] = {}
    for trial in trials:
        task_name = trial.get("task_name", "")
        checksum = trial.get("task_checksum", "")
        if task_name and checksum:
            task_checksums.setdefault(task_name, set()).add(checksum)
    mixed = {t: cs for t, cs in task_checksums.items() if len(cs) > 1}
    if mixed:
        import sys

        print(
            f"Warning: {len(mixed)} task(s) have mixed checksums across models "
            f"(results may not be comparable): {', '.join(sorted(mixed))}",
            file=sys.stderr,
        )

    data = aggregate(trials, categories)
    data["jobs_dir"] = str(jobs_dir.resolve())
    report_path = suite_report_output_path(jobs_dir, output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_html(data))
    print(f"Run reports refreshed: {refreshed_reports}, already current: {skipped_reports}")
    print(f"Report written to {report_path}")
    expected_shots_per_task = max((m["shots_per_task"] for m in data["models"]), default=1)
    expected_trials_per_variant = len(data["all_tasks"]) * expected_shots_per_task
    if not quiet:
        print(f"Using jobs dir {jobs_dir.resolve()}")
        print(
            f"Expected full run per variant: {len(data['all_tasks'])} tasks x {expected_shots_per_task} trials = {expected_trials_per_variant} trials"
        )

    print(
        f"\n{'Model':<35} {'Pass^k':>8} {'Pass@k':>8} {'Mean':>7} {'Tokens':>8} {'Cost':>8} {'Time':>6} {'Trials':>11}"
    )
    print("-" * 108)
    for m in data["models"]:
        print(
            f"{m['label']:<35} {pct(m['pass_hat_rate']):>8} {pct(m['pass_rate']):>8} {pct(m['mean_score']):>7} {format_compact_count(m['total_tokens']):>8} ${m['total_cost_usd']:>7.2f} {m['agent_mins']:>5.0f}m {format_trials(m['n_valid_trials'], m['expected_trials']):>11}"
        )
    return report_path


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate o11y-bench HTML leaderboard from Harbor jobs"
    )
    parser.add_argument(
        "--jobs-dir",
        default=None,
        help="Harbor jobs directory. When omitted, uses the latest jobs/full-suite-* directory.",
    )
    parser.add_argument(
        "--tasks-dir", default="tasks", help="Harbor tasks directory (default: tasks)"
    )
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output HTML file")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    jobs_dir = resolve_jobs_dir(args.jobs_dir, quiet=args.quiet)

    tasks_dir = Path(args.tasks_dir)
    write_report(jobs_dir, tasks_dir=tasks_dir, output=args.output, quiet=args.quiet)


if __name__ == "__main__":
    main()
