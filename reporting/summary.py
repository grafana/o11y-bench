#!/usr/bin/env python3
"""Shared aggregation helpers for Harbor report scripts."""

from collections import defaultdict
from typing import TypedDict


class TrialRow(TypedDict):
    task_name: str
    score: float | None
    cost_usd: float
    agent_secs: float
    n_input_tokens: int
    n_output_tokens: int
    n_cache_tokens: int
    tool_calls: int
    invalid_infra: bool
    counts_as_pass: bool


class TaskSummary(TypedDict):
    scores: list[float | None]
    passed: bool
    consistent: bool
    mean_score: float
    best_score: float
    cost_usd: float


class TrialsSummary(TypedDict):
    per_task: dict[str, TaskSummary]
    n_tasks: int
    n_valid_trials: int
    n_passed: int
    n_consistent: int
    pass_rate: float
    pass_hat_rate: float
    mean_score: float
    total_cost_usd: float
    total_agent_secs: float
    total_tokens_in: int
    total_tokens_out: int
    total_tokens_cache: int
    total_tool_calls: int
    shots_per_task: int
    steps_per_trial: float


def summarize_trials(rows: list[TrialRow]) -> TrialsSummary:
    by_task: dict[str, list[TrialRow]] = defaultdict(list)
    for row in rows:
        by_task[row["task_name"]].append(row)

    valid_rows = [row for row in rows if not row["invalid_infra"]]

    per_task: dict[str, TaskSummary] = {}
    for task_name, task_rows in by_task.items():
        valid_task_rows = [row for row in task_rows if not row["invalid_infra"]]
        if not valid_task_rows:
            continue

        scores = [row["score"] for row in valid_task_rows]
        valid_scores = [score for score in scores if score is not None]
        per_task[task_name] = {
            "scores": scores,
            "passed": any(row["counts_as_pass"] for row in valid_task_rows),
            "consistent": bool(scores)
            and len(valid_scores) == len(scores)
            and all(row["counts_as_pass"] for row in valid_task_rows),
            "mean_score": sum(valid_scores) / len(valid_scores) if valid_scores else 0.0,
            "best_score": max(valid_scores) if valid_scores else 0.0,
            "cost_usd": sum(row["cost_usd"] for row in valid_task_rows),
        }

    total_cost_usd = sum(row["cost_usd"] for row in valid_rows)
    total_agent_secs = sum(row["agent_secs"] for row in valid_rows)
    total_tokens_in = sum(row["n_input_tokens"] for row in valid_rows)
    total_tokens_out = sum(row["n_output_tokens"] for row in valid_rows)
    total_tokens_cache = sum(row["n_cache_tokens"] for row in valid_rows)
    total_tool_calls = sum(row["tool_calls"] for row in valid_rows)
    n_tasks = len(per_task)
    n_passed = sum(1 for stats in per_task.values() if stats["passed"])
    n_consistent = sum(1 for stats in per_task.values() if stats["consistent"])
    mean_score = (
        sum(row["score"] if row["score"] is not None else 0.0 for row in valid_rows)
        / len(valid_rows)
        if valid_rows
        else 0.0
    )
    shots_per_task = max((len(stats["scores"]) for stats in per_task.values()), default=1)
    pass_rate = n_passed / n_tasks if n_tasks else 0.0
    pass_hat_rate = n_consistent / n_tasks if n_tasks else 0.0
    steps_per_trial = total_tool_calls / len(valid_rows) if valid_rows else 0.0

    return {
        "per_task": per_task,
        "n_tasks": n_tasks,
        "n_valid_trials": len(valid_rows),
        "n_passed": n_passed,
        "n_consistent": n_consistent,
        "pass_rate": pass_rate,
        "pass_hat_rate": pass_hat_rate,
        "mean_score": mean_score,
        "total_cost_usd": total_cost_usd,
        "total_agent_secs": total_agent_secs,
        "total_tokens_in": total_tokens_in,
        "total_tokens_out": total_tokens_out,
        "total_tokens_cache": total_tokens_cache,
        "total_tool_calls": total_tool_calls,
        "shots_per_task": shots_per_task,
        "steps_per_trial": steps_per_trial,
    }
