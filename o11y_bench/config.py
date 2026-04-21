"""Shared constants, dataclasses, and suite configuration."""

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from reporting.report_paths import (
    SUITE_PREFIX,
    is_suite_dir,
    latest_suite_dir,
    normalize_repo_path,
)

ROOT = Path(__file__).resolve().parent.parent
JOB_CONFIG = ROOT / "job.yaml"
TASKS_DIR = ROOT / "tasks"
DEFAULT_AGENT_IMPORT_PATH = "agents.o11y_agent:O11yBenchAgent"
DEFAULT_N_ATTEMPTS = 3
DEFAULT_N_CONCURRENT = 8

STANDARD_SUITE: list[tuple[str, str, str]] = [
    ("anthropic", "claude-haiku-4-5-20251001", "off"),
    ("anthropic", "claude-haiku-4-5-20251001", "low"),
    ("anthropic", "claude-haiku-4-5-20251001", "high"),
    ("anthropic", "claude-opus-4-6", "off"),
    ("anthropic", "claude-opus-4-6", "low"),
    ("anthropic", "claude-opus-4-6", "high"),
    ("anthropic", "claude-opus-4-7", "off"),
    ("anthropic", "claude-opus-4-7", "low"),
    ("anthropic", "claude-opus-4-7", "high"),
    ("anthropic", "claude-sonnet-4-5", "off"),
    ("anthropic", "claude-sonnet-4-5", "high"),
    ("anthropic", "claude-sonnet-4-6", "off"),
    ("anthropic", "claude-sonnet-4-6", "low"),
    ("anthropic", "claude-sonnet-4-6", "high"),
    ("openai", "gpt-5.1-codex-mini", "off"),
    ("openai", "gpt-5.1-codex-mini", "high"),
    ("openai", "gpt-5.2-codex", "off"),
    ("openai", "gpt-5.2-codex", "high"),
    ("openai", "gpt-5.2-2025-12-11", "off"),
    ("openai", "gpt-5.2-2025-12-11", "high"),
    ("openai", "gpt-5.4-2026-03-05", "off"),
    ("openai", "gpt-5.4-2026-03-05", "low"),
    ("openai", "gpt-5.4-2026-03-05", "high"),
    ("openai", "gpt-5.4-mini", "off"),
    ("openai", "gpt-5.4-mini", "low"),
    ("openai", "gpt-5.4-mini", "high"),
    ("openai", "gpt-5.4-nano", "off"),
    ("openai", "gpt-5.4-nano", "low"),
    ("openai", "gpt-5.4-nano", "high"),
    ("google", "gemini-3-flash-preview", "off"),
    ("google", "gemini-3-flash-preview", "high"),
    ("google", "gemini-3.1-pro-preview", "off"),
    ("google", "gemini-3.1-pro-preview", "low"),
    ("google", "gemini-3.1-pro-preview", "high"),
    ("google", "gemini-3.1-flash-lite-preview", "off"),
    ("google", "gemini-3.1-flash-lite-preview", "low"),
    ("google", "gemini-3.1-flash-lite-preview", "high"),
]
PROVIDERS = tuple(sorted({provider for provider, _, _ in STANDARD_SUITE}))


@dataclass(frozen=True)
class JobSpec:
    """Everything needed to plan, run, and finalize a single Harbor job."""

    jobs_dir: Path
    job_name: str
    tasks_dir: Path
    model: str
    reasoning_effort: str
    n_attempts: int
    n_concurrent: int
    agent_import_path: str = DEFAULT_AGENT_IMPORT_PATH
    agent: str | None = None
    override_cpus: int | None = None
    override_memory_mb: int | None = None
    override_storage_mb: int | None = None
    task_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class SuiteOpts:
    """Options for a full suite run."""

    jobs_dir: Path | None
    resume: bool
    dry_run: bool
    quiet: bool
    n_attempts: int
    n_concurrent: int
    tasks_dir: Path
    override_cpus: int | None = None
    override_memory_mb: int | None = None
    override_storage_mb: int | None = None


def load_job_config_overrides() -> dict[str, Any]:
    data = yaml.safe_load(JOB_CONFIG.read_text()) or {}
    return data if isinstance(data, dict) else {}


def provider_variants(provider: str) -> list[tuple[str, str]]:
    return [
        (model, reasoning_effort)
        for item_provider, model, reasoning_effort in STANDARD_SUITE
        if item_provider == provider
    ]


def make_job_name(
    provider: str,
    model: str,
    reasoning_effort: str,
    n_attempts: int,
    *,
    agent: str | None = None,
    agent_import_path: str = DEFAULT_AGENT_IMPORT_PATH,
) -> str:
    name = f"{provider}-{model}-{reasoning_effort}"
    if agent:
        name = f"{name}-{agent}"
    elif agent_import_path != DEFAULT_AGENT_IMPORT_PATH:
        variant = re.sub(r"[^A-Za-z0-9]+", "-", agent_import_path).strip("-").lower()
        name = f"{name}-{variant}"
    name = f"{name}-k{n_attempts}"
    return name.replace("/", "-").replace(".", "-")


def default_suite_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{SUITE_PREFIX}{timestamp}"


def choose_suite_dir(jobs_root: Path, allow_resume: bool) -> Path:
    if allow_resume:
        suite_dir = latest_suite_dir(jobs_root)
        if suite_dir is not None:
            return suite_dir
    return jobs_root / default_suite_id()


def resolve_suite_dir(jobs_dir: Path | None, allow_resume: bool) -> Path:
    if jobs_dir is None:
        return choose_suite_dir(ROOT / "jobs", allow_resume=allow_resume)

    requested = normalize_repo_path(ROOT, jobs_dir)
    if requested.exists() and not requested.is_dir():
        raise SystemExit(f"--jobs-dir must be a directory: {requested}")
    if is_suite_dir(requested):
        return requested
    return choose_suite_dir(requested, allow_resume=allow_resume)


def build_suite_job_spec(
    suite_dir: Path,
    provider: str,
    model: str,
    reasoning_effort: str,
    opts: SuiteOpts,
) -> JobSpec:
    return JobSpec(
        jobs_dir=suite_dir,
        job_name=make_job_name(provider, model, reasoning_effort, opts.n_attempts),
        tasks_dir=opts.tasks_dir,
        model=f"{provider}/{model}",
        reasoning_effort=reasoning_effort,
        n_attempts=opts.n_attempts,
        n_concurrent=opts.n_concurrent,
        override_cpus=opts.override_cpus,
        override_memory_mb=opts.override_memory_mb,
        override_storage_mb=opts.override_storage_mb,
    )
