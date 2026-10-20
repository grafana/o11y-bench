"""Harbor subprocess invocation and signal handling."""

import signal
import subprocess
import sys
import threading
from pathlib import Path
from types import FrameType

from .config import DEFAULT_AGENT_IMPORT_PATH, JOB_CONFIG, ROOT, TASKS_DIR, JobSpec

PREFLIGHT_SCRIPT = ROOT / "scripts" / "harbor_preflight.sh"


def builtin_prompt_template(spec: JobSpec) -> Path:
    """Where a built-in agent's prompt template lives, beside the job dir."""
    return spec.jobs_dir / f"{spec.job_name}.prompt.md"


def write_builtin_prompt_template(spec: JobSpec, scenario_time_iso: str) -> None:
    """Give a built-in agent (Claude Code, OpenCode) the prompt our agent sends, with the
    scenario clock: trials start hours after the clock is set, so real now is not it."""
    agents = ROOT / "agents"
    system = (agents / "system_prompt.txt").read_text().strip()
    task = (agents / "task_prompt.txt").read_text()
    path = builtin_prompt_template(spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    prompt = f"{system}\n\n" + task.format(current_time=scenario_time_iso, statement="\0")
    # Harbor renders the file with Jinja: keep our text literal and leave only the instruction.
    path.write_text(
        "{% raw %}"
        + prompt.replace("\0", "{% endraw %}{{ instruction }}{% raw %}")
        + "{% endraw %}"
    )


def build_command(spec: JobSpec, *, quiet: bool = False) -> list[str]:
    """Build a `harbor run` command from a JobSpec (used by suite orchestration)."""
    command = [
        "uv",
        "run",
        "harbor",
        "run",
        "--yes",
        "--config",
        str(JOB_CONFIG),
        "--path",
        str(spec.tasks_dir),
        # Harbor's --agent takes a built-in name or an import path.
        "--agent",
        spec.agent or spec.agent_import_path,
        "--jobs-dir",
        str(spec.jobs_dir),
        "--job-name",
        spec.job_name,
        "--model",
        spec.model,
        "--n-attempts",
        str(spec.n_attempts),
        "--n-concurrent",
        str(spec.n_concurrent),
    ]
    # Harbor rejects --ak kwargs outside an agent's schema, and not every agent takes
    # reasoning_effort (or "off"), so only pass a real level.
    if spec.reasoning_effort != "off":
        command.extend(["--ak", f"reasoning_effort={spec.reasoning_effort}"])
    if spec.agent:
        command.extend(["--ak", f"prompt_template_path={builtin_prompt_template(spec)}"])
    for task_name in spec.task_names:
        command.extend(["--include-task-name", task_name])
    if spec.override_cpus is not None:
        command.extend(["--override-cpus", str(spec.override_cpus)])
    if spec.override_memory_mb is not None:
        command.extend(["--override-memory-mb", str(spec.override_memory_mb)])
    if spec.override_storage_mb is not None:
        command.extend(["--override-storage-mb", str(spec.override_storage_mb)])
    command.extend(spec.harbor_args)
    if quiet:
        command.append("--quiet")
    return command


def build_resume_command(
    config_path: str, *, quiet: bool = False, harbor_args: tuple[str, ...] = ()
) -> list[str]:
    """Build a `harbor run` command that resumes an existing job from its saved config."""
    command = [
        "uv",
        "run",
        "harbor",
        "run",
        "--yes",
        "--config",
        config_path,
    ]
    command.extend(harbor_args)
    if quiet:
        command.append("--quiet")
    return command


def build_command_from_args(harbor_args: list[str], *, quiet: bool = False) -> list[str]:
    """Build a `harbor run` command from raw CLI args, adding repo defaults."""
    command = ["uv", "run", "harbor", "run"]
    if not any(name in harbor_args for name in ("--yes", "-y")):
        command.append("--yes")
    if not any(name in harbor_args for name in ("--config", "-c")):
        command.extend(["--config", str(JOB_CONFIG)])
    if not any(name in harbor_args for name in ("--path", "-p", "--dataset", "-d")):
        command.extend(["--path", str(TASKS_DIR)])
    if not any(name in harbor_args for name in ("--agent", "-a", "--agent-import-path")):
        command.extend(["--agent", DEFAULT_AGENT_IMPORT_PATH])
    if quiet and not any(name in harbor_args for name in ("--quiet", "-q")):
        command.append("--quiet")
    command.extend(harbor_args)
    return command


def run(command: list[str], *, forward_signals: bool = True) -> int:
    """Run Harbor as a subprocess, forwarding SIGINT/SIGTERM for graceful shutdown."""
    if forward_signals and threading.current_thread() is not threading.main_thread():
        raise RuntimeError(
            "signal forwarding requires the main thread; "
            "call run(..., forward_signals=False) from a worker thread"
        )

    process = subprocess.Popen(command)
    if not forward_signals:
        exit_code = process.wait()
        run_cleanup()
        return exit_code

    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)

    def forward(signum: int, _frame: FrameType | None) -> None:
        process.send_signal(signum)

    signal.signal(signal.SIGINT, forward)
    signal.signal(signal.SIGTERM, forward)
    try:
        return process.wait()
    finally:
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)
        run_cleanup()


def run_preflight(*, quiet: bool = False) -> None:
    """Clean stale Docker projects and pre-build shared images."""
    result = subprocess.run(
        ["bash", str(PREFLIGHT_SCRIPT)],
        check=False,
        capture_output=quiet,
        text=quiet,
    )
    if result.returncode != 0:
        if quiet:
            output = (result.stderr or result.stdout or "").strip()
            if output:
                print(output, file=sys.stderr)
        print("Harbor preflight failed", file=sys.stderr)
        raise SystemExit(result.returncode)


def run_cleanup(*, quiet: bool = True) -> None:
    """Best-effort cleanup of stale Harbor Docker projects."""
    result = subprocess.run(
        ["bash", str(PREFLIGHT_SCRIPT), "--cleanup-only"],
        check=False,
        capture_output=quiet,
        text=quiet,
    )
    if result.returncode != 0:
        if quiet:
            output = (result.stderr or result.stdout or "").strip()
            if output:
                print(output, file=sys.stderr)
        print("Harbor cleanup failed", file=sys.stderr)
