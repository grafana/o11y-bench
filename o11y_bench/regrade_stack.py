import os
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from grading.models import (
    DashboardStateParams,
    DatasourceDetailStateParams,
    DatasourceInventoryStateParams,
    Problem,
    TempoTraceServiceInventoryStateParams,
    ToolTraceIdGroundingParams,
)


def problem_requires_live_stack(problem: Problem) -> bool:
    if any(item.fact is not None for item in problem.rubric):
        return True
    return any(check_requires_live_stack(check.params) for check in problem.checks)


def check_requires_live_stack(params: object) -> bool:
    match params:
        case ToolTraceIdGroundingParams():
            return False
        case (
            DashboardStateParams()
            | DatasourceInventoryStateParams()
            | DatasourceDetailStateParams()
            | TempoTraceServiceInventoryStateParams()
        ):
            return True
        case _:
            return True


@contextmanager
def running_regrade_stack(
    *,
    task_dir: Path,
    trial_dir: Path,
    scenario_time_iso: str,
    image: str = "o11y-bench-o11y-stack:latest",
    timeout_sec: float = 150.0,
) -> Iterator[None]:
    setup_path = task_dir / "environment" / "setup.json"
    artifacts_dir = trial_dir / "artifacts" / "regrade-sidecar"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    container_name = f"o11y-regrade-{trial_dir.name.lower()}-{uuid4().hex[:8]}"

    command = [
        "docker",
        "run",
        "-d",
        "--rm",
        "--name",
        container_name,
        "-e",
        f"O11Y_SCENARIO_TIME_ISO={scenario_time_iso}",
        "-v",
        f"{setup_path.resolve()}:/task/setup.json:ro",
        "-v",
        f"{artifacts_dir.resolve()}:/logs/artifacts",
        "-p",
        "127.0.0.1::3000",
        "-p",
        "127.0.0.1::9090",
        "-p",
        "127.0.0.1::3100",
        "-p",
        "127.0.0.1::3200",
        "-p",
        "127.0.0.1::8080",
        image,
    ]
    subprocess.run(["docker", "rm", "-f", container_name], check=False, capture_output=True)
    subprocess.run(command, check=True, capture_output=True, text=True)

    previous_env = {name: os.environ.get(name) for name in _STACK_ENV_NAMES}
    try:
        ports = {
            "GRAFANA_URL": f"http://127.0.0.1:{docker_host_port(container_name, 3000)}",
            "PROMETHEUS_URL": f"http://127.0.0.1:{docker_host_port(container_name, 9090)}",
            "LOKI_URL": f"http://127.0.0.1:{docker_host_port(container_name, 3100)}",
            "TEMPO_URL": f"http://127.0.0.1:{docker_host_port(container_name, 3200)}",
            "MCP_URL": f"http://127.0.0.1:{docker_host_port(container_name, 8080)}/mcp",
        }
        wait_for_http_ok(f"{ports['MCP_URL'][:-4]}/", timeout_sec=timeout_sec)
        os.environ.update(ports)
        yield
    finally:
        save_container_logs(container_name, artifacts_dir / "docker.log")
        subprocess.run(["docker", "rm", "-f", container_name], check=False, capture_output=True)
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def docker_host_port(container_name: str, container_port: int) -> int:
    result = subprocess.run(
        ["docker", "port", container_name, f"{container_port}/tcp"],
        check=True,
        capture_output=True,
        text=True,
    )
    mapping = result.stdout.strip().splitlines()[0].strip()
    return int(mapping.rsplit(":", 1)[1])


def wait_for_http_ok(url: str, *, timeout_sec: float) -> None:
    deadline = time.monotonic() + timeout_sec
    last_error = ""
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if 200 <= response.status < 500:
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for stack at {url}: {last_error or 'no response'}")


def save_container_logs(container_name: str, output_path: Path) -> None:
    result = subprocess.run(
        ["docker", "logs", container_name],
        check=False,
        capture_output=True,
        text=True,
    )
    output_path.write_text((result.stdout or "") + (result.stderr or ""))


_STACK_ENV_NAMES = ("GRAFANA_URL", "PROMETHEUS_URL", "LOKI_URL", "TEMPO_URL", "MCP_URL")
