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

    bind_host = regrade_bind_host()
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
        *[
            arg
            for port in (3000, 9090, 3100, 3200, 8080)
            for arg in ("-p", f"{bind_host}::{port}" if bind_host else f"{port}")
        ],
        image,
    ]
    subprocess.run(["docker", "rm", "-f", container_name], check=False, capture_output=True)
    subprocess.run(command, check=True, capture_output=True, text=True)

    previous_env = {name: os.environ.get(name) for name in _STACK_ENV_NAMES}
    try:
        ports = {
            "GRAFANA_URL": f"http://{docker_host_endpoint(container_name, 3000)}",
            "PROMETHEUS_URL": f"http://{docker_host_endpoint(container_name, 9090)}",
            "LOKI_URL": f"http://{docker_host_endpoint(container_name, 3100)}",
            "TEMPO_URL": f"http://{docker_host_endpoint(container_name, 3200)}",
            "MCP_URL": f"http://{docker_host_endpoint(container_name, 8080)}/mcp",
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


def regrade_bind_host() -> str:
    """Address the sidecar publishes on.

    Defaults to loopback. Set O11Y_REGRADE_BIND_HOST to override, or to the empty
    string to let the daemon choose — needed when DOCKER_HOST points at a daemon
    that does not share this process's loopback (a remote or proxied socket),
    where a loopback-bound publish is unreachable.
    """
    return os.environ.get("O11Y_REGRADE_BIND_HOST", "127.0.0.1").strip()


def docker_host_endpoint(container_name: str, container_port: int) -> str:
    """Return "host:port" for a published port, as the daemon actually bound it.

    The bound address is read back rather than assumed: with a remote or proxied
    DOCKER_HOST the daemon may publish on an address that is not this process's
    loopback, and connecting to 127.0.0.1 then fails with connection refused.
    """
    result = subprocess.run(
        ["docker", "port", container_name, f"{container_port}/tcp"],
        check=True,
        capture_output=True,
        text=True,
    )
    mappings = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
    if not mappings:
        raise RuntimeError(f"{container_name} has no published mapping for {container_port}/tcp")

    # Prefer IPv4; a bracketed IPv6 mapping is only usable as a fallback.
    mapping = next((m for m in mappings if not m.startswith("[")), mappings[0])
    host, _, port = mapping.rpartition(":")
    host = host.strip("[]")

    # A wildcard bind is not a connectable address.
    if host in ("0.0.0.0", "::", ""):
        host = "127.0.0.1"
    override = os.environ.get("O11Y_REGRADE_CONNECT_HOST", "").strip()
    if override:
        host = override
    return f"{host}:{int(port)}"


def wait_for_http_ok(url: str, *, timeout_sec: float) -> None:
    deadline = time.monotonic() + timeout_sec
    last_error = ""
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if 200 <= response.status < 500:
                    return
        except urllib.error.HTTPError as exc:
            # Any status the server answers means it is listening, which is all
            # this waits for. mcp-grafana serves on /mcp and 404s "/", and
            # urlopen raises on 4xx instead of returning, so this has to be
            # caught rather than read off the response.
            if 200 <= exc.code < 500:
                return
            last_error = str(exc)
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
