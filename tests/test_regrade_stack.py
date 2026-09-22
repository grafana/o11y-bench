import urllib.error
from types import SimpleNamespace

import pytest

from o11y_bench import regrade_stack


def _fake_docker_port(monkeypatch, stdout: str) -> None:
    monkeypatch.setattr(
        regrade_stack.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout=stdout, stderr="", returncode=0),
    )


def test_docker_host_endpoint_uses_the_address_the_daemon_reported(monkeypatch) -> None:
    # A proxied or remote DOCKER_HOST can publish somewhere that is not this
    # process's loopback, so the bound address has to be read back, not assumed.
    _fake_docker_port(monkeypatch, "10.44.194.142:32769\n")
    assert regrade_stack.docker_host_endpoint("c", 8080) == "10.44.194.142:32769"


def test_docker_host_endpoint_prefers_ipv4_mapping(monkeypatch) -> None:
    _fake_docker_port(monkeypatch, "[::1]:32770\n127.0.0.1:32769\n")
    assert regrade_stack.docker_host_endpoint("c", 8080) == "127.0.0.1:32769"


def test_docker_host_endpoint_falls_back_to_ipv6_when_only_option(monkeypatch) -> None:
    _fake_docker_port(monkeypatch, "[::1]:32770\n")
    assert regrade_stack.docker_host_endpoint("c", 8080) == "::1:32770"


@pytest.mark.parametrize("wildcard", ["0.0.0.0", "::"])
def test_docker_host_endpoint_rewrites_wildcard_bind_to_loopback(monkeypatch, wildcard) -> None:
    _fake_docker_port(monkeypatch, f"{wildcard}:32769\n")
    assert regrade_stack.docker_host_endpoint("c", 8080) == "127.0.0.1:32769"


def test_docker_host_endpoint_connect_host_override(monkeypatch) -> None:
    _fake_docker_port(monkeypatch, "127.0.0.1:32769\n")
    monkeypatch.setenv("O11Y_REGRADE_CONNECT_HOST", "docker.internal")
    assert regrade_stack.docker_host_endpoint("c", 8080) == "docker.internal:32769"


def test_docker_host_endpoint_raises_when_nothing_published(monkeypatch) -> None:
    _fake_docker_port(monkeypatch, "\n")
    with pytest.raises(RuntimeError, match="no published mapping"):
        regrade_stack.docker_host_endpoint("c", 8080)


def test_regrade_bind_host_defaults_to_loopback(monkeypatch) -> None:
    monkeypatch.delenv("O11Y_REGRADE_BIND_HOST", raising=False)
    assert regrade_stack.regrade_bind_host() == "127.0.0.1"


def test_regrade_bind_host_empty_lets_daemon_choose(monkeypatch) -> None:
    monkeypatch.setenv("O11Y_REGRADE_BIND_HOST", "")
    assert regrade_stack.regrade_bind_host() == ""


def test_wait_for_http_ok_accepts_a_404(monkeypatch) -> None:
    # mcp-grafana serves on /mcp and 404s "/". urlopen raises on 4xx rather
    # than returning, so a 404 has to be caught to count as "listening".
    def raise_404(*a, **k):
        raise urllib.error.HTTPError("http://x/", 404, "Not Found", {}, None)

    monkeypatch.setattr(regrade_stack.urllib.request, "urlopen", raise_404)
    regrade_stack.wait_for_http_ok("http://x/", timeout_sec=1)


def test_wait_for_http_ok_rejects_a_500(monkeypatch) -> None:
    def raise_500(*a, **k):
        raise urllib.error.HTTPError("http://x/", 500, "Server Error", {}, None)

    monkeypatch.setattr(regrade_stack.urllib.request, "urlopen", raise_500)
    monkeypatch.setattr(regrade_stack.time, "sleep", lambda _s: None)
    with pytest.raises(TimeoutError):
        regrade_stack.wait_for_http_ok("http://x/", timeout_sec=0.01)


def test_wait_for_http_ok_times_out_when_nothing_listens(monkeypatch) -> None:
    def refuse(*a, **k):
        raise urllib.error.URLError("[Errno 111] Connection refused")

    monkeypatch.setattr(regrade_stack.urllib.request, "urlopen", refuse)
    monkeypatch.setattr(regrade_stack.time, "sleep", lambda _s: None)
    with pytest.raises(TimeoutError, match="Connection refused"):
        regrade_stack.wait_for_http_ok("http://x/", timeout_sec=0.01)
