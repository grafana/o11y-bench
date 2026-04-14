import json
from pathlib import Path

from o11y_stack import provision_task_resources


class _Response:
    def __init__(self, body: str = "{}", code: int = 200):
        self._body = body
        self._code = code

    def read(self) -> bytes:
        return self._body.encode()

    def getcode(self) -> int:
        return self._code

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_provision_task_resources_posts_dashboards(monkeypatch, tmp_path: Path, capsys) -> None:
    setup_path = tmp_path / "setup.json"
    setup_path.write_text(
        json.dumps(
            {
                "setup_dashboards": [{"uid": "service-overview", "title": "Service Overview"}],
            }
        )
    )

    seen_urls: list[str] = []

    def _urlopen(request_or_url, timeout=0):
        if hasattr(request_or_url, "full_url"):
            seen_urls.append(request_or_url.full_url)
            return _Response('{"status":"ok"}')
        seen_urls.append(request_or_url)
        return _Response(code=200)

    monkeypatch.setattr(provision_task_resources.urllib.request, "urlopen", _urlopen)

    provision_task_resources.provision_task_resources(str(setup_path))

    assert f"{provision_task_resources.GRAFANA_URL}/api/dashboards/db" in seen_urls
    assert (
        f"{provision_task_resources.GRAFANA_URL}/api/dashboards/uid/service-overview" in seen_urls
    )

    out = capsys.readouterr().out
    assert "Provisioned dashboard service-overview" in out
    assert "Verified dashboard uid=service-overview is readable" in out
