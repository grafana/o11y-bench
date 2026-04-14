"""Provision task-scoped Grafana resources from setup.json."""

import json
import sys
import time
import urllib.error
import urllib.request

GRAFANA_URL = "http://localhost:3000"


def build_dashboard(payload: dict) -> dict:
    dashboard = {
        "id": None,
        "editable": True,
        "schemaVersion": 41,
        "version": 0,
        "refresh": "",
        "timezone": "browser",
        "tags": [],
        "time": {"from": "now-6h", "to": "now"},
        "templating": {"list": []},
        "annotations": {"list": []},
        "panels": [],
    }
    dashboard.update({key: value for key, value in payload.items() if key != "folderId"})
    dashboard["id"] = None
    return dashboard


def post_json(url: str, payload: dict) -> str:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.read().decode()


def wait_dashboard_visible(uid: str, timeout_sec: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_sec
    url = f"{GRAFANA_URL}/api/dashboards/uid/{uid}"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.getcode() == 200:
                    print(f"Verified dashboard uid={uid} is readable")
                    return
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                time.sleep(0.5)
                continue
            raise
        except OSError:
            time.sleep(0.5)
    raise RuntimeError(f"dashboard {uid!r} not visible via GET after provision")


def provision_task_resources(setup_path: str) -> None:
    with open(setup_path) as f:
        setup = json.load(f)

    dashboards = setup.get("setup_dashboards") or []

    if not dashboards:
        print("No task Grafana resources to provision")
        return

    for payload in dashboards:
        body = {
            "dashboard": build_dashboard(payload),
            "overwrite": True,
            "message": "o11y-bench task setup",
            "folderId": payload.get("folderId", 0),
        }
        response = post_json(f"{GRAFANA_URL}/api/dashboards/db", body)
        print(f"Provisioned dashboard {payload['uid']}: {response}")
        wait_dashboard_visible(payload["uid"])


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m o11y_stack.provision_task_resources <setup.json>")
    provision_task_resources(sys.argv[1])


if __name__ == "__main__":
    main()
