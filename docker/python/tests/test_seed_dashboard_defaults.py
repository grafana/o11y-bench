import pytest
from o11y_stack.provision_task_resources import build_dashboard

from grading.dashboard_diff import seed_dashboard

# What Grafana, not the task, assigns when it stores a dashboard.
STORE_ASSIGNED = {"id", "version", "schemaVersion"}


@pytest.mark.parametrize(
    "payload",
    [{}, {"uid": "svc", "title": "Svc", "folderId": 5, "tags": ["a"], "panels": [{"id": 1}]}],
    ids=["empty", "with-overrides-and-folder"],
)
def test_the_diff_baseline_matches_the_dashboard_the_provisioner_stores(payload: dict) -> None:
    # The judge diffs the saved dashboard against `seed_dashboard`; if the provisioner fills in a
    # default it does not, every untouched dashboard reads as edited.
    stored = {k: v for k, v in build_dashboard(payload).items() if k not in STORE_ASSIGNED}

    assert seed_dashboard(payload) == stored
