from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

from o11y_bench import agento11y_suite


@dataclass
class FakeCase:
    test_case_id: str
    name: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)
    category: str = ""
    input: object = None
    expected: object = None
    metadata: dict = field(default_factory=dict)


@dataclass
class FakeSuite:
    suite_id: str
    name: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)
    test_cases: list[FakeCase] = field(default_factory=list)

    @property
    def cases(self):
        return self.test_cases


def test_build_test_suite_uses_task_yaml_as_source_of_truth(monkeypatch, tmp_path):
    spec = tmp_path / "investigation" / "incident.yaml"
    spec.parent.mkdir()
    spec.write_text(
        """\
id: incident
category: investigation
tags: [metrics, logs]
statement: Investigate the incident.
checks:
  - name: queried_metrics
    weight: 1
rubric:
  - criterion: The answer identifies the cause.
    weight: 1
"""
    )
    fake = SimpleNamespace(TestCase=FakeCase, TestSuite=FakeSuite)
    monkeypatch.setattr(agento11y_suite, "_import_agento11y", lambda: fake)

    suite = agento11y_suite.build_test_suite(tmp_path)

    assert suite.suite_id == "o11y-bench"
    assert len(suite.cases) == 1
    case = suite.cases[0]
    assert case.test_case_id == "incident"
    assert case.input == "Investigate the incident."
    assert case.expected["checks"][0]["name"] == "queried_metrics"
    assert case.expected["rubric"][0]["criterion"] == "The answer identifies the cause."
    assert case.category == "investigation"
    assert case.tags == ["metrics", "logs"]


def test_build_test_suite_rejects_duplicate_ids(monkeypatch, tmp_path):
    for directory in ("one", "two"):
        path = tmp_path / directory / "task.yaml"
        path.parent.mkdir()
        path.write_text("id: duplicate\ncategory: test\nstatement: Run it.\n")
    fake = SimpleNamespace(TestCase=FakeCase, TestSuite=FakeSuite)
    monkeypatch.setattr(agento11y_suite, "_import_agento11y", lambda: fake)

    try:
        agento11y_suite.build_test_suite(tmp_path)
    except ValueError as exc:
        assert "duplicate task id" in str(exc)
    else:
        raise AssertionError("expected duplicate task ids to fail")


def test_suite_url_uses_configured_grafana_app_control_url(monkeypatch):
    monkeypatch.setenv(
        "AGENTO11Y_CONTROL_ENDPOINT",
        "https://dev.grafana-dev.net/a/grafana-agento11y-app",
    )

    assert agento11y_suite.test_suite_url("o11y-bench/jack") == (
        "https://dev.grafana-dev.net/a/grafana-agento11y-app/experiments/test-suites/o11y-bench%2Fjack"
    )


def test_suite_url_uses_grafana_url_for_direct_control_endpoint(monkeypatch):
    monkeypatch.setenv("AGENTO11Y_CONTROL_ENDPOINT", "http://agento11y:8080/api/v1/eval")
    monkeypatch.setenv("AGENTO11Y_GRAFANA_URL", "http://localhost:3000")

    assert agento11y_suite.test_suite_url("o11y-bench") == (
        "http://localhost:3000/a/grafana-agento11y-app/experiments/test-suites/o11y-bench"
    )
