from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from o11y_bench import agento11y_publish


@dataclass
class FakeEvaluator:
    evaluator_id: str
    version: str
    kind: str


@dataclass
class FakeEvaluationResult:
    evaluator: FakeEvaluator
    value: object
    passed: bool
    explanation: str = ""
    score_key: str = "final"
    metadata: dict | None = None


class FakeEvaluatorKind:
    CUSTOM = SimpleNamespace(value="custom")
    DETERMINISTIC = SimpleNamespace(value="deterministic")
    LLM_JUDGE = SimpleNamespace(value="llm_judge")


class FakeClient:
    def __init__(self):
        self.generations = []
        self.generation_flushes = 0
        self.trial_updates = []

    def record_generation(self, generation_id, **kwargs):
        self.generations.append((generation_id, kwargs))

    def flush_generations(self):
        self.generation_flushes += 1

    def update_trial(self, experiment_id, trial_id, **kwargs):
        self.trial_updates.append((experiment_id, trial_id, kwargs))


class FakeTrial:
    def __init__(self, owner, attempt):
        self.owner = owner
        self.attempt = attempt
        self.trial_id = f"trial-{attempt}"
        self.pending = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def bind_generation(self, generation_id, *, conversation_id):
        self.owner.bound.append((generation_id, conversation_id))

    def set_usage(self, **usage):
        self.owner.usage.append(usage)

    def record_evaluation(self, result):
        self.owner.events.append(("record", result.score_key, result.evaluator.kind))
        self.pending += 1

    def flush(self):
        self.owner.events.append(("flush", self.pending))
        count = self.pending
        self.pending = 0
        return count

    def artifact(self, name, *, path):
        self.owner.artifacts.append((name, path))


class FakeExperiment:
    def __init__(self):
        self.client = FakeClient()
        self.suite = SimpleNamespace(
            cases=[
                SimpleNamespace(
                    test_case_id="task-a",
                    input="Investigate.",
                    category="investigation",
                )
            ]
        )
        self.events = []
        self.bound = []
        self.usage = []
        self.artifacts = []
        self.exit = None
        self.url = "https://example.test/experiment/run-1"
        self.create_kwargs = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, _tb):
        self.exit = (exc_type, exc)
        return False

    def trial(self, _case, *, attempt, metadata):
        assert metadata["harbor_trial_id"] == "task-a__abc"
        return FakeTrial(self, attempt)


def _fake_modules(experiment):
    def experiment_from_suite(*_args, **kwargs):
        experiment.create_kwargs = kwargs
        return experiment

    sdk = SimpleNamespace(TokenUsage=lambda **kwargs: kwargs)
    experiments = SimpleNamespace(
        EvaluationResult=FakeEvaluationResult,
        Evaluator=FakeEvaluator,
        EvaluatorKind=FakeEvaluatorKind,
        experiment_from_suite=experiment_from_suite,
        stable_id=lambda prefix, *_parts: f"{prefix}-stable",
    )
    return sdk, experiments


def _write_trial(tmp_path):
    tasks_dir = tmp_path / "tasks"
    task_dir = tasks_dir / "task-a"
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "tests" / "problem.yaml").write_text(
        """\
id: task-a
category: investigation
statement: Investigate.
checks:
  - name: queried_metrics
    weight: 1
rubric:
  - criterion: grounded answer
    weight: 1
"""
    )
    job_dir = tmp_path / "jobs" / "job-a"
    trial_dir = job_dir / "task-a__abc"
    (trial_dir / "verifier").mkdir(parents=True)
    (trial_dir / "config.json").write_text(json.dumps({"task": {"path": str(task_dir)}}))
    (trial_dir / "verifier" / "grading_details.json").write_text(
        json.dumps(
            {
                "queried_metrics": 1.0,
                "grounded answer": 1.0,
                "explanation:queried_metrics": "metric query found",
                "explanation:grounded answer": "answer used evidence",
            }
        )
    )
    result = {
        "__result_path": str(trial_dir / "result.json"),
        "task_name": "task-a",
        "task_checksum": "checksum-1",
        "agent_info": {"model_info": {"provider": "anthropic", "name": "claude-test"}},
        "agent_result": {
            "n_input_tokens": 10,
            "n_cache_tokens": 3,
            "n_output_tokens": 5,
            "cost_usd": 0.01,
            "metadata": {"reasoning_effort": "high"},
        },
        "agent_execution": {
            "started_at": "2026-01-01T00:00:00Z",
            "finished_at": "2026-01-01T00:00:02Z",
        },
        "verifier_result": {
            "rewards": {
                "reward": 0.75,
            }
        },
    }
    return job_dir, tasks_dir, result


def test_live_publisher_flushes_every_score_and_preserves_evaluator_kind(monkeypatch, tmp_path):
    job_dir, tasks_dir, result = _write_trial(tmp_path)
    experiment = FakeExperiment()
    monkeypatch.setattr(agento11y_publish, "_import_agento11y", lambda: _fake_modules(experiment))
    monkeypatch.setattr(agento11y_publish, "load_trials", lambda _path: [result])
    monkeypatch.setattr(agento11y_publish, "_final_output", lambda _path: "Final answer")
    monkeypatch.setattr(agento11y_publish, "_artifact_paths", lambda _path: [])

    publisher = agento11y_publish.Agento11yLivePublisher(
        job_dir,
        tasks_dir,
        model="anthropic/claude-test",
        agent_name="o11y-bench",
        agent_version="test",
        options=agento11y_publish.Agento11yPublishOptions(
            experiment_id="run-1", suite_version="v1"
        ),
        planned_trial_count=3,
    )
    publisher._connect()
    publisher.scan_once()

    assert experiment.create_kwargs["planned_trial_count"] == 3
    assert experiment.events == [
        ("record", "final", "custom"),
        ("flush", 1),
        ("record", "check.queried-metrics", "deterministic"),
        ("flush", 1),
        ("record", "rubric.grounded-answer", "llm_judge"),
        ("flush", 1),
    ]
    assert publisher._score_count == 3
    assert publisher._trial_count == 1
    assert experiment.client.generation_flushes == 1
    assert experiment.usage == [{"cost": 0.01}]
    assert experiment.client.trial_updates[0][2]["duration_ms"] == 2000
    assert "input_tokens" not in experiment.client.trial_updates[0][2]

    state = json.loads((job_dir / ".agento11y-publish.json").read_text())
    assert state["attempts"]["task-a\u001ftask-a__abc"] == 1
    assert state["exported_trials"] == ["task-a__abc"]


def test_publisher_reuses_persisted_attempt_mapping(monkeypatch, tmp_path):
    job_dir, tasks_dir, result = _write_trial(tmp_path)
    experiment = FakeExperiment()
    monkeypatch.setattr(agento11y_publish, "_import_agento11y", lambda: _fake_modules(experiment))
    monkeypatch.setattr(agento11y_publish, "load_trials", lambda _path: [result])
    monkeypatch.setattr(agento11y_publish, "_final_output", lambda _path: "Final answer")
    monkeypatch.setattr(agento11y_publish, "_artifact_paths", lambda _path: [])
    options = agento11y_publish.Agento11yPublishOptions(experiment_id="run-1", suite_version="v1")
    first = agento11y_publish.Agento11yLivePublisher(
        job_dir,
        tasks_dir,
        model="anthropic/claude-test",
        agent_name="o11y-bench",
        agent_version="test",
        options=options,
    )
    first._connect()
    first.scan_once()

    second = agento11y_publish.Agento11yLivePublisher(
        job_dir,
        tasks_dir,
        model="anthropic/claude-test",
        agent_name="o11y-bench",
        agent_version="test",
        options=options,
    )
    assert second._attempt_for("task-a", "task-a__abc") == 1
    second._connect()
    second.scan_once()
    assert second._trial_count == 1


def test_publisher_fallback_url_uses_current_experiment_route(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENTO11Y_GRAFANA_URL", "https://example.grafana.net/")
    publisher = agento11y_publish.Agento11yLivePublisher(
        tmp_path / "job",
        tmp_path / "tasks",
        model="anthropic/claude-test",
        agent_name="o11y-bench",
        agent_version="test",
        options=agento11y_publish.Agento11yPublishOptions(experiment_id="run-1"),
    )

    assert publisher.url == (
        "https://example.grafana.net/a/grafana-agento11y-app/experiments/runs/run-1"
    )


def test_publisher_error_marks_experiment_failed(monkeypatch, tmp_path):
    experiment = FakeExperiment()
    publisher = agento11y_publish.Agento11yLivePublisher(
        tmp_path / "job",
        tmp_path / "tasks",
        model="anthropic/claude-test",
        agent_name="o11y-bench",
        agent_version="test",
        options=agento11y_publish.Agento11yPublishOptions(
            experiment_id="run-1", suite_version="v1"
        ),
    )
    publisher._experiment = experiment
    publisher._error = ValueError("score export failed")
    monkeypatch.setattr(publisher, "scan_once", lambda: None)

    with pytest.raises(RuntimeError, match="score export failed"):
        publisher.finish(succeeded=True)

    assert experiment.exit is not None
    assert experiment.exit[0] is RuntimeError
    assert "score export failed" in str(experiment.exit[1])


def test_long_score_slugs_are_bounded_and_collision_resistant():
    common = "A very long criterion " * 10
    first = agento11y_publish._slug(common + "first")
    second = agento11y_publish._slug(common + "second")

    assert len(first) == 80
    assert len(second) == 80
    assert first != second
