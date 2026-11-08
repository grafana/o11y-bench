from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

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
        self.shutdown_calls = 0

    def record_generation(self, generation_id, **kwargs):
        self.generations.append((generation_id, kwargs))

    def flush_generations(self):
        self.generation_flushes += 1

    def update_trial(self, experiment_id, trial_id, **kwargs):
        self.trial_updates.append((experiment_id, trial_id, kwargs))

    def shutdown(self):
        self.shutdown_calls += 1


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

    def close(self):
        pass


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
        self.finalizations = []
        self.claimed_attempts = set()
        self.url = "https://example.test/experiment/run-1"
        self.create_kwargs = {}

    def __enter__(self):
        return self

    def finalize(self, status, *, error):
        self.finalizations.append((status, error))

    def trial(self, _case, *, attempt, metadata):
        assert metadata["harbor_trial_id"] == "task-a__abc"
        if attempt in self.claimed_attempts:
            raise ValueError("trial already exists; increment attempt for a retry")
        self.claimed_attempts.add(attempt)
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


def test_live_publisher_batches_scores_and_preserves_evaluator_kind(monkeypatch, tmp_path):
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
        ("record", "check.queried-metrics", "deterministic"),
        ("record", "rubric.grounded-answer", "llm_judge"),
        ("flush", 3),
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

    assert experiment.finalizations == [("failed", "score export failed")]
    assert experiment.client.shutdown_calls == 1


def _transport_error(status=503):
    error = RuntimeError(f"transport failed: status {status}")
    error.__cause__ = HTTPError("https://example.test/ingest", status, "timeout", {}, None)
    return error


@pytest.fixture
def publishing(monkeypatch, tmp_path):
    job_dir, tasks_dir, result = _write_trial(tmp_path)
    experiment = FakeExperiment()
    monkeypatch.setattr(agento11y_publish, "_import_agento11y", lambda: _fake_modules(experiment))
    monkeypatch.setattr(agento11y_publish, "load_trials", lambda _path: [result])
    monkeypatch.setattr(agento11y_publish, "_final_output", lambda _path: "Final answer")
    monkeypatch.setattr(agento11y_publish, "_artifact_paths", lambda _path: [])
    monkeypatch.setattr(agento11y_publish.time, "sleep", lambda _delay: None)
    publisher = agento11y_publish.Agento11yLivePublisher(
        job_dir,
        tasks_dir,
        model="anthropic/claude-test",
        agent_name="o11y-bench",
        agent_version="test",
        options=agento11y_publish.Agento11yPublishOptions(
            experiment_id="run-1", suite_version="v1"
        ),
    )
    publisher._connect()
    return publisher, experiment


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504, 400, 401, 403, 404, 409, 422])
def test_retryable_http_errors_are_classified_by_cause(status):
    assert agento11y_publish._is_retryable_publish_error(_transport_error(status)) == (
        status == 429 or status >= 500
    )


@pytest.mark.parametrize("cause", [URLError("connection lost"), TimeoutError(), ConnectionError()])
def test_retryable_network_errors_are_classified_by_cause(cause):
    error = RuntimeError("SDK transport failed")
    error.__cause__ = cause
    assert agento11y_publish._is_retryable_publish_error(error)


@pytest.mark.parametrize(
    "error",
    [
        ValueError("invalid score"),
        FileNotFoundError("artifact"),
        RuntimeError("status 503 is only text"),
    ],
)
def test_local_and_validation_errors_are_not_retried(error):
    assert not agento11y_publish._is_retryable_publish_error(error)


def test_retry_delay_is_exponential_jittered_and_capped(monkeypatch):
    ranges = []

    def upper_bound(low, high):
        ranges.append((low, high))
        return high

    monkeypatch.setattr(agento11y_publish.random, "uniform", upper_bound)
    assert [agento11y_publish._retry_delay(n) for n in [0, 1, 2, 10000]] == [5, 10, 20, 60]
    assert ranges == [(2.5, 5), (5, 10), (10, 20), (30, 60)]


def test_poll_loop_recovers_and_resets_backoff(publishing, monkeypatch):
    publisher, _experiment = publishing
    outcomes = iter([_transport_error(), None, _transport_error()])
    waits = []
    attempts = []

    def scan():
        outcome = next(outcomes)
        attempts.append(outcome)
        if outcome:
            raise outcome
        assert publisher._error is not None

    def wait(delay):
        waits.append(delay)
        return len(attempts) == 3

    monkeypatch.setattr(publisher, "scan_once", scan)
    monkeypatch.setattr(publisher._stop, "wait", wait)
    monkeypatch.setattr(agento11y_publish, "_retry_delay", lambda attempt: 5 * 2**attempt)
    publisher._poll_loop()

    assert len(attempts) == 3
    assert waits == [1, 5, 1, 5]
    assert not publisher._stop.is_set()


def test_poll_loop_stops_on_permanent_failure(publishing, monkeypatch):
    publisher, _experiment = publishing
    error = _transport_error(403)

    def failed_scan():
        raise error

    monkeypatch.setattr(publisher._stop, "wait", lambda _delay: publisher._stop.is_set())
    monkeypatch.setattr(publisher, "scan_once", failed_scan)

    publisher._poll_loop()

    assert publisher._stop.is_set()
    assert publisher._error is error


def test_finish_recovers_stale_transport_error(publishing):
    publisher, experiment = publishing
    publisher._error = _transport_error()

    result = publisher.finish(succeeded=True)

    assert (result.trial_count, result.score_count) == (1, 3)
    assert publisher._error is None
    assert experiment.finalizations == [("completed", "")]
    assert experiment.client.shutdown_calls == 1


def test_finish_waits_for_worker_before_reading_its_error(publishing):
    publisher, experiment = publishing

    class Worker:
        def join(self):
            assert publisher._stop.is_set()
            publisher._error = _transport_error()

    publisher._thread = Worker()
    publisher.finish(succeeded=True)

    assert publisher._error is None
    assert experiment.finalizations == [("completed", "")]


def test_live_retry_wait_is_interruptible(publishing, monkeypatch):
    publisher, experiment = publishing
    retrying = threading.Event()

    def failed_scan():
        retrying.set()
        raise _transport_error()

    monkeypatch.setattr(publisher, "scan_once", failed_scan)
    publisher.poll_interval_seconds = 0.001
    publisher._thread = threading.Thread(target=publisher._poll_loop)
    publisher._thread.start()
    try:
        assert retrying.wait(timeout=2)
    finally:
        publisher._stop.set()
        publisher._thread.join(timeout=2)
    assert not publisher._thread.is_alive()
    assert experiment.finalizations == []


@pytest.mark.parametrize("succeeded", [True, False])
def test_unresolved_outage_leaves_run_open_and_persists_outcome(publishing, monkeypatch, succeeded):
    publisher, experiment = publishing
    calls = []

    def scan():
        calls.append(1)
        raise _transport_error()

    monkeypatch.setattr(publisher, "scan_once", scan)
    with pytest.raises(RuntimeError, match="status 503"):
        publisher.finish(succeeded=succeeded, error="" if succeeded else "Harbor failed")

    assert len(calls) == 3
    assert experiment.finalizations == []
    assert experiment.client.shutdown_calls == 1
    state = json.loads((publisher.job_dir / ".agento11y-publish.json").read_text())
    assert state["benchmark_succeeded"] is succeeded
    assert state["benchmark_error"] == ("" if succeeded else "Harbor failed")

    recovered_experiment = FakeExperiment()
    monkeypatch.setattr(
        agento11y_publish, "_import_agento11y", lambda: _fake_modules(recovered_experiment)
    )
    recovered = agento11y_publish.Agento11yLivePublisher(
        publisher.job_dir,
        publisher.tasks_dir,
        model=publisher.model,
        agent_name=publisher.agent_name,
        agent_version=publisher.agent_version,
        options=publisher.options,
    )
    result = recovered.publish_existing()
    assert (result.trial_count, result.score_count) == (1, 3)
    assert recovered_experiment.finalizations == [
        ("completed", "") if succeeded else ("failed", "Harbor failed")
    ]
    assert recovered._attempt_for("task-a", "task-a__abc") == 1


@pytest.mark.parametrize("stage", ["generation", "enter", "scores", "artifact", "close", "update"])
def test_partial_trial_reuses_sdk_trial_and_commits_counts_once(publishing, monkeypatch, stage):
    publisher, experiment = publishing
    calls = []
    targets = {
        "generation": (experiment.client, "record_generation"),
        "enter": (FakeTrial, "__enter__"),
        "scores": (FakeTrial, "flush"),
        "artifact": (FakeTrial, "artifact"),
        "close": (FakeTrial, "close"),
        "update": (experiment.client, "update_trial"),
    }
    owner, name = targets[stage]
    original = getattr(owner, name)

    def fail_once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise _transport_error()
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, fail_once)
    monkeypatch.setattr(
        agento11y_publish, "_artifact_paths", lambda _path: [("trajectory", publisher.job_dir)]
    )
    with pytest.raises(RuntimeError, match="status 503"):
        publisher.scan_once()
    assert publisher._score_count == 0
    assert publisher._trial_count == 0
    state = json.loads((publisher.job_dir / ".agento11y-publish.json").read_text())
    assert state["score_count"] == 0
    assert state["exported_trials"] == []

    publisher.scan_once()
    publisher.scan_once()

    assert publisher._score_count == 3
    assert publisher._trial_count == 1
    assert publisher._pending_trials == {}
    assert experiment.claimed_attempts == {1}
    assert len([event for event in experiment.events if event[0] == "record"]) == 3
    assert experiment.artifacts == [("trajectory", str(publisher.job_dir))]
    state = json.loads((publisher.job_dir / ".agento11y-publish.json").read_text())
    assert state["score_count"] == 3
    assert state["exported_trials"] == ["task-a__abc"]


def test_scores_accepted_before_later_failure_are_not_counted_twice_after_restart(
    publishing, monkeypatch
):
    publisher, experiment = publishing

    def failed_update(*_args, **_kwargs):
        raise _transport_error()

    monkeypatch.setattr(experiment.client, "update_trial", failed_update)
    with pytest.raises(RuntimeError, match="status 503"):
        publisher.scan_once()
    assert (publisher._trial_count, publisher._score_count) == (0, 0)

    replacement = FakeExperiment()
    monkeypatch.setattr(agento11y_publish, "_import_agento11y", lambda: _fake_modules(replacement))
    recovered = agento11y_publish.Agento11yLivePublisher(
        publisher.job_dir,
        publisher.tasks_dir,
        model=publisher.model,
        agent_name=publisher.agent_name,
        agent_version=publisher.agent_version,
        options=publisher.options,
    )
    result = recovered.publish_existing()
    assert (result.trial_count, result.score_count) == (1, 3)
    assert replacement.claimed_attempts == {1}


def test_finalization_transport_failure_is_retried(publishing, monkeypatch):
    publisher, experiment = publishing
    finalize = experiment.finalize
    calls = []

    def fail_once(status, *, error):
        calls.append(status)
        if len(calls) == 1:
            raise _transport_error()
        finalize(status, error=error)

    monkeypatch.setattr(experiment, "finalize", fail_once)
    publisher.finish(succeeded=True)
    assert calls == ["completed", "completed"]
    assert experiment.finalizations == [("completed", "")]
    assert experiment.client.shutdown_calls == 1


def test_finalization_outage_preserves_uploaded_checkpoint_for_recovery(publishing, monkeypatch):
    publisher, experiment = publishing
    finalize = experiment.finalize
    calls = []

    def unavailable(*_args, **_kwargs):
        calls.append(1)
        raise _transport_error()

    monkeypatch.setattr(experiment, "finalize", unavailable)
    with pytest.raises(RuntimeError, match="status 503"):
        publisher.finish(succeeded=True)
    assert len(calls) == 3
    assert experiment.finalizations == []
    assert experiment.client.shutdown_calls == 1

    monkeypatch.setattr(experiment, "finalize", finalize)
    recovered = agento11y_publish.Agento11yLivePublisher(
        publisher.job_dir,
        publisher.tasks_dir,
        model=publisher.model,
        agent_name=publisher.agent_name,
        agent_version=publisher.agent_version,
        options=publisher.options,
    )
    result = recovered.publish_existing()
    assert (result.trial_count, result.score_count) == (1, 3)
    assert experiment.finalizations == [("completed", "")]
    # The restored checkpoint skips the SDK's already-claimed attempt entirely.
    assert experiment.claimed_attempts == {1}
    assert len([event for event in experiment.events if event[0] == "record"]) == 3


def test_atomic_checkpoint_keeps_previous_state_when_replace_fails(publishing, monkeypatch):
    publisher, _experiment = publishing
    publisher._save_state()
    state_path = publisher.job_dir / ".agento11y-publish.json"
    original = state_path.read_text()
    publisher._score_count = 123

    def failed_replace(*_args):
        raise OSError("disk error")

    monkeypatch.setattr(type(state_path), "replace", failed_replace)
    with pytest.raises(OSError, match="disk error"):
        publisher._save_state()
    assert state_path.read_text() == original


def test_benchmark_failure_still_finalizes_as_failed(publishing):
    publisher, experiment = publishing
    publisher.finish(succeeded=False, error="Harbor exited 1")
    assert experiment.finalizations == [("failed", "Harbor exited 1")]


@pytest.mark.parametrize(
    "stage", ["trial-create", "scores", "artifact", "trial-close", "duration-update"]
)
def test_real_sdk_replays_partial_trial_without_duplicate_scores(publishing, monkeypatch, stage):
    sdk = pytest.importorskip("agento11y")
    experiments = pytest.importorskip("agento11y.experiments")
    publisher, _fake_experiment = publishing

    class Backend(FakeClient):
        """In-memory transport; the SDK Experiment, Trial, and score IDs are real."""

        use_experimental_otel = False

        def __init__(self):
            super().__init__()
            self.failed = False
            self.trials = {}
            self.scores = {}
            self.score_requests = []
            self.artifacts = []
            self.finalizations = []

        def fail_once(self, current):
            if current == stage and not self.failed:
                self.failed = True
                raise _transport_error()

        def upsert_trial(self, _experiment_id, **kwargs):
            self.trials.setdefault(kwargs["trial_id"], kwargs)
            self.fail_once("trial-create")

        def export_scores(self, scores):
            self.score_requests.append([score.score_id for score in scores])
            for score in scores:
                self.scores[score.score_id] = score
            # Simulate a lost response after the server accepted the batch.
            self.fail_once("scores")
            return len(scores)

        def upload_artifact(self, **kwargs):
            self.fail_once("artifact")
            self.artifacts.append(kwargs)
            return {"artifact_id": "artifact-1"}

        def update_trial(self, experiment_id, trial_id, **kwargs):
            self.fail_once(
                "duration-update" if kwargs.get("duration_ms") == 2000 else "trial-close"
            )
            self.trials[trial_id].update(kwargs)
            super().update_trial(experiment_id, trial_id, **kwargs)

        def finalize(self, _experiment_id, status, **_kwargs):
            assert all(trial["status"] == "completed" for trial in self.trials.values())
            self.finalizations.append(status)

        def experiment_url(self, _experiment_id):
            return "https://example.test/experiment/run-1"

    backend = Backend()
    suite = experiments.TestSuite(
        suite_id="suite-1",
        version="v1",
        test_cases=[
            experiments.TestCase(
                test_case_id="task-a", input="Investigate.", category="investigation"
            )
        ],
    )
    experiment = experiments.Experiment(backend, experiment_id="run-1", suite=suite)
    publisher._sdk = sdk
    publisher._experiments = experiments
    publisher._experiment = experiment
    publisher._suite_cases = {case.test_case_id: case for case in suite.cases}
    monkeypatch.setattr(agento11y_publish, "_import_agento11y", lambda: (sdk, experiments))
    artifact = publisher.job_dir / "trajectory.json"
    artifact.write_text("{}")
    monkeypatch.setattr(
        agento11y_publish, "_artifact_paths", lambda _path: [("trajectory", artifact)]
    )

    with pytest.raises(RuntimeError, match="status 503"):
        publisher.scan_once()
    published = publisher.finish(succeeded=True)

    assert backend.failed
    assert (published.trial_count, published.score_count) == (1, 3)
    assert len(backend.trials) == 1
    assert len(backend.scores) == 3
    assert all(score.trial_id in backend.trials for score in backend.scores.values())
    assert {score.value.number for score in backend.scores.values()} == {0.75, 1.0}
    assert backend.finalizations == ["completed"]
    assert len(backend.artifacts) == 1
    if stage == "scores":
        assert backend.score_requests[0] == backend.score_requests[1]


def test_long_score_slugs_are_bounded_and_collision_resistant():
    common = "A very long criterion " * 10
    first = agento11y_publish._slug(common + "first")
    second = agento11y_publish._slug(common + "second")

    assert len(first) == 80
    assert len(second) == 80
    assert first != second
