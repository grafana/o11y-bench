from __future__ import annotations

import json
from types import SimpleNamespace
from typing import ClassVar

import sigil_sdk

from o11y_bench import sigil_export


class FakeRecorder:
    """Stands in for the SDK GenerationRecorder.

    ExperimentRun.produced_generation_ids reads ``last_generation.id`` after the
    recorder context exits, so we surface the id the run assigned to the start.
    """

    def __init__(self, client, start):
        self.client = client
        self.start = start
        self._result = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.client.generations.append((self.start, self._result))
        return False

    def set_result(self, generation):
        self._result = generation

    @property
    def last_generation(self):
        return SimpleNamespace(id=self.start.id)


class FakeClient:
    """Records the control-plane + score-export calls the exporter makes.

    Implements the surface ExperimentRun and the exporter use, so the real SDK
    ExperimentRun/ScoreOutput/GenerationStart types drive the mapping without
    touching the network.
    """

    instances: ClassVar[list[FakeClient]] = []

    def __init__(self):
        self.experiments = []
        self.generations = []
        self.scores = []
        self.completed = []
        self.flushed = False
        self.closed = False
        FakeClient.instances.append(self)

    def create_experiment(self, request):
        self.experiments.append(request)

    def update_experiment(self, run_id, request):
        self.experiments.append((run_id, request))

    def start_generation(self, start):
        return FakeRecorder(self, start)

    def flush(self):
        self.flushed = True

    def export_scores(self, scores):
        self.scores.extend(scores)
        return SimpleNamespace(accepted_count=len(scores))

    def complete_experiment(self, run_id, **kwargs):
        self.completed.append((run_id, kwargs))

    def experiment_url(self, run_id):
        return f"https://sigil.example/a/grafana-sigil-app/evaluation/experiments/{run_id}"

    def shutdown(self):
        self.closed = True


def _write_trial(tmp_path):
    tasks_dir = tmp_path / "tasks"
    task_dir = tasks_dir / "query-cpu-metrics"
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text('[metadata]\ncategory = "prometheus_query"\n')

    job_dir = tmp_path / "jobs" / "openai-gpt-5-4-nano-off-k1"
    trial_dir = job_dir / "query-cpu-metrics__abc123"
    agent_dir = trial_dir / "agent"
    verifier_dir = trial_dir / "verifier"
    agent_dir.mkdir(parents=True)
    verifier_dir.mkdir()
    (agent_dir / "transcript.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"type": "user", "message": "Find CPU usage"}),
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {"content": [{"type": "text", "text": "CPU is high on api-1"}]},
                    }
                ),
            ]
        )
        + "\n"
    )
    (verifier_dir / "grading_details.json").write_text(
        json.dumps({"score": 1.0, "explanation:rubric": "accurate"})
    )
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "agent_info": {"model_info": {"provider": "openai", "name": "openai/gpt-5.4-nano"}},
                "agent_result": {
                    "n_input_tokens": 10,
                    "n_cache_tokens": 2,
                    "n_output_tokens": 5,
                    "cost_usd": 0.001,
                    "metadata": {"reasoning_effort": "off"},
                },
                "agent_execution": {
                    "started_at": "2026-05-20T00:00:00Z",
                    "finished_at": "2026-05-20T00:00:12Z",
                },
                "task_name": "query-cpu-metrics",
                "task_checksum": "checksum-1",
                "verifier_result": {"rewards": {"reward": 1.0}},
            }
        )
    )
    return job_dir, tasks_dir


def test_export_job_to_sigil_maps_trials_to_generations_and_scores(monkeypatch, tmp_path):
    FakeClient.instances = []
    # Use the real SDK types (ExperimentRun/ScoreOutput/GenerationStart/...) but
    # swap the network-backed Client for a recorder fake.
    monkeypatch.setattr(sigil_export, "_import_sdk", lambda sdk_path=None: sigil_sdk)
    monkeypatch.setattr(sigil_export, "_make_client", lambda sdk, options: FakeClient())

    job_dir, tasks_dir = _write_trial(tmp_path)

    result = sigil_export.export_job_to_sigil(
        job_dir,
        tasks_dir,
        sigil_export.SigilExportOptions(run_id="o11y-run-1", tags=["candidate"]),
    )

    assert result.run_id == "o11y-run-1"
    assert result.generation_count == 1
    assert result.score_count == 1
    assert "o11y-run-1" in result.url

    client = FakeClient.instances[0]
    assert client.flushed is True
    assert client.closed is True
    assert client.experiments[0].tags == ["o11y-bench", "candidate"]

    # one generation, tagged with the experiment run_id + o11y-bench labels
    assert len(client.generations) == 1
    start, _generation = client.generations[0]
    assert start.conversation_title == "query-cpu-metrics"
    assert start.tags["experiment.run_id"] == "o11y-run-1"
    assert start.tags["task_id"] == "query-cpu-metrics"
    assert start.metadata["experiment_run_id"] == "o11y-run-1"

    # one score, attributed to the run + generation, with grouping metadata
    assert len(client.scores) == 1
    score = client.scores[0]
    assert score.run_id == "o11y-run-1"
    assert score.generation_id == _generation.id
    assert score.value.number == 1.0
    assert score.passed is True
    assert score.source.kind == "experiment"
    assert score.metadata["dataset_id"] == "o11y-bench"
    assert score.metadata["item_id"] == "query-cpu-metrics"
    assert score.metadata["trial_id"] == "query-cpu-metrics__abc123"
    assert score.metadata["task_id"] == "query-cpu-metrics"
    assert score.metadata["task_category"] == "prometheus_query"
    assert score.metadata["total_tokens"] == 17
    assert score.metadata["candidate"] == {"job_name": "openai-gpt-5-4-nano-off-k1"}
    assert score.explanation == "accurate"

    assert client.completed == [
        (
            "o11y-run-1",
            {
                "status": "succeeded",
                "score_count": 1,
                "error": "",
                "metadata": {
                    "source": "o11y-bench",
                    "job_name": "openai-gpt-5-4-nano-off-k1",
                    "exported_generations": 1,
                    "exported_scores": 1,
                },
            },
        )
    ]


def test_default_run_id_is_stable(tmp_path):
    job_dir = tmp_path / "jobs" / "demo"
    job_dir.mkdir(parents=True)

    first = sigil_export._default_run_id(job_dir)
    second = sigil_export._default_run_id(job_dir)

    assert first == second
    assert first.startswith("o11y-bench-demo-")
