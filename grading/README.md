# Grading

The grading system is intentionally small:

- `checks`: deterministic pass/fail checks for exact state and grounding
- `rubric`: one LLM-as-a-judge pass over short YES/NO criteria
- `fact`: optional canonical evidence attached to a rubric item

The design goal is to check correctness against controlled data without forcing one exact wording or one exact solution path.

## Main entrypoints

- [verifier.py](verifier.py): standalone Harbor verifier entrypoint
- [checks.py](checks.py): deterministic check execution
- [judge.py](judge.py): prompt building, judge call, and YES/NO parsing
- [facts.py](facts.py): canonical fact fetching, rendering, and per-run caching
- [models.py](models.py): typed grading/task models
- [env_context.py](env_context.py): HTTP fetch helpers for Grafana, Prometheus, Loki, and Tempo

## Flow

1. Parse `problem.yaml` into typed models from [models.py](models.py).
2. Parse the agent transcript with [transcript_parser.py](transcript_parser.py).
3. Run deterministic `checks`.
4. Build rubric criteria for the judge.
5. If a rubric item has `fact`, resolve it once and inline a short `Source of truth:` line into that criterion.
6. Run one judge call for the full rubric.
7. Combine check and rubric subscores with [scoring.py](scoring.py).

The verifier writes:

- `score`
- `checks_passed`
- `rubric_passed`
- one flat score entry per check or rubric criterion
- `explanation:<name>` entries for debugging

## YAML shape

Tasks should use:

```yaml
rubric:
- criterion: The final response identifies the backend with the highest 5xx share.
  weight: 0.4
  fact:
    kind: query
    backend: prometheus
    query: topk(1, ...)

- criterion: The final response states that backend's 5xx share accurately.
  weight: 0.4
  fact:
    kind: query
    backend: prometheus
    query: topk(1, ...)
```

Keep `statement` user-voiced. Put grading detail in `checks`, `rubric`, and `fact`, not in the prompt.

## Deterministic checks

Only three families are supported:

- `grounding`
- `state`

Use checks for things that are exact and wording-independent:

- cited trace IDs or other concrete entities
- saved Grafana state
- datasource inventory or detail

Do not use deterministic response parsing as the main fact checker for semantic answers.

## Facts

`fact` is for canonical truth that can be computed but may be expressed many valid ways in the response.

Supported kinds:

- `query`
  - backends: `prometheus`, `loki`, `tempo`
- `resource`
  - resources: `dashboard`, `datasource_inventory`, `datasource_detail`

The grader auto-renders concise evidence from the fetched result. Task authors should shape the canonical query or resource selection so the result already reflects the truth they want checked. Avoid extra transforms or response-side parsing rules.

Prometheus facts are parsed and validated at spec load time, so invalid canonical PromQL fails before the benchmark runs.

If the truth is intentionally fixed by scenario design and not worth fetching, write it directly into the rubric criterion instead of adding a fake fact type.

## Design rules

- Prefer one canonical fact plus one short rubric criterion over brittle regex extraction.
- Prefer exact checks for artifact state and grounding.
- Prefer a single judge call over multiple rubric passes.
- Prefer native backend APIs under the hood:
  - Prometheus `/api/v1/query`
  - Loki `/loki/api/v1/query`
  - Tempo `/api/search` and `/api/v2/search/tag/.../values`
  - Grafana REST APIs for saved resources

## Regrading existing jobs

If only grading changed and transcripts are still valid, rerun the verifier without rerunning agents:

```bash
uv run python -m o11y_bench.cli regrade --jobs-dir jobs --job-name <job-name> --path tasks
```

That reuses existing transcripts and updates verifier outputs and reports only.

## Files by responsibility

- [models.py](models.py): typed spec and transcript models
- [checks.py](checks.py): deterministic grading logic
- [facts.py](facts.py): canonical evidence resolution
- [judge.py](judge.py): judge prompt and parsing
- [dashboard_snapshot.py](dashboard_snapshot.py): dashboard loading and normalization
- [dashboard_state.py](dashboard_state.py): dashboard state checks
- [dashboard_queries.py](dashboard_queries.py): saved-query semantics and execute-case checks
- [helpers.py](helpers.py): shared transcript and utility helpers

## Quick guidance for maintainers

- Start in `tasks-spec/`, not `tasks/`.
- If a task is semantically hard to parse but easy to compute, move truth into `fact`.
- If a task requires exact saved state or exact citation, keep that in `checks`.
- Keep tests behavior-focused:
  - spec parsing
  - fact rendering
  - representative check behavior
  - one judge-call integration
