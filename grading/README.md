# Grading

A trial is scored from deterministic `checks` (state read back from the stack) and a `rubric`
judged by one LLM call. A rubric item may carry a `fact`: evidence read from the stack, shown to the
judge as the criterion's source of truth. A criterion passes on the outcome the task asks for,
never on the method or the wording.

## Main entrypoints

- [verifier.py](verifier.py): standalone Harbor verifier entrypoint, outcomes and status
- [checks.py](checks.py): deterministic check execution
- [judge.py](judge.py): criteria with their source of truth, the judge call, and parsing
- [judge_system.txt](judge_system.txt) / [judge_request.txt](judge_request.txt): the judge prompt
- [facts.py](facts.py): fact fetching, rendering, and the saved `truth.json`
- [models.py](models.py): typed grading/task models and the transcript the judge reads
- [env_context.py](env_context.py): HTTP fetch helpers for Grafana, Prometheus, Loki, and Tempo

## Flow

1. Parse `problem.yaml` into typed models from [models.py](models.py).
2. Parse the agent transcript with [transcript_parser.py](transcript_parser.py).
3. Run deterministic `checks`.
4. Resolve each rubric `fact` once and add it to its criterion as a `Source of truth`.
5. Run one judge call for every criterion whose truth could be read.
6. Give every check and criterion an outcome, then the trial a status.
7. Combine scores with [scoring.py](scoring.py).

## What the judge reads

The transcript is one line per step: `User:`, `Agent:` and `Agent (final response):`, with tool
calls inline (`name(k=v) ✓` or `✗` for an error) and a short excerpt of each tool result
(`↳ name result: ...`, newest first). The first, last and final-response steps are shown whole;
other agent text and tool results share fixed budgets (constants in `models.py`).

The final response is the last agent message with text after the last user message. If that
message also calls tools it is a planning turn and there is no answer. A closing note under
200 characters gets the nearest earlier long message prepended.

The judge scores each criterion 0, 0.5 or 1 and quotes the words of the final response the
score rests on. A criterion passes at 1. Numbers follow one rule in the prompt: a continuous
value within 1%, counts and identifiers exactly, no 10x slips, and a bound or range is not the
value. Don't write rounding or tolerance rules into criteria.

The judge model is `DEFAULT_GRADING_MODEL` in `judge.py`; `GRADING_MODEL=model[@effort]` overrides it for regrades.

## Outcomes and status

Every check and criterion gets an `outcome:<name>` in `grading_details.json`:

| Outcome | Meaning |
| --- | --- |
| `pass` | Met. |
| `wrong` | The answer contradicts the truth, or the check failed. |
| `not_stated` | A fact-backed criterion the answer never addresses (empty quote). |
| `no_answer` | The agent gave no final response. |
| `ungrounded` | The judge's quote is not in the answer. The score stands; review it. |
| `error/truth` | The fact could not be read (transport error, or empty without `allow_empty`). |
| `error/judge` | The judge call failed or returned no block for the criterion. |
| `error/check` | An offline regrade had no saved result for a state check. |

The trial `status` is `passed`, `failed`, or `error` when any item has an `error/*` outcome.
`reward.txt` stays a float. Error trials are left out of pass rates and are not rerun on
resume; regrade them instead.

Facts are fail-loud: a query that returns nothing is a grading error, not a zero. When "nothing
matches" is the truth, set `allow_empty: true` on the fact.

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

Every check reads state back from the stack after the agent finished (`type: state`):

- saved Grafana state: dashboards, datasource inventory or detail, Tempo service coverage
- `cited_trace`: a trace the final response cites by its full ID (32 hex, or as Tempo prints it with leading zeros dropped)

```yaml
checks:
- name: The final response cites a failed POST /api/orders trace by its full ID
  weight: 10
  type: state
  params:
    mode: cited_trace
    match: '{ span.http.route = "/api/orders" && status = error }'
    lookback_hours: 24   # default
```

`cited_trace` passes when one cited trace is in Tempo over the window and matches `match`, a
TraceQL spanset filter for what the prompt asked the example to be (failing, slow, on a route). That
trace is the task's example: the first cited ID that matches. The check and every `cited_trace`
fact read the same example, so a task has at most one `cited_trace` check, and a `cited_trace` fact
needs one. Both read only the final response and Tempo, never the agent's tool output, so they hold
for every harness, and an invented ID fails. A prefix such as `0595f4e0` can't be read back, so it
doesn't count.

Do not use deterministic response parsing as the main fact checker for semantic answers.

## Facts

`fact` is for canonical truth that can be computed but may be expressed many valid ways in the response.

Supported kinds:

- `query`
  - backends: `prometheus`, `loki`, `tempo`
  - A Tempo query with a metrics stage (`| rate()`, `| count_over_time()`,
    `| quantile_over_time(...)` and the other `_over_time` functions) is evaluated once over the
    window, giving one value per series. Any other Tempo query is a trace search, and its truth
    lists the matching traces and spans.
- `resource`
  - resources: `dashboard`, `dashboard_diff`, `datasource_inventory`, `datasource_detail`
- `cited_trace`
  - The task's example trace (see `cited_trace` under checks), read back from Tempo. The truth
    states what the trace shows, computed rather than left to the judge: its root span and
    duration, its services outermost first, its spans with error status, its error origin (the
    deepest of them) and the span with the most own time (its duration minus its child spans). Then the span tree:
    service, operation, kind, duration, start offset, HTTP status and error status per span, capped
    at 40 spans. Use it to grade claims about the agent's own example (what failed first, where the
    time goes, the call chain). When the response cites no trace in Tempo that matches, the
    criterion fails without a judge call: claims about an example that doesn't exist earn nothing.
    Pair it with window-wide `query` facts for claims about the population ("which service",
    "most or only some").
- `grafana`
  - `path`: any Grafana API read under `/api/` or `/apis/`, such as `/api/folders` or
    `/api/v1/provisioning/alert-rules`. The truth is the JSON response. One over 8000 characters is an
    `error/truth`, not a cut sample: point the fact at a narrower path.
    A 404 or empty response is the truth only with `allow_empty: true`. A dashboard path is graded
    as saved state, like the `dashboard` resource.

The grader auto-renders concise evidence from the fetched result. Task authors should shape the canonical query or resource selection so the result already reflects the truth they want checked. Avoid extra transforms or response-side parsing rules.

Prometheus facts are parsed and validated at spec load time, so invalid canonical PromQL fails before the benchmark runs.

### Dashboard diff

`dashboard_diff` answers "did only the target change?" for tasks that edit a seeded dashboard
(`setup_dashboards`). Handed the whole dashboard, a judge cannot reliably tell which panel moved,
so [dashboard_diff.py](dashboard_diff.py) compares the seed (the before state, with the defaults
the provisioner fills in) with the dashboard read from Grafana after the agent finished (the
after state), and the judge reads the compact diff:

```yaml
- criterion: The existing Error Rate panel was not modified or removed.
  weight: 15
  fact:
    kind: resource
    resource: dashboard_diff
    uid: service-overview   # or title; must be a dashboard the task seeds
```

Elements are keyed `panel-<id>` and rendered with their titles (`panel-2 "Error Rate"`; a rename
shows both). The diff covers elements, `gridPos` moves, variables by name, and the dashboard's
title, description, tags, time settings, annotations and links. Array settings are itemised per
entry, matched by name. Layout is compared apart from the element, so Grafana pushing panels down
to make room is a move and not an edit. Empty values, `pluginVersion`, the built-in annotation and
a variable's resolved `current`/`options` are ignored, since no agent authors them.

The render is size-capped (limits in `dashboard_diff.py`, stated in the diff when they bite).
Only classic dashboard JSON is supported; the judge, not an assertion grammar, decides.

The diff is saved in `truth.json`, so `regrade --offline` needs no stack. `mise run lint:tasks`
flags a diff fact whose dashboard the task does not seed.

If the truth is fixed by scenario design and not worth fetching, write it into the criterion.

## Design rules

- One canonical fact plus one short criterion beats response-side parsing.
- Exact checks for artifact state and for the trace an answer cites; one judge call for the rest.
- Facts read the native APIs: Prometheus `/api/v1/query`, Loki `/loki/api/v1/query`, Tempo
  `/api/search`, `/api/v2/traces/<id>` and tag values, Grafana REST for saved resources.

## Regrading existing jobs

If only grading changed and transcripts are still valid, rerun the verifier without rerunning agents:

```bash
uv run python -m o11y_bench.cli regrade --jobs-dir jobs --job-name <job-name> --path tasks
```

That starts a stack per task, reads the truth again, and updates verifier outputs and reports.
Tasks that grade what the agent left in Grafana (dashboards, API reads) cannot be read from a
fresh stack, so they regrade from their saved truths even without `--offline`.

Every grade saves the truth it read to `verifier/truth.json`. To grade again from that saved
truth, with no stack and no live reads, add `--offline`. State checks keep their saved result;
trials without a `truth.json` are skipped. Both modes write
`<job>/regrade.json` with each item's score before and after.

## Quick guidance for maintainers

- Start in `tasks-spec/`, not `tasks/`.
- If a task is semantically hard to parse but easy to compute, move truth into `fact`.
- If a task requires exact saved state or exact citation, keep that in `checks`.
- Keep tests behavior-focused:
  - spec parsing
  - fact rendering
  - representative check behavior
  - one judge-call integration
