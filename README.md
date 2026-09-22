# o11y-bench

[![Leaderboard](https://img.shields.io/badge/Leaderboard-o11ybench.ai-blue)](https://o11ybench.ai/)
[![Tasks Explorer](https://img.shields.io/badge/Tasks_Explorer-Browse_Tasks-green)](https://o11ybench.ai/tasks/)
[![Harbor Docs](https://img.shields.io/badge/Harbor-Docs-orange)](https://www.harborframework.com/docs)
[![Submit Results](https://img.shields.io/badge/Submit_Results-Hugging_Face-yellow)](https://huggingface.co/datasets/grafanalabs/o11y-bench-leaderboard)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-red.svg)](./LICENSE)

---

- [Quick Start](#quick-start)
- [Running A Single Job](#running-a-single-job)
- [Running With Different Agents](#running-with-different-agents)
- [Running Your Own Models](#running-your-own-models)
- [Publishing Results To Agent Observability](#publishing-results-to-agent-observability)
- [Submitting Results To The Leaderboard](#submitting-results-to-the-leaderboard)

---

`o11y-bench` is an open benchmark for evaluating LLM agents on observability and SRE tasks.
It is built on top of [Harbor](https://harborframework.com) and runs agents against a real
Grafana stack with Prometheus, Loki, and Tempo.

The repo includes:

- benchmark task specs
- a default custom Harbor agent
- grading and reporting logic
- Docker images and config for the synthetic observability environment

Each run produces machine-readable artifacts plus HTML reports for either a single job or a full
comparison suite.

`tasks-spec/` is the source of truth for benchmark scenarios.
`tasks/` is generated output and should not be edited by hand.

## What This Repo Does

At a high level, a benchmark run does the following:

1. Materializes benchmark tasks from `tasks-spec/`
2. Starts the Harbor task container plus the observability sidecar stack
3. Runs an agent against one or more tasks
4. Grades the result against deterministic checks plus rubric criteria
5. Writes job artifacts and HTML reports under `jobs/`

The default agent lives in `agents/o11y_agent.py`, but you can also run Harbor built-in agents or
your own custom Harbor agent class by import path. See the [Harbor agent docs](https://www.harborframework.com/docs/agents) for more on agent types.

Each run also persists one scenario clock in `scenario_time.txt` under the job or suite directory.
That keeps reruns and regrades aligned to the same synthetic data window.

## Requirements

You need all of the following installed locally:

- [`mise`](https://mise.jdx.dev/)
- [`uv`](https://docs.astral.sh/uv/)
- [Docker](https://docs.docker.com/get-docker/)

You also need model-provider API keys in your environment.

Minimum environment variables:

- `ANTHROPIC_API_KEY`
  Used by the grading pipeline.
- Provider key(s) for the model you want to run:
  - `OPENAI_API_KEY`
  - `ANTHROPIC_API_KEY`
  - `GOOGLE_API_KEY` or `GEMINI_API_KEY`
  - `OPENROUTER_API_KEY` — run any model available on [OpenRouter](https://openrouter.ai/)

Optional environment variables:

- `OPENAI_API_BASE`
  Use this for OpenAI-compatible endpoints.
- `O11Y_SCENARIO_TIME_ISO`
  Override the scenario clock for debugging.

## Setup

Clone the repo and install the pinned toolchain and Python environment:

```bash
git clone <your-fork-or-repo-url>
cd o11y-bench
mise install
uv sync
```

That is the normal one-time setup.

If you want to confirm the local toolchain is working before running a benchmark:

```bash
mise run setup:sync
mise run lint
mise run test
```

## Quick Start

Run a single task with the default repo agent:

```bash
mise run bench:job -- --model openai/gpt-5.4-nano --task-name query-cpu-metrics --n-concurrent 1
```

This will:

- regenerate `tasks/` from `tasks-spec/`
- run the selected task with 3 attempts by default
- write artifacts under `jobs/<job-name>/`
- generate `jobs/<job-name>/run_report.html`

If you want a quiet version of the same command:

```bash
mise run bench:job:quiet -- --model openai/gpt-5.4-nano --task-name query-cpu-metrics --n-concurrent 1
```

Browse all available tasks on the [Tasks Explorer](https://o11ybench.ai/tasks/) or see how models compare on the [Leaderboard](https://o11ybench.ai/).

## Setup Details

To regenerate tasks explicitly:

```bash
mise run setup:sync
```

If you invoke raw Harbor commands yourself instead of `mise run bench:*` or
`uv run python -m o11y_bench ...`, run preflight first:

```bash
mise run setup:preflight
```

That pre-builds shared images and cleans stale Harbor Docker projects.

## Running A Single Job

Run one model across the benchmark:

```bash
mise run bench:job -- --model anthropic/claude-sonnet-4-6
```

Run a single task only:

```bash
mise run bench:job -- --model anthropic/claude-sonnet-4-6 --task-name query-cpu-metrics --n-concurrent 1
```

Run with a different reasoning level:

```bash
mise run bench:job -- --model openai/gpt-5.4-mini --reasoning-effort high
```

Run with a custom output location or name:

```bash
mise run bench:job -- --model openai/gpt-5.4-mini --jobs-dir /tmp/o11y-bench-jobs --job-name my-smoke-run
```

### Job Resume Behavior

`bench:job` resumes by job directory.
If the job already exists and the saved config is compatible, it reuses completed work and reruns
only missing or retryable trials.

This means:

- rerunning the same command usually resumes
- changing the model, reasoning effort, or agent configuration creates a distinct job variant
- you can always force separation with `--job-name`

Example default auto-generated job names:

- `openai-gpt-5-4-nano-off-k3`
- `openai-gpt-5-4-nano-high-k3`
- `openai-gpt-5-4-nano-off-opencode-k3`
- `openai-gpt-5-4-nano-off-agents-langchain-o11y-agent-langchaino11ybenchagent-k3`

If you want a fresh run instead of resuming, pass a fresh `--job-name`.

## Running With Different Agents

### Default Repo Agent

This is the normal path and uses the custom repo agent:

```bash
mise run bench:job -- --model openai/gpt-5.4-nano --task-name query-cpu-metrics
```

### Harbor Built-In Agent

You can switch to a [Harbor built-in agent](https://www.harborframework.com/docs/agents) with `--agent`:

```bash
mise run bench:job -- --model openai/gpt-5.4-nano --task-name query-cpu-metrics --agent opencode
```

### Custom Agent Import Path

You can run any importable Harbor agent class with `--agent-import-path`:

```bash
mise run bench:job -- --model openai/gpt-5.4-nano --task-name query-cpu-metrics --agent-import-path agents.langchain_o11y_agent:LangChainO11yBenchAgent
```

Use either `--agent` or `--agent-import-path`, not both.

The LangChain agent in this repo is intentionally a simple example of wiring a custom Harbor agent
entrypoint through the existing benchmark flow.

### gcx CLI Mode

By default, agents interact with Grafana through mcp-grafana (MCP tools). To benchmark
agents using the gcx CLI instead, use the `GcxOpenCodeAgent` agent class:

```bash
mise run bench:job -- --model anthropic/claude-sonnet-4-6 --task-name query-cpu-metrics --agent-import-path agents.gcx_opencode_agent:GcxOpenCodeAgent
```

This runs OpenCode with gcx and gcx skills pre-installed in the container, but
with MCP tools stripped so the agent can only reach Grafana through gcx.
The container image has `GRAFANA_SERVER` and `GRAFANA_ORG_ID` set so gcx
connects to the sidecar Grafana automatically.

This agent can only run models available in OpenCode.

### Using A Local gcx Build

To test gcx changes before they are on main, you can use a locally-built binary
instead of the published release.

Set `LOCAL_GCX` to the path of a gcx executable **for Linux** when running preflight or bench runs.

When set, the Docker image will use your local binary instead of downloading
from GitHub. The `environment/gcx` file is gitignored and cleaned up after build.

Use `GOARCH=amd64` if your Docker is running x86_64 images.

```bash
cd ~/workspace/gcx && GOOS=linux GOARCH=arm64 mise run build
LOCAL_GCX=/path/to/gcx/bin/gcx-linux mise run bench:job -- --model openai/gpt-5.4-nano
```

### Using A Local mcp-grafana Build

The sidecar normally installs the mcp-grafana release pinned in `docker/Dockerfile`.
To benchmark an unreleased build — comparing a branch against `main`, say — set
`LOCAL_MCP_GRAFANA` to the path of an mcp-grafana executable **for Linux**.

When set, the sidecar image uses your local binary instead of downloading from
GitHub. The `docker/mcp-grafana` file is gitignored and cleaned up after build.

Build statically, since the sidecar image is Alpine-based:

```bash
cd ~/workspace/mcp-grafana && CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o /tmp/mcp-grafana-branch ./cmd/mcp-grafana
LOCAL_MCP_GRAFANA=/tmp/mcp-grafana-branch mise run bench:job -- --model openai/gpt-5.4-nano
```

To A/B two builds, run them as separate jobs **sequentially** with distinct
`--job-name` values. The sidecar image tag is shared, so each run rebuilds it —
running two arms concurrently would swap one arm's binary out from under it.
Pin `O11Y_SCENARIO_TIME_ISO` across both so they see identical synthetic data.


## Running Your Own Models

If your model is reachable through Harbor and LiteLLM, pass it as `provider/model`.

Examples:

```bash
mise run bench:job -- --model openai/gpt-5.4-mini
mise run bench:job -- --model anthropic/claude-haiku-4-5-20251001
mise run bench:job -- --model google/gemini-3-flash-preview
```

### OpenRouter Models

You can run any model available on [OpenRouter](https://openrouter.ai/) by using the `openrouter/` prefix.
Set `OPENROUTER_API_KEY` in your environment, then:

```bash
mise run bench:job -- --model openrouter/deepseek/deepseek-v3.2 --job-name openrouter-deepseek-v3-2
```

### Dry Run

You can dry-run the job planner without executing Harbor:

```bash
uv run python -m o11y_bench job --model openai/gpt-5.4-nano --task-name query-cpu-metrics --dry-run
```

## Running The Standard Suite

Run the standard comparison suite:

```bash
mise run bench:suite
```

By default, `bench:suite` resumes the latest suite directory when possible.

To force a fresh suite directory:

```bash
mise run bench:suite -- --jobs-dir jobs/full-suite-$(date +%Y%m%d-%H%M%S)
```

To disable resume explicitly:

```bash
mise run bench:suite -- --no-resume --jobs-dir jobs/full-suite-$(date +%Y%m%d-%H%M%S)
```

To reduce local resource pressure:

```bash
mise run bench:suite -- --jobs-dir jobs/full-suite-$(date +%Y%m%d-%H%M%S) --n-concurrent 1
```

### Standard Suite Variants

The standard suite currently covers these provider/model/reasoning combinations:

- Anthropic
  - `claude-haiku-4-5-20251001`: `off`, `low`, `high`
  - `claude-opus-4-6`: `off`, `low`, `high`
  - `claude-sonnet-4-5`: `off`, `high`
  - `claude-sonnet-4-6`: `off`, `low`, `high`
- OpenAI
  - `gpt-5.1-codex-mini`: `off`, `high`
  - `gpt-5.2-codex`: `off`, `high`
  - `gpt-5.2-2025-12-11`: `off`, `high`
  - `gpt-5.4-2026-03-05`: `off`, `low`, `high`
  - `gpt-5.4-mini`: `off`, `low`, `high`
  - `gpt-5.4-nano`: `off`, `low`, `high`
- Google
  - `gemini-3-flash-preview`: `off`, `high`
  - `gemini-3.1-pro-preview`: `off`, `low`, `high`
  - `gemini-3.1-flash-lite-preview`: `off`, `low`, `high`

The suite uses the default repo agent.
If you want to benchmark a custom agent across the same matrix, run `bench:job` variants yourself
or extend suite orchestration in code.

## Publishing Results To Agent Observability

Publishing is optional. Harbor remains the experiment runner and ATIF remains the native trajectory
format. The integration sends experiment and trial lifecycle events, candidate generations,
token usage and cost, verifier scores, and links to Harbor trajectory and grading artifacts.
Each scored trial is sent as soon as Harbor finishes it rather than waiting for the full job.

Install the published `agento11y` package through the optional project extra:

```bash
mise run agento11y:setup
```

### Configure Agent Observability

Copy `.env.sample` to `.env` and fill in the `AGENTO11Y_*` values. Mise loads `.env` automatically.
Do not commit that file.

The integration uses two credential planes:

| Variable | Purpose | Where to find it |
|---|---|---|
| `AGENTO11Y_ENDPOINT` | Experiment, trial, generation, and score ingest URL | Grafana Cloud **Agent Observability > Configuration > Connection** |
| `AGENTO11Y_AUTH_TENANT_ID` | Grafana Cloud stack/tenant ID used for ingest basic auth | Agent Observability connection details |
| `AGENTO11Y_AUTH_TOKEN` | Access policy token with `sigil:write` for ingest | Agent Observability connection details |
| `AGENTO11Y_CONTROL_ENDPOINT` | Grafana app URL used to publish and retrieve test suites | `https://<stack>.grafana.net/a/grafana-agento11y-app` |
| `AGENTO11Y_SERVICE_ACCOUNT_TOKEN` | Grafana service-account token used by test-suite APIs | Grafana **Administration > Users and access > Service accounts** |
| `AGENTO11Y_INGEST_ACTOR` | Tenant or owner label attached to published experiments | Choose a stable identifier for the publishing team or user |

`AGENTO11Y_SUITE_ID` selects the remote test-suite name and defaults to `o11y-bench`.
`AGENTO11Y_AGENT_NAME`, `AGENTO11Y_AGENT_VERSION`, `AGENTO11Y_EXPERIMENT_NAME`,
`AGENTO11Y_EXPERIMENT_DESCRIPTION`, and `AGENTO11Y_EXPERIMENT_TAGS` customize run labels.
See [.env.sample](.env.sample) for the complete configuration.

### Publish The Dataset

The YAML files under `tasks-spec/` remain the only source of truth for the dataset. Publish them to
Agent Observability before sending benchmark results:

```bash
mise run agento11y:sync-suite
```

The task validates every YAML spec, converts it to an Agent Observability test case, synchronizes the
configured `AGENTO11Y_SUITE_ID`, removes remote cases no longer present in YAML, and publishes the
new suite version. It prints progress and a direct test-suite link. Running it again with unchanged
YAML is idempotent.

Choose a different remote suite name without changing the source dataset:

```bash
AGENTO11Y_SUITE_ID=o11y-bench-my-team mise run agento11y:sync-suite
```

### Publish Benchmark Results

Run a job through the Agent Observability mise task:

```bash
mise run agento11y:job -- \
  --model anthropic/claude-haiku-4-5-20251001 \
  --task-name promql-error-rate \
  --n-attempts 3
```

This task installs the published SDK extra for the command and supplies the explicit
`--agento11y-publish` opt-in flag. The equivalent direct command is:

```bash
uv run --extra agento11y python -m o11y_bench job \
  --model anthropic/claude-haiku-4-5-20251001 \
  --task-name promql-error-rate \
  --n-attempts 3 \
  --agento11y-publish
```

Use `mise run agento11y:suite -- <options>` to publish the standard multi-model suite. Publishing
state is retained in each job directory, so resume and retry operations preserve stable experiment,
trial, and attempt identities instead of duplicating completed results.

## Reports And Artifacts

Single-job report:

```text
jobs/<job-name>/run_report.html
```

Full-suite report:

```text
jobs/<suite-id>/comparison.html
```

Useful artifacts inside each trial directory:

- `agent/instruction.txt`
- `agent/trajectory.json`
- `agent/command-0/stdout.txt`
- `verifier/grading_details.json`
- `verifier/reward.txt`
- `result.json`

## Regrading Existing Runs

If you changed grading and want to reuse existing transcripts without rerunning agents:

```bash
uv run python -m o11y_bench regrade --jobs-dir jobs/<suite-id> --path tasks
```

To regrade a specific job:

```bash
uv run python -m o11y_bench regrade --jobs-dir jobs --job-name <job-name> --path tasks
```

`regrade` reruns the verifier against saved transcripts and rewrites verifier outputs in place.
For tasks whose checks need the live Grafana stack, it also starts a temporary local sidecar stack.

## Manual Reporting Commands

Rebuild a single run report:

```bash
uv run python -m reporting.run_report --job-dir jobs/<job-name>
```

Rebuild a suite report:

```bash
mise run report -- --jobs-dir jobs/<suite-id>
```

Compare two job directories directly:

```bash
uv run python -m reporting.compare_report --job-dir jobs/<suite-id>/<job-a> --job-dir jobs/<suite-id>/<job-b>
```

## Submitting Results To The Leaderboard

After completing a benchmark run, you can submit your results to the
[o11y-bench leaderboard](https://o11ybench.ai/) via the
[Hugging Face submission repo](https://huggingface.co/datasets/grafanalabs/o11y-bench-leaderboard).

To submit:

1. Fork the [submission repo](https://huggingface.co/datasets/grafanalabs/o11y-bench-leaderboard)
2. Create a branch and add your completed job directory under `submissions/o11y-bench/1.0/<agent>__<model>/`
3. Include a `metadata.yaml` with agent and model info
4. Open a Pull Request

See the [submission repo README](https://huggingface.co/datasets/grafanalabs/o11y-bench-leaderboard) for
the full submission structure, validation rules, and example layout.

## Common Local Commands

```bash
mise run setup:sync
mise run setup:preflight
mise run setup:smoke
mise run agento11y:setup
mise run agento11y:sync-suite
mise run lint
mise run format
mise run typecheck
mise run test
mise run bench:job -- --model openai/gpt-5.4-nano --task-name query-cpu-metrics --n-concurrent 1
mise run bench:suite
```

## Notes For Contributors

- Edit `tasks-spec/`, not `tasks/`
- Regenerate tasks after task-spec changes with `mise run setup:sync`
- Keep tests small and behavior-focused
- Be conservative with local concurrency if Docker resources are limited

## License

This project is licensed under the [GNU Affero General Public License v3.0](./LICENSE).
