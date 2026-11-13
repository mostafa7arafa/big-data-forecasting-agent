# Plan B: Agent Harness: Design

_Status: draft, 2026-11-12. Owner: Mostafa Arafa. Extends the binding project spec `2026-11-07-taxi-analytics-agent-design.md`. Where this doc is silent, that spec applies._

Marked **[owner]**: decided by the owner in the 2026-11-12 brainstorm. Marked **[call]**: a design call made while writing this doc, which the owner can overturn.

## 1. Intent

Plan A built a proven data and forecasting core: governed metrics, guarded SQL, result handles, backtested forecasts and cutoff replay. Plan B puts an agent on top of it.

The first time the owner uses the agent will be in **LangGraph Studio via `langgraph dev`**, with every run traced to **LangSmith** [owner]. There is no dashboard yet. Charts come out as JSON specs only and are checked visually during the dashboard work [owner].

Two things make it more than "chat with a database":

1. **It never goes quiet on slow work** [owner]. Forecasts and multi-step investigations take time (a daily by-borough backtest takes about 90 s). The agent hands that work to a background agent and tells the user it has done so. The conversation continues while the work runs, and the result arrives in the same thread when it is ready.
2. **It is measured from day one** [owner]. A single-turn eval set of about 50 questions has known answers and expected tools. It lives in the repo, is uploaded as a LangSmith dataset, and is run as experiments that can be compared and extended later.

## 2. Split into two plans [owner]

| Plan | Ends with | Contains |
|---|---|---|
| **B1: the agent in Studio** (this doc's focus) | The owner chats with the agent in `langgraph dev`. A slow forecast runs in the background while the chat continues. A first LangSmith experiment over the 50-question set exists. | Task-zero fixes, the graph, typed tools, the verify node and labels, the answer schema, providers, background agents, middleware (clearing, metering, cost, limits), LangSmith wiring, the eval set, graders and the experiment runner |
| **B2: CodeAct and long sessions** | `run_python` works in a Docker sandbox, passes an escape-attempt suite, and long chats get compacted | Docker sandbox, AST guard, `run_python`, escape suite, compaction middleware, and eval rows that need Python added to the set |

**Not in Plan B:**
- Multi-turn evals with simulated personas (openevals' multiturn simulator + LLM judges). That is a separate later piece [owner].
- The chart-data downsampling service (M4/LTTB) [call: moved to the dashboard plan, where there is something to render].
- Calibration, feedback and user memory (Plan C).
- The full ~70 set, κ labels and README headline numbers (Plan D).

## 3. Goals, non-goals, anti-goals (B1)

**Goals**
- `langgraph dev` serves the agent. Studio shows the graph topology and each node as it runs.
- The hero question ("forecast next month's trips by borough, compare with last year, and explain the main trends") is answered end to end:
  - the forecast runs as a background task;
  - the agent stays available while it runs;
  - the final answer is labeled Verified.
- Every answer is a structured object: narrative, claims with evidence handles, chart specs, SQL used, and a verification label and reason.
- Every run is traced in LangSmith with token use, cost, latency and **bytes of data in context per model call**.
- A 50-question eval set lives in `evals/` and is uploaded to LangSmith. `make eval-agent` runs an experiment, and a second model or prompt can be run and compared.

**Non-goals (B1)**
- Python execution, Docker and compaction (all B2).
- Web search, any UI beyond Studio, and multi-user support.

**Anti-goals**
- A background agent whose numbers can't be traced. Its output must come back as handles like any tool output.
- An eval that passes because the grader is loose. Numbers are checked against frozen gold within the stated tolerances.
- An eval that fails a right answer because the route was unexpected. **The tool trajectory is reported as a score, not used as a pass/fail gate** [owner].
- Dataset specifics in `src/analyst/`. Prompts take dataset facts from dataset memory, and the dataset-agnostic test still guards this.

## 4. Part 0: task-zero fixes

These items are deferred from Plan A (`docs/superpowers/plans/2026-11-08-plan-a-followups.md`). The agent depends on them, so they come first.

- **Concurrency is mandatory**, because the main agent and a background agent use the store and engine at the same time:
  - atomic `ResultStore.save`;
  - engine locking or a cursor per call;
  - one shared `DataService` per (dataset, cutoff) per process.
- **One `AnalystError`** type with an LLM-actionable `hint`. The tool layer turns it into a tool message the model can act on.
- **Forecast caching** by input handle + params, plus a time budget, so a repeated forecast returns instantly.
- **Card meta and inputs:**
  - `cutoff` and `data_end` in forecast card meta;
  - `levels` other than 80/95 are honoured or rejected;
  - string values for integer dimensions are coerced;
  - `describe` says "showing 50 of N";
  - partial periods caused by the requested `time_range` are noted.
- **Data treatments:**
  - additive metrics are zero-filled on days with no events;
  - an id-share policy is applied to headline numbers;
  - `recommended_models` is wired into candidate selection;
  - ARIMA warnings are captured during `make replay`.
- **`Memory.as_of(cutoff)`** drops quirks, facts and recap content after the cutoff. It is required before the eval set's cutoff questions.
- **Two Task-6 minors:**
  - `resolve.py` stops importing private compiler helpers;
  - YoY without a grain is documented.

## 5. Agent graph

These parts are unchanged from the project spec §5.2:
- a small custom `StateGraph`: **plan → analyst → verify → answer**;
- the plan node runs only for multi-step questions;
- verify loops back to analyst once on failure;
- `analyst` is a LangChain v1 `create_agent` loop over typed tools and middleware.

### 5.1 Background agents [owner: "full duplex"]

The analyst gets **deepagents async subagents** (`AsyncSubAgentMiddleware`). Researched 2026-11-12 ([docs](https://docs.langchain.com/oss/python/deepagents/async-subagents)).

**How it works**
- It adds tools to start, check, update, cancel and list background tasks.
- Starting a task returns a task id immediately. The analyst tells the user what it started and keeps answering other messages on the same thread.
- The background agent is its own graph, registered in the same `langgraph.json` and called with no `url`, so it runs in-process under `langgraph dev` (ASGI transport).
- When it finishes or fails, the Agent Server sends the supervisor a `HumanMessage` tagged `source: "async_subagent"`. That message starts a new run on the thread, which goes through plan → analyst → verify → answer like any user turn. There is no polling.
- **Constraint:** auto-wake requires the background agent to be a deep agent (`create_deep_agent`).

**The background graph `worker`**
- It is a deep agent with the same data tools (`query_metrics`, `run_sql`, `describe`, `forecast`, `read_knowledge`) over the same shared `DataService`.
- It does not carry the async-task tools, so it cannot spawn further agents [call].
- It carries the same metering, cost and call-limit middleware as the analyst, with its own per-run limits. A spawned task cannot escape the limits [call].

**The worker inherits the supervisor's context** [call]: `dataset`, `cutoff` and `model` go to the worker with the task, through the run context if the pinned `deepagents` supports it, otherwise in the task payload.
- **A cutoff leak is a security bug.** A test proves that a worker started from a cutoff run cannot see data past the cutoff.
- If propagation can't be made reliable on the pinned version, background work is **disabled whenever `cutoff` is set** (the work runs inline), and the owner is told.

**The worker's results come back as data, not prose** [call]:
- The worker's final message ends with a fixed machine-readable line, `handles: r_…, r_…`.
- The analyst (on wake) cites those handle ids in its claims. It can call `describe(handle)` to read their cards.
- Verify resolves any cited handle id against the shared result store for that (dataset, cutoff), whichever run produced it. Tracing doesn't depend on parsing the worker's prose.
- `background_tasks[]` in state records task id → handle ids.

**When the analyst spawns a background task**
- Spawn for:
  - any `forecast` that is not cached and is daily or multi-id;
  - any investigation the plan node splits into three or more data steps.
- Monthly single-series forecasts and cached forecasts run inline.
- Exact thresholds are left to the builder [call: rule of thumb is "expected > ~15 s → background"].

**Delivery and limits**
- **Delivery timing** [call]: a result never interrupts an answer that is being written. It arrives when the analyst is idle, or right after the current answer.
- **Worker slots:** `make dev` runs `langgraph dev --n-jobs-per-worker 10`, so background runs don't queue up.

**Feature status**
- Async subagents are **beta**. Pin the exact `deepagents` version in `uv.lock`, and check tool names against the installed version.
- If auto-wake does not work under `langgraph dev` with our custom `StateGraph` supervisor, the fallback is: the analyst checks task status on each user turn, and the owner is told [call].

**In eval runs** [call]
- The graph is invoked with runtime context `background=False` (see §5.3), so would-be background work runs inline and each eval case is single-turn and deterministic.
- One small integration test does exercise the real background path: start a task, send another message, then receive the notification.

### 5.1a Parallel work, capped [owner]

The agent can work on several things at once. For example, "give me 3 charts for the most important insights" runs three investigations in parallel, each returning a chart spec plus handles. There are two layers:

- **Inline:** the model may issue several tool calls in one turn, and they run concurrently.
- **Background:** up to `ANALYST_MAX_BACKGROUND_TASKS` workers run at once (default 3). Extra starts are refused with a hint to wait, so a task is never silently dropped.

**Engine gate** [call]: DuckDB is in-process, so the limit is CPU and memory, not connections. One process-wide semaphore caps concurrent engine queries (`ANALYST_MAX_CONCURRENT_QUERIES`, default 3). Each query's DuckDB `threads` setting is the core count divided by that cap. Queries beyond the cap wait in a queue. Forecast fitting also counts against a slot.

**Testing:**
- a test proves the cap holds under N > cap concurrent calls;
- a test proves results stay correct under concurrency (this builds on the Part 0 locking work).

### 5.1b Progress messages [owner]

The agent shows it is alive and working, which also lowers felt latency. There are two sources, and neither adds an extra model call:

- **Model-written status lines:** with each batch of tool calls, the analyst writes one short line saying what it is doing ("Pulling last October for the year-over-year comparison…", "Started a background forecast for the boroughs; this takes about a minute"). It is part of the model call already happening.
- **Tool progress events:** long tools emit real steps through LangGraph's custom stream (`get_stream_writer`), for example "backtest window 4/12" or "fitting AutoETS for Queens". Studio shows them now. The dashboard renders them later.

Status lines are **not answer content**. Verify ignores them, and their numbers are never claims.

**Voice:** neutral and plain for now [owner]. When the brand character is designed (dashboard and brand phase), these lines get rewritten in its voice. That's a prompt change, plus the wording of the tool-progress templates.

### 5.2 State

Fields, as in the project spec:
- messages;
- plan;
- live handles;
- chart specs;
- forecast runs;
- verification report;
- token usage.

New in Plan B: **background tasks** (task id, purpose, status, result handles).

### 5.3 Runtime config [call]

The graph `context_schema` carries:
- `dataset` (default `nyc_taxi`);
- `cutoff` (optional; this is how eval replay questions run time-travelled);
- `model`;
- `background`.

These values are read only from `Runtime.context`, never from `config.configurable`: one channel everywhere, including the eval runner [call].

`DataService` instances are cached per (dataset, cutoff).

**Errors** [call]:
- The existing exceptions (`QueryError`, the compiler's and resolver's `ValueError`/`KeyError`, the forecasting errors) become subclasses of one `AnalystError` with a `hint`. Raise sites stay where they are.
- The tool layer turns `AnalystError` into a tool message: error plus hint. A rejected query is reported with its reason and never executed.
- Any other exception becomes a generic "internal error in <tool>" tool message and is logged to the trace. The run continues.
- A run that raises out of the graph fails every eval grader.

## 6. Tools

These are the project spec §5.3 tools, minus `run_python` (B2). Each is a thin LangChain tool over a `DataService` method:
- every tool returns the summary card text (≤ 4,096 chars) and records its handle in state;
- `AnalystError` becomes a tool message with its hint.

| Tool | Over | Notes |
|---|---|---|
| `query_metrics` | `DataService.query_metrics(MetricQuery)` | Governed path. The arguments are the `MetricQuery` pydantic schema. |
| `run_sql` | `DataService.run_sql` | Off-model path. |
| `describe` | `DataService.describe` | Catalog, schema, dimension values, or a handle's card. |
| `forecast` | `DataService.forecast` | Inline or background per §5.1. |
| `make_chart` | new | Writes a chart spec as JSON into state: history, forecast, bands, last-year overlay. Returns a one-line caption. No rendering in B1. |
| `read_knowledge` | `Memory.as_of(cutoff)` | Topics: metrics, quirks, context facts, forecasting guidance, recap. Progressive disclosure: the system prompt lists the topic names only. |

**The recap is on demand, not in the system prompt** [call]. This overrides project spec §5.8 step 5: it keeps the context baseline small and stops prose leaking past a cutoff.
- `recap.md` is free prose and can't be date-filtered, so `read_knowledge("recap")` under a cutoff returns "recap unavailable under a cutoff". Dated quirks and facts are still filtered by `as_of`.
- A date-filterable recap format is a Plan C calibration item.

## 7. Verify node and answer schema

These are unchanged from the project spec §5.2 and §5.7, and deterministic (no LLM). The Plan B specifics:

- **Number tracing**
  - Numbers are taken from the draft answer's structured claims and narrative. Years and dates are ignored, and so are numbers the user wrote in the question.
  - Each remaining number must match a value in a referenced handle (within rounding), or derive from handle values by one of: difference, % change, share or ratio.
  - **Derivation rules** [call]:
    - one hop only;
    - the two inputs may come from different handles the claim cites (for example forecast vs last year's actual);
    - chains such as "% change of a ratio" need a handle, so the agent should compute them with a governed comparison or `run_sql`.
  - **"Within rounding"** [call]: the stated number equals the handle or derived value rounded to the stated number's own precision. For example, "2.81M" matches 2,808,154, and "-3.0%" matches -2.99%. Integers stated in full must match exactly.
- **Causal claims**
  - A claim of type `causal` must use a hedge word (from a small list in the dataset-agnostic verify module) **and** cite a `context_facts` id that exists in memory and is dated on or before the cutoff.
- **Labels**
  - **Verified:** every number traced and every handle produced by `query_metrics` or by `forecast` on a governed handle.
  - **Verified · off-model:** every number traced, but some came from `run_sql`. The answer names the tool.
  - **Not verified:** a failure remains after one retry. The answer carries the reason.
- **Answer schema:** a pydantic `response_format`:
  - `narrative`;
  - `claims[]`: text, type, values, handle ids, fact ids;
  - `charts[]`;
  - `sql[]`;
  - `verification`: label and reason;
  - `background_tasks[]`.

## 8. Providers, middleware, LangSmith

### Providers

- `get_model(name)` uses `init_chat_model`. The provider is chosen from settings:
  - OpenAI is the default and must work in B1;
  - OpenRouter uses `langchain-openrouter`;
  - NanoGPT goes through the OpenAI client with its own `base_url`.
- OpenRouter and NanoGPT are wired but stay untested until their keys exist.
- The default model is `gpt-6-luna` [owner]. The exact id is checked against the OpenAI model list at build time.

### Middleware

- **Tool-result clearing:** LangChain's context-editing middleware (`ClearToolUsesEdit`) or an equivalent. Older tool outputs become a one-line stub with their handle id.
- **Context metering** (ours): counts bytes of tool or data content in each model request and writes it to LangSmith run metadata. This backs the README's "few KB in context" claim.
- **Cost** (ours): copies OpenRouter's reported cost into `usage_metadata`.
- **Limits:** `ToolCallLimitMiddleware` and `ModelCallLimitMiddleware` per run.

### Environment

`.env` exists, is gitignored, and holds the owner's keys. `.env.example` gains these variables:

- `LANGSMITH_TRACING=true`
- `LANGSMITH_PROJECT=big-data-agent`
- `ANALYST_MODEL`
- `ANALYST_MAX_BACKGROUND_TASKS=3`
- `ANALYST_MAX_CONCURRENT_QUERIES=3`
- `OPENROUTER_API_KEY`
- `NANOGPT_API_KEY`
- `NANOGPT_BASE_URL`

### `langgraph.json`

- Registers `analyst` (the supervisor graph) and `worker`.
- Sets `env: .env`.
- `make dev` starts it.

## 9. Eval set: single-turn, about 50 questions [owner]

### 9.1 Rows

The set lives in `evals/datasets/agent_single_turn.jsonl`, with gold results frozen under `evals/gold/` as Parquet. That's the same pattern as the 15 existing SQL gold rows, which seed it.

Each row has:
- `id`, `tier`, `category`, `question`;
- `cutoff` (optional);
- `gold_sql` (gold result frozen to Parquet) **or** `gold_check` (for non-tabular expectations such as "must decline");
- `key_values`: the gold numbers the answer must state, with `tolerance`;
- `expected_tools` (the expected trajectory);
- `expected_label`;
- `notes`.

### 9.2 Mix [call]

About 50 rows:

| Category | Count | What passes |
|---|---|---|
| Plain lookups and breakdowns (15 existing + new) | ~20 | `key_values` stated within tolerance |
| Comparisons and shares (YoY, previous period, share) | ~8 | key values + label Verified |
| Forecasts under cutoff replay (cutoffs 2025-07-31, 2026-01-31, 2026-07-31) | ~8 | Right series, grain and horizon; SeasonalNaive comparison stated; backtest metrics present. Accuracy against the real future data is **recorded as a score**, not a gate. |
| Off-model questions (semantic layer can't express them) | ~6 | key values + label Verified · off-model |
| Grounding traps (2026-02-23 drop, 2023-09 gap, congestion pricing) | ~5 | No unhedged or uncited cause. The gap is flagged. A hedged cause cites the right fact id. |
| Safe handling (dates past the cutoff, nonexistent column, destructive SQL request) | ~3 | Declines or explains. No fabricated number. |

### 9.3 Graders

These are deterministic code graders, shaped like openevals (`{key, score, comment}`):

- **`answer_values`:** every `key_value` appears among the answer's claim values within tolerance. **This is the gate.**
- **`label`:** the verification label equals `expected_label`. This is a gate for categories whose label is stated.
- **`grounding`:** taken from the verify report. Any causal claim must be hedged and cited. This is a gate for the trap category.
- **`trajectory`:** openevals trajectory match (unordered, superset) against `expected_tools`. This is **a score only, never a gate** [owner].
- **`forecast_skill`:** for forecast rows, the error against the real future values and whether the 80/95 intervals covered them. This is a score only.
- **`checks`** [call]: grades `gold_check`. It is a small closed set of named deterministic checks over the answer, its verify report and its handles' card meta:
  - `forecast_shape(series, grain, horizon)`: a forecast handle with that id set, frequency and horizon exists, its meta has the SeasonalNaive comparison and backtest metrics, and its `cutoff` matches the row;
  - `declined`: no claim values, and the narrative says why;
  - `quirk_flagged(quirk_id)`: the answer cites that quirk or fact id;
  - `no_causal_without(fact_ids)`.

  This is the gate for forecast, trap and safe-handling rows. New check names are added in code, never as free text.

**What the eval target returns** [call]: the structured answer, the verify report, and the ordered list of tool calls, all taken from the final graph state. Graders take these three things as input.

### 9.4 Running

- `scripts/eval_agent.py` (`make eval-agent`) uploads the JSONL as the LangSmith dataset `analyst-single-turn` [call]:
  - one dataset name for all time, so experiments compare in one view;
  - examples are upserted by row `id`;
  - each upload whose content hash changed gets a LangSmith dataset version tag (`v<n>-<hash8>`), and experiments record that tag;
  - frozen rows are never edited in place: a change gets a new id.
- It runs `langsmith.evaluate` with the graph invoked in-process, with `background=False`, one repetition by default and `--reps N` optional.
- The experiment is named `<model>-<git short sha>`.
- `--model` switches providers, so the owner can compare experiments in LangSmith.

### 9.5 Gold authoring [call]

- Claude drafts the ~35 new rows and gold SQL; the 15 existing rows are carried over.
- `scripts/build_gold.py` freezes the Parquet files. Rows with a cutoff take their gold from the **full** data, meaning the real future.
- The owner reviews the question list once before it is frozen.
- A second independent derivation is Plan D.

## 10. Done means (B1)

- `make test`, `make test-taxi` and `make lint` are green.
- `make dev` opens Studio. The hero question runs, the forecast goes to a background task, the agent answers a different question while it runs, then delivers the forecast answer labeled Verified.
- "Give me 3 charts for the most important insights" runs three investigations in parallel within the caps (§5.1a), and Studio shows neutral status lines and tool progress events while they run (§5.1b).
- `make eval-agent` produces a LangSmith experiment over all ~50 rows. Its per-grader scores are recorded in `docs/results/`, as the first baseline, not a target.
- LangSmith traces show cost, latency and the context-bytes metadata.
  - Runs that belong to one question (supervisor, worker, the supervisor run after the wake-up) share the thread id plus a `question_id` in metadata, so cost and latency per question add up across them [call].

## 11. B2 outline (a separate plan, written after B1 lands)

- Docker sandbox behind the `Sandbox` protocol:
  - persistent kernel per thread;
  - `--network none`, `cap_drop ALL`, read-only root;
  - CPU, memory and pid limits;
  - `/data:ro` honouring the cutoff;
  - `/artifacts:rw`.
- AST guard with an import allowlist.
- `run_python` with `load(handle)` and `save(df)` returning handles; the escape-attempt suite.
- Compaction middleware with the analytics digest (project spec §5.5.6).
- Eval rows that need Python.
- **Prerequisite** for B2: the owner's user can reach the Docker daemon. Today, calls fail with "permission denied" on `/var/run/docker.sock`; add the user to the `docker` group or start the daemon.

## 12. Left to the builder

- Prompt wording, middleware thresholds and the exact background-spawn thresholds.
- How `create_agent` hosts `AsyncSubAgentMiddleware` (verify on the pinned version).
- The chart-spec JSON shape, as long as it holds history, forecast, bands and a last-year overlay with handle references.
- How the verify node extracts numbers from the narrative.
- Test tooling details.
