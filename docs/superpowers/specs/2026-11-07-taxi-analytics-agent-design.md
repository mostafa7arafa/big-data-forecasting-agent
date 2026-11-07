# Big-Data Forecasting Agent: Design (Project 1)

_Status: draft, 2026-11-07. Owner: Mostafa Arafa. Research behind this doc is in `docs/research/01-03`._

Marked **[owner]**: decided by the owner. Marked **[call]**: a design call made while writing this doc. The owner can overturn any [call] item. Everything not marked follows from the original brief.

## 1. Intent

This is an AI data analyst that answers plain-language questions over **~162M NYC taxi trips**. It forecasts future demand with real statistical models and presents charts with explanations that stay grounded in the data.

The question it is built to answer well: *"Forecast next month's trips by borough, compare with last year, and explain the main trends."*

**Why it exists.** It is a portfolio flagship for GitHub and LinkedIn [owner], and the first module of a larger data system [owner]. The reader is a senior AI engineer. In the first 30 seconds they should see:

1. **An organized, eye-catching architecture.** Clear layers, each with one job, assembled from well-known libraries (LangChain/LangGraph, LangSmith, openevals, DuckDB, StatsForecast) rather than hand-rolled ones [owner].
2. **Proof it works, as numbers.** A results table with SQL accuracy, forecast skill against a baseline, the rate of made-up causes, and cost and latency per question.
3. **A product feel.** A polished chat-and-chart dashboard with an original brand character (in the spirit of Omni's "Blobby"), plus a demo video [owner].

**The distinguishing claim.** *"The model reasoned over 162M rows, its context never held more than a few KB of data, and its forecasts beat a seasonal baseline on held-out history."* This sentence goes on the README and must be backed by measured numbers.

## 2. Goals, non-goals, anti-goals

**Goals**
- Answer historical questions correctly over the full dataset without the data ever entering model context.
- Produce forecasts with prediction intervals, backtested on held-out periods, and compared against a seasonal-naive baseline.
- Explanations cite trends and seasonality from computed evidence. Causal claims appear only when hedged or backed by an allowed context fact.
- An agent harness that feels like a real analytics product: governed metrics, a visible plan, inspectable queries, and a verified/not-verified label on each answer.
- A swappable dataset and a swappable model, both changed through config [owner: both "plug and play" meanings are required]. New datasets are onboarded by **calibration** (§5.8), not by code changes.
- An agent that improves with use: a feedback loop into reviewed dataset memory, plus long-term memory per user across sessions (§5.9).
- Everything measurable through LangSmith: traces, latency, cost and eval experiments.

**Non-goals (v1)**
- Web search and runtime package installs. These are deferred [owner]. When added later, outside facts must be labeled as external and kept separate from data-derived evidence.
- Signing in with a subscription (Codex login, etc.) and a multi-provider product layer. That is Project 2 [owner]. v1 only needs API-key providers behind one interface.
- Multiple datasets at once, multiple users, auth and hosting. That is Project 3.
- Voice and multi-agent swarms.
- Building on archived libraries: `langgraph-codeact`, `langchain-sandbox`, `agent-chat-ui` (all confirmed archived on GitHub, 2026-11-07).

**Anti-goals: failure even if it "works"**
- Junky, hard-to-read code, or custom code where a well-known library already does the job [owner].
- Raw query results dumped into the prompt.
- Explanations that invent causes ("trips fell because of the storm") without an allowed fact.
- Headline numbers that cannot be reproduced from the repo.
- A generic "chat with a database" look.

## 3. Constraints

- **Local-first.** The owner's laptop: Linux, 16 cores, 30 GB RAM. Python 3.12 managed with `uv`, versions pinned in `uv.lock`. Docker is needed for the sandbox.
- **Models** come from **OpenAI** (baseline), **OpenRouter** and **NanoGPT** [owner], through `langchain-openai` / `langchain-openrouter` (first-party, 0.2.9). Anthropic is reachable via API key through OpenRouter or `langchain-anthropic` [owner: "Claude yes, via API"].
- **Stack** [owner]: LangChain v1 `create_agent` inside a LangGraph graph, LangSmith for observability and experiments, openevals for evaluators.
- **Data.** NYC TLC yellow taxi, 2023-01 through 2026-09: 45 monthly Parquet files, 2.6 GB, 161,781,947 rows. This is a frozen snapshot, and `scripts/download_data.sh` reproduces it. Zone lookup: `taxi_zone_lookup.csv`.
- **DuckDB 2.0.0** (released 2026-10-21) is new. Pin it and test the read-only lock-down settings explicitly.

## 4. Data facts the design must handle

These come from profiling the full dataset (7 s).

- **Coverage:** 1,369 days. Median 118k trips/day.
- **Bad timestamps:** 176 rows fall outside the file's month (some dated 2001). The base view excludes them.
- **Data gap:** 2023-09-21 to 2023-09-24 has near-zero trips (lowest 4.6k). It must be flagged, not silently modeled. It becomes a "data quirk" fact and an eval case.
- **Unexplained anomaly:** 2026-02-23 is low (26k). The data does not say why. It is a built-in grounding test.
- **Borough skew:** Manhattan ≈ 3.3M trips/month, Queens ≈ 350k, Brooklyn ≈ 150k. Bronx and Staten Island are tiny. Per-borough forecasts are a headline only for Manhattan, Queens and Brooklyn.
- **Schema drift:** later files add `cbd_congestion_fee` (congestion pricing, Jan 2025) and `request_source`. Views use `union_by_name`.
- **Dirty rows:** about 2.5% have fare ≤ 0 or distance ≤ 0, and 0.4% have an unknown zone. Each metric definition states whether these are included.

## 5. Architecture

```
 Dashboard: Rust + Leptos (WASM), WebGL charts, character
        │  HTTP + SSE (LangGraph Agent Server API)   │ chart data (downsampled, LOD on zoom)
 ┌──────▼──────────────── Agent harness (LangGraph) ───────────────────────┐
 │  plan ─► analyst (create_agent + middleware) ─► verify ─► answer       │
 │              │ tools                                                     │
 │   query_metrics  run_sql  describe  forecast  make_chart  run_python   │
 │   read_knowledge                                                         │
 └──────┬──────────────┬───────────────┬──────────────────┬──────────────────┘
   Semantic layer   DuckDB (RO)    StatsForecast     Docker sandbox (CodeAct)
   (YAML metrics)   over Parquet   + utilsforecast   /data:ro  /artifacts:rw
        └──────────────── Result store: data/artifacts/<session>/<id>.parquet ┘
 LangSmith: traces, cost, latency, datasets, experiments    Evals: pytest + openevals
```

### 5.1 Layers, one job each

| Layer | Job | Built on |
|---|---|---|
| `datasets/` config | Declares a dataset: files, base views, quirks, allowed context facts | YAML [call] |
| Semantic layer | Named metrics (trips, revenue, avg fare…), dimensions (borough, zone, hour…), time grains. Compiles a metric request to DuckDB SQL | Own small compiler over YAML [call] (see §5.3) |
| Data engine | Executes SQL read-only and writes every result to the result store | DuckDB |
| Result store | Every result is a handle (`r_<id>`) holding a Parquet file plus a summary card | Files + a small index |
| Forecasting | Backtested model selection and forecasts with intervals | StatsForecast, utilsforecast |
| Sandbox | Runs model-written Python in a persistent, locked-down kernel | Docker + Jupyter kernel [call] |
| Agent harness | Planning, tool use, context management, verification | LangGraph + `create_agent` + middleware |
| Model provider | One `get_model(config)` hiding OpenAI, OpenRouter and Anthropic | LangChain chat models |
| Observability | Traces, cost, latency, experiment comparison | LangSmith |
| Evals | Golden sets, graders, regression runs | pytest + `langsmith` testing + openevals + custom graders |
| Chart data service | Server-side downsampling (M4 / LTTB) and level-of-detail re-queries on zoom | `tsdownsample` (Rust core) or DuckDB `lttb` extension |
| UI | Chat, charts, plan, query inspection, review queue, character | Rust + Leptos (WASM) [owner] (see §8) |

### 5.2 Agent graph

This is a small custom LangGraph `StateGraph`, so the topology shows up in LangGraph Studio. Its main node is a `create_agent` agent [owner: graph agent + create_agent].

1. **plan.** For multi-step questions, writes a short visible task list (Omni/Genie style). Simple questions skip it.
2. **analyst.** A `create_agent` loop with ReAct over typed tools, plus `run_python` as the CodeAct tool [owner: ReAct + CodeAct]. Typed tools are the default. `run_python` is for analysis they can't express.
3. **verify.** Deterministic checks, no LLM:
   - every number in the draft answer must match a value in a referenced result handle (within rounding);
   - each causal claim must be hedged or cite an allowed fact;
   - the forecast must carry its backtest metrics.

   On failure it loops back to analyst once with the specific error, then answers with a "not verified" label.
4. **answer.** Structured output (`response_format`):
   - narrative;
   - claims, each with type (descriptive, comparative, forecast or causal) and the evidence handles behind it;
   - chart specs;
   - SQL and code used;
   - a verification status and reason (Verified / Not verified: off-model query, unverified number, unsupported claim).

State carries: messages, plan, result handles, chart specs, forecast runs, verification report and token usage.

**Verification labels** [call]:
- **Number tracing.** Every number in an answer must trace to a handle. Any tool's handle counts. The verify step also recomputes simple arithmetic between handle values (difference, % change, share, ratio) and accepts the number if it matches within rounding. Anything else needs a handle.
- **Comparisons are first-class.** `query_metrics` supports period-over-period comparison (`compare_to: previous_period | same_period_last_year`) and shares. The hero question's "compare with last year" therefore comes from the governed path, not model arithmetic.
- **Three labels:**
  - **Verified:** all numbers traced and produced only by governed tools (`query_metrics`, and `forecast` on a governed handle).
  - **Verified · off-model:** all numbers traced, but some came from `run_sql` or `run_python`. The chip shows which tool.
  - **Not verified:** an untraced number or an unsupported causal claim remains after one retry. The chip shows the reason.

### 5.3 Tools

All tools return a **summary card**, never raw data. A card holds the handle id, row count, columns with types, a 10-row preview, basic stats per numeric column, the SQL or code that produced it, and the elapsed time.

| Tool | Purpose |
|---|---|
| `query_metrics(metrics, dimensions, filters, grain, time_range)` | **Governed path, the default.** Compiles through the semantic layer. Answers from it can be "Verified". |
| `run_sql(sql)` | Raw read-only SELECT over the base views, used when the semantic layer can't express a question. Answers that depend on it are labeled "off-model". The guards from the spike apply: single SELECT only, external file access off, configuration locked, timeout, row cap. |
| `describe(target)` | Schema, metric catalog, dimension values (fixes "Manhattan" vs "manhattan"), or a result handle's card. |
| `forecast(handle, horizon, freq, levels=[80,95])` | Backtested model choice and a forecast. Returns a handle and metrics. |
| `make_chart(handle(s), spec)` | Builds a chart spec (history, forecast, interval bands, last-year overlay). The UI renders it. Returns a caption to the model. |
| `run_python(code)` | CodeAct. Runs in the sandbox session. `load(handle)` gives a DataFrame, and `save(df, name)` returns a new handle. Stdout is capped. |
| `read_knowledge(topic)` | Loads on demand: metric definitions, data quirks, allowed context facts, and forecasting guidance. Uses skill-style progressive disclosure, so none of this sits in the prompt by default. |

Semantic layer [call]: write a small YAML-to-SQL compiler (about 300 lines) instead of adopting a heavy semantic-layer product. The reason is that the governed path is the project's centerpiece and must stay readable. The builder may swap in a lightweight open-source semantic layer if one fits DuckDB cleanly and keeps the layer readable.

### 5.4 Forecasting

- **Models:** AutoETS, AutoARIMA and MSTL (seasons 7 and 365) for daily data. Baselines are SeasonalNaive (7) and HistoricAverage.
- **Model selection:** rolling-origin backtest picks the model. The default for daily data is horizon 28, step 7, with 12 or more windows.
- **Metrics:** MASE, relative MAE vs SeasonalNaive, coverage at 80/95, and Winkler/scaled CRPS, all via `utilsforecast.losses`.
- **Holidays and known shocks** are reported as separate slices. They are not dropped and not hand-tuned.
- **The 2023-09 gap** is treated as missing, and the treatment is stated in the output.
- **Every forecast carries its backtest scorecard to the UI.** If no model beats the seasonal baseline, the answer says so.
- **Forecast origin** [call]. Forecasts start at the end of the data (2026-09-30), not at today's date. "Next month" means the first full month after the data ends (October 2026). Every forecast answer states the data cutoff. Requests that reach further out are allowed, but the horizon is said explicitly and wider intervals are expected.
- **Forecast at the grain asked** [call]. Monthly questions use a monthly model on the monthly series (season 12; 45 points, enough for 2+ yearly cycles). Daily questions use the daily models above. Daily intervals are never summed into monthly intervals. The hero question ("next month by borough") is therefore a monthly forecast. Its backtest uses rolling origins with horizon 1–3 months over the last 12 months, and a daily forecast for the same month is offered as a drill-down chart.

### 5.5 Context management (core differentiator)

1. **Handles, not data.** The model sees summary cards only. Full results stay on disk or in the sandbox.
2. **Capped output.** The sandbox truncates stdout and tracebacks to about 4 KB. DataFrames print as compact summaries.
3. **Stateful sandbox.** Variables persist across `run_python` calls in a session, so earlier work is referenced, not re-sent.
4. **On-demand knowledge** via `read_knowledge`.
5. **Tool-result clearing.** Older tool outputs in history get replaced by a one-line stub with their handle id.
6. **Compaction, Claude Code style** [owner]. Near a token threshold, the conversation gets replaced by a structured digest: goal, confirmed findings with their numbers and handles, live handles, open questions and user preferences. This is our own middleware, inspired by LangChain's `SummarizationMiddleware`, because the digest format is analytics-specific.
7. **Measured.** Context size per model call is logged to LangSmith as metadata, so the "few KB of data in context" claim is measured, not asserted.

### 5.6 Sandbox and guards

Arbitrary Python is allowed inside the sandbox, with guards [owner].

- **Container:**
  - Docker running a persistent Jupyter kernel per session;
  - `--network none` and `cap_drop ALL`;
  - read-only root filesystem;
  - CPU, memory and pid limits;
  - Parquet mounted at `/data:ro`, artifacts at `/artifacts:rw`;
  - a timeout on each call.
- **Pre-exec check:** a static (AST) check with an import allowlist (duckdb, pandas, polars, numpy, scipy, statsmodels, statsforecast, plotly) and a blocklist (`subprocess`, `socket`, `os.system`, `ctypes`, …). A violation returns a clear error to the model.
- **Behind a `Sandbox` protocol** (`run(code) -> ExecResult`), so E2B or LangSmith Sandboxes can replace Docker for a hosted demo. Docker is not a hard security boundary. That is acceptable locally, and the README says so.
- **Middleware limits:** `ToolCallLimit` and `ModelCallLimit` per run.

### 5.7 Grounding policy

- **Allowed claim types:** descriptive (from data), comparative (from data), forecast (from a forecast handle) and causal.
- **Causal claims** must be hedged ("coincides with") **and** cite an entry from the dataset's allowed context facts. Examples: US holiday calendar, congestion pricing start 2025-01-05, known data gaps.
- **Anything else** is phrased as an observation without a cause. The 2026-02-23 drop gets described, not explained.

### 5.8 Calibration: learning a new dataset [owner]

The code is **dataset-agnostic** [owner]. Nothing in `src/` knows about taxis. Everything dataset-specific lives in that dataset's **memory**, which a one-time calibration run produces when the dataset is connected. Taxi is the first calibrated dataset, not a special case. Research: `docs/research/04-dataset-calibration.md`.

**Pipeline**
1. **Profile** (deterministic code, no LLM):
   - schema, row counts;
   - time column and grain (event rows vs pre-aggregated), coverage, gaps, implausible timestamps;
   - distributions and distinct values; measures vs dimensions;
   - hierarchies via functional-dependency tests (zone → borough), joins to lookup tables;
   - dirty-row rates.

   For each main series it also computes seasonality (ACF/STL), anomalies (robust z-score) and forecastability. Output: `profile.json`.
2. **Draft** (LLM). Reads the profile, runs its own exploratory queries, and drafts the memory files below. It names, describes and proposes. It never invents statistics; every number comes from the profile or a query.
3. **Check by running.** Every drafted metric and verified query is executed. Anything that errors or gives implausible values is fixed or dropped before review.
4. **Review gate** (human). The owner sees the draft as a diff in the dashboard and accepts, edits or rejects each item. Only approved items are **governed**, the requirement for the "Verified" label.
5. **Use.** The recap goes into the system prompt. The other files load on demand via `read_knowledge`.
6. **Drift check.** When new data lands (e.g. a new monthly file), a cheap re-profile flags new columns, new gaps, distribution shifts and coverage changes. The result is a proposal for re-calibration, never a silent change.

**Memory files per dataset** (`datasets/<name>/memory/`). Every item carries `source` (profile, llm, owner, feedback), `status` (proposed, approved, rejected) and `confidence`.
- `semantic.yaml`: metrics, dimensions, synonyms, time grains, joins, dirty-row filters as SQL predicates.
- `quirks.yaml`: gaps, anomalies, schema changes, known data issues.
- `series.yaml`: forecastable series, season lengths, recommended model family.
- `context_facts.yaml`: allowed external facts for hedged causal claims (holidays, policy dates). Owner-curated.
- `verified_queries.yaml`: question → query pairs, learned over time.
- `lessons.md`: procedural notes ("borough totals must exclude zone 264/265"), plus a list of failed approaches.
- `recap.md`: a short plain-language summary of the dataset, written for the agent.

**The second dataset** [call]. Near the end, calibrate a second dataset unrelated to taxis to prove the code is generic. Candidate: a public retail or energy-demand time series, chosen at that point. Score calibration quality against a hand-written gold semantic layer by running each metric, not by text match.

### 5.9 Feedback loop and long-term memory [owner]

There are two kinds of memory, kept apart:

| | Dataset memory (§5.8) | User memory |
|---|---|---|
| Scope | One dataset, shared by everyone | One user, all sessions, all datasets |
| Holds | Definitions, quirks, verified queries, lessons | Preferences (units, chart style, default borough), past findings and their handles, ongoing questions |
| Writes | Proposals; human approval required for governed items | Agent writes directly; user can view, edit and delete |
| Authority | Defines what is "Verified" | Never overrides dataset definitions |

- **Feedback loop.** Each answer gets 👍/👎 with an optional correction. A 👍 on a governed-path answer proposes a verified query. A 👎 with a correction becomes a proposed lesson. Proposals pile up in a review queue in the dashboard, and nothing becomes governed without approval. Benchmark and eval questions are never used for learning, so evals can't leak.
- **Storage** [call]: LangGraph's long-term `Store`, namespaced `(user_id, ...)` and `(dataset, ...)`. Use SQLite or Postgres locally, behind LangGraph's store interface. v1 runs single-user locally, but every memory read and write already carries a `user_id`, so multi-user (Project 3) needs no redesign.
- **Measured.** The eval suite runs the SQL set three ways: no memory, after calibration, and after a simulated feedback round. That shows whether learning actually helps.

## 6. Evaluation

Graders are deterministic first. LLM judges are calibrated against hand labels.

| Eval set | Size | Grader |
|---|---|---|
| SQL/metrics questions, 3 difficulty tiers | ~70 | Per-question result validators (numeric tolerance, set match), DAB-style. Gold answers derived twice, independently, and cached as Parquet. 3 runs each, report pass@1 |
| Forecast backtests | Manhattan, Queens, Brooklyn, all-boroughs | MASE, relative MAE vs SeasonalNaive, interval coverage |
| Agent forecasting tasks | ~15 | Trajectory (forecast tool used, horizon right), reported numbers match tool output, chart has intervals |
| Explanation grounding | ~40 | Claim extraction → numeric check vs evidence → causal-claim rubric judge, with κ vs owner labels |
| Adversarial | ~20 | Out-of-range dates, missing columns, causal bait, prompt injection, destructive SQL, sandbox escape attempts. Graded on refusal or safe handling. False-refusal rate measured on matched controls |

- **Cutoff replay ("time travel")** [owner]. The engine can be started with a data cutoff, e.g. 2026-07-31. Every view, the semantic layer, the sandbox mount and the knowledge files see only data up to the cutoff. The cutoff is enforced in the engine, so the agent cannot see past it. The agent then answers forecasting questions about the following months, and its answers and forecasts are scored against the real data we hold. This tests the whole agent end to end (questions, forecasts, intervals, explanations), not just the models.
  - Run it at several cutoffs to avoid luck from one season. My call: 2025-07-31, 2026-01-31 and 2026-07-31.
  - The same replay is how we **settle the forecasting model lineup**. A model stays in the default candidate set only if it beats SeasonalNaive at most cutoffs.
- **Ground-truth labels** [owner]. The owner hand-labels ~40 explanations for the grounding judge's agreement score (κ). A subagent on a different model derives SQL gold answers a second time, and the owner reviews only the disagreements.
- **Models** [owner]:
  - OpenAI is the baseline model for development and the README headline, because it is fast and cheap. The owner named `gpt-6-luna`; check the exact model id against the OpenAI model list at build time.
  - OpenRouter and NanoGPT (both credited) provide comparison models. NanoGPT is OpenAI-compatible, so it is reached through the same provider interface with a different `base_url` [call].
  - At least one comparison model goes in the README as a LangSmith experiment.
- **README headline table:**
  - SQL pass@1 by tier, with 95% CI;
  - forecast MASE and skill vs baseline, with coverage;
  - unsupported-causal-claim rate;
  - adversarial pass and false-refusal rates;
  - median and p95 latency;
  - cost per question;
  - peak data bytes in context.
- **Also on the README:** a failure-mode breakdown, pinned versions, and links to the LangSmith experiments.
- **Cost tracking for OpenRouter:** a small middleware writes OpenRouter's reported cost into `usage_metadata`, because LangSmith has no price entries for OpenRouter model ids.

## 7. Repo layout [call]

This follows LangChain's `new-langgraph-project` template conventions.

```
src/analyst/
  graph.py  state.py  prompts.py  config.py
  providers/        get_model(): openai | openrouter | anthropic
  semantic/         YAML loader + compiler
  data/             DuckDB engine (read-only), result store, summary cards
  forecasting/      StatsForecast wrapper, backtests, metrics
  sandbox/          base.py (protocol), docker.py, ast_guard.py, kernel image
  tools/            one file per tool
  middleware/       compaction, tool-result clearing, cost, context metering
  verify/           number and claim checks
  calibration/      profiler, LLM drafter, run-checker, drift check
  memory/           dataset memory (proposals, approvals), user memory (LangGraph Store)
datasets/nyc_taxi/  dataset.yaml (files, views, cutoff)  memory/ (semantic.yaml, quirks.yaml, series.yaml,
                    context_facts.yaml, verified_queries.yaml, lessons.md, recap.md, profile.json)
evals/              datasets/*.jsonl  graders/  test_*.py
web/                Rust + Leptos dashboard (own Cargo workspace)
docker/  scripts/  docs/  langgraph.json  pyproject.toml  Makefile  .env.example  README.md
```

The README leads with an architecture diagram, the results table, a GIF and the video link, then "run it in 3 commands".

## 8. Dashboard and brand

- **Shape:**
  - a chat pane;
  - a visible plan;
  - chart cards with history, forecast, 80/95 bands and a last-year overlay;
  - a verification chip per answer;
  - "Query details" showing SQL, code and handles;
  - a backtest scorecard per forecast.
- **Character:** an original character that represents the brand, animated in the UI and used in the README and video. It must not resemble Blobby. Name and design are to be decided in the design phase.
- **Front end: Rust + Leptos, compiled to WebAssembly** [owner]. The agent is an isolated Python service, and the web app is a separate client that talks to it over HTTP and streaming [owner]. The client speaks the LangGraph Agent Server's REST + SSE API (threads, runs, streaming events) through a small typed Rust client module. That replaces the archived `agent-chat-ui` and the JS-only `useStream`. Frontend design uses the "impeccable" design skills [owner], to be installed before UI work; they apply to the HTML/CSS layer.

**Large-data charts** [owner: rendering must not lag with millions of points]. The fix comes in three layers:
1. **Never ship raw points.** The chart contract is a JSON or Arrow payload produced on the server, never raw query results:
   - series are downsampled to roughly the chart's pixel width using **M4** (keeps first/last/min/max per pixel column, so spikes and gaps survive) or **LTTB**;
   - the payload states the original row count and the method used.
2. **Level of detail on zoom.** Zoom and pan request the visible range again from the chart data service at full detail where it fits. This works like Reflex `xy`.
3. **GPU rendering.** Charts render through **WebGL** from typed arrays, not per-point objects. Long chats stay smooth: off-screen charts are virtualized and frozen to images, and only visible charts hold GPU buffers.

**Performance budgets**, which become tests:
- A 1M-point scatter, or a downsampled 10M-row series, renders in < 200 ms and pans or zooms at ≥ 50 fps on the owner's laptop.
- A chat with 30 charts scrolls without dropped frames.

**Left to the builder:** a Rust-native WebGL renderer versus a JS WebGL chart library bridged through `wasm-bindgen`, chosen by a spike against these budgets. Rust-native is preferred if it meets them in reasonable effort.

**Toolchain:** the Rust toolchain is not installed yet. Install `rustup`, the `wasm32-unknown-unknown` target and `cargo-leptos` at build time. CSR vs SSR in Leptos is the builder's call, as long as the budgets hold.
- **Demo video:** a ~2 minute screen recording of the hero question end to end, plus a context-size and eval-results segment [owner: video; format is a call].

## 9. Build order

The SQL and forecasting tools must be proven independently before they are connected to the agent [brief]. Subagents build each step, and the owner's reviewer (Claude) reviews it [owner].

1. Foundation: config, dataset registration, DuckDB engine with cutoff support, result store and cards.
2. Calibration step 1 (deterministic profiler), run on taxi data.
3. Semantic layer and `query_metrics` (incl. comparisons), with taxi memory files written by hand at first and the SQL golden set started.
4. Forecasting module, backtests and cutoff replay at the model level. This settles the model lineup.
5. Sandbox, AST guard and `run_python`, with an escape-attempt test suite.
6. Agent graph, providers, middleware (context, compaction, cost), verify node, LangSmith wiring.
7. Calibration steps 2–6 (LLM draft, check, review gate, drift). Taxi's hand-written memory becomes the gold set to score the agent's own draft against.
8. Feedback loop and long-term memory.
9. Full eval suite, agent-level cutoff replay and README numbers.
10. Dashboard (chat, charts, review queue, memory views), character, video.
11. Second dataset calibration as the generality proof.

The existing code (`src/taxi_forecast_agent/sql_tool.py`, `forecast_tool.py`, `scripts/inspect_data.py`) is a throwaway spike. Ideas from it may be reused, but not the structure.

## 10. Left to the builder

- Exact middleware thresholds, prompt wording, and preview and summary sizes.
- The semantic-layer YAML schema and the compiler internals.
- The sandbox exec mechanism (Jupyter kernel vs a small exec server), provided state persists.
- The chart renderer (Rust-native vs bridged JS WebGL library) within the §8 performance budgets, plus the chart payload schema details.
- Test tooling details beyond pytest + LangSmith.
