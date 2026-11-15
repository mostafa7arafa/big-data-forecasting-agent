# Dashboard design (Plans W1 and W2)

Status: draft for owner approval, 2026-11-15 (builder check done). Binding parent spec: `2026-11-07-taxi-analytics-agent-design.md` (§8 updated to Nuxt + Vue the same day). Product record for the design work: `PRODUCT.md`. Visual direction contract: `.impeccable/surfaces/web-app-pages-index-vue.md`.

## 1. Intent

The dashboard is the agent's product face and the centrepiece of the owner's portfolio. A senior AI engineer who opens it should think: *this person knows what matters to me*. That means two things are on screen at all times [owner]:

1. **Charts that tell the whole story.** The agent never draws only its result. A forecast is history in one colour continuing into the forecast in another, with its ranges, a marked origin, last year for comparison and a track record against a baseline.
2. **The machinery, visible.** Tokens (in, out, cached), estimated cost, context-window fill, the reasoning summary, a timeline of steps with durations, time to first token, total time, the model, and why the answer got its verification label. Plus the "full duplex" behaviour [owner]: the user keeps chatting while long work runs in the background, and can steer or stop the agent mid-answer.

The order changed on 2026-11-15 [owner]: the dashboard comes before Plan C (calibration and memory), so charts and design can be seen and judged first, and Plan C's review queue lands in a real UI instead of a throwaway CLI.

## 2. Goals, non-goals, anti-goals

**Goals**
- The main workspace screen: thread rail, chat with streamed answers and charts, run-trace panel, background-task tray, steer/queue/stop input.
- A typed chart catalogue the agent chooses from (§4), drawn beautifully by our renderer in light and night themes.
- A per-run trace recorded by the backend, streamed live and saved with the thread (§5).
- Steer, queue and stop on a running answer; a visible background-task tray (§6).
- The swarm character (§7).
- Large-data performance budgets as automated tests (§8).
- 👍/👎 with optional correction on every answer, recorded now and consumed by Plan C.

**Non-goals (this design)**
- Review queue, dataset/user memory views, dataset switcher and calibration run view: built with Plan C.
- Auth, multi-user, hosting (Project 3). Per-user character customization (deferred by owner).
- Phone-first layout. Desktop and laptop first; the screen stays usable down to tablet width [call].
- The demo video and README animation (after W2; they reuse the character).

**Anti-goals** (failure even if it works)
- A chat demo that draws only the forecasted months.
- Any forecast chart where predicted values are not visually distinct from history.
- Any number in the UI that does not come from a result handle or the recorded trace. No invented metrics, no placeholder stats in the shipped app.
- A debug-panel look, or the generic dark neon "AI dashboard".
- Anything resembling Omni's Blobby.
- Taxi specifics hard-coded in `src/analyst/` or in the web app's code.

## 3. Constraints

- Stack [owner]: Nuxt + Vue 3 + TypeScript, client-side rendered (`ssr: false`), its own pnpm workspace in `web/`. Agent API through `@langchain/vue` `useStream`. Don't use the archived `agent-chat-ui`.
- One backend process: the existing LangGraph Agent Server. New HTTP endpoints are custom FastAPI routes registered through `http.app` in `langgraph.json`.
- `src/analyst/` stays dataset-agnostic (the existing test enforces it). Dataset-specific chart assets such as map geometry live in `datasets/<name>/`.
- The engine and sandbox security model is unchanged. The chart service reads **result handles only**, through the ResultStore segment for the run's (dataset, cutoff). It never queries raw data, so charts can't leak past a cutoff.
- Tools return cards and handles, never raw rows (spec rule). Chart data travels over the chart endpoint, not through the model's context.
- Visual direction [owner]: Survey Sheet (USGS topographic quadrangle world), light and night themes both first-class. Contract in the surface brief; DESIGN.md is written from the built result at the end of W2 by the impeccable documenter.

## 4. Charts

### 4.1 Who does what (approach A, owner-approved)

- **The agent decides what to show.** It calls `make_chart` with a kind and a few fields: which result handle(s), which columns, a title, which extras (last year, origin). It never writes chart code or chart definitions and never touches colour, axes or layout.
- **Server code checks and prepares.** `make_chart` validates the choice against the handle's card (handles exist, columns exist and have the right types for the kind) and records a `ChartSpec`. The chart service later loads the data from the handle, downsamples it, computes formats and units, and returns a draw-ready payload.
- **The Vue renderer draws.** One renderer per kind, styled only by design tokens, so a chart cannot come out ugly whatever the agent picks.

Rejected: (B) the agent writes Vega-Lite specs (ugly or invalid output, token cost, and per-kind downsampling becomes hard); (C) typed kinds plus a Vega-Lite escape hatch (reintroduces B's risk; can be revisited later).

### 4.2 The catalogue

`ChartSpec` becomes a discriminated union on `kind`. Every kind carries `title` and at least one handle. Fields below are the agent-facing arguments; everything else is derived.

| Kind | Agent supplies | Notes |
|---|---|---|
| `line` | handle, x (time), y, optional series | Up to 6 series; more collapse into "other" with a note. |
| `area` | handle, x, y, series, `stacked` | Stacked shares when `stacked`. |
| `forecast_fan` | history handle, forecast handle, y, `last_year` flag, optional `series_id` | Origin, 80/95 bands, model name and backtest scorecard come from the forecast handle's card and run record. A multi-series forecast draws one series (`series_id`, default the first headline id); several series go in `small_multiples` of fans [call]. Optional `replay`: the most recent backtest fold's predictions drawn against what actually happened, from the forecast run's own backtest (no offline replay files involved) [call]. The SeasonalNaive baseline is stored with the forecast and drawn as a toggleable faint line, off by default; the scorecard always shows it [call]. |
| `bar` | handle, category, value, optional series, `orientation`, `mode` (grouped/stacked), optional `top_n` | Ranked bars sort by value; `top_n` folds the rest into "other". |
| `heatmap` | handle, x, y, value | e.g. hour × weekday. Sequential palette. |
| `map` | handle, key column, value, `geometry` name | Geometry from `datasets/<name>/geo/<geometry>.geojson`, named in `dataset.yaml`. Datasets without geometry don't offer this kind (the tool rejects it with a hint). |
| `histogram` | handle, value column, or an already-binned handle (bin, count) | Bins computed by the chart service from the handle's rows, capped; binning across raw data is a query the agent runs first. |
| `box` | handle, group, value | Quartiles from the handle's rows. |
| `scatter` | handle, x, y, optional color | Above a point cap the payload switches to a density grid. |
| `kpi` | handle, value column, label, optional comparison column, optional trend handle | Headline number plus a small trend line. |
| `small_multiples` | handle, facet column, inner kind (`line`/`bar`/`area`/`forecast_fan`), inner fields | One mini chart per facet value, shared scale. The hero question ("next month by borough") is small multiples of fans. |

The verifier treats every handle a chart references as cited (today `claims.py` reads three fixed fields; it moves to a `ChartSpec.handles()` method). The system prompt gets a short chart-choice policy (forecast → `forecast_fan`, share over time → stacked `area`, ranking → ranked `bar`, two time dimensions → `heatmap`, geography → `map`). The agent eval's chart checks move to the new spec (forecast answers must use `forecast_fan` with bands).

### 4.3 Styling rules (owned by the renderer)

- **Colour roles.** History/actuals: ink. Agent predictions: survey red, **reserved for predicted marks only** (forecast lines, predicted values in the narrative). Comparison and last year: water blue, dashed. Uncertainty ranges: contour brown hairlines with index labels (80, 95) and a woodland tint between them. Categorical series: a fixed, colour-blind-checked survey palette of six. Sequential (heatmap, map): hypsometric tints.
- **Origin.** Every forecast chart marks the forecast start with a vertical line, a label with its date, and a benchmark triangle at the axis.
- **Numbers.** Tabular figures; compact units (3.93M, 12.4K); currency and percent from the column's semantic type when the card provides it; axis ticks at round values.
- **Themes.** Every token has a light and a night value; charts switch with the UI.
- **States.** Each chart has loading (skeleton of its own shape), empty, error (with the reason) and "downsampled from N points" states.

### 4.4 Chart service

- `make_chart` stores the spec and returns a chart id plus a short caption to the model.
- `GET /charts/{dataset}/{segment}/{chart_id}?width=<px>&from=&to=` (custom route) loads the referenced handles from the ResultStore segment of the chart's (dataset, cutoff), recorded in the spec, and returns a columnar payload: typed arrays for x/y as JSON or Arrow, at about one point per pixel column using M4 (keeps first, last, min and max, so spikes and gaps survive).
- Zoom and pan re-request the visible range at full detail where it fits (level of detail).
- `tsdownsample` (Rust core) for M4/LTTB [call]; the builder may use DuckDB's equivalent if simpler.
- Payload schema is versioned and tested with fixtures shared by the backend tests and the web tests.

## 5. The run trace (approach A, owner-approved)

### 5.1 What is recorded

Per run, a `RunTrace` saved in thread state (a new `trace` state key, one record per turn) and streamed live as custom stream events:

- **Steps:** each node and tool call with start/end time, status (ok, error, cancelled), and for model calls tokens in, out, cached and reasoning (as the provider reports them in `usage_metadata`).
- **Totals:** tokens, estimated cost, time to first token, total time.
- **Cost:** from a per-model price table in config (input, cached input, output per million tokens). A model without prices shows "cost n/a", never a guess.
- **Context fill:** the last model call's input tokens against the model's context window (from the same config table), plus the compaction trigger.
- **Reasoning summary:** streamed and saved when the provider exposes one (OpenAI Responses reasoning summaries); otherwise the trace says "not provided by this model".
- **Events:** compaction, background task started/finished (linked to the task's own trace), interruptions (steer/stop), the verification label and its reason.

### 5.2 Where it lives

- The trace belongs to the thread in the Agent Server's own storage. Reopening a thread shows its full trace.
- LangSmith is a mirror for debugging and eval experiments only; the dashboard never reads it. The free plan's ~14-day trace retention therefore doesn't affect the product [owner question answered].
- W1 verifies that local storage (`langgraph dev` persists to `.langgraph_api/`) survives a restart. If it doesn't, threads move to a durable checkpointer (SQLite) for local runs.

### 5.3 How it is collected

A middleware on the analyst agent and the worker (model-call and tool-call wrappers) plus the outer graph nodes write step records and emit stream events. It must not block the event loop (blockbuster in `langgraph dev`). The trace is tested with a fake model that reports known usage.

## 6. Full duplex

- **Background tasks (exist since B1, now visible).** The rail's tray lists running and recent tasks from state, with live progress events from the worker, elapsed time, and a link to the task's trace. When a task finishes, its result arrives in the conversation as its own answer (the existing notification run), with its chart and trace. The user keeps chatting meanwhile.
- **Steer.** Sending while a run streams offers *steer* (stop the current run and start a new one with the new message added) or *queue* (answer after the current run). Both map onto the Agent Server's multitask strategies. The builder picks between "interrupt" and "rollback" for steer: the interrupted turn's partial output stays visible and marked as interrupted, and the thread is left consistent (no dangling tool calls).
- **Stop.** Cancels the run.
- **Cleanup is a requirement.** A cancelled run releases its sandbox and in-flight queries, and records the interruption in the trace. Tested.

## 6b. Cutoff and model pickers, and "reveal what happened" [owner, 2026-11-15]

- **Pickers.** The top bar has a data-cutoff picker (latest data, or a past date) and a model picker (the configured provider specs). They are set when a thread starts and are fixed for that thread; changing either starts a new thread. Every run of the thread passes them through the existing run context (`AnalystContext.cutoff`, `.model`), so every chart in a thread comes from one (dataset, cutoff) segment. The thread rail shows each thread's cutoff and model.
- **Reveal.** In a thread with a past cutoff, a forecast chart offers "Reveal what happened". It draws the real values after the cutoff over the forecast, in ink, with the forecast's error stated (e.g. "actual 3.88M, forecast 3.93M, inside the 80% range").
- **The agent never sees the reveal.** It is a chart-service endpoint (`GET /charts/{dataset}/{segment}/{chart_id}/reveal`) that re-runs the history handle's recorded source (the governed metric request or the validated SELECT) on the uncut dataset, limited to the forecast horizon, and returns only those points to the browser. Nothing is written into the thread's messages, state or ResultStore segment, so the agent's later turns stay blind. Only available after the answer has finished. Tested: the reveal's output never reaches a model call.

## 7. The swarm character [owner-approved]

- About 300 dots drawn on a canvas. Each state is a set of target positions the dots ease toward; new states are cheap to add.
- States, driven by stream events: **idle** (a loose creature with two eyes, breathing), **thinking** (orbiting swirl, while reasoning streams), **querying** (dots pulled through a funnel, while `query_metrics`/`run_sql`/`describe` run), **forecasting** (a rising line fanning into futures), **comparing** (bars), **heatmap** (a grid), **done** (settles, eyes smile), **unsure** (scatters, flickers, shrugs; on "not verified" or a declined causal claim).
- Sizes: large on the empty start screen, small beside each streaming answer.
- `prefers-reduced-motion`: a static pose per state, no easing.
- Colours from the theme tokens (ink dots, red when forecasting).
- Name: open, chosen by the owner once it moves. Progress lines are rewritten in its voice afterwards (the Plan B status-line follow-up).

## 8. Performance budgets (tests, from spec §8)

- A 1M-point line chart: first render under 200 ms, measured client-side from a 1M-point payload already in memory [call]. This is a renderer stress test independent of downsampling (normal payloads are about pixel width, but a zoomed level-of-detail request can hand over full detail).
- Pan and zoom at 50 fps or better.
- A thread with 30 charts scrolls without dropped frames; off-screen charts are virtualized or frozen to images, only visible charts hold GPU or canvas buffers.

Measured with Playwright in a headless Chromium. Each budget is a test that fails when missed.

## 9. Plans and order

**W1: dashboard backend** (Python, no UI, testable on its own)
1. `ChartSpec` union, the new `make_chart`, verifier and grader updates, chart-choice prompt policy.
2. Chart service: custom route, handle loading per (dataset, cutoff), M4/LTTB, level of detail, per-kind payloads, fixtures.
3. Map geometry for taxi zones in `datasets/nyc_taxi/geo/`, declared in `dataset.yaml`.
4. Run trace: middleware, state key, stream events, price and context-window table, reasoning summaries.
5. Steer/queue/stop support and cancellation cleanup; storage durability check.
6. Feedback endpoint: records (thread, run, rating, optional correction) locally for Plan C.
6b. Reveal endpoint (§6b) and the thread's cutoff/model recorded in thread metadata.
7. Agent eval re-run on the new chart spec. Merge gate [call]: at least 38/55 (within two questions of the 40/55 baseline, for LLM noise), python still 5/5, no category down by more than one question. Questions that newly fail only on the stricter chart check count as regressions and get fixed in the prompt policy, not waived. Forecast changes in W1 (baseline predictions and backtest folds stored with the forecast) keep the Plan A forecasting tests green.

**W2: web app** (Nuxt + Vue)
1. Renderer spike against the budgets: uPlot (canvas, fast time series) vs a WebGL/WebGPU library; heatmap and map candidates; pick per kind.
2. App shell, theme tokens (light and night), Survey Sheet components, `useStream` wiring.
3. Answer card: narrative with predicted values in red, verification mark with reason, chart cards, query details, scorecard, 👍/👎.
4. Chart renderers for every kind with all states.
5. Trace panel with live counters, context budget, step profile, reasoning.
6. Background tray, steer/queue/stop, cutoff and model pickers, reveal on forecast charts.
7. Swarm character.
8. Performance tests, impeccable critique/audit/polish, finish review, DESIGN.md.

Then Plan C (calibration and memory, with the review queue and memory views in this UI), then Plan D (evals at scale).

## 10. Left to the builder

- The renderer per chart kind, decided by the W2 spike against §8.
- Payload encoding (JSON typed arrays vs Arrow) and the exact payload schema.
- `interrupt` vs `rollback` for steer, within §6's requirements.
- The night theme's exact tokens (within the surface brief's description), the categorical palette's exact values (colour-blind checked).
- Nuxt and library versions (pinned, at least two weeks old).
- Point caps for scatter density mode and histogram binning.
