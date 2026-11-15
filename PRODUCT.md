# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Nuxt + Vue 3 with TypeScript, client-side rendered (`ssr: false`), its own pnpm workspace under `web/`. Talks to the Python LangGraph Agent Server over HTTP + SSE through `@langchain/vue` `useStream`; chart-data endpoints are custom FastAPI routes inside that server. Owner decision 2026-11-15 (replaced Rust + Leptos).

## Users

- **Primary: a senior AI engineer evaluating the owner's work** (hiring manager, peer, LinkedIn/GitHub visitor). They open the dashboard or its demo video to judge whether the owner knows what matters when building agents. They are fluent in LLM internals: tokens, context windows, reasoning, latency, tool calls, evals.
- **Operator: the owner**, an AI engineer, who runs it locally to ask questions, demo it, and (from Plan C) review what the agent learned.

## Product Purpose

An AI data analyst over ~162M NYC taxi trips (DuckDB over Parquet). It answers plain-language questions, forecasts with real statistical models, and explains results without inventing causes. The dashboard is its product face. Success: a senior AI engineer leaves thinking "this person knows what matters to me" — the agent's performance and insight are visible, measured and honest, not a chat demo that draws charts.

## Positioning

The agent shows its work and its costs. Every answer carries a verification label (Verified / Verified · off-model / Not verified) traced to result handles; forecasts carry a seasonal-naive baseline and a backtest; the engine enforces a data cutoff so the agent can be replayed into the past and scored against what actually happened. The machinery — tokens, cached tokens, cost, context fill, reasoning stream, tool-call timeline, latency, background agents — is on display, not hidden.

## Operating Context

- Chat is the main workflow: ask, watch the agent plan and stream, inspect the answer.
- **Background work:** long forecasts run as background tasks; the user keeps chatting, the task shows progress in a tray, and its result lands in the conversation as its own answer.
- **Steering:** the user can interrupt mid-answer (steer: restart with the new message; queue: answer after; stop: cancel).
- **Trace per answer:** tokens in/out/cached, cost estimate, context-window meter, reasoning summary streamed separately from the answer, tool-call timeline with durations, time to first token, total time, model name, why the verification label was given.
- Used locally on a desktop browser by the owner; also screen-recorded for a ~2 minute demo video and README.

## Capabilities and Constraints

- Agent tools: `query_metrics` (governed semantic layer), `run_sql` (off-model), `describe`, `forecast`, `make_chart`, `run_python` (Docker sandbox), `read_knowledge`.
- **Charts are a core capability, and the agent picks the shape.** Planned set: line, area, stacked area; forecast fan (history solid in one colour, forecast continuing in a second colour with 80/95 bands, a marked origin with its date, last-year overlay, backtest scorecard vs baseline, optional replay of a past forecast against actuals); grouped/stacked/ranked bars; hour×weekday heatmap; zone map (choropleth, geometry lives in the dataset folder); histogram/box; scatter; KPI tiles and sparklines; small multiples. Chart styling is rule-driven so a freely chosen shape can't come out ugly. Today `make_chart` supports only line and bar.
- **Large data:** the server downsamples to about pixel width (M4/LTTB) and re-queries on zoom; budgets are tests — a 1M-point chart renders in < 200 ms, pan/zoom ≥ 50 fps, a chat with 30 charts scrolls without dropped frames.
- Code under `src/analyst/` is dataset-agnostic; the dashboard must not hard-code taxi specifics outside dataset-provided content.
- Later (Plan C): review queue (proposals as diffs, accept/edit/reject per item), dataset and user memory views, dataset switcher and calibration run view. 👍/👎 buttons appear now and get wired in Plan C.
- Out of scope: auth, multi-user, hosting; per-user character customization (deferred by owner).

## Brand Commitments

- **The character is a swarm of a few hundred data points** (owner-approved 2026-11-15). Idle: a loose friendly creature shape with two eyes, gently breathing. Working: the dots rearrange into the shape of the task — swirl (thinking), funnel stream (querying), a rising line fanning into futures (forecasting), bars (comparing), a grid (heatmap); settles with "smiling" eyes when done; scatters, flickers and shrugs when unsure or declining to guess a cause. Its body literally becomes the chart. Same animation in the UI, README and demo video. Respects reduced motion. Name to be chosen.
- It must not resemble Omni's Blobby (a blob/jelly mascot).
- Progress/status lines get rewritten in the character's voice once it is designed.

## Evidence on Hand

- Real data: 45 monthly NYC TLC yellow-taxi files (2023-01..2026-09) in `data/raw/`.
- Real results: `docs/results/agent-baseline.md` — 73% pass on a 55-question agent eval (LangSmith experiment `openai-gpt-6-luna-81b55b5-aae16c18`), python 5/5 Verified · off-model; model-level cutoff replay results in `docs/results/`.
- No users, testimonials or customers exist; never invent them.

## Product Principles

1. Show the work: every number traces to a result, every label has a reason a viewer can open.
2. Honest uncertainty: forecasts always show their range and their track record; causes are never invented.
3. The machinery is a feature: tokens, time, context and parallel agents are first-class UI, not a debug panel.
4. Charts tell the whole story: context (history, last year, origin) around the result, never the result alone.
5. Built for the expert eye: precision and restraint over decoration; delight lives in the character and in moments.
