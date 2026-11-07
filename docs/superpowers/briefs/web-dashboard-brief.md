# Brief: Web Dashboard (separate session)

Read first: `docs/superpowers/specs/2026-11-07-taxi-analytics-agent-design.md`, especially §8, §5.2 (answer structure and verification labels), §5.8 (review gate) and §5.9 (feedback and memory). Start this session only after Plans A and B have landed, because the dashboard consumes the agent's real API.

## What it is

The product face of the analyst. A senior AI engineer opening it should feel "this is a real analytics product", in the spirit of Omni, not a chat demo. It is a separate client. The agent is an isolated Python service (LangGraph Agent Server), and the dashboard talks to it over HTTP + SSE.

## Decided

- **Rust + Leptos, compiled to WebAssembly** (owner). Its own Cargo workspace under `web/`.
- **A small typed Rust client** for the LangGraph Agent Server API: threads, runs and streaming events. Don't use the archived `agent-chat-ui`.
- **Large-data charts** follow the spec's three layers:
  1. downsampled payloads (M4/LTTB) from the chart-data service;
  2. level of detail re-queried on zoom;
  3. WebGL rendering from typed arrays, with off-screen charts virtualized or frozen to images.
- **Performance budgets** are tests:
  - 1M-point chart renders in < 200 ms;
  - pan and zoom at ≥ 50 fps;
  - a chat with 30 charts scrolls without dropped frames.
- **Design** uses the "impeccable" design skills. Install them at the start of the session.
- **An original brand character** that is the AI's persona, animated in the UI and reused in the README and the demo video. It must not resemble Omni's Blobby.

## Screens and parts

1. **Chat.** Streaming answers, a visible plan or task list as the agent works, and a character state (thinking, querying, done, unsure).
2. **Answer card:**
   - the narrative;
   - a verification chip (Verified / Verified · off-model / Not verified, with the reason on hover);
   - chart cards;
   - "Query details" (SQL, code, result handles, elapsed time);
   - the data cutoff;
   - for forecasts, a backtest scorecard vs the seasonal baseline;
   - 👍/👎 with an optional correction.
3. **Forecast chart.** History, forecast line, 80/95 bands, a last-year overlay and a marked origin. Quirk days (gaps, anomalies) are annotated.
4. **Review queue.** Calibration and feedback proposals shown as diffs, with accept, edit or reject per item.
5. **Memory views.** Dataset memory (recap, metrics, quirks, verified queries, lessons) and user memory (view, edit, delete).
6. **Dataset switcher and calibration run view** (progress, profile summary).

## Open for that session

- The character's name, look and motion.
- Visual identity: palette, type, tone.
- Renderer: Rust-native WebGL vs a JS WebGL library bridged through `wasm-bindgen`, decided by a spike against the budgets. Rust-native is preferred.
- CSR vs SSR in Leptos.
- The demo-video script (~2 min): the hero question end to end, a context-size meter, and the eval results.

## Out of scope

Auth, multi-user and hosting (Project 3); subscription sign-in (Project 2).
