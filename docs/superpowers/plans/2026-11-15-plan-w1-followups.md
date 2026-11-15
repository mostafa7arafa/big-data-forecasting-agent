# Plan W1 rulings and deferred items

_Plan W1 (dashboard backend: chart catalogue and service, map geometry, run trace and model catalogue, steer and stop cleanup) was built on branch `feat/plan-w1-dashboard-backend`. This file keeps every decision made while executing it, the durability check, the deferred minors, and the handoff to W2. The source is the SDD ledger, which is deleted after merge._

## Rulings

- Ruling R1: T1's `validate_spec` only checks that a forecast_fan's `forecast_handle` exists with `produced_by == "forecast"` (and that `series_id`, when given, is among the handle's unique ids); T4's payload builder raises `InvalidRequestError("forecast predates chart support; re-run forecast")` when the time/value column meta is missing. Cost if wrong: an old cached forecast charted between T2 and T3 errors; re-running it fixes it.
- Ruling R2: T4 implements every kind except `map` (it raised "map needs dataset geometry" until T5); T5 implements the map payload and its golden fixture. Cost if wrong: none beyond sequencing.
- Ruling R3: in compact, repair first, then the compaction check runs on the repaired list; at most one REMOVE_ALL in the node's update, since two in one step would be ambiguous. Cost if wrong: a review finding in T8.
- Ruling R4: T2 may edit `evals/target.py` (Step 3 names it); the plan text is authoritative over its Files list. Cost if wrong: none.
- Ruling R5: the thread id comes from `langgraph.config.get_config()["configurable"]["thread_id"]` inside try/except; no thread id means no registration (eval and scripts run without a thread). Cost if wrong: cancellation does not interrupt queries in that context, which only matters for UI runs, and those always have a thread.
- Ruling R6: `test_real_forecast_passes_forecast_shape` (taxi) broke from T2's `chart_kind` check. The test is about `forecast_shape` only, so it grades only `forecast_shape` entries; the agent eval still grades `chart_kind`. Cost if wrong: none.
- Ruling R7: `fastapi` added as a main dependency (langgraph-api 0.18 does not ship it) because custom routes need it. Cost if wrong: one extra pinned dependency.
- Ruling R8: CORS narrowed from the langgraph default `*` to `http://localhost:3000` and `https://smith.langchain.com` via `langgraph.json` `http.cors`; the `CORS_ALLOW_ORIGINS` env var is now ignored. Cost if wrong: a Studio user on another host (for example EU Studio) gets CORS errors until that origin is added.
- Ruling R9: `reasoning_summary: true` for `openai:gpt-6-luna` (Responses API), because the owner wants reasoning visible. Cost if wrong: the eval regresses and the flag flips back. Result: it did not regress (41/55, see `docs/results/agent-baseline.md`, W1 section).
- Ruling R10: an interrupted run's inner-agent steps never reach the thread (the analyst node commits only on return), so a saved, steered thread keeps the question plus an interrupted trace record. The partial output is visible in W2 only from the live stream. Making the inner loop checkpoint progressively is out of W1 scope. Cost if wrong: reloading a steered thread loses the partial steps (owner to be told; candidate follow-up).
- Ruling R11: keep the widened interruption detection: any prior turn that ended without an answer gets `interrupted=True`, since a crash and a cancel look the same afterwards. Cost if wrong: an errored run is labelled "interrupted" rather than "failed" in the trace.

## Durability check

Task 8: a SIGINT restart of the dev server, then the thread and its trace were read back successfully. SIGKILL was not tested.

## Task 9 live check

On `make dev`, the question "How many trips should we expect next month by borough, and how sure are you?" ran to completion on graph `analyst`. The answer listed one chart id; `GET /charts/nyc_taxi/full/<id>?width=800` returned a `small_multiples` chart with inner `forecast_fan`. The thread state held a `trace` record with per-step tokens, cost, reasoning summaries and timings (176 s total for that run).

## Deferred minors

**Task 1:** the inner wrap copies fields by hand and validates the inner fan twice (use `model_dump`, one place). `_inner.py` duplicates the outer models' fields (could subclass). `type: ignore` on `trend_x`/`trend_y` in `validate.py`. The `is_numeric`/`is_temporal` type-string helpers are substring matches (`interval`, `list<int64>`). Corrupt chart JSON surfaces as a raw `JSONDecodeError`/`ValidationError`. Two validation tests assert loosely (hint truthy, `or`). The kpi trend has no rejection test; the test file lacks a trailing newline.

**Task 2:** `_facts` has an unused `handles` param. The success-side 4,096 cap is untested. No test for `verify` with `charts=None` and chart ids. `chart_kind` passes if any chart matches.

**Task 3:** fold tests do not pin the `best_model` identity or the refit-fallback re-derivation, nor `fold_handle` None in the service (the reviewer verified the behaviour manually). A single meta cutoff versus per-series fold cutoffs (moot while series align to the data end; add a comment). The baseline fit computes unused intervals; `BacktestReport.cv` is kept in memory.

**Task 4:** null series keys are dropped, not folded. A real category named "other" collides. The small_multiples `y_range` ignores last_year/baseline/replay and stacked totals. Fan facets repeat scorecard and history with no overall size cap. An empty prebinned histogram raises `IndexError`. Decimal labels. Weak `or` asserts in the bar test. No leap-day or NaN-history fan tests. Fixtures embed handle ids (stable only while `_make_handle` is unchanged). All-null x into m4 on empty arrays is untested.

**Task 5:** the geometry file is re-read per payload, and a missing or malformed file raises raw errors. `"properties": null` raises `AttributeError`. Duplicate zone names share a join key (document it, or key on location_id). Ring winding is clockwise (see For W2). `scripts/` is not linted; `_dp`/`_ring` are untested. (The containment duplication was resolved in Task 6.)

**Task 6:** reveal should also clear `limit` and `compare_to`. Reveal is uncached (`run_sql` re-runs the full history and holds a gate slot). Up to four uncut engines times an 8 GB limit. A reveal over 1M rows gives an unhelpful 500. `/meta` retries a failed coverage query on every call. A tz-aware `run_sql` time column gives an all-null reveal.

**Task 7:** `trace_hidden` persists in the checkpoint (including reasoning) until the next turn. Eval usage disagrees with trace totals (switch the eval to `state["trace"][-1]`?). The worker `TraceMiddleware` is not truly outermost (deepagents built-ins and summarization are untraced; fix the docstring). `_trace_inputs` is duplicated. The trace list is unbounded per thread. `run_id` is unverified on a real server. `has_prices` ignores `cached_input`. Plan and compaction steps are named "plan"/"compaction" without the model name (the run-level model is in `trace.model`); `plan_cut` is computed twice.

**Task 8 (final review to triage):** `abort()` during sandbox cold start is lost (the cell runs to its 30 s timeout); a one-line fix is `proc.kill()` on the local docker client in `abort()`. A late `get_sandbox` from the old run can create an untracked container. Queries waiting for a gate slot are not cancellable. `cancel_thread` lock contention under blockbuster (dev only). Engine register/timer start sit outside the try. `_failed` runs before `_stop_thread_work`. Repair assumes tool results follow their AI message.

**Task 9 (eval):** the latency median rose to 33.9 s (B2: 21.2 s) with reasoning summaries on; one run cannot separate that from noise. The 20 Not-verified answers include two "untraced number 2026" cases (a year the model wrote as a number).

## For W2

- Steer is a run with `multitask_strategy: "interrupt"`, queue is `"enqueue"`, stop is a cancel of the run. Interrupted threads are repaired on the next turn and carry an interrupted trace record (R10, R11).
- The chart-payload fixtures in `tests/fixtures/charts` (11 files, one per chart kind) are the W2 contract; build the front end against them.
- GeoJSON ring winding is clockwise (d3-geo friendly, not RFC 7946's counter-clockwise exterior rings). Tools that assume RFC 7946 winding will fill the complement.
- CORS allows `http://localhost:3000` and `https://smith.langchain.com` only (R8); add the dashboard's origin to `langgraph.json` `http.cors` when it is deployed elsewhere.
- `make dev` serves `/meta`, `/charts`, `/geo`, `/feedback` next to the LangGraph routes. The thread state key `trace` holds the per-run trace records.
