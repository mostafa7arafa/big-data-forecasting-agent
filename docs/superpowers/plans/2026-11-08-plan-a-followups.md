# Plan A — rulings and follow-ups

Plan A (data and forecasting core) merged from branch `feat/plan-a-data-core`. This file keeps the decisions made during the build and the items deferred to later plans. It is the input for writing Plans B–D. It was copied from the build ledger before the ledger was deleted.

## Plan B task zero (from the final whole-branch review)

- Zero-fill additive metrics for days with no events (today they are interpolated and reported as "imputed")
- `Memory.as_of(cutoff)`: drop quirks/facts/recap content after the cutoff before any agent-level replay
- One `AnalystError` type with an LLM-actionable hint (today: KeyError / ValueError / QueryError / ValidationError)
- `cutoff` and `data_end` in forecast card meta; honour or reject `levels` other than 80/95
- Forecast caching by input handle + params, and a time budget (daily by-borough ≈ 90 s)
- Atomic `ResultStore.save` and engine locking/cursors for concurrent tool calls
- Coerce string values for integer dimensions; `describe` says "showing 50 of N"; note partial periods caused by the requested time_range
- Id-share policy for headline numbers (service averages all ids; replay drops ids < 1% share)
- Wire `recommended_models` into candidate selection before the README says replay settles the lineup
- `make replay` still prints ARIMA convergence warnings outside pytest; capture them as one adjustment

## Rulings

- work on branch feat/plan-a-data-core in the main checkout, not a separate worktree — 2.6 GB data/raw (gitignored) lives here and taxi integration tests need it — cost if wrong: none beyond a less isolated tree; master is untouched.
- implementers = external delegates per owner (Cursor composer-2.5 / grok-4.7-medium at normal speed; Pi openai-codex/gpt-6.1-sol), run visibly in Herdr panes; task reviewers = Claude subagents (sonnet), final review on opus. Delegates don't commit; controller commits after review — cost if wrong: commit granularity is per task instead of per step.
- (T2 ↔ T5) Conflict: schema doesn't exist at T2. Ruling: T2 fixture writes no memory files; T5 adds synthetic memory to the fixture — cost if wrong: none
- (T1 ↔ T10) Ruling: T1 adds `replay` target that runs scripts/replay_models.py; it fails until T10 — cost: none
- (T5 ↔ T10) Ruling: T10 sets the SeriesSpec item's own source=profile, status=proposed when it writes recommended_models (item-level status) — cost if wrong: an approved series reads as proposed until owner approves in Plan C
- Task 11 may extend the synthetic fixture so weekend days have fewer rows (e.g. every other hour) to give daily counts a weekly cycle for test_weekly_acf_detected; any earlier tests that hard-code row counts are updated in the same task — implementer flagged daily counts are flat — cost if wrong: one fixture tweak touches Task 3/7 count assertions
- DatasetConfig gains `dir: Path` (datasets/<name>); `memory_dir = dir / "memory"`; `root` stays repo_root for resolving files — plan text said root/"memory" but the spec layout puts memory under the dataset folder — cost if wrong: one property
- engine validates the query AST (json_serialize_sql) before execution — every BASE_TABLE must be an engine view or a CTE defined in the query; TABLE_FUNCTION allowlist = {range, generate_series, unnest}; anything else → QueryError("query may only read the dataset views: …") — spec §6 requires the cutoff be enforced in the engine so the agent cannot see past it — cost if wrong: some legit table functions (e.g. read_csv on a temp file) are refused; Plan B's sandbox path is separate
- BASE_TABLE and CTE names must be plain identifiers (^[A-Za-z_][A-Za-z0-9_]*$) AND CTE names are scoped to the query node that defines them — defense in depth; a file path can never pass the identifier rule — cost if wrong: quoted identifiers with spaces/dots in dataset view names are disallowed (our views are plain)
- share=True raises ValueError for metrics whose expr is not a single sum(...)/count(...) aggregate (shares of ratios/averages are meaningless) — cost if wrong: a user asking share of avg_fare gets an explanatory error
- neq / not_in keep rows whose dimension is NULL (`IS DISTINCT FROM` / `(x NOT IN (...) OR x IS NULL)`) — "not Manhattan" should include unknown zones — cost if wrong: unknown rows appear in negative filters
- for previous_period at day/hour grain, align cur and cmp by offset from their own range start (cur.period - cur_start = cmp.period - cmp_start); unmatched trailing periods get NULL compare; result row count == number of periods in the current range — position-matching is the natural reading of "same day of the previous period" — cost if wrong: day-of-week no longer aligned for month comparisons (YoY day grain keeps 364-day weekday alignment)
- hour grain is never partial (coverage.end is the start of the last hour present, so every returned hour bucket is complete) — cost if wrong: a truncated final hour of raw events would not be flagged
- MS series — a month overlapping an impute quirk is scaled pro-rata: y * days_in_month / (days_in_month - gap_days_in_month), with adjustment "scaled 2023-09 for 4 known-gap days (2023-09-21..2023-09-24)" — honest, simple, keeps the month usable — cost if wrong: small bias if gap days weren't typical days
- daily AutoARIMA uses approximation=True (stepwise) to bound runtime; target < 30 s per daily series for the full backtest; candidates unchanged — spec lists AutoARIMA for daily — cost if wrong: slightly worse ARIMA fits; cutoff replay (Task 10) decides lineup anyway
- a non-baseline candidate that fails to fit (in backtest or refit) is skipped with adjustment "<model> skipped (fit failed: <first line>)"; SeasonalNaive failure still raises; never use StatsForecast fallback_model — reviewer recommendation — cost if wrong: a broken model hides behind a note rather than an error
- only quirks with status "approved" are applied — governed memory semantics (spec §5.8) — cost if wrong: none
- full backtest scorecard saved as its own handle (produced_by "backtest"); card meta keeps backtest_handle + summary {best_model, mean_relative_mae, mean_mase, mean_coverage_80, mean_coverage_95, baseline_mean_mase} and per-series rows only when ≤ 5 series — cost if wrong: agent must open a second handle for per-series detail on wide forecasts
- SummaryCard.render() caps the meta line at 1,000 chars with " …(meta truncated)" so the 4096 budget always holds; floats in preview/stats are display-rounded (|x|≥1 → 2 decimals, else 4 significant digits); stored parquet keeps full precision — touches Task 4 code — cost if wrong: display-only precision loss
- render() orders meta keys by length of their serialized value (shortest first, ties by key name) before applying the 1,000-char cap — cost if wrong: meta key order in render differs from stored JSON order
- (supersedes Task 2 ledger ruling) Task 11 builds its own tmp dataset with weekday-varying row counts for the weekly-ACF test; the shared fixture stays unchanged — avoids churn in Tasks 3–9 tests — cost if wrong: none
- Task 12 gold SQL is hand-written (no compiler) but encodes the same business definitions as semantic.yaml (valid_trip, payment CASE) — otherwise semantic-vs-gold compares different questions — cost if wrong: gold shares definition errors with the semantic layer (a second-model derivation in Plan D covers that)
- Task 12 grader also normalizes int→float, DATE≡TIMESTAMP midnight, NULL≡NULL; gold parquet files are committed with explicit end bounds — cost if wrong: grader too lenient on type differences
- first_non_null_month stays literal per brief; new field first_material_month (≥1% non-null, falls back to literal) carries the "when did it really start" answer (taxi: cbd fee 2025-01 vs stray 2024-12) — cost if wrong: one extra profile field
- gap detection keeps the implementer's soft-neighbor expansion (<50% days adjacent to a hard <10% day join the gap) — catches partial edge days like 2023-09-21 — cost if wrong: an unusually quiet day next to a real gap gets folded into it
- Task 11: parked — lookup join detection only considers id-named integer key columns and role=id main columns, so string-keyed lookups (e.g. country code) are never detected — Ruling: real but deferred to Plan C (second-dataset calibration); widening it now adds ~one 162M-row distinct scan per lookup-col × main-col pair and taxi joins on integer ids — cost if wrong: a string-keyed second dataset gets no lookup dependencies until Plan C fixes it
- final fix wave = handle-id validation, origin-vs-data-end check (raise if whole series short; note per short id), freq-vs-grain check (run_sql handles: reject >20% imputed), one selection criterion (relative MAE; serve SeasonalNaive when nothing beats it), descending order_by (q004 becomes a real check), series status no longer demoted by replay (new recommended_models_status field — amends the T5↔T10 item-status ruling), warning filter, hermetic replay test, agnostic-guard path fix, elapsed_s display rounding — cost if wrong: small API changes Plan B builds on
- deferred to Plan B task zero: zero-fill additive metrics for empty days, Memory.as_of(cutoff) before any agent-level replay, unified AnalystError with hints, cutoff/data_end in forecast meta, forecast caching + time budget, atomic save + engine locking for concurrent tools, int-coerce string dimension values, describe "showing 50 of N", id-share policy for headline numbers, wire recommended_models into candidate selection — cost if wrong: Plan B grows by these items
- hour-grain-never-partial ruling keeps its behavior; its premise is corrected (coverage.end is max timestamp, not last-hour start; true only for feeds ending on hour boundaries) — cost if wrong: a mid-hour feed shows a partial last hour unflagged
- deferred to Plan C: duplicated quoting/type-sniffing helpers, extra="forbid" + cross-ref validation before LLM-drafted calibration, Task 1–3/5/6/11 minors as triaged; to Plan D: grader exact-for-integers default, gold freshness test, CRPS from more quantiles

## Deferred minors

- Task 1: minor (deferred): Settings dir fields typed Path|None; test_settings_defaults doesn't clear ANALYST_* env; dataset-agnostic guard uses cwd-relative path (can pass vacuously if run elsewhere); .gitignore has .claude/settings.local.json (controller-added)
- Task 2: minor (deferred): oversized fixture test; dead datasets_dir fallback; data_dirs unsorted / absolute-glob risk
- Task 3: minor (deferred): set python_enable_replacements=false; validator CTE scope wider than DuckDB (harmless); validate dataset/table/lookup names are plain identifiers in load_dataset; current_setting reveals data paths
- Task 4: minor (deferred): _is_numeric_type_str substring "int" matches list<int64>; preview separator detection uses "|---" in line; save() not atomic across parquet+card; wide tables drop stats
- Task 5: minor (deferred): models lack extra="forbid" (YAML typos silently dropped — matters for LLM-drafted calibration in Plan C); no cross-reference validation (metric.filters/dimension.join/series refs); data_issue facts lumped into one quirk; only observed day stored for weekend holidays; recap wording on 176 rows
- Task 6: minor (deferred), 8 items from the task review:
  1. neq/not_in and NULLs: settled by ruling (NULLs kept)
  2. share of non-additive metrics: settled by ruling (error)
  3. dimension values present only in the comparison period are dropped from compare_to results. Plan B: state it as an answer caveat
  4. time column is unqualified in compiled SQL (ambiguous if a joined table has the same column). Plan C
  5. alias capture inside scalar subqueries in metric exprs. Plan C
  6. filters can't declare the joins they need. Plan C
  7. resolve.py imports private compiler helpers. Plan B
  8. YoY without a grain is undocumented. Plan B
- Task 7: minor (deferred): limit can drop partial period from result; compare_to windows before data start not flagged; describe dimension output no width guard; _period_threshold raises for hour (guarded)
- Task 8: minor (deferred): multi-id series with missing trailing periods forecast from own last date while origin is global; scaled CRPS from 4 quantiles; adjustment wording differs daily vs monthly; ARIMA convergence warnings in tests
- Task 9: minor (deferred): 523 AutoARIMA convergence warnings in test output; repeated forecast call recomputes backtest
- Task 10: minor (deferred): monthly report reuses daily baseline wording ("last week"); series.yaml reformatted by yaml writer
- Task 11: minor (deferred): ACF on raw counts flags every lag on trending series (consider differencing in Plan C); anomaly baseline is a global day-of-week median (rolling would survive level shifts); first-month query scans once per column; arbitrary_types_allowed unused
- Task 11: minor (deferred): _parse_bound via float() loses precision for BIGINT/HUGEINT > 2^53; monthly no-gaps test weak (short series)
- Task 11: minor (deferred): _is_numeric/_is_integer substring-match type names; one join-overlap query per key × main-column pair
- Task 12: minor (deferred): grader default rel_tol=1e-6 loosens integer counts unless caller maps tolerance null→exact (only the integration test does); no check that frozen parquet still matches current gold_sql; 6-sig-digit sort key can misalign near-equal rows; build_gold leaves orphan parquets
- PR #1 CodeRabbit round: minor (deferred): unit tests other than test_config can still pick up a developer .env's ANALYST_ARTIFACTS_DIR (add an autouse conftest fixture that disables env_file); service/replay import private _is_additive/_metric_additive
