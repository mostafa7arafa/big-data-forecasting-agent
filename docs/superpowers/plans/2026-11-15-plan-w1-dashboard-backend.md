# Plan W1: Dashboard Backend, Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Everything the dashboard needs from Python, testable without a UI: a typed chart catalogue the agent picks from, a chart-data service with downsampling and level of detail, map geometry, a per-run trace streamed live and saved with the thread, steer/stop that leaves threads consistent, the cutoff "reveal" endpoint, and a feedback endpoint.

**Architecture:**
- **Charts:** `make_chart` takes one `spec` argument, a Pydantic discriminated union on `kind` (11 kinds). It validates the spec against the handles' cards, stores it content-addressed in a `ChartStore` inside the run's (dataset, cutoff) segment, and returns a chart id. `Answer.charts` becomes a list of chart ids.
- **Chart service:** `analyst.charts.payload.build_payload()` loads the referenced handles and returns a versioned JSON payload per kind (M4-downsampled to the requested width, optional `from`/`to` range). FastAPI routes in `analyst.http.app` (registered as `http.app` in `langgraph.json`) serve it next to the Agent Server.
- **Trace:** a `TraceMiddleware` on the analyst and worker agents stamps timing on every AI and tool message and emits live `trace` stream events; the outer `answer` node folds the turn's messages into a `RunTrace` saved under a new `trace` state key. Cost and context fill come from a model catalogue (`config/models.yaml`).
- **Steer/stop:** the outer `compact` node repairs dangling tool calls left by a cancelled run (REMOVE_ALL + rebuilt list, the compaction pattern) and records the interruption; tool-level cancellation interrupts in-flight DuckDB queries and closes the thread's sandbox.

**Tech Stack:** Python 3.12 via `uv`; LangChain 1.5 / LangGraph 1.2 / langgraph-api 0.18; FastAPI (already a langgraph-api dependency); `tsdownsample==0.1.5.1` (new); `pyshp==3.1.8` + `pyproj` (new, dev group only, for the one-off geometry script); pytest.

**Spec:** `docs/superpowers/specs/2026-11-15-dashboard-design.md` (§4 charts, §5 trace, §6 duplex, §6b pickers and reveal, §9 W1 list). Parent: `docs/superpowers/specs/2026-11-07-taxi-analytics-agent-design.md` §8. Product record: `PRODUCT.md`.

## Global Constraints

- **`src/analyst/` stays dataset-agnostic.** The existing test fails on `taxi`, `borough`, `tpep_`, `PULocationID`, `DOLocationID` under `src/analyst/`; that includes `analyst.charts` and `analyst.http`. Geometry names, keys and files come from `dataset.yaml`.
- **The chart service reads result handles only**, from the ResultStore segment `segment_root(dataset, cutoff)`. It never queries raw data. The one exception is the reveal endpoint (Task 6), which runs a recorded query on an uncut `Engine` and **writes nothing anywhere**.
- **Tools return cards and handles, never raw rows**; tool content ≤ 4,096 chars (`run_tool` cap). Chart data travels only over HTTP.
- **Every HTTP path parameter is validated** before touching the filesystem: dataset must be a registered dataset name; segment is `full` or an ISO date; chart id matches `^c_[0-9a-f]{10}$`; geometry name matches `^[a-z0-9_]{1,40}$` and must be declared in `dataset.yaml`. Anything else → 404, never a path join.
- **No invented numbers.** Model prices and context windows in `config/models.yaml` are real values or `null`; `null` renders as "n/a" downstream. The trace's cost uses a provider-reported `usage_metadata["total_cost"]` first (OpenRouter, existing `OpenRouterCost`), then the price table, else `None`.
- **Blocking work stays off the event loop** (`langgraph dev` runs blockbuster): file reads, DuckDB and payload building in async routes and nodes go through `asyncio.to_thread`.
- **Payload schema version is `1`**, carried as `"version": 1` in every chart payload.
- **Width parameter:** default 1200, clamped to [100, 4000]. M4 keeps at most `4 * width` points per series.
- **Scatter switches to a density grid above 50,000 points** (grid 200 × 200). **Histogram default 40 bins.** **Line/area: at most 6 series**, the rest summed into `other` with a note. **Small multiples: at most 12 facets.** **Bar `top_n`: 1–50.**
- **Delegates never commit.** The controller gates, reviews and commits.

## Review Focus

1. **A chart spec that names a handle from another segment or a nonexistent column.** `make_chart` must reject it with a hint listing the handle's columns, and the HTTP route must 404 rather than read another segment. Owned by Task 1 (`test_validate_rejects_unknown_column_with_hint`) and Task 6 (`test_chart_route_rejects_cross_segment_and_bad_ids`).
2. **A reveal that leaks into the agent.** The reveal output must never appear in thread state, messages or any ResultStore segment. Owned by Task 6 (`test_reveal_writes_nothing_and_needs_past_cutoff`).
3. **A steered/cancelled run leaving an AIMessage with tool calls and no ToolMessages**, so the next model call fails at the provider. Owned by Task 8 (`test_compact_repairs_dangling_tool_calls`).
4. **A multi-series forecast drawn as one line through all series.** `forecast_fan` must draw exactly one series (`series_id`, default first headline id) and `small_multiples` of fans one per id. Owned by Task 4 (`test_forecast_fan_payload_single_series`).
5. **A model without prices or a provider that reports no usage.** The trace must still build, with `cost_usd=None` and token fields `None`, never 0. Owned by Task 7 (`test_trace_without_usage_or_prices`).

---

## File map

| Path | Responsibility | Task |
|---|---|---|
| `src/analyst/charts/spec.py` | `ChartSpec` union, per-kind models, `handles()` | 1 |
| `src/analyst/charts/store.py` | `ChartStore`, `StoredChart`, content-addressed ids | 1 |
| `src/analyst/charts/validate.py` | `validate_spec(spec, store, geometries)` | 1 |
| `src/analyst/tools/make_chart.py` | rewritten tool | 2 |
| `src/analyst/state.py` | `Answer.charts: list[str]`; `trace` state key | 2, 7 |
| `src/analyst/verify/claims.py`, `src/analyst/prompts.py` | chart ids in verify; chart-choice policy | 2 |
| `evals/graders/agent.py`, `evals/datasets/agent_single_turn.jsonl` | `chart_kind(...)` check | 2 |
| `src/analyst/forecasting/core.py`, `src/analyst/data/service.py` | baseline column, last backtest fold handle, column names in meta | 3 |
| `src/analyst/charts/downsample.py`, `src/analyst/charts/payload.py` | M4 wrapper; per-kind payload builders | 4 |
| `scripts/build_geometry.py`, `datasets/nyc_taxi/geo/taxi_zones.geojson`, `datasets/nyc_taxi/dataset.yaml`, `src/analyst/data/dataset.py` | geometry asset and declaration | 5 |
| `src/analyst/http/app.py`, `src/analyst/charts/reveal.py`, `src/analyst/feedback.py`, `langgraph.json` | routes: meta, chart, geo, reveal, feedback | 6 |
| `config/models.yaml`, `src/analyst/providers/catalog.py`, `src/analyst/providers/__init__.py` | model catalogue; reasoning summaries opt-in | 7 |
| `src/analyst/trace/model.py`, `src/analyst/trace/middleware.py`, `src/analyst/trace/build.py`, `src/analyst/graph.py`, `src/analyst/worker.py` | trace schema, live events, fold into state | 7 |
| `src/analyst/graph.py`, `src/analyst/data/engine.py`, `src/analyst/cancel.py` | dangling-call repair, interruption record, cancellation cleanup | 8 |
| `docs/results/agent-baseline.md`, `README.md`, `CLAUDE.md`, `docs/superpowers/plans/2026-11-15-plan-w1-followups.md` | eval gate and docs | 9 |

---

### Task 1: Chart spec union, chart store, validation

**Files:**
- Create: `src/analyst/charts/__init__.py`, `src/analyst/charts/spec.py`, `src/analyst/charts/store.py`, `src/analyst/charts/validate.py`
- Test: `tests/unit/test_chart_spec.py`

**Interfaces:**
- Produces:
  - `ChartSpec = Annotated[LineChart | AreaChart | ForecastFanChart | BarChart | HeatmapChart | MapChart | HistogramChart | BoxChart | ScatterChart | KpiChart | SmallMultiplesChart, Field(discriminator="kind")]`. Every model has `model_config = ConfigDict(extra="forbid")`, `kind: Literal[...]`, `title: str = Field(min_length=1, max_length=120)`.
  - Fields per kind (all column fields are `str` column names):
    - `LineChart`: `handle, x, y, series: str | None = None`
    - `AreaChart`: `handle, x, y, series: str | None = None, stacked: bool = False`
    - `ForecastFanChart`: `forecast_handle, series_id: str | None = None, last_year: bool = True, baseline: bool = False, replay: bool = False` (history, value and time columns come from the forecast card's meta, Task 3)
    - `BarChart`: `handle, category, value, series: str | None = None, orientation: Literal["vertical","horizontal"] = "vertical", mode: Literal["grouped","stacked"] = "grouped", top_n: int | None = Field(None, ge=1, le=50)`
    - `HeatmapChart`: `handle, x, y, value`
    - `MapChart`: `handle, key, value, geometry: str`
    - `HistogramChart`: `handle, value: str | None = None, bin: str | None = None, count: str | None = None` (exactly one of `value` or the pair `bin`+`count`; model validator)
    - `BoxChart`: `handle, group, value`
    - `ScatterChart`: `handle, x, y, color: str | None = None`
    - `KpiChart`: `handle, value, label: str = Field(max_length=60), comparison: str | None = None, trend_handle: str | None = None, trend_x: str | None = None, trend_y: str | None = None` (trend fields all or none)
    - `SmallMultiplesChart`: `facet: str, inner: InnerSpec` where `InnerSpec` is the same discriminated union over `line`, `bar`, `area`, `forecast_fan` without `title` (models `InnerLine`, `InnerBar`, `InnerArea`, `InnerFan`); for an inner `forecast_fan`, `facet` must be `"unique_id"`.
  - `def chart_handles(spec: ChartSpec) -> list[str]` — every handle id the spec references, in field order, no duplicates (for `forecast_fan` only the forecast handle; Task 3's meta links the rest).
  - `StoredChart(BaseModel)`: `chart_id: str, dataset: str, cutoff: date | None, spec: ChartSpec, created_at: datetime`.
  - `class ChartStore(root: Path)` with `save(spec: ChartSpec, *, dataset: str, cutoff: date | None) -> StoredChart`, `get(chart_id: str) -> StoredChart` (raises `UnknownNameError`), `exists(chart_id: str) -> bool`. Files at `<root>/charts/<chart_id>.json`, atomic write (tmp + `os.replace`). `chart_id = "c_" + sha256(canonical spec JSON + dataset + cutoff)[:10]`; saving the same spec twice returns the same id.
  - `CHART_ID_RE = re.compile(r"^c_[0-9a-f]{10}$")`; `get`/`exists` validate the id first.
  - `def validate_spec(spec: ChartSpec, store: ResultStore, geometries: Mapping[str, GeometrySpec]) -> None` — raises `InvalidRequestError` with a hint. Column-type rules use the card's column type strings: time axes (`line.x`, `area.x`, `kpi.trend_x`) must be date/timestamp types; value fields (`y`, `value`, `count`, `comparison`, `trend_y`) numeric; `map.geometry` must be in `geometries` (else hint "this dataset has no geometry named …; available: …" or "this dataset has no map geometry"). `GeometrySpec` is defined in Task 5; until then the parameter type is `Mapping[str, Any]` and tests pass `{}`.

- [ ] **Step 1: Write the failing tests**

```python
def test_union_parses_each_kind(): ...          # one minimal valid dict per kind → right class
def test_extra_fields_rejected():                # {"kind":"line",...,"colour":"red"} → ValidationError
def test_histogram_needs_value_or_bin_count(): ...
def test_small_multiples_fan_requires_unique_id_facet(): ...
def test_chart_handles_collects_all_refs():
    assert chart_handles(KpiChart(kind="kpi", title="t", handle="r_aaaaaaaaaa", value="v",
        label="l", trend_handle="r_bbbbbbbbbb", trend_x="d", trend_y="v")) == ["r_aaaaaaaaaa", "r_bbbbbbbbbb"]
def test_store_is_content_addressed(tmp_path):   # same spec → same id; different cutoff → different id
def test_store_rejects_bad_ids(tmp_path):        # "../x", "c_ZZ", "c_" + 11 hex → UnknownNameError, no file access
def test_validate_rejects_unknown_column_with_hint(store_with_handle):
    # hint lists the handle's columns
def test_validate_rejects_non_time_x_for_line(store_with_handle): ...
def test_validate_rejects_unknown_geometry(store_with_handle): ...
```

- [ ] **Step 2: Run** `uv run pytest tests/unit/test_chart_spec.py -v` — expected: FAIL (module not found).
- [ ] **Step 3: Implement** the three modules per the Interfaces block. Reuse `_is_numeric_type_str` from `analyst.data.results` (promote it to a public `is_numeric_type_str`, keeping the old name as an alias) and add `is_temporal_type_str` beside it.
- [ ] **Step 4: Run** the test file — expected: PASS; then `make test` and `make lint` — expected: all green.
- [ ] **Step 5: Hand back for review** (controller commits: `feat(charts): chart spec union, content-addressed chart store, validation`).

---

### Task 2: `make_chart`, chart ids in answers, verify, prompt policy, eval check

**Files:**
- Modify: `src/analyst/tools/make_chart.py`, `src/analyst/state.py` (`ChartSpec` model removed; `Answer.charts: list[str]`), `src/analyst/verify/claims.py:133-136`, `src/analyst/prompts.py`, `evals/graders/agent.py`, `evals/datasets/agent_single_turn.jsonl`, `evals/datasets/agent_single_turn_review.md`
- Test: `tests/unit/test_tools.py`, `tests/unit/test_make_chart.py` (new), `tests/unit/test_verify*.py` (existing chart case), `tests/unit/test_agent_graders.py`

**Interfaces:**
- Consumes: Task 1 `ChartSpec`, `ChartStore`, `validate_spec`, `chart_handles`.
- Produces:
  - `make_chart(spec: ChartSpec, runtime) -> Result` — content: `chart <chart_id> queued: <kind> '<title>'` plus one line per derived fact the model should know (e.g. "forecast_fan of r_… (series Manhattan); last year shown"); artifact `{"chart_id": ..., "kind": ..., "handles": [...]}`. Store root: `service.store`'s segment root (the same `segment_root(dataset, cutoff)`).
  - `DataService.charts: ChartStore` (constructed beside `self.store`, same root).
  - `Answer.charts: list[str]` (chart ids). Verify: each id must exist in the segment's `ChartStore` (else problem `unknown chart <id>`), and every handle from `chart_handles` is resolved as cited.
  - Grader check `chart_kind(kind: str)`: passes when some chart id in `outputs["answer"]["charts"]` resolves (via `outputs["charts"]`, a `{chart_id: stored chart dict}` map the eval target adds) to that kind, or to `small_multiples` whose inner kind is that kind.
- Prompt policy text (add to the system prompt where `make_chart` is described, verbatim):
  > Pick the chart kind from the question: a forecast → `forecast_fan` (several series → `small_multiples` of `forecast_fan` on `unique_id`); a share over time → stacked `area`; a ranking → horizontal `bar` with `top_n`; two time dimensions such as hour × weekday → `heatmap`; a geographic breakdown → `map` when the dataset has geometry; a distribution → `histogram` or `box`; two measures against each other → `scatter`; one headline number → `kpi`. List the chart ids `make_chart` returned in `charts`.

- [ ] **Step 1: Write failing tests**: `test_make_chart_stores_and_returns_id`, `test_make_chart_rejects_bad_column_as_tool_error` (ToolMessage status error, hint present), `test_make_chart_schema_is_openai_valid` (extend the existing `test_openai_tool_schema_arrays_have_items` parametrization; also assert the `spec` parameter's schema has a `oneOf`/`anyOf` with 11 branches), `test_verify_flags_unknown_chart_id`, `test_verify_resolves_chart_handles`, `test_grade_chart_kind_pass_and_fail`, `test_grade_chart_kind_accepts_small_multiples_inner`.
- [ ] **Step 2: Run** them — expected FAIL.
- [ ] **Step 3: Implement.** Add `chart_kind('forecast_fan')` to the `gold_check` list of all 8 rows whose `category` is `forecast` (keep their existing checks); update `_VALIDATORS` and `_CHECKS` together (the module asserts they match). Update `evals/target.py` to include `charts` in `outputs` by reading each answer chart id from the run's `ChartStore`. Note the change in `agent_single_turn_review.md`.
- [ ] **Step 4: Run** `make test`, `make lint` — expected green.
- [ ] **Step 5: Hand back** (`feat(charts): make_chart over the chart catalogue, chart ids in answers, chart_kind eval check`).

---

### Task 3: Forecast handle carries baseline, last backtest fold, and its columns

**Files:**
- Modify: `src/analyst/forecasting/core.py` (`BacktestReport`, `forecast()`), `src/analyst/data/service.py` (`_FORECAST_COLUMNS`, `_compute_forecast`)
- Test: `tests/unit/test_forecasting_core.py`, `tests/unit/test_service_forecast.py` (existing files; add cases)

**Interfaces:**
- Produces:
  - Forecast table columns: `unique_id, ds, yhat, lo_80, hi_80, lo_95, hi_95, baseline` — `baseline` is SeasonalNaive's point forecast for the same `ds` (fit with `fit_predict(..., models=[SeasonalNaive(season)])`, the season the backtest used). When the winner *is* SeasonalNaive, `baseline == yhat`.
  - `BacktestReport.last_fold: pd.DataFrame | None` — the best model's predictions on the most recent backtest window: columns `unique_id, ds, yhat, lo_80, hi_80, lo_95, hi_95, actual` (from the cross-validation frame's `y`). `None` when no window ran.
  - Forecast card meta gains: `"time_col"`, `"value_col"`, `"id_col"` (the arguments), `"season_length"`, and `"fold_handle"` (a handle saved with `produced_by="backtest_fold"`, `index=False`, same `governed` as the input) or `None`.
  - The forecast `source` string gets a `v2` marker so cached pre-W1 forecasts (no `baseline` column) are recomputed, not served.

- [ ] **Step 1: Write failing tests**: `test_forecast_has_baseline_column`, `test_baseline_equals_yhat_when_seasonal_naive_wins`, `test_last_fold_is_latest_window_with_actuals`, `test_forecast_meta_names_columns_and_fold_handle`, `test_old_cached_forecast_not_served` (a pre-seeded index entry under the old source is ignored).
- [ ] **Step 2: Run** — expected FAIL.
- [ ] **Step 3: Implement.** Keep the per-model CV results already collected in `backtest()`; the fold is the rows with the maximum `cutoff` for `report.best_model` after any refit fallback (rename `<model>-lo-80` etc. as `forecast()` already does).
- [ ] **Step 4: Run** `make test`; `make test-taxi` — expected green (Plan A forecast tests unchanged in substance; update fixtures that list forecast columns).
- [ ] **Step 5: Hand back** (`feat(forecast): store seasonal baseline, last backtest fold and column names with each forecast`).

---

### Task 4: Downsampling and per-kind payloads

**Files:**
- Create: `src/analyst/charts/downsample.py`, `src/analyst/charts/payload.py`
- Modify: `pyproject.toml` (add `tsdownsample==0.1.5.1`), `uv.lock`
- Test: `tests/unit/test_chart_payload.py`, fixtures `tests/fixtures/charts/*.json` (one golden payload per kind, shared with W2)

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces:
  - `def m4(x: np.ndarray, y: np.ndarray, width: int) -> np.ndarray` — indices to keep (sorted), using `tsdownsample.M4Downsampler` (NaNs: `NaNM4Downsampler`); returns all indices when `len(x) <= 4 * width`.
  - `def build_payload(stored: StoredChart, store: ResultStore, *, width: int = 1200, x_from: datetime | None = None, x_to: datetime | None = None, geometries: Mapping[str, GeometrySpec] | None = None) -> dict`. Common keys: `version` (1), `chart_id`, `kind`, `title`, `dataset`, `cutoff` (ISO or null), `spec` (the stored spec), `notes` (list of strings), `downsampled` (`{"from": n, "to": m}` or null), `data` (per kind below). Times are ISO-8601 strings; numbers are floats or null.
  - `data` per kind:
    - `line`/`area`: `{"series": [{"name", "x": [...], "y": [...]}]}` — each series M4-downsampled on its own; `other` series per the 6-series rule.
    - `forecast_fan`: `{"series_id", "history": {"x","y"}, "forecast": {"x","yhat","lo_80","hi_80","lo_95","hi_95"}, "origin", "last_year": {"x","y"} | null, "baseline": {"x","y"} | null, "replay": {"x","yhat","lo_80","hi_80","lo_95","hi_95","actual"} | null, "model", "scorecard": {"mase","relative_mae","coverage_80","coverage_95","windows_used","windows","beats_baseline"}}` — `history` from the forecast's `input_handle` filtered to `series_id`; `last_year` = history values at `ds - 1 year` (monthly) or `ds - 364 days` (daily) for each forecast `ds`, null where missing; `baseline` only when `spec.baseline`, `replay` only when `spec.replay` and a fold exists; scorecard from meta `backtest_summary`.
    - `bar`: `{"categories": [...], "series": [{"name","values"}], "other": bool}` — sorted by total value descending when `top_n` is set.
    - `heatmap`: `{"x": [...], "y": [...], "values": [[...]]}` (rows = y).
    - `map`: `{"geometry", "key_property", "keys": [...], "values": [...], "unmatched": [...]}`.
    - `histogram`: `{"edges": [...], "counts": [...]}`.
    - `box`: `{"groups": [{"name","min","q1","median","q3","max","n"}]}`.
    - `scatter`: `{"mode": "points", "x", "y", "color"}` or `{"mode": "density", "x_edges", "y_edges", "counts"}`.
    - `kpi`: `{"value", "label", "comparison", "change_pct", "trend": {"x","y"} | null}`.
    - `small_multiples`: `{"inner_kind", "facets": [{"name", "data": <inner kind data>}], "y_range": [min, max]}`.
  - `x_from`/`x_to` filter time-x kinds (`line`, `area`, `forecast_fan` history) before downsampling; other kinds ignore them.

- [ ] **Step 1: Write failing tests**: `test_m4_keeps_extremes_and_endpoints`, `test_m4_noop_when_small`, `test_line_payload_downsamples_to_width` (1M-row handle, width 1000 → ≤ 4000 points, `downsampled.from == 1_000_000`), `test_line_payload_range_filter`, `test_line_more_than_six_series_folds_other`, `test_forecast_fan_payload_single_series`, `test_forecast_fan_last_year_monthly_and_daily`, `test_forecast_fan_replay_and_baseline_optional`, `test_bar_top_n_and_other`, `test_heatmap_grid_shape`, `test_histogram_bins_and_prebinned`, `test_box_quartiles`, `test_scatter_density_above_cap`, `test_kpi_change_pct`, `test_small_multiples_fan_one_facet_per_id`, `test_payloads_match_golden_fixtures` (writes fixtures with `UPDATE_FIXTURES=1`).
- [ ] **Step 2: Run** — expected FAIL.
- [ ] **Step 3: Implement**; add the dependency with `uv add tsdownsample==0.1.5.1`.
- [ ] **Step 4: Run** `make test`, `make lint` — green.
- [ ] **Step 5: Hand back** (`feat(charts): M4 downsampling and per-kind chart payloads`).

---

### Task 5: Map geometry

**Files:**
- Create: `scripts/build_geometry.py`, `datasets/nyc_taxi/geo/taxi_zones.geojson`
- Modify: `datasets/nyc_taxi/dataset.yaml`, `src/analyst/data/dataset.py`, `src/analyst/charts/payload.py` (map), `pyproject.toml` (dev group: `pyshp==3.1.8`, `pyproj`)
- Test: `tests/unit/test_dataset_geometry.py`, `tests/integration/test_taxi_geometry.py` (marker `taxi`)

**Interfaces:**
- Produces:
  - `dataset.yaml` key:
    ```yaml
    geometries:
      - {name: taxi_zones, file: geo/taxi_zones.geojson, key_property: zone, dimension: pickup_zone_name}
    ```
  - `GeometrySpec(BaseModel)`: `name, file, key_property, dimension: str | None = None`; `DatasetConfig.geometries: list[GeometrySpec] = []` and `DatasetConfig.geometry(name) -> GeometrySpec | None`; file resolved under `DatasetConfig.dir`, must stay inside it.
  - `scripts/build_geometry.py --src <shapefile.zip> --out datasets/nyc_taxi/geo/taxi_zones.geojson`: reads the TLC taxi-zone shapefile (https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip, EPSG:2263), reprojects to EPSG:4326, simplifies to ≤ 1.5 MB with coordinates rounded to 5 decimals, writes properties `{"location_id", "zone", "borough"}`. The zip is downloaded into a fresh directory under the scratch area and read as data; the script never executes anything from it.
- [ ] **Step 1: Write failing tests**: `test_geometry_declared_and_resolved`, `test_geometry_path_cannot_escape_dataset_dir`, `test_map_payload_matches_keys_and_reports_unmatched`; taxi test `test_taxi_zones_geojson_covers_zone_names` (every `zone` value in `taxi_zone_lookup.csv` except `N/A`/`Outside of NYC` matches a feature).
- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement**; run the script once and commit its output (record the source URL and download date in the file's top-level `"metadata"` member).
- [ ] **Step 4: Run** `make test`, `make test-taxi`, `make lint` — green.
- [ ] **Step 5: Hand back** (`feat(geo): taxi-zone geometry declared in dataset.yaml; map payloads`).

---

### Task 6: HTTP routes: meta, charts, geometry, reveal, feedback

**Files:**
- Create: `src/analyst/http/__init__.py`, `src/analyst/http/app.py`, `src/analyst/charts/reveal.py`, `src/analyst/feedback.py`
- Modify: `langgraph.json` (add `"http": {"app": "./src/analyst/http/app.py:app"}`)
- Test: `tests/unit/test_http_app.py` (FastAPI `TestClient`)

**Interfaces:**
- Consumes: Tasks 1, 4, 5; `get_service`, `segment_root`, `load_dataset`, `compile_query`, `Engine`.
- Produces (all JSON; blocking work in `asyncio.to_thread`):
  - `GET /meta` → `{"datasets": [{"name", "description", "coverage": {"start","end"}, "geometries": [...]}], "models": [<catalog entries from Task 7, keys only: spec, label, context_window, has_prices, reasoning_summary>], "default_model"}`. Until Task 7 lands, `models` is `[]`.
  - `GET /charts/{dataset}/{segment}/{chart_id}?width=&from=&to=` → Task 4 payload.
  - `GET /geo/{dataset}/{name}` → the GeoJSON file, `Cache-Control: max-age=86400`.
  - `GET /charts/{dataset}/{segment}/{chart_id}/reveal` → `{"series": [{"series_id", "x": [...], "actual": [...], "yhat": [...], "inside_80": [...], "inside_95": [...]}]}`; 404 when `segment == "full"`, when the chart is not a `forecast_fan` or a `small_multiples` of fans, or when the forecast's input handle was produced by `run_python`.
  - `def reveal_actuals(stored: StoredChart, service: DataService, dataset: DatasetConfig) -> dict` in `charts/reveal.py`: for a `query_metrics` input, rebuild its `MetricQuery` from card meta with `time_range = (first forecast ds, last forecast ds + one grain)` and run `compile_query` on a fresh `Engine(dataset, cutoff=None)`; for a `run_sql` input, re-run its source SQL on that engine and keep rows whose time column is in the forecast's `ds` set. **No `ResultStore.save`, no state write.**
  - `POST /feedback` body `{"thread_id": str, "run_id": str, "rating": "up"|"down", "correction": str | None (≤ 2000 chars)}` → 204; appends one JSON line `{..., "user_id": "local", "at": ISO}` to `<artifacts_dir>/feedback/feedback.jsonl` under a process lock.
  - CORS: allow origin `http://localhost:3000` (the Nuxt dev server) via langgraph.json `http.cors` or FastAPI middleware, whichever langgraph-api 0.18 honours for custom routes.
- [ ] **Step 1: Write failing tests**: `test_meta_lists_dataset_and_coverage`, `test_chart_route_returns_payload`, `test_chart_route_rejects_cross_segment_and_bad_ids` (unknown dataset, `segment="../x"`, `segment="2026-13-01"`, bad chart id, a valid chart id that lives in another segment → 404), `test_width_clamped`, `test_geo_route_serves_declared_geometry_only`, `test_reveal_writes_nothing_and_needs_past_cutoff` (snapshot every file under the artifacts dir before/after; `full` segment → 404), `test_reveal_returns_actuals_after_cutoff` (synthetic dataset with data beyond the cutoff), `test_feedback_appends_line_and_validates`.
- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `make test`, `make lint`; then `make dev` and `curl -s localhost:2024/meta` — expected the meta JSON next to the Agent Server's own routes (stop the server by its PID afterwards).
- [ ] **Step 5: Hand back** (`feat(http): chart, geometry, reveal, meta and feedback routes on the Agent Server`).

---

### Task 7: Model catalogue, reasoning summaries, run trace

**Files:**
- Create: `config/models.yaml`, `src/analyst/providers/catalog.py`, `src/analyst/trace/__init__.py`, `src/analyst/trace/model.py`, `src/analyst/trace/middleware.py`, `src/analyst/trace/build.py`
- Modify: `src/analyst/providers/__init__.py`, `src/analyst/middleware/__init__.py` (add `TraceMiddleware` outermost), `src/analyst/worker.py`, `src/analyst/graph.py` (`answer` node), `src/analyst/state.py` (`trace` key), `src/analyst/http/app.py` (`/meta` models)
- Test: `tests/unit/test_model_catalog.py`, `tests/unit/test_trace.py`

**Interfaces:**
- Produces:
  - `config/models.yaml`: list of `{spec, label, context_window: int | null, prices: {input, cached_input, output} | null (USD per 1M tokens), reasoning_summary: bool}`. Ship entries for the configured default (`openai:gpt-6-luna`) and every model spec in `.env.example` comments, with `null` for any number not copied from the provider's pricing/model page (record the page URL and date in a YAML comment per entry).
  - `ModelInfo` and `def model_info(spec: str) -> ModelInfo | None` in `catalog.py` (cached; unknown spec → `None`).
  - `get_model()`: for provider `openai` with `reasoning_summary: true`, build `ChatOpenAI(..., use_responses_api=True, reasoning={"summary": "auto"})`; everything else unchanged.
  - `trace/model.py`:
    - `StepTrace(BaseModel)`: `id: str, kind: Literal["model","tool","node"], name: str, started_at: datetime, ended_at: datetime | None, status: Literal["ok","error","cancelled"], tokens_in: int | None, tokens_out: int | None, tokens_cached: int | None, tokens_reasoning: int | None, cost_usd: float | None, reasoning: str | None`
    - `RunTrace(BaseModel)`: `run_id: str | None, model: str, started_at, ended_at, ttft_s: float | None, total_s: float | None, steps: list[StepTrace], tokens_in, tokens_out, tokens_cached, cost_usd: float | None, context_tokens: int | None, context_window: int | None, compaction_trigger: int, events: list[dict], verification: Verification | None, interrupted: bool = False`
  - `TraceMiddleware(AgentMiddleware)`: `awrap_model_call`/`wrap_model_call` and `awrap_tool_call`/`wrap_tool_call` record wall-clock start/end into the resulting message's `response_metadata["analyst_trace"] = {"started_at", "ended_at", "first_token_at"}` (tool messages: `ToolMessage.response_metadata`), and emit `{"type": "trace", "event": "step_started"|"step_finished", "step": StepTrace dict}` through `get_stream_writer()` (no-op outside a graph). `first_token_at` comes from an `AsyncCallbackHandler.on_llm_new_token` attached to the call; it stays `None` when the model doesn't stream.
  - `def build_trace(turn: Sequence[BaseMessage], *, model: str, run_id: str | None, verification: Verification | None, events: list[dict], compaction_trigger: int) -> RunTrace` in `trace/build.py`: one step per AI message (tokens from `usage_metadata`, `input_token_details.cache_read` → cached, `output_token_details.reasoning` → reasoning tokens, reasoning text from `content_blocks` of type `reasoning`) and per tool message; totals; cost per step = `usage_metadata["total_cost"]` if present, else price table, else `None` (a run total is `None` if any priced step is `None`); `context_tokens` = the last AI step's `tokens_in`; `events` gets `{"type": "compaction"}` when the turn starts from a compaction summary and `{"type": "background_started", "task_id"}` for each `start_async_task` tool result, in addition to the events passed in.
  - State: `trace: NotRequired[Annotated[list[dict], operator.add]]` — the `answer` node appends `build_trace(...).model_dump(mode="json")` for the turn; the worker graph does the same in its final state for its own runs.
- [ ] **Step 1: Write failing tests**: `test_catalog_loads_and_unknown_is_none`, `test_openai_reasoning_summary_opt_in`, `test_trace_middleware_stamps_timing_and_emits_events` (fake model + a fake tool, collect `stream_mode="custom"`), `test_build_trace_tokens_cost_and_context`, `test_trace_without_usage_or_prices` (all token fields `None`, `cost_usd is None`), `test_trace_prefers_reported_total_cost`, `test_reasoning_text_from_content_blocks`, `test_answer_node_appends_trace_per_turn` (two turns → two records), `test_meta_lists_models`.
- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `make test`, `make lint` — green.
- [ ] **Step 5: Hand back** (`feat(trace): model catalogue, reasoning summaries, per-run trace streamed and saved`).

---

### Task 8: Steer/stop consistency, cancellation cleanup, storage durability

**Files:**
- Create: `src/analyst/cancel.py`
- Modify: `src/analyst/graph.py` (`compact` node), `src/analyst/data/engine.py` (register in-flight cursors), `src/analyst/trace/middleware.py` (cancellation hook), `src/analyst/tools/run_python.py`
- Test: `tests/unit/test_steer.py`, `tests/unit/test_engine.py` (add case)

**Interfaces:**
- Produces:
  - In `compact`: if the thread's messages contain an `AIMessage` whose tool calls have no matching `ToolMessage`, rebuild the list with a `ToolMessage(content="cancelled: the user interrupted this run", status="error", tool_call_id=...)` inserted right after it, emitted as `[RemoveMessage(id=REMOVE_ALL_MESSAGES), *rebuilt]` (the compaction pattern), and append a trace record `{"interrupted": True, ...}` for the interrupted turn built from its messages with `build_trace(...)` (Task 7). Runs before compaction's own check, in the same node.
  - `cancel.py`: `def register(thread_id: str, cancel: Callable[[], None]) -> Callable[[], None]` (returns an unregister function) and `def cancel_thread(thread_id: str) -> int` (calls every registered callback, returns the count). `Engine._execute` registers `cursor.interrupt` for the current thread id (read from the LangGraph config var when present) while a query runs.
  - `TraceMiddleware.awrap_tool_call`: on `asyncio.CancelledError`, call `cancel_thread(thread_id)` and, when the tool is `run_python`, `close_sandbox(thread_id)`, then re-raise.
  - Steer uses the Agent Server's `multitask_strategy="interrupt"` [call]: the interrupted run's checkpoint is kept, and this task's repair makes it consistent. Queue uses `"enqueue"`. Stop is run cancel. (Recorded for W2; no backend code beyond the above.)
- [ ] **Step 1: Write failing tests**: `test_compact_repairs_dangling_tool_calls` (provider-shaped history after repair: every tool call followed by its ToolMessage), `test_compact_records_interrupted_trace`, `test_compact_no_op_on_consistent_history`, `test_cancel_interrupts_running_query` (a slow DuckDB query in a thread; `cancel_thread` makes it raise `QueryError` well before the 60 s timeout), `test_cancelled_run_python_closes_sandbox` (fake sandbox).
- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `make test`, `make lint` — green. Then the **durability check** (manual, recorded in the followups doc): `make dev`, create a thread and one run via the SDK, stop the server by PID, start it again, read the thread back with its `trace` key. If the thread is gone, stop and bring a Ruling to the owner (durable checkpointer for local runs) instead of building one silently.
- [ ] **Step 5: Hand back** (`feat(steer): repair interrupted threads, cancel in-flight queries and sandboxes`).

---

### Task 9: Eval gate and docs

**Files:**
- Modify: `docs/results/agent-baseline.md`, `README.md`, `CLAUDE.md`
- Create: `docs/superpowers/plans/2026-11-15-plan-w1-followups.md`

- [ ] **Step 1: Run** `make eval-agent` (LangSmith experiment, needs `.env` keys and the sandbox image). **Gate:** at least 38/55 passed, `python` category 5/5, no category down by more than one question vs the B2 run (`openai-gpt-6-luna-81b55b5-aae16c18`: lookup 80%, comparison 75%, forecast 88%, off_model 33%, trap 40%, safe 67%). A forecast row that fails only on `chart_kind` is a regression: fix the prompt policy and re-run; don't waive it.
- [ ] **Step 2: Live check:** `make dev`, ask "How many trips should we expect next month by borough, and how sure are you?"; confirm the answer lists a chart id, `GET /charts/...` returns a `small_multiples` of `forecast_fan` payload, and the thread state holds a `trace` record with tokens and timings.
- [ ] **Step 3: Write** the W1 section of `agent-baseline.md` (experiment name, pass rate, per-category, chart_kind pass count), the README "How it's built" line for the chart service and trace, `CLAUDE.md` commands (none new beyond `make dev` serving the routes; mention `config/models.yaml`), and the followups doc (every Ruling from the ledger, the durability result, deferred items).
- [ ] **Step 4: Run** `make test`, `make test-docker`, `make test-taxi`, `make lint` — all green.
- [ ] **Step 5: Hand back** (`docs: W1 eval baseline, followups`).
