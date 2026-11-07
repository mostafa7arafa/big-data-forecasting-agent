# Plan A: Data & Forecasting Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the governed data layer and forecasting core as a tested Python library with no LLM yet. Pieces: dataset registration, a read-only DuckDB engine with cutoff, a result store with summary cards, the semantic layer and compiler, the deterministic profiler, backtested forecasting and model-level cutoff replay. The SQL and forecasting tools are proven independently before any agent exists.

**Architecture:**
- Code in `src/analyst/` is dataset-agnostic. Everything taxi-specific lives in `datasets/nyc_taxi/` (YAML and Markdown).
- A `DataService` facade owns an `Engine` (DuckDB, locked down, optional cutoff), a `ResultStore` (Parquet handles plus summary cards) and the dataset's `Memory` (semantic model, quirks, series).
- Every tool-shaped method returns a `SummaryCard`, never raw data. Plan B wraps these methods as LangChain tools.

**Tech Stack:** Python 3.12, uv, duckdb 2.0.0, pyarrow, pandas, pydantic v2, pydantic-settings, PyYAML, statsforecast 2.x, utilsforecast 0.2.x, pytest.

**Spec:** `docs/superpowers/specs/2026-11-07-taxi-analytics-agent-design.md` (§3, §4, §5.1, §5.3, §5.4, §5.8 step 1, §6 cutoff replay, §9 steps 1–4). Executors read both.

## Global Constraints

- Python 3.12 via `uv`. Pin `duckdb==2.0.0`. Every dependency is pinned in `uv.lock`.
- **No dataset-specific strings in `src/`.** `taxi`, `borough`, `tpep_`, `PULocationID`, `DOLocationID` must not appear there. This is enforced by a test in Task 1.
- Unit tests run on a tiny synthetic dataset (fixture in Task 2). Taxi tests are integration tests marked `@pytest.mark.taxi` and skip when `data/raw` is missing.
- Taxi data window: 2023-01-01 ≤ pickup < 2026-10-01 (161,781,947 raw rows; 176 outside the window).
- The engine is read-only: single `SELECT` only, `enable_external_access=false` with `allowed_directories` set to the dataset's data directories, `lock_configuration=true`, a 60 s query timeout and a 1,000,000-row result cap.
- Summary cards: a 10-row preview, rendered text ≤ 4,096 characters.
- Forecast levels `[80, 95]`. Baseline `SeasonalNaive`. Daily defaults: horizon 28, step 7, 12 windows, seasons 7 and 365. Monthly defaults: horizon 3, step 1, 12 windows, season 12.
- Forecast origin is the end of available data (respecting the cutoff), never today's date.
- Cutoff replay cutoffs: `2025-07-31`, `2026-01-31`, `2026-07-31`. A model stays in the lineup if `relative_mae < 1` vs SeasonalNaive at ≥ 2 of 3 cutoffs.
- Every memory item carries `source` (`profile|llm|owner|feedback`), `status` (`proposed|approved|rejected`) and `confidence` (0–1).
- Run all commands from the repo root. Tests use `uv run pytest`.

## Review Focus

1. **A time range past the end of the data**, e.g. December 2026 actuals. The answer must be an empty result with an explicit "data ends at …" note, never zeros. Covered by a test in Task 6.
2. **Dimension values with a different case or spacing** (`"manhattan"`, `" Staten island "`). These resolve to the canonical value, or fail with a "did you mean" list. Covered by a test in Task 6.
3. **Known gap days inside a training window** (2023-09-21..24). Imputed and reported in the forecast's `adjustments`, never trained on as near-zero. Covered by a test in Task 8.
4. **A partial last period** (cutoff 2026-07-15 with monthly grain). The trailing partial month is flagged in the card and dropped before forecasting. Covered by tests in Tasks 6 and 8.
5. **A raw query returning more rows than the cap.** Fails fast with an "aggregate further" error, without loading everything into memory. Covered by a test in Task 3.

---

## File Structure

```
pyproject.toml                  package "analyst", deps, pytest markers
Makefile                        test, test-taxi, lint, replay
.env.example
src/analyst/
  __init__.py
  config.py                     Settings (paths)
  data/
    dataset.py                  DatasetConfig + load_dataset()
    engine.py                   Engine (DuckDB, lock-down, cutoff, coverage)
    results.py                  ResultStore, SummaryCard, Handle
    service.py                  DataService facade (query_metrics, run_sql, describe, forecast)
  semantic/
    model.py                    SemanticModel pydantic schema + MemoryItem base
    compiler.py                 MetricQuery, compile_query()
    resolve.py                  dimension value resolution
  memory/
    files.py                    Memory: load semantic/quirks/series/context_facts/recap
  calibration/
    profile.py                  profile_dataset() → Profile, write profile.json
  forecasting/
    models.py                   candidate_models(freq)
    core.py                     prepare_series(), backtest(), forecast()
    replay.py                   replay(), lineup()
datasets/nyc_taxi/
  dataset.yaml
  memory/semantic.yaml quirks.yaml series.yaml context_facts.yaml recap.md
evals/
  datasets/sql_golden.jsonl
  graders/result_match.py
  gold/                         (generated, gitignored)
scripts/
  download_data.sh              (exists)
  build_gold.py                 cache gold results
  replay_models.py              run replay → docs/results/model_replay.md + series.yaml lineup
tests/
  conftest.py                   synthetic dataset fixture, taxi marker skip
  unit/...  integration/...
```

Delete the spike code: `src/taxi_forecast_agent/`, `tests/test_sql_tool.py`, `tests/test_forecast_tool.py`, `scripts/inspect_data.py`. Ideas from it are reused (lock-down, timeout via `con.interrupt`).

---

### Task 1: Project skeleton, settings, dataset-agnostic guard

**Files:**
- Modify: `pyproject.toml` (name `analyst`, package `src/analyst`, drop `[project.scripts]`, add deps `utilsforecast pydantic pydantic-settings pyyaml`, dev `pytest ruff`, pytest markers `taxi`, `slow`)
- Create: `src/analyst/__init__.py`, `src/analyst/config.py`, `Makefile`, `.env.example`
- Delete: spike files listed above
- Test: `tests/unit/test_config.py`, `tests/unit/test_dataset_agnostic.py`

**Interfaces:**
- Produces: `analyst.config.Settings` (pydantic-settings, env prefix `ANALYST_`), with fields `repo_root: Path`, `datasets_dir: Path = repo_root/"datasets"`, `artifacts_dir: Path = repo_root/"data"/"artifacts"`. Also `get_settings() -> Settings` (cached).

- [ ] **Step 1: Write the failing tests**

```python
def test_settings_defaults(monkeypatch):
    s = Settings()
    assert s.datasets_dir == s.repo_root / "datasets"
    assert s.artifacts_dir == s.repo_root / "data" / "artifacts"

def test_settings_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("ANALYST_ARTIFACTS_DIR", str(tmp_path))
    assert Settings().artifacts_dir == tmp_path

FORBIDDEN = ["taxi", "borough", "tpep_", "PULocationID", "DOLocationID"]
def test_src_has_no_dataset_specific_strings():
    hits = [(p, w) for p in Path("src/analyst").rglob("*.py")
            for w in FORBIDDEN if w.lower() in p.read_text().lower()]
    assert hits == []
```

- [ ] **Step 2: Run** `uv run pytest tests/unit -v`. Expected: FAIL (`analyst` not importable).
- [ ] **Step 3: Implement** `Settings`. `repo_root` defaults to the directory three levels above `config.py`. Restructure `pyproject.toml`, delete the spike, and add a Makefile with targets `test` (`uv run pytest -m "not taxi"`), `test-taxi` (`uv run pytest -m taxi`), `lint` (`uv run ruff check src tests`) and `replay`. `.env.example` lists `ANALYST_ARTIFACTS_DIR`. Add `data/` and `evals/gold/` to `.gitignore` (data/ already there).
- [ ] **Step 4: Run** `uv run pytest tests/unit -v && uv run ruff check src tests`. Expected: PASS, no lint errors.
- [ ] **Step 5: Commit** `git add -A && git commit -m "chore: analyst package skeleton, settings, dataset-agnostic guard"`

---

### Task 2: Dataset registration + synthetic test fixture

**Files:**
- Create: `src/analyst/data/__init__.py`, `src/analyst/data/dataset.py`, `datasets/nyc_taxi/dataset.yaml`, `tests/conftest.py`
- Test: `tests/unit/test_dataset.py`

**Interfaces:**
- Produces:
  - `Lookup(name: str, file: Path)`
  - `DatasetConfig(name: str, description: str, files: str (glob, relative to repo_root), table: str, time_column: str, valid_time_range: tuple[date, date] (start incl., end excl.), lookups: list[Lookup], root: Path)`, with property `data_dirs -> list[Path]` (unique parent dirs of `files` and lookup files) and property `memory_dir -> Path` (`root/"memory"`)
  - `load_dataset(name: str, settings: Settings | None = None) -> DatasetConfig`. Reads `datasets/<name>/dataset.yaml` and resolves paths against `repo_root`.
  - pytest fixture `synthetic_dataset(tmp_path_factory) -> DatasetConfig`, session-scoped. It creates `events_2024-01.parquet` and `events_2024-02.parquet` with columns `ts TIMESTAMP, region_id INT, amount DOUBLE, qty INT`. That is 60 days of hourly rows, regions 1–4, weekly pattern in `amount`, and days 2024-01-20..21 with 2 rows each (a gap). Also `regions.csv` (`region_id,area`: 1,North / 2,North / 3,South / 4,South), a `dataset.yaml` with `table: events`, `time_column: ts`, `valid_time_range: [2024-01-01, 2024-03-01]`, and a `memory/semantic.yaml` (written in Task 5's schema; Task 5 extends this fixture). A second file has an extra column `note VARCHAR` (schema drift).
  - The fixture also writes 3 rows with `ts` in 2001 (implausible timestamps).
  - `conftest.py` registers a `taxi` skip: skip if `data/raw/yellow_tripdata_2023-01.parquet` is missing.

Taxi `dataset.yaml`:
```yaml
name: nyc_taxi
description: NYC TLC yellow taxi trip records
files: data/raw/yellow_tripdata_*.parquet
table: trips
time_column: tpep_pickup_datetime
valid_time_range: [2023-01-01, 2026-10-01]
lookups:
  - {name: zones, file: data/raw/taxi_zone_lookup.csv}
```

- [ ] **Step 1: Write failing tests**: `test_load_taxi_dataset` (`table == "trips"`, `valid_time_range == (date(2023,1,1), date(2026,10,1))`, `lookups[0].name == "zones"`), `test_data_dirs_unique` (taxi → `[repo_root/"data"/"raw"]`), `test_missing_dataset_raises` (`FileNotFoundError` naming the path), `test_bad_yaml_field_raises` (missing `time_column` → `pydantic.ValidationError`), `test_synthetic_fixture_loads` (two parquet files exist, `table == "events"`).
- [ ] **Step 2: Run** `uv run pytest tests/unit/test_dataset.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `dataset.py` and the fixture.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(data): dataset registration and synthetic fixture`

---

### Task 3: Read-only DuckDB engine with cutoff and coverage

**Files:**
- Create: `src/analyst/data/engine.py`
- Test: `tests/unit/test_engine.py`, `tests/integration/test_engine_taxi.py`

**Interfaces:**
- Consumes: `DatasetConfig`.
- Produces:
  - `class QueryError(Exception)`
  - `Coverage(start: datetime, end: datetime)`. `end` is the max timestamp present.
  - `class Engine`:
    - `__init__(self, dataset: DatasetConfig, cutoff: date | None = None, *, timeout_s: float = 60, max_rows: int = 1_000_000)`
    - `cutoff: date | None`
    - `execute(self, sql: str) -> pyarrow.Table`
    - `coverage(self) -> Coverage` (cached)
    - `columns(self) -> list[tuple[str, str]]`
  - Views:
    - `<table>`: valid time range **and** `time_column < cutoff + 1 day` when a cutoff is set;
    - `<table>_raw`: no valid-range filter, still cut off;
    - one view per lookup.
  - Lock-down order: create views → `SET allowed_directories=[...data_dirs]` → `SET enable_external_access=false` → `SET lock_configuration=true` (verified working on duckdb 2.0.0).
  - `execute` accepts exactly one `SELECT` statement (`con.extract_statements`, type `SELECT`), else `QueryError`. A `threading.Timer(timeout_s, con.interrupt)` turns an interrupt into `QueryError("query exceeded 60s timeout")`. Results are fetched with `.limit(max_rows + 1)`, raising `QueryError("result has more than N rows; aggregate further")` when exceeded. DuckDB errors become `QueryError` with the first line of the message only.

- [ ] **Step 1: Write failing unit tests** on `synthetic_dataset`:
  - `test_view_filters_valid_range` (`select count(*) from events where ts < '2024-01-01'` → 0; same on `events_raw` → 3);
  - `test_cutoff_hides_later_rows` (`Engine(ds, cutoff=date(2024,1,31)).coverage().end < datetime(2024,2,1)`, and `select max(ts) from events_raw` also < 2024-02-01);
  - `test_union_by_name_schema_drift` (`note` is in `columns()`);
  - `test_rejects_non_select` (parametrized: `DROP VIEW events`, `CREATE TABLE x AS SELECT 1`, `COPY (SELECT 1) TO 'x.csv'`, `SELECT 1; SELECT 2`, `SET threads=1`, `INSTALL httpfs`, `ATTACH 'x.db'`);
  - `test_blocks_other_files` (`select * from read_csv('/etc/passwd')` → `QueryError`);
  - `test_row_cap` (`Engine(ds, max_rows=10).execute("select * from events")` → `QueryError` containing "aggregate further");
  - `test_timeout` (`Engine(ds, timeout_s=0.5)` running `select count(*) from range(10_000_000_000)` raises `QueryError` containing "timeout");
  - `test_error_is_one_line`.
- [ ] **Step 2: Write failing taxi integration test** `test_month_count_matches_pyarrow` (marked `taxi`): count 2024-03 rows via pyarrow compute on the single file (the spike's approach) == `execute` count.
- [ ] **Step 3: Run** `uv run pytest tests/unit/test_engine.py -v`. Expected: FAIL.
- [ ] **Step 4: Implement** `engine.py`.
- [ ] **Step 5: Run** `uv run pytest tests/unit/test_engine.py -v && uv run pytest -m taxi tests/integration/test_engine_taxi.py -v`. Expected: PASS.
- [ ] **Step 6: Commit** `feat(data): locked-down DuckDB engine with cutoff`

---

### Task 4: Result store and summary cards

**Files:**
- Create: `src/analyst/data/results.py`
- Test: `tests/unit/test_results.py`

**Interfaces:**
- Produces:
  - `ColumnStats(name: str, type: str, nulls: int, min: float | str | None, max: float | str | None, mean: float | None)`
  - `SummaryCard` (pydantic):
    - `handle_id: str`
    - `rows: int`
    - `columns: list[tuple[str, str]]`
    - `preview: str` (markdown, ≤ 10 rows)
    - `stats: list[ColumnStats]`
    - `produced_by: str` (tool name)
    - `source: str` (SQL or code)
    - `governed: bool`
    - `elapsed_s: float`
    - `notes: list[str] = []`
    - `meta: dict = {}`

    Method `render() -> str` (≤ 4,096 chars; truncates the preview first, then stats, appending `"…(truncated)"`).
  - `class ResultStore`:
    - `__init__(self, root: Path)`
    - `save(self, table: pa.Table, *, produced_by: str, source: str, governed: bool, elapsed_s: float, notes: list[str] = (), meta: dict | None = None) -> SummaryCard`
    - `load(self, handle_id: str) -> pa.Table`
    - `card(self, handle_id: str) -> SummaryCard`
    - `exists(self, handle_id: str) -> bool`
  - Storage: `root/<handle_id>.parquet` plus `root/<handle_id>.card.json`. `handle_id = "r_" + sha256(source + produced_by + json(meta, sort_keys)).hexdigest()[:10]`.
  - Unknown handle → `KeyError("unknown result handle r_…")`.

- [ ] **Step 1: Write failing tests**:
  - `test_roundtrip` (save → load equals the table; `card()` equals the returned card);
  - `test_preview_max_10_rows` (1,000-row table → preview has 10 data rows);
  - `test_stats` (numeric min/max/mean/nulls correct for `[1, 2, None, 5]`);
  - `test_render_capped` (200 columns × 10 rows → `len(card.render()) <= 4096` and it ends with `"…(truncated)"`);
  - `test_same_source_same_handle`;
  - `test_unknown_handle`.
- [ ] **Step 2: Run.** Expected: FAIL. **Step 3: Implement.** **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(data): result store with summary cards`

---

### Task 5: Dataset memory files, schemas and the taxi memory

**Files:**
- Create: `src/analyst/semantic/__init__.py`, `src/analyst/semantic/model.py`, `src/analyst/memory/__init__.py`, `src/analyst/memory/files.py`
- Create: `datasets/nyc_taxi/memory/{semantic.yaml,quirks.yaml,series.yaml,context_facts.yaml,recap.md}`
- Modify: `tests/conftest.py` (synthetic `memory/` files)
- Test: `tests/unit/test_memory.py`, `tests/integration/test_taxi_memory.py`

**Interfaces:**
- Produces:
  - `MemoryItem` (pydantic base): `source: Literal["profile","llm","owner","feedback"] = "owner"`, `status: Literal["proposed","approved","rejected"] = "approved"`, `confidence: float = 1.0` (0–1, validated).
  - `TimeDimension(name, column, grains: list[Literal["hour","day","week","month"]])`
  - `Join(MemoryItem: name, table, on)`
  - `Dimension(MemoryItem: name, expr, join: str | None, synonyms: list[str], description)`
  - `Filter(MemoryItem: name, expr, description)`
  - `Metric(MemoryItem: name, expr, filters: list[str], synonyms, description, unit)`
  - `SemanticModel(table, time_dimension, joins, dimensions, filters, metrics)`, with `metric(name)`, `dimension(name)` (each matching name or synonym case-insensitively; `KeyError` listing available names) and `approved_only() -> SemanticModel`
  - `Quirk(MemoryItem: kind: Literal["gap","anomaly","schema_change","data_issue"], start: date, end: date | None, description, treatment: Literal["impute","flag","exclude"] | None)`
  - `SeriesSpec(MemoryItem: name, metric, dimensions: list[str], freq: Literal["D","MS"], season_lengths: list[int], recommended_models: list[str])`
  - `ContextFact(MemoryItem: date: date, end: date | None, label, kind: Literal["holiday","policy","data"])`
  - `Memory(semantic, quirks: list[Quirk], series: list[SeriesSpec], context_facts: list[ContextFact], recap: str)` and `load_memory(dataset: DatasetConfig) -> Memory`

Taxi memory content (from spec §4, all `source: owner`, `status: approved`):
- `semantic.yaml`:
  - **joins:** `pickup_zone`: `zones` on `trips.PULocationID = pickup_zone.LocationID`; `dropoff_zone`: likewise on `DOLocationID`.
  - **dimensions:**
    - `pickup_borough` (`pickup_zone.Borough`, synonyms `[borough]`), `pickup_zone_name` (`pickup_zone.Zone`, synonyms `[zone, neighborhood]`), `dropoff_borough`;
    - `hour` (`hour(tpep_pickup_datetime)`), `weekday` (`dayname(tpep_pickup_datetime)`);
    - `payment_type` (CASE mapping 1 Credit card, 2 Cash, 3 No charge, 4 Dispute, else Other).
  - **filters:** `valid_trip`: `fare_amount > 0 AND trip_distance > 0`; `known_zone`: `PULocationID NOT IN (264, 265)`.
  - **metrics:**
    - `trips` (`count(*)`, no filters; the description says it includes dirty rows);
    - `revenue` (`sum(total_amount)`, `[valid_trip]`, unit USD);
    - `avg_fare` (`avg(fare_amount)`, `[valid_trip]`, unit USD);
    - `avg_distance` (`avg(trip_distance)`, `[valid_trip]`, unit miles);
    - `tip_rate` (`sum(tip_amount)/nullif(sum(fare_amount),0)`, `[valid_trip]`);
    - `congestion_fees` (`sum(cbd_congestion_fee)`, unit USD).
  - **time_dimension:** `pickup_time` on `tpep_pickup_datetime`, grains `[hour, day, week, month]`.
- `quirks.yaml`:
  - gap 2023-09-21..2023-09-24, treatment `impute`;
  - anomaly 2026-02-23, treatment `flag`, description "unusually low (≈26k trips); cause not in data";
  - schema_change 2025-01-05 `cbd_congestion_fee` added;
  - schema_change `request_source` added (date = first month the column is non-null, found by query in Step 3);
  - data_issue: 176 rows outside window; ~2.5% fare/distance ≤ 0; 0.4% unknown zone.
- `context_facts.yaml`:
  - US federal holidays 2023-01-01..2026-12-31 (enumerate them);
  - policy 2025-01-05 "Manhattan congestion pricing starts";
  - data 2023-09-21..24 "known data gap".
- `series.yaml`:
  - `daily_trips` (metric trips, no dims, D, [7, 365], recommended_models [] until Task 10);
  - `daily_trips_by_borough` (dims [pickup_borough], D);
  - `monthly_trips_by_borough` (MS, [12]).
- `recap.md`: ≤ 25 lines covering what a row is, coverage, borough skew (Manhattan ≈3.3M/month, Queens ≈350k, Brooklyn ≈150k, Bronx and Staten Island tiny), the quirks above and the dirty-row conventions.

- [ ] **Step 1: Write failing tests**:
  - `test_load_synthetic_memory`;
  - `test_metric_lookup_by_synonym` (taxi: `memory.semantic.dimension("borough").name == "pickup_borough"`);
  - `test_unknown_metric_lists_available` (`KeyError` text contains `"trips"`);
  - `test_confidence_validated` (1.5 → `ValidationError`);
  - `test_approved_only_drops_proposed`;
  - taxi integration `test_every_expr_compiles`: for each dimension, filter and metric, `Engine.execute` a `SELECT <expr> FROM trips LEFT JOIN … LIMIT 0` succeeds.
- [ ] **Step 2: Run.** Expected: FAIL. **Step 3: Implement** the schemas, loader and YAML files. **Step 4: Run** unit and taxi tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat(memory): dataset memory schemas and hand-written taxi memory`

---

### Task 6: Semantic compiler (incl. comparisons, value resolution, coverage notes)

**Files:**
- Create: `src/analyst/semantic/compiler.py`, `src/analyst/semantic/resolve.py`
- Test: `tests/unit/test_compiler.py`, `tests/integration/test_compiler_taxi.py`

**Interfaces:**
- Consumes: `SemanticModel`, `Engine`.
- Produces:
  - `FilterSpec(dimension: str, op: Literal["eq","in","neq","not_in"], values: list[str | int])`
  - `MetricQuery` (pydantic):
    - `metrics: list[str]` (≥ 1)
    - `dimensions: list[str] = []`
    - `filters: list[FilterSpec] = []`
    - `grain: Literal["hour","day","week","month"] | None = None`
    - `time_range: tuple[date, date] | None = None` (start incl., end excl.)
    - `compare_to: Literal["previous_period","same_period_last_year"] | None = None`
  - `share: bool = False`
    - `order_by: str | None = None`
    - `limit: int | None = None`
  - `compile_query(model: SemanticModel, q: MetricQuery, *, allow_proposed: bool = False) -> str`
  - `resolve_values(engine: Engine, model: SemanticModel, dimension: str, values: list) -> list` (case and whitespace-insensitive match to distinct values; raises `ValueError("unknown value 'X' for pickup_borough; did you mean: …")`, using difflib with the top 5 matches)
  - Output columns, in order:
    1. `period` (DATE, if grain set);
    2. one column per dimension;
    3. one column per metric;
    4. with `compare_to`, per metric `<m>_compare` and `<m>_change_pct` (= `(m - m_compare) / nullif(m_compare, 0) * 100`, rounded to 2);
    5. with `share`, per metric `<m>_share_pct` (= `m / sum(m) OVER (PARTITION BY period) * 100`, rounded to 2; partitioned over everything when there is no grain).

**Semantics.** A metric's filters apply via `FILTER (WHERE …)` per metric, so metrics with different filters can share one query. Time range filters on the time dimension column. With `compare_to`, compile two CTEs: `cur` (time_range) and `cmp` (time_range shifted back). Shift by:
- the range length, for `previous_period`;
- `same_period_last_year`, by grain: month → 12 months; week → 52 weeks; day → 364 days (same weekday); hour → 364 days.

Then `cur LEFT JOIN cmp ON cur.period = cmp.period + shift AND` the dimensions match. `compare_to` requires `time_range`, else `ValueError`. Proposed or rejected items raise unless `allow_proposed`.

- [ ] **Step 1: Write failing unit tests** on the synthetic dataset:
  - `test_total_metric`: `sum(amount)` matches hand-written SQL;
  - `test_grain_and_dimension`: day × area matches a hand-written `GROUP BY`;
  - `test_metric_filter_applied`: a metric with filter `qty > 0` matches;
  - `test_compare_previous_period`: Feb vs Jan totals, with `change_pct` equal to `(feb - jan) / jan * 100` rounded to 2;
  - `test_compare_requires_time_range`;
  - `test_share_sums_to_100` (per day, the `amount_share_pct` across areas sums to 100 ± 0.05);
  - `test_unknown_dimension_raises`;
  - `test_proposed_metric_rejected_unless_allowed`;
  - `test_resolve_case_insensitive`: `"north"` → `"North"`; `" South "` → `"South"`;
  - `test_resolve_unknown_suggests`: `"Nrth"` → `ValueError` containing `"North"`.
- [ ] **Step 2: Write failing taxi tests:**
  - `test_monthly_trips_by_borough_2024` equals hand-written SQL over `trips JOIN zones`;
  - `test_yoy_manhattan_2025_months`: `trips_compare` for 2025-03 equals the 2024-03 Manhattan count.
- [ ] **Step 3: Run.** Expected: FAIL. **Step 4: Implement.** **Step 5: Run** unit and taxi tests. Expected: PASS.
- [ ] **Step 6: Commit** `feat(semantic): metric compiler with period comparisons and value resolution`

---

### Task 7: DataService facade: query_metrics, run_sql, describe

**Files:**
- Create: `src/analyst/data/service.py`
- Test: `tests/unit/test_service.py`

**Interfaces:**
- Consumes: `Engine`, `ResultStore`, `Memory`, `compile_query`, `resolve_values`.
- Produces `class DataService`:
  - `__init__(self, dataset: DatasetConfig, *, cutoff: date | None = None, store_root: Path | None = None)`. The default store root is `settings.artifacts_dir / dataset.name / (cutoff or "full")`.
  - Attributes `engine`, `store`, `memory`.
  - `query_metrics(self, q: MetricQuery) -> SummaryCard`. Resolves filter values first, runs the compiled SQL and saves with `produced_by="query_metrics"` and `governed=True`. `source` is the compiled SQL.
  - `run_sql(self, sql: str) -> SummaryCard`, with `produced_by="run_sql"` and `governed=False`.
  - `describe(self, target: str) -> str`. Targets:
    - `"metrics"`: name, synonyms, unit, description;
    - `"dimensions"`: the same for dimensions;
    - `"dimension:<name>"`: top 50 distinct values with counts;
    - `"schema"`: the table's columns;
    - `"coverage"`: start, end, cutoff;
    - a handle id: `card.render()`.

    An unknown target raises `ValueError` listing the valid targets.
  - **Coverage notes**, added to `card.notes` by `query_metrics`:
    - When `time_range` extends beyond `coverage().end` (or starts before `coverage().start`): `"requested range extends beyond available data (2024-01-01 .. 2024-02-29 23:00)"`.
    - When the grain is set and the last period is partial (`coverage().end` is before the period's end): `"last period <period> is partial (data ends <end>)"`, plus `meta["partial_last_period"] = "<period>"`.

- [ ] **Step 1: Write failing tests**:
  - `test_query_metrics_card_governed`;
  - `test_run_sql_not_governed`;
  - `test_range_beyond_data_note`: a query for 2024-03 returns `rows == 0` and the note contains "beyond available data";
  - `test_partial_last_period_note`: `DataService(ds, cutoff=date(2024,2,15))` with a monthly grain → `meta["partial_last_period"] == "2024-02-01"`;
  - `test_describe_dimension_values` (the output contains `North` and `South`);
  - `test_describe_unknown_target`;
  - `test_handles_isolated_by_cutoff` (the same query with a different cutoff gives a different store root).
- [ ] **Step 2: Run.** Expected: FAIL. **Step 3: Implement.** **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(data): DataService facade with coverage-aware cards`

---

### Task 8: Forecasting core: candidates, series prep, backtest, forecast

**Files:**
- Create: `src/analyst/forecasting/__init__.py`, `models.py`, `core.py`
- Test: `tests/unit/test_forecasting.py`

**Interfaces:**
- Consumes: `Quirk` (for gap imputation).
- Produces:
  - `candidate_models(freq: Literal["D","MS"]) -> list`:
    - `D`: `AutoETS(season_length=7)`, `AutoARIMA(season_length=7)`, `MSTL(season_length=[7, 365], trend_forecaster=AutoETS(model="ZZN"))`, `SeasonalNaive(season_length=7)`, `HistoricAverage()`;
    - `MS`: `AutoETS(12)`, `AutoARIMA(12)`, `SeasonalNaive(12)`, `HistoricAverage()`.
  - `DEFAULTS = {"D": dict(horizon=28, step=7, n_windows=12, season=7), "MS": dict(horizon=3, step=1, n_windows=12, season=12)}`
  - `PreparedSeries(df: pd.DataFrame (unique_id, ds, y), adjustments: list[str])`
  - `prepare_series(df: pd.DataFrame, *, freq, data_end: datetime, quirks: list[Quirk] = (), id_col: str | None, time_col: str, value_col: str) -> PreparedSeries`. It:
    - renames columns to `unique_id`/`ds`/`y`;
    - drops a trailing partial period (`MS`: the last month when `data_end` is before that month's last day 23:00; `D`: the last day when `data_end.hour < 23`), adding an adjustment;
    - replaces `y` on `impute` quirk dates by linear interpolation in time, and on missing dates by reindexing to the full frequency and interpolating, recording each as an adjustment, e.g. `"imputed 4 days 2023-09-21..2023-09-24 (known gap)"`.
  - `BacktestReport(scores: pd.DataFrame` with columns `unique_id, model, mae, mase, relative_mae, coverage_80, coverage_95, scaled_crps`, `best_model: str, beats_baseline: bool, windows: int, horizon: int)`
  - `backtest(series: PreparedSeries, *, freq, models=None, horizon=None, step=None, n_windows=None, levels=(80, 95)) -> BacktestReport`. Uses `StatsForecast.cross_validation` and `utilsforecast.evaluation.evaluate`, with `functools.partial(mase, seasonality=season)`, `mae`, `coverage` and `scaled_crps`. `relative_mae` = model MAE ÷ SeasonalNaive MAE per id. `best_model` = lowest mean `mase` across ids. `beats_baseline` = the best model's mean `relative_mae < 1`. When there is too little history for the windows (needs `2 * season + horizon * n_windows` points per id), reduce `n_windows` down to a minimum of 3; below that, raise `ValueError("series too short …")`.
  - `ForecastRun(forecast: pd.DataFrame` with columns `unique_id, ds, yhat, lo_80, hi_80, lo_95, hi_95`, `model: str, origin: datetime, backtest: BacktestReport, adjustments: list[str])`
  - `fit_predict(series: PreparedSeries, *, freq, horizon, models, levels=(80, 95)) -> pd.DataFrame`. Fits the given models on the full series and returns `unique_id, ds` plus `<model>`, `<model>-lo-<l>`, `<model>-hi-<l>` per model (the StatsForecast `forecast` layout), with no backtest.
  - `forecast(series: PreparedSeries, *, freq, horizon, levels=(80, 95), models=None) -> ForecastRun`. Backtests, then calls `fit_predict` with the best model. The origin is the last `ds`.

- [ ] **Step 1: Write failing tests**, using synthetic series built in the test:
  - `test_candidates_daily_include_mstl_and_baseline`;
  - `test_interval_ordering`: `lo_95 ≤ lo_80 ≤ yhat ≤ hi_80 ≤ hi_95` for all rows;
  - `test_horizon_length` (28 rows per id);
  - `test_baseline_relative_mae_is_one`;
  - `test_trend_series_beats_baseline` (weekly pattern + trend → `beats_baseline is True`);
  - `test_white_noise_reports_not_beating_baseline_honestly` (`beats_baseline` is a bool and the report carries the SeasonalNaive row; no assertion on the value);
  - `test_gap_quirk_imputed_and_reported` (4 near-zero days matching an impute quirk → `y` on those days within 20% of neighbors, and an adjustment mentions "known gap");
  - `test_missing_dates_reindexed`;
  - `test_partial_last_month_dropped` (`data_end = 2024-07-15` → last `ds` is 2024-06-01, with an adjustment);
  - `test_multi_series`;
  - `test_too_short_raises`.
- [ ] **Step 2: Run.** Expected: FAIL. **Step 3: Implement.** **Step 4: Run.** Expected: PASS (mark tests that take > 10 s as `slow`).
- [ ] **Step 5: Commit** `feat(forecasting): backtested model selection with honest baseline comparison`

---

### Task 9: DataService.forecast

**Files:**
- Modify: `src/analyst/data/service.py`
- Test: `tests/unit/test_service_forecast.py`, `tests/integration/test_forecast_taxi.py`

**Interfaces:**
- Consumes: `prepare_series`, `forecast` (Task 8), `ResultStore`.
- Produces: `DataService.forecast(self, handle_id: str, *, time_col: str = "period", value_col: str, id_col: str | None = None, freq: Literal["D","MS"], horizon: int | None = None, levels: tuple[int, int] = (80, 95)) -> SummaryCard`.
  - Loads the handle and passes `data_end = engine.coverage().end` and `memory.quirks`.
  - Saves the forecast table (`unique_id, ds, yhat, lo_80, hi_80, lo_95, hi_95`) as a handle with `produced_by="forecast"` and `governed` equal to the input handle's `governed`.
  - `meta` holds `model`, `origin`, `horizon`, `freq`, `beats_baseline`, `backtest` (list of score rows for the best model and SeasonalNaive) and `input_handle`.
  - `notes` holds the adjustments, plus `"no model beat the seasonal baseline"` when applicable.
  - Errors: unknown column → `ValueError` listing the columns. `horizon` defaults to `DEFAULTS[freq]["horizon"]`.

- [ ] **Step 1: Write failing unit tests:**
  - `test_forecast_from_query_handle` (synthetic daily amount by area → the card has `rows == 28 * 2` and `meta["model"]` among the candidates);
  - `test_forecast_governed_propagates`;
  - `test_forecast_bad_column`.
- [ ] **Step 2: Write the failing taxi test** `test_hero_monthly_by_borough_cutoff` (`slow`, `taxi`):
  - `DataService(load_dataset("nyc_taxi"), cutoff=date(2026,7,31))`;
  - `query_metrics(trips by pickup_borough, grain month, filter pickup_borough in [Manhattan, Queens, Brooklyn])`;
  - `forecast(..., id_col="pickup_borough", value_col="trips", freq="MS", horizon=2)`.

  Expected: 6 rows; `ds` is 2026-08-01 and 2026-09-01; the backtest is in `meta`.
- [ ] **Step 3: Run.** Expected: FAIL. **Step 4: Implement.** **Step 5: Run.** Expected: PASS.
- [ ] **Step 6: Commit** `feat(data): forecast tool on result handles`

---

### Task 10: Model-level cutoff replay and lineup

**Files:**
- Create: `src/analyst/forecasting/replay.py`, `scripts/replay_models.py`
- Test: `tests/unit/test_replay.py`

**Interfaces:**
- Consumes: `DataService`, `MetricQuery`, `prepare_series`, `fit_predict`, `candidate_models` (Task 8).
- Produces:
  - `ReplayReport(rows: pd.DataFrame` with columns `cutoff, unique_id, model, mae, mase, relative_mae, coverage_80, coverage_95)`
  - `replay(dataset: DatasetConfig, query: MetricQuery, *, cutoffs: list[date], freq, horizon, id_col: str | None, value_col: str) -> ReplayReport`. For each cutoff:
    1. build `DataService(cutoff=c)`;
    2. run the query, prepare the series and `fit_predict` **every** candidate model (not just the best);
    3. score against truth from `DataService(cutoff=None)` for the `horizon` periods after the cutoff;
    4. compute `mase` using the cut-off training data.
  - `lineup(report: ReplayReport, *, min_wins: int = 2) -> list[str]`. Returns the models with mean `relative_mae < 1` at ≥ `min_wins` cutoffs, always plus `"SeasonalNaive"` as the baseline.
  - `scripts/replay_models.py --dataset nyc_taxi --series <name>`:
    - reads the series spec from `series.yaml` (metric, dimensions, freq);
    - runs at cutoffs `2025-07-31, 2026-01-31, 2026-07-31`, with horizon `DEFAULTS[freq]["horizon"]`, capped at 2 for `MS` at the 2026-07-31 cutoff because the truth ends in 2026-09;
    - writes `docs/results/model_replay_<series>.md` (a table per cutoff plus the lineup);
    - updates that series' `recommended_models` in `series.yaml` with `source: profile` and `status: proposed`.
  - Makefile target `replay` runs it for `daily_trips` and `monthly_trips_by_borough`.

- [ ] **Step 1: Write failing tests** on the synthetic dataset (cutoffs 2024-02-10 and 2024-02-17, freq D, horizon 7):
  - `test_replay_scores_all_models_each_cutoff` (rows = cutoffs × ids × candidates);
  - `test_truth_not_visible_to_forecast`: the training max `ds` < cutoff + 1 day. Assert via an internal hook or a recorded `train_end` column;
  - `test_lineup_rule` (a hand-built report where model A wins 2 of 3 and B wins 1 of 3 → `["A", "SeasonalNaive"]`).
- [ ] **Step 2: Run.** Expected: FAIL. **Step 3: Implement.** **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Run the real replay:** `make replay`. Expected: two markdown reports in `docs/results/`, and `series.yaml` updated with proposed lineups. Paste the lineups into the commit message.
- [ ] **Step 6: Commit** `feat(forecasting): cutoff replay settles model lineup` (include `docs/results/*.md` and `series.yaml`)

---

### Task 11: Deterministic profiler (calibration step 1)

**Files:**
- Create: `src/analyst/calibration/__init__.py`, `src/analyst/calibration/profile.py`
- Test: `tests/unit/test_profile.py`, `tests/integration/test_profile_taxi.py`

**Interfaces:**
- Consumes: `Engine`, `DatasetConfig`.
- Produces:
  - `ColumnProfile(name, type, null_rate: float, distinct_approx: int, role: Literal["time","dimension","measure","id","other"], top_values: list[tuple[str, int]] | None, min, max, nonpositive_rate: float | None, first_non_null_month: str | None)`
  - `Dependency(determinant: str, dependent: str, via_lookup: str | None)`
  - `SeriesProfile(grain: Literal["day","month"], points: int, gaps: list[tuple[date, date]], anomalies: list[tuple[date, float]], acf: dict[int, float], seasonal_lags: list[int])`
  - `Profile(dataset: str, rows: int, raw_rows: int, implausible_timestamps: int, coverage: Coverage, columns: list[ColumnProfile], dependencies: list[Dependency], daily: SeriesProfile, monthly: SeriesProfile, elapsed_s: float)`
  - `profile_dataset(dataset: DatasetConfig) -> Profile` and `write_profile(profile, dataset) -> Path` (to `memory/profile.json`).

**Rules:**
- **Role:** time = the dataset's time column or a TIMESTAMP type. Measure = numeric with `distinct_approx > 50` and not ending in `ID`/`_id`. Id = integer whose name ends in `ID`/`_id`, or `distinct_approx > 0.5 * rows`. Dimension = string or integer with `distinct_approx ≤ 1000`.
- **Gaps:** daily counts below 10% of the centered 28-day rolling median, merged into ranges.
- **Anomalies:** daily robust z-score (median/MAD on residuals after removing the day-of-week median) with |z| > 4, excluding gap days.
- **ACF:** at lags 1, 7, 14, 30, 365 on daily counts (365 only with ≥ 2 years of data). `seasonal_lags` = lags with ACF > 0.3 among 7, 14, 30, 365.
- **Dependencies:** for each id or dimension column joined to a lookup (the join key is the lookup column with the same values, found by checking `LocationID`-style overlap ≥ 95% of distinct values), test the functional dependency `lookup.col → lookup.other_col` with `count(distinct other) per col == 1`. Generic logic only, no names.
- **Performance:** everything is pushed into DuckDB aggregates. It must profile 162M taxi rows in < 60 s.

- [ ] **Step 1: Write failing unit tests** on the synthetic dataset:
  - `test_roles` (`ts` time, `amount` measure, `region_id` id or dimension);
  - `test_gap_detected` (2024-01-20..21);
  - `test_implausible_timestamps == 3`;
  - `test_schema_drift_first_non_null` (`note` → `"2024-02"`);
  - `test_weekly_acf_detected` (7 in `seasonal_lags`);
  - `test_lookup_dependency` (`region_id → area` via `regions`);
  - `test_profile_json_roundtrip`.
- [ ] **Step 2: Write failing taxi test** (`taxi`, `slow`): the gap list contains a range covering 2023-09-21..2023-09-24; `implausible_timestamps == 176`; `cbd_congestion_fee.first_non_null_month == "2025-01"`; `elapsed_s < 60`.
- [ ] **Step 3: Run.** Expected: FAIL. **Step 4: Implement.** **Step 5: Run.** Expected: PASS.
- [ ] **Step 6: Commit** `feat(calibration): deterministic dataset profiler`

---

### Task 12: SQL golden-set seed and result-set grader

**Files:**
- Create: `evals/graders/result_match.py`, `evals/datasets/sql_golden.jsonl`, `scripts/build_gold.py`
- Test: `tests/unit/test_result_match.py`

**Interfaces:**
- Produces:
  - `GradeResult(key: str, score: bool, comment: str)`, which matches the openevals return shape `{key, score, comment}` via `.as_dict()`.
  - `result_set_match(got: pa.Table, expected: pa.Table, *, rel_tol: float = 1e-6, abs_tol: float = 1e-9, ignore_order: bool = True, ignore_column_names: bool = True) -> GradeResult`. It normalizes by: casting decimals to float; rounding floats to 6 significant digits before sorting; sorting rows by all columns when `ignore_order`; and matching columns positionally when `ignore_column_names`, otherwise by name. The comment describes the first difference (shape, column, or row index and values).
  - `sql_golden.jsonl` rows: `{"id": "q001", "tier": 1|2|3, "question": str, "gold_sql": str, "tolerance": {"rel": float} | null, "notes": str}`. Seed 15 questions:
    - **5 at tier 1:** a single-metric total for a month, trips on a given date, average fare 2024, the top 5 pickup zones in 2025, the payment-type split in 2023;
    - **5 at tier 2:** monthly trips by borough 2024, YoY Manhattan trips for March 2026, the hour-of-day profile on weekdays vs weekends, the tip rate by payment type, the airport-pickup share;
    - **5 at tier 3:** the busiest day per month in 2025, a borough's share change between 2023 and 2026, average trips per weekday excluding the known gap, the revenue per trip trend by quarter, the congestion fees total since 2025-01-05.

    Gold SQL is written against the base views (`trips`, `zones`) **independently of the semantic layer**.
  - `scripts/build_gold.py` executes each gold SQL with `Engine(load_dataset("nyc_taxi"))` and writes `evals/gold/<id>.parquet`.

- [ ] **Step 1: Write failing tests:**
  - `test_exact_match`;
  - `test_row_order_ignored`;
  - `test_float_tolerance` (1.0000001 vs 1.0 passes);
  - `test_value_mismatch_comment_names_row`;
  - `test_shape_mismatch`;
  - `test_column_names_ignored_by_default`;
  - `test_as_dict_shape`;
  - `test_golden_file_valid` (15 rows, unique ids, tiers 1–3, each `gold_sql` parses as exactly one SELECT via `duckdb.connect().extract_statements`).
- [ ] **Step 2: Run.** Expected: FAIL. **Step 3: Implement**, then write the 15 questions. **Step 4: Run** the unit tests, then `uv run python scripts/build_gold.py`. Expected: 15 parquet files and no errors.
- [ ] **Step 5: Add the taxi test** `tests/integration/test_semantic_vs_gold.py`. For each of the tier-1 and tier-2 seed questions, hand-write the equivalent `MetricQuery` in the test, run it through `DataService.query_metrics`, and assert `result_set_match(...).score`. This proves the governed path agrees with independent SQL. Questions the semantic layer can't express are listed in the test as `xfail` with a reason. **Run** `uv run pytest -m taxi tests/integration/test_semantic_vs_gold.py -v`. Expected: PASS or XFAIL only.
- [ ] **Step 6: Commit** `feat(evals): result-set grader, seed golden set, semantic-vs-gold check`

---

## Later plans (written after Plan A lands, from real interfaces)

- **Plan B: Agent harness.** Docker sandbox with AST guard and `run_python`; `create_agent` analyst inside the LangGraph `StateGraph` (plan → analyst → verify → answer); providers (OpenAI baseline, OpenRouter, NanoGPT); middleware (tool-result clearing, compaction digest, cost injection, context metering); LangChain tool wrappers over `DataService`; LangSmith wiring; chart-data service with M4/LTTB downsampling.
- **Plan C: Calibration and memory.** LLM draft, run-check, review-queue API, drift check; feedback loop; LangGraph Store user memory.
- **Plan D: Evals and results.** Full golden set (~70, second derivation by a different model plus owner review), forecast and agent cutoff replay, grounding set with owner labels and κ, adversarial set, LangSmith experiments across models, README results table, second-dataset calibration.
- **Web:** separate session, from `docs/superpowers/briefs/web-dashboard-brief.md`.
