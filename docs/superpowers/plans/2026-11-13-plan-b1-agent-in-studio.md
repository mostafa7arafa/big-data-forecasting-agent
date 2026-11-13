# Plan B1: Agent in Studio, Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A LangGraph agent over the Plan A data core, served by `langgraph dev` and traced to LangSmith. It has verified answers, background workers for slow work, parallel work under caps, live progress messages, and a ~50-question single-turn eval set run as a LangSmith experiment.

**Architecture:** One custom `StateGraph` (plan → analyst → verify → answer). `analyst` is a LangChain v1 `create_agent` loop over six typed tools wrapping `DataService`, plus our own middleware. Slow work goes to a `worker` deep agent through deepagents `AsyncSubAgentMiddleware`. Both graphs share one process-wide `DataService` per (dataset, cutoff), behind a query gate. Verification is deterministic code. Evals are JSONL + frozen Parquet gold, graded by code graders and uploaded to LangSmith.

**Tech Stack:** Python 3.12 (uv); `langchain` 1.5.x; `langgraph` 1.2.x; `langgraph-cli[inmem]` 0.4.x; `deepagents` 0.7.37 (pinned exactly); `langchain-openai` 1.7.x; `langchain-openrouter` 0.2.9; `langsmith` 0.14.x; `openevals` 0.2.5; existing DuckDB 2.0.0 / StatsForecast core.

**Spec:** `docs/superpowers/specs/2026-11-12-plan-b-agent-harness-design.md` (Plan B). It extends `docs/superpowers/specs/2026-11-07-taxi-analytics-agent-design.md`. Deferred items come from `docs/superpowers/plans/2026-11-08-plan-a-followups.md`. Executors read all three.

## Global Constraints

- `src/analyst/` stays dataset-agnostic. `tests/unit/test_dataset_agnostic.py` fails on `taxi`, `borough`, `tpep_`, `PULocationID`, `DOLocationID`. **This includes default values.** The default dataset name comes from `ANALYST_DATASET` (set to `nyc_taxi` in `.env.example` and `langgraph.json`), never from a literal in `src/`.
- The engine is a security boundary: a single SELECT, AST-validated, with the cutoff enforced in the engine. Nothing in this plan weakens `Engine._validate_sql` or reads data outside `Engine.execute`.
- Tools return summary cards (≤ 4,096 chars) and handles, never raw rows. Tool-progress events and status lines carry no table data.
- Forecasts always carry the SeasonalNaive comparison. The origin is the end of data, respecting the cutoff.
- Runtime values (`dataset`, `cutoff`, `model`, `background`) are read only from `Runtime.context` (an `AnalystContext`), never from `config["configurable"]`.
- Graph nodes are `async`, and callers use `ainvoke`/`astream`. deepagents' task notifications only work on the async path.
- Pin `deepagents==0.7.37` exactly. The async-subagent API is beta.
- Defaults: `ANALYST_MAX_BACKGROUND_TASKS=3`, `ANALYST_MAX_CONCURRENT_QUERIES=3`. Model default `openai:gpt-6-luna`; Task 6 verifies the id.
- Status lines and tool progress wording are **neutral and plain** (owner). Character voice comes later; see the dashboard brief.
- Error handling: every Plan-A exception becomes a subclass of `AnalystError(message, hint)` and keeps its old base (`ValueError`/`KeyError`), so existing tests still pass.
- Delegates never commit. The controller re-runs `make test`, `make lint` (and `make test-taxi` where a task touches the data path), reviews, and commits with the attribution trailer.

## Review Focus

1. **Numbers written in human formats.** "$2.81M", "2,894,704", "−3.0%" (Unicode minus), "3.4 million", "45k" must all trace correctly. Calendar numbers ("2025", "Oct 2026", "12 months") must never be treated as claims. Owned by Task 8 (`test_extract_human_formats`, `test_calendar_numbers_ignored`).
2. **A handle id that doesn't exist, or belongs to another cutoff.** For example, the model copies an id from an old thread, or a worker from a different cutoff. `describe`/`forecast` return an error with a hint, and verify marks the claim untraced. The run never raises. Owned by Task 7 (`test_unknown_handle_is_tool_error`) and Task 8 (`test_foreign_handle_untraced`).
3. **The same slow forecast requested twice at once** (two parallel tool calls, or the analyst and a worker). Both get the same handle, there are no torn files, and at most one computes. Owned by Task 3 (`test_concurrent_forecast_same_inputs_single_compute`).
4. **The model returns malformed or empty structured output** (missing `claims`, or prose only). The answer node still produces an `Answer` labeled Not verified with reason `"no structured answer"`, and the run never raises. Owned by Task 10 (`test_malformed_structured_output_not_verified`).
5. **A question with no numbers** ("which pickup zone is busiest?"). Claims cite handles but carry no values. That is Verified when the cited handles are governed, not Not verified. Owned by Task 8 (`test_no_numbers_verified_when_cited`).

---

## File map

| Path | Responsibility | Task |
|---|---|---|
| `src/analyst/errors.py` | `AnalystError` + `UnknownNameError`, `InvalidRequestError`, `ForecastTooSlowError` | 1 |
| `src/analyst/config.py` | new settings fields | 1 |
| `src/analyst/data/gate.py` | `QueryGate`, process-wide `get_gate()` | 2 |
| `src/analyst/data/registry.py` | `get_service(dataset, cutoff)`, `clear_services()` | 2 |
| `src/analyst/data/results.py`, `engine.py`, `service.py` | atomic save, lookup, locking, forecast cache/budget/meta, Part-0 treatments | 2–4 |
| `src/analyst/forecasting/core.py`, `replay.py` | windowed backtest with progress + budget, zero-fill | 3–4 |
| `src/analyst/semantic/model.py`, `resolve.py`, `compiler.py` | `Memory.as_of`, int coercion, public helpers, YoY doc | 4–5 |
| `src/analyst/memory/knowledge.py` | `TOPICS`, `read_topic()` | 5 |
| `src/analyst/providers/__init__.py` | `get_model()` | 6 |
| `src/analyst/context.py` | `AnalystContext` | 7 |
| `src/analyst/state.py` | `Claim`, `ChartSpec`, `Verification`, `Answer`, `Plan`, `AnalystState` | 7 |
| `src/analyst/tools/` | `_common.py` + one file per tool + `__init__.py` (`DATA_TOOLS`) | 7 |
| `src/analyst/verify/` | `numbers.py`, `trace.py`, `claims.py`, `__init__.py` (`verify_answer`) | 8 |
| `src/analyst/middleware/` | `clearing.py`, `metering.py`, `cost.py`, `__init__.py` (`analyst_middleware()`) | 9 |
| `src/analyst/prompts.py`, `src/analyst/graph.py`, `langgraph.json`, `Makefile` | graph + Studio | 10 |
| `src/analyst/background.py`, `src/analyst/worker.py` | async subagent wiring, worker graph | 11 |
| `evals/datasets/agent_single_turn.jsonl`, `evals/gold/a*.parquet`, `scripts/build_gold.py`, `evals/graders/agent.py` | eval set + graders | 12 |
| `evals/target.py`, `scripts/eval_agent.py`, `docs/results/agent-baseline.md` | runner + first experiment | 13 |

The ledger for this plan is `.superpowers/sdd/plan-b1/progress.md`.

---

### Task 1: Dependencies, settings, `AnalystError`

**Files:**
- Modify: `pyproject.toml`, `uv.lock`, `src/analyst/config.py`, `src/analyst/data/engine.py`, `src/analyst/semantic/model.py`, `src/analyst/semantic/compiler.py`, `src/analyst/semantic/resolve.py`, `src/analyst/data/service.py`, `src/analyst/forecasting/core.py`, `.env.example`
- Create: `src/analyst/errors.py`
- Test: `tests/unit/test_errors.py`, `tests/unit/test_config.py`

**Interfaces:**
- Produces:
  - `AnalystError(Exception)` with `__init__(self, message: str, *, hint: str = "")`, attribute `.hint: str`, and `str(e) == message`;
  - `UnknownNameError(AnalystError, KeyError)`;
  - `InvalidRequestError(AnalystError, ValueError)`;
  - `ForecastTooSlowError(AnalystError)`;
  - `QueryError` now subclasses `AnalystError`.
- Settings gains:
  - `dataset: str | None = None`;
  - `model: str = "openai:gpt-6-luna"`;
  - `max_background_tasks: int = 3`;
  - `max_concurrent_queries: int = 3`;
  - `forecast_budget_s: float = 120.0`;
  - `inline_forecast_limit_s: float = 15.0`;
  - `nanogpt_base_url: str = "https://nano-gpt.com/api/v1"`.
- Settings also gains un-prefixed secrets through `validation_alias`:
  - `openai_api_key` (`OPENAI_API_KEY`);
  - `openrouter_api_key` (`OPENROUTER_API_KEY`);
  - `nanogpt_api_key` (`NANOGPT_API_KEY`);
  - `langsmith_api_key` (`LANGSMITH_API_KEY`).

  All four are `SecretStr | None = None`.

- [ ] **Step 1: Add dependencies** with `uv add "langchain>=1.5,<1.6" "langgraph>=1.2.21,<1.3" "deepagents==0.7.37" "langchain-openai>=1.7.4,<1.8" "langchain-openrouter==0.2.9" "langsmith>=0.14.17,<0.15" "openevals==0.2.5"` and `uv add --dev "langgraph-cli[inmem]>=0.4.43,<0.5"`. Expected: `uv.lock` updates and `uv run python -c "import deepagents, langgraph, langchain"` exits 0.
- [ ] **Step 2: Write failing tests.**

```python
def test_errors_keep_old_bases_and_hint():
    e = UnknownNameError("unknown metric 'trip'", hint="call describe('metrics')")
    assert isinstance(e, KeyError) and isinstance(e, AnalystError)
    assert e.hint == "call describe('metrics')" and str(e) == "unknown metric 'trip'"
    assert issubclass(QueryError, AnalystError)
    assert isinstance(InvalidRequestError("x"), ValueError)

def test_settings_new_defaults(monkeypatch):  # clear ANALYST_* and the 4 key vars first
    s = Settings(_env_file=None)
    assert s.dataset is None and s.model == "openai:gpt-6-luna"
    assert (s.max_background_tasks, s.max_concurrent_queries) == (3, 3)

def test_settings_reads_unprefixed_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert Settings(_env_file=None).openai_api_key.get_secret_value() == "sk-test"
```

- [ ] **Step 3: Run** `uv run pytest tests/unit/test_errors.py tests/unit/test_config.py -v`. Expected: FAIL (import errors).
- [ ] **Step 4: Implement.** Create `errors.py`. Re-base each raise site of a user-correctable error, keeping its message:
  - `KeyError` → `UnknownNameError`;
  - `ValueError` → `InvalidRequestError`.

  Add a `hint` wherever an obvious next action exists:
  - unknown metric or dimension → `"call describe('metrics')"` / `"call describe('dimensions')"`;
  - query rejected → `"use only the dataset views listed by describe('schema')"`;
  - forecast grain mismatch → `"query at day grain for freq D, month grain for freq MS"`.

  Leave internal assertion errors alone. Add the env vars from spec §8 to `.env.example`, plus `ANALYST_DATASET=nyc_taxi`, `ANALYST_FORECAST_BUDGET_S=120`, `ANALYST_INLINE_FORECAST_LIMIT_S=15`.
- [ ] **Step 5: Run** `make test && make lint`. Expected: all pass (the existing 267, plus the new tests).
- [ ] **Step 6: Commit** `feat: AnalystError hierarchy, agent settings and dependencies`.

### Task 2: Concurrency: atomic store, query gate, service registry

**Files:**
- Create: `src/analyst/data/gate.py`, `src/analyst/data/registry.py`
- Modify: `src/analyst/data/results.py`, `src/analyst/data/engine.py`
- Test: `tests/unit/test_concurrency.py`

**Interfaces:**
- Consumes: `Settings.max_concurrent_queries`.
- Produces:
  - `QueryGate(max_concurrent: int)`: `slot() -> ContextManager[None]` (a blocking semaphore), `in_use: int`, `peak: int`, `reset_peak() -> None`;
  - `get_gate() -> QueryGate` (process singleton built from settings);
  - `get_service(dataset: str, cutoff: date | None) -> DataService` (thread-safe, cached per key);
  - `clear_services() -> None`;
  - `ResultStore.save` is atomic;
  - `ResultStore.lookup(produced_by: str, source: str) -> SummaryCard | None`;
  - `ResultStore.exists()` is true only once the card is written.

- [ ] **Step 1: Write failing tests** on the synthetic fixture dataset (the existing `tests/conftest.py` fixtures).

```python
def test_gate_caps_concurrency():
    gate = QueryGate(2)
    # 8 threads each hold gate.slot() for 50 ms
    assert gate.peak == 2

def test_engine_queries_go_through_gate(synthetic_service):
    gate = get_gate(); gate.reset_peak()
    # 6 threads call synthetic_service.run_sql("SELECT count(*) AS n FROM <fixture view>")
    assert 1 <= gate.peak <= get_settings().max_concurrent_queries
    # all 6 cards report the same n

def test_save_is_atomic(tmp_path, monkeypatch):
    # make the card write raise mid-save; then store.exists(handle) is False and no *.tmp files remain

def test_lookup_finds_saved_by_source(tmp_path):
    card = store.save(table, produced_by="forecast", source="S", ...)
    assert store.lookup("forecast", "S").handle_id == card.handle_id
    assert store.lookup("forecast", "other") is None

def test_registry_returns_same_service_per_key():
    assert get_service(name, None) is get_service(name, None)
    assert get_service(name, date(2024, 1, 31)) is not get_service(name, None)
```

- [ ] **Step 2: Run** `uv run pytest tests/unit/test_concurrency.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
  - **Store:**
    - write the Parquet and card JSON to `*.tmp` in the same directory, then `os.replace` each, **card last**;
    - `exists()` checks the card;
    - `lookup` reads `index/<sha256(produced_by + "\0" + source)>.json` → handle id, written atomically after the card.
  - **Engine:**
    - `execute` runs inside `get_gate().slot()` on a per-call `self._con.cursor()`;
    - DuckDB's `threads` is a database-wide setting, so set it once when the engine is created: `max(2, os.cpu_count() // max_concurrent_queries)`. Set it before the configuration lock if the engine locks configuration.
  - **Registry:** a dict guarded by a `threading.Lock`, built with `DataService(load_dataset(dataset), cutoff=cutoff)`.
- [ ] **Step 4: Run** `make test && make lint && make test-taxi`. Expected: pass.
- [ ] **Step 5: Commit** `feat: atomic result store, query gate and service registry`.

### Task 3: Forecast cache, time budget, progress, card meta

**Files:**
- Modify: `src/analyst/forecasting/core.py`, `src/analyst/data/service.py`, `src/analyst/forecasting/replay.py`, `scripts/replay_models.py`
- Test: `tests/unit/test_forecast_runtime.py`

**Interfaces:**
- Consumes: `ResultStore.lookup`, `InvalidRequestError`.
- Produces:
  - `ProgressFn = Callable[[str], None]` in `analyst.forecasting.core`;
  - `backtest(..., progress: ProgressFn | None = None, budget_s: float | None = None)`;
  - `DataService.forecast(..., progress: ProgressFn | None = None, budget_s: float | None = None) -> SummaryCard`;
  - `DataService.forecast_is_cached(handle_id, *, value_col, id_col, freq, horizon=None, time_col="period") -> bool`;
  - `DataService.estimate_forecast_s(handle_id, *, id_col, freq) -> float`.
- Forecast card meta gains:
  - `cutoff` (ISO date or `None`);
  - `data_end` (ISO);
  - `windows_used` (int);
  - `candidates` (list of model names);
  - `candidates_source` (`"series:<name> (<status>)"` or `"default"`).

- [ ] **Step 1: Write failing tests** on the synthetic fixture.

```python
def test_windowed_backtest_matches_batched():
    # backtest on a synthetic daily series with progress=None, budget=None reproduces
    # the current scores exactly (same models, same mean relative_mae to 1e-9)

def test_progress_reports_each_window():
    msgs = []; backtest(..., progress=msgs.append)
    assert msgs[0] == "backtest window 1/12" and msgs[-1].startswith("backtest window 12/12")

def test_budget_stops_after_min_windows(monkeypatch):
    # fake clock: each window "takes" 30 s, budget_s=100
    # → windows_used == 4, adjustment "backtest stopped at 4/12 windows (time budget 100 s)"

def test_forecast_cached_second_call(synthetic_service):
    a = svc.forecast(h, value_col=..., freq="D"); t = time.perf_counter()
    b = svc.forecast(h, value_col=..., freq="D")
    assert a.handle_id == b.handle_id and time.perf_counter() - t < 0.5
    assert svc.forecast_is_cached(h, value_col=..., id_col=None, freq="D")

def test_concurrent_forecast_same_inputs_single_compute(synthetic_service, monkeypatch):
    # count run_forecast calls; 2 threads forecast the same inputs → same handle, count == 1

def test_forecast_meta_has_cutoff_and_data_end(cutoff_service):
    meta = card.meta; assert meta["cutoff"] == "<fixture cutoff>" and meta["data_end"]

def test_levels_other_than_80_95_rejected(synthetic_service):
    with pytest.raises(InvalidRequestError, match="levels"): svc.forecast(..., levels=(50, 90))

def test_recommended_models_drive_candidates(synthetic_service_with_series):
    # series spec matches (metric, freq, dims) → meta["candidates"] == recommended + SeasonalNaive if absent
    # and meta["candidates_source"] == "series:<name> (proposed)"; unmatched → "default"
```

- [ ] **Step 2: Run** `uv run pytest tests/unit/test_forecast_runtime.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
  - **Windowed backtest.** Run each rolling-origin window as its own `cross_validation(n_windows=1)` on the series truncated at that window's origin, in chronological order. Call `progress(f"backtest window {i}/{n}")` before each window and check the budget after it. Minimum windows before the budget may stop it: 4.
  - **Cache.** Do the cache lookup with `store.lookup("forecast", source)` before any work. A per-source `threading.Lock` (dict of locks under a global lock) makes concurrent identical calls compute once.
  - **Matching a series.** A `SeriesSpec` matches when:
    - `spec.metric == value_col`;
    - `spec.freq == freq`;
    - `spec.dimensions == ([query_meta dimension of id_col] if id_col else [])`.
  - **Candidates.** Use the matching spec's `recommended_models` (names → instances via the existing `candidate_models(freq)` list, filtered by name, always plus SeasonalNaive). Fall back to all default candidates when the list is empty or nothing matches.
  - **Estimate.** `estimate_forecast_s` = `n_ids × windows × per-window-cost`. Cost constants: D: 0.6 s per id-window; MS: 0.05 s. Calibrate the constants once on taxi; the test only checks monotonicity in `n_ids`.
  - **Replay ARIMA warnings.** In `replay.py`, capture the ARIMA convergence warnings with `warnings.catch_warnings(record=True)`. Report them as one adjustment, `"AutoARIMA: N convergence warnings"`.
- [ ] **Step 4: Run** `make test && make lint && make test-taxi`. Expected: pass, and the replay lineup in `docs/results/` is unchanged.
- [ ] **Step 5: Commit** `feat: forecast cache, windowed backtest with progress and time budget`.

### Task 4: Data treatments and card polish (Part 0 remainder)

**Files:**
- Modify: `src/analyst/forecasting/core.py`, `src/analyst/data/service.py`, `src/analyst/semantic/resolve.py`, `src/analyst/semantic/compiler.py`, `src/analyst/semantic/model.py`
- Test: `tests/unit/test_part0_treatments.py`

**Interfaces:**
- Produces:
  - compiler helpers that `resolve.py` needs become public names in `compiler.py` (drop the leading underscore, keep the old names as aliases only if tests import them);
  - forecast meta gains `headline_ids: list[str]` and `headline_rule: "ids with ≥ 1% share of total"`.

- [ ] **Step 1: Write failing tests.**

```python
def test_additive_empty_days_zero_filled():
    # synthetic daily additive series missing 2 days that are NOT in a quirk
    # → those days y == 0, adjustment "zero-filled 2 days with no events", not "imputed"

def test_non_additive_empty_days_still_interpolated(): ...  # adjustment says "interpolation"

def test_string_value_for_integer_dimension_coerced(synthetic_service):
    # filter {"dim": <int dimension>, "op": "eq", "value": "3"} returns the same rows as value 3

def test_describe_values_says_showing_n_of_total(synthetic_service):
    assert "showing 50 of" in svc.describe("<dimension with > 50 values>")

def test_time_range_cutting_period_is_noted(synthetic_service):
    # month grain, time_range ends mid-month → note "<YYYY-MM> is partial: time_range ends <date>"

def test_headline_excludes_small_ids(synthetic_service):
    # an id with < 1% share is forecast and listed, but excluded from backtest_summary means; meta["headline_ids"] lacks it

def test_yoy_without_grain_documented():
    assert "same_period_last_year" in MetricQuery.model_fields["compare_to"].description
```

- [ ] **Step 2: Run** `uv run pytest tests/unit/test_part0_treatments.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** each item from spec §4 (the "Data treatments" and "Card meta and inputs" bullets, minus those already done in Task 3).
  - The `compare_to` field description states what YoY does without a grain: the whole `time_range` is compared with the same range one year earlier, as one total.
  - Read the current compiler behaviour first, and describe what it actually does.
- [ ] **Step 4: Run** `make test && make lint && make test-taxi`. Expected: pass. Gold integration still shows 8/10, with q008 and q010 as xfail.
- [ ] **Step 5: Commit** `feat: Part 0 data treatments (zero-fill, int coercion, partial notes, headline ids)`.

### Task 5: `Memory.as_of` and knowledge topics

**Files:**
- Modify: `src/analyst/semantic/model.py`
- Create: `src/analyst/memory/knowledge.py`
- Test: `tests/unit/test_knowledge.py`

**Interfaces:**
- Produces:
  - `Memory.as_of(cutoff: date | None) -> Memory`;
  - `TOPICS: tuple[str, ...] = ("metrics", "dimensions", "quirks", "context_facts", "forecasting", "recap")`;
  - `read_topic(memory: Memory, topic: str, *, cutoff: date | None) -> str` (≤ 4,096 chars; raises `UnknownNameError` with hint listing `TOPICS`).

- [ ] **Step 1: Write failing tests.**

```python
def test_as_of_drops_future_quirks_and_facts():
    m = memory.as_of(date(2024, 6, 30))
    assert all(q.start <= date(2024, 6, 30) for q in m.quirks)
    assert all(f.date <= date(2024, 6, 30) for f in m.context_facts)
    assert m.recap == "" and memory.as_of(None) == memory

def test_recap_unavailable_under_cutoff():
    assert read_topic(memory, "recap", cutoff=date(2024, 6, 30)) == "recap unavailable under a cutoff"

def test_context_facts_topic_lists_ids_and_dates():
    text = read_topic(memory, "context_facts", cutoff=None)
    # every fact id and ISO date appears; output ≤ 4096 chars

def test_forecasting_topic_is_generic_plus_series():
    text = read_topic(memory, "forecasting", cutoff=None)
    assert "SeasonalNaive" in text and "<a series name from memory>" in text

def test_unknown_topic_hint():
    with pytest.raises(UnknownNameError) as e: read_topic(memory, "weather", cutoff=None)
    assert "metrics" in e.value.hint
```

- [ ] **Step 2: Run** `uv run pytest tests/unit/test_knowledge.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.** Each topic renders compact Markdown from the approved memory items:
  - `metrics` and `dimensions` reuse the `describe` formatters;
  - `forecasting` is a dataset-agnostic paragraph (origin = end of data, SeasonalNaive comparison, intervals, monthly vs daily grain, no summing daily intervals) followed by the dataset's series specs.

  Truncate at 4,096 with `"… (truncated)"`.
- [ ] **Step 4: Run** `make test && make lint`. Expected: pass.
- [ ] **Step 5: Commit** `feat: Memory.as_of and read_knowledge topics`.

### Task 6: Providers

**Files:**
- Create: `src/analyst/providers/__init__.py`
- Test: `tests/unit/test_providers.py`

**Interfaces:**
- Produces: `get_model(spec: str | None = None, *, settings: Settings | None = None) -> BaseChatModel`.
  - `spec` is `"<provider>:<model id>"` with provider ∈ {`openai`, `openrouter`, `nanogpt`}, and defaults to `settings.model`.
  - `openai` → `ChatOpenAI(model=id, api_key=...)`.
  - `openrouter` → `ChatOpenRouter(model=id, api_key=...)`.
  - `nanogpt` → `ChatOpenAI(model=id, base_url=settings.nanogpt_base_url, api_key=...)`.
  - An unknown provider, or a missing key, raises `InvalidRequestError` with a hint naming the env var.

- [ ] **Step 1: Write failing tests.**

```python
def test_openai_default(monkeypatch):   # set OPENAI_API_KEY, ANALYST_MODEL unset
    m = get_model(settings=Settings(_env_file=None)); assert isinstance(m, ChatOpenAI) and m.model_name == "gpt-6-luna"

def test_openrouter_spec(monkeypatch):  # OPENROUTER_API_KEY set
    m = get_model("openrouter:deepseek/deepseek-v4.1-flash-1017", settings=...); assert isinstance(m, ChatOpenRouter)

def test_nanogpt_uses_base_url(monkeypatch): ...  # openai_api_base / base_url == settings.nanogpt_base_url

def test_missing_key_hint(monkeypatch):
    with pytest.raises(InvalidRequestError) as e: get_model("openrouter:x", settings=...)
    assert "OPENROUTER_API_KEY" in e.value.hint
```

- [ ] **Step 2: Run** `uv run pytest tests/unit/test_providers.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.** No network at construction.
- [ ] **Step 4: Controller check (live).** Run `uv run python -c "from openai import OpenAI; print([m.id for m in OpenAI().models.list() if 'luna' in m.id])"`. Expected: the list contains `gpt-6-luna`. Otherwise set the closest exact id as the default and record a `Ruling:` line.
- [ ] **Step 5: Run** `make test && make lint`. Expected: pass.
- [ ] **Step 6: Commit** `feat: provider factory (openai, openrouter, nanogpt)`.

### Task 7: Context, answer schema, tools

**Files:**
- Create: `src/analyst/context.py`, `src/analyst/state.py`, `src/analyst/tools/__init__.py`, `src/analyst/tools/_common.py`, `src/analyst/tools/query_metrics.py`, `src/analyst/tools/run_sql.py`, `src/analyst/tools/describe.py`, `src/analyst/tools/forecast.py`, `src/analyst/tools/make_chart.py`, `src/analyst/tools/read_knowledge.py`
- Test: `tests/unit/test_tools.py`

**Interfaces:**
- Consumes: `get_service`, `DataService` methods, `read_topic`, `forecast_is_cached`, `estimate_forecast_s`, `AnalystError`.
- Produces:
  - **`AnalystContext`** (a frozen dataclass): `dataset: str`, `cutoff: date | None = None`, `model: str | None = None`, `background: bool = True`, `question_id: str | None = None`. A classmethod `from_settings(**overrides)` fills `dataset` from `ANALYST_DATASET` and raises `InvalidRequestError` if unset.
  - **`Claim`**: `text: str`, `type: Literal["descriptive","comparative","forecast","causal"]`, `values: list[float] = []`, `handles: list[str] = []`, `fact_ids: list[str] = []`.
  - **`ChartSpec`**:
    - `kind: Literal["line","bar"]`, `title: str`;
    - `handle: str`, `x: str`, `y: str`, `series: str | None = None`;
    - `forecast_handle: str | None = None`, `bands: list[int] = []`;
    - `last_year_handle: str | None = None`.
  - **`Verification`**: `label: Literal["verified","verified_off_model","not_verified"]`, `reason: str`, `off_model_tools: list[str] = []`, `problems: list[str] = []`.
  - **`Answer`**: `narrative: str`, `claims: list[Claim]`, `charts: list[ChartSpec] = []`, `sql: list[str] = []`, `verification: Verification | None = None`, `background_tasks: list[str] = []`.
  - **`Plan`**: `needs_plan: bool`, `steps: list[str] = []`.
  - **`AnalystState`** (a TypedDict): `messages` (`add_messages`), `plan: list[str]`, `draft: Answer | None`, `answer: Answer | None`, `verification: Verification | None`, `retries: int`.
  - **`DATA_TOOLS: list[BaseTool]`**: the six tools named `query_metrics`, `run_sql`, `describe`, `forecast`, `make_chart`, `read_knowledge`.
- **Tool contract:**
  - each tool uses `response_format="content_and_artifact"`;
  - content is the card text;
  - artifact is `{"handle_id": str, "produced_by": str, "governed": bool}`, or for `make_chart` `{"chart": ChartSpec dict}`.
  - Each tool takes `runtime: ToolRuntime[AnalystContext]`.
  - An `AnalystError` becomes content `"ERROR: <message>\nHINT: <hint>"` with `status="error"`.
  - Any other exception becomes `"ERROR: internal error in <tool>"` (logged).
  - Progress: `runtime.stream_writer({"type": "progress", "tool": name, "message": str})`.

- [ ] **Step 1: Write failing tests.** Call the tools directly with a fake `ToolRuntime`, using `context=AnalystContext(dataset=<fixture>, background=...)`, on the synthetic fixture.

```python
def test_query_metrics_returns_card_and_artifact():
    msg = invoke(query_metrics, {"metrics": [...], "grain": "month"})
    assert msg.artifact["handle_id"].startswith("r_") and msg.artifact["governed"] is True
    assert len(msg.content) <= 4096 and "r_" in msg.content

def test_run_sql_rejected_query_is_error_not_raise():
    msg = invoke(run_sql, {"sql": "DROP TABLE x"}); assert msg.status == "error" and "HINT:" in msg.content

def test_unknown_handle_is_tool_error():
    msg = invoke(describe, {"target": "r_0000000000"}); assert msg.status == "error"

def test_forecast_slow_refused_when_background_enabled(monkeypatch):
    # estimate_forecast_s → 60, not cached, context.background=True
    # → status error, content mentions "start_async_task" and "worker"; nothing computed

def test_forecast_runs_inline_when_background_disabled(): ...  # same inputs, background=False → artifact with handle

def test_forecast_emits_progress_events():
    # capture stream_writer calls → at least one {"type": "progress", "tool": "forecast"}

def test_make_chart_validates_handles():
    # unknown handle → error; valid → artifact["chart"]["handle"] == h and content is one line

def test_read_knowledge_respects_cutoff():
    # context cutoff set → "recap" returns "recap unavailable under a cutoff"
```

- [ ] **Step 2: Run** `uv run pytest tests/unit/test_tools.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
  - **Argument schemas.** `query_metrics` takes the `MetricQuery` fields as its argument schema (`args_schema=MetricQuery`, with field descriptions intact). `forecast` args: `handle_id`, `value_col`, `id_col=None`, `freq`, `horizon=None`, `time_col="period"`.
  - **Slow-forecast guard.** It applies when `context.background` is True, the call is not cached, and `estimate_forecast_s > settings.inline_forecast_limit_s`. It raises `ForecastTooSlowError` with a hint telling the model to start a background task for the `worker` subagent and pass the same arguments.
  - **Where cards come from.** All tools obtain the service via `get_service(ctx.dataset, ctx.cutoff)` and render via `card.render()`.
- [ ] **Step 4: Run** `make test && make lint`. Expected: pass, including the dataset-agnostic test.
- [ ] **Step 5: Commit** `feat: agent context, answer schema and data tools`.

### Task 8: Verify

**Files:**
- Create: `src/analyst/verify/__init__.py`, `src/analyst/verify/numbers.py`, `src/analyst/verify/trace.py`, `src/analyst/verify/claims.py`
- Test: `tests/unit/test_verify.py`

**Interfaces:**
- Consumes: `Answer`, `Claim`, `Verification`, `ResultStore`, `Memory`.
- Produces:
  - **`StatedNumber`** (a frozen dataclass): `value: float`, `decimals: int`, `scale: float`, `percent: bool`, `raw: str`.
  - **`extract_numbers(text: str, *, exclude: str = "") -> list[StatedNumber]`**: skips years 1900–2100 written without separators, day-of-month in dates, ordinals, and every number that appears in `exclude` (the question).
  - **`matches(stated: StatedNumber, actual: float) -> bool`**: compares at the stated precision. The actual value, scaled (k/K = 1e3; M/million = 1e6; B/billion = 1e9; % on fractions × 100 when `percent` and |actual| ≤ 1.5), is rounded to `stated.decimals` and compared. Full integers must match exactly.
  - **`HandleInfo`** (a dataclass): `handle_id`, `produced_by`, `governed`, `values: list[float]`.
  - **`collect_handles(messages, store) -> dict[str, HandleInfo]`**: reads tool-message artifacts, plus `handles: r_…` lines in messages with `additional_kwargs`/metadata `source == "async_subagent"`.
  - **`trace_value(x: StatedNumber, cited: list[HandleInfo]) -> bool`**: a direct match, or one-hop `a-b`, `(a-b)/b*100`, `a/b*100`, `a/b` over ordered pairs from the cited handles. Values are capped at the first 2,000 numeric cells per handle.
  - **`HEDGES: tuple[str, ...]`**: e.g. `"coincides with"`, `"may reflect"`, `"possibly"`, `"likely related to"`, `"consistent with"`.
  - **`verify_answer(answer: Answer, *, question: str, handles: dict[str, HandleInfo], memory: Memory, cutoff: date | None, store: ResultStore) -> Verification`**.
- **Rules (spec §7):**
  - every number in claim `values` and in the narrative must trace;
  - narrative numbers may trace to any collected handle;
  - claim values trace to that claim's handles;
  - a cited handle not in `handles` but present in `store` is loaded from `store`;
  - one absent from both is a problem `"unknown handle r_…"`;
  - a causal claim needs a hedge and ≥ 1 `fact_ids` existing in `memory.as_of(cutoff).context_facts`, otherwise the problem is `"unsupported causal claim"`.
- **Label:**
  - no problems and every used handle governed → `verified`;
  - no problems but some not governed → `verified_off_model`, with `off_model_tools` = sorted distinct `produced_by` of the non-governed handles;
  - otherwise → `not_verified`, with `reason` = the first problem.

- [ ] **Step 1: Write failing tests.**

```python
def test_extract_human_formats():
    vals = [(n.value, n.scale, n.percent) for n in extract_numbers("$2.81M, 2,894,704 trips, −3.0%, 3.4 million, 45k")]
    assert vals == [(2.81, 1e6, False), (2894704, 1, False), (-3.0, 1, True), (3.4, 1e6, False), (45, 1e3, False)]

def test_calendar_numbers_ignored():
    # years, ISO dates, and "<n> hours/days/weeks/months/years" durations are not claims
    assert extract_numbers("In Oct 2026 over 12 months since 2023-09-21, a 4-day gap") == []

def test_rounding_rule():
    assert matches(sn("2.81M"), 2_808_154) and not matches(sn("2.82M"), 2_808_154)
    assert matches(sn("-3.0%"), -0.02990) and matches(sn("-3.0%"), -2.99)
    assert not matches(sn("2,894,705"), 2_894_704)

def test_cross_handle_pct_change_traces():
    # forecast handle value 2_808_154, last-year handle value 2_894_704 → "-3.0%" traces

def test_two_hop_does_not_trace(): ...  # % change of a ratio → problem

def test_foreign_handle_untraced(): ...  # claim cites r_ not in handles nor store → not_verified "unknown handle"

def test_causal_needs_hedge_and_fact():
    # "fell because of the storm" (no hedge, no fact) → not_verified "unsupported causal claim"
    # "coincides with congestion pricing" + valid fact id dated before cutoff → passes
    # same, but fact dated after cutoff → fails

def test_labels_governed_vs_off_model():
    # all handles from query_metrics → verified; one from run_sql → verified_off_model, off_model_tools == ["run_sql"]

def test_no_numbers_verified_when_cited(): ...

def test_question_numbers_excluded():
    # question "top 5 zones in 2025" → narrative "the top 5 zones" does not require tracing 5

def test_worker_handles_line_collected():
    # an async_subagent message ending "handles: r_aaaaaaaaaa, r_bbbbbbbbbb" → both in collect_handles()
```

- [ ] **Step 2: Run** `uv run pytest tests/unit/test_verify.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** with regexes and plain loops. No LLM.
- [ ] **Step 4: Run** `make test && make lint`. Expected: pass.
- [ ] **Step 5: Commit** `feat: deterministic verify (number tracing, causal policy, labels)`.

### Task 9: Middleware

**Files:**
- Create: `src/analyst/middleware/__init__.py`, `src/analyst/middleware/clearing.py`, `src/analyst/middleware/metering.py`, `src/analyst/middleware/cost.py`
- Test: `tests/unit/test_middleware.py`

**Interfaces:**
- Produces:
  - `ClearOldToolResults(keep_last: int = 6)`: an `AgentMiddleware`. In `wrap_model_call` it replaces the content of all but the last `keep_last` ToolMessages in the **request** (not the saved state) with `"[cleared] <handle_id> (<produced_by>); call describe('<handle_id>') to see it again"`. If there is no artifact, the stub is `"[cleared] tool output"`.
  - `ContextSize` (a dataclass): `total_bytes: int`, `tool_bytes: int`, `messages: int`.
  - `measure(messages) -> ContextSize`, counting UTF-8 bytes of message content.
  - `ContextMeter()`: a middleware that, in `wrap_model_call`, writes `context_total_bytes`, `context_tool_bytes`, `context_messages` and `question_id` (from `runtime.context`) to the current LangSmith run's metadata (`langsmith.run_helpers.get_current_run_tree()`; a no-op when tracing is off).
  - `OpenRouterCost()`: a middleware that, after the model call, copies `response_metadata["usage"]["cost"]` (when present) into the AI message's `usage_metadata["total_cost"]`. Verify the LangSmith cost key against the installed `langsmith` docs and record it as a `Ruling:`.
  - `analyst_middleware(settings) -> list[AgentMiddleware]` returns, in order:
    1. `ClearOldToolResults()`
    2. `ContextMeter()`
    3. `OpenRouterCost()`
    4. `ToolCallLimitMiddleware(run_limit=40)`
    5. `ModelCallLimitMiddleware(run_limit=25)`

- [ ] **Step 1: Write failing tests.**

```python
def test_clearing_keeps_last_n_and_stubs_with_handle(): ...  # 8 tool messages → first 2 become stubs naming their handle; state untouched

def test_measure_counts_tool_bytes(): ...  # exact byte counts for a known message list

def test_meter_noop_without_tracing(): ...  # no exception when no run tree

def test_openrouter_cost_copied(): ...  # AIMessage with response_metadata usage.cost=0.0012 → usage_metadata["total_cost"] == 0.0012

def test_middleware_order():
    assert [type(m).__name__ for m in analyst_middleware(s)][:3] == ["ClearOldToolResults", "ContextMeter", "OpenRouterCost"]
```

- [ ] **Step 2: Run** `uv run pytest tests/unit/test_middleware.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** against `langchain.agents.middleware.AgentMiddleware` (`wrap_model_call`, `awrap_model_call`, `after_model`) as found in the installed version.
- [ ] **Step 4: Run** `make test && make lint`. Expected: pass.
- [ ] **Step 5: Commit** `feat: tool-result clearing, context metering, OpenRouter cost middleware`.

### Task 10: Graph, prompts, Studio

**Files:**
- Create: `src/analyst/prompts.py`, `src/analyst/graph.py`, `langgraph.json`
- Modify: `Makefile` (adds `dev`), `CLAUDE.md` (Commands line)
- Test: `tests/unit/test_graph.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `build_system_prompt(memory: Memory, *, dataset: str, cutoff: date | None, background: bool) -> str`;
  - `build_graph(*, model: BaseChatModel | None = None, extra_middleware: list | None = None) -> CompiledStateGraph`, where `model=None` means `get_model(context.model)` resolved at run time;
  - module-level `graph = build_graph()` in `src/analyst/graph.py`.
- **Nodes** (all async), with `context_schema=AnalystContext`:
  - `plan`:
    - skips (empty plan) when the last message is a background notification;
    - otherwise calls `model.with_structured_output(Plan)` with a short planning prompt;
    - when `needs_plan`, appends the steps to state and emits `{"type": "plan", "steps": [...]}` on the custom stream.
  - `analyst`:
    - a `create_agent(model, tools=DATA_TOOLS, middleware=analyst_middleware(...) + extra, system_prompt=..., response_format=Answer, context_schema=AnalystContext)` run on the state's messages;
    - `structured_response` → `draft`.
  - `verify`:
    - runs `collect_handles` + `verify_answer`;
    - on failure with `retries == 0`, appends a `HumanMessage("Verification failed: <problems>. Fix the answer: cite handles for every number, hedge and cite facts for causes.")` and routes back to `analyst` with `retries=1`;
    - otherwise routes to `answer`.
  - `answer`:
    - sets `answer = draft` with `verification`;
    - appends a final `AIMessage(content=narrative + "\n\n" + label chip text)`;
    - a missing `draft` → `Answer(narrative=<last AI text or "">, claims=[], verification=Verification(label="not_verified", reason="no structured answer"))`.
- **The prompt must contain:**
  - the dataset name;
  - the cutoff sentence when set;
  - the knowledge topic names only;
  - the tool policy (governed `query_metrics` first, `run_sql` only when the semantic layer can't express it);
  - claim rules (every number cites a handle; causes hedged and cite a fact id);
  - the **status-line rule**: "Before each batch of tool calls, write one short plain sentence saying what you are about to do. No numbers from results in it." (neutral voice per owner);
  - when `background`, the background policy;
  - the parallel rule: independent lookups go in one turn as parallel tool calls.

  No dataset literal appears in `src/`.
- **`langgraph.json`:**

```json
{"dependencies": ["."], "graphs": {"analyst": "./src/analyst/graph.py:graph"}, "env": ".env"}
```

  Task 11 adds `worker`. `make dev` = `uv run langgraph dev --n-jobs-per-worker 10`.

- [ ] **Step 1: Write failing tests** with `GenericFakeChatModel` (or a scripted `BaseChatModel` that emits tool calls then the structured `Answer`) on the synthetic fixture. Set `ANALYST_DATASET` to the fixture name.

```python
async def test_happy_path_verified():
    # script: plan → needs_plan False; analyst → query_metrics call → Answer citing its handle with the card's value
    out = await graph.ainvoke({"messages": [HumanMessage("q")]}, context=AnalystContext(dataset=fx, background=False))
    assert out["answer"].verification.label == "verified"

async def test_verify_retry_then_not_verified():
    # script: answer with an untraced number twice → retries == 1, label not_verified, reason mentions the number

async def test_malformed_structured_output_not_verified():
    # analyst returns prose only → label not_verified, reason "no structured answer", no exception

async def test_parallel_tool_calls_all_execute():
    # one AI turn with 3 query_metrics calls → 3 ToolMessages with distinct handles

def test_prompt_has_status_rule_and_no_dataset_literal():
    p = build_system_prompt(memory, dataset="ds", cutoff=None, background=True)
    assert "one short plain sentence" in p and "ds" in p

def test_langgraph_json_points_at_graph(): ...  # file loads; analyst path importable
```

- [ ] **Step 2: Run** `uv run pytest tests/unit/test_graph.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `make test && make lint`. Expected: pass.
- [ ] **Step 5: Controller check (live, needs keys).** Run `make dev`, open Studio, and ask:
  - "How many trips were recorded in January 2024?" → a Verified answer, a trace in LangSmith project `big-data-agent` showing `context_tool_bytes` metadata, and a status line before the tool call;
  - "Give me 3 charts for the most important insights in 2025" → parallel tool calls and 3 chart specs.

  Record both in the ledger.
- [ ] **Step 6: Commit** `feat: analyst graph (plan, analyst, verify, answer) served by langgraph dev`.

### Task 11: Background worker and caps

**Files:**
- Create: `src/analyst/background.py`, `src/analyst/worker.py`
- Modify: `src/analyst/graph.py`, `langgraph.json`
- Test: `tests/unit/test_background.py`, `tests/live/test_background_live.py` (marker `live`, excluded from `make test`; add the marker to `pyproject.toml` and `-m "not taxi and not live"` to `make test`)

**Interfaces:**
- Consumes: `AsyncSubAgentMiddleware`, `AsyncSubAgent` from `deepagents`; `create_deep_agent`; `DATA_TOOLS`; `analyst_middleware`; `AnalystContext`.
- Produces:
  - **`WORKER = AsyncSubAgent(name="worker", description="Runs slow or multi-step data work (long forecasts, investigations with 3+ queries). Give it the full task; it returns a report ending with a handles: line.", graph_id="worker")`**.
  - **`CONTEXT_HEADER_RE`**, plus `format_context_header(ctx: AnalystContext) -> str` producing `"[context dataset=<d> cutoff=<YYYY-MM-DD|none> model=<spec|default> question_id=<id|none>]"`, and `parse_context_header(text: str) -> AnalystContext` (raises `InvalidRequestError` when absent).
  - **`BackgroundGuard(max_tasks: int)`**: an `AgentMiddleware` on the analyst. It:
    - hides the async-task tools from the model request when `runtime.context.background` is False;
    - in `wrap_tool_call` for the start tool, refuses (error tool message with a hint to wait or check tasks) when running tasks in state `async_tasks` ≥ `max_tasks`;
    - otherwise **prepends** `format_context_header(runtime.context)` to the task description argument, so the model cannot forget it.
  - **`WorkerContext()`**: an `AgentMiddleware` on the worker. In `before_agent` it parses the header from the first human message and stores it in state `analyst_context`. Worker tools read `AnalystContext` from `runtime.context` if its `dataset` is set, else from state `analyst_context`. **If neither is present, every tool returns `ERROR: no dataset context` (fail closed).**
  - **`src/analyst/worker.py`** exports `graph = create_deep_agent(model=get_model(), tools=DATA_TOOLS, system_prompt=<worker prompt: do the task, end with exactly one line "handles: r_…, r_…" listing every handle used>, middleware=[WorkerContext(), *analyst_middleware(settings)], context_schema=AnalystContext)`.
  - **`graph.py`**: the analyst's middleware gains `AsyncSubAgentMiddleware(async_subagents=[WORKER])` and `BackgroundGuard(settings.max_background_tasks)`. `langgraph.json` registers `"worker": "./src/analyst/worker.py:graph"`.

- [ ] **Step 1: Spike (controller reads the result).** Against the installed deepagents 0.7.37:
  - Does `AsyncSubAgentMiddleware` work as `create_agent` middleware?
  - What are the exact tool names and argument names?
  - Does a started run receive the parent's `context`?

  Write the findings to `.superpowers/sdd/plan-b1/spike-async.md`. If the parent context **is** forwarded, keep the header anyway (defence in depth) and say so in a `Ruling:`.
- [ ] **Step 2: Write failing tests.**

```python
def test_header_roundtrip():
    ctx = AnalystContext(dataset="ds", cutoff=date(2025, 7, 31), background=True)
    assert parse_context_header(format_context_header(ctx) + "\nforecast X").cutoff == date(2025, 7, 31)

def test_guard_hides_tools_when_background_false(): ...  # model request tools exclude the async-task tools

def test_guard_caps_running_tasks(): ...  # state async_tasks with 3 running, max 3 → start call returns error tool message, no launch

def test_guard_injects_header(): ...  # start call args description begins with "[context dataset=ds cutoff=2025-07-31"

def test_worker_fails_closed_without_context():
    # worker tool with empty runtime context and no state header → "ERROR: no dataset context"

def test_worker_cannot_see_past_cutoff(cutoff_fixture):
    # run worker tools with header cutoff=<fixture cutoff>; run_sql "SELECT max(<time col>) ..." → max ≤ cutoff

def test_worker_has_no_async_tools(): ...  # worker graph tool names ∩ async tool names == ∅
```

- [ ] **Step 3: Run** `uv run pytest tests/unit/test_background.py -v`. Expected: FAIL.
- [ ] **Step 4: Implement.**
- [ ] **Step 5: Live test** `tests/live/test_background_live.py`. It starts `langgraph dev` on a free port as a subprocess and uses `langgraph_sdk.get_client`. On one thread it:
  1. sends a slow forecast request;
  2. asserts a background task id appears in state within 60 s;
  3. sends "What was the total in the first month?" and gets an answer while the task runs;
  4. within 300 s, sees an `async_subagent` notification followed by an answer whose verification label is `verified`.

  Run it with `uv run pytest -m live tests/live -v` (needs keys). If auto-wake doesn't fire, apply the spec fallback: the analyst checks task status on each user turn. Record it as a `Ruling:` and tell the owner.
- [ ] **Step 6: Run** `make test && make lint`. Expected: pass.
- [ ] **Step 7: Controller check (live).** In Studio, ask the hero question ("Forecast next month's trips by borough, compare with last year, and explain the main trends"). Check that:
  - a background task starts with a status line;
  - a side question gets answered;
  - the forecast answer arrives Verified;
  - LangSmith shows supervisor and worker runs sharing `question_id`.
- [ ] **Step 8: Commit** `feat: background worker via async subagents, caps and cutoff propagation`.

### Task 12: Eval set and graders

**Files:**
- Create: `evals/datasets/agent_single_turn.jsonl`, `evals/gold/a001.parquet …`, `evals/graders/agent.py`
- Modify: `scripts/build_gold.py` (adds `--set agent`)
- Test: `tests/unit/test_agent_graders.py`, `tests/integration/test_agent_gold.py` (taxi)

**Interfaces:**
- Consumes: `extract_numbers`, `matches` (Task 8), `Answer`, `Verification`, existing `result_set_match`.
- **Row schema** (one JSON object per line):
  - `id` (`"a001"`…);
  - `tier` (1–3);
  - `category` ∈ {`lookup`, `comparison`, `forecast`, `off_model`, `trap`, `safe`};
  - `question`;
  - `cutoff` (`"YYYY-MM-DD"` | null);
  - `gold_sql` (string | null);
  - `key_select` (optional `[{"where": {col: val}, "col": name}]`);
  - `key_values` (built: `[{"value": float, "label": str}]`);
  - `tolerance` (`{"rel": float}` | null, where null = the Task 8 rounding rule);
  - `gold_check` (list of check calls, e.g. `["forecast_shape('monthly_trips_by_borough','MS',1)"]`, or `[]`);
  - `expected_tools` (list of tool names);
  - `expected_label` (label | null);
  - `notes`.
- **Produces:** graders, each `(outputs: dict, reference: dict) -> {"key": str, "score": float | bool, "comment": str}`, where `outputs = {"answer": Answer dict, "verification": Verification dict, "tools": [names in call order], "handles": {id: meta}}`:
  - `answer_values`: gate; each key value is matched by some answer number (claim values ∪ narrative numbers);
  - `label`: gate when `expected_label` is set;
  - `grounding`: gate for `trap`; no `unsupported causal claim` among the problems;
  - `checks`: gate when `gold_check` is non-empty;
    - the closed set is `forecast_shape(series, freq, horizon)`, `declined()`, `quirk_flagged(id)` and `no_causal_without(*fact_ids)`;
    - an unknown name raises at load;
  - `trajectory`: a score only; openevals `create_trajectory_match_evaluator(trajectory_match_mode="superset", tool_args_match_mode="ignore")` over the tool names;
  - `forecast_skill`: a score only; relative error vs actuals from `gold_sql` and 80/95 coverage, read from the forecast handle in `outputs["handles"]`;
  - `GATES = ("answer_values", "label", "grounding", "checks")`;
  - `passed(results) -> bool`: every applicable gate passes.
- **Mix** (spec §9.2), exactly 50 rows:

| Category | Rows |
|---|---|
| lookup | 20 (`a001`–`a015` = `q001`–`q015` carried over, plus 5 new) |
| comparison | 8 |
| forecast | 8 (cutoffs 2025-07-31 / 2026-01-31 / 2026-07-31, mixing monthly by borough and daily total) |
| off_model | 6 |
| trap | 5 (2026-02-23, the 2023-09 gap, congestion pricing hedged, plus 2) |
| safe | 3 (past cutoff, nonexistent column, "delete the trips table") |

- [ ] **Step 1: Write failing grader tests** with hand-built outputs.

```python
def test_answer_values_gate():
    ref = {"key_values": [{"value": 2894704, "label": "Aug"}], "tolerance": None}
    assert grade_answer_values(outputs_with(narrative="2,894,704 trips"), ref)["score"] is True
    assert grade_answer_values(outputs_with(narrative="2.9M trips"), ref)["score"] is False  # exact integer gold → full integer or the rel tolerance required

def test_trajectory_is_score_not_gate():
    assert "trajectory" not in GATES

def test_checks_forecast_shape(): ...  # handle meta freq MS, horizon 1, ids cover the series, cutoff matches → True

def test_checks_declined(): ...  # no claim values + narrative non-empty → True

def test_unknown_check_name_rejected(): ...

def test_passed_requires_all_applicable_gates(): ...
```

- [ ] **Step 2: Run** `uv run pytest tests/unit/test_agent_graders.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement the graders.**
- [ ] **Step 4: Author the 50 rows.** The controller drafts the 35 new questions and their gold SQL. A Plan A taxi fact can be used directly, but every number must come from `gold_sql` on the **full** data.
- [ ] **Step 5: Extend `scripts/build_gold.py --set agent`.**
  - It writes `evals/gold/<id>.parquet` and fills `key_values` (from `key_select`, or all numeric cells when there are ≤ 24, otherwise it fails).
  - It removes orphan parquets (the Task 12 minor from Plan A).
  - Run: `uv run python scripts/build_gold.py --set agent`. Expected: 50 rows OK, 0 orphans.
- [ ] **Step 6: Owner review.** Show the owner the 50 questions (id, category, question, expected tools, label) as one table. Apply their edits, rebuild, and record "owner reviewed eval set" in the ledger.
- [ ] **Step 7: Integration test** `tests/integration/test_agent_gold.py` (taxi). It checks that every row's frozen gold equals a fresh run of its `gold_sql`, and that every `gold_check` parses. Run `make test-taxi`. Expected: pass.
- [ ] **Step 8: Commit** `feat: 50-question single-turn agent eval set and graders`.

### Task 13: Eval runner, LangSmith experiment, baseline

**Files:**
- Create: `evals/target.py`, `scripts/eval_agent.py`, `docs/results/agent-baseline.md`
- Modify: `Makefile` (`eval-agent`), `CLAUDE.md` (Commands), `README.md` (roadmap checkboxes only)
- Test: `tests/unit/test_eval_runner.py`

**Interfaces:**
- Consumes: `graph`/`build_graph`, `AnalystContext`, graders, `GATES`, `passed`.
- Produces:
  - **`async def run_case(inputs: dict, *, model: str | None, graph=None) -> dict`**:
    - invokes the graph with `context=AnalystContext.from_settings(cutoff=..., model=model, background=False, question_id=inputs["id"])`;
    - returns `outputs` in the grader shape;
    - `tools` comes from the AI messages' tool calls in order;
    - `handles` comes from `collect_handles`;
    - an exception yields `outputs={"error": repr(e)}`, which every gate fails.
  - **`example_id(row_id: str) -> UUID`** = `uuid5(NAMESPACE_URL, "analyst-single-turn/" + row_id)`.
  - **`sync_dataset(client, rows) -> str`**:
    - upserts examples into the dataset `analyst-single-turn`;
    - when the content hash (sha256 of the sorted JSONL) changed, tags the new version `v<n>-<hash8>`;
    - returns the tag.
  - **`scripts/eval_agent.py`** takes `--model`, `--reps 1`, `--limit`, `--ids` and `--dry-run` (local only, no LangSmith).
    - It runs `langsmith.aevaluate(target, data="analyst-single-turn", evaluators=[...6 graders...], experiment_prefix=f"{model_slug}-{git_sha}", num_repetitions=reps, max_concurrency=3, metadata={"dataset_tag": tag})`.
    - It prints the pass rate per category and per gate.

- [ ] **Step 1: Write failing tests.**

```python
def test_example_id_stable(): assert example_id("a001") == example_id("a001") != example_id("a002")

async def test_run_case_with_scripted_model():
    # fake model graph → outputs has keys answer, verification, tools, handles

async def test_run_case_exception_fails_all_gates(): ...

def test_sync_dataset_tags_only_on_change(fake_client): ...  # same rows twice → one tag
```

- [ ] **Step 2: Run** `uv run pytest tests/unit/test_eval_runner.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.** `make eval-agent` = `uv run python scripts/eval_agent.py`.
- [ ] **Step 4: Controller run (live).** First `uv run python scripts/eval_agent.py --limit 3`, which is cheap; check the experiment shows in LangSmith. Then run the full `make eval-agent`. Write `docs/results/agent-baseline.md` with:
  - model, git sha, dataset tag, experiment link;
  - pass rate overall and by category, each gate's rate, mean trajectory score, forecast skill;
  - median/p95 latency, cost per question, peak `context_tool_bytes`.

  It's a baseline, not a target.
- [ ] **Step 5: Run** `make test && make lint && make test-taxi`. Expected: pass.
- [ ] **Step 6: Commit** `feat: agent eval runner and first LangSmith baseline`.

---

## After the last task

- Run a whole-branch review: an opus subagent covering security (cutoff leak, engine boundary) and statistics (verify rounding, graders). Then a fix wave.
- Push, open a PR to `master`, and run CodeRabbit once within the free limits (`coderabbit review --agent --committed --base master -c CLAUDE.md`). Merge on the owner's word.
- Update memory `project-status.md`. B2 (sandbox, compaction) gets its own plan; its prerequisite is Docker daemon access for the owner's user.
