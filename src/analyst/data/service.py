"""Facade tying Engine, semantic memory, and result handles together."""

from __future__ import annotations

import json
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Literal

import pandas as pd
import pyarrow as pa

from analyst.config import get_settings
from analyst.data.dataset import DatasetConfig
from analyst.data.engine import Coverage, Engine
from analyst.data.gate import get_gate
from analyst.data.results import ResultStore, SummaryCard, is_result_handle_id
from analyst.errors import InvalidRequestError, UnknownNameError
from analyst.forecasting.core import ProgressFn, prepare_series
from analyst.forecasting.core import forecast as run_forecast
from analyst.forecasting.models import DEFAULTS, candidate_models
from analyst.memory.files import load_memory
from analyst.memory.knowledge import describe_dimensions, describe_metrics
from analyst.semantic.compiler import (
    FilterSpec,
    MetricQuery,
    _is_additive,
    compile_query,
    from_sql,
)
from analyst.semantic.model import Memory, SemanticModel
from analyst.semantic.resolve import resolve_values

_FORECAST_COLUMNS = (
    "unique_id",
    "ds",
    "yhat",
    "lo_80",
    "hi_80",
    "lo_95",
    "hi_95",
)
_SCORE_FIELDS = (
    "mae",
    "mase",
    "relative_mae",
    "coverage_80",
    "coverage_95",
    "scaled_crps",
)
_BASELINE_NOTE = "no model beat the seasonal baseline"
_BACKTEST_SERIES_LIMIT = 5
_FORECAST_LEVELS = (80, 95)
_VALUE_LIMIT = 50
# Same id-share rule as analyst.forecasting.replay.min_share.
_MIN_HEADLINE_SHARE = 0.01
_HEADLINE_RULE = "ids with ≥ 1% share of total"
# Seconds per (series id x backtest window); rough single-core estimates.
_WINDOW_COST_S = {"D": 0.6, "MS": 0.05}

_forecast_locks: dict[str, threading.Lock] = {}
_forecast_locks_guard = threading.Lock()


def _forecast_lock(key: str) -> threading.Lock:
    with _forecast_locks_guard:
        return _forecast_locks.setdefault(key, threading.Lock())

_DESCRIBE_TARGETS = (
    "metrics",
    "dimensions",
    "dimension:<name>",
    "schema",
    "coverage",
    "<handle_id>",
)


class DataService:
    def __init__(
        self,
        dataset: DatasetConfig,
        *,
        cutoff: date | None = None,
        store_root: Path | None = None,
    ) -> None:
        settings = get_settings()
        if store_root is None:
            segment = cutoff.isoformat() if cutoff is not None else "full"
            store_root = settings.artifacts_dir / dataset.name / segment
        self.engine = Engine(dataset, cutoff=cutoff)
        self.store = ResultStore(store_root)
        self.memory: Memory = load_memory(dataset)

    def query_metrics(self, q: MetricQuery) -> SummaryCard:
        model = self.memory.semantic
        resolved_filters: list[FilterSpec] = []
        for spec in q.filters:
            canonical = resolve_values(
                self.engine, model, spec.dimension, list(spec.values)
            )
            resolved_filters.append(spec.model_copy(update={"values": canonical}))
        q_resolved = q.model_copy(update={"filters": resolved_filters})
        sql = compile_query(model, q_resolved)
        t0 = time.perf_counter()
        table = self.engine.execute(sql)
        elapsed_s = time.perf_counter() - t0
        cov = self.engine.coverage()
        notes, extra_meta = _coverage_notes(cov, q_resolved, table)
        meta: dict = {
            "query": q_resolved.model_dump(mode="json"),
            "cutoff": self.engine.cutoff.isoformat() if self.engine.cutoff else None,
            **extra_meta,
        }
        return self.store.save(
            table,
            produced_by="query_metrics",
            source=sql,
            governed=True,
            elapsed_s=elapsed_s,
            notes=notes,
            meta=meta,
        )

    def run_sql(self, sql: str) -> SummaryCard:
        t0 = time.perf_counter()
        table = self.engine.execute(sql)
        elapsed_s = time.perf_counter() - t0
        return self.store.save(
            table,
            produced_by="run_sql",
            source=sql,
            governed=False,
            elapsed_s=elapsed_s,
        )

    def describe(self, target: str) -> str:
        if is_result_handle_id(target) and self.store.exists(target):
            return self.store.card(target).render()
        if target == "metrics":
            return describe_metrics(self.memory.semantic)
        if target == "dimensions":
            return describe_dimensions(self.memory.semantic)
        if target.startswith("dimension:"):
            name = target.split(":", 1)[1]
            try:
                return _describe_dimension_values(
                    self.engine, self.memory.semantic, name
                )
            except KeyError as exc:
                available = ", ".join(d.name for d in self.memory.semantic.dimensions)
                raise InvalidRequestError(
                    f"unknown dimension {name!r}; valid dimensions: {available}",
                    hint="call describe('dimensions')",
                ) from exc
        if target == "schema":
            return _describe_schema(self.engine)
        if target == "coverage":
            return _describe_coverage(self.engine)
        choices = ", ".join(_DESCRIBE_TARGETS)
        raise InvalidRequestError(
            f"unknown describe target {target!r}; valid: {choices}",
            hint=f"call describe('metrics') with one of: {choices}",
        )

    def _forecast_source(
        self,
        handle_id: str,
        *,
        freq: str,
        horizon: int,
        value_col: str,
        id_col: str | None,
        time_col: str,
        levels,
    ) -> tuple[str, list | None]:
        """Cache key plus the candidate models it was built from."""
        query_meta = self.store.card(handle_id).meta.get("query")
        models, names, candidates_source = self._candidates(
            value_col=value_col, id_col=id_col, freq=freq, query_meta=query_meta
        )
        source = json.dumps(
            {
                "input_handle": handle_id,
                "freq": freq,
                "horizon": horizon,
                "value_col": value_col,
                "id_col": id_col,
                "time_col": time_col,
                "levels": sorted(levels),
                "candidates": names,
                "candidates_source": candidates_source,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        return source, models

    def forecast_is_cached(
        self,
        handle_id: str,
        *,
        value_col: str,
        id_col: str | None,
        freq: Literal["D", "MS"],
        horizon: int | None = None,
        time_col: str = "period",
    ) -> bool:
        if horizon is None:
            horizon = DEFAULTS[freq]["horizon"]
        try:
            source, _ = self._forecast_source(
                handle_id,
                freq=freq,
                horizon=horizon,
                value_col=value_col,
                id_col=id_col,
                time_col=time_col,
                levels=_FORECAST_LEVELS,
            )
        except UnknownNameError:
            return False
        return self.store.lookup("forecast", source) is not None

    def estimate_forecast_s(
        self, handle_id: str, *, id_col: str | None, freq: Literal["D", "MS"]
    ) -> float:
        """Rough wall-clock seconds for an uncached forecast of this handle."""
        n_ids = 1
        if id_col is not None:
            table = self.store.load(handle_id)
            _require_columns(table.column_names, [id_col])
            n_ids = max(len(set(table.column(id_col).to_pylist())), 1)
        windows = DEFAULTS[freq]["n_windows"]
        return n_ids * windows * _WINDOW_COST_S[freq]

    def _candidates(
        self,
        *,
        value_col: str,
        id_col: str | None,
        freq: str,
        query_meta: dict | None,
    ) -> tuple[list | None, list[str], str]:
        """Models for the matching series spec, else the defaults."""
        defaults = candidate_models(freq)
        if id_col is None:
            dims: list[str] = []
        elif query_meta is not None and id_col not in query_meta.get("dimensions", []):
            return None, [str(m) for m in defaults], "default"
        else:
            dims = [id_col]
        for spec in self.memory.series:
            if spec.metric != value_col or spec.freq != freq or spec.dimensions != dims:
                continue
            wanted = set(spec.recommended_models)
            chosen = [m for m in defaults if str(m) in wanted and str(m) != "SeasonalNaive"]
            if not chosen:
                break
            ordered = [m for name in spec.recommended_models for m in chosen if str(m) == name]
            baseline = next(m for m in defaults if str(m) == "SeasonalNaive")
            models = [*dict.fromkeys(ordered), baseline]
            status = spec.recommended_models_status or "proposed"
            return models, [str(m) for m in models], f"series:{spec.name} ({status})"
        return None, [str(m) for m in defaults], "default"

    def forecast(
        self,
        handle_id: str,
        *,
        time_col: str = "period",
        value_col: str,
        id_col: str | None = None,
        freq: Literal["D", "MS"],
        horizon: int | None = None,
        levels: tuple[int, int] = (80, 95),
        progress: ProgressFn | None = None,
        budget_s: float | None = None,
    ) -> SummaryCard:
        if sorted(levels) != list(_FORECAST_LEVELS):
            raise InvalidRequestError(
                "forecast levels must be (80, 95)",
                hint="omit levels; intervals are always 80% and 95%",
            )
        if horizon is None:
            horizon = DEFAULTS[freq]["horizon"]
        source, models = self._forecast_source(
            handle_id,
            freq=freq,
            horizon=horizon,
            value_col=value_col,
            id_col=id_col,
            time_col=time_col,
            levels=levels,
        )
        cached = self.store.lookup("forecast", source)
        if cached is not None:
            return cached
        with _forecast_lock(f"{self.store._root}\0{source}"):
            cached = self.store.lookup("forecast", source)
            if cached is not None:
                return cached
            return self._compute_forecast(
                handle_id,
                source=source,
                models=models,
                time_col=time_col,
                value_col=value_col,
                id_col=id_col,
                freq=freq,
                horizon=horizon,
                levels=levels,
                progress=progress,
                budget_s=budget_s,
            )

    def _compute_forecast(
        self,
        handle_id: str,
        *,
        source: str,
        models: list | None,
        time_col: str,
        value_col: str,
        id_col: str | None,
        freq: Literal["D", "MS"],
        horizon: int,
        levels: tuple[int, int],
        progress: ProgressFn | None,
        budget_s: float | None,
    ) -> SummaryCard:
        table = self.store.load(handle_id)
        input_card = self.store.card(handle_id)
        query_meta = input_card.meta.get("query")
        if query_meta is not None and {"day": "D", "month": "MS"}.get(query_meta.get("grain")) != freq:
            raise InvalidRequestError(
                "forecast supports query grain/frequency pairs day→D and month→MS only",
                hint="query at day grain for freq D, month grain for freq MS",
            )
        _require_columns(
            table.column_names,
            [time_col, value_col, *([id_col] if id_col else [])],
        )
        if id_col is None:
            periods = table.column(time_col).to_pylist()
            if len(periods) != len(set(periods)):
                raise InvalidRequestError(
                    "handle has multiple rows per period; pass id_col",
                    hint="pass id_col so each id is forecast as its own series",
                )
        t0 = time.perf_counter()
        additive = query_meta is not None and _metric_additive(self.memory, value_col)
        prepared = prepare_series(
            table.to_pandas(),
            freq=freq,
            data_end=self.engine.coverage().end,
            quirks=self.memory.quirks,
            id_col=id_col,
            time_col=time_col,
            value_col=value_col,
            additive=additive,
        )
        if query_meta is None and prepared.imputed_points > 0.2 * len(prepared.df):
            raise InvalidRequestError(
                "more than 20% of series points were imputed; aggregate to the right grain first",
                hint="aggregate to the right grain first, or query a wider time range",
            )
        data_end = pd.Timestamp(self.engine.coverage().end)
        complete_end = data_end.to_period("D" if freq == "D" else "M").start_time
        boundary = (
            complete_end + pd.offsets.MonthEnd(0) if freq == "MS" else complete_end
        ) + pd.Timedelta(hours=23)
        if data_end < boundary:
            complete_end -= pd.tseries.frequencies.to_offset(freq)
        series_end = prepared.df.ds.max()
        if series_end < complete_end:
            raise InvalidRequestError(
                f"handle ends {series_end.date()} but data runs to {complete_end.date()}; "
                "re-query without an earlier end (forecast origin must be the end of data)"
            )
        for uid, end in prepared.df.groupby("unique_id").ds.max().items():
            if end < series_end:
                prepared.adjustments.append(
                    f"{uid} ends {end.date()}, before the series end {series_end.date()}"
                )
        _, candidate_names, candidates_source = self._candidates(
            value_col=value_col, id_col=id_col, freq=freq, query_meta=query_meta
        )
        if budget_s is None:
            budget_s = get_settings().forecast_budget_s
        # Fitting counts against the process-wide concurrency cap.
        with get_gate().slot():
            run = run_forecast(
                prepared,
                freq=freq,
                horizon=horizon,
                levels=levels,
                models=models,
                n_jobs=1,
                progress=progress,
                budget_s=budget_s,
            )
        elapsed_s = time.perf_counter() - t0
        notes = list(run.adjustments)
        if not run.backtest.beats_baseline and _BASELINE_NOTE not in notes:
            notes.append(_BASELINE_NOTE)
        frame = run.forecast.loc[:, list(_FORECAST_COLUMNS)]
        forecast_table = pa.Table.from_pandas(frame, preserve_index=False)
        kept = _kept_scores(run.backtest.scores, run.model)
        headline_ids = _headline_ids(table, id_col=id_col, value_col=value_col)
        headline_kept = kept.loc[kept["unique_id"].map(str).isin(set(headline_ids))]
        # A budget-truncated run is returned but never served from the cache.
        complete = run.backtest.windows_used >= run.backtest.windows
        backtest_card = self.store.save(
            pa.Table.from_pandas(kept, preserve_index=False),
            produced_by="backtest",
            source=source + ":backtest",
            governed=input_card.governed,
            elapsed_s=elapsed_s,
            index=complete,
        )
        meta = {
            "model": run.model,
            "origin": run.origin.isoformat(),
            "horizon": horizon,
            "freq": freq,
            "beats_baseline": bool(run.backtest.beats_baseline),
            "backtest_handle": backtest_card.handle_id,
            "backtest_summary": _backtest_summary(
                headline_kept if len(headline_kept) else kept, run.model
            ),
            "input_handle": handle_id,
            "cutoff": self.engine.cutoff.isoformat() if self.engine.cutoff else None,
            "data_end": data_end.isoformat(),
            "windows_used": int(run.backtest.windows_used),
            "candidates": candidate_names,
            "candidates_source": candidates_source,
            "headline_ids": headline_ids,
            "headline_rule": _HEADLINE_RULE,
        }
        if kept["unique_id"].nunique() <= _BACKTEST_SERIES_LIMIT:
            meta["backtest"] = _backtest_rows(kept)
        return self.store.save(
            forecast_table,
            produced_by="forecast",
            source=source,
            governed=input_card.governed,
            elapsed_s=elapsed_s,
            notes=notes,
            meta=meta,
            index=complete,
        )


def _metric_additive(memory: Memory, value_col: str) -> bool:
    semantic = getattr(memory, "semantic", None)
    if semantic is None:
        return False
    for metric in semantic.metrics:
        if metric.name == value_col:
            return _is_additive(metric.expr)
    return False


def _require_columns(columns: list[str], required: list[str]) -> None:
    missing = [name for name in required if name not in columns]
    if missing:
        available = ", ".join(columns)
        unknown = ", ".join(missing)
        raise InvalidRequestError(
            f"unknown column(s): {unknown}; columns: {available}",
            hint="use the output column names of the handle",
        )


def _kept_scores(scores, best_model: str):
    columns = ["unique_id", "model", *_SCORE_FIELDS]
    return scores.loc[scores["model"].isin([best_model, "SeasonalNaive"]), columns].reset_index(
        drop=True
    )


def _backtest_summary(kept, best_model: str) -> dict:
    best = kept.loc[kept["model"] == best_model]
    baseline = kept.loc[kept["model"] == "SeasonalNaive"]

    def mean(frame, column: str) -> float:
        return round(float(frame[column].mean()), 4)

    return {
        "best_model": best_model,
        "mean_relative_mae": mean(best, "relative_mae"),
        "mean_mase": mean(best, "mase"),
        "mean_coverage_80": mean(best, "coverage_80"),
        "mean_coverage_95": mean(best, "coverage_95"),
        "baseline_mean_mase": mean(baseline, "mase"),
    }


def _backtest_rows(kept) -> list[dict]:
    rows: list[dict] = []
    for record in kept.to_dict(orient="records"):
        rows.append(
            {
                "unique_id": str(record["unique_id"]),
                "model": str(record["model"]),
                **{field: round(float(record[field]), 4) for field in _SCORE_FIELDS},
            }
        )
    return rows


def _describe_dimension_values(
    engine: Engine, model: SemanticModel, name: str
) -> str:
    dimension = model.dimension(name)
    source = from_sql(model, [dimension])
    sql = (
        f"SELECT {dimension.expr} AS value, count(*) AS n\n{source}\n"
        f"GROUP BY 1\nORDER BY n DESC\nLIMIT {_VALUE_LIMIT}"
    )
    table = engine.execute(sql)
    lines = [f"{row['value']}\t{row['n']}" for row in table.to_pylist()]
    if table.num_rows >= _VALUE_LIMIT:
        total = engine.execute(
            f"SELECT count(DISTINCT {dimension.expr}) AS n\n{source}"
        )[0][0].as_py()
        if total > _VALUE_LIMIT:
            lines.insert(0, f"showing {_VALUE_LIMIT} of {total} values")
    return "\n".join(lines)


def _describe_schema(engine: Engine) -> str:
    return "\n".join(f"{name}: {typ}" for name, typ in engine.columns())


def _describe_coverage(engine: Engine) -> str:
    cov = engine.coverage()
    cutoff = engine.cutoff.isoformat() if engine.cutoff else "none"
    return (
        f"start: {_format_bound(cov.start)}\n"
        f"end: {_format_bound(cov.end)}\n"
        f"cutoff: {cutoff}"
    )


def _format_bound(dt: datetime) -> str:
    if dt.hour == 0 and dt.minute == 0 and dt.second == 0 and dt.microsecond == 0:
        return dt.date().isoformat()
    return dt.strftime("%Y-%m-%d %H:%M")


def _headline_ids(
    table: pa.Table, *, id_col: str | None, value_col: str
) -> list[str]:
    """Ids whose total is at least 1% of the headline total."""
    if id_col is None:
        return ["series"]
    frame = table.select([id_col, value_col]).to_pandas()
    totals = frame.groupby(id_col)[value_col].sum()
    shares = totals / totals.sum()
    headline = totals[shares >= _MIN_HEADLINE_SHARE]
    # A degenerate total (all zero/negative) cannot rank ids; keep every id.
    wanted = headline.index if not headline.empty else totals.index
    return [str(uid) for uid in wanted]


def _coverage_notes(
    cov: Coverage,
    q: MetricQuery,
    table: pa.Table,
) -> tuple[list[str], dict]:
    notes: list[str] = []
    extra: dict = {}
    if q.time_range is not None:
        start, end = q.time_range
        end_exclusive = datetime.combine(end, datetime.min.time())
        beyond = end_exclusive > cov.end + timedelta(hours=1)
        before = datetime.combine(start, datetime.min.time()) < cov.start
        if before or beyond:
            span = f"{_format_bound(cov.start)} .. {_format_bound(cov.end)}"
            notes.append(
                f"requested range extends beyond available data ({span})"
            )
    if (
        q.grain
        and q.grain != "hour"
        and table.num_rows > 0
        and "period" in table.column_names
    ):
        last_period = _max_period(table.column("period"))
        if last_period is not None and _is_partial_period(
            cov.end, last_period, q.grain
        ):
            period_label = _period_label(last_period)
            notes.append(
                f"last period {period_label} is partial "
                f"(data ends {_format_bound(cov.end)})"
            )
            extra["partial_last_period"] = period_label
    if (
        q.time_range is not None
        and q.grain in {"week", "month"}
        and table.num_rows > 0
        and "period" in table.column_names
    ):
        start, end = q.time_range
        first_period = _min_period(table.column("period"))
        last_period = _max_period(table.column("period"))
        if first_period is not None and datetime.combine(
            start, datetime.min.time()
        ) > _period_start(first_period, q.grain):
            notes.append(
                f"{_grain_label(first_period, q.grain)} is partial: "
                f"time_range starts {start.isoformat()}"
            )
        if last_period is not None and datetime.combine(
            end, datetime.min.time()
        ) <= _period_threshold(last_period, q.grain):
            notes.append(
                f"{_grain_label(last_period, q.grain)} is partial: "
                f"time_range ends {end.isoformat()}"
            )
    return notes, extra


def _min_period(column: pa.ChunkedArray) -> date | datetime | None:
    values = [v for v in column.to_pylist() if v is not None]
    if not values:
        return None
    return min(values)


def _period_start(period: date | datetime, grain: str) -> datetime:
    if isinstance(period, datetime):
        period = period.date()
    if grain == "month":
        return datetime(period.year, period.month, 1)
    if grain == "week":
        monday = period - timedelta(days=period.weekday())
        return datetime.combine(monday, datetime.min.time())
    return datetime.combine(period, datetime.min.time())


def _grain_label(period: date | datetime, grain: str) -> str:
    if grain == "month":
        return f"{period:%Y-%m}"
    return _period_label(period)


def _max_period(column: pa.ChunkedArray) -> date | datetime | None:
    values = [v for v in column.to_pylist() if v is not None]
    if not values:
        return None
    return max(values)


def _period_label(period: date | datetime) -> str:
    if isinstance(period, datetime):
        if period.time() == datetime.min.time():
            return period.date().isoformat()
        return period.strftime("%Y-%m-%d %H:%M")
    return period.isoformat()


def _period_threshold(period: date | datetime, grain: str) -> datetime:
    if isinstance(period, datetime):
        period = period.date()
    if grain == "month":
        if period.month == 12:
            next_start = date(period.year + 1, 1, 1)
        else:
            next_start = date(period.year, period.month + 1, 1)
    elif grain == "week":
        next_start = period + timedelta(days=7)
    elif grain == "day":
        next_start = period + timedelta(days=1)
    else:
        raise InvalidRequestError(f"unsupported grain {grain!r}")
    return datetime.combine(next_start, datetime.min.time()) - timedelta(hours=1)


def _is_partial_period(
    coverage_end: datetime, period: date | datetime, grain: str
) -> bool:
    threshold = _period_threshold(period, grain)
    return coverage_end < threshold
