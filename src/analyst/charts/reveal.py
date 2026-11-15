"""Actuals for a forecast made at a cutoff: the data the forecast could not see.

Read-only by construction. The uncut data is read through its own cached
:class:`Engine` (the same single-SELECT validation as every other query); no
:class:`DataService` is built for it, nothing is saved, no directory is made.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Any

import pandas as pd

from analyst.charts._inner import InnerFan
from analyst.charts.payload import MAX_FACETS, _fan_default_series
from analyst.charts.spec import ForecastFanChart, SmallMultiplesChart
from analyst.charts.store import StoredChart
from analyst.data.dataset import DatasetConfig
from analyst.data.engine import Coverage, Engine
from analyst.data.service import DataService
from analyst.errors import UnknownNameError
from analyst.memory.files import load_memory
from analyst.semantic.compiler import MetricQuery, compile_query

# Inputs that can be re-run on uncut data: governed queries and validated SQL.
REVEALABLE_INPUTS = frozenset({"query_metrics", "run_sql"})


class NotRevealableError(UnknownNameError):
    """The chart has no actuals to reveal (maps to 404)."""


@lru_cache(maxsize=4)
def _engine_for(dataset_json: str) -> Engine:
    return Engine(DatasetConfig.model_validate_json(dataset_json), cutoff=None)


def uncut_engine(dataset: DatasetConfig) -> Engine:
    """One cached cutoff-free engine per dataset configuration."""
    return _engine_for(dataset.model_dump_json())


def clear_engine_cache() -> None:
    _engine_for.cache_clear()


def reveal_actuals(stored: StoredChart, service: DataService, dataset: DatasetConfig) -> dict:
    if stored.cutoff is None:
        raise NotRevealableError("this chart has no cutoff; there is nothing to reveal")
    handle, wanted = _fan_target(stored)
    store = service.store
    meta = store.card(handle).meta
    time_col, value_col = meta.get("time_col"), meta.get("value_col")
    id_col, freq = meta.get("id_col"), meta.get("freq")
    input_handle = meta.get("input_handle")
    if not time_col or not value_col or freq not in {"D", "MS"} or not input_handle:
        raise NotRevealableError("this forecast predates reveal support")
    if not store.exists(input_handle):
        raise NotRevealableError("this forecast's input history is not stored")
    input_card = store.card(input_handle)
    if input_card.produced_by not in REVEALABLE_INPUTS:
        raise NotRevealableError(
            f"history produced by {input_card.produced_by} cannot be re-run on uncut data",
            hint="reveal works for forecasts of query_metrics or run_sql results",
        )

    forecast = store.load(handle).to_pandas()
    forecast["unique_id"] = forecast["unique_id"].astype(str)
    forecast["ds"] = pd.to_datetime(forecast["ds"])
    series_ids = _series_ids(stored, forecast, meta, wanted)
    forecast = forecast.loc[forecast["unique_id"].isin(series_ids)]
    if forecast.empty:
        return {"series": []}
    first, last = forecast["ds"].min(), forecast["ds"].max()

    engine = uncut_engine(dataset)
    if input_card.produced_by == "query_metrics":
        query = MetricQuery.model_validate(input_card.meta["query"])
        query = query.model_copy(
            update={"time_range": (first.date(), _next_period(last, freq).date())}
        )
        sql = compile_query(load_memory(dataset).semantic, query)
    else:
        sql = input_card.source
    table = engine.execute(sql)
    actuals = _actuals(
        table.to_pandas(),
        time_col=time_col,
        value_col=value_col,
        id_col=id_col,
        horizon=set(forecast["ds"]),
        complete_until=_complete_until(engine.coverage()),
        freq=freq,
    )

    out = []
    for sid in series_ids:
        rows = forecast.loc[forecast["unique_id"] == sid].sort_values("ds")
        actual = [actuals.get((sid, ds)) for ds in rows["ds"]]
        out.append(
            {
                "series_id": sid,
                "x": [ds.isoformat() for ds in rows["ds"]],
                "actual": actual,
                "yhat": [_num(v) for v in rows["yhat"]],
                "inside_80": _inside(actual, rows["lo_80"], rows["hi_80"]),
                "inside_95": _inside(actual, rows["lo_95"], rows["hi_95"]),
            }
        )
    return {"series": out}


def _fan_target(stored: StoredChart) -> tuple[str, str | None]:
    spec = stored.spec
    if isinstance(spec, ForecastFanChart):
        return spec.forecast_handle, spec.series_id
    if isinstance(spec, SmallMultiplesChart) and isinstance(spec.inner, InnerFan):
        return spec.inner.forecast_handle, None
    raise NotRevealableError("only forecast charts have actuals to reveal")


def _series_ids(stored: StoredChart, forecast: pd.DataFrame, meta: dict, wanted: str | None) -> list[str]:
    """The series the chart draws: the fan's one series, or the payload's facets."""
    if isinstance(stored.spec, ForecastFanChart):
        return [wanted or _fan_default_series(meta, forecast)]
    totals = forecast.groupby("unique_id", sort=False)["yhat"].sum()
    return totals.sort_values(ascending=False, kind="stable").index.tolist()[:MAX_FACETS]


def _next_period(ts: pd.Timestamp, freq: str) -> pd.Timestamp:
    return ts + (pd.DateOffset(months=1) if freq == "MS" else pd.Timedelta(days=1))


def _complete_until(coverage: Coverage) -> pd.Timestamp:
    """A period ending at or before this is fully covered by the data.

    Same rule as the service's partial-period check: a period is complete once
    the data reaches its last hour. A partial period would understate the
    actual, so it is left out (null) rather than revealed.
    """
    return pd.Timestamp(coverage.end) + pd.Timedelta(hours=1)


def _actuals(
    frame: pd.DataFrame,
    *,
    time_col: str,
    value_col: str,
    id_col: str | None,
    horizon: set,
    complete_until: pd.Timestamp,
    freq: str,
) -> dict[tuple[str, pd.Timestamp], float | None]:
    if time_col not in frame.columns or value_col not in frame.columns:
        return {}
    if id_col is not None and id_col not in frame.columns:
        return {}
    times = pd.to_datetime(frame[time_col])
    ids = frame[id_col].astype(str) if id_col is not None else pd.Series("series", index=frame.index)
    out: dict[tuple[str, pd.Timestamp], float | None] = {}
    for sid, ts, value in zip(ids, times, frame[value_col], strict=True):
        if ts not in horizon or _next_period(ts, freq) > complete_until:
            continue
        out[(sid, ts)] = _num(value)
    return out


def _inside(actual: list[float | None], lo, hi) -> list[bool | None]:
    out: list[bool | None] = []
    for a, low, high in zip(actual, lo, hi, strict=True):
        low, high = _num(low), _num(high)
        out.append(None if a is None or low is None or high is None else low <= a <= high)
    return out


def _num(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, date | datetime | timedelta):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
