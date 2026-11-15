"""Per-kind JSON payloads for stored charts, built from result handles only."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analyst.charts._inner import InnerArea, InnerBar, InnerFan, InnerLine
from analyst.charts.downsample import m4
from analyst.charts.spec import (
    AreaChart,
    BarChart,
    BoxChart,
    ForecastFanChart,
    HeatmapChart,
    HistogramChart,
    KpiChart,
    LineChart,
    MapChart,
    ScatterChart,
    SmallMultiplesChart,
)
from analyst.charts.store import StoredChart
from analyst.data.dataset import GeometrySpec, contained_geometry_path
from analyst.data.results import ResultStore
from analyst.errors import InvalidRequestError

PAYLOAD_VERSION = 1
DEFAULT_WIDTH = 1200
MIN_WIDTH = 100
MAX_WIDTH = 4000
MAX_SERIES = 6
MAX_FACETS = 12
HISTOGRAM_BINS = 40
SCATTER_POINT_CAP = 50_000
DENSITY_GRID = 200


class _Ctx:
    def __init__(self, store: ResultStore, width: int, x_from, x_to) -> None:
        self.store = store
        self.width = width
        self.x_from = x_from
        self.x_to = x_to
        self.notes: list[str] = []
        self.n_from = 0
        self.n_to = 0

    def table(self, handle: str) -> pd.DataFrame:
        return self.store.load(handle).to_pandas()

    def note(self, text: str) -> None:
        if text not in self.notes:
            self.notes.append(text)


def build_payload(
    stored: StoredChart,
    store: ResultStore,
    *,
    width: int = DEFAULT_WIDTH,
    x_from: datetime | None = None,
    x_to: datetime | None = None,
    geometries: Mapping[str, GeometrySpec] | None = None,
    geometry_dir: Path | None = None,
) -> dict:
    ctx = _Ctx(store, max(MIN_WIDTH, min(MAX_WIDTH, int(width))), x_from, x_to)
    spec = stored.spec
    data = _build_data(spec, ctx, geometries, geometry_dir)
    downsampled = {"from": ctx.n_from, "to": ctx.n_to} if ctx.n_to < ctx.n_from else None
    return {
        "version": PAYLOAD_VERSION,
        "chart_id": stored.chart_id,
        "kind": spec.kind,
        "title": spec.title,
        "dataset": stored.dataset,
        "cutoff": stored.cutoff.isoformat() if stored.cutoff else None,
        "spec": spec.model_dump(mode="json"),
        "notes": ctx.notes,
        "downsampled": downsampled,
        "data": data,
    }


def _build_data(spec, ctx: _Ctx, geometries, geometry_dir) -> dict:
    if isinstance(spec, LineChart | AreaChart):
        return _line_data(ctx, ctx.table(spec.handle), spec.x, spec.y, spec.series)
    if isinstance(spec, ForecastFanChart):
        return _fan_data(ctx, spec.forecast_handle, spec)
    if isinstance(spec, BarChart):
        return _bar_data(ctx.table(spec.handle), spec.category, spec.value, spec.series, spec.top_n)
    if isinstance(spec, HeatmapChart):
        return _heatmap_data(ctx.table(spec.handle), spec.x, spec.y, spec.value)
    if isinstance(spec, MapChart):
        return _map_data(ctx.table(spec.handle), spec, geometries or {}, geometry_dir)
    if isinstance(spec, HistogramChart):
        return _histogram_data(ctx.table(spec.handle), spec)
    if isinstance(spec, BoxChart):
        return _box_data(ctx.table(spec.handle), spec.group, spec.value)
    if isinstance(spec, ScatterChart):
        return _scatter_data(ctx.table(spec.handle), spec)
    if isinstance(spec, KpiChart):
        return _kpi_data(ctx, spec)
    if isinstance(spec, SmallMultiplesChart):
        return _multiples_data(ctx, spec)
    raise InvalidRequestError(f"unsupported chart kind {spec.kind}")  # pragma: no cover


# ---------------------------------------------------------------- helpers


def _num(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _nums(values) -> list[float | None]:
    return [_num(v) for v in values]


def _iso(v: Any) -> str:
    return pd.Timestamp(v).isoformat()


def _isos(values) -> list[str]:
    return [_iso(v) for v in values]


def _label(v: Any) -> Any:
    if isinstance(v, pd.Timestamp | datetime):
        return _iso(v)
    if hasattr(v, "item") and not isinstance(v, str):
        v = v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return v


def _to_time(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series)


def _filter_range(ctx: _Ctx, df: pd.DataFrame, col: str) -> pd.DataFrame:
    if ctx.x_from is None and ctx.x_to is None:
        return df
    t = _to_time(df[col])
    mask = pd.Series(True, index=df.index)
    for bound, op in ((ctx.x_from, "ge"), (ctx.x_to, "le")):
        if bound is None:
            continue
        b = pd.Timestamp(bound)
        if t.dt.tz is not None and b.tzinfo is None:
            b = b.tz_localize("UTC")
        elif t.dt.tz is None and b.tzinfo is not None:
            b = b.tz_convert("UTC").tz_localize(None)
        mask &= t >= b if op == "ge" else t <= b
    return df.loc[mask]


def _downsample_xy(ctx: _Ctx, x: pd.Series, y: pd.Series) -> tuple[list[str], list[float | None]]:
    """Sort by time, M4-downsample, return ISO x and clean y."""
    t = _to_time(x).reset_index(drop=True)
    yv = pd.to_numeric(y.reset_index(drop=True), errors="coerce")
    null_t = t.isna()
    if null_t.any():
        ctx.note(f"{int(null_t.sum())} rows with no time dropped")
        t = t[~null_t].reset_index(drop=True)
        yv = yv[~null_t].reset_index(drop=True)
    order = np.argsort(t.to_numpy(), kind="stable")
    t = t.iloc[order].reset_index(drop=True)
    yv = yv.iloc[order].reset_index(drop=True)
    xi = t.dt.tz_localize(None).to_numpy().astype("datetime64[ns]").astype(np.int64) if t.dt.tz is not None else t.to_numpy().astype("datetime64[ns]").astype(np.int64)
    ya = yv.to_numpy(dtype=np.float64)
    ya = np.where(np.isfinite(ya), ya, np.nan)
    keep = m4(xi, ya, ctx.width)
    ctx.n_from += len(t)
    ctx.n_to += len(keep)
    return _isos(t.iloc[keep]), _nums(ya[keep])


# ---------------------------------------------------------------- line / area


def _line_data(ctx: _Ctx, df: pd.DataFrame, x: str, y: str, series: str | None) -> dict:
    df = _filter_range(ctx, df, x)
    groups: list[tuple[str, pd.DataFrame]]
    if series is None:
        groups = [(y, df[[x, y]])]
    else:
        totals = df.groupby(series, sort=False)[y].sum(min_count=1).fillna(0.0)
        ranked = totals.sort_values(ascending=False, kind="stable").index.tolist()
        groups = [(str(k), df.loc[df[series] == k, [x, y]]) for k in ranked[:MAX_SERIES]]
        rest = ranked[MAX_SERIES:]
        if rest:
            folded = df.loc[df[series].isin(rest), [x, y]].groupby(x, sort=True)[y].sum(min_count=1)
            groups.append(("other", folded.reset_index()))
            ctx.note(f"{len(rest)} smaller series summed into 'other' (showing top {MAX_SERIES})")
    out = []
    for name, g in groups:
        xs, ys = _downsample_xy(ctx, g[x], g[y])
        out.append({"name": name, "x": xs, "y": ys})
    return {"series": out}


# ---------------------------------------------------------------- forecast fan


def _fan_meta(ctx: _Ctx, handle: str):
    card = ctx.store.card(handle)
    meta = card.meta
    if not meta.get("time_col") or not meta.get("value_col"):
        raise InvalidRequestError(
            "forecast predates chart support; re-run forecast",
            hint="call forecast again to get a handle the chart can draw",
        )
    return card, meta


def _fan_default_series(meta: dict, table: pd.DataFrame) -> str:
    ids = [str(i) for i in meta.get("headline_ids") or []]
    if ids:
        return ids[0]
    return str(table["unique_id"].iloc[0])


def _fan_data(ctx: _Ctx, handle: str, spec, series_id: str | None = None) -> dict:
    _card, meta = _fan_meta(ctx, handle)
    table = ctx.table(handle)
    sid = series_id or spec.series_id or _fan_default_series(meta, table)
    fc = table.loc[table["unique_id"].astype(str) == sid].copy()
    if fc.empty:
        available = sorted(table["unique_id"].astype(str).unique())
        shown = ", ".join(available[:20]) + (" ..." if len(available) > 20 else "")
        raise InvalidRequestError(
            f"series_id {sid!r} is not in forecast {handle}",
            hint=f"available series ids: {shown}",
        )
    fc["ds"] = _to_time(fc["ds"])
    fc = fc.sort_values("ds")

    time_col, value_col, id_col = meta["time_col"], meta["value_col"], meta.get("id_col")
    input_handle = meta.get("input_handle")
    if not input_handle or not ctx.store.exists(input_handle):
        raise InvalidRequestError(
            f"forecast {handle} has no usable input history handle",
            hint="re-run forecast so its input handle is stored",
        )
    hist = ctx.table(input_handle)
    if id_col and id_col in hist.columns:
        hist = hist.loc[hist[id_col].astype(str) == sid]
    hist = hist[[time_col, value_col]].copy()
    hist[time_col] = _to_time(hist[time_col])
    hist = hist.sort_values(time_col)

    hx, hy = _downsample_xy(ctx, *_range(ctx, hist, time_col, value_col))
    data: dict[str, Any] = {
        "series_id": sid,
        "history": {"x": hx, "y": hy},
        "forecast": {
            "x": _isos(fc["ds"]),
            **{k: _nums(fc[k]) for k in ("yhat", "lo_80", "hi_80", "lo_95", "hi_95")},
        },
        "origin": meta.get("origin"),
        "last_year": None,
        "baseline": None,
        "replay": None,
        "model": meta.get("model"),
        "scorecard": _scorecard(ctx, meta),
    }
    if spec.last_year:
        data["last_year"] = _last_year(fc["ds"], hist, time_col, value_col, meta.get("freq"))
    if spec.baseline and "baseline" in fc.columns:
        data["baseline"] = {"x": _isos(fc["ds"]), "y": _nums(fc["baseline"])}
    if spec.replay and meta.get("fold_handle"):
        fold = ctx.table(meta["fold_handle"])
        fold = fold.loc[fold["unique_id"].astype(str) == sid].copy()
        if len(fold):
            fold["ds"] = _to_time(fold["ds"])
            fold = fold.sort_values("ds")
            data["replay"] = {
                "x": _isos(fold["ds"]),
                **{
                    k: _nums(fold[k])
                    for k in ("yhat", "lo_80", "hi_80", "lo_95", "hi_95", "actual")
                },
            }
    return data


def _range(ctx: _Ctx, df: pd.DataFrame, x: str, y: str) -> tuple[pd.Series, pd.Series]:
    df = _filter_range(ctx, df, x)
    return df[x], df[y]


def _last_year(ds: pd.Series, hist: pd.DataFrame, time_col: str, value_col: str, freq) -> dict:
    lookup = dict(zip(hist[time_col], hist[value_col], strict=True))
    monthly = freq == "MS"
    xs: list[str] = []
    ys: list[float | None] = []
    for d in ds:
        prior = d - pd.DateOffset(years=1) if monthly else d - pd.Timedelta(days=364)
        xs.append(_iso(d))
        ys.append(_num(lookup.get(prior)))
    return {"x": xs, "y": ys}


def _scorecard(ctx: _Ctx, meta: dict) -> dict:
    s = meta.get("backtest_summary") or {}
    windows = meta.get("windows_used")
    backtest = meta.get("backtest_handle")
    if backtest and ctx.store.exists(backtest):
        windows = ctx.store.card(backtest).meta.get("windows", windows)
    return {
        "mase": _num(s.get("mean_mase")),
        "relative_mae": _num(s.get("mean_relative_mae")),
        "coverage_80": _num(s.get("mean_coverage_80")),
        "coverage_95": _num(s.get("mean_coverage_95")),
        "windows_used": meta.get("windows_used"),
        "windows": windows,
        "beats_baseline": meta.get("beats_baseline"),
    }


# ---------------------------------------------------------------- map


def _geo_key(k: Any) -> str | None:
    """One string form for feature and handle keys: 1, 1.0 and "1" all become "1"."""
    if k is None or (not isinstance(k, str) and pd.isna(k)):
        return None
    if isinstance(k, float) and k.is_integer():
        return str(int(k))
    if isinstance(k, np.floating) and float(k).is_integer():
        return str(int(k))
    return str(k)


def _map_data(
    df: pd.DataFrame,
    spec: MapChart,
    geometries: Mapping[str, GeometrySpec],
    geometry_dir: Path | None,
) -> dict:
    geo = geometries.get(spec.geometry)
    if geo is None or geometry_dir is None:
        raise InvalidRequestError(
            f"this dataset has no geometry named {spec.geometry!r}",
            hint="map charts need a geometry declared by the dataset; use a bar chart",
        )
    try:
        path = contained_geometry_path(geometry_dir, geo)
    except ValueError as exc:
        raise InvalidRequestError(str(exc)) from exc
    try:
        features = json.loads(path.read_text(encoding="utf-8")).get("features", [])
    except (OSError, ValueError, AttributeError) as exc:
        raise InvalidRequestError(
            f"geometry {geo.name!r} is missing or not valid GeoJSON",
            hint="map charts need a readable geometry file; use a bar chart",
        ) from exc
    feature_keys = {
        _geo_key((f.get("properties") or {}).get(geo.key_property))
        for f in features
        if isinstance(f, dict)
    } - {None}
    keys = [_geo_key(k) for k in df[spec.key]]
    values = [_num(v) for v in df[spec.value]]
    unmatched = sorted({k for k in keys if k is not None and k not in feature_keys})
    return {
        "geometry": geo.name,
        "key_property": geo.key_property,
        "keys": keys,
        "values": values,
        "unmatched": unmatched,
    }


# ---------------------------------------------------------------- bar / heatmap


def _bar_data(df: pd.DataFrame, category: str, value: str, series: str | None, top_n: int | None) -> dict:
    name = value if series is None else None
    if series is None:
        wide = df.groupby(category, sort=False)[value].sum(min_count=1).to_frame(value)
    else:
        wide = df.pivot_table(index=category, columns=series, values=value, aggfunc="sum", sort=False)
        order = wide.sum(axis=0).sort_values(ascending=False, kind="stable").index
        wide = wide[order]
    other = False
    if top_n is not None:
        totals = wide.sum(axis=1)
        ranked = totals.sort_values(ascending=False, kind="stable").index.tolist()
        keep, rest = ranked[:top_n], ranked[top_n:]
        head = wide.loc[keep]
        if rest:
            tail = wide.loc[rest].sum(axis=0, min_count=1).to_frame("other").T
            head = pd.concat([head, tail])
            other = True
        wide = head
    return {
        "categories": [_label(c) for c in wide.index],
        "series": [
            {"name": name if name is not None else str(col), "values": _nums(wide[col])}
            for col in wide.columns
        ],
        "other": other,
    }


def _heatmap_data(df: pd.DataFrame, x: str, y: str, value: str) -> dict:
    grid = df.pivot_table(index=y, columns=x, values=value, aggfunc="sum", sort=True)
    grid = grid.sort_index(axis=0).sort_index(axis=1)
    return {
        "x": [_label(c) for c in grid.columns],
        "y": [_label(r) for r in grid.index],
        "values": [_nums(row) for row in grid.to_numpy(dtype=np.float64)],
    }


# ---------------------------------------------------------------- histogram / box / scatter


def _histogram_data(df: pd.DataFrame, spec: HistogramChart) -> dict:
    if spec.value is not None:
        vals = pd.to_numeric(df[spec.value], errors="coerce").to_numpy(dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if not len(vals):
            return {"edges": [], "counts": []}
        counts, edges = np.histogram(vals, bins=HISTOGRAM_BINS)
        return {"edges": _nums(edges), "counts": _nums(counts)}
    bins = pd.to_numeric(df[spec.bin], errors="coerce")
    if bins.isna().any():
        raise InvalidRequestError(
            f"histogram bin column {spec.bin!r} must be numeric",
            hint="use the numeric lower edge of each bin",
        )
    if not len(bins):
        return {"edges": [], "counts": []}
    order = np.argsort(bins.to_numpy(), kind="stable")
    lows = bins.to_numpy(dtype=np.float64)[order]
    counts = pd.to_numeric(df[spec.count], errors="coerce").to_numpy(dtype=np.float64)[order]
    step = float(lows[-1] - lows[-2]) if len(lows) > 1 else 1.0
    edges = [*lows.tolist(), float(lows[-1]) + step]
    return {"edges": _nums(edges), "counts": _nums(counts)}


def _box_data(df: pd.DataFrame, group: str, value: str) -> dict:
    groups = []
    for key, g in df.groupby(group, sort=True):
        v = pd.to_numeric(g[value], errors="coerce").to_numpy(dtype=np.float64)
        v = v[np.isfinite(v)]
        if not len(v):
            continue
        q = np.percentile(v, [0, 25, 50, 75, 100])
        groups.append(
            {
                "name": str(_label(key)),
                "min": _num(q[0]),
                "q1": _num(q[1]),
                "median": _num(q[2]),
                "q3": _num(q[3]),
                "max": _num(q[4]),
                "n": len(v),
            }
        )
    return {"groups": groups}


def _scatter_data(df: pd.DataFrame, spec: ScatterChart) -> dict:
    x = pd.to_numeric(df[spec.x], errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(df[spec.y], errors="coerce").to_numpy(dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if int(ok.sum()) > SCATTER_POINT_CAP:
        counts, xe, ye = np.histogram2d(x[ok], y[ok], bins=DENSITY_GRID)
        return {
            "mode": "density",
            "x_edges": _nums(xe),
            "y_edges": _nums(ye),
            "counts": [[int(c) for c in row] for row in counts.T],
        }
    color = None
    if spec.color is not None:
        color = [_label(c) for c in df[spec.color].to_numpy()[ok]]
    return {"mode": "points", "x": _nums(x[ok]), "y": _nums(y[ok]), "color": color}


# ---------------------------------------------------------------- kpi


def _first(df: pd.DataFrame, col: str) -> float | None:
    return _num(df[col].iloc[0]) if len(df) else None


def _kpi_data(ctx: _Ctx, spec: KpiChart) -> dict:
    df = ctx.table(spec.handle)
    value = _first(df, spec.value)
    comparison = _first(df, spec.comparison) if spec.comparison else None
    change = None
    if value is not None and comparison not in (None, 0.0):
        change = (value - comparison) / abs(comparison) * 100.0
    trend = None
    if spec.trend_handle is not None:
        tdf = ctx.table(spec.trend_handle)
        xs, ys = _downsample_xy(ctx, tdf[spec.trend_x], tdf[spec.trend_y])
        trend = {"x": xs, "y": ys}
    return {
        "value": value,
        "label": spec.label,
        "comparison": comparison,
        "change_pct": change,
        "trend": trend,
    }


# ---------------------------------------------------------------- small multiples


def _multiples_data(ctx: _Ctx, spec: SmallMultiplesChart) -> dict:
    inner = spec.inner
    facets: list[dict] = []
    values: list[float] = []
    if isinstance(inner, InnerFan):
        table = ctx.table(inner.forecast_handle)
        totals = table.groupby(table["unique_id"].astype(str), sort=False)["yhat"].sum()
        names = _top_facets(ctx, totals)
        for name in names:
            data = _fan_data(ctx, inner.forecast_handle, inner, series_id=name)
            facets.append({"name": name, "data": data})
            values += _fan_numbers(data)
    else:
        df = ctx.table(inner.handle)
        measure = inner.value if isinstance(inner, InnerBar) else inner.y
        totals = df.groupby(spec.facet, sort=False)[measure].sum(min_count=1).fillna(0.0)
        for key in _top_facets(ctx, totals):
            sub = df.loc[df[spec.facet].astype(str) == key]
            if isinstance(inner, InnerBar):
                data = _bar_data(sub, inner.category, inner.value, inner.series, inner.top_n)
                values += [v for s in data["series"] for v in s["values"] if v is not None]
            else:
                assert isinstance(inner, InnerLine | InnerArea)
                data = _line_data(ctx, sub, inner.x, inner.y, inner.series)
                values += [v for s in data["series"] for v in s["y"] if v is not None]
            facets.append({"name": key, "data": data})
    y_range = [min(values), max(values)] if values else [0.0, 1.0]
    return {"inner_kind": inner.kind, "facets": facets, "y_range": y_range}


def _top_facets(ctx: _Ctx, totals: pd.Series) -> list[str]:
    totals = totals.copy()
    totals.index = totals.index.astype(str)
    ranked = totals.sort_values(ascending=False, kind="stable").index.tolist()
    if len(ranked) > MAX_FACETS:
        ctx.note(f"showing the {MAX_FACETS} largest of {len(ranked)} facets")
    return ranked[:MAX_FACETS]


def _fan_numbers(data: dict) -> list[float]:
    out: list[Any] = list(data["history"]["y"])
    for key in ("yhat", "lo_95", "hi_95"):
        out += data["forecast"][key]
    return [v for v in out if v is not None]
