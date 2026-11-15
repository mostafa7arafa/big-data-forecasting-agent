from __future__ import annotations

import json
import math
import os
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from analyst.charts.downsample import m4
from analyst.charts.payload import build_payload
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
from analyst.data.dataset import GeometrySpec
from analyst.data.results import ResultStore
from analyst.errors import InvalidRequestError

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "charts"


def _stored(spec, cutoff: date | None = date(2026, 6, 30)) -> StoredChart:
    return StoredChart(
        chart_id="c_0123456789",
        dataset="demo",
        cutoff=cutoff,
        spec=spec,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _save(store: ResultStore, table: pa.Table, *, produced_by="query", meta=None, tag="a"):
    return store.save(
        table,
        produced_by=produced_by,
        source=f"src-{tag}-{produced_by}",
        governed=True,
        elapsed_s=0.1,
        meta=meta or {},
    ).handle_id


@pytest.fixture
def store(tmp_path):
    return ResultStore(tmp_path / "results")


def test_m4_keeps_extremes_and_endpoints():
    n = 10_000
    x = np.arange(n)
    y = np.sin(x / 50.0)
    y[1234] = 50.0
    y[7777] = -50.0
    idx = m4(x, y, 100)
    assert len(idx) <= 400
    assert idx[0] == 0 and idx[-1] == n - 1
    assert 1234 in idx and 7777 in idx
    assert list(idx) == sorted(idx)


def test_m4_noop_when_small():
    x = np.arange(300)
    idx = m4(x, np.random.rand(300), 100)
    assert list(idx) == list(range(300))


def test_line_payload_downsamples_to_width(store):
    n = 1_000_000
    ds = pd.date_range("2020-01-01", periods=n, freq="min")
    table = pa.table({"ds": ds, "v": np.random.default_rng(0).random(n)})
    h = _save(store, table)
    spec = LineChart(title="t", handle=h, x="ds", y="v")
    p = build_payload(_stored(spec), store, width=1000)
    assert p["version"] == 1
    assert p["downsampled"] == {"from": 1_000_000, "to": len(p["data"]["series"][0]["x"])}
    assert len(p["data"]["series"][0]["x"]) <= 4000
    assert isinstance(p["data"]["series"][0]["x"][0], str)


def test_line_payload_range_filter(store):
    ds = pd.date_range("2024-01-01", periods=100, freq="D")
    h = _save(store, pa.table({"ds": ds, "v": np.arange(100.0)}))
    spec = LineChart(title="t", handle=h, x="ds", y="v")
    p = build_payload(
        _stored(spec), store, x_from=datetime(2024, 2, 1), x_to=datetime(2024, 2, 10)
    )
    xs = p["data"]["series"][0]["x"]
    assert xs[0].startswith("2024-02-01") and xs[-1].startswith("2024-02-10")
    assert len(xs) == 10


def test_line_more_than_six_series_folds_other(store):
    ds = pd.date_range("2024-01-01", periods=5, freq="D")
    rows = []
    for i in range(8):
        for d in ds:
            rows.append({"ds": d, "s": f"s{i}", "v": float(10 - i)})
    h = _save(store, pa.Table.from_pandas(pd.DataFrame(rows)))
    spec = LineChart(title="t", handle=h, x="ds", y="v", series="s")
    p = build_payload(_stored(spec), store)
    names = [s["name"] for s in p["data"]["series"]]
    assert len(names) == 7 and names[-1] == "other"
    assert names[0] == "s0"
    assert p["data"]["series"][-1]["y"] == [4.0 + 3.0] * 5
    assert p["notes"]


def _forecast_pair(store, *, freq="MS", n_hist=24, horizon=3, with_fold=True, id_col="unique_id"):
    start = "2022-01-01"
    hist_ds = pd.date_range(start, periods=n_hist, freq=freq)
    hist = pd.DataFrame(
        {
            "unique_id": ["a"] * n_hist + ["b"] * n_hist,
            "ds": list(hist_ds) * 2,
            "y": [float(i) for i in range(n_hist)] + [float(100 + i) for i in range(n_hist)],
        }
    )
    hh = _save(store, pa.Table.from_pandas(hist), tag="fhist")
    fds = pd.date_range(hist_ds[-1], periods=horizon + 1, freq=freq)[1:]
    rows = []
    for uid, base in (("a", 30.0), ("b", 130.0)):
        for i, d in enumerate(fds):
            v = base + i
            rows.append(
                {
                    "unique_id": uid,
                    "ds": d,
                    "yhat": v,
                    "lo_80": v - 1,
                    "hi_80": v + 1,
                    "lo_95": v - 2,
                    "hi_95": v + 2,
                    "baseline": v - 0.5,
                }
            )
    fold_handle = None
    if with_fold:
        fold = pd.DataFrame(
            {
                "unique_id": ["a"] * 3 + ["b"] * 3,
                "ds": list(hist_ds[-3:]) * 2,
                "yhat": [1.0] * 6,
                "lo_80": [0.0] * 6,
                "hi_80": [2.0] * 6,
                "lo_95": [-1.0] * 6,
                "hi_95": [3.0] * 6,
                "actual": [1.5] * 6,
            }
        )
        fold_handle = _save(store, pa.Table.from_pandas(fold), produced_by="backtest_fold", tag="f")
    meta = {
        "input_handle": hh,
        "time_col": "ds",
        "value_col": "y",
        "id_col": id_col,
        "season_length": 12,
        "fold_handle": fold_handle,
        "model": "AutoETS",
        "origin": hist_ds[-1].isoformat(),
        "freq": freq,
        "headline_ids": ["b", "a"],
        "backtest_summary": {
            "best_model": "AutoETS",
            "mean_relative_mae": 0.8,
            "mean_mase": 0.9,
            "mean_coverage_80": 0.75,
            "mean_coverage_95": 0.9,
            "baseline_mean_mase": 1.1,
        },
        "beats_baseline": True,
        "windows_used": 3,
    }
    fh = _save(store, pa.Table.from_pandas(pd.DataFrame(rows)), produced_by="forecast", meta=meta, tag="fc")
    return fh


def test_forecast_fan_payload_single_series(store):
    fh = _forecast_pair(store)
    p = build_payload(_stored(ForecastFanChart(title="f", forecast_handle=fh, series_id="a")), store)
    d = p["data"]
    assert d["series_id"] == "a"
    assert len(d["history"]["x"]) == 24 and d["history"]["y"][-1] == 23.0
    assert d["forecast"]["yhat"] == [30.0, 31.0, 32.0]
    assert d["model"] == "AutoETS"
    assert d["scorecard"]["mase"] == 0.9 and d["scorecard"]["beats_baseline"] is True
    assert d["baseline"] is None and d["replay"] is None
    # default series: first headline id
    p2 = build_payload(_stored(ForecastFanChart(title="f", forecast_handle=fh)), store)
    assert p2["data"]["series_id"] == "b"


def test_forecast_fan_last_year_monthly_and_daily(store):
    fh = _forecast_pair(store)  # history 2022-01..2023-12, forecast 2024-01..03
    p = build_payload(_stored(ForecastFanChart(title="f", forecast_handle=fh, series_id="a")), store)
    ly = p["data"]["last_year"]
    assert ly["y"] == [12.0, 13.0, 14.0]  # 2023-01..03
    store2 = ResultStore(store._root.parent / "r2")
    fh2 = _forecast_pair(store2, freq="D", n_hist=400, horizon=3)
    p2 = build_payload(_stored(ForecastFanChart(title="f", forecast_handle=fh2, series_id="a")), store2)
    # first forecast ds = idx 400; minus 364 days = idx 36
    assert p2["data"]["last_year"]["y"] == [36.0, 37.0, 38.0]
    store3 = ResultStore(store._root.parent / "r3")
    fh3 = _forecast_pair(store3, freq="D", n_hist=100, horizon=2)
    p3 = build_payload(_stored(ForecastFanChart(title="f", forecast_handle=fh3, series_id="a")), store3)
    assert p3["data"]["last_year"]["y"] == [None, None]
    off = build_payload(
        _stored(ForecastFanChart(title="f", forecast_handle=fh, series_id="a", last_year=False)), store
    )
    assert off["data"]["last_year"] is None


def test_forecast_fan_replay_and_baseline_optional(store):
    fh = _forecast_pair(store)
    spec = ForecastFanChart(title="f", forecast_handle=fh, series_id="a", baseline=True, replay=True)
    d = build_payload(_stored(spec), store)["data"]
    assert d["baseline"]["y"] == [29.5, 30.5, 31.5]
    assert len(d["replay"]["x"]) == 3 and d["replay"]["actual"] == [1.5] * 3
    store2 = ResultStore(store._root.parent / "r2")
    fh2 = _forecast_pair(store2, with_fold=False)
    spec2 = ForecastFanChart(title="f", forecast_handle=fh2, series_id="a", replay=True)
    assert build_payload(_stored(spec2), store2)["data"]["replay"] is None


def test_forecast_without_chart_meta_raises(store):
    t = pa.table({"unique_id": ["a"], "ds": pd.to_datetime(["2024-01-01"]), "yhat": [1.0]})
    fh = _save(store, t, produced_by="forecast", meta={"id_col": "unique_id"})
    with pytest.raises(InvalidRequestError, match="predates chart support"):
        build_payload(_stored(ForecastFanChart(title="f", forecast_handle=fh)), store)


def test_forecast_fan_unknown_series_id_rejected(store):
    fh = _forecast_pair(store)
    spec = ForecastFanChart(title="f", forecast_handle=fh, series_id="zzz")
    with pytest.raises(InvalidRequestError) as e:
        build_payload(_stored(spec), store)
    assert "a, b" in e.value.hint


def test_forecast_fan_missing_input_handle_rejected(store):
    fh = _forecast_pair(store)
    card = store.card(fh)
    meta = dict(card.meta, input_handle="r_0000000000")
    t = store.load(fh)
    fh2 = _save(store, t, produced_by="forecast", meta=meta, tag="bad")
    with pytest.raises(InvalidRequestError, match="input history"):
        build_payload(_stored(ForecastFanChart(title="f", forecast_handle=fh2, series_id="a")), store)
    meta.pop("input_handle")
    fh3 = _save(store, t, produced_by="forecast", meta=meta, tag="bad2")
    with pytest.raises(InvalidRequestError, match="input history"):
        build_payload(_stored(ForecastFanChart(title="f", forecast_handle=fh3, series_id="a")), store)


def test_line_payload_drops_null_times(store):
    ds = pd.to_datetime(["2024-01-01", None, "2024-01-03"])
    h = _save(store, pa.table({"ds": ds, "v": [1.0, 2.0, 3.0]}))
    p = build_payload(_stored(LineChart(title="t", handle=h, x="ds", y="v")), store)
    assert len(p["data"]["series"][0]["x"]) == 2
    assert not any("NaT" in x for x in p["data"]["series"][0]["x"])
    assert "1 rows with no time dropped" in p["notes"]


def test_bar_top_n_and_other(store):
    t = pa.table({"c": [f"c{i}" for i in range(10)], "v": [float(i) for i in range(10)]})
    h = _save(store, t)
    spec = BarChart(title="b", handle=h, category="c", value="v", top_n=3)
    d = build_payload(_stored(spec), store)["data"]
    assert d["categories"] == ["c9", "c8", "c7", "c6"] or d["categories"][:3] == ["c9", "c8", "c7"]
    assert d["other"] is True
    assert d["categories"][-1] == "other"
    assert d["series"][0]["values"] == [9.0, 8.0, 7.0, sum(range(7)) * 1.0]
    d2 = build_payload(_stored(BarChart(title="b", handle=h, category="c", value="v")), store)["data"]
    assert d2["other"] is False and len(d2["categories"]) == 10


def test_heatmap_grid_shape(store):
    t = pa.table({"x": [1, 2, 3, 1, 2], "y": ["a", "a", "a", "b", "b"], "v": [1.0, 2, 3, 4, 5]})
    h = _save(store, t)
    d = build_payload(_stored(HeatmapChart(title="h", handle=h, x="x", y="y", value="v")), store)["data"]
    assert d["x"] == [1, 2, 3] and d["y"] == ["a", "b"]
    assert d["values"] == [[1.0, 2.0, 3.0], [4.0, 5.0, None]]


def test_histogram_bins_and_prebinned(store):
    vals = np.concatenate([np.zeros(10), np.ones(10) * 10])
    h = _save(store, pa.table({"v": vals}))
    d = build_payload(_stored(HistogramChart(title="h", handle=h, value="v")), store)["data"]
    assert len(d["counts"]) == 40 and len(d["edges"]) == 41 and sum(d["counts"]) == 20
    h2 = _save(store, pa.table({"b": [0.0, 1.0, 2.0], "n": [5, 6, 7]}), tag="pre")
    d2 = build_payload(_stored(HistogramChart(title="h", handle=h2, bin="b", count="n")), store)["data"]
    assert d2["counts"] == [5.0, 6.0, 7.0] and len(d2["edges"]) == 4


def test_box_quartiles(store):
    t = pa.table({"g": ["a"] * 5, "v": [1.0, 2, 3, 4, 5]})
    h = _save(store, t)
    g = build_payload(_stored(BoxChart(title="b", handle=h, group="g", value="v")), store)["data"]["groups"][0]
    assert (g["min"], g["q1"], g["median"], g["q3"], g["max"], g["n"]) == (1.0, 2.0, 3.0, 4.0, 5.0, 5)


def test_scatter_density_above_cap(store):
    rng = np.random.default_rng(1)
    small = _save(store, pa.table({"x": [1.0, 2.0], "y": [3.0, 4.0]}), tag="s")
    d = build_payload(_stored(ScatterChart(title="s", handle=small, x="x", y="y")), store)["data"]
    assert d["mode"] == "points" and d["x"] == [1.0, 2.0] and d["color"] is None
    n = 60_000
    big = _save(store, pa.table({"x": rng.random(n), "y": rng.random(n)}), tag="b")
    d = build_payload(_stored(ScatterChart(title="s", handle=big, x="x", y="y")), store)["data"]
    assert d["mode"] == "density"
    assert len(d["counts"]) == 200 and len(d["counts"][0]) == 200
    assert len(d["x_edges"]) == 201 and sum(map(sum, d["counts"])) == n


def test_kpi_change_pct(store):
    h = _save(store, pa.table({"v": [110.0], "c": [100.0]}))
    spec = KpiChart(title="k", handle=h, value="v", label="Trips", comparison="c")
    d = build_payload(_stored(spec), store)["data"]
    assert d["value"] == 110.0 and d["comparison"] == 100.0
    assert d["change_pct"] == pytest.approx(10.0) and d["trend"] is None
    h0 = _save(store, pa.table({"v": [1.0], "c": [0.0]}), tag="z")
    d0 = build_payload(
        _stored(KpiChart(title="k", handle=h0, value="v", label="L", comparison="c")), store
    )["data"]
    assert d0["change_pct"] is None


def test_small_multiples_fan_one_facet_per_id(store):
    fh = _forecast_pair(store)
    spec = SmallMultiplesChart(
        title="m", facet="unique_id", inner={"kind": "forecast_fan", "forecast_handle": fh}
    )
    d = build_payload(_stored(spec), store)["data"]
    assert d["inner_kind"] == "forecast_fan"
    assert [f["name"] for f in d["facets"]] == ["a", "b"] or {f["name"] for f in d["facets"]} == {"a", "b"}
    assert d["facets"][0]["data"]["series_id"] == d["facets"][0]["name"]
    assert d["y_range"][0] <= d["y_range"][1]


def test_small_multiples_caps_facets(store):
    rows = [{"ds": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i), "k": f"k{j:02d}", "v": float(j)}
            for j in range(15) for i in range(3)]
    h = _save(store, pa.Table.from_pandas(pd.DataFrame(rows)))
    spec = SmallMultiplesChart(
        title="m", facet="k", inner={"kind": "line", "handle": h, "x": "ds", "y": "v"}
    )
    p = build_payload(_stored(spec), store)
    assert len(p["data"]["facets"]) == 12 and p["notes"]


def test_no_nan_in_json(store):
    ds = pd.date_range("2024-01-01", periods=4, freq="D")
    h = _save(store, pa.table({"ds": ds, "v": [1.0, math.nan, 3.0, 4.0]}))
    p = build_payload(_stored(LineChart(title="t", handle=h, x="ds", y="v")), store)
    json.dumps(p, allow_nan=False)


def test_payloads_match_golden_fixtures(tmp_path):
    store = ResultStore(tmp_path / "g")
    ds = pd.date_range("2024-01-01", periods=6, freq="D")
    line_h = _save(
        store,
        pa.Table.from_pandas(
            pd.DataFrame(
                {"ds": list(ds) * 2, "s": ["x"] * 6 + ["y"] * 6, "v": [float(i) for i in range(6)] + [2.0 * i for i in range(6)]}
            )
        ),
        tag="line",
    )
    bar_h = _save(store, pa.table({"c": ["a", "b", "c", "d"], "v": [4.0, 3.0, 2.0, 1.0]}), tag="bar")
    heat_h = _save(
        store, pa.table({"x": [1, 2, 1, 2], "y": ["p", "p", "q", "q"], "v": [1.0, 2.0, 3.0, 4.0]}), tag="heat"
    )
    hist_h = _save(store, pa.table({"v": [float(i % 5) for i in range(20)]}), tag="hist")
    box_h = _save(
        store, pa.table({"g": ["a"] * 5 + ["b"] * 5, "v": [1.0, 2, 3, 4, 5, 2, 4, 6, 8, 10]}), tag="box"
    )
    sc_h = _save(store, pa.table({"x": [1.0, 2.0, 3.0], "y": [2.0, 1.0, 3.0], "k": ["u", "v", "u"]}), tag="sc")
    kpi_h = _save(store, pa.table({"v": [120.0], "c": [100.0]}), tag="kpi")
    fh = _forecast_pair(store)
    geo_dir = tmp_path / "geo_ds"
    (geo_dir / "geo").mkdir(parents=True)
    (geo_dir / "geo" / "areas.geojson").write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"area": name},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[x, 0], [x + 1, 0], [x + 1, 1], [x, 0]]],
                        },
                    }
                    for name, x in (("A", 0), ("B", 2))
                ],
            }
        )
    )
    geometries = {
        "areas": GeometrySpec(name="areas", file="geo/areas.geojson", key_property="area")
    }
    map_h = _save(store, pa.table({"k": ["A", "B", "Z"], "v": [1.0, 2.0, 3.0]}), tag="map")
    specs = {
        "line": LineChart(title="Line", handle=line_h, x="ds", y="v", series="s"),
        "area": AreaChart(title="Area", handle=line_h, x="ds", y="v", series="s", stacked=True),
        "forecast_fan": ForecastFanChart(
            title="Fan", forecast_handle=fh, series_id="a", baseline=True, replay=True
        ),
        "bar": BarChart(title="Bar", handle=bar_h, category="c", value="v", top_n=3),
        "heatmap": HeatmapChart(title="Heat", handle=heat_h, x="x", y="y", value="v"),
        "map": MapChart(title="Map", handle=map_h, key="k", value="v", geometry="areas"),
        "histogram": HistogramChart(title="Hist", handle=hist_h, value="v"),
        "box": BoxChart(title="Box", handle=box_h, group="g", value="v"),
        "scatter": ScatterChart(title="Scatter", handle=sc_h, x="x", y="y", color="k"),
        "kpi": KpiChart(title="Kpi", handle=kpi_h, value="v", label="Trips", comparison="c"),
        "small_multiples": SmallMultiplesChart(
            title="Multiples", facet="unique_id", inner={"kind": "forecast_fan", "forecast_handle": fh}
        ),
    }
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for kind, spec in specs.items():
        payload = build_payload(
            _stored(spec), store, width=1200, geometries=geometries, geometry_dir=geo_dir
        )
        text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        path = FIXTURES / f"{kind}.json"
        if os.environ.get("UPDATE_FIXTURES"):
            path.write_text(text, encoding="utf-8")
        assert path.read_text(encoding="utf-8") == text, kind
