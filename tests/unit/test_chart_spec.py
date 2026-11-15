from __future__ import annotations

from datetime import date, datetime

import pyarrow as pa
import pytest
from pydantic import TypeAdapter, ValidationError

from analyst.charts.spec import (
    AreaChart,
    BarChart,
    BoxChart,
    ChartSpec,
    ForecastFanChart,
    HeatmapChart,
    HistogramChart,
    KpiChart,
    LineChart,
    MapChart,
    ScatterChart,
    SmallMultiplesChart,
    chart_handles,
)
from analyst.charts.store import ChartStore
from analyst.charts.validate import validate_spec
from analyst.data.results import ResultStore
from analyst.errors import InvalidRequestError, UnknownNameError

_ADAPTER = TypeAdapter(ChartSpec)


def _minimal_specs() -> list[tuple[dict, type]]:
    handle = "r_aabbccddee"
    return [
        ({"kind": "line", "title": "T", "handle": handle, "x": "d", "y": "v"}, LineChart),
        ({"kind": "area", "title": "T", "handle": handle, "x": "d", "y": "v"}, AreaChart),
        (
            {"kind": "forecast_fan", "title": "T", "forecast_handle": handle},
            ForecastFanChart,
        ),
        (
            {"kind": "bar", "title": "T", "handle": handle, "category": "c", "value": "v"},
            BarChart,
        ),
        (
            {"kind": "heatmap", "title": "T", "handle": handle, "x": "a", "y": "b", "value": "v"},
            HeatmapChart,
        ),
        (
            {
                "kind": "map",
                "title": "T",
                "handle": handle,
                "key": "k",
                "value": "v",
                "geometry": "zones",
            },
            MapChart,
        ),
        ({"kind": "histogram", "title": "T", "handle": handle, "value": "v"}, HistogramChart),
        (
            {"kind": "box", "title": "T", "handle": handle, "group": "g", "value": "v"},
            BoxChart,
        ),
        ({"kind": "scatter", "title": "T", "handle": handle, "x": "a", "y": "b"}, ScatterChart),
        (
            {"kind": "kpi", "title": "T", "handle": handle, "value": "v", "label": "L"},
            KpiChart,
        ),
        (
            {
                "kind": "small_multiples",
                "title": "T",
                "facet": "f",
                "inner": {"kind": "line", "handle": handle, "x": "d", "y": "v"},
            },
            SmallMultiplesChart,
        ),
    ]


def test_union_parses_each_kind():
    for data, cls in _minimal_specs():
        parsed = _ADAPTER.validate_python(data)
        assert isinstance(parsed, cls)
        assert parsed.kind == data["kind"]


def test_extra_fields_rejected():
    with pytest.raises(ValidationError):
        _ADAPTER.validate_python(
            {
                "kind": "line",
                "title": "T",
                "handle": "r_aabbccddee",
                "x": "d",
                "y": "v",
                "colour": "red",
            }
        )


def test_histogram_needs_value_or_bin_count():
    with pytest.raises(ValidationError):
        HistogramChart(
            kind="histogram",
            title="T",
            handle="r_aabbccddee",
            value="v",
            bin="b",
            count="c",
        )
    with pytest.raises(ValidationError):
        HistogramChart(kind="histogram", title="T", handle="r_aabbccddee")
    HistogramChart(kind="histogram", title="T", handle="r_aabbccddee", value="v")
    HistogramChart(
        kind="histogram", title="T", handle="r_aabbccddee", bin="b", count="c"
    )


def test_small_multiples_fan_requires_unique_id_facet():
    with pytest.raises(ValidationError):
        SmallMultiplesChart(
            kind="small_multiples",
            title="T",
            facet="borough",
            inner={
                "kind": "forecast_fan",
                "forecast_handle": "r_aabbccddee",
            },
        )
    SmallMultiplesChart(
        kind="small_multiples",
        title="T",
        facet="unique_id",
        inner={"kind": "forecast_fan", "forecast_handle": "r_aabbccddee"},
    )


def test_chart_handles_collects_all_refs():
    assert chart_handles(
        KpiChart(
            kind="kpi",
            title="t",
            handle="r_aaaaaaaaaa",
            value="v",
            label="l",
            trend_handle="r_bbbbbbbbbb",
            trend_x="d",
            trend_y="v",
        )
    ) == ["r_aaaaaaaaaa", "r_bbbbbbbbbb"]


def test_store_is_content_addressed(tmp_path):
    store = ChartStore(tmp_path)
    spec = LineChart(
        kind="line", title="T", handle="r_aabbccddee", x="d", y="v"
    )
    a = store.save(spec, dataset="demo", cutoff=None)
    b = store.save(spec, dataset="demo", cutoff=None)
    assert a.chart_id == b.chart_id
    c = store.save(spec, dataset="demo", cutoff=date(2024, 1, 1))
    assert c.chart_id != a.chart_id


def test_store_rejects_bad_ids(tmp_path):
    store = ChartStore(tmp_path)
    spec = LineChart(
        kind="line", title="T", handle="r_aabbccddee", x="d", y="v"
    )
    stored = store.save(spec, dataset="demo", cutoff=None)
    charts_dir = tmp_path / "charts"
    assert charts_dir.is_dir()
    only_file = charts_dir / f"{stored.chart_id}.json"

    for bad in ("../x", "c_ZZ", "c_" + "a" * 11):
        with pytest.raises(UnknownNameError):
            store.get(bad)
        with pytest.raises(UnknownNameError):
            store.exists(bad)
    assert list(charts_dir.iterdir()) == [only_file]


@pytest.fixture
def store_with_handle(tmp_path):
    table = pa.table(
        {
            "ds": pa.array(
                [datetime(2024, 1, 1), datetime(2024, 1, 2)], type=pa.timestamp("us")
            ),
            "v": pa.array([1.0, 2.0], type=pa.float64()),
            "cat": pa.array(["a", "b"], type=pa.string()),
            "zone": pa.array([1, 2], type=pa.int64()),
            "bin": pa.array([0.0, 1.0], type=pa.float64()),
            "count": pa.array([10, 20], type=pa.int64()),
        }
    )
    store = ResultStore(tmp_path)
    card = store.save(
        table,
        produced_by="run_sql",
        source="select",
        governed=True,
        elapsed_s=0.01,
    )
    return store, card.handle_id


@pytest.fixture
def forecast_handle(tmp_path):
    table = pa.table(
        {
            "unique_id": pa.array(["series_a", "series_b"], type=pa.string()),
            "ds": pa.array(
                [datetime(2024, 1, 1), datetime(2024, 1, 2)], type=pa.timestamp("us")
            ),
            "yhat": pa.array([1.0, 2.0], type=pa.float64()),
        }
    )
    store = ResultStore(tmp_path)
    card = store.save(
        table,
        produced_by="forecast",
        source="forecast v2",
        governed=True,
        elapsed_s=0.01,
        meta={"id_col": "unique_id"},
        index=False,
    )
    sql_card = store.save(
        pa.table({"x": [1]}),
        produced_by="run_sql",
        source="q",
        governed=True,
        elapsed_s=0.01,
        index=False,
    )
    return store, card.handle_id, sql_card.handle_id


def test_validate_rejects_unknown_column_with_hint(store_with_handle):
    store, handle = store_with_handle
    spec = LineChart(kind="line", title="T", handle=handle, x="missing", y="v")
    with pytest.raises(InvalidRequestError) as exc:
        validate_spec(spec, store, {})
    assert "ds" in exc.value.hint
    assert "v" in exc.value.hint


def test_validate_rejects_non_time_x_for_line(store_with_handle):
    store, handle = store_with_handle
    spec = LineChart(kind="line", title="T", handle=handle, x="cat", y="v")
    with pytest.raises(InvalidRequestError) as exc:
        validate_spec(spec, store, {})
    assert exc.value.hint


def test_validate_rejects_unknown_geometry(store_with_handle):
    store, handle = store_with_handle
    spec = MapChart(
        kind="map",
        title="T",
        handle=handle,
        key="zone",
        value="v",
        geometry="zones",
    )
    with pytest.raises(InvalidRequestError) as exc:
        validate_spec(spec, store, {})
    assert "geometry" in exc.value.hint.lower() or "map" in exc.value.hint.lower()


@pytest.mark.parametrize(
    "spec",
    [
        LineChart(kind="line", title="T", handle="H", x="ds", y="v"),
        AreaChart(kind="area", title="T", handle="H", x="ds", y="v"),
        BarChart(kind="bar", title="T", handle="H", category="cat", value="v"),
        HeatmapChart(kind="heatmap", title="T", handle="H", x="cat", y="zone", value="v"),
        HistogramChart(kind="histogram", title="T", handle="H", value="v"),
        HistogramChart(kind="histogram", title="T", handle="H", bin="bin", count="count"),
        BoxChart(kind="box", title="T", handle="H", group="cat", value="v"),
        ScatterChart(kind="scatter", title="T", handle="H", x="v", y="zone"),
        MapChart(
            kind="map",
            title="T",
            handle="H",
            key="zone",
            value="v",
            geometry="zones",
        ),
        SmallMultiplesChart(
            kind="small_multiples",
            title="T",
            facet="cat",
            inner={"kind": "line", "handle": "H", "x": "ds", "y": "v"},
        ),
    ],
    ids=[
        "line",
        "area",
        "bar",
        "heatmap",
        "histogram_value",
        "histogram_binned",
        "box",
        "scatter",
        "map",
        "small_multiples_line",
    ],
)
def test_validate_accepts_valid_specs(store_with_handle, spec):
    store, handle = store_with_handle
    if isinstance(spec, SmallMultiplesChart):
        inner = spec.inner.model_copy(update={"handle": handle})
        spec = spec.model_copy(update={"inner": inner})
    else:
        spec = spec.model_copy(update={"handle": handle})
    geo = {"zones": {}} if spec.kind == "map" else {}
    validate_spec(spec, store, geo)


def test_validate_kpi_with_trend_passes(store_with_handle):
    store, handle = store_with_handle
    spec = KpiChart(
        kind="kpi",
        title="T",
        handle=handle,
        value="v",
        label="L",
        trend_handle=handle,
        trend_x="ds",
        trend_y="v",
    )
    validate_spec(spec, store, {})


def test_validate_forecast_fan_rejects_wrong_produced_by(forecast_handle):
    store, _forecast, sql = forecast_handle
    spec = ForecastFanChart(kind="forecast_fan", title="T", forecast_handle=sql)
    with pytest.raises(InvalidRequestError) as exc:
        validate_spec(spec, store, {})
    assert "forecast" in str(exc.value).lower() or "forecast" in exc.value.hint.lower()


def test_validate_forecast_fan_rejects_unknown_series_id(forecast_handle):
    store, forecast, _sql = forecast_handle
    spec = ForecastFanChart(
        kind="forecast_fan", title="T", forecast_handle=forecast, series_id="nope"
    )
    with pytest.raises(InvalidRequestError) as exc:
        validate_spec(spec, store, {})
    assert exc.value.hint


def test_validate_forecast_fan_accepts_series_id(forecast_handle):
    store, forecast, _sql = forecast_handle
    spec = ForecastFanChart(
        kind="forecast_fan", title="T", forecast_handle=forecast, series_id="series_a"
    )
    validate_spec(spec, store, {})


def test_validate_small_multiples_rejects_bad_facet(store_with_handle):
    store, handle = store_with_handle
    spec = SmallMultiplesChart(
        kind="small_multiples",
        title="T",
        facet="missing_facet",
        inner={"kind": "line", "handle": handle, "x": "ds", "y": "v"},
    )
    with pytest.raises(InvalidRequestError) as exc:
        validate_spec(spec, store, {})
    assert "ds" in exc.value.hint
    assert "cat" in exc.value.hint