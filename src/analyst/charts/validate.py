from __future__ import annotations

from collections.abc import Mapping

from analyst.charts._inner import InnerArea, InnerBar, InnerFan, InnerLine, InnerSpec
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
from analyst.data.dataset import GeometrySpec
from analyst.data.results import (
    INTERVAL_HINT,
    ResultStore,
    SummaryCard,
    is_interval_type_str,
    is_numeric_type_str,
    is_temporal_type_str,
)
from analyst.errors import InvalidRequestError


def validate_spec(
    spec: ChartSpec,
    store: ResultStore,
    geometries: Mapping[str, GeometrySpec],
) -> None:
    for handle_id in chart_handles(spec):
        card = _require_handle(store, handle_id)
        if isinstance(spec, ForecastFanChart) and handle_id == spec.forecast_handle:
            _validate_forecast_fan(spec, store, card)
        elif isinstance(spec, SmallMultiplesChart) and spec.inner.kind == "forecast_fan":
            inner: InnerFan = spec.inner  # type: ignore[assignment]
            if handle_id == inner.forecast_handle:
                _validate_forecast_fan_inner(inner, store, card)

    if isinstance(spec, (LineChart, AreaChart)):
        card = _require_handle(store, spec.handle)
        _require_column(card, spec.handle, spec.x)
        _require_column(card, spec.handle, spec.y)
        _require_temporal(card, spec.handle, spec.x, role="x")
        _require_numeric(card, spec.handle, spec.y, role="y")
        if spec.series is not None:
            _require_column(card, spec.handle, spec.series)
    elif isinstance(spec, ForecastFanChart):
        pass
    elif isinstance(spec, BarChart):
        card = _require_handle(store, spec.handle)
        _require_column(card, spec.handle, spec.category)
        _require_column(card, spec.handle, spec.value)
        _require_numeric(card, spec.handle, spec.value, role="value")
        if spec.series is not None:
            _require_column(card, spec.handle, spec.series)
    elif isinstance(spec, HeatmapChart):
        card = _require_handle(store, spec.handle)
        for col in (spec.x, spec.y, spec.value):
            _require_column(card, spec.handle, col)
        _require_numeric(card, spec.handle, spec.value, role="value")
    elif isinstance(spec, MapChart):
        card = _require_handle(store, spec.handle)
        _require_column(card, spec.handle, spec.key)
        _require_column(card, spec.handle, spec.value)
        _require_numeric(card, spec.handle, spec.value, role="value")
        _require_geometry(spec.geometry, geometries)
    elif isinstance(spec, HistogramChart):
        card = _require_handle(store, spec.handle)
        if spec.value is not None:
            _require_column(card, spec.handle, spec.value)
            _require_numeric(card, spec.handle, spec.value, role="value")
        else:
            assert spec.bin is not None and spec.count is not None
            _require_column(card, spec.handle, spec.bin)
            _require_column(card, spec.handle, spec.count)
            _require_numeric(card, spec.handle, spec.count, role="count")
    elif isinstance(spec, BoxChart):
        card = _require_handle(store, spec.handle)
        _require_column(card, spec.handle, spec.group)
        _require_column(card, spec.handle, spec.value)
        _require_numeric(card, spec.handle, spec.value, role="value")
    elif isinstance(spec, ScatterChart):
        card = _require_handle(store, spec.handle)
        _require_column(card, spec.handle, spec.x)
        _require_column(card, spec.handle, spec.y)
        if spec.color is not None:
            _require_column(card, spec.handle, spec.color)
    elif isinstance(spec, KpiChart):
        card = _require_handle(store, spec.handle)
        _require_column(card, spec.handle, spec.value)
        _require_numeric(card, spec.handle, spec.value, role="value")
        if spec.comparison is not None:
            _require_column(card, spec.handle, spec.comparison)
            _require_numeric(card, spec.handle, spec.comparison, role="comparison")
        if spec.trend_handle is not None:
            trend_card = _require_handle(store, spec.trend_handle)
            _require_column(trend_card, spec.trend_handle, spec.trend_x)  # type: ignore[arg-type]
            _require_column(trend_card, spec.trend_handle, spec.trend_y)  # type: ignore[arg-type]
            _require_temporal(trend_card, spec.trend_handle, spec.trend_x, role="trend_x")  # type: ignore[arg-type]
            _require_numeric(trend_card, spec.trend_handle, spec.trend_y, role="trend_y")  # type: ignore[arg-type]
    elif isinstance(spec, SmallMultiplesChart):
        _validate_inner(spec.inner, store, facet=spec.facet)


def _validate_inner(
    inner: InnerSpec,
    store: ResultStore,
    *,
    facet: str | None = None,
) -> None:
    if isinstance(inner, (InnerLine, InnerBar, InnerArea)):
        card = _require_handle(store, inner.handle)
        if facet is not None:
            _require_column(card, inner.handle, facet)
    if isinstance(inner, InnerLine):
        spec = LineChart(
            kind="line",
            title=".",
            handle=inner.handle,
            x=inner.x,
            y=inner.y,
            series=inner.series,
        )
        validate_spec(spec, store, {})
    elif isinstance(inner, InnerBar):
        spec = BarChart(
            kind="bar",
            title=".",
            handle=inner.handle,
            category=inner.category,
            value=inner.value,
            series=inner.series,
            orientation=inner.orientation,
            mode=inner.mode,
            top_n=inner.top_n,
        )
        validate_spec(spec, store, {})
    elif isinstance(inner, InnerArea):
        spec = AreaChart(
            kind="area",
            title=".",
            handle=inner.handle,
            x=inner.x,
            y=inner.y,
            series=inner.series,
            stacked=inner.stacked,
        )
        validate_spec(spec, store, {})
    elif isinstance(inner, InnerFan):
        _require_handle(store, inner.forecast_handle)
        card = store.card(inner.forecast_handle)
        _validate_forecast_fan_inner(inner, store, card)


def _validate_forecast_fan(
    spec: ForecastFanChart,
    store: ResultStore,
    card: SummaryCard,
) -> None:
    if card.produced_by != "forecast":
        raise InvalidRequestError(
            f"handle {spec.forecast_handle} was not produced by forecast",
            hint="use a forecast result handle for forecast_fan",
        )
    if spec.series_id is not None:
        _require_series_id(store, spec.forecast_handle, spec.series_id, card)


def _validate_forecast_fan_inner(
    spec: InnerFan,
    store: ResultStore,
    card: SummaryCard,
) -> None:
    if card.produced_by != "forecast":
        raise InvalidRequestError(
            f"handle {spec.forecast_handle} was not produced by forecast",
            hint="use a forecast result handle for forecast_fan",
        )
    if spec.series_id is not None:
        _require_series_id(store, spec.forecast_handle, spec.series_id, card)


def _require_series_id(
    store: ResultStore,
    handle_id: str,
    series_id: str,
    card: SummaryCard,
) -> None:
    id_col = str(card.meta.get("id_col", "unique_id"))
    if id_col not in {name for name, _ in card.columns}:
        raise InvalidRequestError(
            f"forecast handle {handle_id} has no id column {id_col!r}",
            hint=_column_hint(card),
        )
    table = store.load(handle_id)
    values = {str(v) for v in table.column(id_col).to_pylist() if v is not None}
    if series_id not in values:
        listed = ", ".join(sorted(values)[:20])
        raise InvalidRequestError(
            f"unknown series_id {series_id!r} on forecast handle {handle_id}",
            hint=f"unique_id values include: {listed}",
        )


def _require_handle(store: ResultStore, handle_id: str) -> SummaryCard:
    if not store.exists(handle_id):
        raise InvalidRequestError(
            f"unknown result handle {handle_id}",
            hint="use a handle id returned by an earlier tool call",
        )
    return store.card(handle_id)


def _column_names(card: SummaryCard) -> list[str]:
    return [name for name, _ in card.columns]


def _column_hint(card: SummaryCard) -> str:
    return f"columns on {card.handle_id}: {', '.join(_column_names(card))}"


def _column_type(card: SummaryCard, column: str) -> str:
    for name, typ in card.columns:
        if name == column:
            return typ
    raise AssertionError(f"missing column {column!r}")


def _require_column(card: SummaryCard, handle_id: str, column: str) -> None:
    names = _column_names(card)
    if column not in names:
        raise InvalidRequestError(
            f"column {column!r} not found on handle {handle_id}",
            hint=_column_hint(card),
        )


def _require_temporal(
    card: SummaryCard,
    handle_id: str,
    column: str,
    *,
    role: str,
) -> None:
    typ = _column_type(card, column)
    if not is_temporal_type_str(typ):
        raise InvalidRequestError(
            f"column {column!r} on handle {handle_id} is not a time axis ({role})",
            hint=INTERVAL_HINT
            if is_interval_type_str(typ)
            else "pick a date or timestamp column for the time axis",
        )


def _require_numeric(
    card: SummaryCard,
    handle_id: str,
    column: str,
    *,
    role: str,
) -> None:
    typ = _column_type(card, column)
    if not is_numeric_type_str(typ):
        raise InvalidRequestError(
            f"column {column!r} on handle {handle_id} is not numeric ({role})",
            hint=INTERVAL_HINT if is_interval_type_str(typ) else "pick a numeric column",
        )


def _require_geometry(name: str, geometries: Mapping[str, GeometrySpec]) -> None:
    if not geometries:
        raise InvalidRequestError(
            "this dataset has no map geometry",
            hint="this dataset has no map geometry",
        )
    if name not in geometries:
        available = ", ".join(sorted(geometries))
        raise InvalidRequestError(
            f"this dataset has no geometry named {name!r}",
            hint=f"this dataset has no geometry named {name!r}; available: {available}",
        )