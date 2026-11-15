from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from analyst.charts._inner import InnerSpec

_EXTRA_FORBID = ConfigDict(extra="forbid")


class LineChart(BaseModel):
    model_config = _EXTRA_FORBID

    kind: Literal["line"] = "line"
    title: str = Field(min_length=1, max_length=120)
    handle: str
    x: str
    y: str
    series: str | None = None


class AreaChart(BaseModel):
    model_config = _EXTRA_FORBID

    kind: Literal["area"] = "area"
    title: str = Field(min_length=1, max_length=120)
    handle: str
    x: str
    y: str
    series: str | None = None
    stacked: bool = False


class ForecastFanChart(BaseModel):
    model_config = _EXTRA_FORBID

    kind: Literal["forecast_fan"] = "forecast_fan"
    title: str = Field(min_length=1, max_length=120)
    forecast_handle: str
    series_id: str | None = None
    last_year: bool = True
    baseline: bool = False
    replay: bool = False


class BarChart(BaseModel):
    model_config = _EXTRA_FORBID

    kind: Literal["bar"] = "bar"
    title: str = Field(min_length=1, max_length=120)
    handle: str
    category: str
    value: str
    series: str | None = None
    orientation: Literal["vertical", "horizontal"] = "vertical"
    mode: Literal["grouped", "stacked"] = "grouped"
    top_n: int | None = Field(None, ge=1, le=50)


class HeatmapChart(BaseModel):
    model_config = _EXTRA_FORBID

    kind: Literal["heatmap"] = "heatmap"
    title: str = Field(min_length=1, max_length=120)
    handle: str
    x: str
    y: str
    value: str


class MapChart(BaseModel):
    model_config = _EXTRA_FORBID

    kind: Literal["map"] = "map"
    title: str = Field(min_length=1, max_length=120)
    handle: str
    key: str
    value: str
    geometry: str


class HistogramChart(BaseModel):
    model_config = _EXTRA_FORBID

    kind: Literal["histogram"] = "histogram"
    title: str = Field(min_length=1, max_length=120)
    handle: str
    value: str | None = None
    bin: str | None = None
    count: str | None = None

    @model_validator(mode="after")
    def _value_or_bin_count(self) -> HistogramChart:
        has_value = self.value is not None
        has_bin = self.bin is not None
        has_count = self.count is not None
        if has_value and (has_bin or has_count):
            raise ValueError("histogram accepts either value or bin+count, not both")
        if has_value:
            return self
        if has_bin and has_count:
            return self
        raise ValueError("histogram needs value or both bin and count")


class BoxChart(BaseModel):
    model_config = _EXTRA_FORBID

    kind: Literal["box"] = "box"
    title: str = Field(min_length=1, max_length=120)
    handle: str
    group: str
    value: str


class ScatterChart(BaseModel):
    model_config = _EXTRA_FORBID

    kind: Literal["scatter"] = "scatter"
    title: str = Field(min_length=1, max_length=120)
    handle: str
    x: str
    y: str
    color: str | None = None


class KpiChart(BaseModel):
    model_config = _EXTRA_FORBID

    kind: Literal["kpi"] = "kpi"
    title: str = Field(min_length=1, max_length=120)
    handle: str
    value: str
    label: str = Field(max_length=60)
    comparison: str | None = None
    trend_handle: str | None = None
    trend_x: str | None = None
    trend_y: str | None = None

    @model_validator(mode="after")
    def _trend_all_or_none(self) -> KpiChart:
        trend = (self.trend_handle, self.trend_x, self.trend_y)
        if any(t is not None for t in trend) and not all(t is not None for t in trend):
            raise ValueError("kpi trend_handle, trend_x and trend_y must all be set or all omitted")
        return self


class SmallMultiplesChart(BaseModel):
    model_config = _EXTRA_FORBID

    kind: Literal["small_multiples"] = "small_multiples"
    title: str = Field(min_length=1, max_length=120)
    facet: str
    inner: InnerSpec

    @model_validator(mode="after")
    def _fan_needs_unique_id_facet(self) -> SmallMultiplesChart:
        if self.inner.kind == "forecast_fan" and self.facet != "unique_id":
            raise ValueError("small_multiples of forecast_fan requires facet unique_id")
        return self


ChartSpec = Annotated[
    LineChart
    | AreaChart
    | ForecastFanChart
    | BarChart
    | HeatmapChart
    | MapChart
    | HistogramChart
    | BoxChart
    | ScatterChart
    | KpiChart
    | SmallMultiplesChart,
    Field(discriminator="kind"),
]


def chart_handles(spec: ChartSpec) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []

    def add(handle: str | None) -> None:
        if handle is not None and handle not in seen:
            seen.add(handle)
            out.append(handle)

    if isinstance(spec, LineChart | AreaChart | BarChart | HeatmapChart | MapChart):
        add(spec.handle)
    elif isinstance(spec, ForecastFanChart):
        add(spec.forecast_handle)
    elif isinstance(spec, HistogramChart | BoxChart | ScatterChart):
        add(spec.handle)
    elif isinstance(spec, KpiChart):
        add(spec.handle)
        add(spec.trend_handle)
    elif isinstance(spec, SmallMultiplesChart):
        inner = spec.inner
        if inner.kind == "line" or inner.kind == "bar" or inner.kind == "area":
            add(inner.handle)
        else:
            add(inner.forecast_handle)
    return out