from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

_EXTRA_FORBID = ConfigDict(extra="forbid")


class InnerLine(BaseModel):
    model_config = _EXTRA_FORBID

    kind: Literal["line"] = "line"
    handle: str
    x: str
    y: str
    series: str | None = None


class InnerBar(BaseModel):
    model_config = _EXTRA_FORBID

    kind: Literal["bar"] = "bar"
    handle: str
    category: str
    value: str
    series: str | None = None
    orientation: Literal["vertical", "horizontal"] = "vertical"
    mode: Literal["grouped", "stacked"] = "grouped"
    top_n: int | None = Field(None, ge=1, le=50)


class InnerArea(BaseModel):
    model_config = _EXTRA_FORBID

    kind: Literal["area"] = "area"
    handle: str
    x: str
    y: str
    series: str | None = None
    stacked: bool = False


class InnerFan(BaseModel):
    model_config = _EXTRA_FORBID

    kind: Literal["forecast_fan"] = "forecast_fan"
    forecast_handle: str
    series_id: str | None = None
    last_year: bool = True
    baseline: bool = False
    replay: bool = False


InnerSpec = Annotated[
    InnerLine | InnerBar | InnerArea | InnerFan,
    Field(discriminator="kind"),
]