from typing import Literal

from langchain.tools import ToolRuntime

from analyst.context import AnalystContext
from analyst.errors import InvalidRequestError, UnknownNameError
from analyst.state import ChartSpec
from analyst.tools._common import Result, data_tool, run_tool, service_for


@data_tool("make_chart")
def make_chart(
    kind: Literal["line", "bar"],
    title: str,
    handle: str,
    x: str,
    y: str,
    runtime: ToolRuntime[AnalystContext],
    series: str | None = None,
    forecast_handle: str | None = None,
    bands: list[int] | None = None,
    last_year_handle: str | None = None,
) -> Result:
    """Declare a chart over result handles; the UI renders it. x, y and series
    are column names of `handle`. Optionally overlay a forecast handle with
    interval bands (80, 95) or last year's handle."""

    def body() -> Result:
        service = service_for(runtime)
        for ref in (handle, forecast_handle, last_year_handle):
            if ref is not None and not service.store.exists(ref):
                raise UnknownNameError(
                    f"unknown handle {ref!r}",
                    hint="use a handle_id returned by a data tool",
                )
        columns = {name for name, _ in service.store.card(handle).columns}
        for col in (x, y, series):
            if col is not None and col not in columns:
                raise InvalidRequestError(
                    f"column {col!r} not in {handle}; columns: {sorted(columns)}",
                    hint="pick x, y and series from the handle's columns",
                )
        spec = ChartSpec(
            kind=kind,
            title=title,
            handle=handle,
            x=x,
            y=y,
            series=series,
            forecast_handle=forecast_handle,
            bands=bands or [],
            last_year_handle=last_year_handle,
        )
        return f"chart queued: {kind} '{title}' of {y} by {x} from {handle}", {
            "chart": spec.model_dump()
        }

    return run_tool("make_chart", runtime, body)
