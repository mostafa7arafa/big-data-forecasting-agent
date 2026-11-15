from langchain.tools import ToolRuntime

from analyst.charts import ChartSpec, chart_handles, validate_spec
from analyst.charts.spec import ForecastFanChart, SmallMultiplesChart
from analyst.context import AnalystContext
from analyst.tools._common import Result, data_tool, run_tool, service_for


def _facts(spec: ChartSpec, handles: list[str]) -> list[str]:
    """One line per derived fact the model should know about the stored chart."""
    if isinstance(spec, ForecastFanChart):
        series = f"series {spec.series_id}" if spec.series_id else "all series"
        year = "last year shown" if spec.last_year else "no last-year overlay"
        return [f"forecast_fan of {spec.forecast_handle} ({series}); {year}"]
    if isinstance(spec, SmallMultiplesChart):
        return [f"small_multiples of {spec.inner.kind} faceted by {spec.facet}"]
    return []


@data_tool("make_chart")
def make_chart(spec: ChartSpec, runtime: ToolRuntime[AnalystContext]) -> Result:
    """Declare a chart over result handles; the UI renders it. `spec` is one
    chart kind (its `kind` field picks the shape) whose handles and columns
    come from data tool results. Returns a chart id to list in the answer's
    `charts`."""

    def body() -> Result:
        service = service_for(runtime)
        validate_spec(spec, service.store, geometries={})
        ctx = runtime.context
        stored = service.charts.save(spec, dataset=ctx.dataset, cutoff=ctx.cutoff)
        handles = chart_handles(spec)
        title = spec.title
        lines = [f"chart {stored.chart_id} queued: {spec.kind} '{title}'"]
        lines += _facts(spec, handles)
        artifact = {"chart_id": stored.chart_id, "kind": spec.kind, "handles": handles}
        return "\n".join(lines), artifact

    return run_tool("make_chart", runtime, body)
