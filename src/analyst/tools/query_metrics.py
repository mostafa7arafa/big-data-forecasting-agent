from langchain.tools import ToolRuntime

from analyst.context import AnalystContext
from analyst.semantic.compiler import MetricQuery
from analyst.tools._common import Result, card_result, data_tool, run_tool, service_for


@data_tool("query_metrics", args_schema=MetricQuery)
def query_metrics(runtime: ToolRuntime[AnalystContext], **kwargs) -> Result:
    """Run a governed metric query (metrics, dimensions, filters, grain, time range).

    Returns a summary card and a result handle; prefer this over run_sql.
    """
    return run_tool(
        "query_metrics",
        runtime,
        lambda: card_result(service_for(runtime).query_metrics(MetricQuery(**kwargs))),
    )
