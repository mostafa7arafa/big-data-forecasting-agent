from datetime import date

from langchain.tools import ToolRuntime
from pydantic import Field

from analyst.context import AnalystContext
from analyst.semantic.compiler import MetricQuery
from analyst.tools._common import Result, card_result, data_tool, run_tool, service_for


class MetricQueryArgs(MetricQuery):
    """Model-facing ``MetricQuery``: ``time_range`` is a two-item list.

    A tuple renders as JSON-schema ``prefixItems`` without ``items``, which the
    OpenAI function-schema validator rejects; the tool converts back to
    ``MetricQuery`` before compiling.
    """

    time_range: list[date] | None = Field(
        default=None,
        min_length=2,
        max_length=2,
        description="[start, end) as two ISO dates; end is exclusive",
    )


@data_tool("query_metrics", args_schema=MetricQueryArgs)
def query_metrics(runtime: ToolRuntime[AnalystContext], **kwargs) -> Result:
    """Run a governed metric query (metrics, dimensions, filters, grain, time range).

    Returns a summary card and a result handle; prefer this over run_sql.
    """
    return run_tool(
        "query_metrics",
        runtime,
        lambda: card_result(service_for(runtime).query_metrics(MetricQuery(**kwargs))),
    )
