from langchain.tools import ToolRuntime

from analyst.context import AnalystContext
from analyst.tools._common import Result, card_result, data_tool, run_tool, service_for


@data_tool("run_sql")
def run_sql(sql: str, runtime: ToolRuntime[AnalystContext]) -> Result:
    """Run one read-only SELECT over the dataset views (ungoverned escape hatch).

    Use only when query_metrics cannot express the question. Returns a summary
    card and a result handle.
    """
    return run_tool(
        "run_sql", runtime, lambda: card_result(service_for(runtime).run_sql(sql))
    )
