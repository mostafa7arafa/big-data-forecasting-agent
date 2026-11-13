from langchain.tools import ToolRuntime

from analyst.context import AnalystContext
from analyst.memory.knowledge import read_topic
from analyst.tools._common import Result, data_tool, run_tool, service_for


@data_tool("read_knowledge")
def read_knowledge(topic: str, runtime: ToolRuntime[AnalystContext]) -> Result:
    """Read a knowledge topic: metrics, dimensions, quirks, context_facts,
    forecasting or recap. Cutoff-aware."""
    return run_tool(
        "read_knowledge",
        runtime,
        lambda: (
            read_topic(
                service_for(runtime).memory, topic, cutoff=runtime.context.cutoff
            ),
            None,
        ),
    )
