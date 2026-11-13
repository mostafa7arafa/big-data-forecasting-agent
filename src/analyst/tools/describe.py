from langchain.tools import ToolRuntime

from analyst.context import AnalystContext
from analyst.data.results import is_result_handle_id
from analyst.tools._common import Result, card_result, data_tool, run_tool, service_for


@data_tool("describe")
def describe(target: str, runtime: ToolRuntime[AnalystContext]) -> Result:
    """Describe 'metrics', 'dimensions', 'dimension:<name>', 'schema',
    'coverage', or a result handle id (returns its summary card)."""

    def body() -> Result:
        service = service_for(runtime)
        text = service.describe(target)
        if is_result_handle_id(target) and service.store.exists(target):
            return card_result(service.store.card(target))
        return text, None

    return run_tool("describe", runtime, body)
