"""Shared plumbing: service lookup, progress events and error shaping."""

import logging
from collections.abc import Callable
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool, ToolException, tool

from analyst.context import AnalystContext
from analyst.data.registry import get_service
from analyst.data.results import SummaryCard
from analyst.data.service import DataService
from analyst.errors import AnalystError

logger = logging.getLogger(__name__)

Result = tuple[str, dict[str, Any] | None]


def service_for(runtime: ToolRuntime[AnalystContext]) -> DataService:
    ctx = runtime.context
    return get_service(ctx.dataset, ctx.cutoff)


def progress(runtime: ToolRuntime[AnalystContext], tool: str, message: str) -> None:
    runtime.stream_writer({"type": "progress", "tool": tool, "message": message})


def card_result(card: SummaryCard) -> Result:
    artifact = {
        "handle_id": card.handle_id,
        "produced_by": card.produced_by,
        "governed": card.governed,
    }
    return card.render(), artifact


def run_tool(
    name: str, runtime: ToolRuntime[AnalystContext], body: Callable[[], Result]
) -> Result:
    """Run a tool body; turn failures into ``status="error"`` tool messages.

    Errors are raised as ``ToolException`` and the tools set
    ``handle_tool_error=True``, so langchain emits a ToolMessage with
    ``status="error"`` carrying the text.
    """
    progress(runtime, name, f"{name} started")
    try:
        return body()
    except AnalystError as exc:
        raise ToolException(f"ERROR: {exc.message}\nHINT: {exc.hint}") from exc
    except ToolException:
        raise
    except Exception as exc:
        logger.exception("internal error in %s", name)
        raise ToolException(f"ERROR: internal error in {name}") from exc


def data_tool(name: str, **kwargs: Any) -> Callable[[Callable[..., Result]], BaseTool]:
    """``@tool`` with the content_and_artifact format and error messages on."""

    def wrap(fn: Callable[..., Result]) -> BaseTool:
        built = tool(name, response_format="content_and_artifact", **kwargs)(fn)
        built.handle_tool_error = True
        return built

    return wrap
