"""Agent middleware: tool-result clearing, context metering, cost and limits."""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
)

from analyst.middleware.clearing import ClearOldToolResults
from analyst.middleware.cost import OpenRouterCost
from analyst.middleware.metering import ContextMeter, ContextSize, measure

__all__ = [
    "ClearOldToolResults",
    "ContextMeter",
    "ContextSize",
    "OpenRouterCost",
    "analyst_middleware",
    "measure",
]

TOOL_CALL_LIMIT = 40
MODEL_CALL_LIMIT = 25


def analyst_middleware(
    settings: Any = None, *, record_trace: bool = False
) -> list[AgentMiddleware]:
    """Middleware for the analyst agent, outermost first.

    ``settings`` is accepted for symmetry with the other factories; the
    middleware itself has no per-dataset configuration. The trace middleware
    is outermost so its timing covers the whole call and the stamped message
    already carries the provider cost. ``record_trace`` also appends the run
    trace to the agent's ``trace`` state (for agents with no answer node).
    """
    del settings
    # Imported here: analyst.trace imports analyst.middleware.compaction, so a
    # module-level import would make the two packages import each other.
    from analyst.trace.middleware import RunTraceRecorder, TraceMiddleware

    return [
        RunTraceRecorder() if record_trace else TraceMiddleware(),
        ClearOldToolResults(),
        ContextMeter(),
        OpenRouterCost(),
        ToolCallLimitMiddleware(run_limit=TOOL_CALL_LIMIT),
        ModelCallLimitMiddleware(run_limit=MODEL_CALL_LIMIT),
    ]
