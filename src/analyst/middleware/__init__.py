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


def analyst_middleware(settings: Any = None) -> list[AgentMiddleware]:
    """Middleware for the analyst agent, outermost first.

    ``settings`` is accepted for symmetry with the other factories; the
    middleware itself has no per-dataset configuration.
    """
    del settings
    return [
        ClearOldToolResults(),
        ContextMeter(),
        OpenRouterCost(),
        ToolCallLimitMiddleware(run_limit=TOOL_CALL_LIMIT),
        ModelCallLimitMiddleware(run_limit=MODEL_CALL_LIMIT),
    ]
