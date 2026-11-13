"""Agent middleware: tool-result clearing, context metering, cost and limits."""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
)

from analyst.middleware.clearing import ClearOldToolResults
from analyst.middleware.compaction import CompactionMiddleware
from analyst.middleware.cost import OpenRouterCost
from analyst.middleware.metering import ContextMeter, ContextSize, measure

__all__ = [
    "ClearOldToolResults",
    "CompactionMiddleware",
    "ContextMeter",
    "ContextSize",
    "OpenRouterCost",
    "analyst_middleware",
    "measure",
]

TOOL_CALL_LIMIT = 40
MODEL_CALL_LIMIT = 25


def analyst_middleware(
    settings: Any = None, *, model: Any = None, compaction: bool = True
) -> list[AgentMiddleware]:
    """Middleware for the analyst agent, outermost first.

    ``settings`` supplies the compaction thresholds (defaults when absent).
    ``model`` is the chat model (or factory of the run's model spec) used for
    the compaction digest; by default the run's own model is built from the
    context. ``compaction=False`` leaves compaction out (the worker, whose
    deep-agent harness summarises on its own and whose runs are short).
    """
    trigger = getattr(settings, "compaction_trigger_tokens", 60_000)
    keep = getattr(settings, "compaction_keep_messages", 8)
    return [
        *([CompactionMiddleware(trigger, keep, model)] if compaction else []),
        ClearOldToolResults(),
        ContextMeter(),
        OpenRouterCost(),
        ToolCallLimitMiddleware(run_limit=TOOL_CALL_LIMIT),
        ModelCallLimitMiddleware(run_limit=MODEL_CALL_LIMIT),
    ]
