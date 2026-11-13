"""Copy OpenRouter's per-call cost into the AI message usage metadata.

OpenRouter reports the actual charged cost in ``response_metadata["usage"]["cost"]``.
LangSmith reads cost from the message's ``usage_metadata`` under the
``total_cost`` key (``langsmith.run_trees.validate_extracted_usage_metadata``
allows ``input_cost``, ``output_cost`` and ``total_cost``), so the cost is
copied there after the model call.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import (
    AgentMiddleware,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, BaseMessage

COST_KEY = "total_cost"


def _usage(message: AIMessage) -> dict[str, Any]:
    usage = (message.response_metadata or {}).get("usage")
    return usage if isinstance(usage, dict) else {}


def _with_cost(message: BaseMessage) -> BaseMessage:
    """Return ``message`` with ``usage_metadata["total_cost"]`` set, if known."""
    if not isinstance(message, AIMessage):
        return message

    usage = _usage(message)
    cost = usage.get("cost")
    if cost is None:
        return message

    metadata: dict[str, Any] = dict(message.usage_metadata or {})
    if not metadata:
        # No token metadata yet: build it from the raw usage block so the
        # result is still a well-formed UsageMetadata for message additions.
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if prompt_tokens is None or completion_tokens is None:
            return message
        metadata = {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": usage.get("total_tokens")
            or prompt_tokens + completion_tokens,
        }

    metadata[COST_KEY] = cost
    return message.model_copy(update={"usage_metadata": metadata})


def _apply(result: Any) -> Any:
    """Apply the cost copy to whichever model-call result shape came back."""
    if isinstance(result, AIMessage):
        return _with_cost(result)
    if isinstance(result, ExtendedModelResponse):
        return ExtendedModelResponse(
            model_response=_apply(result.model_response), command=result.command
        )
    if isinstance(result, ModelResponse):
        return ModelResponse(
            result=[_with_cost(message) for message in result.result],
            structured_response=result.structured_response,
        )
    return result


class OpenRouterCost(AgentMiddleware):
    """Copy ``response_metadata["usage"]["cost"]`` into ``usage_metadata``."""

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], object],
    ) -> object:
        """Run the model call, then attach the reported cost."""
        return _apply(handler(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[object]],
    ) -> object:
        """Async variant of ``wrap_model_call``."""
        return _apply(await handler(request))
