"""Measure the context handed to the model and record it on the LangSmith run.

The size numbers land in the current run's metadata (``context_total_bytes``,
``context_tool_bytes``, ``context_messages``) together with the question id, so
context growth per question is visible in LangSmith. When tracing is off there
is no run tree and the middleware does nothing.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain_core.messages import BaseMessage, ToolMessage
from langsmith.run_helpers import get_current_run_tree


@dataclass(frozen=True)
class ContextSize:
    """Byte and message counts for one model request."""

    total_bytes: int
    tool_bytes: int
    messages: int


def _content_bytes(content: Any) -> int:
    if isinstance(content, str):
        return len(content.encode("utf-8"))
    if isinstance(content, list):
        return sum(
            _content_bytes(part.get("text", "") if isinstance(part, dict) else part)
            for part in content
        )
    return len(str(content).encode("utf-8"))


def measure(messages: Sequence[BaseMessage]) -> ContextSize:
    """Count UTF-8 bytes of message content, and of tool results separately."""
    total_bytes = 0
    tool_bytes = 0
    for message in messages:
        size = _content_bytes(message.content)
        total_bytes += size
        if isinstance(message, ToolMessage):
            tool_bytes += size
    return ContextSize(
        total_bytes=total_bytes, tool_bytes=tool_bytes, messages=len(messages)
    )


def _question_id(request: ModelRequest) -> str | None:
    context = getattr(request.runtime, "context", None)
    return getattr(context, "question_id", None)


class ContextMeter(AgentMiddleware):
    """Write context size and question id to the current LangSmith run."""

    def _record(self, request: ModelRequest) -> None:
        run_tree = get_current_run_tree()
        if run_tree is None:
            return
        size = measure(request.messages)
        run_tree.add_metadata(
            {
                "context_total_bytes": size.total_bytes,
                "context_tool_bytes": size.tool_bytes,
                "context_messages": size.messages,
                "question_id": _question_id(request),
            }
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], object],
    ) -> object:
        """Record the request size, then run the model call."""
        self._record(request)
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[object]],
    ) -> object:
        """Async variant of ``wrap_model_call``."""
        self._record(request)
        return await handler(request)
