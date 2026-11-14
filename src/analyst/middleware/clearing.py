"""Stub older tool results in the request sent to the model.

The saved conversation keeps every tool result. Only the copy handed to the
model is trimmed: all but the last ``keep_last`` tool messages are replaced by
a short stub naming the result handle, so the model can still ask for the full
card again with ``describe``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence

from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain_core.messages import AnyMessage, ToolMessage

DEFAULT_STUB = "[cleared] tool output"


def stub_for(message: ToolMessage) -> str:
    """Build the replacement text for one cleared tool message."""
    artifact = message.artifact if isinstance(message.artifact, dict) else {}
    many = artifact.get("handles")
    if not artifact.get("handle_id") and isinstance(many, list):
        entries = [a for a in many if isinstance(a, dict) and a.get("handle_id")]
        if entries:
            labels = ", ".join(
                f"{a['handle_id']} ({a.get('produced_by') or 'unknown'})" for a in entries
            )
            return f"[cleared] {labels}; call describe('<handle>') to see it again"
    handle_id = artifact.get("handle_id")
    if not handle_id:
        return DEFAULT_STUB
    produced_by = artifact.get("produced_by") or "unknown"
    return (
        f"[cleared] {handle_id} ({produced_by}); "
        f"call describe('{handle_id}') to see it again"
    )


def cleared_messages(
    messages: Sequence[AnyMessage], keep_last: int
) -> list[AnyMessage]:
    """Return a copy of ``messages`` with old tool results replaced by stubs."""
    tool_positions = [
        position
        for position, message in enumerate(messages)
        if isinstance(message, ToolMessage)
    ]
    if keep_last > 0:
        to_clear = set(tool_positions[:-keep_last])
    else:
        to_clear = set(tool_positions)

    cleared: list[AnyMessage] = []
    for position, message in enumerate(messages):
        if position in to_clear and isinstance(message, ToolMessage):
            cleared.append(message.model_copy(update={"content": stub_for(message)}))
        else:
            cleared.append(message)
    return cleared


class ClearOldToolResults(AgentMiddleware):
    """Replace all but the last ``keep_last`` tool results with handle stubs."""

    def __init__(self, keep_last: int = 6) -> None:
        super().__init__()
        self.keep_last = keep_last

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], object],
    ) -> object:
        """Send the model a trimmed copy of the conversation."""
        return handler(
            request.override(messages=cleared_messages(request.messages, self.keep_last))
        )

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[object]],
    ) -> object:
        """Async variant of ``wrap_model_call``."""
        return await handler(
            request.override(messages=cleared_messages(request.messages, self.keep_last))
        )
