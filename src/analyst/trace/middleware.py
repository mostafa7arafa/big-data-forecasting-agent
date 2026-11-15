"""Stamp every model and tool call with wall-clock timing; stream step events.

:class:`TraceMiddleware` sits outermost in the agent's middleware, so its
stamp covers the whole call (inner middleware included) and the message it
stamps already carries the provider cost. The stamp lands in the result
message's ``response_metadata["analyst_trace"]`` as ISO strings
``{"step_id", "started_at", "ended_at", "first_token_at"}``; tool results are
stamped on the ``ToolMessage`` (inside a ``Command`` update too).

Live events go through LangGraph's stream writer as ``{"type": "trace",
"event": "step_started" | "step_finished", "step": <StepTrace dict>}``; they
are visible to a ``stream_mode="custom"`` consumer and silently dropped
outside a graph.

Time to first token: the agent loop calls the model with no per-call config,
so a callback cannot be attached to the request itself (``with_config`` is
lost when the agent calls ``bind_tools``). Instead a context-scoped handler is
registered with ``langchain_core.tracers.context.register_configure_hook`` —
LangChain's supported mechanism for context-local callbacks, the one its own
tracers use. The middleware sets it around ``handler(request)`` only, so it
sees exactly this model call's tokens. It never forces streaming: when the
model does not stream, ``first_token_at`` stays ``None``.
"""

from __future__ import annotations

import asyncio
import logging
import operator
import uuid
from collections.abc import Awaitable, Callable, Mapping
from contextvars import ContextVar
from dataclasses import replace
from datetime import UTC, datetime
from typing import Annotated, Any, NotRequired

from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)
from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tracers.context import register_configure_hook
from langgraph.config import get_config, get_stream_writer
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command

from analyst.cancel import cancel_thread, current_thread_id, try_cancel_thread
from analyst.providers.catalog import ModelInfo, cached_model_info, model_spec_of
from analyst.trace.build import STAMP_KEY, build_trace, model_step, tool_step
from analyst.trace.model import StepTrace

__all__ = [
    "RunTraceRecorder",
    "TraceMiddleware",
    "atimed_model_call",
    "current_run_id",
    "emit",
    "run_id_of",
]


logger = logging.getLogger(__name__)

#: the tool whose cancellation also closes the thread's sandbox.
SANDBOX_TOOL = "run_python"


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


class _FirstToken(AsyncCallbackHandler):
    """Records when the first streamed token of a model call arrived."""

    def __init__(self) -> None:
        self.at: datetime | None = None

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        if self.at is None:
            self.at = _now()


class _SyncFirstToken(BaseCallbackHandler):
    """The same probe for the sync agent path."""

    def __init__(self) -> None:
        self.at: datetime | None = None

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        if self.at is None:
            self.at = _now()


_first_token: ContextVar[BaseCallbackHandler | None] = ContextVar(
    "analyst_first_token", default=None
)
# Not inheritable: the probe joins the model run's own callback manager only.
register_configure_hook(_first_token, inheritable=False)


def emit(event: str, step: StepTrace | None) -> None:
    """Write a trace event to the graph's custom stream; a no-op outside a graph."""
    try:
        writer = get_stream_writer()
    except RuntimeError:  # not inside a runnable context
        return
    writer(
        {
            "type": "trace",
            "event": event,
            "step": step.model_dump(mode="json") if step is not None else None,
        }
    )


def run_id_of(runtime: Any) -> str | None:
    """The run id of the current run: the node runtime's first, then the config.

    The Agent Server passes the run id as top-level ``config["run_id"]``, which
    LangGraph drops from node configs but keeps in ``runtime.execution_info``.
    """
    info = getattr(runtime, "execution_info", None)
    value = getattr(info, "run_id", None)
    if value is not None:
        return str(value)
    return current_run_id()


def current_run_id() -> str | None:
    """The LangGraph run id of the current run, when there is one."""
    try:
        config = get_config()
    except RuntimeError:
        return None
    for source in (config.get("metadata"), config.get("configurable"), config):
        if isinstance(source, Mapping) and source.get("run_id") is not None:
            return str(source["run_id"])
    return None


def _live_info(model: Any) -> ModelInfo | None:
    """The catalogue entry for a live step's model, without blocking.

    Prices a step whose provider reports tokens but no cost the way
    ``build_trace`` prices the saved one. ``get_model`` loaded the catalogue
    off the loop, so this is an in-memory lookup.
    """
    spec = model_spec_of(model)
    return cached_model_info(spec) if spec is not None else None


def _model_name(request: ModelRequest) -> str:
    for attr in ("model_name", "model"):
        value = getattr(request.model, attr, None)
        if isinstance(value, str) and value:
            return value
    return "model"


def _stamped(message: BaseMessage, marks: dict[str, Any]) -> BaseMessage:
    meta = {**(message.response_metadata or {}), STAMP_KEY: marks}
    return message.model_copy(update={"response_metadata": meta})


def _stamp_model_result(result: Any, marks: dict[str, Any]) -> tuple[Any, AIMessage | None]:
    """``result`` with its AI message stamped, and that message."""
    if isinstance(result, AIMessage):
        stamped = _stamped(result, marks)
        return stamped, stamped
    if isinstance(result, ExtendedModelResponse):
        inner, message = _stamp_model_result(result.model_response, marks)
        return ExtendedModelResponse(model_response=inner, command=result.command), message
    if isinstance(result, ModelResponse):
        found: AIMessage | None = None
        messages = []
        for m in result.result:
            if isinstance(m, AIMessage) and found is None:
                m = _stamped(m, marks)
                found = m
            messages.append(m)
        return (
            ModelResponse(result=messages, structured_response=result.structured_response),
            found,
        )
    return result, None


def _stamp_tool_result(result: Any, marks: dict[str, Any], call_id: str) -> tuple[Any, ToolMessage | None]:
    if isinstance(result, ToolMessage):
        stamped = _stamped(result, marks)
        return stamped, stamped
    if isinstance(result, Command) and isinstance(result.update, Mapping):
        messages = result.update.get("messages")
        if isinstance(messages, list):
            found: ToolMessage | None = None
            out = []
            for m in messages:
                if isinstance(m, ToolMessage) and m.tool_call_id == call_id and found is None:
                    m = _stamped(m, marks)
                    found = m
                out.append(m)
            return replace(result, update={**result.update, "messages": out}), found
    return result, None


async def atimed_model_call(
    name: str,
    call: Callable[[], Awaitable[Any]],
    raw_of: Callable[[Any], Any] = lambda result: result,
    *,
    model: Any = None,
) -> tuple[Any, AIMessage | None]:
    """Run a model call outside the agent loop as a traced step.

    For graph nodes that call a model directly (plan, compaction): the same
    live events, first-token probe and stamp the middleware gives agent model
    calls. ``raw_of(result)`` picks the call's raw AI message; the stamped
    copy is returned for the node to hand to the run trace (``None`` when the
    call produced no AI message). ``model`` is the chat model the call uses;
    it prices a step whose provider reports no cost.
    """
    started = _now()
    step = StepTrace(
        id=f"model-{uuid.uuid4().hex[:12]}",
        kind="model",
        name=name,
        started_at=started,
        ended_at=None,
        status="ok",
    )
    emit("step_started", step)
    probe = _FirstToken()
    token = _first_token.set(probe)
    try:
        result = await call()
    except BaseException as exc:
        status = "cancelled" if isinstance(exc, asyncio.CancelledError) else "error"
        emit("step_finished", step.model_copy(update={"ended_at": _now(), "status": status}))
        raise
    finally:
        _first_token.reset(token)
    ended = _now()
    raw = raw_of(result)
    if not isinstance(raw, AIMessage):
        emit("step_finished", step.model_copy(update={"ended_at": ended}))
        return result, None
    marks = {
        "step_id": step.id,
        "started_at": _iso(started),
        "ended_at": _iso(ended),
        "first_token_at": _iso(probe.at),
        "name": name,
    }
    stamped = _stamped(raw, marks)
    emit("step_finished", model_step(stamped, _live_info(model)))
    return result, stamped


async def _stop_thread_work(tool_name: str) -> None:
    """A cancelled tool call: stop its thread's in-flight work.

    The awaited sync tool keeps running in its executor thread after the
    task is cancelled; interrupt its queries (``analyst.cancel``) and, for
    the sandbox tool, kill the thread's sandbox. Closing a sandbox runs the
    docker CLI, so it goes off the loop, shielded so a second cancel cannot
    abandon it half done. Cleanup errors are logged, never raised.
    """
    thread_id = current_thread_id()
    if thread_id is None:
        return
    try:
        # Never wait for the registry lock on the loop; on contention retry
        # off it, shielded, so the cancel is never dropped.
        if try_cancel_thread(thread_id) is None:
            await asyncio.shield(asyncio.to_thread(cancel_thread, thread_id))
    except asyncio.CancelledError:
        pass  # the shielded retry still runs; the caller re-raises either way
    except Exception:
        logger.warning("cancelling the work of a cancelled run failed", exc_info=True)
    if tool_name != SANDBOX_TOOL:
        return
    from analyst.sandbox import close_sandbox

    try:
        await asyncio.shield(asyncio.to_thread(close_sandbox, thread_id))
    except asyncio.CancelledError:
        pass  # the caller re-raises the cancellation either way
    except Exception:
        logger.warning("closing the sandbox of a cancelled run failed", exc_info=True)


class TraceMiddleware(AgentMiddleware):
    """Wall-clock stamps on model and tool calls, plus live trace events."""

    # --- model calls

    def _start_model(self, request: ModelRequest) -> tuple[StepTrace, datetime]:
        started = _now()
        step = StepTrace(
            id=f"model-{uuid.uuid4().hex[:12]}",
            kind="model",
            name=_model_name(request),
            started_at=started,
            ended_at=None,
            status="ok",
        )
        emit("step_started", step)
        return step, started

    def _finish_model(
        self,
        step: StepTrace,
        started: datetime,
        first: datetime | None,
        result: Any,
        model: Any = None,
    ) -> Any:
        ended = _now()
        marks = {
            "step_id": step.id,
            "started_at": _iso(started),
            "ended_at": _iso(ended),
            "first_token_at": _iso(first),
        }
        result, message = _stamp_model_result(result, marks)
        if message is not None:
            done = model_step(message, _live_info(model)).model_copy(
                update={"name": step.name}
            )
        else:
            done = step.model_copy(update={"ended_at": ended})
        emit("step_finished", done)
        return result

    def _failed(self, step: StepTrace, exc: BaseException) -> None:
        status = "cancelled" if isinstance(exc, asyncio.CancelledError) else "error"
        emit("step_finished", step.model_copy(update={"ended_at": _now(), "status": status}))

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Any]
    ) -> Any:
        step, started = self._start_model(request)
        probe = _SyncFirstToken()
        token = _first_token.set(probe)
        try:
            result = handler(request)
        except BaseException as exc:
            self._failed(step, exc)
            raise
        finally:
            _first_token.reset(token)
        return self._finish_model(step, started, probe.at, result, request.model)

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[Any]]
    ) -> Any:
        step, started = self._start_model(request)
        probe = _FirstToken()
        token = _first_token.set(probe)
        try:
            result = await handler(request)
        except BaseException as exc:
            self._failed(step, exc)
            raise
        finally:
            _first_token.reset(token)
        return self._finish_model(step, started, probe.at, result, request.model)

    # --- tool calls

    def _start_tool(self, request: ToolCallRequest) -> tuple[StepTrace, datetime]:
        started = _now()
        call = request.tool_call
        step = StepTrace(
            id=str(call.get("id") or f"tool-{uuid.uuid4().hex[:12]}"),
            kind="tool",
            name=str(call.get("name") or "tool"),
            started_at=started,
            ended_at=None,
            status="ok",
        )
        emit("step_started", step)
        return step, started

    def _finish_tool(self, step: StepTrace, started: datetime, result: Any) -> Any:
        ended = _now()
        marks = {
            "step_id": step.id,
            "started_at": _iso(started),
            "ended_at": _iso(ended),
            "first_token_at": None,
        }
        result, message = _stamp_tool_result(result, marks, step.id)
        if message is not None:
            done = tool_step(message).model_copy(update={"id": step.id, "name": step.name})
        else:
            done = step.model_copy(update={"ended_at": ended})
        emit("step_finished", done)
        return result

    def wrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Any]
    ) -> Any:
        step, started = self._start_tool(request)
        try:
            result = handler(request)
        except BaseException as exc:
            self._failed(step, exc)
            raise
        return self._finish_tool(step, started, result)

    async def awrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Awaitable[Any]]
    ) -> Any:
        step, started = self._start_tool(request)
        try:
            result = await handler(request)
        except asyncio.CancelledError as exc:
            self._failed(step, exc)
            await _stop_thread_work(step.name)
            raise  # never swallow the cancellation
        except BaseException as exc:
            self._failed(step, exc)
            raise
        return self._finish_tool(step, started, result)


class TraceState(AgentState):
    #: one RunTrace dict per finished run of this agent.
    trace: NotRequired[Annotated[list[dict], operator.add]]


def _run_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """The messages after the latest human message (this run's task)."""
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return messages[i + 1 :]
    return messages


def _context_model(context: Any) -> str | None:
    value = context.get("model") if isinstance(context, Mapping) else getattr(context, "model", None)
    return value if isinstance(value, str) and value else None


def _run_model(context: Any, state: Any) -> str | None:
    """The model the run actually used.

    The worker resolves its run context from the forwarded context and the
    task's stored header (see ``WorkerContext``); the trace does the same.
    """
    from analyst.background import _ContextError, resolve_worker_context
    from analyst.errors import AnalystError

    stored = state.get("analyst_context") if isinstance(state, Mapping) else None
    try:
        return resolve_worker_context(context, stored).model
    except (_ContextError, AnalystError):  # no resolvable context: fall back
        return _context_model(context)


def _trace_inputs(context: Any, state: Any = None) -> tuple[str, int]:
    """Model spec and compaction trigger; reads settings and warms the catalogue."""
    from analyst.config import get_settings
    from analyst.providers.catalog import model_info

    settings = get_settings()
    spec = _run_model(context, state) or settings.model
    model_info(spec, settings=settings)
    return spec, settings.compaction_trigger_tokens


class RunTraceRecorder(TraceMiddleware):
    """:class:`TraceMiddleware` that also appends the run's trace to ``trace``.

    For agents with no outer answer node (the background worker): when the
    agent finishes, the run's messages are folded into one record.
    """

    state_schema = TraceState

    def _record(
        self, state: Any, runtime: Runtime, spec: str, trigger: int
    ) -> dict[str, Any]:
        trace = build_trace(
            _run_messages(list(state.get("messages") or [])),
            model=spec,
            run_id=run_id_of(runtime),
            verification=None,
            events=[],
            compaction_trigger=trigger,
        )
        return {"trace": [trace.model_dump(mode="json")]}

    def after_agent(self, state: Any, runtime: Runtime) -> dict[str, Any] | None:
        spec, trigger = _trace_inputs(getattr(runtime, "context", None), state)
        return self._record(state, runtime, spec, trigger)

    async def aafter_agent(self, state: Any, runtime: Runtime) -> dict[str, Any] | None:
        # Settings and the catalogue may touch the disk: keep them off the loop.
        spec, trigger = await asyncio.to_thread(
            _trace_inputs, getattr(runtime, "context", None), state
        )
        return self._record(state, runtime, spec, trigger)
