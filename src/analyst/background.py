"""Background work: the worker subagent spec, the analyst's guard, the worker's context.

Slow work (long forecasts, investigations) runs as a deepagents async subagent
on the same Agent Server. The run context reaches the worker two ways:

1. deepagents forwards the analyst's run context to the worker run (primary);
2. :class:`BackgroundGuard` prepends a ``[context ...]`` header to every task
   description, and :class:`WorkerContext` reads it back (defence in depth).

The worker trusts the narrower of the two: the same dataset or nothing, and
the earliest cutoff either names. With neither, every worker tool fails
closed, so a worker can never query past the cutoff of the run that started it.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import replace
from datetime import date
from functools import lru_cache
from typing import Annotated, Any, NotRequired

from deepagents import AsyncSubAgent, AsyncSubAgentMiddleware
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents.middleware.types import (
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command

from analyst.context import AnalystContext
from analyst.errors import AnalystError, InvalidRequestError

__all__ = [
    "ASYNC_TASK_TOOL_NAMES",
    "CONTEXT_HEADER_RE",
    "NOTIFICATION_SOURCE",
    "START_TOOL",
    "WORKER",
    "BackgroundGuard",
    "WorkerContext",
    "format_context_header",
    "is_background_notification",
    "parse_context_header",
    "resolve_worker_context",
]

WORKER = AsyncSubAgent(
    name="worker",
    description=(
        "Runs slow or multi-step data work (long forecasts, investigations with "
        "3+ queries). Give it the full task; it returns a report ending with a "
        "handles: line."
    ),
    graph_id="worker",
)

#: the tools AsyncSubAgentMiddleware adds, read from the middleware itself.
ASYNC_TASK_TOOL_NAMES: frozenset[str] = frozenset(
    t.name for t in AsyncSubAgentMiddleware(async_subagents=[WORKER]).tools
)
START_TOOL = "start_async_task"

#: task statuses that still occupy a background slot (deepagents' unfinished set).
_OCCUPYING = frozenset({"running", "pending", "waiting"})

# --- notifications -----------------------------------------------------------

#: ``lc_source`` of a deepagents task notification (``source`` on older shapes).
NOTIFICATION_SOURCE = "async_subagent"
_SOURCE_KEYS = ("lc_source", "source")
_DEPRECATED_NOTIFICATION_KEY = "deepagents_notification"


def _get(message: object, name: str) -> Any:
    if isinstance(message, Mapping):
        return message.get(name)
    return getattr(message, name, None)


def is_background_notification(message: object) -> bool:
    """Whether ``message`` is a background task's notification to the analyst.

    Real notifications carry ``additional_kwargs["lc_source"] ==
    "async_subagent"`` (via ``add_notification_metadata``). Anyone who can send
    input can set these keys, so this only locates worker reports: it never
    authorizes anything (cited handles must still exist in the result store).
    """
    for location in ("additional_kwargs", "response_metadata", "metadata"):
        meta = _get(message, location)
        if not isinstance(meta, Mapping):
            continue
        if any(meta.get(key) == NOTIFICATION_SOURCE for key in _SOURCE_KEYS):
            return True
        if location == "additional_kwargs" and isinstance(
            meta.get(_DEPRECATED_NOTIFICATION_KEY), Mapping
        ):
            return True
    return False


# --- context header ----------------------------------------------------------

_VALUE = r"[^\s\]]+"
CONTEXT_HEADER_RE = re.compile(
    rf"\[context dataset=(?P<dataset>{_VALUE}) "
    r"cutoff=(?P<cutoff>\d{4}-\d{2}-\d{2}|none) "
    rf"model=(?P<model>{_VALUE}) "
    rf"question_id=(?P<question_id>{_VALUE})\]"
)
#: anything that starts like a header, well-formed or not.
_HEADER_LIKE = re.compile(r"\[context\b[^\]\n]*\]?")
_HEADER_PREFIX = "[context"


def _header_value(value: str | None, default: str) -> str:
    if value is None or value == "":
        return default
    return re.sub(r"[\s\]\[]", "_", value)


def format_context_header(ctx: AnalystContext) -> str:
    """``[context dataset=<d> cutoff=<YYYY-MM-DD|none> model=<spec|default> question_id=<id|none>]``."""
    if not ctx.dataset or re.search(r"[\s\]\[]", ctx.dataset):
        raise InvalidRequestError(
            f"dataset name {ctx.dataset!r} cannot go in a context header",
            hint="use a dataset name without spaces or brackets",
        )
    cutoff = ctx.cutoff.isoformat() if ctx.cutoff is not None else "none"
    return (
        f"[context dataset={ctx.dataset} cutoff={cutoff} "
        f"model={_header_value(ctx.model, 'default')} "
        f"question_id={_header_value(ctx.question_id, 'none')}]"
    )


def parse_context_header(text: str) -> AnalystContext:
    """The context in the header that leads ``text``; the worker runs inline.

    Raises ``InvalidRequestError`` when ``text`` does not start with a
    well-formed header (a header anywhere else is ignored).
    """
    match = CONTEXT_HEADER_RE.match(text.lstrip())
    if match is None:
        raise InvalidRequestError(
            "no context header",
            hint="start the task with format_context_header(context)",
        )
    try:
        cutoff = (
            None if match["cutoff"] == "none" else date.fromisoformat(match["cutoff"])
        )
    except ValueError as exc:
        raise InvalidRequestError(
            f"invalid cutoff {match['cutoff']!r} in context header",
            hint="the cutoff is an ISO date, YYYY-MM-DD",
        ) from exc
    return AnalystContext(
        dataset=match["dataset"],
        cutoff=cutoff,
        model=None if match["model"] == "default" else match["model"],
        background=False,
        question_id=None if match["question_id"] == "none" else match["question_id"],
    )


def _strip_headers(text: str) -> str:
    return _HEADER_LIKE.sub("", text).strip()


def _error(request: ToolCallRequest, text: str) -> ToolMessage:
    return ToolMessage(
        content=text,
        tool_call_id=request.tool_call["id"],
        name=request.tool_call["name"],
        status="error",
    )


def _tool_names(tools: Iterable[Any]) -> list[str]:
    return [t["name"] if isinstance(t, Mapping) else t.name for t in tools]


# --- analyst side --------------------------------------------------------------


class BackgroundGuard(AgentMiddleware):
    """Gates the analyst's async-task tools on the run context and a task cap.

    - ``context.background`` False: the async-task tools are hidden from the
      model, and a start call is refused anyway.
    - A start call is refused once ``max_tasks`` tasks occupy a slot, counting
      earlier start calls in the same model turn (parallel tool calls).
    - Otherwise the run's context header is prepended to the task description
      (any header the model wrote is removed first).
    """

    def __init__(self, max_tasks: int) -> None:
        super().__init__()
        self.max_tasks = max_tasks

    @staticmethod
    def _background(context: object) -> bool:
        return isinstance(context, AnalystContext) and context.background

    def _filter(self, request: ModelRequest) -> ModelRequest:
        if self._background(request.runtime.context if request.runtime else None):
            return request
        tools = [
            t
            for t in request.tools
            if (t["name"] if isinstance(t, Mapping) else t.name)
            not in ASYNC_TASK_TOOL_NAMES
        ]
        return request.override(tools=tools)

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        return handler(self._filter(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(self._filter(request))

    def _occupied(self, request: ToolCallRequest) -> int:
        state = request.state if isinstance(request.state, Mapping) else {}
        tasks = state.get("async_tasks") or {}
        running = sum(
            1
            for task in tasks.values()
            if isinstance(task, Mapping) and task.get("status") in _OCCUPYING
        )
        # Parallel start calls in this turn see the same state: count the ones
        # ahead of this call in the model message that issued it.
        call_id = request.tool_call.get("id")
        for message in reversed(state.get("messages") or []):
            calls = getattr(message, "tool_calls", None) or []
            ids = [c.get("id") for c in calls if c.get("name") == START_TOOL]
            if isinstance(message, AIMessage) and call_id in ids:
                running += ids.index(call_id)
                break
        return running

    def _prepare(self, request: ToolCallRequest) -> ToolCallRequest | ToolMessage:
        context = request.runtime.context if request.runtime else None
        if not self._background(context):
            return _error(
                request,
                "ERROR: background tasks are off for this run\n"
                "HINT: do the work inline in this turn",
            )
        occupied = self._occupied(request)
        if occupied >= self.max_tasks:
            return _error(
                request,
                f"ERROR: {occupied} background tasks are already running "
                f"(limit {self.max_tasks})\n"
                "HINT: wait for one to finish, or check them with "
                "check_async_task / list_async_tasks",
            )
        try:
            header = format_context_header(context)
        except AnalystError as exc:
            return _error(request, f"ERROR: {exc.message}\nHINT: {exc.hint}")
        args = dict(request.tool_call.get("args") or {})
        description = _strip_headers(str(args.get("description") or ""))
        args["description"] = f"{header}\n{description}"
        return request.override(tool_call={**request.tool_call, "args": args})

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        if request.tool_call["name"] != START_TOOL:
            return handler(request)
        prepared = self._prepare(request)
        return prepared if isinstance(prepared, ToolMessage) else handler(prepared)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        if request.tool_call["name"] != START_TOOL:
            return await handler(request)
        prepared = self._prepare(request)
        if isinstance(prepared, ToolMessage):
            return prepared
        return await handler(prepared)


# --- worker side ---------------------------------------------------------------

NO_CONTEXT = "ERROR: no dataset context"
_NO_CONTEXT_HINT = (
    "HINT: start the worker from the analyst (start_async_task), which sends "
    "the run context, or pass a run context with a dataset"
)


class _ContextError(Exception):
    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.text = text


class WorkerState(AgentState):
    #: the header's context as JSON data; ``{"invalid": ...}`` for a bad header.
    analyst_context: NotRequired[Annotated[dict[str, Any] | None, PrivateStateAttr]]


def _first_human_text(messages: Iterable[Any]) -> str | None:
    for message in messages:
        if isinstance(message, HumanMessage) or _get(message, "type") == "human":
            content = _get(message, "content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(
                    part if isinstance(part, str) else str(part.get("text", ""))
                    for part in content
                    if isinstance(part, str | dict)
                )
            return ""
    return None


def _header_state(messages: Iterable[Any]) -> dict[str, Any] | None:
    text = _first_human_text(messages)
    if text is None or not text.lstrip().startswith(_HEADER_PREFIX):
        return None
    try:
        ctx = parse_context_header(text)
    except AnalystError as exc:
        return {"invalid": exc.message}
    return {
        "dataset": ctx.dataset,
        "cutoff": ctx.cutoff.isoformat() if ctx.cutoff else None,
        "model": ctx.model,
        "question_id": ctx.question_id,
    }


def _runtime_context(context: object) -> AnalystContext | None:
    if isinstance(context, Mapping):
        try:
            context = AnalystContext(**context)
        except (AnalystError, TypeError):
            return None
    if isinstance(context, AnalystContext) and context.dataset:
        return context
    return None


def resolve_worker_context(context: object, stored: object) -> AnalystContext:
    """The context worker tools run with: the narrower of run context and header.

    Raises ``_ContextError`` (fail closed) when neither names a dataset, the
    header is malformed, or the two disagree on the dataset.
    """
    runtime_ctx = _runtime_context(context)
    header_ctx: AnalystContext | None = None
    if isinstance(stored, Mapping):
        if "invalid" in stored:
            raise _ContextError(
                f"ERROR: invalid context header ({stored['invalid']})\n"
                + _NO_CONTEXT_HINT
            )
        header_ctx = AnalystContext(
            dataset=stored["dataset"],
            cutoff=stored.get("cutoff"),
            model=stored.get("model"),
            background=False,
            question_id=stored.get("question_id"),
        )
    if runtime_ctx is None and header_ctx is None:
        raise _ContextError(f"{NO_CONTEXT}\n{_NO_CONTEXT_HINT}")
    if runtime_ctx is None:
        return header_ctx  # type: ignore[return-value]
    if header_ctx is None:
        return replace(runtime_ctx, background=False)
    if runtime_ctx.dataset != header_ctx.dataset:
        raise _ContextError(
            "ERROR: conflicting dataset context "
            f"({runtime_ctx.dataset!r} vs {header_ctx.dataset!r})\n" + _NO_CONTEXT_HINT
        )
    cutoffs = [c for c in (runtime_ctx.cutoff, header_ctx.cutoff) if c is not None]
    return replace(
        runtime_ctx,
        cutoff=min(cutoffs) if cutoffs else None,
        model=runtime_ctx.model or header_ctx.model,
        question_id=runtime_ctx.question_id or header_ctx.question_id,
        background=False,
    )


@lru_cache(maxsize=8)
def _model_for(spec: str) -> BaseChatModel:
    from analyst.providers import get_model

    return get_model(spec)


class WorkerContext(AgentMiddleware):
    """The worker's context and tool allowlist.

    ``before_agent`` reads the header from the thread's first human message
    into state ``analyst_context`` (overwriting anything the input set). Every
    tool call then runs with :func:`resolve_worker_context` as its context;
    without one it returns an error and the tool never runs. Only the data
    tools may run (no ``task``, file or shell tools): the deep agent's built-in
    subagents would otherwise run data tools outside this middleware. A model
    spec in the context (the analyst's) replaces the worker's default model.
    """

    state_schema = WorkerState

    def __init__(self, allowed_tools: Iterable[str] | None = None) -> None:
        super().__init__()
        if allowed_tools is None:
            from analyst.tools import DATA_TOOLS

            allowed_tools = (t.name for t in DATA_TOOLS)
        self.allowed = frozenset(allowed_tools)

    def before_agent(self, state: WorkerState, runtime: Runtime) -> dict[str, Any]:
        return {"analyst_context": _header_state(state.get("messages") or [])}

    async def abefore_agent(
        self, state: WorkerState, runtime: Runtime
    ) -> dict[str, Any]:
        return self.before_agent(state, runtime)

    def _filter(self, request: ModelRequest) -> ModelRequest:
        tools = [
            t
            for t, name in zip(request.tools, _tool_names(request.tools), strict=True)
            if name in self.allowed
        ]
        return request.override(tools=tools)

    @staticmethod
    def _model_spec(request: ModelRequest) -> str | None:
        """The run's model spec, so the worker runs on the analyst's model."""
        state = request.state if isinstance(request.state, Mapping) else {}
        try:
            ctx = resolve_worker_context(
                request.runtime.context if request.runtime else None,
                state.get("analyst_context"),
            )
        except (_ContextError, AnalystError):
            return None
        return ctx.model

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        request = self._filter(request)
        if spec := self._model_spec(request):
            request = request.override(model=_model_for(spec))
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        request = self._filter(request)
        if spec := self._model_spec(request):
            # Building a client touches the disk: keep it off the event loop.
            model = await asyncio.to_thread(_model_for, spec)
            request = request.override(model=model)
        return await handler(request)

    def _prepare(self, request: ToolCallRequest) -> ToolCallRequest | ToolMessage:
        name = request.tool_call["name"]
        if name not in self.allowed:
            return _error(
                request,
                f"ERROR: tool {name!r} is not available to the worker\n"
                "HINT: use the data tools: " + ", ".join(sorted(self.allowed)),
            )
        state = request.state if isinstance(request.state, Mapping) else {}
        runtime = request.runtime
        if runtime is None:
            return _error(request, f"{NO_CONTEXT}\n{_NO_CONTEXT_HINT}")
        try:
            ctx = resolve_worker_context(runtime.context, state.get("analyst_context"))
        except _ContextError as exc:
            return _error(request, exc.text)
        except AnalystError as exc:
            return _error(request, f"ERROR: {exc.message}\nHINT: {exc.hint}")
        return request.override(runtime=replace(runtime, context=ctx))

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        prepared = self._prepare(request)
        return prepared if isinstance(prepared, ToolMessage) else handler(prepared)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        prepared = self._prepare(request)
        if isinstance(prepared, ToolMessage):
            return prepared
        return await handler(prepared)
