"""Fold a run's messages into a :class:`RunTrace`.

Pure in-memory work: timings come from the ``analyst_trace`` stamp the
:class:`~analyst.trace.middleware.TraceMiddleware` puts in each message's
``response_metadata``, tokens from ``usage_metadata``, prices from the model
catalogue (cached; warm it off the event loop before calling from async code).

Cost per model step: the provider-reported ``usage_metadata["total_cost"]``
when present, else the catalogue price table, else ``None``. A run total is
``None`` as soon as one model step's cost is unknown.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from analyst.middleware.compaction import is_compaction_summary
from analyst.providers.catalog import ModelInfo, model_info
from analyst.state import Verification
from analyst.trace.model import RunTrace, StepTrace

__all__ = ["STAMP_KEY", "build_trace", "model_step", "tool_step"]

#: ``response_metadata`` key holding a step's wall-clock stamp.
STAMP_KEY = "analyst_trace"
#: the async-task tool whose result starts background work.
START_TOOL = "start_async_task"
_TASK_ID = re.compile(r"task_id:\s*(\S+)")
_PER_TOKEN = 1_000_000


def stamp(message: BaseMessage) -> Mapping[str, Any]:
    meta = getattr(message, "response_metadata", None) or {}
    value = meta.get(STAMP_KEY)
    return value if isinstance(value, Mapping) else {}


def _time(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _detail(usage: Mapping[str, Any], group: str, key: str) -> int | None:
    details = usage.get(group)
    return _int(details.get(key)) if isinstance(details, Mapping) else None


def _reasoning(message: AIMessage) -> str | None:
    try:
        blocks = message.content_blocks
    except Exception:  # noqa: BLE001  (an odd provider payload must not break the trace)
        return None
    texts = [
        b["reasoning"]
        for b in blocks
        if isinstance(b, Mapping)
        and b.get("type") == "reasoning"
        and isinstance(b.get("reasoning"), str)
        and b["reasoning"].strip()
    ]
    return "\n\n".join(texts) if texts else None


def _price(
    tokens_in: int | None,
    tokens_out: int | None,
    tokens_cached: int | None,
    info: ModelInfo | None,
) -> float | None:
    """Cost from the price table, or ``None`` when any needed number is unknown.

    ``input_tokens`` includes cache reads (LangChain's usage convention), so
    the cached part is priced at the cached-input rate and the rest at input.
    """
    if info is None or info.prices is None or tokens_in is None or tokens_out is None:
        return None
    prices = info.prices
    if prices.input is None or prices.output is None:
        return None
    cached = tokens_cached or 0
    if cached and prices.cached_input is None:
        return None
    cached_cost = cached * prices.cached_input if cached else 0.0
    return (
        (tokens_in - cached) * prices.input + cached_cost + tokens_out * prices.output
    ) / _PER_TOKEN


def model_step(
    message: AIMessage, info: ModelInfo | None = None, *, fallback_id: str = "model"
) -> StepTrace:
    """The trace step for one model call's AI message."""
    marks = stamp(message)
    usage = message.usage_metadata or {}
    tokens_in = _int(usage.get("input_tokens"))
    tokens_out = _int(usage.get("output_tokens"))
    tokens_cached = _detail(usage, "input_token_details", "cache_read")
    reported = usage.get("total_cost")
    if isinstance(reported, int | float) and not isinstance(reported, bool):
        cost: float | None = float(reported)
    else:
        cost = _price(tokens_in, tokens_out, tokens_cached, info)
    name = marks.get("name") or (message.response_metadata or {}).get("model_name")
    return StepTrace(
        id=str(marks.get("step_id") or message.id or fallback_id),
        kind="model",
        name=name if isinstance(name, str) and name else "model",
        started_at=_time(marks.get("started_at")),
        ended_at=_time(marks.get("ended_at")),
        status="ok",
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        tokens_cached=tokens_cached,
        tokens_reasoning=_detail(usage, "output_token_details", "reasoning"),
        cost_usd=cost,
        reasoning=_reasoning(message),
    )


def tool_step(message: ToolMessage) -> StepTrace:
    """The trace step for one tool call's result message."""
    marks = stamp(message)
    return StepTrace(
        id=message.tool_call_id,
        kind="tool",
        name=message.name or "tool",
        started_at=_time(marks.get("started_at")),
        ended_at=_time(marks.get("ended_at")),
        status="error" if message.status == "error" else "ok",
    )


def _first_token(message: BaseMessage) -> datetime | None:
    return _time(stamp(message).get("first_token_at"))


def _total(values: list[Any]) -> Any:
    """Sum of every value, or ``None`` when there are none or one is unknown."""
    if not values or any(v is None for v in values):
        return None
    return sum(values)


def _task_id(message: ToolMessage) -> str | None:
    if isinstance(message.artifact, Mapping) and isinstance(
        message.artifact.get("task_id"), str
    ):
        return message.artifact["task_id"]
    content = message.content if isinstance(message.content, str) else str(message.content)
    found = _TASK_ID.search(content)
    return found.group(1) if found else None


def build_trace(
    turn: Sequence[BaseMessage],
    *,
    model: str,
    run_id: str | None,
    verification: Verification | None,
    events: list[dict],
    compaction_trigger: int,
) -> RunTrace:
    """One run's trace: a step per AI and tool message, totals, context fill.

    Steps are ordered by their stamped start when every step has one, else
    kept in message order. ``events`` gets ``{"type": "compaction"}`` when the
    turn starts from a compaction summary and ``{"type":
    "background_started", "task_id"}`` per ``start_async_task`` result.
    """
    info = model_info(model)
    steps: list[StepTrace] = []
    firsts: list[datetime] = []
    extra: list[dict] = []
    if turn and is_compaction_summary(turn[0]):
        extra.append({"type": "compaction"})
    for i, message in enumerate(turn):
        if isinstance(message, AIMessage):
            steps.append(model_step(message, info, fallback_id=f"model-{i}"))
            if (first := _first_token(message)) is not None:
                firsts.append(first)
        elif isinstance(message, ToolMessage):
            steps.append(tool_step(message))
            if message.name == START_TOOL and message.status != "error":
                task_id = _task_id(message)
                if task_id is not None:
                    extra.append({"type": "background_started", "task_id": task_id})
    if steps and all(s.started_at is not None for s in steps):
        steps.sort(key=lambda s: s.started_at)

    models = [s for s in steps if s.kind == "model"]
    starts = [s.started_at for s in steps if s.started_at is not None]
    ends = [s.ended_at for s in steps if s.ended_at is not None]
    started_at = min(starts) if starts else None
    ended_at = max(ends) if ends else None
    total_s = (
        (ended_at - started_at).total_seconds()
        if started_at is not None and ended_at is not None
        else None
    )
    ttft_s = (
        (min(firsts) - started_at).total_seconds()
        if firsts and started_at is not None
        else None
    )
    return RunTrace(
        run_id=run_id,
        model=model,
        started_at=started_at,
        ended_at=ended_at,
        ttft_s=ttft_s,
        total_s=total_s,
        steps=steps,
        tokens_in=_total([s.tokens_in for s in models]),
        tokens_out=_total([s.tokens_out for s in models]),
        tokens_cached=_total([s.tokens_cached for s in models]),
        cost_usd=_total([s.cost_usd for s in models]),
        context_tokens=models[-1].tokens_in if models else None,
        context_window=info.context_window if info is not None else None,
        compaction_trigger=compaction_trigger,
        events=[*events, *extra],
        verification=verification,
    )
