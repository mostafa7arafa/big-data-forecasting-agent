"""The analyst graph: plan -> analyst -> verify -> answer, served by ``langgraph dev``.

The module-level ``graph`` imports without API keys or ``ANALYST_DATASET``:
the model, dataset, cutoff and background flag are all resolved per run from
``Runtime.context``. The analyst's system prompt depends on that context, so
the inner ``create_agent`` is built per run and cached by
(model spec, dataset, cutoff, background).

The nodes are async and run on the server's event loop, so everything that
touches the disk or builds clients (settings, services, the result store,
model and agent construction, verification) goes through ``asyncio.to_thread``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import replace
from datetime import date
from functools import lru_cache
from typing import Any, Literal

from deepagents import AsyncSubAgentMiddleware
from langchain.agents import create_agent
from langchain.agents.structured_output import StructuredOutputError
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import ValidationError

from analyst.background import (
    ASYNC_TASK_TOOL_NAMES,
    WORKER,
    BackgroundGuard,
    is_background_notification,
    task_context_for_notification,
)
from analyst.config import get_settings
from analyst.context import AnalystContext
from analyst.data.registry import get_service
from analyst.middleware import analyst_middleware
from analyst.prompts import PLAN_PROMPT, build_system_prompt
from analyst.providers import get_model
from analyst.state import AnalystState, Answer, Plan, Verification
from analyst.tools import DATA_TOOLS
from analyst.verify import NO_CLAIMS, collect_handles, verify_answer

logger = logging.getLogger(__name__)

__all__ = ["LABEL_CHIPS", "build_graph", "graph", "label_chip"]

#: tag on messages the graph itself adds to the thread (verify feedback, plan).
SOURCE_KEY = "source"
_VERIFY_SOURCE = "verify"
_PLAN_SOURCE = "plan"
#: the async-task tools the analyst starts background work with. Background
#: work (prompt policy, forecast refusal) is on only while this is non-empty.
ASYNC_TASK_TOOLS: list[str] = sorted(ASYNC_TASK_TOOL_NAMES)
#: inner-agent state the thread keeps between turns (background task records).
_CARRIED_KEYS = ("async_tasks", "background_contexts")

RETRY_TEXT = (
    "Verification failed: {problems}. Fix the answer: cite handles for every "
    "number, hedge and cite facts for causes."
)

LABEL_CHIPS = {
    "verified": "Verified",
    "verified_off_model": "Verified · off-model",
    "not_verified": "Not verified",
}

_AGENT_CACHE_SIZE = 32


def label_chip(verification: Verification) -> str:
    """One-line label shown under the answer, e.g. ``Verified · off-model (run_sql)``."""
    chip = LABEL_CHIPS[verification.label]
    if verification.label == "verified_off_model" and verification.off_model_tools:
        return f"{chip} ({', '.join(verification.off_model_tools)})"
    if verification.label == "not_verified":
        return f"{chip}: {verification.reason}"
    return chip


def _source(message: Any) -> str | None:
    for location in ("additional_kwargs", "response_metadata"):
        meta = getattr(message, location, None)
        if isinstance(meta, dict) and isinstance(meta.get(SOURCE_KEY), str):
            return meta[SOURCE_KEY]
    return None


def _text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(
        part if isinstance(part, str) else str(part.get("text", ""))
        for part in content
        if isinstance(part, str | dict)
    )


def _question(messages: Sequence[BaseMessage]) -> HumanMessage | None:
    """The latest real user turn: not one the graph wrote, not a worker note.

    Numbers in the question are exempt from tracing, so a background task's
    notification must never stand in for it.
    """
    for message in reversed(messages):
        if (
            isinstance(message, HumanMessage)
            and _source(message) not in (_VERIFY_SOURCE, _PLAN_SOURCE)
            and not is_background_notification(message)
        ):
            return message
    return None


def _turn(messages: Sequence[BaseMessage]) -> Sequence[BaseMessage]:
    """This turn's messages: those after the latest real user question.

    A woken run's worker notification comes after that question, so its
    handles count as this turn's. Without a question, every message counts.
    """
    question = _question(messages)
    if question is None:
        return messages
    start = next(i for i, m in enumerate(messages) if m is question) + 1
    return messages[start:]


_ANSWER_TOOL = Answer.__name__


def _without_answer_call(message: AIMessage, answer_ids: set[str]) -> AIMessage | None:
    """``message`` minus its ``Answer`` tool call, or None when nothing else is left.

    The call is removed from ``tool_calls`` and from any provider copies (content
    blocks, raw ``additional_kwargs["tool_calls"]``) so no orphan call survives.
    """
    rest = [tc for tc in message.tool_calls if tc["id"] not in answer_ids]
    if not rest:
        return None
    update: dict[str, Any] = {"tool_calls": rest}
    if isinstance(message.content, list):
        update["content"] = [
            block
            for block in message.content
            if not (
                isinstance(block, dict)
                and (block.get("id") in answer_ids or block.get("call_id") in answer_ids)
            )
        ]
    raw = message.additional_kwargs.get("tool_calls")
    if isinstance(raw, list):
        update["additional_kwargs"] = {
            **message.additional_kwargs,
            "tool_calls": [
                tc for tc in raw if not (isinstance(tc, dict) and tc.get("id") in answer_ids)
            ],
        }
    return message.model_copy(update=update)


def _drop_structured_response(
    new: list[BaseMessage], *, structured: bool
) -> list[BaseMessage]:
    """The agent's new messages minus the raw structured answer.

    The answer node posts the visible final answer, so the agent's own copy is
    clutter: under the tool strategy an ``Answer`` tool call and its tool
    message, under the provider strategy the closing JSON-text AI message.
    Status lines, data tool calls and their results stay, including data calls
    made in the same AI message as the ``Answer`` call.
    """
    answer_ids = {
        tc["id"]
        for m in new
        if isinstance(m, AIMessage)
        for tc in m.tool_calls
        if tc["name"] == _ANSWER_TOOL
    }
    kept: list[BaseMessage] = []
    for m in new:
        if isinstance(m, AIMessage) and any(tc["id"] in answer_ids for tc in m.tool_calls):
            stripped = _without_answer_call(m, answer_ids)
            if stripped is not None:
                kept.append(stripped)
        elif not (isinstance(m, ToolMessage) and m.tool_call_id in answer_ids):
            kept.append(m)
    if (
        structured
        and not answer_ids
        and kept
        and isinstance(kept[-1], AIMessage)
        and not kept[-1].tool_calls
    ):
        kept.pop()  # provider strategy: the final message is the JSON answer
    return kept


def _narrow(ctx: AnalystContext, wake: Any) -> AnalystContext:
    """``ctx`` narrowed to a woken task's recorded start context.

    A run started by a background notification carries whatever context the
    worker sent back; the task's recorded dataset and cutoff win, and the
    earlier cutoff applies, so a wake-up never widens what the task saw.
    """
    if not isinstance(wake, dict) or not wake.get("dataset"):
        return ctx
    recorded = wake.get("cutoff")
    recorded = date.fromisoformat(recorded) if recorded else None
    if wake["dataset"] != ctx.dataset:
        return replace(ctx, dataset=wake["dataset"], cutoff=recorded)
    cutoffs = [c for c in (ctx.cutoff, recorded) if c is not None]
    return replace(ctx, cutoff=min(cutoffs) if cutoffs else None)


def _context(runtime: Runtime[AnalystContext], wake: Any = None) -> AnalystContext:
    """The run's context with background work gated on the async-task tools.

    ``wake`` is the turn's ``wake_context`` (see :func:`_narrow`).
    Blocking (settings may read ``.env``): call it through ``asyncio.to_thread``.
    """
    ctx = runtime.context
    if ctx is None:
        ctx = AnalystContext.from_settings()
    elif isinstance(ctx, dict):
        ctx = AnalystContext.from_settings(**ctx)
    ctx = _narrow(ctx, wake)
    background = ctx.background and bool(ASYNC_TASK_TOOLS)
    return ctx if ctx.background == background else replace(ctx, background=background)


def build_graph(
    *, model: BaseChatModel | None = None, extra_middleware: list | None = None
) -> CompiledStateGraph:
    """Compile the analyst graph.

    ``model=None`` resolves ``get_model(context.model)`` at run time; a given
    model (tests) is used for both the plan and analyst nodes.
    """
    extra = list(extra_middleware or [])

    def resolve_model(spec: str | None) -> BaseChatModel:
        return model if model is not None else get_model(spec)

    @lru_cache(maxsize=_AGENT_CACHE_SIZE)
    def analyst_agent(
        spec: str | None, dataset: str, cutoff: date | None, background: bool
    ):
        memory = get_service(dataset, cutoff).memory
        settings = get_settings()
        return create_agent(
            resolve_model(spec),
            tools=DATA_TOOLS,
            middleware=[
                *analyst_middleware(settings),
                AsyncSubAgentMiddleware(async_subagents=[WORKER]),
                BackgroundGuard(settings.max_background_tasks),
                *extra,
            ],
            system_prompt=build_system_prompt(
                memory, dataset=dataset, cutoff=cutoff, background=background
            ),
            response_format=Answer,
            context_schema=AnalystContext,
            name="analyst_agent",
        )

    async def plan(state: AnalystState, runtime: Runtime[AnalystContext]) -> dict:
        # Every turn starts clean: a checkpointed thread keeps the last turn's
        # draft, verification and retry count otherwise.
        reset: dict[str, Any] = {
            "plan": [],
            "draft": None,
            "verification": None,
            "retries": 0,
            "wake_context": None,
        }
        messages = state["messages"]
        if messages and is_background_notification(messages[-1]):
            # Decided once per turn: later nodes see verify feedback last.
            reset["wake_context"] = task_context_for_notification(
                messages[-1], state.get("background_contexts")
            )
            return reset
        question = _question(messages)
        if question is None:
            return reset
        ctx = await asyncio.to_thread(_context, runtime, None)
        chat = await asyncio.to_thread(resolve_model, ctx.model)
        planner = chat.with_structured_output(Plan)
        try:
            result = await planner.ainvoke(
                [SystemMessage(PLAN_PROMPT), HumanMessage(_text(question))]
            )
            if isinstance(result, dict):
                result = Plan.model_validate(result)
        except Exception:  # a planning failure must not fail the question
            logger.warning("plan node failed; continuing without a plan", exc_info=True)
            return reset
        if not isinstance(result, Plan) or not result.needs_plan or not result.steps:
            return reset
        runtime.stream_writer({"type": "plan", "steps": list(result.steps)})
        return {**reset, "plan": list(result.steps)}

    async def analyst(state: AnalystState, runtime: Runtime[AnalystContext]) -> dict:
        ctx = await asyncio.to_thread(_context, runtime, state.get("wake_context"))
        agent = await asyncio.to_thread(
            analyst_agent, ctx.model, ctx.dataset, ctx.cutoff, ctx.background
        )
        messages = list(state["messages"])
        steps = state.get("plan") or []
        if steps and not state.get("retries"):
            # The plan rides along for this call only; it is not thread history.
            numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))
            messages.append(
                HumanMessage(
                    f"Plan for this question:\n{numbered}",
                    additional_kwargs={SOURCE_KEY: _PLAN_SOURCE},
                )
            )
        # Stream state snapshots so the work done so far survives a
        # structured-output failure; that failure must never escape the graph
        # (it becomes the answer node's "no structured answer" path).
        # Background task records ride through the inner agent and back.
        carried = {key: dict(state.get(key) or {}) for key in _CARRIED_KEYS}
        last: dict[str, Any] = {"messages": messages}
        try:
            async for snapshot in agent.astream(
                {"messages": messages, **carried},
                context=ctx,
                stream_mode="values",
            ):
                last = snapshot
        except (StructuredOutputError, ValidationError, ValueError):
            logger.warning("analyst gave no valid structured answer", exc_info=True)
            last = {**last, "structured_response": None}
        draft = last.get("structured_response")
        if not isinstance(draft, Answer):
            draft = None
        new = _drop_structured_response(
            list(last["messages"][len(messages) :]), structured=draft is not None
        )
        update: dict[str, Any] = {"messages": new, "draft": draft}
        for key, before in carried.items():
            if last.get(key) and last[key] != before:
                update[key] = last[key]
        return update

    async def verify(
        state: AnalystState, runtime: Runtime[AnalystContext]
    ) -> Command[Literal["analyst", "answer"]]:
        draft = state.get("draft")
        if draft is None:
            return Command(goto="answer")
        ctx = await asyncio.to_thread(_context, runtime, state.get("wake_context"))
        messages = state["messages"]
        question = _question(messages)

        def check() -> Verification:  # reads the result store: off the loop
            service = get_service(ctx.dataset, ctx.cutoff)
            return verify_answer(
                draft,
                question=_text(question) if question is not None else "",
                handles=collect_handles(_turn(messages), service.store),
                memory=service.memory,
                cutoff=ctx.cutoff,
                store=service.store,
            )

        verification = await asyncio.to_thread(check)
        # A decline or acknowledgement (no claims, no numbers) is final: a
        # "cite handles" retry would only push the model to invent numbers.
        if (
            verification.label == "not_verified"
            and verification.reason != NO_CLAIMS
            and not state.get("retries")
        ):
            problems = "; ".join(verification.problems or [verification.reason])
            feedback = HumanMessage(
                RETRY_TEXT.format(problems=problems),
                additional_kwargs={SOURCE_KEY: _VERIFY_SOURCE},
            )
            return Command(
                goto="analyst",
                update={
                    "messages": [feedback],
                    "retries": 1,
                    "verification": verification,
                },
            )
        return Command(goto="answer", update={"verification": verification})

    async def answer(state: AnalystState) -> dict:
        draft = state.get("draft")
        verification = state.get("verification")
        if draft is None:
            # The latest prose reply of this turn, never an earlier turn's answer.
            last_ai = next(
                (
                    m
                    for m in reversed(_turn(state["messages"]))
                    if isinstance(m, AIMessage) and not m.tool_calls
                ),
                None,
            )
            verification = Verification(
                label="not_verified", reason="no structured answer"
            )
            final = Answer(
                narrative=_text(last_ai) if last_ai is not None else "",
                claims=[],
                verification=verification,
            )
        else:
            if verification is None:  # unreachable via verify; keep the label honest
                verification = Verification(label="not_verified", reason="not checked")
            final = draft.model_copy(update={"verification": verification})
        message = AIMessage(content=final.narrative + "\n\n" + label_chip(verification))
        return {"answer": final, "verification": verification, "messages": [message]}

    builder = StateGraph(AnalystState, context_schema=AnalystContext)
    builder.add_node("plan", plan)
    builder.add_node("analyst", analyst)
    builder.add_node("verify", verify)
    builder.add_node("answer", answer)
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "analyst")
    builder.add_edge("analyst", "verify")
    builder.add_edge("answer", END)
    return builder.compile(name="analyst")


graph = build_graph()
