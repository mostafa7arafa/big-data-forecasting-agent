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

from analyst.context import AnalystContext
from analyst.data.registry import get_service
from analyst.middleware import analyst_middleware
from analyst.prompts import PLAN_PROMPT, build_system_prompt
from analyst.providers import get_model
from analyst.state import AnalystState, Answer, Plan, Verification
from analyst.tools import DATA_TOOLS
from analyst.verify import collect_handles, verify_answer

logger = logging.getLogger(__name__)

__all__ = ["LABEL_CHIPS", "build_graph", "graph", "label_chip"]

#: tag on messages the graph itself adds to the thread (verify feedback, plan).
SOURCE_KEY = "source"
_VERIFY_SOURCE = "verify"
_PLAN_SOURCE = "plan"
#: tag the Agent Server puts on a background task's completion message
#: (``additional_kwargs["lc_source"]`` today; ``source`` kept for older shapes).
_BACKGROUND_SOURCE = "async_subagent"
_BACKGROUND_KEYS = ("lc_source", SOURCE_KEY)

#: the async-task tools the analyst can start background work with. Task 11
#: fills this; until then background work is off whatever the context says.
ASYNC_TASK_TOOLS: list = []

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


def _is_background_note(message: Any) -> bool:
    for location in ("additional_kwargs", "response_metadata"):
        meta = getattr(message, location, None)
        if isinstance(meta, dict) and any(
            meta.get(key) == _BACKGROUND_SOURCE for key in _BACKGROUND_KEYS
        ):
            return True
    return False


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
            and not _is_background_note(message)
        ):
            return message
    return None


_ANSWER_TOOL = Answer.__name__


def _drop_structured_response(
    new: list[BaseMessage], *, structured: bool
) -> list[BaseMessage]:
    """The agent's new messages minus the raw structured answer.

    The answer node posts the visible final answer, so the agent's own copy is
    clutter: under the tool strategy an ``Answer`` tool call and its tool
    message, under the provider strategy the closing JSON-text AI message.
    Status lines, data tool calls and their results stay.
    """
    answer_ids = {
        tc["id"]
        for m in new
        if isinstance(m, AIMessage)
        for tc in m.tool_calls
        if tc["name"] == _ANSWER_TOOL
    }
    kept = [
        m
        for m in new
        if not (
            isinstance(m, AIMessage)
            and any(tc["name"] == _ANSWER_TOOL for tc in m.tool_calls)
        )
        and not (isinstance(m, ToolMessage) and m.tool_call_id in answer_ids)
    ]
    if (
        structured
        and not answer_ids
        and kept
        and isinstance(kept[-1], AIMessage)
        and not kept[-1].tool_calls
    ):
        kept.pop()  # provider strategy: the final message is the JSON answer
    return kept


def _context(runtime: Runtime[AnalystContext]) -> AnalystContext:
    """The run's context with background work gated on the async-task tools.

    Blocking (settings may read ``.env``): call it through ``asyncio.to_thread``.
    """
    ctx = runtime.context
    if ctx is None:
        ctx = AnalystContext.from_settings()
    elif isinstance(ctx, dict):
        ctx = AnalystContext.from_settings(**ctx)
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
        return create_agent(
            resolve_model(spec),
            tools=DATA_TOOLS,
            middleware=[*analyst_middleware(), *extra],
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
        }
        messages = state["messages"]
        if messages and _is_background_note(messages[-1]):
            return reset
        question = _question(messages)
        if question is None:
            return reset
        ctx = await asyncio.to_thread(_context, runtime)
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
        ctx = await asyncio.to_thread(_context, runtime)
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
        last: dict[str, Any] = {"messages": messages}
        try:
            async for snapshot in agent.astream(
                {"messages": messages}, context=ctx, stream_mode="values"
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
        return {"messages": new, "draft": draft}

    async def verify(
        state: AnalystState, runtime: Runtime[AnalystContext]
    ) -> Command[Literal["analyst", "answer"]]:
        draft = state.get("draft")
        if draft is None:
            return Command(goto="answer")
        ctx = await asyncio.to_thread(_context, runtime)
        messages = state["messages"]
        question = _question(messages)

        def check() -> Verification:  # reads the result store: off the loop
            service = get_service(ctx.dataset, ctx.cutoff)
            return verify_answer(
                draft,
                question=_text(question) if question is not None else "",
                handles=collect_handles(messages, service.store),
                memory=service.memory,
                cutoff=ctx.cutoff,
                store=service.store,
            )

        verification = await asyncio.to_thread(check)
        if verification.label == "not_verified" and not state.get("retries"):
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
            messages = state["messages"]
            question = _question(messages)
            start = 0
            if question is not None:
                start = next(i for i, m in enumerate(messages) if m is question) + 1
            last_ai = next(
                (
                    m
                    for m in reversed(messages[start:])
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
