"""The run trace: TraceMiddleware timing and live events, build_trace, the answer node."""

import asyncio
import json
from datetime import UTC, datetime, timedelta

import pytest
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    ToolMessage,
)
from langchain_core.messages.notification import add_notification_metadata
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import tool

from analyst.config import get_settings
from analyst.context import AnalystContext
from analyst.middleware.compaction import COMPACTION_SOURCE
from analyst.state import Verification
from analyst.trace import RunTrace, TraceMiddleware, build_trace
from tests.unit.test_graph import (  # noqa: F401  (events_service is an autouse fixture)
    COUNT,
    CTX,
    ScriptedModel,
    call,
    cite_total,
    events_service,
    no_plan,
    tool_turn,
    untraced,
)

LUNA = "openai:gpt-6-luna"  # priced in config/models.yaml: 0.10 / 0.01 / 0.50 per 1M
T0 = datetime(2026, 11, 15, 12, 0, 0, tzinfo=UTC)


class StreamingModel(BaseChatModel):
    """Streams each scripted reply word by word (``streaming=True`` forces it)."""

    replies: list[AIMessage]
    streaming: bool = False

    @property
    def _llm_type(self) -> str:
        return "streaming-fake"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self.replies.pop(0))])

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        reply = self.replies.pop(0)
        for word in str(reply.content).split(" "):
            yield ChatGenerationChunk(message=AIMessageChunk(content=word + " "))
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {
                        "name": tc["name"],
                        "args": json.dumps(tc["args"]),
                        "id": tc["id"],
                        "index": i,
                    }
                    for i, tc in enumerate(reply.tool_calls)
                ],
                usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110},
            )
        )


@tool
def lookup(x: int) -> str:
    """Look a number up."""
    return f"value {x}"


def _collect(model: BaseChatModel):
    agent = create_agent(model, tools=[lookup], middleware=[TraceMiddleware()])

    async def go():
        events, last = [], None
        async for mode, chunk in agent.astream(
            {"messages": [HumanMessage("hi")]}, stream_mode=["custom", "values"]
        ):
            if mode == "custom":
                events.append(chunk)
            else:
                last = chunk
        return events, last

    return asyncio.run(go())


def _replies():
    return [
        AIMessage("Looking it up now", tool_calls=[call("lookup", {"x": 1}, "t1")]),
        AIMessage("The value is one"),
    ]


@pytest.mark.parametrize("streaming", [True, False])
def test_trace_middleware_stamps_timing_and_emits_events(streaming):
    model = StreamingModel(replies=_replies(), streaming=streaming)
    events, state = _collect(model)

    ai = [m for m in state["messages"] if isinstance(m, AIMessage)]
    tools = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    assert len(ai) == 2 and len(tools) == 1
    for message in [*ai, *tools]:
        stamp = message.response_metadata["analyst_trace"]
        started = datetime.fromisoformat(stamp["started_at"])
        ended = datetime.fromisoformat(stamp["ended_at"])
        assert started <= ended
        if isinstance(message, ToolMessage):
            assert stamp["first_token_at"] is None
        elif streaming:
            first = datetime.fromisoformat(stamp["first_token_at"])
            assert started <= first <= ended
        else:
            assert stamp["first_token_at"] is None  # the model did not stream

    trace = [e for e in events if e.get("type") == "trace"]
    kinds = [(e["event"], e["step"]["kind"]) for e in trace]
    assert kinds == [
        ("step_started", "model"),
        ("step_finished", "model"),
        ("step_started", "tool"),
        ("step_finished", "tool"),
        ("step_started", "model"),
        ("step_finished", "model"),
    ]
    started, finished = trace[0]["step"], trace[1]["step"]
    assert started["id"] == finished["id"]
    assert started["ended_at"] is None and finished["ended_at"] is not None
    assert finished["status"] == "ok"
    tool_step = trace[3]["step"]
    assert tool_step["name"] == "lookup" and tool_step["id"] == "t1"
    if streaming:
        assert finished["tokens_in"] == 100 and finished["tokens_out"] == 10


def test_trace_middleware_is_silent_outside_a_graph():
    """No graph context: no stream writer, nothing raised."""
    from analyst.trace.middleware import emit

    emit("step_started", None)


def _ai(
    *,
    usage: dict | None,
    start: float,
    end: float,
    first: float | None = None,
    content="",
    tool_calls=None,
    msg_id=None,
) -> AIMessage:
    stamp = {
        "started_at": (T0 + timedelta(seconds=start)).isoformat(),
        "ended_at": (T0 + timedelta(seconds=end)).isoformat(),
        "first_token_at": None if first is None else (T0 + timedelta(seconds=first)).isoformat(),
    }
    return AIMessage(
        content=content,
        tool_calls=tool_calls or [],
        usage_metadata=usage,
        response_metadata={"analyst_trace": stamp},
        id=msg_id,
    )


def _tool(name: str, call_id: str, content: str, *, start: float, end: float) -> ToolMessage:
    stamp = {
        "started_at": (T0 + timedelta(seconds=start)).isoformat(),
        "ended_at": (T0 + timedelta(seconds=end)).isoformat(),
        "first_token_at": None,
    }
    return ToolMessage(
        content, tool_call_id=call_id, name=name, response_metadata={"analyst_trace": stamp}
    )


def _usage(i: int, o: int, cached: int | None = None, reasoning: int | None = None, cost=None):
    usage: dict = {"input_tokens": i, "output_tokens": o, "total_tokens": i + o}
    if cached is not None:
        usage["input_token_details"] = {"cache_read": cached}
    if reasoning is not None:
        usage["output_token_details"] = {"reasoning": reasoning}
    if cost is not None:
        usage["total_cost"] = cost
    return usage


def test_build_trace_tokens_cost_and_context():
    turn = [
        HumanMessage("how many?"),
        _ai(
            usage=_usage(1_000_000, 100_000, cached=500_000, reasoning=40_000),
            start=0,
            end=2,
            first=0.5,
            tool_calls=[call("query_metrics", {}, "c1")],
        ),
        _tool("query_metrics", "c1", "card", start=2, end=3),
        _tool("start_async_task", "c2", "Launched async subagent. task_id: abc-123", start=3, end=4),
        _ai(usage=_usage(2_000, 1_000, cached=0), start=4, end=6),
    ]
    verification = Verification(label="verified", reason="ok")
    trace = build_trace(
        turn,
        model=LUNA,
        run_id="run-1",
        verification=verification,
        events=[{"type": "plan"}],
        compaction_trigger=60_000,
    )
    assert isinstance(trace, RunTrace)
    assert [s.kind for s in trace.steps] == ["model", "tool", "tool", "model"]
    first = trace.steps[0]
    assert (first.tokens_in, first.tokens_out, first.tokens_cached, first.tokens_reasoning) == (
        1_000_000,
        100_000,
        500_000,
        40_000,
    )
    # uncached 0.5M * 0.10 + cached 0.5M * 0.01 + 0.1M output * 0.50
    assert first.cost_usd == pytest.approx(0.05 + 0.005 + 0.05)
    second = trace.steps[3]
    assert second.cost_usd == pytest.approx(2_000 * 0.10 / 1e6 + 1_000 * 0.50 / 1e6)
    assert trace.tokens_in == 1_002_000
    assert trace.tokens_out == 101_000
    assert trace.tokens_cached == 500_000
    assert trace.cost_usd == pytest.approx(first.cost_usd + second.cost_usd)
    assert trace.context_tokens == 2_000  # the last model step's input
    assert trace.context_window == 1_050_000
    assert trace.compaction_trigger == 60_000
    assert trace.ttft_s == pytest.approx(0.5)
    assert trace.total_s == pytest.approx(6.0)
    assert trace.run_id == "run-1" and trace.model == LUNA
    assert trace.verification == verification
    assert trace.events == [{"type": "plan"}, {"type": "background_started", "task_id": "abc-123"}]
    assert trace.interrupted is False
    assert trace.steps[1].name == "query_metrics" and trace.steps[1].status == "ok"
    RunTrace.model_validate(trace.model_dump(mode="json"))


def test_trace_without_usage_or_prices():
    turn = [
        _ai(usage=None, start=0, end=1, tool_calls=[call("query_metrics", {}, "c1")]),
        _tool("query_metrics", "c1", "card", start=1, end=2),
        _ai(usage=_usage(10, 5), start=2, end=3),
    ]
    trace = build_trace(
        turn, model="openai:unknown", run_id=None, verification=None, events=[],
        compaction_trigger=1,
    )
    first = trace.steps[0]
    assert first.tokens_in is None and first.tokens_out is None
    assert first.tokens_cached is None and first.tokens_reasoning is None
    assert first.cost_usd is None
    assert trace.steps[2].cost_usd is None  # tokens known, but no price table
    assert trace.tokens_in is None and trace.tokens_out is None
    assert trace.tokens_cached is None
    assert trace.cost_usd is None
    assert trace.context_window is None
    assert trace.ttft_s is None

    # A priced model with one unpriceable step: the run total is unknown.
    priced = build_trace(
        turn, model=LUNA, run_id=None, verification=None, events=[], compaction_trigger=1
    )
    assert priced.steps[2].cost_usd is not None
    assert priced.cost_usd is None

    empty = build_trace([], model=LUNA, run_id=None, verification=None, events=[],
                        compaction_trigger=1)
    assert empty.steps == [] and empty.cost_usd is None and empty.tokens_in is None
    assert empty.context_tokens is None and empty.total_s is None


def test_trace_prefers_reported_total_cost():
    turn = [_ai(usage=_usage(1_000_000, 0, cost=0.42), start=0, end=1)]
    trace = build_trace(
        turn, model=LUNA, run_id=None, verification=None, events=[], compaction_trigger=1
    )
    assert trace.steps[0].cost_usd == pytest.approx(0.42)
    assert trace.cost_usd == pytest.approx(0.42)


def test_reasoning_text_from_content_blocks():
    message = _ai(
        usage=_usage(10, 20, reasoning=15),
        start=0,
        end=1,
        content=[
            {"type": "reasoning", "reasoning": "First, count the rows."},
            {"type": "reasoning", "reasoning": "Then compare."},
            {"type": "text", "text": "Done."},
        ],
    )
    trace = build_trace(
        [message], model=LUNA, run_id=None, verification=None, events=[], compaction_trigger=1
    )
    step = trace.steps[0]
    assert step.reasoning == "First, count the rows.\n\nThen compare."
    assert step.tokens_reasoning == 15
    plain = build_trace(
        [_ai(usage=None, start=0, end=1, content="hi")],
        model=LUNA, run_id=None, verification=None, events=[], compaction_trigger=1,
    )
    assert plain.steps[0].reasoning is None


def test_compaction_event_when_turn_starts_from_summary():
    summary = add_notification_metadata(
        HumanMessage("## Earlier conversation (compacted)"), source=COMPACTION_SOURCE, data={}
    )
    trace = build_trace(
        [summary, HumanMessage("q"), _ai(usage=None, start=0, end=1)],
        model=LUNA, run_id=None, verification=None, events=[], compaction_trigger=1,
    )
    assert trace.events == [{"type": "compaction"}]


def _run(graph, state):
    return asyncio.run(graph.ainvoke(state, context=CTX))


def test_answer_node_appends_trace_per_turn():
    from analyst.graph import build_graph

    model = ScriptedModel(
        script=[
            no_plan(),
            tool_turn(call("query_metrics", COUNT, "c1")),
            cite_total,
            no_plan(),
            tool_turn(call("query_metrics", COUNT, "c2")),
            cite_total,
        ]
    )
    graph = build_graph(model=model)
    first = _run(graph, {"messages": [HumanMessage("how many events?")]})
    assert len(first["trace"]) == 1
    second = _run(
        graph,
        {
            "messages": [*first["messages"], HumanMessage("and again?")],
            "trace": first["trace"],
        },
    )
    records = second["trace"]
    assert len(records) == 2
    for record in records:
        trace = RunTrace.model_validate(record)
        # the plan call, the model call with tools, the tool, then the final
        # (Answer) model call, which the thread drops but the trace keeps.
        assert [s.kind for s in trace.steps] == ["model", "model", "tool", "model"]
        assert trace.steps[0].name == "plan"
        assert trace.verification.label == "verified"
        assert trace.model == get_settings().model  # CTX names no model: the default
    assert RunTrace.model_validate(records[1]).steps[2].id == "c2"


def test_trace_events_stream_through_the_analyst_graph():
    from analyst.graph import build_graph

    model = ScriptedModel(
        script=[no_plan(), tool_turn(call("query_metrics", COUNT, "c1")), cite_total]
    )
    graph = build_graph(model=model)

    async def go():
        return [
            chunk
            async for chunk in graph.astream(
                {"messages": [HumanMessage("how many events?")]},
                context=CTX,
                stream_mode="custom",
            )
        ]

    events = [e for e in asyncio.run(go()) if isinstance(e, dict) and e.get("type") == "trace"]
    assert [e["event"] for e in events].count("step_finished") == 4  # plan, model, tool, model


def test_worker_records_its_own_trace():
    from analyst.worker import build_worker

    worker = build_worker(model=StreamingModel(replies=[AIMessage("report done")]))
    out = asyncio.run(worker.ainvoke({"messages": [HumanMessage("task")]}))
    assert len(out["trace"]) == 1
    trace = RunTrace.model_validate(out["trace"][0])
    assert [s.kind for s in trace.steps] == ["model"]
    assert trace.verification is None


def test_run_boundary_woken_run_and_compaction():
    from analyst.background import NOTIFICATION_SOURCE
    from analyst.graph import _run_messages

    summary = add_notification_metadata(
        HumanMessage("digest"), source=COMPACTION_SOURCE, data={}
    )
    earlier = _ai(usage=None, start=0, end=1, content="earlier run", msg_id="old")
    note = add_notification_metadata(
        HumanMessage("task t1 done"), source=NOTIFICATION_SOURCE, data={"task_id": "t1"}
    )
    woken = _ai(usage=None, start=5, end=6, content="woken run", msg_id="new")
    hidden = _ai(usage=None, start=6, end=7, msg_id="hidden")
    messages = [summary, HumanMessage("q"), earlier, note, woken]

    run = _run_messages({"messages": messages, "trace_hidden": [hidden]})
    assert [m.id for m in run] == ["new", "hidden"]  # not the earlier run's steps
    compacted = _run_messages({"messages": messages, "compacted": True})
    assert compacted[0] is summary and [m.id for m in compacted[1:]] == ["new"]


def test_reasoning_text_from_openai_responses_shape():
    message = AIMessage(
        content=[
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [
                    {"type": "summary_text", "text": "Count first."},
                    {"type": "summary_text", "text": "Then compare."},
                ],
            },
            {"type": "text", "text": "Done."},
        ],
        response_metadata={"model_provider": "openai"},
    )
    trace = build_trace(
        [message], model=LUNA, run_id=None, verification=None, events=[], compaction_trigger=1
    )
    assert trace.steps[0].reasoning == "Count first.\n\nThen compare."


def test_trace_keeps_both_answer_calls_on_verify_retry():
    from analyst.graph import build_graph

    model = ScriptedModel(
        script=[
            no_plan(),
            tool_turn(call("query_metrics", COUNT, "c1")),
            untraced("a1"),
            cite_total,
        ]
    )
    out = _run(build_graph(model=model), {"messages": [HumanMessage("how many events?")]})
    assert out["retries"] == 1
    trace = RunTrace.model_validate(out["trace"][-1])
    assert [s.kind for s in trace.steps] == ["model", "model", "tool", "model", "model"]


def test_worker_trace_uses_the_task_header_model():
    from analyst.background import format_context_header
    from analyst.worker import build_worker

    header = format_context_header(
        AnalystContext(dataset="events", model="openai:gpt-6-luna", background=False)
    )
    worker = build_worker(model=StreamingModel(replies=[AIMessage("report done")]))
    out = asyncio.run(worker.ainvoke({"messages": [HumanMessage(f"{header}\ndo the task")]}))
    assert RunTrace.model_validate(out["trace"][-1]).model == "openai:gpt-6-luna"


def test_trace_events_with_subgraphs_stream():
    """With ``subgraphs=True`` each event still arrives once, on the root namespace."""
    from analyst.graph import build_graph

    model = ScriptedModel(
        script=[no_plan(), tool_turn(call("query_metrics", COUNT, "c1")), cite_total]
    )
    graph = build_graph(model=model)

    async def go():
        return [
            (ns, chunk)
            async for ns, chunk in graph.astream(
                {"messages": [HumanMessage("how many events?")]},
                context=CTX,
                stream_mode="custom",
                subgraphs=True,
            )
        ]

    events = [
        (ns, e) for ns, e in asyncio.run(go()) if isinstance(e, dict) and e.get("type") == "trace"
    ]
    assert all(ns == () for ns, _ in events)
    assert [e["event"] for _, e in events].count("step_finished") == 4  # plan, model, tool, model


def _with_usage(step, i: int, o: int):
    """A script step whose AI message reports ``i``/``o`` tokens."""

    def run(messages):
        message = step(messages) if callable(step) else step
        return message.model_copy(
            update={"usage_metadata": {"input_tokens": i, "output_tokens": o, "total_tokens": i + o}}
        )

    return run


def test_trace_includes_plan_model_call():
    from analyst.graph import build_graph

    model = ScriptedModel(
        script=[
            _with_usage(no_plan(), 100, 10),
            _with_usage(tool_turn(call("query_metrics", COUNT, "c1")), 200, 20),
            _with_usage(cite_total, 300, 30),
        ]
    )
    out = _run(build_graph(model=model), {"messages": [HumanMessage("how many events?")]})
    trace = RunTrace.model_validate(out["trace"][-1])
    plan = trace.steps[0]
    assert (plan.name, plan.tokens_in, plan.tokens_out) == ("plan", 100, 10)
    assert trace.tokens_in == 600 and trace.tokens_out == 60
    assert trace.context_tokens == 300  # still the last agent call
    assert trace.started_at == plan.started_at  # TTFT/total measure from the first call


def test_trace_plan_without_usage_makes_totals_unknown():
    from analyst.graph import build_graph

    model = ScriptedModel(
        script=[
            no_plan(),
            _with_usage(tool_turn(call("query_metrics", COUNT, "c1")), 200, 20),
            _with_usage(cite_total, 300, 30),
        ]
    )
    out = _run(build_graph(model=model), {"messages": [HumanMessage("how many events?")]})
    trace = RunTrace.model_validate(out["trace"][-1])
    assert trace.steps[0].tokens_in is None
    assert trace.tokens_in is None and trace.cost_usd is None


def test_trace_includes_compaction_model_call(monkeypatch):
    from analyst.graph import build_graph

    monkeypatch.setenv("ANALYST_COMPACTION_TRIGGER_TOKENS", "1500")
    monkeypatch.setenv("ANALYST_COMPACTION_KEEP_MESSAGES", "1")
    get_settings.cache_clear()
    history = [
        HumanMessage("how many events last time?"),
        AIMessage("", tool_calls=[call("query_metrics", COUNT, "c0")]),
        ToolMessage("card " + "x" * 8000, tool_call_id="c0"),
        AIMessage("There were some events.\n\nVerified"),
        HumanMessage("how many events now?"),
    ]
    digest = AIMessage("", tool_calls=[call("Digest", {"goal": "events count"}, "d0")])
    model = ScriptedModel(
        script=[
            _with_usage(digest, 5_000, 50),
            _with_usage(no_plan(), 100, 10),
            _with_usage(tool_turn(call("query_metrics", COUNT, "c1")), 200, 20),
            _with_usage(cite_total, 300, 30),
        ]
    )
    out = _run(build_graph(model=model), {"messages": history})
    trace = RunTrace.model_validate(out["trace"][-1])
    assert {"type": "compaction"} in trace.events
    assert [s.name for s in trace.steps[:2]] == ["compaction", "plan"]
    assert trace.steps[0].tokens_in == 5_000
    assert trace.tokens_in == 5_600 and trace.tokens_out == 110


def test_trace_run_id_from_top_level_config():
    """The Agent Server passes the run id as top-level ``config["run_id"]``."""
    import uuid

    from analyst.graph import build_graph

    run_id = uuid.uuid4()
    model = ScriptedModel(
        script=[no_plan(), tool_turn(call("query_metrics", COUNT, "c1")), cite_total]
    )
    graph = build_graph(model=model)
    out = asyncio.run(
        graph.ainvoke(
            {"messages": [HumanMessage("how many events?")]},
            config={"run_id": run_id, "configurable": {"thread_id": "t-run-id"}},
            context=CTX,
        )
    )
    assert RunTrace.model_validate(out["trace"][-1]).run_id == str(run_id)


def test_worker_trace_run_id_from_top_level_config():
    import uuid

    from analyst.worker import build_worker

    run_id = uuid.uuid4()
    worker = build_worker(model=StreamingModel(replies=[AIMessage("report done")]))
    out = asyncio.run(
        worker.ainvoke(
            {"messages": [HumanMessage("task")]},
            config={"run_id": run_id, "configurable": {"thread_id": "t-worker"}},
        )
    )
    assert RunTrace.model_validate(out["trace"][-1]).run_id == str(run_id)


def test_live_step_cost_matches_saved_trace_from_catalogue():
    """Tokens without a provider cost: the live step is priced like build_trace."""
    from analyst.providers.catalog import MODEL_SPEC_KEY, load_catalog

    load_catalog()  # get_model warms it off the loop before any model runs
    model = StreamingModel(replies=_replies(), streaming=True, metadata={MODEL_SPEC_KEY: LUNA})
    events, state = _collect(model)
    live = [
        e["step"]["cost_usd"]
        for e in events
        if e.get("type") == "trace" and e["event"] == "step_finished" and e["step"]["kind"] == "model"
    ]
    saved = build_trace(
        [m for m in state["messages"] if not isinstance(m, HumanMessage)],
        model=LUNA, run_id=None, verification=None, events=[], compaction_trigger=1000,
    )
    saved_costs = [s.cost_usd for s in saved.steps if s.kind == "model"]
    assert live == pytest.approx(saved_costs)
    assert live == pytest.approx([(100 * 0.10 + 10 * 0.50) / 1e6] * 2)


def test_timed_model_call_prices_from_the_model_spec(monkeypatch):
    from analyst.providers.catalog import MODEL_SPEC_KEY, load_catalog
    from analyst.trace import middleware

    load_catalog()
    seen: list = []
    monkeypatch.setattr(middleware, "emit", lambda event, step: seen.append((event, step)))
    model = StreamingModel(replies=[], metadata={MODEL_SPEC_KEY: LUNA})

    async def call():
        return AIMessage("plan", usage_metadata=_usage(1_000, 100))

    asyncio.run(middleware.atimed_model_call("plan", call, model=model))
    finished = [s for e, s in seen if e == "step_finished"]
    assert finished[0].cost_usd == pytest.approx((1_000 * 0.10 + 100 * 0.50) / 1e6)

    seen.clear()

    async def reported():
        return AIMessage("plan", usage_metadata=_usage(1_000, 100, cost=0.5))

    asyncio.run(middleware.atimed_model_call("plan", reported, model=model))
    assert next(s for e, s in seen if e == "step_finished").cost_usd == 0.5
