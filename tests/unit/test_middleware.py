"""Unit tests for the agent middleware (clearing, metering, cost, limits)."""

import asyncio
from types import SimpleNamespace

import pytest
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from analyst.context import AnalystContext
from analyst.middleware import (
    ClearOldToolResults,
    ContextMeter,
    ContextSize,
    OpenRouterCost,
    analyst_middleware,
    measure,
)


def _request(messages, runtime=None):
    return ModelRequest(
        model=None,
        messages=list(messages),
        runtime=runtime if runtime is not None else SimpleNamespace(context=None),
    )


def _tool_message(index, handle=None):
    artifact = None
    if handle is not None:
        artifact = {"handle_id": handle, "produced_by": "run_sql", "governed": False}
    return ToolMessage(
        content=f"result {index} " + "x" * 10,
        tool_call_id=f"call-{index}",
        artifact=artifact,
    )


def _run_sync(middleware, request):
    seen = {}

    def handler(req):
        seen["request"] = req
        return ModelResponse(result=[AIMessage(content="done")])

    result = middleware.wrap_model_call(request, handler)
    return result, seen["request"]


def test_clearing_keeps_last_n_and_stubs_with_handle():
    messages = [HumanMessage(content="question")]
    # an artifact-less tool message first, to exercise the fallback stub
    messages.append(ToolMessage(content="plain output", tool_call_id="call-bare"))
    messages.extend(_tool_message(i, handle=f"r{i:03d}") for i in range(7))

    pipeline = ClearOldToolResults(keep_last=6)
    original_contents = [m.content for m in messages]

    _result, sent = _run_sync(pipeline, _request(messages))

    # State (the caller's list) is untouched.
    assert [m.content for m in messages] == original_contents
    assert len(sent.messages) == len(messages)

    sent_tools = [m for m in sent.messages if isinstance(m, ToolMessage)]
    assert len(sent_tools) == 8

    stubbed, kept = sent_tools[:2], sent_tools[2:]
    assert stubbed[0].content == "[cleared] tool output"
    assert stubbed[1].content == (
        "[cleared] r000 (run_sql); call describe('r000') to see it again"
    )
    assert all(m.content.startswith("result ") for m in kept)
    # Stubbing preserves the call id and the artifact.
    assert stubbed[1].tool_call_id == "call-0"
    assert stubbed[1].artifact["handle_id"] == "r000"


def test_clearing_keeps_everything_when_short():
    messages = [_tool_message(i, handle=f"r{i}") for i in range(3)]
    _result, sent = _run_sync(ClearOldToolResults(keep_last=6), _request(messages))
    assert [m.content for m in sent.messages] == [m.content for m in messages]


def test_clearing_async_matches_sync():
    messages = [_tool_message(i, handle=f"r{i}") for i in range(8)]
    pipeline = ClearOldToolResults(keep_last=3)

    async def handler(req):
        return ModelResponse(result=[AIMessage(content="done")])

    async def run_async():
        seen = {}

        async def capture(req):
            seen["request"] = req
            return await handler(req)

        await pipeline.awrap_model_call(_request(messages), capture)
        return seen["request"]

    sent = asyncio.run(run_async())
    assert len(sent.messages) == 8
    assert [m.content for m in sent.messages if isinstance(m, ToolMessage)][:2] == [
        "[cleared] r0 (run_sql); call describe('r0') to see it again",
        "[cleared] r1 (run_sql); call describe('r1') to see it again",
    ]


def test_measure_counts_tool_bytes():
    messages = [
        ToolMessage(content="abc", tool_call_id="a"),
        ToolMessage(content="héllo", tool_call_id="b"),
        AIMessage(content="hi"),
    ]
    size = measure(messages)
    assert isinstance(size, ContextSize)
    assert size.tool_bytes == 3 + 6
    assert size.total_bytes == 3 + 6 + 2
    assert size.messages == 3


def test_meter_noop_without_tracing(monkeypatch):
    monkeypatch.setattr("analyst.middleware.metering.get_current_run_tree", lambda: None)
    called = {}

    def handler(req):
        called["ran"] = True
        return ModelResponse(result=[AIMessage(content="done")])

    result = ContextMeter().wrap_model_call(_request([HumanMessage(content="hi")]), handler)
    assert called["ran"] is True
    assert result.result[0].content == "done"


def test_meter_writes_sizes_and_question_id_to_run_metadata(monkeypatch):
    written = {}

    class FakeTree:
        def add_metadata(self, metadata):
            written.update(metadata)

    monkeypatch.setattr(
        "analyst.middleware.metering.get_current_run_tree", lambda: FakeTree()
    )
    runtime = SimpleNamespace(
        context=AnalystContext(dataset="ds", question_id="question-7")
    )
    messages = [
        HumanMessage(content="hé"),
        ToolMessage(content="abc", tool_call_id="a"),
    ]

    def handler(req):
        return ModelResponse(result=[AIMessage(content="done")])

    ContextMeter().wrap_model_call(_request(messages, runtime=runtime), handler)

    assert written["context_total_bytes"] == 3 + 3
    assert written["context_tool_bytes"] == 3
    assert written["context_messages"] == 2
    assert written["question_id"] == "question-7"


def test_meter_tolerates_missing_context(monkeypatch):
    written = {}

    class FakeTree:
        def add_metadata(self, metadata):
            written.update(metadata)

    monkeypatch.setattr(
        "analyst.middleware.metering.get_current_run_tree", lambda: FakeTree()
    )

    def handler(req):
        return ModelResponse(result=[AIMessage(content="done")])

    ContextMeter().wrap_model_call(_request([HumanMessage(content="hi")]), handler)
    assert written["question_id"] is None


def test_openrouter_cost_copied():
    ai = AIMessage(
        content="answer",
        response_metadata={"usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.0012}},
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )

    def handler(req):
        return ModelResponse(result=[ai])

    result = OpenRouterCost().wrap_model_call(_request([HumanMessage(content="hi")]), handler)
    sent = result.result[0]
    assert sent.usage_metadata["total_cost"] == 0.0012
    assert sent.usage_metadata["total_tokens"] == 15
    # the incoming message was not mutated
    assert "total_cost" not in (ai.usage_metadata or {})


def test_openrouter_cost_without_usage_is_noop():
    ai = AIMessage(content="answer", usage_metadata=None)

    def handler(req):
        return ModelResponse(result=[ai])

    result = OpenRouterCost().wrap_model_call(_request([HumanMessage(content="hi")]), handler)
    assert result.result[0].usage_metadata is None


def test_openrouter_cost_async_matches_sync():
    ai = AIMessage(
        content="answer",
        response_metadata={
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "cost": 0.5}
        },
    )

    async def handler(req):
        return ModelResponse(result=[ai])

    result = asyncio.run(
        OpenRouterCost().awrap_model_call(_request([HumanMessage(content="hi")]), handler)
    )
    assert result.result[0].usage_metadata["total_cost"] == 0.5


def test_middleware_order():
    settings = SimpleNamespace()
    names = [type(m).__name__ for m in analyst_middleware(settings)]
    assert names == [
        "CompactionMiddleware",
        "ClearOldToolResults",
        "ContextMeter",
        "OpenRouterCost",
        "ToolCallLimitMiddleware",
        "ModelCallLimitMiddleware",
    ]


@pytest.mark.parametrize("index,limit", [(4, 40), (5, 25)])
def test_middleware_limits(index, limit):
    middleware = analyst_middleware(SimpleNamespace())[index]
    assert middleware.run_limit == limit


def test_middleware_composes_with_create_agent():
    from langchain.agents import create_agent
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

    model = GenericFakeChatModel(messages=iter([AIMessage(content="hello there")]))
    agent = create_agent(model=model, middleware=analyst_middleware(None))
    out = asyncio.run(agent.ainvoke({"messages": [("human", "hi")]}))
    assert out["messages"][-1].content == "hello there"
