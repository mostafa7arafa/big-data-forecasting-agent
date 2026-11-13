"""The analyst graph (plan -> analyst -> verify -> answer) on the synthetic dataset.

A scripted chat model stands in for the LLM: each call pops the next step of
its script, which is either a fixed AIMessage or a function of the messages it
was sent (so an answer can cite the handle a tool call just produced).
"""

import asyncio
import importlib
import json
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from analyst.config import get_settings
from analyst.context import AnalystContext
from analyst.data import registry
from analyst.data.service import DataService
from analyst.graph import build_graph
from analyst.memory.knowledge import TOPICS
from analyst.prompts import build_system_prompt
from analyst.state import Answer, Claim

REPO = Path(__file__).resolve().parents[2]
CTX = AnalystContext(dataset="events", background=False)

Step = AIMessage | Callable[[list[BaseMessage]], AIMessage]


class ScriptedModel(BaseChatModel):
    """Replays a script of AI turns; bind_tools is a no-op."""

    script: list[Any]
    seen: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        if not self.script:
            raise AssertionError("scripted model ran out of steps")
        step = self.script.pop(0)
        message = step(messages) if callable(step) else step
        return ChatResult(generations=[ChatGeneration(message=message)])


def call(name: str, args: dict, call_id: str) -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def no_plan() -> AIMessage:
    return AIMessage("", tool_calls=[call("Plan", {"needs_plan": False}, "p0")])


def tool_turn(*calls: dict, text: str = "Pulling the totals.") -> AIMessage:
    return AIMessage(text, tool_calls=list(calls))


def handles_in(messages: list[BaseMessage]) -> list[str]:
    return [
        m.artifact["handle_id"]
        for m in messages
        if isinstance(m, ToolMessage)
        and isinstance(m.artifact, dict)
        and "handle_id" in m.artifact
    ]


def total(handle_id: str) -> float:
    """The single value of a one-row result, from whichever service made it."""
    store = next(
        s.store for s in registry._services.values() if s.store.exists(handle_id)
    )
    table = store.load(handle_id)
    return float(table.column(table.num_columns - 1)[0].as_py())


def answer_turn(narrative: str, claims: list[dict], call_id: str = "a1") -> AIMessage:
    args = {"narrative": narrative, "claims": claims}
    return AIMessage("", tool_calls=[call("Answer", args, call_id)])


def cite_total(messages: list[BaseMessage]) -> AIMessage:
    hid = handles_in(messages)[-1]
    n = int(total(hid))
    text = f"There were {n:,} events."
    return answer_turn(
        text,
        [{"text": text, "type": "descriptive", "values": [n], "handles": [hid]}],
    )


def untraced(call_id: str) -> Callable[[list[BaseMessage]], AIMessage]:
    def step(messages: list[BaseMessage]) -> AIMessage:
        hid = handles_in(messages)[-1]
        text = "There were 987654 events."
        return answer_turn(
            text,
            [
                {
                    "text": text,
                    "type": "descriptive",
                    "values": [987654],
                    "handles": [hid],
                }
            ],
            call_id,
        )

    return step


COUNT = {"metrics": ["event_count"]}


@pytest.fixture(autouse=True)
def events_service(synthetic_dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "load_dataset", lambda name: synthetic_dataset)
    original = DataService.__init__

    def init(self, dataset, *, cutoff=None, store_root=None):
        original(self, dataset, cutoff=cutoff, store_root=tmp_path / f"s-{cutoff}")

    monkeypatch.setattr(DataService, "__init__", init)
    monkeypatch.setenv("ANALYST_DATASET", "events")
    get_settings.cache_clear()
    registry.clear_services()
    yield
    registry.clear_services()
    get_settings.cache_clear()


def run(model: ScriptedModel, messages, context=CTX) -> dict:
    graph = build_graph(model=model)
    return asyncio.run(graph.ainvoke({"messages": messages}, context=context))


def test_happy_path_verified():
    model = ScriptedModel(
        script=[no_plan(), tool_turn(call("query_metrics", COUNT, "c1")), cite_total]
    )
    out = run(model, [HumanMessage("how many events?")])
    assert out["answer"].verification.label == "verified", out["answer"].verification
    assert out["retries"] == 0
    final = out["messages"][-1]
    assert isinstance(final, AIMessage)
    assert final.content.startswith(out["answer"].narrative + "\n\n")
    assert "Verified" in final.content


def test_context_from_settings_when_missing():
    model = ScriptedModel(
        script=[no_plan(), tool_turn(call("query_metrics", COUNT, "c1")), cite_total]
    )
    out = run(model, [HumanMessage("how many events?")], context=None)
    assert out["answer"].verification.label == "verified"


def test_verify_retry_then_not_verified():
    model = ScriptedModel(
        script=[
            no_plan(),
            tool_turn(call("query_metrics", COUNT, "c1")),
            untraced("a1"),
            untraced("a2"),
        ]
    )
    out = run(model, [HumanMessage("how many events?")])
    assert out["retries"] == 1
    verification = out["answer"].verification
    assert verification.label == "not_verified"
    assert "987654" in verification.reason
    feedback = [
        m
        for m in out["messages"]
        if isinstance(m, HumanMessage) and m.content.startswith("Verification failed")
    ]
    assert len(feedback) == 1
    assert "cite handles for every number" in feedback[0].content
    assert not model.script


def test_retry_recovers_to_verified():
    model = ScriptedModel(
        script=[
            no_plan(),
            tool_turn(call("query_metrics", COUNT, "c1")),
            untraced("a1"),
            cite_total,
        ]
    )
    out = run(model, [HumanMessage("how many events?")])
    assert out["retries"] == 1
    assert out["answer"].verification.label == "verified"


def test_malformed_structured_output_not_verified():
    model = ScriptedModel(
        script=[
            no_plan(),
            tool_turn(call("query_metrics", COUNT, "c1")),
            AIMessage("Lots of events happened."),
        ]
    )
    out = run(model, [HumanMessage("how many events?")])
    answer = out["answer"]
    assert answer.verification.label == "not_verified"
    assert answer.verification.reason == "no structured answer"
    assert answer.narrative == "Lots of events happened."
    assert answer.claims == []


def test_parallel_tool_calls_all_execute():
    model = ScriptedModel(
        script=[
            no_plan(),
            tool_turn(
                call("query_metrics", {"metrics": ["event_count"]}, "c1"),
                call("query_metrics", {"metrics": ["total_amount"]}, "c2"),
                call(
                    "query_metrics",
                    {"metrics": ["event_count"], "grain": "month"},
                    "c3",
                ),
            ),
            cite_total,
        ]
    )
    out = run(model, [HumanMessage("three things")])
    tool_messages = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    by_call = {m.tool_call_id: m for m in tool_messages}
    assert {"c1", "c2", "c3"} <= set(by_call)
    handles = [by_call[c].artifact["handle_id"] for c in ("c1", "c2", "c3")]
    assert len(set(handles)) == 3


def test_tool_argument_validation_error_reaches_model():
    model = ScriptedModel(
        script=[
            no_plan(),
            tool_turn(call("query_metrics", {"metrics": []}, "bad")),
            tool_turn(call("query_metrics", COUNT, "c1")),
            cite_total,
        ]
    )
    out = run(model, [HumanMessage("how many events?")])
    bad = next(
        m
        for m in out["messages"]
        if isinstance(m, ToolMessage) and m.tool_call_id == "bad"
    )
    assert bad.status == "error"
    assert "metrics" in bad.content
    assert out["answer"].verification.label == "verified"


def test_plan_emits_steps_on_custom_stream():
    plan = AIMessage(
        "",
        tool_calls=[
            call("Plan", {"needs_plan": True, "steps": ["total", "by month"]}, "p0")
        ],
    )
    model = ScriptedModel(
        script=[plan, tool_turn(call("query_metrics", COUNT, "c1")), cite_total]
    )
    graph = build_graph(model=model)

    async def go():
        custom, last = [], None
        # Studio streams subgraphs, so tool events from the inner agent show up.
        async for namespace, mode, chunk in graph.astream(
            {"messages": [HumanMessage("multi-step question")]},
            context=CTX,
            stream_mode=["custom", "values"],
            subgraphs=True,
        ):
            if mode == "custom":
                custom.append(chunk)
            elif namespace == ():
                last = chunk
        return custom, last

    custom, out = asyncio.run(go())
    assert {"type": "plan", "steps": ["total", "by month"]} in custom
    assert any(e.get("type") == "progress" for e in custom)  # tool events reach Studio
    assert out["plan"] == ["total", "by month"]
    assert out["answer"].verification.label == "verified"


def test_plan_skipped_for_background_notification():
    note = HumanMessage(
        "Task t1 finished.\nhandles: r_0000000000",
        additional_kwargs={"source": "async_subagent"},
    )
    model = ScriptedModel(
        script=[tool_turn(call("query_metrics", COUNT, "c1")), cite_total]
    )
    out = run(model, [note])
    assert out["plan"] == []
    assert out["answer"].verification.label == "verified"
    assert not model.script


def test_analyst_prompt_uses_run_context():
    model = ScriptedModel(
        script=[no_plan(), tool_turn(call("query_metrics", COUNT, "c1")), cite_total]
    )
    ctx = AnalystContext(dataset="events", cutoff=date(2024, 2, 15), background=False)
    run(model, [HumanMessage("how many events?")], context=ctx)
    system = model.seen[1][0]
    assert system.type == "system"
    assert "events" in system.content and "2024-02-15" in system.content


def test_prompt_has_status_rule_and_no_dataset_literal():
    memory = registry.get_service("events", None).memory
    p = build_system_prompt(memory, dataset="ds", cutoff=None, background=True)
    assert "one short plain sentence" in p and "ds" in p
    assert (
        "Before each batch of tool calls, write one short plain sentence saying what "
        "you are about to do. No numbers from results in it."
    ) in p
    for topic in TOPICS:
        assert topic in p
    assert "Synthetic hourly events" not in p  # topic names only, not content
    assert "query_metrics" in p and "run_sql" in p
    assert "parallel" in p
    assert "background" in p.lower()
    assert "cutoff" not in p.lower()


def test_prompt_cutoff_and_background_toggle():
    memory = registry.get_service("events", None).memory
    on = build_system_prompt(
        memory, dataset="ds", cutoff=date(2024, 2, 1), background=True
    )
    off = build_system_prompt(
        memory, dataset="ds", cutoff=date(2024, 2, 1), background=False
    )
    assert "2024-02-01" in on and "2024-02-01" in off
    assert len(off) < len(on)


def test_langgraph_json_points_at_graph(monkeypatch, tmp_path):
    config = json.loads((REPO / "langgraph.json").read_text())
    assert config == {
        "dependencies": ["."],
        "graphs": {
            "analyst": "./src/analyst/graph.py:graph",
            "worker": "./src/analyst/worker.py:graph",
        },
        "env": ".env",
    }
    for var in ("ANALYST_DATASET", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)  # no .env: the graphs import without keys
    get_settings.cache_clear()
    assert get_settings().openai_api_key is None
    for name, spec in config["graphs"].items():
        path, _, attr = spec.partition(":")
        assert (REPO / path).is_file()
        module_name = "analyst." + Path(path).stem
        module = importlib.reload(importlib.import_module(module_name))
        assert hasattr(getattr(module, attr), "ainvoke"), name


# --- fix round 1 -----------------------------------------------------------


@pytest.mark.parametrize("key", ["lc_source", "source"])
def test_background_note_numbers_are_not_exempt(key):
    note = HumanMessage(
        "Task t1 finished: 4242 events.", additional_kwargs={key: "async_subagent"}
    )
    untraced_answer = answer_turn("The worker found 4242 events.", [])
    model = ScriptedModel(script=[untraced_answer, untraced_answer.model_copy()])
    out = run(model, [HumanMessage("how many events?"), note])
    assert out["plan"] == []
    verification = out["answer"].verification
    assert verification.label == "not_verified"
    assert "4242" in verification.reason


def test_no_blocking_calls_on_event_loop(monkeypatch):
    import threading

    from blockbuster import blockbuster_ctx

    loop_threads: set[int] = set()
    tool_threads: set[int] = set()
    original = DataService.query_metrics

    def spy(self, q):
        tool_threads.add(threading.get_ident())
        return original(self, q)

    monkeypatch.setattr(DataService, "query_metrics", spy)
    model = ScriptedModel(
        script=[no_plan(), tool_turn(call("query_metrics", COUNT, "c1")), cite_total]
    )
    graph = build_graph(model=model)

    async def go():
        loop_threads.add(threading.get_ident())
        with blockbuster_ctx():
            return await graph.ainvoke(
                {"messages": [HumanMessage("how many events?")]}, context=None
            )

    out = asyncio.run(go())
    assert out["answer"].verification.label == "verified"
    assert tool_threads and not tool_threads & loop_threads  # tools ran off-loop


def test_plan_malformed_result_is_empty_plan():
    from langchain_core.runnables import RunnableLambda

    class BadPlanner(ScriptedModel):
        def with_structured_output(self, schema, **kwargs):
            return RunnableLambda(lambda _: {"needs_plan": "maybe", "steps": 3})

    model = BadPlanner(
        script=[tool_turn(call("query_metrics", COUNT, "c1")), cite_total]
    )
    out = run(model, [HumanMessage("how many events?")])
    assert out["plan"] == []
    assert out["answer"].verification.label == "verified"


def test_background_policy_on_with_async_tools():
    from analyst.graph import ASYNC_TASK_TOOLS

    assert "start_async_task" in ASYNC_TASK_TOOLS
    model = ScriptedModel(
        script=[no_plan(), tool_turn(call("query_metrics", COUNT, "c1")), cite_total]
    )
    ctx = AnalystContext(dataset="events", background=True)
    run(model, [HumanMessage("how many events?")], context=ctx)
    system = model.seen[1][0].content
    assert "## Background work" in system
    assert 'start_async_task(description, subagent_type="worker")' in system


def test_plan_skipped_for_real_notification_shape():
    from langchain_core.messages.notification import add_notification_metadata

    note = add_notification_metadata(
        HumanMessage(content=[{"type": "text", "text": "Task t1 finished."}]),
        source="async_subagent",
        data={"task_id": "t1", "agent_name": "worker", "status": "success"},
    )
    model = ScriptedModel(
        script=[tool_turn(call("query_metrics", COUNT, "c1")), cite_total]
    )
    out = run(model, [HumanMessage("how many events?"), note])
    assert out["plan"] == []
    assert out["answer"].verification.label == "verified"
    assert not model.script


def test_async_tasks_survive_the_analyst_node():
    task = {
        "task_id": "t1",
        "agent_name": "worker",
        "thread_id": "t1",
        "run_id": "r1",
        "status": "success",
        "created_at": "2026-01-01T00:00:00Z",
        "last_checked_at": "2026-01-01T00:00:00Z",
        "last_updated_at": "2026-01-01T00:00:00Z",
    }
    model = ScriptedModel(
        script=[no_plan(), tool_turn(call("query_metrics", COUNT, "c1")), cite_total]
    )
    graph = build_graph(model=model)
    out = asyncio.run(
        graph.ainvoke(
            {
                "messages": [HumanMessage("how many events?")],
                "async_tasks": {"t1": task},
            },
            context=CTX,
        )
    )
    assert out["async_tasks"]["t1"]["status"] == "success"


def test_prompt_quirk_ids_in_fact_ids():
    memory = registry.get_service("events", None).memory
    p = build_system_prompt(memory, dataset="ds", cutoff=None, background=False)
    assert 'read_knowledge("quirks")' in p
    assert "gap:2023-09-21" in p and "fact_ids" in p
    assert "context fact id" in p  # causal claims still need a context fact


# --- fix round 2 -----------------------------------------------------------


def provider_model(script: list[Any]) -> ScriptedModel:
    """A model whose profile makes create_agent use the provider strategy:
    the final answer arrives as JSON text, as it does from OpenAI."""
    return ScriptedModel(script=script, profile={"structured_output": True})


def json_turn(payload: dict) -> AIMessage:
    return AIMessage(json.dumps(payload))


def cite_total_json_with_nulls(messages: list[BaseMessage]) -> AIMessage:
    hid = handles_in(messages)[-1]
    n = int(total(hid))
    text = f"There were {n:,} events."
    return json_turn(
        {
            "narrative": text,
            "claims": [
                {
                    "text": text,
                    "type": "descriptive",
                    "values": [n],
                    "handles": [hid],
                    "fact_ids": None,
                }
            ],
            "charts": [
                {
                    "kind": "bar",
                    "title": "Events",
                    "handle": hid,
                    "x": "event_count",
                    "y": "event_count",
                    "bands": None,
                }
            ],
            "sql": None,
            "background_tasks": None,
        }
    )


def test_answer_schema_coerces_null_lists():
    answer = Answer.model_validate(
        {
            "narrative": "n",
            "claims": None,
            "charts": [
                {
                    "kind": "line",
                    "title": "t",
                    "handle": "r_1",
                    "x": "a",
                    "y": "b",
                    "bands": None,
                }
            ],
            "sql": None,
            "background_tasks": None,
        }
    )
    assert answer.claims == [] and answer.sql == [] and answer.background_tasks == []
    assert answer.charts[0].bands == []
    claim = Claim.model_validate(
        {
            "text": "t",
            "type": "descriptive",
            "values": None,
            "handles": None,
            "fact_ids": None,
        }
    )
    assert claim.values == [] and claim.handles == [] and claim.fact_ids == []


def test_answer_json_schema_arrays_have_items():
    from langchain_core.utils.function_calling import convert_to_openai_tool

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "array":
                assert "items" in node and "prefixItems" not in node, node
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(Answer.model_json_schema())
    walk(convert_to_openai_tool(Answer))


def test_provider_null_list_fields_do_not_raise():
    model = provider_model(
        [
            no_plan(),
            tool_turn(call("query_metrics", COUNT, "c1")),
            cite_total_json_with_nulls,
        ]
    )
    out = run(model, [HumanMessage("how many events?")])
    answer = out["answer"]
    assert answer.verification.label == "verified", answer.verification
    assert answer.charts[0].bands == []


def test_unrepairable_structured_output_is_not_verified():
    model = provider_model(
        [
            no_plan(),
            tool_turn(call("query_metrics", COUNT, "c1")),
            json_turn({"narrative": {"oops": 1}, "claims": "nope"}),
        ]
    )
    out = run(model, [HumanMessage("how many events?")])
    verification = out["answer"].verification
    assert verification.label == "not_verified"
    assert verification.reason == "no structured answer"
    # the work done before the bad answer stays in the thread
    assert any(
        isinstance(m, ToolMessage) and m.tool_call_id == "c1" for m in out["messages"]
    )


def _is_answer_json(text: str) -> bool:
    try:
        Answer.model_validate_json(text)
    except ValueError:
        return False
    return True


@pytest.mark.parametrize("strategy", ["tool", "provider"])
def test_raw_structured_response_not_persisted(strategy):
    if strategy == "tool":
        model = ScriptedModel(
            script=[
                no_plan(),
                tool_turn(call("query_metrics", COUNT, "c1")),
                cite_total,
            ]
        )
    else:
        model = provider_model(
            [
                no_plan(),
                tool_turn(call("query_metrics", COUNT, "c1")),
                cite_total_json_with_nulls,
            ]
        )
    out = run(model, [HumanMessage("how many events?")])
    assert out["answer"].verification.label == "verified"
    messages = out["messages"]
    ai = [m for m in messages if isinstance(m, AIMessage)]
    assert not any(_is_answer_json(str(m.content)) for m in ai)
    assert not any(tc["name"] == "Answer" for m in ai for tc in m.tool_calls)
    answer_calls = {"a1"}
    assert not any(
        isinstance(m, ToolMessage) and m.tool_call_id in answer_calls for m in messages
    )
    # status line + tool call and its result are kept; the chip message is last
    assert any(m.tool_calls and m.content == "Pulling the totals." for m in ai)
    assert any(isinstance(m, ToolMessage) and m.tool_call_id == "c1" for m in messages)
    assert messages[-1].content.startswith(out["answer"].narrative + "\n\n")


def test_task_cap_sees_thread_tasks(monkeypatch):
    """The analyst's inner agent gets the thread's task records, so the cap holds."""
    from analyst.background import AsyncSubAgentMiddleware

    async def no_pull(self, tasks):  # no Agent Server in unit tests
        return {}

    monkeypatch.setattr(AsyncSubAgentMiddleware, "_apull", no_pull)
    tasks = {
        f"t{i}": {
            "task_id": f"t{i}",
            "agent_name": "worker",
            "thread_id": f"t{i}",
            "run_id": f"r{i}",
            "status": "running",
            "created_at": "2026-01-01T00:00:00Z",
            "last_checked_at": "2026-01-01T00:00:00Z",
            "last_updated_at": "2026-01-01T00:00:00Z",
        }
        for i in range(3)
    }
    start = call(
        "start_async_task",
        {"description": "forecast daily", "subagent_type": "worker"},
        "s1",
    )
    model = ScriptedModel(
        script=[
            no_plan(),
            tool_turn(start),
            tool_turn(call("query_metrics", COUNT, "c1")),
            cite_total,
        ]
    )
    graph = build_graph(model=model)
    out = asyncio.run(
        graph.ainvoke(
            {"messages": [HumanMessage("forecast daily events")], "async_tasks": tasks},
            context=AnalystContext(dataset="events", background=True),
        )
    )
    refused = next(
        m
        for m in out["messages"]
        if isinstance(m, ToolMessage) and m.tool_call_id == "s1"
    )
    assert refused.status == "error" and "limit 3" in refused.content
    assert set(out["async_tasks"]) == set(tasks)


def test_wake_up_run_narrowed_to_task_cutoff():
    """A notification-started run with no cutoff runs under the task's cutoff."""
    from langchain_core.messages.notification import add_notification_metadata

    cutoff = date(2024, 1, 31)
    note = add_notification_metadata(
        HumanMessage(content=[{"type": "text", "text": "Task t1 finished."}]),
        source="async_subagent",
        data={"task_id": "t1", "agent_name": "worker", "status": "success"},
    )
    model = ScriptedModel(
        script=[tool_turn(call("query_metrics", COUNT, "c1")), cite_total]
    )
    graph = build_graph(model=model)
    out = asyncio.run(
        graph.ainvoke(
            {
                "messages": [HumanMessage("how many events?"), note],
                "background_contexts": {
                    "t1": {"dataset": "events", "cutoff": cutoff.isoformat()}
                },
            },
            context=AnalystContext(dataset="events", background=False),
        )
    )
    assert "2024-01-31" in model.seen[0][0].content  # the system prompt
    handle = handles_in(out["messages"])[0]
    assert registry.get_service("events", cutoff).store.exists(handle)
    assert not registry.get_service("events", None).store.exists(handle)
    assert out["answer"].verification.label == "verified"
    assert out["wake_context"] == {"dataset": "events", "cutoff": "2024-01-31"}


def test_wake_context_cleared_on_user_turn():
    model = ScriptedModel(
        script=[no_plan(), tool_turn(call("query_metrics", COUNT, "c1")), cite_total]
    )
    graph = build_graph(model=model)
    out = asyncio.run(
        graph.ainvoke(
            {
                "messages": [HumanMessage("how many events?")],
                "wake_context": {"dataset": "events", "cutoff": "2024-01-31"},
            },
            context=CTX,
        )
    )
    assert out["wake_context"] is None
    assert "2024-01-31" not in model.seen[1][0].content


def test_start_async_task_through_the_graph(monkeypatch):
    """A real start call: header injected, context forwarded, task recorded."""
    from deepagents.middleware.async_subagents import _ClientCache

    from analyst.background import AsyncSubAgentMiddleware, parse_context_header

    launched: list[dict] = []

    class FakeClient:
        def __init__(self):
            self.threads, self.runs = self, self

        async def create(self, **kwargs):
            if "assistant_id" in kwargs:  # runs.create
                launched.append(kwargs)
                return {"run_id": "run-1"}
            return {"thread_id": "task-1"}  # threads.create

    async def no_pull(self, tasks):
        return {}

    fake = FakeClient()
    monkeypatch.setattr(_ClientCache, "get_async", lambda self, name: fake)
    monkeypatch.setattr(AsyncSubAgentMiddleware, "_apull", no_pull)
    cutoff = date(2024, 1, 31)
    start = call(
        "start_async_task",
        {"description": "forecast daily events", "subagent_type": "worker"},
        "s1",
    )
    model = ScriptedModel(
        script=[
            no_plan(),
            tool_turn(start),
            tool_turn(call("query_metrics", COUNT, "c1")),
            cite_total,
        ]
    )
    graph = build_graph(model=model)
    out = asyncio.run(
        graph.ainvoke(
            {"messages": [HumanMessage("forecast daily events")]},
            context=AnalystContext(dataset="events", cutoff=cutoff, background=True),
        )
    )
    assert len(launched) == 1
    run = launched[0]
    assert run["assistant_id"] == "worker" and run["thread_id"] == "task-1"
    description = run["input"]["messages"][0]["content"]
    assert parse_context_header(description).cutoff == cutoff
    assert description.endswith("forecast daily events")
    assert run["context"]["dataset"] == "events"
    assert run["context"]["cutoff"] == cutoff
    assert out["async_tasks"]["task-1"]["status"] == "running"
    assert out["background_contexts"]["task-1"] == {
        "dataset": "events",
        "cutoff": "2024-01-31",
    }
