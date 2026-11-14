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

import pyarrow as pa
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
    assert verification.reason == "no evidence cited"
    assert any("4242" in p for p in verification.problems)


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
    assert "kind:YYYY-MM-DD" in p and "fact_ids" in p
    assert "gap:2023-09-21" not in p
    assert "context fact id" in p
    assert "still needs a context fact id" not in p
    assert "context fact or quirk id" in p


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


# --- final fix wave ------------------------------------------------------------


def _paired(messages: list[BaseMessage]) -> None:
    """Every tool call has its ToolMessage and every ToolMessage its call."""
    calls = {tc["id"] for m in messages if isinstance(m, AIMessage) for tc in m.tool_calls}
    results = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    assert calls == results, (calls, results)


def test_answer_beside_data_call_keeps_the_data_call():
    both = AIMessage(
        "Pulling the totals.",
        tool_calls=[
            call("query_metrics", COUNT, "c1"),
            # a number with no claim fails verify, so the retry runs
            call("Answer", {"narrative": "There were 5 events.", "claims": []}, "a1"),
        ],
    )
    model = ScriptedModel(script=[no_plan(), both, cite_total])
    out = run(model, [HumanMessage("how many events?")])
    messages = out["messages"]
    _paired(messages)
    ai = [m for m in messages if isinstance(m, AIMessage)]
    assert not any(tc["name"] == "Answer" for m in ai for tc in m.tool_calls)
    assert any([tc["id"] for tc in m.tool_calls] == ["c1"] for m in ai)
    assert not any(isinstance(m, ToolMessage) and m.tool_call_id == "a1" for m in messages)
    assert out["answer"].verification.label == "verified"


def test_drop_structured_response_strips_only_the_answer_call():
    from analyst.graph import _drop_structured_response

    both = AIMessage(
        "status",
        tool_calls=[
            call("query_metrics", COUNT, "c1"),
            call("Answer", {"narrative": "x", "claims": []}, "a1"),
        ],
    )
    answer_only = AIMessage("", tool_calls=[call("Answer", {"narrative": "y", "claims": []}, "a2")])
    new = [
        both,
        ToolMessage("Returning structured response", tool_call_id="a1"),
        ToolMessage("card", tool_call_id="c1"),
        answer_only,
        ToolMessage("Returning structured response", tool_call_id="a2"),
    ]
    kept = _drop_structured_response(new, structured=True)
    assert len(kept) == 2
    assert [tc["id"] for tc in kept[0].tool_calls] == ["c1"]
    assert kept[0].content == "status"
    assert kept[1].tool_call_id == "c1"
    assert both.tool_calls[1]["id"] == "a1"  # the input message is not mutated
    _paired(kept)


def test_narrative_does_not_trace_to_an_earlier_turn():
    store = registry.get_service("events", None).store
    old = store.save(
        pa.table({"v": pa.array([123_456.0])}),
        produced_by="query_metrics",
        source="earlier turn",
        governed=True,
        elapsed_s=0.0,
    ).handle_id
    history = [
        HumanMessage("how many events last time?"),
        AIMessage("", tool_calls=[call("query_metrics", COUNT, "c0")]),
        ToolMessage(
            "card",
            tool_call_id="c0",
            artifact={"handle_id": old, "produced_by": "query_metrics", "governed": True},
        ),
        AIMessage("There were 123,456 events.\n\nVerified"),
        HumanMessage("how many events now?"),
    ]

    def stale(messages: list[BaseMessage]) -> AIMessage:
        hid = handles_in(messages)[-1]
        n = int(total(hid))
        text = f"There were {n:,} events, against 123,456 before."
        claim = {"text": f"{n:,} events", "type": "descriptive", "values": [n], "handles": [hid]}
        return answer_turn(text, [claim])

    model = ScriptedModel(
        script=[no_plan(), tool_turn(call("query_metrics", COUNT, "c1")), stale, stale]
    )
    out = run(model, history)
    verification = out["answer"].verification
    assert verification.label == "not_verified"
    assert "untraced number 123,456" in verification.problems


def test_narrative_traces_to_this_turns_worker_handles():
    from langchain_core.messages.notification import add_notification_metadata  # noqa: F401

    store = registry.get_service("events", None).store
    h = store.save(
        pa.table({"v": pa.array([4242.0])}),
        produced_by="run_sql",
        source="worker",
        governed=False,
        elapsed_s=0.0,
    ).handle_id
    note = HumanMessage(
        f"Task t1 finished.\nhandles: {h}", additional_kwargs={"source": "async_subagent"}
    )
    claim = {"text": "4242 events", "type": "descriptive", "values": [4242], "handles": [h]}
    model = ScriptedModel(script=[answer_turn("The worker found 4242 events.", [claim])])
    out = run(model, [HumanMessage("how many events?"), note])
    assert out["answer"].verification.label == "verified_off_model"


def test_drop_structured_response_strips_provider_copies_of_the_answer_call():
    from analyst.graph import _drop_structured_response

    both = AIMessage(
        [
            {"type": "text", "text": "status"},
            {"type": "tool_use", "id": "c1", "name": "query_metrics", "input": {}},
            {"type": "tool_use", "id": "a1", "name": "Answer", "input": {}},
            {"type": "function_call", "call_id": "a1", "name": "Answer", "arguments": "{}"},
        ],
        tool_calls=[call("query_metrics", COUNT, "c1"), call("Answer", {}, "a1")],
        additional_kwargs={
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "query_metrics"}},
                {"id": "a1", "type": "function", "function": {"name": "Answer"}},
            ]
        },
    )
    (kept, result) = _drop_structured_response(
        [both, ToolMessage("card", tool_call_id="c1"), ToolMessage("done", tool_call_id="a1")],
        structured=True,
    )
    assert [b.get("id") or b.get("call_id") for b in kept.content] == [None, "c1"]
    assert [tc["id"] for tc in kept.additional_kwargs["tool_calls"]] == ["c1"]
    assert result.tool_call_id == "c1"


def test_zero_claim_answer_is_not_retried():
    ack = answer_turn("I started a background task; I'll report back when it finishes.", [])
    model = ScriptedModel(script=[no_plan(), ack])
    out = run(model, [HumanMessage("forecast daily events")])
    assert out["retries"] == 0
    verification = out["answer"].verification
    assert verification.label == "not_verified"
    assert verification.reason == "no claims (decline or acknowledgement)"
    assert not any(
        isinstance(m, HumanMessage) and str(m.content).startswith("Verification failed")
        for m in out["messages"]
    )


# --- compaction at the graph level ------------------------------------------


def _saved(store, value: float, produced_by: str = "query_metrics", governed: bool = True) -> str:
    return store.save(
        pa.table({"v": pa.array([value])}),
        produced_by=produced_by,
        source="seeded",
        governed=governed,
        elapsed_s=0.0,
    ).handle_id


def _digest_turn(goal: str) -> AIMessage:
    return AIMessage("", tool_calls=[call("Digest", {"goal": goal}, "d0")])


@pytest.mark.parametrize(
    ("produced_by", "governed", "label"),
    [("query_metrics", True, "verified"), ("run_python", False, "verified_off_model")],
)
def test_compaction_graph_keeps_turn_handles_and_prior_handle(
    monkeypatch, produced_by, governed, label
):
    from analyst.middleware.compaction import is_compaction_summary

    monkeypatch.setenv("ANALYST_COMPACTION_TRIGGER_TOKENS", "1500")
    monkeypatch.setenv("ANALYST_COMPACTION_KEEP_MESSAGES", "1")
    get_settings.cache_clear()
    store = registry.get_service("events", None).store
    handle_a = _saved(store, 123_456.0, produced_by, governed)
    history = [
        HumanMessage("how many events last time?"),
        AIMessage("", tool_calls=[call(produced_by, COUNT, "c0")]),
        ToolMessage(
            "card " + "x" * 8000,
            tool_call_id="c0",
            artifact={"handle_id": handle_a, "produced_by": produced_by, "governed": governed},
        ),
        AIMessage("There were 123,456 events.\n\nVerified"),
        HumanMessage("how many events now?"),
    ]

    def cite_both(messages: list[BaseMessage]) -> AIMessage:
        hid_b = handles_in(messages)[-1]
        n = int(total(hid_b))
        text = f"There were {n:,} events, against 123,456 before."
        claims = [
            {"text": f"{n:,} events", "type": "descriptive", "values": [n], "handles": [hid_b]},
            {"text": "123,456 before", "type": "descriptive", "values": [123456], "handles": [handle_a]},
        ]
        return answer_turn(text, claims)

    model = ScriptedModel(
        script=[
            _digest_turn("events count"),
            no_plan(),
            tool_turn(call("query_metrics", COUNT, "c1")),
            cite_both,
        ]
    )
    out = run(model, history)
    messages = out["messages"]
    assert is_compaction_summary(messages[0])
    assert handle_a in messages[0].content
    assert messages[1].content == "how many events now?"
    assert out["answer"].verification.label == label, out["answer"].verification

    # second turn: the thread is already small, so no new digest is made
    model2 = ScriptedModel(
        script=[no_plan(), tool_turn(call("query_metrics", COUNT, "c2")), cite_total]
    )
    out2 = run(model2, [*messages, HumanMessage("and again?")])
    assert is_compaction_summary(out2["messages"][0])
    assert sum(is_compaction_summary(m) for m in out2["messages"]) == 1
    assert not any(
        "Summarise the earlier part" in str(m.content)
        for call_seen in model2.seen
        for m in call_seen
    )
    assert out2["answer"].verification.label == "verified"


def test_question_skips_compaction_summary():
    from langchain_core.messages.notification import add_notification_metadata

    from analyst.graph import _question
    from analyst.middleware.compaction import COMPACTION_SOURCE

    summary = add_notification_metadata(
        HumanMessage("Goal: 5 things"), source=COMPACTION_SOURCE
    )
    real = HumanMessage("real question")
    assert _question([real, summary]) is real


def test_compaction_default_model_comes_from_get_model(monkeypatch):
    import analyst.graph as graph_mod
    from analyst.middleware.compaction import is_compaction_summary

    monkeypatch.setenv("ANALYST_COMPACTION_TRIGGER_TOKENS", "200")
    monkeypatch.setenv("ANALYST_COMPACTION_KEEP_MESSAGES", "1")
    get_settings.cache_clear()
    specs = []
    model = ScriptedModel(
        script=[_digest_turn("g"), no_plan(), AIMessage("Nothing to compute.")]
    )
    monkeypatch.setattr(
        graph_mod, "get_model", lambda spec=None: (specs.append(spec), model)[1]
    )
    history = [
        HumanMessage("first " + "x" * 2000),
        AIMessage("ok"),
        HumanMessage("thanks"),
    ]
    out = asyncio.run(
        build_graph().ainvoke({"messages": history}, context=CTX)
    )
    assert is_compaction_summary(out["messages"][0])
    assert specs  # resolved through get_model, not an injected model
