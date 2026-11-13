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


def test_langgraph_json_points_at_graph(monkeypatch):
    config = json.loads((REPO / "langgraph.json").read_text())
    assert config == {
        "dependencies": ["."],
        "graphs": {"analyst": "./src/analyst/graph.py:graph"},
        "env": ".env",
    }
    path, _, attr = config["graphs"]["analyst"].partition(":")
    assert (REPO / path).is_file()
    for var in ("ANALYST_DATASET", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    get_settings.cache_clear()
    module = importlib.reload(importlib.import_module("analyst.graph"))
    assert hasattr(getattr(module, attr), "ainvoke")
