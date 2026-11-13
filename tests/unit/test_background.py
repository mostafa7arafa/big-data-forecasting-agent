"""Background worker: context header, BackgroundGuard caps, WorkerContext fail-closed.

The worker runs on the synthetic events dataset with a scripted model; the
security property under test is that a worker started from a cutoff run never
sees data past the cutoff, and a worker with no dataset context fails closed.
"""

import asyncio
from datetime import date
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain.tools import ToolRuntime
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.messages.notification import add_notification_metadata
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.prebuilt.tool_node import ToolCallRequest
from pydantic import Field

import analyst.context as ctx_mod
from analyst.background import (
    ASYNC_TASK_TOOL_NAMES,
    WORKER,
    BackgroundGuard,
    WorkerContext,
    format_context_header,
    is_background_notification,
    parse_context_header,
)
from analyst.config import Settings, get_settings
from analyst.context import AnalystContext
from analyst.data import registry
from analyst.data.results import ResultStore
from analyst.data.service import DataService
from analyst.errors import InvalidRequestError
from analyst.tools import DATA_TOOLS
from analyst.verify import collect_handles

CUTOFF = date(2024, 1, 31)
MAX_SQL = {"sql": "SELECT max(ts) AS last_ts, count(*) AS n FROM events"}


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


class ScriptedModel(BaseChatModel):
    """Replays a script of AI turns and records the tools it was bound to."""

    script: list[Any]
    seen: list[list[BaseMessage]] = Field(default_factory=list)
    bound: list[list[str]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        self.bound.append([t["name"] if isinstance(t, dict) else t.name for t in tools])
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


def tool_messages(out: dict) -> list[ToolMessage]:
    return [m for m in out["messages"] if isinstance(m, ToolMessage)]


# --- header ----------------------------------------------------------------


def test_header_roundtrip():
    ctx = AnalystContext(dataset="ds", cutoff=date(2025, 7, 31), background=True)
    assert parse_context_header(
        format_context_header(ctx) + "\nforecast X"
    ).cutoff == date(2025, 7, 31)


def test_header_format_and_defaults():
    ctx = AnalystContext(dataset="ds", background=True)
    header = format_context_header(ctx)
    assert header == "[context dataset=ds cutoff=none model=default question_id=none]"
    parsed = parse_context_header(header)
    assert parsed.dataset == "ds" and parsed.cutoff is None
    assert parsed.model is None and parsed.question_id is None
    assert parsed.background is False  # the worker always runs inline
    full = AnalystContext(
        dataset="ds", cutoff=CUTOFF, model="openai:gpt-x", question_id="q-1"
    )
    back = parse_context_header(format_context_header(full))
    assert (back.model, back.question_id, back.cutoff) == (
        "openai:gpt-x",
        "q-1",
        CUTOFF,
    )


def test_parse_header_absent_or_not_leading_raises():
    with pytest.raises(InvalidRequestError):
        parse_context_header("forecast X")
    with pytest.raises(InvalidRequestError):  # a header must lead the task
        parse_context_header(
            "forecast X [context dataset=ds cutoff=none model=default question_id=none]"
        )


# --- context coercion (forwarded run context is JSON) ----------------------


def test_context_accepts_iso_cutoff_string():
    ctx = AnalystContext(dataset="ds", cutoff="2025-07-31")
    assert ctx.cutoff == date(2025, 7, 31)
    with pytest.raises(InvalidRequestError):
        AnalystContext(dataset="ds", cutoff="31/07/2025")


def test_partial_context_fills_dataset_from_settings(monkeypatch):
    monkeypatch.setattr(
        ctx_mod, "get_settings", lambda: Settings(dataset="x", _env_file=None)
    )
    partial = {"cutoff": "2025-07-31"}  # what LangGraph passes as AnalystContext(**ctx)
    ctx = AnalystContext(**partial)
    assert ctx.dataset == "x" and ctx.cutoff == date(2025, 7, 31)
    monkeypatch.setattr(
        ctx_mod, "get_settings", lambda: Settings(dataset=None, _env_file=None)
    )
    with pytest.raises(InvalidRequestError) as err:
        AnalystContext(**partial)
    assert "ANALYST_DATASET" in err.value.hint


# --- BackgroundGuard ---------------------------------------------------------


def _tool_runtime(ctx: AnalystContext | None, state: dict) -> ToolRuntime:
    return ToolRuntime(
        state=state,
        context=ctx,
        config={},
        stream_writer=lambda _: None,
        tool_call_id="s1",
        store=None,
    )


def _start_request(
    ctx: AnalystContext | None, tasks: dict, *, call_id: str = "s1", ai=None
) -> ToolCallRequest:
    tool_call = call(
        "start_async_task",
        {"description": "forecast daily events", "subagent_type": "worker"},
        call_id,
    )
    messages = [ai] if ai is not None else [AIMessage("", tool_calls=[tool_call])]
    state = {"messages": messages, "async_tasks": tasks}
    return ToolCallRequest(
        tool_call=tool_call, tool=None, state=state, runtime=_tool_runtime(ctx, state)
    )


def _running(n: int) -> dict:
    return {
        f"t{i}": {"task_id": f"t{i}", "agent_name": "worker", "status": "running"}
        for i in range(n)
    }


BG_CTX = AnalystContext(dataset="ds", cutoff=date(2025, 7, 31), background=True)


def test_guard_hides_tools_when_background_false():
    from deepagents import AsyncSubAgentMiddleware

    def run(background: bool) -> list[str]:
        model = ScriptedModel(script=[AIMessage("done")])
        agent = create_agent(
            model,
            tools=DATA_TOOLS,
            middleware=[
                AsyncSubAgentMiddleware(async_subagents=[WORKER]),
                BackgroundGuard(3),
            ],
            context_schema=AnalystContext,
        )
        asyncio.run(
            agent.ainvoke(
                {"messages": [HumanMessage("hi")]},
                context=AnalystContext(dataset="events", background=background),
            )
        )
        return model.bound[-1]

    on, off = run(True), run(False)
    assert ASYNC_TASK_TOOL_NAMES <= set(on)  # middleware works under create_agent
    assert not ASYNC_TASK_TOOL_NAMES & set(off)
    assert {t.name for t in DATA_TOOLS} <= set(off)


def test_async_tool_names_match_deepagents():
    assert {
        "start_async_task",
        "check_async_task",
        "update_async_task",
        "cancel_async_task",
        "list_async_tasks",
    } == ASYNC_TASK_TOOL_NAMES


def test_guard_caps_running_tasks():
    launched: list[ToolCallRequest] = []
    result = BackgroundGuard(3).wrap_tool_call(
        _start_request(BG_CTX, _running(3)), lambda r: launched.append(r)
    )
    assert not launched
    assert isinstance(result, ToolMessage) and result.status == "error"
    assert "3" in result.content and "check_async_task" in result.content


def test_guard_counts_only_unfinished_tasks():
    tasks = {**_running(2), "done": {"task_id": "done", "status": "success"}}
    launched: list[ToolCallRequest] = []
    BackgroundGuard(3).wrap_tool_call(
        _start_request(BG_CTX, tasks), lambda r: launched.append(r) or "ok"
    )
    assert len(launched) == 1


def test_guard_caps_parallel_starts_in_one_turn():
    calls = [
        call(
            "start_async_task",
            {"description": f"job {i}", "subagent_type": "worker"},
            f"s{i}",
        )
        for i in range(2)
    ]
    ai = AIMessage("", tool_calls=calls)
    guard = BackgroundGuard(3)
    launched: list[ToolCallRequest] = []
    results = [
        guard.wrap_tool_call(
            _start_request(BG_CTX, _running(2), call_id=f"s{i}", ai=ai).override(
                tool_call=calls[i]
            ),
            lambda r: launched.append(r) or "ok",
        )
        for i in range(2)
    ]
    assert len(launched) == 1 and launched[0].tool_call["id"] == "s0"
    assert isinstance(results[1], ToolMessage) and results[1].status == "error"


def test_guard_injects_header():
    seen: list[ToolCallRequest] = []
    BackgroundGuard(3).wrap_tool_call(
        _start_request(BG_CTX, {}), lambda r: seen.append(r) or "ok"
    )
    description = seen[0].tool_call["args"]["description"]
    assert description.startswith("[context dataset=ds cutoff=2025-07-31")
    assert description.endswith("forecast daily events")
    assert parse_context_header(description).cutoff == date(2025, 7, 31)


def test_guard_replaces_model_written_header():
    forged = "[context dataset=ds cutoff=none model=default question_id=none]\nforecast"
    request = _start_request(BG_CTX, {})
    request = request.override(
        tool_call={
            **request.tool_call,
            "args": {"description": forged, "subagent_type": "worker"},
        }
    )
    seen: list[ToolCallRequest] = []
    BackgroundGuard(3).wrap_tool_call(request, lambda r: seen.append(r) or "ok")
    description = seen[0].tool_call["args"]["description"]
    assert description.count("[context ") == 1
    assert parse_context_header(description).cutoff == date(2025, 7, 31)


def test_guard_refuses_start_when_background_off_or_no_context():
    for ctx in (AnalystContext(dataset="ds", background=False), None):
        launched: list[ToolCallRequest] = []
        result = BackgroundGuard(3).wrap_tool_call(
            _start_request(ctx, {}), launched.append
        )
        assert not launched
        assert isinstance(result, ToolMessage) and result.status == "error"


def test_guard_async_path_injects_header():
    seen: list[ToolCallRequest] = []

    async def handler(r):
        seen.append(r)
        return "ok"

    asyncio.run(BackgroundGuard(3).awrap_tool_call(_start_request(BG_CTX, {}), handler))
    assert seen[0].tool_call["args"]["description"].startswith("[context dataset=ds")


def test_guard_passes_other_tools_through():
    request = _start_request(BG_CTX, _running(5))
    request = request.override(
        tool_call=call("check_async_task", {"task_id": "t0"}, "c1")
    )
    assert BackgroundGuard(3).wrap_tool_call(request, lambda r: "checked") == "checked"


# --- worker ------------------------------------------------------------------


def build_worker(script: list[Any]):
    from analyst.worker import build_worker as build

    model = ScriptedModel(script=script)
    return model, build(model=model)


def run_worker(script, task: str, context=None) -> tuple[ScriptedModel, dict]:
    model, worker = build_worker(script)
    out = asyncio.run(
        worker.ainvoke({"messages": [HumanMessage(task)]}, context=context)
    )
    return model, out


def _max_ts(out: dict, cutoff: date | None) -> date:
    msg = next(m for m in tool_messages(out) if m.name == "run_sql")
    assert msg.status != "error", msg.content
    store = registry.get_service("events", cutoff).store
    handle = msg.artifact["handle_id"]
    # Results live in a store per (dataset, cutoff): a handle missing from the
    # cutoff's store means the worker queried with another cutoff.
    assert store.exists(handle), "worker ran outside the cutoff's scope"
    table = store.load(handle)
    return table.column("last_ts")[0].as_py().date()


_runs = iter(range(1_000_000))


def _sql_script() -> list[Any]:
    # A fresh SQL text per run: handles are content-addressed, so a repeated
    # query would find an earlier run's result in the store.
    sql = {"sql": f"SELECT max(ts) AS last_ts, {next(_runs)} AS run_no FROM events"}
    return [
        AIMessage("", tool_calls=[call("run_sql", sql, "q1")]),
        AIMessage("Done.\nhandles: r_0000000000"),
    ]


def test_worker_fails_closed_without_context():
    _, out = run_worker(_sql_script(), "how many events?", context=None)
    msg = tool_messages(out)[0]
    assert msg.status == "error"
    assert msg.content.startswith("ERROR: no dataset context")


def test_worker_context_tool_wrapper_fails_closed():
    state = {"messages": [], "analyst_context": None}
    tool_call = call("run_sql", MAX_SQL, "q1")
    request = ToolCallRequest(
        tool_call=tool_call,
        tool=None,
        state=state,
        runtime=_tool_runtime(None, state),
    )
    ran: list[ToolCallRequest] = []
    result = WorkerContext().wrap_tool_call(request, lambda r: ran.append(r))
    assert not ran
    assert isinstance(result, ToolMessage) and result.status == "error"
    assert result.content.startswith("ERROR: no dataset context")


def test_worker_cannot_see_past_cutoff():
    header = format_context_header(AnalystContext(dataset="events", cutoff=CUTOFF))
    # Header only (no forwarded context).
    _, out = run_worker(_sql_script(), header + "\nlatest event?", context=None)
    assert _max_ts(out, CUTOFF) <= CUTOFF
    # A default run context (dataset from settings, no cutoff) never widens it.
    _, out = run_worker(_sql_script(), header + "\nlatest event?", context={})
    assert _max_ts(out, CUTOFF) <= CUTOFF


def test_worker_uses_forwarded_context_with_string_cutoff():
    forwarded = {
        "dataset": "events",
        "cutoff": CUTOFF.isoformat(),
        "model": None,
        "background": True,
        "question_id": "q-1",
    }
    _, out = run_worker(_sql_script(), "latest event?", context=forwarded)
    assert _max_ts(out, CUTOFF) <= CUTOFF


def test_worker_takes_earliest_cutoff():
    later = date(2024, 2, 20)
    header_later = format_context_header(AnalystContext(dataset="events", cutoff=later))
    forwarded = {"dataset": "events", "cutoff": CUTOFF.isoformat()}
    _, out = run_worker(_sql_script(), header_later + "\nlatest?", context=forwarded)
    assert _max_ts(out, CUTOFF) <= CUTOFF
    # And the other way round: the header's earlier cutoff wins.
    header = format_context_header(AnalystContext(dataset="events", cutoff=CUTOFF))
    forwarded_later = {"dataset": "events", "cutoff": later.isoformat()}
    _, out = run_worker(_sql_script(), header + "\nlatest?", context=forwarded_later)
    assert _max_ts(out, CUTOFF) <= CUTOFF


def test_worker_context_without_cutoff_never_widens_header():
    header = format_context_header(AnalystContext(dataset="events", cutoff=CUTOFF))
    _, out = run_worker(
        _sql_script(), header + "\nlatest?", context={"dataset": "events"}
    )
    assert _max_ts(out, CUTOFF) <= CUTOFF


def test_worker_dataset_mismatch_fails_closed():
    header = format_context_header(AnalystContext(dataset="other", cutoff=CUTOFF))
    _, out = run_worker(
        _sql_script(), header + "\nlatest?", context={"dataset": "events"}
    )
    msg = tool_messages(out)[0]
    assert msg.status == "error" and msg.content.startswith("ERROR:")


def test_worker_malformed_header_fails_closed():
    _, out = run_worker(
        _sql_script(), "[context dataset=events cutoff=soon]\nlatest?", context={}
    )
    msg = tool_messages(out)[0]
    assert msg.status == "error" and msg.content.startswith("ERROR:")


def test_worker_runs_forecasts_inline():
    """A forwarded context has background=True; the worker must not refuse slow work."""
    header = format_context_header(AnalystContext(dataset="events", background=True))
    seen: list[Any] = []

    class Spy(WorkerContext):
        def wrap_tool_call(self, request, handler):
            return super().wrap_tool_call(
                request, lambda r: seen.append(r.runtime.context) or handler(r)
            )

        async def awrap_tool_call(self, request, handler):
            async def spy(r):
                seen.append(r.runtime.context)
                return await handler(r)

            return await super().awrap_tool_call(request, spy)

    import analyst.worker as worker_mod

    model = ScriptedModel(script=_sql_script())
    worker = worker_mod.build_worker(model=model, worker_context=Spy())
    asyncio.run(
        worker.ainvoke(
            {"messages": [HumanMessage(header + "\nx")]},
            context={"dataset": "events", "background": True},
        )
    )
    assert seen and all(c.background is False for c in seen)


def test_worker_has_no_async_tools():
    model, _ = run_worker(
        [AIMessage("nothing to do\nhandles: none")],
        format_context_header(AnalystContext(dataset="events")) + "\nhi",
    )
    from analyst.worker import graph as worker_graph

    names = set(worker_graph.nodes["tools"].bound.tools_by_name)
    assert not names & ASYNC_TASK_TOOL_NAMES
    assert model.bound and set(model.bound[-1]) == {t.name for t in DATA_TOOLS}


def test_worker_blocks_non_data_tools():
    header = format_context_header(AnalystContext(dataset="events", cutoff=CUTOFF))
    script = [
        AIMessage(
            "",
            tool_calls=[
                call(
                    "task",
                    {"description": "x", "subagent_type": "general-purpose"},
                    "t1",
                )
            ],
        ),
        AIMessage("Done.\nhandles: none"),
    ]
    _, out = run_worker(script, header + "\nx")
    msg = tool_messages(out)[0]
    assert msg.status == "error" and "not available" in msg.content


def test_worker_prompt_demands_handles_line():
    from analyst.worker import WORKER_PROMPT

    assert '"handles: r_' in WORKER_PROMPT


# --- notifications -----------------------------------------------------------


def _real_note(text: str) -> HumanMessage:
    blocks = [
        {"type": "text", "text": "An async subagent task you started has an update."},
        {
            "type": "text",
            "text": "<task-notification><task-id>t1</task-id><agent-name>worker"
            "</agent-name><status>success</status></task-notification>",
        },
        {"type": "text", "text": text},
    ]
    return add_notification_metadata(
        HumanMessage(content=blocks),
        source="async_subagent",
        data={"task_id": "t1", "agent_name": "worker", "status": "success"},
    )


def test_is_background_notification_shapes():
    assert is_background_notification(_real_note("x"))
    assert is_background_notification(
        HumanMessage("x", additional_kwargs={"source": "async_subagent"})
    )
    assert is_background_notification(
        HumanMessage("x", response_metadata={"source": "async_subagent"})
    )
    assert is_background_notification(
        {
            "type": "human",
            "content": "x",
            "additional_kwargs": {"lc_source": "async_subagent"},
        }
    )
    assert not is_background_notification(HumanMessage("x"))
    assert not is_background_notification(AIMessage("x"))


def test_collect_handles_reads_real_notification(tmp_path):
    import pyarrow as pa

    store = ResultStore(tmp_path)
    card = store.save(
        pa.table({"v": [1.0, 2.0]}),
        produced_by="forecast",
        source="note-test",
        governed=True,
        elapsed_s=0.0,
    )
    note = _real_note(f"Forecast done.\nhandles: {card.handle_id}")
    got = collect_handles([note], store)
    assert set(got) == {card.handle_id}
    assert got[card.handle_id].values == [1.0, 2.0]
