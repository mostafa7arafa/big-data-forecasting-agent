"""Steer/stop consistency: repairing interrupted threads, cancelling in-flight work.

Steer is the Agent Server's ``multitask_strategy="interrupt"``, queue is
``"enqueue"``, stop is run cancel; the backend's part is that the next run
starts from a consistent thread and a cancelled run's work actually stops.
"""

import asyncio
import threading
import time

import pytest
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    ToolMessage,
)
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.prebuilt.tool_node import ToolCallRequest

from analyst import cancel
from analyst.data.engine import QueryError
from analyst.graph import CANCELLED_TOOL_TEXT, build_graph, repair_tool_calls
from tests.unit.test_graph import (  # noqa: F401 - the autouse fixture
    COUNT,
    CTX,
    ScriptedModel,
    answer_turn,
    call,
    events_service,
    no_plan,
)


def dangling_history() -> list[BaseMessage]:
    """A steered thread: the first run stopped mid-tool, then a new question."""
    return [
        HumanMessage("how many events?", id="q1"),
        AIMessage(
            "Pulling the totals.",
            tool_calls=[
                call("query_metrics", COUNT, "c1"),
                call("query_metrics", COUNT, "c2"),
            ],
            id="ai1",
        ),
        ToolMessage("rows: 1", tool_call_id="c1", name="query_metrics", id="t1"),
        HumanMessage("actually, just say hello", id="q2"),
    ]


def assert_provider_shaped(messages: list[BaseMessage]) -> None:
    """Every AI tool call is answered by the tool messages right after it."""
    for i, m in enumerate(messages):
        if isinstance(m, AIMessage) and m.tool_calls:
            following = []
            for n in messages[i + 1 :]:
                if not isinstance(n, ToolMessage):
                    break
                following.append(n.tool_call_id)
            assert {tc["id"] for tc in m.tool_calls} <= set(following), (i, following)


def run_updates(model: ScriptedModel, messages: list[BaseMessage]) -> tuple[dict, list[dict]]:
    """Final state and every node update of one graph run."""
    graph = build_graph(model=model)

    async def go():
        updates = []
        async for mode, chunk in graph.astream(
            {"messages": messages}, context=CTX, stream_mode=["updates", "values"]
        ):
            if mode == "updates":
                updates.append(chunk)
            else:
                final = chunk
        return final, updates

    return asyncio.run(go())


def hello() -> AIMessage:
    return answer_turn("Hello.", [])


# --- compact: repair and trace


def test_compact_repairs_dangling_tool_calls():
    model = ScriptedModel(script=[no_plan(), hello()])
    final, updates = run_updates(model, dangling_history())
    compact = next(u["compact"] for u in updates if "compact" in u)
    removals = [m for m in compact["messages"] if isinstance(m, RemoveMessage)]
    assert [r.id for r in removals] == [REMOVE_ALL_MESSAGES]
    assert_provider_shaped(final["messages"])
    # The analyst's model call saw a provider-valid history.
    assert_provider_shaped(model.seen[-1])
    repaired = [m for m in final["messages"] if isinstance(m, ToolMessage) and m.tool_call_id == "c2"]
    assert len(repaired) == 1
    assert repaired[0].content == CANCELLED_TOOL_TEXT
    assert repaired[0].status == "error"
    # Inserted with the AI message's other results, before the new question.
    ids = [m.tool_call_id if isinstance(m, ToolMessage) else m.id for m in final["messages"]]
    assert ids[:5] == ["q1", "ai1", "c1", "c2", "q2"]


def test_compact_records_interrupted_trace():
    model = ScriptedModel(script=[no_plan(), hello()])
    final, _ = run_updates(model, dangling_history())
    first, second = final["trace"]
    assert first["interrupted"] is True
    assert second["interrupted"] is False
    tools = {s["id"]: s["status"] for s in first["steps"] if s["kind"] == "tool"}
    assert tools == {"c1": "ok", "c2": "error"}


def test_compact_records_steer_with_no_saved_steps():
    # An interrupted analyst node leaves only the question in the thread.
    model = ScriptedModel(script=[no_plan(), hello()])
    history = [HumanMessage("how many events?"), HumanMessage("say hello instead")]
    final, updates = run_updates(model, history)
    compact = next(u["compact"] for u in updates if "compact" in u)
    assert "messages" not in compact  # nothing to repair
    first, second = final["trace"]
    assert first["interrupted"] is True and first["steps"] == []
    assert second["interrupted"] is False


def test_compact_no_op_on_consistent_history():
    history = [
        HumanMessage("how many events?"),
        AIMessage("", tool_calls=[call("query_metrics", COUNT, "c1")]),
        ToolMessage("rows: 1", tool_call_id="c1", name="query_metrics"),
        AIMessage("There were 10 events.\n\nVerified"),
        HumanMessage("thanks"),
    ]
    assert repair_tool_calls(history) == (history, False)
    model = ScriptedModel(script=[no_plan(), hello()])
    final, updates = run_updates(model, history)
    compact = next(u["compact"] for u in updates if "compact" in u)
    assert "messages" not in compact and "trace" not in compact
    assert [t["interrupted"] for t in final["trace"]] == [False]


def test_compact_repairs_before_compaction_one_remove_all(monkeypatch):
    import analyst.graph as graph_mod

    seen: list[list[BaseMessage]] = []

    async def fake_compact(messages, chat, **kwargs):
        seen.append(list(messages))
        summary = HumanMessage("summary", additional_kwargs={"source": "compaction"})
        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), summary, *messages[-2:]]}

    monkeypatch.setattr(graph_mod, "compact_messages", fake_compact)
    monkeypatch.setattr(graph_mod, "count_tokens_approximately", lambda m: 10**9)
    monkeypatch.setattr(graph_mod, "plan_cut", lambda *a: 1)
    model = ScriptedModel(script=[no_plan(), hello()])
    _, updates = run_updates(model, dangling_history())
    compact = next(u["compact"] for u in updates if "compact" in u)
    removals = [m for m in compact["messages"] if isinstance(m, RemoveMessage)]
    assert len(removals) == 1
    assert_provider_shaped(seen[0])  # compaction saw the repaired list
    assert compact["trace"][0]["interrupted"] is True


# --- cancel registry


def test_cancel_registry_register_and_unregister():
    hits: list[str] = []
    off = cancel.register("t-a", lambda: hits.append("a"))
    assert cancel.cancel_thread("t-a") == 1
    off()
    assert cancel.cancel_thread("t-a") == 0
    assert hits == ["a"]


def test_cancel_main_and_worker_runs():
    # Workers run on their own thread ids: stopping the main thread leaves them.
    hits: list[str] = []
    offs = [
        cancel.register("main", lambda: hits.append("main")),
        cancel.register("worker", lambda: hits.append("worker")),
    ]
    assert cancel.cancel_thread("main") == 1
    assert hits == ["main"]
    # Two runs on one thread id (main agent and worker): stopping it stops both.
    offs.append(cancel.register("main", lambda: hits.append("worker-same-thread")))
    assert cancel.cancel_thread("main") == 2
    assert sorted(hits) == ["main", "main", "worker-same-thread"]
    for off in offs:
        off()
    assert cancel.cancel_thread("main") == 0


def test_current_thread_id_reads_run_config():
    assert cancel.current_thread_id() is None  # outside a run

    from langchain_core.runnables import RunnableLambda

    seen = RunnableLambda(lambda _: cancel.current_thread_id()).invoke(
        None, {"configurable": {"thread_id": "t-9"}}
    )
    assert seen == "t-9"


# --- engine (the query itself: tests/unit/test_engine.py)


def test_tool_query_registers_under_thread_id(monkeypatch):
    # Through the real async agent path: the sync tool runs in an executor
    # thread and must still see the run's thread id.
    from langgraph.checkpoint.memory import InMemorySaver

    registered: list[str] = []
    real = cancel.register

    def spy(thread_id, fn):
        registered.append(thread_id)
        return real(thread_id, fn)

    monkeypatch.setattr(cancel, "register", spy)
    model = ScriptedModel(
        script=[no_plan(), AIMessage("", tool_calls=[call("query_metrics", COUNT, "c1")]), hello()]
    )
    graph = build_graph(model=model)
    graph.checkpointer = InMemorySaver()
    asyncio.run(
        graph.ainvoke(
            {"messages": [HumanMessage("how many events?")]},
            {"configurable": {"thread_id": "t-tool"}},
            context=CTX,
        )
    )
    assert "t-tool" in registered


# --- middleware


def _request(name: str) -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"name": name, "args": {}, "id": "x1", "type": "tool_call"},
        tool=None,
        state={},
        runtime=None,
    )


def _cancel_tool_call(monkeypatch, tool_name: str, thread_id: str | None):
    """Cancel an in-flight tool call through TraceMiddleware; what was cleaned up."""
    from analyst.trace import middleware as mw

    closed: list[str] = []
    cancelled: list[str] = []
    monkeypatch.setattr("analyst.sandbox.close_sandbox", closed.append)
    monkeypatch.setattr(mw, "try_cancel_thread", lambda t: cancelled.append(t) or 0)
    monkeypatch.setattr(mw, "current_thread_id", lambda: thread_id)

    async def handler(request):
        await asyncio.sleep(30)

    async def go():
        task = asyncio.create_task(mw.TraceMiddleware().awrap_tool_call(_request(tool_name), handler))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(go())
    return cancelled, closed


def test_cancelled_run_python_closes_sandbox(monkeypatch):
    cancelled, closed = _cancel_tool_call(monkeypatch, "run_python", "t-s")
    assert cancelled == ["t-s"]
    assert closed == ["t-s"]


def test_cancelled_other_tool_keeps_sandbox(monkeypatch):
    cancelled, closed = _cancel_tool_call(monkeypatch, "query_metrics", "t-s")
    assert cancelled == ["t-s"]
    assert closed == []


def test_cancel_without_thread_id_cleans_nothing(monkeypatch):
    cancelled, closed = _cancel_tool_call(monkeypatch, "run_python", None)
    assert cancelled == [] and closed == []


def test_try_cancel_thread_never_blocks_on_a_held_lock():
    hits: list[str] = []
    off = cancel.register("t-held", lambda: hits.append("x"))
    try:
        with cancel._lock:  # another thread is (un)registering
            assert cancel.try_cancel_thread("t-held") is None  # contended: no wait
        assert hits == []
        assert cancel.try_cancel_thread("t-held") == 1
        assert hits == ["x"]
    finally:
        off()


def test_cancel_is_not_lost_when_the_registry_lock_is_held(monkeypatch):
    """A contended registry lock neither stalls the loop nor drops the cancel."""
    import time

    from blockbuster import blockbuster_ctx

    from analyst.trace import middleware as mw

    monkeypatch.setattr("analyst.sandbox.close_sandbox", lambda t: None)
    monkeypatch.setattr(mw, "current_thread_id", lambda: "t-lock")
    fired = threading.Event()
    off = cancel.register("t-lock", fired.set)
    release = threading.Timer(0.5, cancel._lock.release)
    gaps: list[float] = []

    async def handler(request):
        await asyncio.sleep(30)

    async def ticker():
        last = time.monotonic()
        while True:
            await asyncio.sleep(0.01)
            now = time.monotonic()
            gaps.append(now - last)
            last = now

    async def go():
        task = asyncio.create_task(
            mw.TraceMiddleware().awrap_tool_call(_request("query_metrics"), handler)
        )
        await asyncio.sleep(0.05)
        tick = asyncio.create_task(ticker())
        await asyncio.sleep(0.03)  # the ticker is running
        cancel._lock.acquire()  # held by "another thread" while the stop lands
        release.start()
        with blockbuster_ctx():
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        await asyncio.sleep(0.03)
        tick.cancel()

    try:
        asyncio.run(go())
    finally:
        release.join()
        off()
    assert fired.is_set()  # retried off the loop once the lock came free
    assert max(gaps) < 0.3  # the loop kept running while the lock was held


def test_cancel_cleanup_failure_still_reraises(monkeypatch):
    from analyst.trace import middleware as mw

    def boom(thread_id):
        raise RuntimeError("docker is gone")

    monkeypatch.setattr("analyst.sandbox.close_sandbox", boom)
    monkeypatch.setattr(mw, "current_thread_id", lambda: "t-b")

    async def handler(request):
        await asyncio.sleep(30)

    async def go():
        task = asyncio.create_task(mw.TraceMiddleware().awrap_tool_call(_request("run_python"), handler))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(go())


def test_run_python_registers_sandbox_abort(monkeypatch):
    from langchain.tools import ToolRuntime

    from analyst.context import AnalystContext
    from analyst.sandbox.base import ExecResult

    rp_mod = __import__("analyst.tools.run_python", fromlist=["x"])
    aborted: list[str] = []

    class AbortableSandbox:
        def run(self, code):
            assert cancel.cancel_thread("t-py") == 1  # registered while it runs
            return ExecResult(ok=True, stdout="hi", error=None)

        def abort(self):
            aborted.append("abort")

        def close(self):
            pass

    monkeypatch.setattr(rp_mod, "get_sandbox", lambda *a: AbortableSandbox())
    runtime = ToolRuntime(
        state={},
        context=AnalystContext(dataset="events"),
        config={"configurable": {"thread_id": "t-py"}},
        stream_writer=lambda e: None,
        tool_call_id="t1",
        store=None,
    )
    rp_mod.run_python.invoke(
        {"name": "run_python", "args": {"code": "print('hi')", "runtime": runtime},
         "id": "t1", "type": "tool_call"}
    )
    deadline = time.monotonic() + 5  # the abort runs on its own thread
    while not aborted and time.monotonic() < deadline:
        time.sleep(0.01)
    assert aborted == ["abort"]
    assert cancel.cancel_thread("t-py") == 0  # unregistered after the cell


def test_repair_path_does_no_blocking_io_on_the_loop():
    from blockbuster import blockbuster_ctx

    model = ScriptedModel(script=[no_plan(), hello()])
    graph = build_graph(model=model)

    async def go():
        with blockbuster_ctx():
            return await graph.ainvoke({"messages": dangling_history()}, context=CTX)

    out = asyncio.run(go())
    assert out["trace"][0]["interrupted"] is True


# --- a real run, cancelled mid-tool


def _cancel_real_run(script, *, thread_id: str, registered: threading.Event):
    """Run the graph on ``thread_id`` under blockbuster; cancel it mid-tool."""
    from blockbuster import blockbuster_ctx
    from langgraph.checkpoint.memory import InMemorySaver

    graph = build_graph(model=ScriptedModel(script=script))
    graph.checkpointer = InMemorySaver()

    async def go():
        with blockbuster_ctx():
            task = asyncio.create_task(
                graph.ainvoke(
                    {"messages": [HumanMessage("how many events?")]},
                    {"configurable": {"thread_id": thread_id}},
                    context=CTX,
                )
            )
            await asyncio.to_thread(registered.wait, 20)
            await asyncio.sleep(0.2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(go())


def test_cancelled_run_stops_its_query(monkeypatch):
    from analyst.data.service import DataService

    registered = threading.Event()
    outcome: dict = {}
    real_register = cancel.register

    def spy(thread_id, fn):
        registered.set()
        return real_register(thread_id, fn)

    def slow(self, q):
        start = time.monotonic()
        try:
            self.engine.execute("select count(*) from range(10_000_000_000) a")
        except QueryError as exc:
            outcome["error"] = str(exc)
        outcome["elapsed"] = time.monotonic() - start
        raise RuntimeError("stopped")

    monkeypatch.setattr(cancel, "register", spy)
    monkeypatch.setattr(DataService, "query_metrics", slow)
    script = [no_plan(), AIMessage("", tool_calls=[call("query_metrics", COUNT, "c1")])]
    _cancel_real_run(script, thread_id="t-real-q", registered=registered)
    deadline = time.monotonic() + 15
    while "elapsed" not in outcome and time.monotonic() < deadline:
        time.sleep(0.05)
    assert "cancelled" in outcome["error"], outcome
    assert outcome["elapsed"] < 10
    assert "t-real-q" not in cancel._callbacks


def test_cancelled_run_aborts_and_closes_its_sandbox(monkeypatch):
    from analyst.sandbox.base import ExecResult

    rp_mod = __import__("analyst.tools.run_python", fromlist=["x"])
    registered = threading.Event()
    killed = threading.Event()
    closed: list[str] = []

    class SlowSandbox:
        def run(self, code):
            registered.set()
            killed.wait(20)
            return ExecResult(ok=False, stdout="", error="killed", restarted=True)

        def abort(self):
            time.sleep(0.05)  # the docker CLI: blocking, must stay off the loop
            killed.set()

        def close(self):
            pass

    monkeypatch.setattr(rp_mod, "get_sandbox", lambda *a: SlowSandbox())
    monkeypatch.setattr("analyst.sandbox.close_sandbox", closed.append)
    script = [no_plan(), AIMessage("", tool_calls=[call("run_python", {"code": "x = 1"}, "c1")])]
    _cancel_real_run(script, thread_id="t-real-py", registered=registered)
    assert killed.wait(5)
    assert closed == ["t-real-py"]
