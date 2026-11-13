"""Compaction middleware: trigger, safe cut, handle survival, failure fallback."""

import asyncio
from types import SimpleNamespace
from typing import ClassVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from analyst.data.results import ResultStore
from analyst.middleware import CompactionMiddleware, analyst_middleware
from analyst.middleware.compaction import Digest, Finding, is_compaction_summary
from analyst.verify import collect_handles

H_LISTED = "r_aaaaaaaaaa"
H_ONLY_IN_TOOL = "r_bbbbbbbbbb"


class FakeChat(BaseChatModel):
    """Structured output returns ``digest`` (or raises ``error``)."""

    digest: object = None
    error: bool = False
    calls: ClassVar[list] = []

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=""))])

    def with_structured_output(self, schema, **kwargs):
        def run(prompt):
            self.calls.append(prompt)
            if self.error:
                raise RuntimeError("boom")
            return self.digest

        return RunnableLambda(run)


def _digest():
    return Digest(
        goal="Why did trips fall in March?",
        findings=[Finding(text="Down 3%", values=[-2.99], handles=[H_LISTED])],
        live_handles=[H_LISTED],
        open_questions=["weather?"],
        preferences=["short answers"],
    )


def _conversation(n_pairs=10):
    msgs = [HumanMessage("Why did trips fall in March?")]
    for i in range(n_pairs):
        handle = H_ONLY_IN_TOOL if i == 1 else f"r_{i:010x}"
        msgs.append(
            AIMessage(
                content="",
                tool_calls=[{"name": "run_sql", "args": {"sql": "x"}, "id": f"c{i}"}],
            )
        )
        msgs.append(
            ToolMessage(
                content="rows " + "y" * 400,
                tool_call_id=f"c{i}",
                artifact={
                    "handle_id": handle,
                    "produced_by": "run_sql",
                    "governed": True,
                },
            )
        )
    return msgs


def _mw(chat, trigger=100, keep=8):
    return CompactionMiddleware(trigger, keep, model=chat)


RUNTIME = SimpleNamespace(context=None)


def _run(mw, msgs):
    return mw.before_model({"messages": msgs}, RUNTIME)


def _no_orphans(msgs):
    for i, m in enumerate(msgs):
        if isinstance(m, AIMessage) and m.tool_calls:
            after = {
                t.tool_call_id for t in msgs[i + 1 :] if isinstance(t, ToolMessage)
            }
            assert {c["id"] for c in m.tool_calls} <= after
        if isinstance(m, ToolMessage):
            before = {
                c["id"]
                for a in msgs[:i]
                if isinstance(a, AIMessage)
                for c in a.tool_calls
            }
            assert m.tool_call_id in before


def test_no_compaction_below_trigger():
    chat = FakeChat(digest=_digest())
    assert _run(_mw(chat, trigger=10**9), _conversation()) is None
    assert chat.calls == []


def test_compacts_above_trigger_keeps_tail_and_pairs():
    msgs = _conversation()
    update = _run(_mw(FakeChat(digest=_digest())), msgs)
    out = update["messages"]
    assert isinstance(out[0], RemoveMessage)
    summary, tail = out[1], out[2:]
    assert is_compaction_summary(summary)
    assert "Why did trips fall in March?" in summary.content
    assert len(tail) == 8 and tail == msgs[-8:]
    assert not isinstance(tail[0], ToolMessage)
    _no_orphans(tail)


def test_cut_landing_on_tool_message_moves_back_to_ai():
    msgs = _conversation()
    # keep=7 would start the tail on a ToolMessage
    out = _run(_mw(FakeChat(digest=_digest()), keep=7), msgs)["messages"]
    tail = out[2:]
    assert isinstance(tail[0], AIMessage)
    _no_orphans(tail)


def test_handles_survive_compaction(tmp_path):
    update = _run(_mw(FakeChat(digest=_digest())), _conversation())
    summary = update["messages"][1]
    assert H_ONLY_IN_TOOL in summary.content and H_LISTED in summary.content
    assert "Handles:" in summary.content

    import pyarrow as pa

    store = ResultStore(tmp_path)
    saved = store.save(
        pa.table({"n": [1.0, 2.0]}),
        produced_by="run_sql",
        source="select 1",
        governed=True,
        elapsed_s=0.0,
    )
    msgs = _conversation()
    msgs[2] = msgs[2].model_copy(
        update={"artifact": {"handle_id": saved.handle_id, "produced_by": "run_sql"}}
    )
    summary = _run(_mw(FakeChat(digest=_digest())), msgs)["messages"][1]
    found = collect_handles([summary], store)
    assert list(found) == [saved.handle_id]  # H_LISTED etc. unknown: dropped


def test_digest_failure_falls_back_to_no_compaction():
    assert _run(_mw(FakeChat(error=True)), _conversation()) is None
    # invalid structured output
    assert _run(_mw(FakeChat(digest={"nope": 1})), _conversation()) is None


def test_async_path_and_metadata(monkeypatch):
    seen = {}
    run = SimpleNamespace(add_metadata=lambda m: seen.update(m))
    monkeypatch.setattr(
        "analyst.middleware.compaction.get_current_run_tree", lambda: run
    )
    mw = _mw(FakeChat(digest=_digest()))
    update = asyncio.run(mw.abefore_model({"messages": _conversation()}, RUNTIME))
    assert update is not None
    assert seen == {"compacted": True, "compacted_messages": 13}


def test_dict_digest_is_validated():
    update = _run(_mw(FakeChat(digest=_digest().model_dump())), _conversation())
    assert update is not None


def test_compaction_is_first_middleware_and_configured():
    from analyst.config import Settings

    s = Settings()
    assert (s.compaction_trigger_tokens, s.compaction_keep_messages) == (60_000, 8)
    first = analyst_middleware(s)[0]
    assert isinstance(first, CompactionMiddleware)
    assert not any(
        isinstance(m, CompactionMiddleware)
        for m in analyst_middleware(s, compaction=False)
    )
