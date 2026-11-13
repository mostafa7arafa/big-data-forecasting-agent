"""Compaction middleware: trigger, safe cut, handle survival, failure fallback."""

import asyncio
from types import SimpleNamespace
from typing import ClassVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from analyst.middleware.compaction import (
    Digest,
    Finding,
    compact_messages,
    is_compaction_summary,
)

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


def _run(chat, msgs, trigger=100, keep=8, protect=None):
    return asyncio.run(
        compact_messages(
            msgs,
            chat,
            trigger_tokens=trigger,
            keep_messages=keep,
            protect_from=len(msgs) if protect is None else protect,
        )
    )


def _no_orphans(msgs):
    for i, m in enumerate(msgs):
        if isinstance(m, AIMessage) and m.tool_calls:
            after = {t.tool_call_id for t in msgs[i + 1 :] if isinstance(t, ToolMessage)}
            assert {c["id"] for c in m.tool_calls} <= after
        if isinstance(m, ToolMessage):
            before = {c["id"] for a in msgs[:i] if isinstance(a, AIMessage) for c in a.tool_calls}
            assert m.tool_call_id in before


def test_no_compaction_below_trigger():
    chat = FakeChat(digest=_digest())
    assert _run(chat, _conversation(), trigger=10**9) is None
    assert chat.calls == []


def test_compacts_above_trigger_keeps_tail_and_pairs():
    msgs = _conversation()
    out = _run(FakeChat(digest=_digest()), msgs)["messages"]
    assert isinstance(out[0], RemoveMessage)
    summary, tail = out[1], out[2:]
    assert is_compaction_summary(summary)
    assert "Why did trips fall in March?" in summary.content
    assert [m.content for m in tail] == [m.content for m in msgs[-8:]] and len(tail) == 8
    assert not isinstance(tail[0], ToolMessage)
    _no_orphans(tail)


def test_cut_landing_on_tool_message_moves_back_to_ai():
    out = _run(FakeChat(digest=_digest()), _conversation(), keep=7)["messages"]
    tail = out[2:]
    assert isinstance(tail[0], AIMessage)
    _no_orphans(tail)


def test_current_question_and_after_always_stay():
    msgs = _conversation() + [HumanMessage("and April?")]
    # an extra message after the question; keep=1 alone would drop the question
    msgs.append(AIMessage("working"))
    out = _run(FakeChat(digest=_digest()), msgs, keep=1, protect=len(msgs) - 2)["messages"]
    assert [m.content for m in out[2:]] == ["and April?", "working"]


def test_nothing_to_drop_when_question_is_first():
    msgs = _conversation()
    assert _run(FakeChat(digest=_digest()), msgs, protect=0) is None


def test_handles_survive_compaction():
    summary = _run(FakeChat(digest=_digest()), _conversation())["messages"][1]
    assert H_ONLY_IN_TOOL in summary.content and H_LISTED in summary.content
    assert "Handles:" in summary.content


def test_digest_failure_falls_back_to_no_compaction():
    assert _run(FakeChat(error=True), _conversation()) is None
    assert _run(FakeChat(digest={"nope": 1}), _conversation()) is None


def test_metadata_on_run(monkeypatch):
    seen = {}
    run = SimpleNamespace(add_metadata=lambda m: seen.update(m))
    monkeypatch.setattr("analyst.middleware.compaction.get_current_run_tree", lambda: run)
    assert _run(FakeChat(digest=_digest()), _conversation()) is not None
    assert seen == {"compacted": True, "compacted_messages": 13}


def test_dict_digest_is_validated():
    assert _run(FakeChat(digest=_digest().model_dump()), _conversation()) is not None


def test_input_messages_not_mutated():
    msgs = _conversation()
    _run(FakeChat(digest=_digest()), msgs)
    assert all(m.id is None for m in msgs)


def test_nothing_droppable_but_summaries_makes_no_model_call():
    from langchain_core.messages.notification import add_notification_metadata

    summary = add_notification_metadata(
        HumanMessage("Goal: old " + "x" * 2000), source="compaction"
    )
    msgs = [summary, HumanMessage("Q1"), *_conversation()[1:3]]
    chat = FakeChat(digest=_digest())
    chat.calls.clear()
    assert _run(chat, msgs, keep=1, protect=1) is None
    assert chat.calls == []
