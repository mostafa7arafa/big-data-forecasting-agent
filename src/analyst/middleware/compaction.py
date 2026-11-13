"""Compact long conversations into a structured analytics digest.

Runs as a step of the outer analyst graph (graph.py ``compact`` node), on the
thread's messages before the turn starts; it is not agent middleware, because
the inner agent's state is not the thread's. Only prior turns are compacted.

When the approximate token count of the thread passes ``trigger_tokens``, every
message except the last ``keep_messages`` is replaced by one harness-added
``HumanMessage`` that renders a :class:`Digest`: the goal, confirmed findings
with their numbers and handles, live handles, open questions and user
preferences. Handle ids survive verbatim: the digest's handle list is the
union of what the model listed and every id found in the removed messages
(text, tool artifacts), and the message carries a ``Handles:`` line. Prior
handles resolve through the result store when a claim cites them.

The first user question is not kept as a separate message: the digest's
``goal`` carries it, and a later compaction folds the earlier digest in.
Any failure to build the digest is logged and leaves the messages unchanged.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Mapping
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
    get_notification_metadata,
)
from langchain_core.messages.notification import add_notification_metadata
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langsmith.run_helpers import get_current_run_tree
from pydantic import BaseModel, Field

__all__ = [
    "COMPACTION_SOURCE",
    "Digest",
    "Finding",
    "compact_messages",
    "is_compaction_summary",
]

logger = logging.getLogger(__name__)

COMPACTION_SOURCE = "compaction"
HANDLE_ID = re.compile(r"r_[0-9a-f]{10}")
_TOOL_TEXT_LIMIT = 2_000

DIGEST_PROMPT = (
    "Summarise the earlier part of an analyst conversation into a digest that "
    "lets the analysis continue. Record: the user's goal (their question, "
    "stated in full), confirmed findings with the exact numbers and the result "
    "handles (ids like r_0123456789) they came from, every handle still worth "
    "reusing, open questions (include the ids of background tasks still running, as 'task <id> running'), and stated user preferences. Copy numbers and "
    "handle ids exactly; never invent or round them."
)


class Finding(BaseModel):
    """One confirmed result: what was found, its numbers, and where it came from."""

    text: str
    values: list[float] = Field(default_factory=list)
    handles: list[str] = Field(default_factory=list)


class Digest(BaseModel):
    """What the analysis needs to keep after older messages are dropped."""

    goal: str
    findings: list[Finding] = Field(default_factory=list)
    live_handles: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    preferences: list[str] = Field(default_factory=list)


def is_compaction_summary(message: object) -> bool:
    """Whether ``message`` is a compaction digest (a locator, not trust)."""
    if isinstance(message, Mapping):
        kwargs = message.get("additional_kwargs")
        return (
            isinstance(kwargs, Mapping) and kwargs.get("lc_source") == COMPACTION_SOURCE
        )
    meta = (
        get_notification_metadata(message)
        if hasattr(message, "additional_kwargs")
        else None
    )
    return meta is not None and meta["source"] == COMPACTION_SOURCE


def _text(message: AnyMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "\n".join(
        part if isinstance(part, str) else str(part.get("text", ""))
        for part in content
        if isinstance(part, str | dict)
    )


def _handles_in(messages: list[AnyMessage]) -> list[str]:
    """Handle ids in the text, tool-call arguments and artifacts, in first-seen order."""
    found: dict[str, None] = {}

    def scan(value: Any) -> None:
        for hid in HANDLE_ID.findall(str(value)):
            found.setdefault(hid)

    for message in messages:
        scan(_text(message))
        if isinstance(message, AIMessage):
            scan(message.tool_calls)
        if isinstance(message, ToolMessage) and message.artifact is not None:
            scan(message.artifact)
    return list(found)


def render_digest(digest: Digest, handles: list[str]) -> str:
    """The digest as Markdown, ending with a ``Handles:`` line."""
    lines = [
        "## Earlier conversation (compacted)",
        "",
        f"**Goal:** {digest.goal}",
    ]
    if digest.findings:
        lines += ["", "**Findings:**"]
        for f in digest.findings:
            extra = []
            if f.values:
                extra.append("values: " + ", ".join(f"{v:g}" for v in f.values))
            if f.handles:
                extra.append("handles: " + ", ".join(f.handles))
            suffix = f" ({'; '.join(extra)})" if extra else ""
            lines.append(f"- {f.text}{suffix}")
    for title, items in (
        ("Open questions", digest.open_questions),
        ("Preferences", digest.preferences),
    ):
        if items:
            lines += ["", f"**{title}:**", *(f"- {item}" for item in items)]
    lines += ["", f"Handles: {', '.join(handles) if handles else 'none'}"]
    return "\n".join(lines)


def safe_cutoff(messages: list[AnyMessage], target: int) -> int:
    """Index where the kept tail starts, at or before ``target``; 0 = nothing to drop.

    Never splits an AI tool call from its ToolMessages, and the tail never
    starts with a ToolMessage.
    """
    cut = target
    if cut <= 0:
        return 0
    while cut < len(messages) and isinstance(messages[cut], ToolMessage):
        ids = set()
        j = cut
        while j < len(messages) and isinstance(messages[j], ToolMessage):
            ids.add(messages[j].tool_call_id)
            j += 1
        for i in range(cut - 1, -1, -1):
            m = messages[i]
            if isinstance(m, AIMessage) and ids & {tc.get("id") for tc in m.tool_calls}:
                return i
        return j  # orphan tool results: drop them with the summarised part
    return cut


def plan_cut(
    messages: list[AnyMessage], trigger_tokens: int, keep_messages: int, protect_from: int
) -> int:
    """Where to cut, or 0 for no compaction.

    ``protect_from`` is the index of the current user question: it and
    everything after it stay in the tail, so only prior turns are summarised.
    Returns 0 too when nothing but earlier digests would be dropped: a model
    call that cannot shrink the thread only adds digest-of-digest drift.
    """
    if count_tokens_approximately(messages) <= trigger_tokens:
        return 0
    cut = safe_cutoff(messages, min(len(messages) - keep_messages, protect_from))
    if all(is_compaction_summary(m) for m in messages[:cut]):
        return 0
    return cut


def _prompt(old: list[AnyMessage]) -> list[AnyMessage]:
    parts = []
    for m in old:
        text = _text(m)
        if isinstance(m, ToolMessage) and len(text) > _TOOL_TEXT_LIMIT:
            text = text[:_TOOL_TEXT_LIMIT] + " ...[truncated]"
        line = f"{m.type}: {text}"
        if isinstance(m, AIMessage) and m.tool_calls:
            line += f" tool_calls={m.tool_calls}"
        if isinstance(m, ToolMessage) and isinstance(m.artifact, dict):
            line += f" artifact={m.artifact}"
        parts.append(line)
    return [SystemMessage(DIGEST_PROMPT), HumanMessage("\n\n".join(parts))]


async def compact_messages(
    messages: list[AnyMessage],
    chat: BaseChatModel,
    *,
    trigger_tokens: int,
    keep_messages: int,
    protect_from: int,
) -> dict | None:
    """A state update replacing old messages with the digest, or None.

    None when below the trigger, when nothing can be dropped, or when the
    digest fails (logged); the run then continues with the messages unchanged.
    The update is ``RemoveMessage(REMOVE_ALL)``, the summary, the tail.
    """
    cut = plan_cut(messages, trigger_tokens, keep_messages, protect_from)
    if cut <= 0:
        return None
    old, tail = messages[:cut], messages[cut:]
    try:
        result = await chat.with_structured_output(Digest).ainvoke(_prompt(old))
        digest = result if isinstance(result, Digest) else Digest.model_validate(result)
    except Exception:
        logger.warning("compaction failed; continuing uncompacted", exc_info=True)
        return None
    handles = list(dict.fromkeys([*digest.live_handles, *_handles_in(old)]))
    handles = [h for h in handles if HANDLE_ID.fullmatch(h)]
    summary = add_notification_metadata(
        HumanMessage(content=render_digest(digest, handles), id=str(uuid.uuid4())),
        source=COMPACTION_SOURCE,
        data={"compacted_messages": len(old), "handles": handles},
    )
    run_tree = get_current_run_tree()
    if run_tree is not None:
        run_tree.add_metadata({"compacted": True, "compacted_messages": len(old)})
    # tail messages need ids for the add_messages reducer to keep them stable
    tail = [m if m.id is not None else m.model_copy(update={"id": str(uuid.uuid4())}) for m in tail]
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), summary, *tail]}
