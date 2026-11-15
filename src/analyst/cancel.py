"""Per-thread cancellation of in-flight work (queries, sandbox cells).

A cancelled run's asyncio task stops at its next ``await``, but a sync tool
keeps running in its executor thread: a DuckDB query or a sandbox cell would
finish (or time out) long after the user pressed stop. Work that can be
interrupted registers a callback here under the thread id while it runs;
:func:`cancel_thread` fires every callback of that thread.

Keying: callbacks are keyed by thread id only, and ``cancel_thread`` is the
explicit API: it cancels everything running for that thread, from any run.
The main agent and a background worker never share a thread id (the worker
runs on its own Agent Server thread), and when two runs do share one (an
interrupted run still draining while its replacement starts), stopping the
thread means stopping all of its work.
"""

from __future__ import annotations

import itertools
import logging
import threading
from collections.abc import Callable

from langgraph.config import get_config

__all__ = ["cancel_thread", "current_thread_id", "register", "try_cancel_thread"]

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_callbacks: dict[str, dict[int, Callable[[], None]]] = {}
_tokens = itertools.count()


def current_thread_id() -> str | None:
    """The LangGraph thread id of the current run; ``None`` outside a run."""
    try:
        thread_id = get_config()["configurable"]["thread_id"]
    except (RuntimeError, KeyError, TypeError):
        return None
    return str(thread_id) if thread_id is not None else None


def register(thread_id: str, cancel: Callable[[], None]) -> Callable[[], None]:
    """Register ``cancel`` for ``thread_id``; returns the unregister function."""
    token = next(_tokens)
    with _lock:
        _callbacks.setdefault(thread_id, {})[token] = cancel

    def unregister() -> None:
        with _lock:
            group = _callbacks.get(thread_id)
            if group is not None:
                group.pop(token, None)
                if not group:
                    del _callbacks[thread_id]

    return unregister


def _fire(callbacks: list[Callable[[], None]]) -> int:
    count = 0
    for cancel in callbacks:  # outside the lock: a callback may unregister
        try:
            cancel()
        except Exception:
            logger.warning("cancel callback failed", exc_info=True)
        count += 1
    return count


def cancel_thread(thread_id: str) -> int:
    """Call every callback registered for ``thread_id``; returns how many ran.

    Waits for the registry lock: call it off the event loop (see
    :func:`try_cancel_thread` for the loop).
    """
    with _lock:
        callbacks = list((_callbacks.get(thread_id) or {}).values())
    return _fire(callbacks)


def try_cancel_thread(thread_id: str) -> int | None:
    """:func:`cancel_thread` without waiting: ``None`` when the lock is contended.

    Safe on the event loop. On ``None`` nothing was cancelled yet; the caller
    must retry with :func:`cancel_thread` off the loop, never drop the cancel.
    """
    if not _lock.acquire(blocking=False):
        return None
    try:
        callbacks = list((_callbacks.get(thread_id) or {}).values())
    finally:
        _lock.release()
    return _fire(callbacks)
