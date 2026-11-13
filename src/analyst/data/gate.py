"""Process-wide cap on concurrently running engine queries."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

from analyst.config import get_settings


class QueryGate:
    def __init__(self, max_concurrent: int) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be at least 1")
        self.max_concurrent = max_concurrent
        self._semaphore = threading.BoundedSemaphore(max_concurrent)
        self._lock = threading.Lock()
        self._in_use = 0
        self._peak = 0

    @property
    def in_use(self) -> int:
        with self._lock:
            return self._in_use

    @property
    def peak(self) -> int:
        with self._lock:
            return self._peak

    def reset_peak(self) -> None:
        with self._lock:
            self._peak = self._in_use

    @contextmanager
    def slot(self) -> Iterator[None]:
        with self._semaphore:
            with self._lock:
                self._in_use += 1
                self._peak = max(self._peak, self._in_use)
            try:
                yield
            finally:
                with self._lock:
                    self._in_use -= 1


_gate: QueryGate | None = None
_gate_lock = threading.Lock()


def get_gate() -> QueryGate:
    global _gate
    with _gate_lock:
        if _gate is None:
            _gate = QueryGate(get_settings().max_concurrent_queries)
        return _gate


def reset_gate() -> None:
    """Drop the singleton so the next get_gate() re-reads settings."""
    global _gate
    with _gate_lock:
        _gate = None
