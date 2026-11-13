"""Sandbox protocol and result types for the CodeAct Python tool."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class SavedFrame:
    name: str
    path: Path  # host path
    rows: int


@dataclass(frozen=True)
class ExecResult:
    ok: bool
    stdout: str
    error: str | None
    saved: list[SavedFrame] = field(default_factory=list)
    truncated: bool = False
    restarted: bool = False


class Sandbox(Protocol):
    def run(self, code: str) -> ExecResult: ...

    def close(self) -> None: ...
