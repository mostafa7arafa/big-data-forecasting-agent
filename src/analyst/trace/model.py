"""The run trace: what one answer cost, how long it took, and its steps.

Every token, cost and timing field is ``None`` when unknown, never 0: a
provider that reports no usage gives an unknown, not a free, run.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from analyst.state import Verification

__all__ = ["RunTrace", "StepTrace"]


class StepTrace(BaseModel):
    """One model call, tool call or graph node inside a run."""

    id: str
    kind: Literal["model", "tool", "node"]
    name: str
    #: ``None`` only for a message recorded without the trace middleware.
    started_at: datetime | None
    ended_at: datetime | None
    status: Literal["ok", "error", "cancelled"]
    tokens_in: int | None = None
    tokens_out: int | None = None
    tokens_cached: int | None = None
    tokens_reasoning: int | None = None
    cost_usd: float | None = None
    reasoning: str | None = None


class RunTrace(BaseModel):
    """One run (one answer) of the agent, folded from its messages."""

    run_id: str | None
    model: str
    started_at: datetime | None
    ended_at: datetime | None
    ttft_s: float | None
    total_s: float | None
    steps: list[StepTrace]
    tokens_in: int | None
    tokens_out: int | None
    tokens_cached: int | None
    cost_usd: float | None
    #: the last model call's input tokens: how full the context window was.
    context_tokens: int | None
    context_window: int | None
    compaction_trigger: int
    events: list[dict[str, Any]] = Field(default_factory=list)
    verification: Verification | None = None
    interrupted: bool = False
