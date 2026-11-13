"""Per-run context handed to the graph and every tool."""

from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any

from analyst.config import get_settings
from analyst.errors import InvalidRequestError


def _default_dataset() -> str:
    """``ANALYST_DATASET``, for a run context that names no dataset.

    LangGraph builds the context as ``AnalystContext(**ctx)`` from the run's
    JSON, so a partial context (``{"cutoff": ...}``) must not fail on a
    missing field.
    """
    dataset = get_settings().dataset
    if not dataset:
        raise InvalidRequestError(
            "no dataset configured",
            hint="set ANALYST_DATASET or pass dataset=... in the run context",
        )
    return dataset


@dataclass(frozen=True)
class AnalystContext:
    dataset: str = field(default_factory=_default_dataset)
    cutoff: date | None = None
    model: str | None = None
    background: bool = True
    question_id: str | None = None

    def __post_init__(self) -> None:
        # A forwarded run context is JSON, so its cutoff arrives as a string.
        if isinstance(self.cutoff, str):
            try:
                cutoff = date.fromisoformat(self.cutoff)
            except ValueError as exc:
                raise InvalidRequestError(
                    f"invalid cutoff {self.cutoff!r}",
                    hint="pass the cutoff as an ISO date, YYYY-MM-DD",
                ) from exc
            object.__setattr__(self, "cutoff", cutoff)

    @classmethod
    def from_settings(cls, **overrides: Any) -> "AnalystContext":
        """Fill ``dataset`` from ``ANALYST_DATASET`` unless overridden."""
        if overrides.get("dataset") is None:
            overrides["dataset"] = _default_dataset()
        return replace(cls(dataset=overrides["dataset"]), **overrides)
