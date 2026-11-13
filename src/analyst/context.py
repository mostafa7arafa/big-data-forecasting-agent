"""Per-run context handed to the graph and every tool."""

from dataclasses import dataclass, replace
from datetime import date
from typing import Any

from analyst.config import get_settings
from analyst.errors import InvalidRequestError


@dataclass(frozen=True)
class AnalystContext:
    dataset: str
    cutoff: date | None = None
    model: str | None = None
    background: bool = True
    question_id: str | None = None

    @classmethod
    def from_settings(cls, **overrides: Any) -> "AnalystContext":
        """Fill ``dataset`` from ``ANALYST_DATASET`` unless overridden."""
        if overrides.get("dataset") is None:
            dataset = get_settings().dataset
            if not dataset:
                raise InvalidRequestError(
                    "no dataset configured",
                    hint="set ANALYST_DATASET or pass dataset=...",
                )
            overrides["dataset"] = dataset
        return replace(cls(dataset=overrides["dataset"]), **overrides)
