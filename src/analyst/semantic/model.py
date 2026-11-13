"""Dataset-agnostic semantic model and memory-item schemas."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from analyst.errors import UnknownNameError


class MemoryItem(BaseModel):
    source: Literal["profile", "llm", "owner", "feedback"] = "owner"
    status: Literal["proposed", "approved", "rejected"] = "approved"
    confidence: float = Field(default=1.0, ge=0, le=1)


class TimeDimension(BaseModel):
    name: str
    column: str
    grains: list[Literal["hour", "day", "week", "month"]]


class Join(MemoryItem):
    name: str
    table: str
    on: str


class Dimension(MemoryItem):
    name: str
    expr: str
    join: str | None = None
    synonyms: list[str] = Field(default_factory=list)
    description: str = ""


class Filter(MemoryItem):
    name: str
    expr: str
    description: str = ""


class Metric(MemoryItem):
    name: str
    expr: str
    filters: list[str] = Field(default_factory=list)
    synonyms: list[str] = Field(default_factory=list)
    description: str = ""
    unit: str | None = None


class SemanticModel(BaseModel):
    table: str
    time_dimension: TimeDimension
    joins: list[Join] = Field(default_factory=list)
    dimensions: list[Dimension] = Field(default_factory=list)
    filters: list[Filter] = Field(default_factory=list)
    metrics: list[Metric] = Field(default_factory=list)

    def metric(self, name: str) -> Metric:
        return _match(self.metrics, name, "metric")

    def dimension(self, name: str) -> Dimension:
        return _match(self.dimensions, name, "dimension")

    def approved_only(self) -> "SemanticModel":
        return self.model_copy(
            update={
                "joins": [item for item in self.joins if item.status == "approved"],
                "dimensions": [
                    item for item in self.dimensions if item.status == "approved"
                ],
                "filters": [item for item in self.filters if item.status == "approved"],
                "metrics": [item for item in self.metrics if item.status == "approved"],
            }
        )


class Quirk(MemoryItem):
    kind: Literal["gap", "anomaly", "schema_change", "data_issue"]
    start: date
    end: date | None = None
    description: str = ""
    treatment: Literal["impute", "flag", "exclude"] | None = None


class SeriesSpec(MemoryItem):
    name: str
    metric: str
    dimensions: list[str] = Field(default_factory=list)
    freq: Literal["D", "MS"]
    season_lengths: list[int] = Field(default_factory=list)
    recommended_models: list[str] = Field(default_factory=list)
    recommended_models_status: Literal["approved", "proposed"] | None = None


class ContextFact(MemoryItem):
    date: date
    end: date | None = None
    label: str
    kind: Literal["holiday", "policy", "data"]


class Memory(BaseModel):
    semantic: SemanticModel
    quirks: list[Quirk]
    series: list[SeriesSpec]
    context_facts: list[ContextFact]
    recap: str


def _match[T: Dimension | Metric](items: list[T], name: str, kind: str) -> T:
    needle = name.casefold()
    for item in items:
        labels = [item.name, *item.synonyms]
        if any(label.casefold() == needle for label in labels):
            return item
    available = ", ".join(item.name for item in items)
    raise UnknownNameError(
        f"unknown {kind} {name!r}; available: {available}",
        hint=f"call describe('{kind}s')",
    )
