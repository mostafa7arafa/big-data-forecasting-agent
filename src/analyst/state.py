"""Answer schema and graph state shared by the agent, verifier and UI."""

from typing import Annotated, Any, Literal, NotRequired, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, field_validator


def _none_to_list(value: object) -> object:
    """Providers may send ``null`` for an optional list; treat it as empty."""
    return [] if value is None else value


def _null_lists(*fields: str):
    """A before-validator coercing ``None`` to ``[]`` on the given list fields.

    The JSON schema stays a plain array (with ``items``), which OpenAI accepts.
    """
    return field_validator(*fields, mode="before")(_none_to_list)


class Claim(BaseModel):
    text: str
    type: Literal["descriptive", "comparative", "forecast", "causal"]
    values: list[float] = Field(default_factory=list)
    handles: list[str] = Field(default_factory=list)
    fact_ids: list[str] = Field(default_factory=list)

    _lists = _null_lists("values", "handles", "fact_ids")


class ChartSpec(BaseModel):
    kind: Literal["line", "bar"]
    title: str
    handle: str
    x: str
    y: str
    series: str | None = None
    forecast_handle: str | None = None
    bands: list[int] = Field(default_factory=list)
    last_year_handle: str | None = None

    _lists = _null_lists("bands")


class Verification(BaseModel):
    label: Literal["verified", "verified_off_model", "not_verified"]
    reason: str
    off_model_tools: list[str] = Field(default_factory=list)
    problems: list[str] = Field(default_factory=list)


class Answer(BaseModel):
    narrative: str
    claims: list[Claim]
    charts: list[ChartSpec] = Field(default_factory=list)
    sql: list[str] = Field(default_factory=list)
    verification: Verification | None = None
    background_tasks: list[str] = Field(default_factory=list)

    _lists = _null_lists("claims", "charts", "sql", "background_tasks")


class Plan(BaseModel):
    needs_plan: bool
    steps: list[str] = Field(default_factory=list)


def _merge_tasks(
    existing: dict[str, Any] | None, update: dict[str, Any] | None
) -> dict[str, Any]:
    """Records by task id; an update replaces a task's record."""
    return {**(existing or {}), **(update or {})}


class AnalystState(TypedDict):
    messages: Annotated[list, add_messages]
    plan: list[str]
    draft: Answer | None
    answer: Answer | None
    verification: Verification | None
    retries: int
    #: background tasks started with ``start_async_task`` (deepagents records).
    async_tasks: NotRequired[Annotated[dict[str, Any], _merge_tasks]]
    #: task id -> {"dataset", "cutoff"} each background task started under.
    background_contexts: NotRequired[Annotated[dict[str, Any], _merge_tasks]]
    #: this turn's woken task context (a run started by a notification), or None.
    wake_context: NotRequired[dict[str, Any] | None]
