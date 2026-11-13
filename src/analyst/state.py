"""Answer schema and graph state shared by the agent, verifier and UI."""

from typing import Annotated, Literal, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class Claim(BaseModel):
    text: str
    type: Literal["descriptive", "comparative", "forecast", "causal"]
    values: list[float] = Field(default_factory=list)
    handles: list[str] = Field(default_factory=list)
    fact_ids: list[str] = Field(default_factory=list)


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


class Plan(BaseModel):
    needs_plan: bool
    steps: list[str] = Field(default_factory=list)


class AnalystState(TypedDict):
    messages: Annotated[list, add_messages]
    plan: list[str]
    draft: Answer | None
    answer: Answer | None
    verification: Verification | None
    retries: int
