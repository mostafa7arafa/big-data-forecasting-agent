"""The background worker: a deep agent the analyst starts with ``start_async_task``.

Served by ``langgraph dev`` as graph ``worker``. It has the data tools only (no
async-task tools, so it cannot start more background work), runs everything
inline, and ends its report with a ``handles:`` line the analyst's verifier
reads. Its dataset and cutoff come from the task's required context header,
narrowed by the forwarded run context (see :mod:`analyst.background`); without
a header every tool fails closed.

Like the analyst graph, the module imports without API keys: a missing key
fails the first model call instead, with the provider's hint.
"""

from __future__ import annotations

from typing import Any

from deepagents import create_deep_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatResult
from langgraph.graph.state import CompiledStateGraph

from analyst.background import WorkerContext
from analyst.config import get_settings
from analyst.context import AnalystContext
from analyst.errors import AnalystError
from analyst.middleware import analyst_middleware
from analyst.providers import get_model
from analyst.tools import DATA_TOOLS

__all__ = ["WORKER_PROMPT", "build_worker", "graph"]

WORKER_PROMPT = """\
You are a background data worker for an analyst. The analyst gave you one task: \
do it with the data tools and report only what the results show.
- The task starts with a [context ...] line naming the dataset and the date the \
data ends; the tools already enforce it. Never describe anything after that date.
- Use `query_metrics` first; `run_sql` only when the semantic layer can't express \
the step. Read `read_knowledge("quirks")` before trusting an odd period.
- Run every forecast and query yourself, inline: you cannot start background tasks.
- Tool results are summary cards with handle ids (`r_…`). Every number you report \
comes from a tool result; never invent, estimate or recall one.
- Never report a number you computed yourself (sums, totals, averages, row or \
period counts). If the task needs one, get it from a tool (`query_metrics` with \
the right grain and time_range, or `run_sql`). Your report lists only numbers \
present in the results of the handles you cite.
- Write a short report of what you found, naming the handle each number came from.
- End your report with exactly one line "handles: r_…, r_…" listing every handle \
id you used."""


class _UnconfiguredModel(BaseChatModel):
    """Stands in when no key is configured, so the graph still imports."""

    reason: str

    @property
    def _llm_type(self) -> str:
        return "unconfigured"

    def bind_tools(self, tools: Any, **kwargs: Any) -> _UnconfiguredModel:
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        raise RuntimeError(self.reason)


def _default_model() -> BaseChatModel:
    try:
        return get_model()
    except AnalystError as exc:
        return _UnconfiguredModel(reason=f"{exc.message} ({exc.hint})")


def build_worker(
    *,
    model: BaseChatModel | None = None,
    worker_context: WorkerContext | None = None,
) -> CompiledStateGraph:
    """Compile the worker; ``model=None`` uses ``ANALYST_MODEL``.

    A run context naming a model (forwarded from the analyst) overrides it per
    run, so the worker runs on the analyst's model.
    """
    settings = get_settings()
    return create_deep_agent(
        model=model if model is not None else _default_model(),
        tools=DATA_TOOLS,
        system_prompt=WORKER_PROMPT,
        middleware=[worker_context or WorkerContext(), *analyst_middleware(settings)],
        context_schema=AnalystContext,
        name="worker",
    )


graph = build_worker()
