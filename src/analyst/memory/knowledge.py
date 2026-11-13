"""On-demand knowledge topics rendered from approved dataset memory.

The agent gets topic *names* in its prompt and pulls the content it needs with
``read_knowledge``. Every topic is cutoff-aware: rendering happens against
``memory.as_of(cutoff)`` so a time-travel run never reads a fact dated after
its cutoff.
"""

from collections.abc import Callable
from datetime import date

from analyst.errors import UnknownNameError
from analyst.semantic.model import (
    ContextFact,
    Memory,
    MemoryItem,
    Quirk,
    SemanticModel,
    SeriesSpec,
)

TOPICS: tuple[str, ...] = (
    "metrics",
    "dimensions",
    "quirks",
    "context_facts",
    "forecasting",
    "recap",
)

MAX_CHARS = 4096
_TRUNCATION = "… (truncated)"
_RECAP_UNAVAILABLE = "recap unavailable under a cutoff"
_NO_RECAP = "no recap is recorded for this dataset"

_FORECASTING_GUIDANCE = """\
## forecasting guidance
- The forecast origin is always the end of available data, never today: query \
the full range and let the engine set the origin (respecting the cutoff).
- Every forecast is backtested against SeasonalNaive; report whether the chosen \
model beat that baseline instead of hiding it (`beats_baseline` in the card \
meta, and the SeasonalNaive rows of the `backtest_handle`).
- Intervals are always 80% and 95%, as the columns `lo_80`, `hi_80`, `lo_95` \
and `hi_95`.
- Grain must match the frequency: day grain with freq "D", month grain with \
freq "MS". Aggregate first — ratios and averages cannot be summed across \
finer periods.
- Intervals never add up: summing daily intervals does not give a monthly \
interval, so forecast at the grain you report.
- Series specs below are canned argument sets, not tool arguments: pass the \
spec's metric, dimensions and freq to `forecast`. A spec's name only labels it \
here and in a forecast card's `candidates_source`.\
"""


def read_topic(memory: Memory, topic: str, *, cutoff: date | None) -> str:
    """Render one knowledge topic as compact Markdown (at most 4,096 chars)."""
    if topic not in TOPICS:
        available = ", ".join(TOPICS)
        raise UnknownNameError(
            f"unknown knowledge topic {topic!r}; available: {available}",
            hint=f"call read_knowledge with one of: {available}",
        )
    if topic == "recap":
        # Free prose cannot be date-filtered, so it never leaks past a cutoff.
        if cutoff is not None:
            return _RECAP_UNAVAILABLE
        return _truncate(memory.recap.strip() or _NO_RECAP)
    view = memory.as_of(cutoff)
    return _truncate(_TOPIC_RENDERERS[topic](view, cutoff))


def describe_metrics(model: SemanticModel) -> str:
    """One line per metric: name, synonyms, unit and description."""
    lines: list[str] = []
    for metric in model.metrics:
        parts = [metric.name]
        if metric.synonyms:
            parts.append(f"synonyms={metric.synonyms}")
        if metric.unit:
            parts.append(f"unit={metric.unit}")
        if metric.description:
            parts.append(metric.description)
        lines.append(": ".join(parts))
    return "\n".join(lines)


def describe_dimensions(model: SemanticModel) -> str:
    """One line per dimension: name, synonyms and description."""
    lines: list[str] = []
    for dimension in model.dimensions:
        parts = [dimension.name]
        if dimension.synonyms:
            parts.append(f"synonyms={dimension.synonyms}")
        if dimension.description:
            parts.append(dimension.description)
        lines.append(": ".join(parts))
    return "\n".join(lines)


def _metrics_topic(memory: Memory, cutoff: date | None) -> str:
    approved = memory.semantic.model_copy(
        update={"metrics": _approved(memory.semantic.metrics)}
    )
    body = describe_metrics(approved) or "no metrics are approved yet"
    return _section("metrics", body)


def _dimensions_topic(memory: Memory, cutoff: date | None) -> str:
    approved = memory.semantic.model_copy(
        update={"dimensions": _approved(memory.semantic.dimensions)}
    )
    body = describe_dimensions(approved) or "no dimensions are approved yet"
    return _section("dimensions", body)


def _quirks_topic(memory: Memory, cutoff: date | None) -> str:
    lines = [_heading("quirks", cutoff)]
    quirks = _approved(memory.quirks)
    if not quirks:
        lines.append("no known quirks are recorded")
    for quirk in quirks:
        span = _span(quirk.start, quirk.end, cutoff)
        lines.append(f"- {quirk.id} {span} {_quirk_tail(quirk)}".rstrip())
    return "\n".join(lines)


def _context_facts_topic(memory: Memory, cutoff: date | None) -> str:
    lines = [_heading("context_facts", cutoff)]
    lines.append("cite these ids as `fact_ids` when a claim rests on context")
    facts = _approved(memory.context_facts)
    if not facts:
        lines.append("no context facts are recorded")
    for fact in facts:
        lines.append(f"- {_fact_line(fact, cutoff)}")
    return "\n".join(lines)


def _forecasting_topic(memory: Memory, cutoff: date | None) -> str:
    specs = _approved(memory.series)
    lines = [_FORECASTING_GUIDANCE, "", "## series specs"]
    if not specs:
        lines.append("no series specs are recorded; the dataset defaults apply")
    lines.extend(f"- {_series_line(spec)}" for spec in specs)
    return "\n".join(lines)


_TOPIC_RENDERERS: dict[str, Callable[[Memory, date | None], str]] = {
    "metrics": _metrics_topic,
    "dimensions": _dimensions_topic,
    "quirks": _quirks_topic,
    "context_facts": _context_facts_topic,
    "forecasting": _forecasting_topic,
}


def _approved[T: MemoryItem](items: list[T]) -> list[T]:
    return [item for item in items if item.status == "approved"]


def _heading(title: str, cutoff: date | None) -> str:
    if cutoff is None:
        return f"## {title}"
    # Dated items after the cutoff are already gone from the rendered view.
    return f"## {title} (as of {cutoff.isoformat()}; later items are hidden)"


def _section(title: str, body: str) -> str:
    return f"## {title}\n{body}"


def _span(start: date, end: date | None, cutoff: date | None) -> str:
    """A date, or a span; under a cutoff an end past it reads ``..ongoing``."""
    if end is None or end == start:
        return start.isoformat()
    if cutoff is not None and end > cutoff:
        return f"{start.isoformat()}..ongoing"
    return f"{start.isoformat()}..{end.isoformat()}"


def _quirk_tail(quirk: Quirk) -> str:
    """Treatment and description; the kind already leads the line as its id."""
    head = f"({quirk.treatment})" if quirk.treatment else ""
    if not quirk.description:
        return head
    return f"{head}: {quirk.description}" if head else quirk.description


def _fact_line(fact: ContextFact, cutoff: date | None) -> str:
    return f"{fact.id} {_span(fact.date, fact.end, cutoff)}: {fact.label}"


def _series_line(spec: SeriesSpec) -> str:
    dimensions = ", ".join(spec.dimensions)
    seasons = ", ".join(str(length) for length in spec.season_lengths)
    models = ", ".join(spec.recommended_models) or "dataset defaults"
    status = f" ({spec.recommended_models_status})" if spec.recommended_models_status else ""
    return (
        f"{spec.name}: metric={spec.metric}, freq={spec.freq}, "
        f"dimensions=[{dimensions}], season_lengths=[{seasons}], "
        f"models=[{models}]{status}"
    )


def _truncate(text: str) -> str:
    if len(text) <= MAX_CHARS:
        return text
    return text[: MAX_CHARS - len(_TRUNCATION)] + _TRUNCATION
