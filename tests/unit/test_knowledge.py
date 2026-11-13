from datetime import date

import pytest

from analyst.errors import UnknownNameError
from analyst.memory import load_memory
from analyst.memory.knowledge import (
    TOPICS,
    describe_dimensions,
    describe_metrics,
    read_topic,
)
from analyst.semantic.model import ContextFact, Memory, Quirk

CUTOFF = date(2024, 6, 30)


@pytest.fixture
def memory(synthetic_dataset) -> Memory:
    """Fixture memory plus dated items on both sides of CUTOFF."""
    base = load_memory(synthetic_dataset)
    return base.model_copy(
        update={
            "quirks": [
                *base.quirks,
                Quirk(
                    kind="anomaly",
                    start=date(2024, 3, 1),
                    end=date(2024, 12, 31),
                    treatment="flag",
                    description="spans the cutoff",
                ),
                Quirk(
                    kind="schema_change",
                    start=date(2024, 9, 1),
                    description="added after the cutoff",
                ),
            ],
            "context_facts": [
                ContextFact(
                    date=date(2024, 1, 1), label="New Year Day", kind="holiday"
                ),
                ContextFact(
                    date=date(2024, 4, 1),
                    end=date(2024, 5, 1),
                    label="fare policy change",
                    kind="policy",
                ),
                ContextFact(
                    date=date(2024, 9, 2), label="Labor Day", kind="holiday"
                ),
            ],
        }
    )


def test_topics_are_the_documented_set():
    assert TOPICS == (
        "metrics",
        "dimensions",
        "quirks",
        "context_facts",
        "forecasting",
        "recap",
    )


def test_as_of_drops_future_quirks_and_facts(memory):
    m = memory.as_of(CUTOFF)
    assert all(q.start <= CUTOFF for q in m.quirks)
    assert all(f.date <= CUTOFF for f in m.context_facts)
    assert m.recap == "" and memory.as_of(None) == memory


def test_as_of_keeps_an_item_that_starts_before_the_cutoff(memory):
    m = memory.as_of(CUTOFF)
    spanning = [q for q in m.quirks if q.description == "spans the cutoff"]
    assert len(spanning) == 1 and spanning[0].end > CUTOFF
    assert [f.label for f in m.context_facts] == ["New Year Day", "fare policy change"]
    assert m.series == memory.series and m.semantic == memory.semantic


def test_recap_unavailable_under_cutoff(memory):
    assert read_topic(memory, "recap", cutoff=CUTOFF) == (
        "recap unavailable under a cutoff"
    )


def test_recap_without_cutoff_is_the_memory_recap(memory):
    assert read_topic(memory, "recap", cutoff=None) == "Synthetic hourly events."


def test_context_facts_topic_lists_ids_and_dates(memory):
    text = read_topic(memory, "context_facts", cutoff=None)
    for fact in memory.context_facts:
        assert fact.id in text
        assert fact.date.isoformat() in text
    assert len(text) <= 4096
    assert "- cf_" in text


def test_context_facts_topic_hides_future_facts(memory):
    text = read_topic(memory, "context_facts", cutoff=CUTOFF)
    assert CUTOFF.isoformat() in text and "Labor Day" not in text
    assert "New Year Day" in text and "fare policy change" in text


def test_quirks_topic_lists_dates_and_treatments(memory):
    text = read_topic(memory, "quirks", cutoff=None)
    assert "2024-01-20..2024-01-21" in text and "impute" in text
    assert "2024-09-01" in text


def test_quirks_topic_hides_future_quirks(memory):
    text = read_topic(memory, "quirks", cutoff=CUTOFF)
    assert "2024-09-01" not in text and "added after the cutoff" not in text
    assert "spans the cutoff" in text


def test_metrics_topic_reuses_describe_output(memory):
    text = read_topic(memory, "metrics", cutoff=None)
    approved = memory.semantic.model_copy(
        update={
            "metrics": [
                m for m in memory.semantic.metrics if m.status == "approved"
            ]
        }
    )
    assert describe_metrics(approved) in text
    assert "total_amount" in text
    assert "draft_metric" not in text


def test_dimensions_topic_reuses_describe_output(memory):
    text = read_topic(memory, "dimensions", cutoff=None)
    assert describe_dimensions(memory.semantic) in text
    assert "zone_group" in text


def test_forecasting_topic_is_generic_plus_series(memory):
    text = read_topic(memory, "forecasting", cutoff=None)
    assert "SeasonalNaive" in text and memory.series[0].name in text
    lowered = text.lower()
    assert "origin" in lowered and "80%" in text and "95%" in text
    assert "MS" in text and "summing daily intervals" in lowered
    # Dataset-agnostic: no schema names from the dataset leak into the prose.
    assert "event_count" in text


def test_unknown_topic_hint(memory):
    with pytest.raises(UnknownNameError) as excinfo:
        read_topic(memory, "weather", cutoff=None)
    assert "metrics" in excinfo.value.hint
    assert "weather" in str(excinfo.value)


def test_long_topic_is_truncated(memory):
    text = read_topic(
        memory.model_copy(update={"recap": "x" * 5000}), "recap", cutoff=None
    )
    assert len(text) <= 4096 and text.endswith("… (truncated)")
