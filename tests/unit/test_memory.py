from datetime import date

import pytest
from pydantic import ValidationError

from analyst.data.dataset import load_dataset
from analyst.memory import load_memory
from analyst.semantic import Metric


def test_load_synthetic_memory(synthetic_dataset):
    memory = load_memory(synthetic_dataset)
    model = memory.semantic

    assert model.table == "events"
    assert model.time_dimension.name == "event_time"
    assert model.time_dimension.column == "ts"
    assert model.time_dimension.grains == ["hour", "day", "week", "month"]

    assert len(model.joins) == 1
    join = model.joins[0]
    assert join.name == "region"
    assert join.table == "regions"
    assert join.on == "events.region_id = region.region_id"

    area = model.dimension("area")
    assert area.expr == "region.area"
    assert area.synonyms == ["zone_group"]
    assert model.dimension("ZONE_GROUP").name == "area"
    region = model.dimension("region_id")
    assert region.expr == "region_id"

    assert [(item.name, item.expr) for item in model.filters] == [
        ("positive_qty", "qty > 0")
    ]
    assert model.metric("total_amount").expr == "sum(amount)"
    assert model.metric("total_amount").filters == []
    assert model.metric("event_count").expr == "count(*)"
    positive = model.metric("positive_amount")
    assert positive.expr == "sum(amount)"
    assert positive.filters == ["positive_qty"]
    assert model.metric("draft_metric").status == "proposed"

    assert len(memory.quirks) == 1
    quirk = memory.quirks[0]
    assert quirk.kind == "gap"
    assert quirk.start == date(2024, 1, 20)
    assert quirk.end == date(2024, 1, 21)
    assert quirk.treatment == "impute"

    assert memory.context_facts == []
    assert len(memory.series) == 1
    series = memory.series[0]
    assert series.name == "daily_events"
    assert series.metric == "event_count"
    assert series.dimensions == []
    assert series.freq == "D"
    assert series.season_lengths == [7]
    assert series.recommended_models == []
    assert memory.recap == "Synthetic hourly events.\n"


@pytest.mark.taxi
def test_metric_lookup_by_synonym():
    memory = load_memory(load_dataset("nyc_taxi"))
    assert memory.semantic.dimension("borough").name == "pickup_borough"


@pytest.mark.taxi
def test_unknown_metric_lists_available():
    memory = load_memory(load_dataset("nyc_taxi"))
    with pytest.raises(KeyError, match="trips"):
        memory.semantic.metric("not_a_metric")


def test_confidence_validated():
    with pytest.raises(ValidationError):
        Metric(name="draft", expr="count(*)", confidence=1.5)


def test_approved_only_drops_proposed(synthetic_dataset):
    approved = load_memory(synthetic_dataset).semantic.approved_only()
    names = [metric.name for metric in approved.metrics]
    assert "draft_metric" not in names
    assert names == ["total_amount", "event_count", "positive_amount"]
