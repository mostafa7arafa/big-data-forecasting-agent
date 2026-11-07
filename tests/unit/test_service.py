from datetime import date

import pytest

from analyst.data.engine import QueryError
from analyst.data.service import DataService
from analyst.semantic.compiler import MetricQuery


def test_query_metrics_card_governed(synthetic_dataset, tmp_path):
    svc = DataService(synthetic_dataset, store_root=tmp_path / "store")
    card = svc.query_metrics(MetricQuery(metrics=["event_count"]))
    assert card.governed is True
    assert card.produced_by == "query_metrics"
    assert card.rows >= 1
    assert svc.store.exists(card.handle_id)


def test_run_sql_not_governed(synthetic_dataset, tmp_path):
    svc = DataService(synthetic_dataset, store_root=tmp_path / "store")
    card = svc.run_sql("SELECT count(*) AS n FROM events")
    assert card.governed is False
    assert card.produced_by == "run_sql"
    assert card.rows == 1


def test_run_sql_bad_sql_raises_query_error(synthetic_dataset, tmp_path):
    svc = DataService(synthetic_dataset, store_root=tmp_path / "store")
    with pytest.raises(QueryError):
        svc.run_sql("SELECT missing_column FROM events")


def test_range_beyond_data_note(synthetic_dataset, tmp_path):
    svc = DataService(synthetic_dataset, store_root=tmp_path / "store")
    card = svc.query_metrics(
        MetricQuery(
            metrics=["event_count"],
            grain="month",
            time_range=(date(2024, 3, 1), date(2024, 4, 1)),
        )
    )
    assert card.rows == 0
    beyond = [n for n in card.notes if "beyond available data" in n]
    assert len(beyond) == 1
    assert "2024-01-01" in beyond[0]
    assert "2024-02-29 23:00" in beyond[0]


def test_range_before_coverage_start_note(synthetic_dataset, tmp_path):
    svc = DataService(synthetic_dataset, store_root=tmp_path / "store")
    card = svc.query_metrics(
        MetricQuery(
            metrics=["event_count"],
            grain="month",
            time_range=(date(2023, 12, 1), date(2024, 3, 1)),
        )
    )
    assert any("beyond available data" in note for note in card.notes)


def test_in_range_query_no_beyond_note(synthetic_dataset, tmp_path):
    svc = DataService(synthetic_dataset, store_root=tmp_path / "store")
    card = svc.query_metrics(
        MetricQuery(
            metrics=["event_count"],
            grain="month",
            time_range=(date(2024, 1, 1), date(2024, 3, 1)),
        )
    )
    assert not any("beyond available data" in note for note in card.notes)


def test_full_month_in_coverage_no_partial_note(synthetic_dataset, tmp_path):
    svc = DataService(synthetic_dataset, store_root=tmp_path / "store")
    card = svc.query_metrics(
        MetricQuery(
            metrics=["event_count"],
            grain="month",
            time_range=(date(2024, 2, 1), date(2024, 3, 1)),
        )
    )
    assert card.rows == 1
    assert "partial_last_period" not in card.meta
    assert not any("partial" in note for note in card.notes)


def test_partial_last_period_note(synthetic_dataset, tmp_path):
    svc = DataService(
        synthetic_dataset,
        cutoff=date(2024, 2, 15),
        store_root=tmp_path / "cut",
    )
    card = svc.query_metrics(
        MetricQuery(
            metrics=["event_count"],
            grain="month",
            time_range=(date(2024, 1, 1), date(2024, 3, 1)),
        )
    )
    assert card.meta.get("partial_last_period") == "2024-02-01"
    partial = [n for n in card.notes if "partial" in n]
    assert len(partial) == 1
    assert "data ends 2024-02-15 23:00" in partial[0]


def test_hour_grain_never_partial(synthetic_dataset, tmp_path):
    svc = DataService(
        synthetic_dataset,
        cutoff=date(2024, 2, 15),
        store_root=tmp_path / "hour",
    )
    card = svc.query_metrics(
        MetricQuery(
            metrics=["event_count"],
            grain="hour",
            time_range=(date(2024, 2, 1), date(2024, 2, 3)),
        )
    )
    assert card.rows > 0
    assert "partial_last_period" not in card.meta
    assert not any("partial" in note for note in card.notes)


def test_describe_dimension_values(synthetic_dataset, tmp_path):
    svc = DataService(synthetic_dataset, store_root=tmp_path / "store")
    text = svc.describe("dimension:area")
    assert "North" in text
    assert "South" in text


def test_describe_unknown_dimension(synthetic_dataset, tmp_path):
    svc = DataService(synthetic_dataset, store_root=tmp_path / "store")
    with pytest.raises(ValueError, match="valid dimensions"):
        svc.describe("dimension:totally_missing")


def test_describe_unknown_target(synthetic_dataset, tmp_path):
    svc = DataService(synthetic_dataset, store_root=tmp_path / "store")
    with pytest.raises(ValueError, match="metrics"):
        svc.describe("not_a_target")


def test_handles_isolated_by_cutoff(synthetic_dataset, tmp_path, monkeypatch):
    from analyst.config import Settings

    monkeypatch.setattr(
        "analyst.data.service.get_settings",
        lambda: Settings(
            repo_root=synthetic_dataset.root,
            datasets_dir=synthetic_dataset.root / "datasets",
            artifacts_dir=tmp_path / "artifacts",
        ),
    )
    svc_jan = DataService(synthetic_dataset, cutoff=date(2024, 1, 31))
    svc_feb = DataService(synthetic_dataset, cutoff=date(2024, 2, 15))
    q = MetricQuery(metrics=["event_count"])
    card_jan = svc_jan.query_metrics(q)
    card_feb = svc_feb.query_metrics(q)
    assert svc_jan.store._root != svc_feb.store._root
    assert card_jan.handle_id != card_feb.handle_id
