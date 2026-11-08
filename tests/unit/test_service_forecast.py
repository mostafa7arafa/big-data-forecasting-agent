"""DataService.forecast on result handles. Series are long enough for daily defaults."""

import json
import shutil
from datetime import date, datetime, timedelta

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from analyst.config import Settings
from analyst.data.dataset import DatasetConfig, load_dataset
from analyst.data.service import DataService
from analyst.forecasting.core import BacktestReport, ForecastRun
from analyst.forecasting.models import DEFAULTS, candidate_models
from analyst.semantic.compiler import MetricQuery, compile_query

_DAILY_MODELS = {str(model) for model in candidate_models("D")}


def _long_events(synthetic_dataset: DatasetConfig, tmp_path) -> DatasetConfig:
    """Daily history past the 3-window daily backtest minimum (98 points)."""
    root = tmp_path / "events-root"
    data = root / "data"
    data.mkdir(parents=True)
    start = date(2024, 1, 1)
    days = 140
    rows = []
    for offset in range(days):
        day = start + timedelta(days=offset)
        for region_id in (1, 2, 3, 4):
            rows.append(
                (
                    datetime(day.year, day.month, day.day, 23),
                    region_id,
                    float(100 + 10 * day.weekday() + region_id),
                    1,
                )
            )
    columns = list(zip(*rows, strict=True))
    pq.write_table(
        pa.table(
            {
                "ts": pa.array(columns[0], type=pa.timestamp("us")),
                "region_id": pa.array(columns[1], type=pa.int32()),
                "amount": pa.array(columns[2], type=pa.float64()),
                "qty": pa.array(columns[3], type=pa.int32()),
            }
        ),
        data / "events_long.parquet",
    )
    (data / "regions.csv").write_text(
        "region_id,area\n1,North\n2,North\n3,South\n4,South\n"
    )
    dataset_dir = root / "datasets" / "events"
    shutil.copytree(synthetic_dataset.dir, dataset_dir)
    end = start + timedelta(days=days)
    (dataset_dir / "dataset.yaml").write_text(
        f"""\
name: events
description: Synthetic hourly events
files: data/events_long.parquet
table: events
time_column: ts
valid_time_range: [{start.isoformat()}, {end.isoformat()}]
lookups:
  - {{name: regions, file: data/regions.csv}}
"""
    )
    settings = Settings(repo_root=root, datasets_dir=root / "datasets")
    return load_dataset("events", settings)


def _four_decimals(value: float) -> bool:
    return round(float(value), 4) == float(value)


@pytest.fixture(scope="module")
def daily_by_area(synthetic_dataset, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("service-forecast")
    dataset = _long_events(synthetic_dataset, tmp)
    svc = DataService(dataset, store_root=tmp / "store")
    query = MetricQuery(
        metrics=["total_amount"],
        dimensions=["area"],
        grain="day",
    )
    card = svc.query_metrics(query)
    forecast_card = svc.forecast(
        card.handle_id,
        id_col="area",
        value_col="total_amount",
        freq="D",
    )
    return svc, card, forecast_card


def test_forecast_from_query_handle(daily_by_area):
    svc, card, forecast_card = daily_by_area
    assert forecast_card.rows == 28 * 2
    assert forecast_card.meta["model"] in _DAILY_MODELS
    assert forecast_card.produced_by == "forecast"
    assert forecast_card.governed is True
    assert forecast_card.meta["horizon"] == DEFAULTS["D"]["horizon"]
    assert forecast_card.meta["freq"] == "D"
    assert forecast_card.meta["input_handle"] == card.handle_id
    assert isinstance(forecast_card.meta["origin"], str)
    assert isinstance(forecast_card.meta["beats_baseline"], bool)
    backtest = forecast_card.meta["backtest"]
    assert backtest
    models = {row["model"] for row in backtest}
    assert models <= {forecast_card.meta["model"], "SeasonalNaive"}
    assert "SeasonalNaive" in models
    assert forecast_card.meta["model"] in models
    score_keys = {
        "unique_id",
        "model",
        "mae",
        "mase",
        "relative_mae",
        "coverage_80",
        "coverage_95",
        "scaled_crps",
    }
    for row in backtest:
        assert set(row) == score_keys
        for key in score_keys - {"unique_id", "model"}:
            assert _four_decimals(row[key])
    expected_source = {
        "input_handle": card.handle_id,
        "freq": "D",
        "horizon": 28,
        "value_col": "total_amount",
        "id_col": "area",
        "time_col": "period",
        "levels": [80, 95],
    }
    assert forecast_card.source == json.dumps(
        expected_source, separators=(",", ":"), sort_keys=True
    )
    if forecast_card.meta["beats_baseline"]:
        assert "no model beat the seasonal baseline" not in forecast_card.notes
    else:
        assert forecast_card.notes[-1] == "no model beat the seasonal baseline"
    assert any("known gap" in note for note in forecast_card.notes)
    table = svc.store.load(forecast_card.handle_id)
    assert table.column_names == [
        "unique_id",
        "ds",
        "yhat",
        "lo_80",
        "hi_80",
        "lo_95",
        "hi_95",
    ]
    assert set(table.column("unique_id").to_pylist()) == {"North", "South"}
    again = svc.forecast(
        card.handle_id,
        id_col="area",
        value_col="total_amount",
        freq="D",
    )
    assert again.handle_id == forecast_card.handle_id


def test_forecast_governed_propagates(daily_by_area):
    svc, card, forecast_card = daily_by_area
    assert card.governed is True
    assert forecast_card.governed is True
    sql = compile_query(
        svc.memory.semantic,
        MetricQuery(metrics=["total_amount"], grain="day"),
    )
    raw = svc.run_sql(sql)
    assert raw.governed is False
    ungoverned = svc.forecast(raw.handle_id, value_col="total_amount", freq="D")
    assert ungoverned.governed is False
    assert ungoverned.produced_by == "forecast"
    loaded = svc.store.load(ungoverned.handle_id)
    assert set(loaded.column("unique_id").to_pylist()) == {"series"}


def _fake_run(ids, *, beats_baseline, adjustments, mae=1.23456789):
    scores = []
    forecast_rows = []
    for index, uid in enumerate(ids):
        scores.append(
            {
                "unique_id": uid,
                "model": "SeasonalNaive",
                "mae": mae,
                "mase": 1.0,
                "relative_mae": 1.0,
                "coverage_80": 0.5,
                "coverage_95": 0.9,
                "scaled_crps": 0.2,
            }
        )
        scores.append(
            {
                "unique_id": uid,
                "model": "AutoETS",
                "mae": mae / 3,
                "mase": 0.4,
                "relative_mae": 1 / 3,
                "coverage_80": 0.8,
                "coverage_95": 0.95,
                "scaled_crps": 0.05,
            }
        )
        forecast_rows.append(
            {
                "unique_id": uid,
                "ds": datetime(2024, 1, 2),
                "yhat": 10.123456789 + index,
                "lo_80": 9.0,
                "hi_80": 11.0,
                "lo_95": 8.0,
                "hi_95": 12.0,
            }
        )
    return ForecastRun(
        pd.DataFrame(forecast_rows),
        "AutoETS",
        datetime(2024, 1, 1),
        BacktestReport(
            pd.DataFrame(scores),
            "AutoETS",
            beats_baseline,
            3,
            1,
        ),
        adjustments,
    )


def _one_series_handle(svc):
    table = pa.table(
        {
            "period": [date(2024, 1, 1)],
            "total_amount": [10.0],
        }
    )
    return svc.store.save(
        table,
        produced_by="run_sql",
        source="one series",
        governed=False,
        elapsed_s=0.0,
    )


def test_forecast_many_series_render_within_budget(synthetic_dataset, tmp_path, monkeypatch):
    svc = DataService(synthetic_dataset, store_root=tmp_path / "store")
    ids = [f"s{i:02d}" for i in range(10)]
    table = pa.table(
        {
            "period": [date(2024, 1, 1)] * len(ids),
            "area": ids,
            "total_amount": [float(i + 1) for i in range(len(ids))],
        }
    )
    card = svc.store.save(
        table,
        produced_by="query_metrics",
        source="ten series",
        governed=True,
        elapsed_s=0.0,
    )

    def fake(series, **kwargs):
        return _fake_run(ids, beats_baseline=True, adjustments=[])

    monkeypatch.setattr("analyst.data.service.run_forecast", fake)
    forecast_card = svc.forecast(
        card.handle_id,
        id_col="area",
        value_col="total_amount",
        freq="D",
        horizon=1,
    )
    rendered = forecast_card.render()
    assert len(rendered) <= 4096
    assert "preview:" in rendered
    summary = forecast_card.meta["backtest_summary"]
    assert summary["best_model"] == "AutoETS"
    assert summary["mean_relative_mae"] == 0.3333
    assert summary["mean_mase"] == 0.4
    assert summary["mean_coverage_80"] == 0.8
    assert summary["mean_coverage_95"] == 0.95
    assert summary["baseline_mean_mase"] == 1.0
    assert "backtest" not in forecast_card.meta
    loaded = svc.store.load(forecast_card.meta["backtest_handle"])
    assert loaded.num_rows == 2 * len(ids)
    assert set(loaded.column("unique_id").to_pylist()) == set(ids)
    backtest_card = svc.store.card(forecast_card.meta["backtest_handle"])
    assert backtest_card.produced_by == "backtest"
    assert backtest_card.governed is True
    assert backtest_card.source == forecast_card.source + ":backtest"
    assert loaded.column("mae").to_pylist()[0] == pytest.approx(1.23456789)


def test_forecast_baseline_note_once(synthetic_dataset, tmp_path, monkeypatch):
    svc = DataService(synthetic_dataset, store_root=tmp_path / "store")
    card = _one_series_handle(svc)
    note = "no model beat the seasonal baseline"

    def fake(series, **kwargs):
        return _fake_run(["series"], beats_baseline=False, adjustments=[])

    monkeypatch.setattr("analyst.data.service.run_forecast", fake)
    forecast_card = svc.forecast(card.handle_id, value_col="total_amount", freq="D", horizon=1)
    assert forecast_card.notes.count(note) == 1


def test_forecast_grouped_without_id_col(synthetic_dataset, tmp_path):
    svc = DataService(synthetic_dataset, store_root=tmp_path / "store")
    card = svc.query_metrics(
        MetricQuery(metrics=["total_amount"], dimensions=["area"], grain="day")
    )
    with pytest.raises(
        ValueError, match=r"handle has multiple rows per period; pass id_col"
    ):
        svc.forecast(card.handle_id, value_col="total_amount", freq="D")


def test_forecast_bad_column(synthetic_dataset, tmp_path):
    svc = DataService(synthetic_dataset, store_root=tmp_path / "store")
    card = svc.query_metrics(
        MetricQuery(metrics=["total_amount"], dimensions=["area"], grain="day")
    )
    with pytest.raises(ValueError, match="not_a_column") as exc:
        svc.forecast(card.handle_id, value_col="not_a_column", freq="D")
    message = str(exc.value)
    for name in ("period", "area", "total_amount"):
        assert name in message
