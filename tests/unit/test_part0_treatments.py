"""Part 0 data treatments and card polish (deferred from Plan A)."""

from datetime import date, datetime

import pandas as pd
import pytest

from analyst.data.service import DataService
from analyst.forecasting.core import BacktestReport, ForecastRun, prepare_series
from analyst.semantic.compiler import MetricQuery
from analyst.semantic.model import Quirk


@pytest.fixture
def synthetic_service(synthetic_dataset, tmp_path):
    return DataService(synthetic_dataset, store_root=tmp_path)


def _daily_frame(missing: tuple[date, ...] = (date(2024, 1, 3), date(2024, 1, 4))):
    days = pd.date_range("2024-01-01", "2024-01-10", freq="D")
    frame = pd.DataFrame({"ds": days, "y": [float(rank) for rank in range(1, 11)]})
    return frame.loc[~frame.ds.dt.date.isin(missing)].reset_index(drop=True)


def test_additive_empty_days_zero_filled():
    quirk = Quirk(
        kind="gap",
        start=date(2024, 1, 20),
        end=date(2024, 1, 21),
        treatment="impute",
    )
    result = prepare_series(
        _daily_frame(),
        freq="D",
        data_end=datetime(2024, 1, 10, 23),
        quirks=[quirk],
        id_col=None,
        time_col="ds",
        value_col="y",
        additive=True,
    )
    values = result.df.set_index("ds").y
    assert values.loc["2024-01-03"] == 0
    assert values.loc["2024-01-04"] == 0
    assert "zero-filled 2 days with no events" in result.adjustments
    assert not any("imputed 2 missing days" in note for note in result.adjustments)


def test_non_additive_empty_days_still_interpolated():
    result = prepare_series(
        _daily_frame(),
        freq="D",
        data_end=datetime(2024, 1, 10, 23),
        quirks=[],
        id_col=None,
        time_col="ds",
        value_col="y",
        additive=False,
    )
    values = result.df.set_index("ds").y
    assert values.loc["2024-01-03"] == pytest.approx(3.0)
    assert values.loc["2024-01-04"] == pytest.approx(4.0)
    assert any("interpolation" in note for note in result.adjustments)


def test_additive_quirk_day_keeps_imputation():
    days = pd.date_range("2024-01-18", "2024-01-24", freq="D")
    frame = pd.DataFrame({"ds": days, "y": [float(rank) for rank in range(1, 8)]})
    frame = frame.loc[frame.ds != pd.Timestamp("2024-01-20")].reset_index(drop=True)
    quirk = Quirk(
        kind="gap", start=date(2024, 1, 20), end=date(2024, 1, 20), treatment="impute"
    )
    result = prepare_series(
        frame,
        freq="D",
        data_end=datetime(2024, 1, 24, 23),
        quirks=[quirk],
        id_col=None,
        time_col="ds",
        value_col="y",
        additive=True,
    )
    values = result.df.set_index("ds").y
    assert values.loc["2024-01-20"] == pytest.approx(3.0)
    assert not any("zero-filled" in note for note in result.adjustments)
    assert any("(known gap)" in note for note in result.adjustments)


def test_additive_missing_month_zero_filled():
    frame = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", "2024-04-01", freq="MS"),
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )
    frame = frame.loc[frame.ds != pd.Timestamp("2024-03-01")].reset_index(drop=True)
    result = prepare_series(
        frame,
        freq="MS",
        data_end=datetime(2024, 4, 30, 23),
        quirks=[],
        id_col=None,
        time_col="ds",
        value_col="y",
        additive=True,
    )
    assert result.df.set_index("ds").y.loc["2024-03-01"] == 0
    assert "zero-filled 1 month with no events" in result.adjustments


def test_string_value_for_integer_dimension_coerced(synthetic_service):
    svc = synthetic_service

    def query(value):
        return svc.query_metrics(
            MetricQuery(
                metrics=["event_count"],
                dimensions=["region_id"],
                filters=[
                    {"dimension": "region_id", "op": "eq", "values": [value]}
                ],
            )
        )

    as_int = svc.store.load(query(3).handle_id)
    as_text = svc.store.load(query("3").handle_id)
    assert as_int.num_rows == 1
    assert as_text.to_pylist() == as_int.to_pylist()


def test_describe_values_says_showing_n_of_total(synthetic_service):
    assert "showing 50 of" in synthetic_service.describe("dimension:region_slot")


def test_time_range_cutting_period_is_noted(synthetic_service):
    card = synthetic_service.query_metrics(
        MetricQuery(
            metrics=["event_count"],
            grain="month",
            time_range=(date(2024, 1, 1), date(2024, 2, 15)),
        )
    )
    assert "2024-02 is partial: time_range ends 2024-02-15" in card.notes


def test_headline_excludes_small_ids(synthetic_service, monkeypatch):
    svc = synthetic_service
    handle = svc.run_sql(
        """
        SELECT CAST(date_trunc('day', ts) AS DATE) AS period,
               CASE WHEN region_id = 1 AND hour(ts) = 0 THEN 'tiny'
                    ELSE 'big' END AS uid,
               sum(CASE WHEN region_id = 1 AND hour(ts) = 0 THEN 0.01
                        ELSE 1 END) AS y
        FROM main.events
        GROUP BY 1, 2
        """
    ).handle_id

    def fake(series, **kwargs):
        scores = pd.DataFrame(
            [
                {
                    "unique_id": uid,
                    "model": model,
                    "mae": 1.0,
                    "mase": mase,
                    "relative_mae": 0.5,
                    "coverage_80": 0.8,
                    "coverage_95": 0.95,
                    "scaled_crps": 0.1,
                }
                for uid, mase in (("big", 1.0), ("tiny", 1000.0))
                for model in ("AutoETS", "SeasonalNaive")
            ]
        )
        forecast = pd.DataFrame(
            [
                {
                    "unique_id": uid,
                    "ds": datetime(2024, 3, 1),
                    "yhat": 1.0,
                    "lo_80": 0.5,
                    "hi_80": 1.5,
                    "lo_95": 0.2,
                    "hi_95": 1.8,
                }
                for uid in ("big", "tiny")
            ]
        )
        report = BacktestReport(scores, "AutoETS", True, 3, 1, [], 3)
        return ForecastRun(forecast, "AutoETS", datetime(2024, 2, 29), report, [])

    monkeypatch.setattr("analyst.data.service.run_forecast", fake)
    card = svc.forecast(handle, id_col="uid", value_col="y", freq="D", horizon=1)
    assert card.meta["headline_ids"] == ["big"]
    assert card.meta["headline_rule"] == "ids with ≥ 1% share of total"
    # The small id is excluded from the headline means but still listed.
    assert card.meta["backtest_summary"]["mean_mase"] == pytest.approx(1.0)
    assert {row["unique_id"] for row in card.meta["backtest"]} == {"big", "tiny"}
    forecast = svc.store.load(card.handle_id)
    assert set(forecast.column("unique_id").to_pylist()) == {"big", "tiny"}


def test_yoy_without_grain_documented():
    description = MetricQuery.model_fields["compare_to"].description
    assert "same_period_last_year" in description
    assert "one year earlier" in description
    assert "one total" in description


def test_resolve_uses_public_compiler_helpers():
    from analyst.semantic import resolve

    assert resolve.approved.__module__ == "analyst.semantic.compiler"
