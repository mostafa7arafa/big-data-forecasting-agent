"""Forecasting public seams from the Task 8 brief; all data is synthetic."""

from copy import deepcopy
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
import pytest
from statsforecast.models import MSTL, AutoETS, HistoricAverage, SeasonalNaive

from analyst.forecasting import (
    DEFAULTS,
    PreparedSeries,
    backtest,
    candidate_models,
    fit_predict,
    forecast,
    prepare_series,
)
from analyst.semantic.model import Quirk


def daily(n=210, ids=("a",), noise=False):
    rng = np.random.default_rng(42)
    frames = []
    for i, uid in enumerate(ids):
        t = np.arange(n)
        y = 100 + 0.5 * t + 8 * np.sin(2 * np.pi * t / 7) + i * 40
        if noise:
            y = 100 + rng.normal(0, 5, n)
        frames.append(
            pd.DataFrame(
                {
                    "unique_id": uid,
                    "ds": pd.date_range("2024-01-01", periods=n, freq="D"),
                    "y": y,
                }
            )
        )
    return PreparedSeries(pd.concat(frames, ignore_index=True), [])


def fast_models():
    return [AutoETS(season_length=7), SeasonalNaive(season_length=7)]


@pytest.fixture(scope="module")
def run():
    # Exercise the full default candidate set, including the shortened MSTL.
    return forecast(daily(), freq="D", horizon=28)


def test_candidates_daily_include_mstl_and_baseline():
    models = candidate_models("D")
    assert [str(m) for m in models] == [
        "AutoETS",
        "AutoARIMA",
        "MSTL",
        "SeasonalNaive",
        "HistoricAverage",
    ]
    assert [m.season_length for m in models[:-1]] == [7, 7, [7, 365], 7]
    assert models[2].trend_forecaster.model == "ZZN"
    assert DEFAULTS == {
        "D": {"horizon": 28, "step": 7, "n_windows": 12, "season": 7},
        "MS": {"horizon": 3, "step": 1, "n_windows": 12, "season": 12},
    }


def test_arima_convergence_warning_is_filtered():
    import warnings

    with warnings.catch_warnings(record=True) as recorded:
        warnings.warn_explicit(
            "possible convergence problem: minimize gave code 2]", UserWarning,
            filename="statsforecast/arima.py", lineno=479, module="statsforecast.arima",
        )
    assert not recorded


def test_monthly_candidates():
    models = candidate_models("MS")
    assert [str(m) for m in models] == [
        "AutoETS",
        "AutoARIMA",
        "SeasonalNaive",
        "HistoricAverage",
    ]
    assert [m.season_length for m in models[:-1]] == [12, 12, 12]


def test_interval_ordering(run):
    df = run.forecast
    assert list(df) == ["unique_id", "ds", "yhat", "lo_80", "hi_80", "lo_95", "hi_95"]
    for lower, upper in zip(
        ["lo_95", "lo_80", "yhat", "hi_80"], ["lo_80", "yhat", "hi_80", "hi_95"]
    ):
        assert (df[lower] <= df[upper]).all()


def test_horizon_length(run):
    assert run.forecast.groupby("unique_id").size().to_dict() == {"a": 28}
    assert run.origin == datetime(2024, 7, 28)
    assert run.forecast.ds.min() == pd.Timestamp("2024-07-29")
    assert run.model == run.backtest.best_model
    assert "MSTL yearly season skipped (insufficient history)" in run.adjustments


def test_baseline_relative_mae_is_one(run):
    scores = run.backtest.scores
    assert set(scores) == {
        "unique_id",
        "model",
        "mae",
        "mase",
        "relative_mae",
        "coverage_80",
        "coverage_95",
        "scaled_crps",
    }
    assert (scores.loc[scores.model == "SeasonalNaive", "relative_mae"] == 1).all()
    assert scores.mase.notna().all()
    assert scores.coverage_80.between(0, 1).all()
    assert scores.coverage_95.between(0, 1).all()
    assert (scores.scaled_crps >= 0).all()


def test_trend_series_beats_baseline(run):
    assert run.backtest.beats_baseline is True
    assert run.backtest.best_model != "SeasonalNaive"


def test_white_noise_reports_not_beating_baseline_honestly():
    report = backtest(daily(noise=True), freq="D", models=fast_models())
    assert isinstance(report.beats_baseline, bool)
    assert "SeasonalNaive" in set(report.scores.model)


def test_gap_quirk_imputed_and_reported():
    df = daily(40).df
    df.loc[20:23, "y"] = 0.001
    quirk = Quirk(
        kind="gap", start=date(2024, 1, 21), end=date(2024, 1, 24), treatment="impute"
    )
    result = prepare_series(
        df,
        freq="D",
        data_end=datetime(2024, 2, 9, 23),
        quirks=[quirk],
        id_col="unique_id",
        time_col="ds",
        value_col="y",
    )
    neighbors = df.loc[[19, 24], "y"].mean()
    assert result.df.loc[20:23, "y"].between(neighbors * 0.8, neighbors * 1.2).all()
    assert "imputed 4 days 2024-01-21..2024-01-24 (known gap)" in result.adjustments
    assert df.loc[20, "y"] == 0.001  # caller data is not changed


def test_missing_dates_reindexed():
    df = pd.DataFrame(
        {"date": pd.date_range("2024-01-01", periods=5), "value": [1, 2, 3, 4, 5]}
    )
    result = prepare_series(
        df.drop(index=[1, 2]),
        freq="D",
        data_end=datetime(2024, 1, 5, 23),
        id_col=None,
        time_col="date",
        value_col="value",
    )
    assert list(result.df) == ["unique_id", "ds", "y"]
    assert result.df.y.tolist() == [1, 2, 3, 4, 5]
    assert result.df.unique_id.nunique() == 1
    assert "imputed 2 missing days 2024-01-02..2024-01-03" in result.adjustments


def test_partial_last_month_dropped():
    df = pd.DataFrame(
        {"ds": pd.date_range("2024-01-01", periods=7, freq="MS"), "y": range(7)}
    )
    result = prepare_series(
        df,
        freq="MS",
        data_end=datetime(2024, 7, 15),
        id_col=None,
        time_col="ds",
        value_col="y",
    )
    assert result.df.ds.max() == pd.Timestamp("2024-06-01")
    assert any("partial" in a and "2024-07" in a for a in result.adjustments)


@pytest.mark.parametrize(
    "freq,end,expected",
    [
        ("D", datetime(2024, 2, 29, 22), "2024-02-28"),
        ("D", datetime(2024, 2, 29, 23), "2024-02-29"),
        ("MS", datetime(2024, 2, 29, 22), "2024-01-01"),
        ("MS", datetime(2024, 2, 29, 23), "2024-02-01"),
    ],
)
def test_partial_period_boundary(freq, end, expected):
    df = pd.DataFrame({"ds": pd.date_range("2024-01-01", end.date(), freq=freq)})
    df["y"] = 1.0
    result = prepare_series(
        df, freq=freq, data_end=end, id_col=None, time_col="ds", value_col="y"
    )
    assert result.df.ds.max() == pd.Timestamp(expected)


def test_multi_series():
    result = forecast(daily(ids=("a", "b")), freq="D", horizon=28, models=fast_models())
    assert result.forecast.groupby("unique_id").size().to_dict() == {"a": 28, "b": 28}
    assert set(result.backtest.scores.unique_id) == {"a", "b"}
    baseline = result.backtest.scores.query("model == 'SeasonalNaive'")
    assert baseline.relative_mae.tolist() == [1, 1]


def test_too_short_raises():
    with pytest.raises(ValueError, match="series too short"):
        backtest(daily(60), freq="D", models=fast_models())


def test_windows_reduced_to_history():
    report = backtest(daily(100), freq="D", models=[SeasonalNaive(7)])
    assert report.windows == 3
    assert report.horizon == 28
    assert report.beats_baseline is False


def test_mase_uses_each_cutoff_training_scale():
    # y=t²: lag-7 scale changes at every cutoff. Expected errors are derived
    # directly from the seasonal-naive definition, not from evaluation helpers.
    n, h, step, windows = 100, 7, 7, 3
    df = daily(n).df
    df["y"] = np.arange(n, dtype=float) ** 2
    report = backtest(
        PreparedSeries(df, []),
        freq="D",
        models=[SeasonalNaive(7)],
        horizon=h,
        step=step,
        n_windows=windows,
    )
    errors, scaled = [], []
    for train_size in (79, 86, 93):
        # mean(t²-(t-7)²) over t=7..train_size-1 = 7*train_size-7
        scale = 7 * train_size - 7
        error = np.mean(
            [(t**2 - (t - 7) ** 2) for t in range(train_size, train_size + h)]
        )
        errors.append(error)
        scaled.append(error / scale)
    row = report.scores.iloc[0]
    assert row.mae == pytest.approx(np.mean(errors))
    assert row.mase == pytest.approx(np.mean(scaled))


def test_fit_predict_has_no_backtest_and_keeps_all_model_columns():
    df = fit_predict(
        daily(40), freq="D", horizon=5, models=[SeasonalNaive(7), HistoricAverage()]
    )
    assert len(df) == 5
    assert {
        "SeasonalNaive",
        "HistoricAverage",
        "SeasonalNaive-lo-80",
        "HistoricAverage-hi-95",
    } <= set(df)


def test_custom_candidates_still_compare_to_baseline():
    report = backtest(daily(100), freq="D", models=[HistoricAverage()])
    assert set(report.scores.model) == {"HistoricAverage", "SeasonalNaive"}


def test_monthly_backtest_and_forecast():
    t = np.arange(75)
    df = pd.DataFrame(
        {
            "unique_id": "m",
            "ds": pd.date_range("2018-01-01", periods=75, freq="MS"),
            "y": 100 + t + 10 * np.sin(2 * np.pi * t / 12),
        }
    )
    result = forecast(
        PreparedSeries(df, []),
        freq="MS",
        horizon=3,
        models=[AutoETS(12), SeasonalNaive(12)],
    )
    assert len(result.forecast) == 3
    assert result.backtest.windows == 12
    assert result.forecast.ds.min() == pd.Timestamp("2024-04-01")


@pytest.mark.parametrize("zero", [False, True])
def test_perfect_baseline_has_honest_degenerate_scores(zero):
    series = daily(100)
    series.df["y"] = 0.0 if zero else np.arange(100) % 7 + 1.0
    report = backtest(series, freq="D", models=[HistoricAverage()])
    baseline = report.scores.query("model == 'SeasonalNaive'").iloc[0]
    assert baseline.mae == 0
    assert baseline.mase == 0
    assert baseline.relative_mae == 1
    assert report.best_model == "SeasonalNaive"
    assert report.beats_baseline is False
    if not zero:
        other = report.scores.query("model == 'HistoricAverage'").iloc[0]
        assert np.isinf(other.mase)
        assert np.isinf(other.relative_mae)


def test_other_unsupported_seasons_are_skipped_without_mutating_candidates():
    series = daily(100)
    model = MSTL(season_length=[7, 30], trend_forecaster=AutoETS(model="ZZN"))
    report = backtest(series, freq="D", models=[model])
    assert "MSTL" in set(report.scores.model)
    assert "MSTL season 30 skipped (insufficient history)" in report.adjustments
    assert series.adjustments == []
    assert model.season_length == [7, 30]


def monthly_frame():
    return pd.DataFrame(
        {
            "ds": pd.date_range("2023-08-01", periods=3, freq="MS"),
            "y": [100.0, 87.0, 100.0],
        }
    )


def prepare_monthly(df, quirks):
    return prepare_series(
        df,
        freq="MS",
        data_end=datetime(2023, 10, 31, 23),
        id_col=None,
        time_col="ds",
        value_col="y",
        quirks=quirks,
    )


def test_monthly_gap_quirk_scaled_pro_rata():
    quirk = Quirk(
        kind="gap", start=date(2023, 9, 21), end=date(2023, 9, 24), treatment="impute"
    )
    result = prepare_monthly(monthly_frame(), [quirk])
    assert result.df.loc[1, "y"] == pytest.approx(87 * 30 / 26)
    assert (
        "scaled 2023-09 for 4 known-gap days (2023-09-21..2023-09-24)"
        in result.adjustments
    )


def test_monthly_whole_gap_interpolated():
    quirk = Quirk(
        kind="gap", start=date(2023, 9, 1), end=date(2023, 9, 30), treatment="impute"
    )
    result = prepare_monthly(monthly_frame(), [quirk])
    assert result.df.loc[1, "y"] == 100
    assert any("imputed" in a and "known gap" in a for a in result.adjustments)


def test_monthly_overlapping_quirks_count_each_gap_day_once():
    quirks = [
        Quirk(
            kind="gap",
            start=date(2023, 9, 21),
            end=date(2023, 9, 23),
            treatment="impute",
        ),
        Quirk(
            kind="gap",
            start=date(2023, 9, 23),
            end=date(2023, 9, 24),
            treatment="impute",
        ),
    ]
    result = prepare_monthly(monthly_frame(), quirks)
    assert result.df.loc[1, "y"] == pytest.approx(87 * 30 / 26)


def test_monthly_gap_spanning_month_boundary():
    quirk = Quirk(
        kind="gap", start=date(2023, 8, 30), end=date(2023, 9, 2), treatment="impute"
    )
    result = prepare_monthly(monthly_frame(), [quirk])
    assert result.df.loc[0, "y"] == pytest.approx(100 * 31 / 29)
    assert result.df.loc[1, "y"] == pytest.approx(87 * 30 / 28)
    assert result.df.loc[2, "y"] == 100


@pytest.mark.parametrize("status", ["proposed", "rejected"])
@pytest.mark.parametrize("freq", ["D", "MS"])
def test_unapproved_impute_quirks_ignored(status, freq):
    df = monthly_frame() if freq == "MS" else daily(40).df
    quirk = Quirk(
        kind="gap",
        start=date(2023, 9, 1) if freq == "MS" else date(2024, 1, 21),
        treatment="impute",
        status=status,
    )
    result = prepare_series(
        df,
        freq=freq,
        data_end=datetime(2023, 10, 31, 23)
        if freq == "MS"
        else datetime(2024, 2, 9, 23),
        quirks=[quirk],
        id_col=None,
        time_col="ds",
        value_col="y",
    )
    assert result.df.y.tolist() == df.y.tolist()
    assert result.adjustments == []


def test_multi_series_adjustments_are_attributed():
    df = daily(40, ids=("Queens", "Bronx")).df
    quirk = Quirk(kind="gap", start=date(2024, 1, 21), treatment="impute")
    result = prepare_series(
        df,
        freq="D",
        data_end=datetime(2024, 2, 9, 23),
        id_col="unique_id",
        time_col="ds",
        value_col="y",
        quirks=[quirk],
    )
    assert len(result.adjustments) == 2
    assert result.adjustments[0].startswith("[Queens] imputed")
    assert result.adjustments[1].startswith("[Bronx] imputed")


def test_backtest_adjustments_do_not_mutate_caller_and_forecast_merges_them():
    series = daily(100)
    series.adjustments.append("original preparation notice")
    report = backtest(series, freq="D", models=[SeasonalNaive(7)])
    assert series.adjustments == ["original preparation notice"]
    assert any("windows reduced" in a for a in report.adjustments)
    result = forecast(series, freq="D", horizon=28, models=[SeasonalNaive(7)])
    assert "original preparation notice" in result.adjustments
    assert any("windows reduced" in a for a in result.adjustments)
    assert series.adjustments == ["original preparation notice"]


def test_timezone_aware_data_end_rejected_clearly():
    with pytest.raises(ValueError, match="data_end must be timezone-naive"):
        prepare_series(
            daily(40).df,
            freq="D",
            data_end=datetime(2024, 2, 9, 23, tzinfo=UTC),
            id_col="unique_id",
            time_col="ds",
            value_col="y",
        )


def test_daily_arima_uses_approximation_with_default_stepwise():
    arima = candidate_models("D")[1]
    assert arima.approximation is True
    assert arima.stepwise is True
    assert candidate_models("MS")[1].approximation is False


class FailingModel:
    """Minimal StatsForecast protocol; refit_only allows CV to succeed first."""

    uses_exog = False
    prediction_intervals = None

    def __init__(self, refit_only=False):
        self.refit_only = refit_only

    def __repr__(self):
        return "FailingModel"

    def new(self):
        return deepcopy(self)

    def fit(self, y, X=None):
        if not self.refit_only or len(y) == 100:
            raise RuntimeError("deliberate fit failure\nnot part of the adjustment")
        return self

    def forecast(self, y, h, X=None, X_future=None, level=None, fitted=False):
        self.fit(y, X)
        return AutoETS(7).forecast(
            y=y, h=h, X=X, X_future=X_future, level=level, fitted=fitted
        )


def test_failing_backtest_candidate_skipped_and_reported():
    series = daily(100)
    report = backtest(series, freq="D", models=[FailingModel()])
    assert set(report.scores.model) == {"SeasonalNaive"}
    assert (
        "FailingModel skipped (fit failed: deliberate fit failure)"
        in report.adjustments
    )
    assert series.adjustments == []
    result = forecast(series, freq="D", horizon=28, models=[FailingModel()])
    assert result.model == "SeasonalNaive"
    assert (
        "FailingModel skipped (fit failed: deliberate fit failure)"
        in result.adjustments
    )


def test_winning_candidate_refit_failure_reselects_remaining_models():
    series = daily(100)
    models = [FailingModel(refit_only=True)]
    initial = backtest(series, freq="D", models=models)
    assert initial.best_model == "FailingModel"
    result = forecast(series, freq="D", horizon=28, models=models)
    assert result.model == result.backtest.best_model == "SeasonalNaive"
    assert result.backtest.beats_baseline is False
    assert set(result.backtest.scores.model) == {"SeasonalNaive"}
    assert (
        "FailingModel skipped (fit failed: deliberate fit failure)"
        in result.adjustments
    )
    assert len(result.forecast) == 28


class FailingBaseline(SeasonalNaive):
    def forecast(self, y, h, X=None, X_future=None, level=None, fitted=False):
        raise RuntimeError("baseline fit failure")


def test_baseline_fit_failure_propagates():
    with pytest.raises(RuntimeError, match="baseline fit failure"):
        backtest(daily(100), freq="D", models=[FailingBaseline(7)])


class RefitFailingBaseline(SeasonalNaive):
    def forecast(self, y, h, X=None, X_future=None, level=None, fitted=False):
        if len(y) == 100:
            raise RuntimeError("baseline refit failure")
        return super().forecast(
            y=y, h=h, X=X, X_future=X_future, level=level, fitted=fitted
        )


def test_baseline_refit_failure_propagates():
    with pytest.raises(RuntimeError, match="baseline refit failure"):
        forecast(daily(100), freq="D", horizon=28, models=[RefitFailingBaseline(7)])


@pytest.mark.parametrize(
    "relative,mase_scores,ets_relative,ets_mase,expected",
    [
        ([0.5, 0.5], [2.0, 2.0], 0.8, 0.1, "HistoricAverage"),
        ([1.1, 1.1], [0.1, 0.1], 1.2, 0.2, "SeasonalNaive"),
        ([1.0, 1.0], [0.1, 0.1], 1.0, 0.2, "SeasonalNaive"),
        ([0.5, 0.5], [2.0, 2.0], 0.5, 0.1, "AutoETS"),
        ([0.1, 1.5], [2.0, 2.0], 0.7, 3.0, "AutoETS"),
    ],
)
def test_selection_uses_relative_mae_not_mase(monkeypatch, relative, mase_scores, ets_relative, ets_mase, expected):
    from analyst.forecasting import core

    def evaluated(window, **kwargs):
        return pd.DataFrame([
            {"unique_id": uid, "metric": metric, "SeasonalNaive": baseline,
             "HistoricAverage": candidate,
             "AutoETS": 10 * ets_relative if metric == "mae" else ets_mase if metric == "mase" else baseline}
            for uid, ratio, scale in zip(["a", "b"], relative, mase_scores, strict=True)
            for metric, baseline, candidate in [
                ("mae", 10.0, 10.0 * ratio), ("mase", 1.0, scale),
                ("coverage_level80", 0.8, 0.8), ("coverage_level95", 0.95, 0.95),
                ("scaled_crps", 0.2, 0.2),
            ]
        ])

    monkeypatch.setattr(core, "evaluate", evaluated)
    result = forecast(daily(100, ids=("a", "b")), freq="D", horizon=28, models=[HistoricAverage(), AutoETS(7)])
    assert result.model == result.backtest.best_model == expected
    assert result.backtest.beats_baseline == (expected != "SeasonalNaive")
    if expected == "SeasonalNaive":
        assert "no model beat the seasonal baseline" in result.adjustments


def test_only_impute_quirks_are_changed():
    df = daily(40).df
    quirks = [
        Quirk(kind="gap", start=date(2024, 1, 21), treatment="flag"),
        Quirk(kind="anomaly", start=date(2024, 1, 22), treatment="impute"),
    ]
    result = prepare_series(
        df,
        freq="D",
        data_end=datetime(2024, 2, 9, 23),
        quirks=quirks,
        id_col="unique_id",
        time_col="ds",
        value_col="y",
    )
    assert result.df.loc[20, "y"] == df.loc[20, "y"]
    assert result.df.loc[21, "y"] == pytest.approx(
        (df.loc[20, "y"] + df.loc[22, "y"]) / 2
    )
    assert len(result.adjustments) == 1
