"""Forecast cache, windowed backtest, progress, time budget and card meta."""

import itertools
import threading
import time
from datetime import date, datetime

import pandas as pd
import pytest

from analyst.data.service import DataService
from analyst.errors import InvalidRequestError
from analyst.forecasting.core import BacktestReport, ForecastRun, backtest
from analyst.semantic.compiler import MetricQuery
from tests.unit.test_forecasting import daily, fast_models


@pytest.fixture
def synthetic_service(synthetic_dataset, tmp_path):
    return DataService(synthetic_dataset, store_root=tmp_path)


def _daily_handle(svc):
    return svc.query_metrics(
        MetricQuery(metrics=["event_count"], dimensions=[], grain="day")
    ).handle_id


# Captured from the batched cross_validation implementation (pre-windowing):
# backtest(daily(n=210, ids=("a", "b"), noise=True), freq="D", models=fast_models())
_BATCHED_EXPECTED = {
    ("a", "SeasonalNaive"): (5.079253054716325, 1.0358211035077756, 1.0, 0.8622448979591837, 0.01890674392760961),
    ("a", "AutoETS"): (3.842765874083774, 0.7831137906583142, 0.756561217306467, 0.6989795918367347, 0.010876721032933124),
    ("b", "SeasonalNaive"): (5.888092742497906, 1.064298503579016, 1.0, 0.9438775510204083, 0.0200114494284554),
    ("b", "AutoETS"): (3.9816528565617832, 0.7197568514464213, 0.676221151855133, 0.8061224489795918, 0.01204511069221057),
}


def test_windowed_backtest_matches_batched():
    report = backtest(
        daily(n=210, ids=("a", "b"), noise=True), freq="D", models=fast_models()
    )
    assert report.windows == 7 and report.windows_used == 7
    cols = ["mae", "mase", "relative_mae", "coverage_80", "scaled_crps"]
    got = report.scores.set_index(["unique_id", "model"])
    assert set(got.index) == set(_BATCHED_EXPECTED)
    for key, expected in _BATCHED_EXPECTED.items():
        for column, value in zip(cols, expected, strict=True):
            assert got.loc[key, column] == pytest.approx(value, abs=1e-9), (key, column)
    assert report.best_model == "AutoETS"


def test_progress_reports_each_window():
    msgs = []
    backtest(daily(n=400), freq="D", models=fast_models(), progress=msgs.append)
    assert msgs[0] == "backtest window 1/12"
    assert msgs[-1].startswith("backtest window 12/12")
    assert len(msgs) == 12


def test_budget_stops_after_min_windows():
    ticks = itertools.count()
    report = backtest(
        daily(n=400),
        freq="D",
        models=fast_models(),
        budget_s=100,
        clock=lambda: 30.0 * next(ticks),
    )
    assert report.windows_used == 4
    assert report.windows == 12
    assert "backtest stopped at 4/12 windows (time budget 100 s)" in report.adjustments
    assert set(report.scores.model) == {"AutoETS", "SeasonalNaive"}


def test_budget_not_exceeded_runs_all_windows():
    report = backtest(
        daily(n=400), freq="D", models=fast_models(), budget_s=1e9
    )
    assert report.windows_used == 12
    assert not any("time budget" in a for a in report.adjustments)


def _fake_run(*, windows_used=3):
    scores = pd.DataFrame(
        [
            {"unique_id": "x", "model": m, "mae": 1.0, "mase": 1.0, "relative_mae": r,
             "coverage_80": 0.8, "coverage_95": 0.95, "scaled_crps": 0.1}
            for m, r in (("SeasonalNaive", 1.0), ("AutoETS", 0.5))
        ]
    )
    frame = pd.DataFrame(
        [{"unique_id": "x", "ds": datetime(2024, 3, 1), "yhat": 1.0, "lo_80": 0.5,
          "hi_80": 1.5, "lo_95": 0.2, "hi_95": 1.8}]
    )
    report = BacktestReport(scores, "AutoETS", True, 3, 1, [], windows_used)
    return ForecastRun(frame, "AutoETS", datetime(2024, 2, 29), report, [])


def test_forecast_cached_second_call(synthetic_service):
    svc = synthetic_service
    h = _daily_handle(svc)
    kw = {"value_col": "event_count", "freq": "D", "horizon": 7}
    assert not svc.forecast_is_cached(h, id_col=None, **kw)
    a = svc.forecast(h, **kw)
    t = time.perf_counter()
    b = svc.forecast(h, **kw)
    assert a.handle_id == b.handle_id and time.perf_counter() - t < 0.5
    assert svc.forecast_is_cached(h, id_col=None, **kw)


def test_concurrent_forecast_same_inputs_single_compute(synthetic_service, monkeypatch):
    svc = synthetic_service
    h = _daily_handle(svc)
    calls = []

    def fake(series, **kwargs):
        calls.append(1)
        time.sleep(0.3)
        return _fake_run()

    monkeypatch.setattr("analyst.data.service.run_forecast", fake)
    results = [None, None]

    def work(i):
        results[i] = svc.forecast(h, value_col="event_count", freq="D", horizon=7)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results[0].handle_id == results[1].handle_id
    assert len(calls) == 1


def test_forecast_meta_has_cutoff_and_data_end(synthetic_dataset, tmp_path, monkeypatch):
    svc = DataService(synthetic_dataset, cutoff=date(2024, 2, 15), store_root=tmp_path)
    monkeypatch.setattr(
        "analyst.data.service.run_forecast", lambda series, **kw: _fake_run(windows_used=4)
    )
    card = svc.forecast(_daily_handle(svc), value_col="event_count", freq="D", horizon=7)
    assert card.meta["cutoff"] == "2024-02-15"
    assert card.meta["data_end"].startswith("2024-02-15")
    assert card.meta["windows_used"] == 4
    full = DataService(synthetic_dataset, store_root=tmp_path / "full")
    card = full.forecast(_daily_handle(full), value_col="event_count", freq="D", horizon=7)
    assert card.meta["cutoff"] is None and card.meta["data_end"]


def test_levels_other_than_80_95_rejected(synthetic_service):
    svc = synthetic_service
    with pytest.raises(InvalidRequestError, match="levels"):
        svc.forecast(
            _daily_handle(svc), value_col="event_count", freq="D", levels=(50, 90)
        )


def test_recommended_models_drive_candidates(synthetic_service, monkeypatch):
    svc = synthetic_service
    h = _daily_handle(svc)
    seen = {}

    def fake(series, **kwargs):
        seen["models"] = [str(m) for m in kwargs["models"]] if kwargs.get("models") else None
        return _fake_run()

    monkeypatch.setattr("analyst.data.service.run_forecast", fake)
    spec = svc.memory.series[0].model_copy(
        update={"recommended_models": ["AutoETS"], "recommended_models_status": "proposed"}
    )
    svc.memory = svc.memory.model_copy(update={"series": [spec]})
    card = svc.forecast(h, value_col="event_count", freq="D", horizon=7)
    assert card.meta["candidates"] == ["AutoETS", "SeasonalNaive"]
    assert card.meta["candidates_source"] == "series:daily_events (proposed)"
    assert seen["models"] == ["AutoETS", "SeasonalNaive"]
    # Unmatched freq/metric -> default candidates.
    svc.memory = svc.memory.model_copy(update={"series": []})
    card = svc.forecast(h, value_col="event_count", freq="D", horizon=6)
    assert card.meta["candidates_source"] == "default"
    assert "AutoARIMA" in card.meta["candidates"]


def test_estimate_forecast_s_monotonic(synthetic_service):
    svc = synthetic_service
    one = svc.estimate_forecast_s(_daily_handle(svc), id_col=None, freq="D")
    by_area = svc.query_metrics(
        MetricQuery(metrics=["event_count"], dimensions=["region_id"], grain="day")
    ).handle_id
    many = svc.estimate_forecast_s(by_area, id_col="region_id", freq="D")
    assert 0 < one < many
