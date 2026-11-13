"""Cutoff replay: honest training, paired scores, and conservative lineups."""

import runpy
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from analyst.forecasting.models import candidate_models
from analyst.forecasting.replay import ReplayReport, lineup, replay
from analyst.semantic.compiler import MetricQuery

CUTOFFS = [date(2024, 2, 10), date(2024, 2, 17)]
QUERY = MetricQuery(metrics=["total_amount"], dimensions=["area"], grain="day")


@pytest.fixture(scope="module")
def daily_report(synthetic_dataset, tmp_path_factory):
    return replay(
        synthetic_dataset, QUERY, cutoffs=CUTOFFS, freq="D", horizon=7,
        id_col="area", value_col="total_amount", n_jobs=1,
        store_root=tmp_path_factory.mktemp("replay"),
    )


def test_replay_scores_all_models_each_cutoff(daily_report):
    rows = daily_report.rows
    names = {str(model) for model in candidate_models("D")}
    assert len(rows) == len(CUTOFFS) * 2 * len(names)
    assert set(rows.model) == names
    assert not rows.duplicated(["cutoff", "unique_id", "model"]).any()
    baseline = rows.loc[rows.model == "SeasonalNaive"]
    assert baseline.mae.notna().all()
    assert (baseline.relative_mae == 1).all()
    assert {"mase", "coverage_80", "coverage_95"} <= set(rows.columns)


def test_truth_not_visible_to_forecast(synthetic_dataset, monkeypatch, tmp_path):
    import analyst.forecasting.replay as module

    original = module.fit_predict
    seen = []

    def checked(series, **kwargs):
        seen.append(series.df.ds.max())
        assert series.df.ds.max() < pd.Timestamp(CUTOFFS[0]) + pd.Timedelta(days=1)
        assert kwargs["n_jobs"] == 1
        return original(series, **kwargs)

    monkeypatch.setattr(module, "fit_predict", checked)
    report = replay(
        synthetic_dataset, QUERY, cutoffs=CUTOFFS[:1], freq="D", horizon=7,
        id_col="area", value_col="total_amount", n_jobs=1, store_root=tmp_path,
    )
    assert seen
    assert (report.rows.train_end < pd.Timestamp(CUTOFFS[0]) + pd.Timedelta(days=1)).all()


def test_lineup_rule():
    rows = pd.DataFrame([
        {"cutoff": cutoff, "unique_id": "x", "model": model, "relative_mae": score}
        for cutoff, scores in enumerate([(0.8, 1.2), (0.9, 0.9), (1.1, 1.2)])
        for model, score in zip(["A", "B", "SeasonalNaive"], [*scores, 1.0], strict=True)
    ])
    assert lineup(ReplayReport(rows)) == ["A", "SeasonalNaive"]


def test_missing_id_prevents_cutoff_win():
    rows = pd.DataFrame([
        {"cutoff": 0, "unique_id": uid, "model": model, "relative_mae": score}
        for uid, model, score in [
            ("x", "A", 0.2), ("y", "A", np.nan),
            ("x", "SeasonalNaive", 1.0), ("y", "SeasonalNaive", 1.0),
        ]
    ])
    assert lineup(ReplayReport(rows), min_wins=1) == ["SeasonalNaive"]
    # An absent row must not let the surviving id silently win either.
    assert lineup(ReplayReport(rows.drop(index=1)), min_wins=1) == ["SeasonalNaive"]


def test_failed_fit_and_min_share(synthetic_dataset, monkeypatch, tmp_path):
    import analyst.forecasting.replay as module

    def fake(series, *, horizon, models, **kwargs):
        name = str(models[0])
        if name == "AutoETS":
            raise ValueError("deliberate fit failure")
        dates = pd.date_range(series.df.ds.max(), periods=horizon + 1, freq="D")[1:]
        result = pd.DataFrame({"unique_id": series.df.unique_id.iloc[0], "ds": dates})
        result[name] = 0.0
        for level in (80, 95):
            result[f"{name}-lo-{level}"] = -1e9
            result[f"{name}-hi-{level}"] = 1e9
        return result

    monkeypatch.setattr(module, "fit_predict", fake)
    report = replay(
        synthetic_dataset, QUERY, cutoffs=CUTOFFS[:1], freq="D", horizon=7,
        id_col="area", value_col="total_amount", min_share=0.5, store_root=tmp_path,
    )
    assert len(report.dropped_ids) == 1
    assert report.dropped_ids[0]["unique_id"] == "North"
    assert len(report.failed_fits) == 1
    failed = report.rows.loc[report.rows.model == "AutoETS"]
    assert failed.mae.isna().all()
    assert (failed.scored_points == 0).all()
    assert failed.forecast_mean.isna().all()
    assert "AutoETS" not in lineup(report, min_wins=1)
    assert (report.rows.loc[report.rows.model != "AutoETS", "coverage_80"] == 1).all()


def test_scores_use_training_scale_and_raw_truth(synthetic_dataset, monkeypatch, tmp_path):
    import analyst.forecasting.replay as module
    from analyst.data.service import DataService

    training = []

    def fake(series, *, horizon, models, **kwargs):
        training.append(series.df.copy())
        name = str(models[0])
        dates = pd.date_range(series.df.ds.max(), periods=horizon + 1, freq="D")[1:]
        result = pd.DataFrame({"unique_id": series.df.unique_id.iloc[0], "ds": dates})
        result[name] = 0.0 if name == "SeasonalNaive" else 100.0
        for level in (80, 95):
            result[f"{name}-lo-{level}"] = -1e9
            result[f"{name}-hi-{level}"] = 1e9
        return result

    monkeypatch.setattr(module, "fit_predict", fake)
    report = replay(
        synthetic_dataset, QUERY, cutoffs=CUTOFFS[:1], freq="D", horizon=7,
        id_col="area", value_col="total_amount", store_root=tmp_path / "replay",
    )
    svc = DataService(synthetic_dataset, store_root=tmp_path / "truth")
    truth = svc.store.load(svc.query_metrics(QUERY).handle_id).to_pandas()
    truth["period"] = pd.to_datetime(truth.period)
    truth = truth.loc[(truth.period > pd.Timestamp(CUTOFFS[0])) &
                      (truth.period <= pd.Timestamp(CUTOFFS[0]) + pd.Timedelta(days=7))]
    for uid, group in truth.groupby("area"):
        train = next(frame for frame in training if frame.unique_id.iloc[0] == uid)
        scale = np.abs(train.y.to_numpy()[7:] - train.y.to_numpy()[:-7]).mean()
        for row in report.rows.loc[report.rows.unique_id == uid].itertuples():
            expected = np.abs(group.total_amount - (0 if row.model == "SeasonalNaive" else 100)).mean()
            assert row.mae == pytest.approx(expected)
            assert row.mase == pytest.approx(expected / scale)
            assert row.relative_mae == pytest.approx(expected / group.total_amount.mean())
            assert row.scored_points == 7
            assert row.actual_mean == pytest.approx(group.total_amount.mean())
            assert row.forecast_mean == (0 if row.model == "SeasonalNaive" else 100)


def test_horizon_capped_by_truth(synthetic_dataset, monkeypatch, tmp_path):
    import analyst.forecasting.replay as module

    monkeypatch.setattr(module, "candidate_models", lambda freq: candidate_models(freq)[-2:-1])
    report = replay(
        synthetic_dataset, QUERY, cutoffs=[date(2024, 2, 27)], freq="D", horizon=7,
        id_col="area", value_col="total_amount", store_root=tmp_path,
    )
    assert report.horizons == {date(2024, 2, 27): 2}
    assert (report.rows.horizon == 2).all()


def test_monthly_july_horizon_is_two(synthetic_dataset, monkeypatch, tmp_path):
    import analyst.forecasting.replay as module

    cutoff = date(2026, 7, 31)

    def service(dataset, *, cutoff=None, store_root=None):
        end = datetime(2026, 9, 30, 23) if cutoff is None else datetime(2026, 7, 31, 23)
        return SimpleNamespace(
            engine=SimpleNamespace(coverage=lambda: SimpleNamespace(end=end)),
            memory=SimpleNamespace(quirks=[]), cutoff=cutoff,
        )

    def frame(service, *args):
        dates = pd.date_range("2023-01-01", "2026-09-01" if service.cutoff is None
                              else "2026-07-01", freq="MS")
        return pd.DataFrame({"unique_id": "x", "ds": dates, "y": np.arange(len(dates)) + 100})

    monkeypatch.setattr(module, "DataService", service)
    monkeypatch.setattr(module, "_query_frame", frame)
    monkeypatch.setattr(module, "candidate_models", lambda freq: candidate_models(freq)[-2:-1])
    report = replay(
        synthetic_dataset, MetricQuery(metrics=["total_amount"], grain="month"),
        cutoffs=[cutoff], freq="MS", horizon=3, id_col=None, value_col="total_amount", store_root=tmp_path,
    )
    assert report.horizons == {cutoff: 2}
    assert not report.failed_fits
    assert report.rows.mae.iloc[0] == pytest.approx(12.0)


def test_script_publishes_report_and_preserves_other_series(synthetic_dataset, tmp_path, monkeypatch):
    import shutil
    import sys

    import yaml

    script = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/replay_models.py"))
    main = script["main"]
    root = tmp_path / "dataset"
    shutil.copytree(synthetic_dataset.dir, root)
    dataset = synthetic_dataset.model_copy(update={"root": tmp_path, "dir": root})
    path = dataset.memory_dir / "series.yaml"
    mapping = yaml.safe_load(path.read_text())
    mapping["series"].append({**mapping["series"][0], "name": "untouched",
                              "recommended_models": ["X"], "status": "approved"})
    mapping["extra"] = {"keep": True}
    path.write_text(yaml.safe_dump(mapping, sort_keys=False))
    original_other = mapping["series"][1].copy()
    report = ReplayReport(pd.DataFrame([
        {"cutoff": cutoff, "unique_id": "series", "model": model,
         "mae": score, "mase": score, "relative_mae": score,
         "coverage_80": 0.8, "coverage_95": 0.95,
         "train_end": pd.Timestamp(cutoff), "scored_points": 28,
         "forecast_mean": 120.0, "actual_mean": 100.0}
        for cutoff in script["CUTOFFS"] for model, score in [("A", 0.9), ("SeasonalNaive", 1.0)]
    ]), horizons=dict.fromkeys(script["CUTOFFS"], 28))
    monkeypatch.setitem(main.__globals__, "load_dataset", lambda name: dataset)
    monkeypatch.setitem(main.__globals__, "replay", lambda *args, **kwargs: report)
    monkeypatch.setattr(sys, "argv", ["replay_models.py", "--dataset", "events", "--series", "daily_events"])
    main()
    updated = yaml.safe_load(path.read_text())
    assert updated["series"][1] == original_other
    assert updated["extra"] == mapping["extra"]
    selected = updated["series"][0]
    assert selected["recommended_models"] == ["A", "SeasonalNaive"]
    assert selected.get("source") == mapping["series"][0].get("source")
    assert selected.get("status") == mapping["series"][0].get("status")
    assert selected["recommended_models_status"] == "proposed"
    from analyst.semantic.model import SeriesSpec

    restored = SeriesSpec.model_validate(selected)
    assert restored.recommended_models_status == "proposed"
    assert restored.status == "approved"
    text = (tmp_path / "docs/results/model_replay_daily_events.md").read_text()
    assert "0.900" in text
    assert "Daily evaluation: 1 series × 28 points per cutoff." in text
    assert "SeasonalNaive mean 120.000 vs actual mean 100.000 (+20.000%)" in text
    for heading in ["Run date:", "Dataset: events", "Horizon: 28", "Wins per model",
                    "Dropped ids", "Failed fits", "Final lineup", "Wall-clock time:"]:
        assert heading in text


def test_markdown_states_sample_sizes_origins_and_caveats(daily_report):
    script = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/replay_models.py"))
    text = script["render_report"](
        daily_report, dataset="events", series="daily_amount", elapsed=1.0,
        min_share=0.01, min_wins=1,
    )
    assert "## How to read this" in text
    assert "Daily evaluation: 2 series × 7 points per cutoff." in text
    assert "single forecast origin" in text
    assert "last 7 days" in text
    assert "MASE > 1" in text
    assert "indicative, not conclusive" in text
    assert "≥ 1 of 2 cutoffs" in text
    for cutoff in CUTOFFS:
        section = text.split(f"## Cutoff {cutoff}")[1].split("\n## ")[0]
        assert "Retained ids: 2 (North, South)" in section
        assert "Points scored per id: 7" in section
        assert "Total scored points: 14" in section
        assert f"| North | 7 | {cutoff} |" in section
        assert f"| South | 7 | {cutoff} |" in section
        assert "Baseline note: SeasonalNaive mean" in section
        assert "vs actual mean" in section


def test_baseline_note_is_computed_from_predictions_and_truth():
    script = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/replay_models.py"))
    report = ReplayReport(pd.DataFrame([
        {"cutoff": CUTOFFS[0], "unique_id": uid, "model": "SeasonalNaive",
         "relative_mae": 1.0, "train_end": pd.Timestamp(CUTOFFS[0]),
         "scored_points": points, "forecast_mean": predicted, "actual_mean": actual}
        for uid, points, predicted, actual in [("x", 2, 200.0, 100.0), ("y", 4, 50.0, 100.0)]
    ]), horizons={CUTOFFS[0]: 4})
    text = script["render_report"](
        report, dataset="events", series="test", elapsed=1.0, min_share=0.01,
    )
    assert "SeasonalNaive mean 100.000 vs actual mean 100.000 (+0.000%)" in text
    assert "Total scored points: 6" in text


def test_missing_truth_is_checked_before_fitting(synthetic_dataset, monkeypatch, tmp_path):
    import analyst.forecasting.replay as module

    original = module._query_frame

    def missing(service, *args):
        frame = original(service, *args)
        if service.engine.cutoff is None:
            frame.loc[frame.ds == pd.Timestamp("2024-02-11"), "y"] = np.nan
        return frame

    def unexpected_fit(*args, **kwargs):
        pytest.fail("missing truth should be checked before any model fits")

    monkeypatch.setattr(module, "_query_frame", missing)
    monkeypatch.setattr(module, "fit_predict", unexpected_fit)
    with pytest.raises(ValueError, match="missing/nonfinite held-out truth"):
        replay(synthetic_dataset, QUERY, cutoffs=CUTOFFS[:1], freq="D", horizon=7,
               id_col="area", value_col="total_amount", store_root=tmp_path)


def test_convergence_warnings_reported_as_one_adjustment(synthetic_dataset, monkeypatch, tmp_path):
    import warnings

    import analyst.forecasting.replay as module

    noisy = [True]

    def fake(series, *, horizon, models, **kwargs):
        name = str(models[0])
        if name == "AutoARIMA" and noisy[0]:
            warnings.warn("possible convergence problem: minimize did not converge", UserWarning)
        dates = pd.date_range(series.df.ds.max(), periods=horizon + 1, freq="D")[1:]
        result = pd.DataFrame({"unique_id": series.df.unique_id.iloc[0], "ds": dates})
        result[name] = 0.0
        for level in (80, 95):
            result[f"{name}-lo-{level}"] = -1e9
            result[f"{name}-hi-{level}"] = 1e9
        return result

    monkeypatch.setattr(module, "fit_predict", fake)
    report = replay(
        synthetic_dataset, QUERY, cutoffs=CUTOFFS, freq="D", horizon=7,
        id_col="area", value_col="total_amount", store_root=tmp_path,
    )
    # one AutoARIMA fit per id (2) per cutoff (2)
    assert report.adjustments == ["AutoARIMA: 4 convergence warnings"]
    noisy[0] = False
    clean = replay(
        synthetic_dataset, QUERY, cutoffs=CUTOFFS[:1], freq="D", horizon=7,
        id_col="area", value_col="total_amount", store_root=tmp_path / "clean",
    )
    assert clean.adjustments == []
