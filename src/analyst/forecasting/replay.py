"""Dataset-independent, engine-enforced cutoff replay of every candidate model."""

import warnings
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from analyst.data.dataset import DatasetConfig
from analyst.data.service import DataService, _metric_additive
from analyst.forecasting.core import PreparedSeries, fit_predict, prepare_series
from analyst.forecasting.models import DEFAULTS, candidate_models
from analyst.semantic.compiler import MetricQuery

SCORE_COLUMNS = ["mae", "mase", "relative_mae", "coverage_80", "coverage_95"]


@dataclass
class ReplayReport:
    rows: pd.DataFrame
    horizons: dict[date, int] = field(default_factory=dict)
    dropped_ids: list[dict] = field(default_factory=list)
    failed_fits: list[dict] = field(default_factory=list)
    adjustments: list[str] = field(default_factory=list)


def cutoff_scores(report: ReplayReport) -> pd.DataFrame:
    """Aggregate equally across ids; incomplete model/cutoff pairs cannot win."""
    records = []
    for cutoff, window in report.rows.groupby("cutoff", sort=True):
        expected_ids = set(window.unique_id)
        for model, group in window.groupby("model", sort=True):
            complete = (
                set(group.unique_id) == expected_ids
                and not group.relative_mae.isna().any()
                and not any(f["cutoff"] == cutoff and f["model"] == model
                            for f in report.failed_fits)
            )
            record = {"cutoff": cutoff, "model": model, "complete": complete}
            for column in SCORE_COLUMNS:
                values = group[column] if column in group else pd.Series(dtype=float)
                record[column] = float(np.mean(values)) if len(values) else np.nan
            # Never silently average away a missing id/fit.
            if not complete:
                record["relative_mae"] = np.nan
            records.append(record)
    return pd.DataFrame(records, columns=["cutoff", "model", "complete", *SCORE_COLUMNS])


def wins(report: ReplayReport) -> dict[str, int]:
    scores = cutoff_scores(report)
    scores["win"] = scores.complete & (scores.relative_mae < 1)
    return scores.groupby("model").win.sum().astype(int).to_dict()


def lineup(report: ReplayReport, *, min_wins: int = 2) -> list[str]:
    """Keep candidates that beat the baseline at enough complete cutoffs."""
    return [name for name, count in wins(report).items()
            if name != "SeasonalNaive" and count >= min_wins] + ["SeasonalNaive"]


def _query_frame(service, query, id_col, value_col):
    card = service.query_metrics(query)
    frame = service.store.load(card.handle_id).to_pandas()
    frame = frame.rename(columns={"period": "ds", value_col: "y"})
    if id_col is None:
        frame["unique_id"] = "series"
    else:
        frame = frame.rename(columns={id_col: "unique_id"})
    frame["ds"] = pd.to_datetime(frame.ds)
    return frame[["unique_id", "ds", "y"]]


def _ratio(error, denominator, *, perfect):
    if denominator == 0:
        return perfect if error == 0 else np.inf
    return error / denominator


def replay(
    dataset: DatasetConfig,
    query: MetricQuery,
    *,
    cutoffs: list[date],
    freq: str,
    horizon: int,
    id_col: str | None,
    value_col: str,
    min_share: float = 0.01,
    n_jobs: int = 1,
    store_root: Path | None = None,
) -> ReplayReport:
    """Fit each id/model in isolation, scoring raw held-out observations only.

    Failed fits retain NaN rows and disqualify that model at the cutoff. MASE
    uses the training seasonal-difference scale. Zero scale gives 0 for a
    perfect forecast and infinity otherwise; zero baseline error gives relative
    MAE 1 for a perfect forecast and infinity otherwise.
    """
    if freq not in DEFAULTS:
        raise ValueError(f"unsupported frequency: {freq!r}")
    if horizon < 1 or not 0 <= min_share <= 1:
        raise ValueError("horizon must be positive and min_share must be in [0, 1]")
    full = DataService(dataset, cutoff=None,
                       store_root=store_root / "full" if store_root is not None else None)
    truth = _query_frame(full, query, id_col, value_col)
    data_end = pd.Timestamp(full.engine.coverage().end)
    rows, dropped_ids, failed_fits = [], [], []
    arima_warnings = 0
    horizons = {}
    season = DEFAULTS[freq]["season"]
    for cutoff in cutoffs:
        service = DataService(dataset, cutoff=cutoff,
                              store_root=store_root / cutoff.isoformat() if store_root is not None else None)
        frame = _query_frame(service, query, id_col, value_col)
        totals = frame.groupby("unique_id").y.sum()
        shares = totals / totals.sum()
        dropped = shares[shares < min_share]
        dropped_ids.extend(
            {"cutoff": cutoff, "unique_id": uid, "share": float(share)}
            for uid, share in dropped.items()
        )
        frame = frame.loc[~frame.unique_id.isin(dropped.index)]
        start = pd.Timestamp(cutoff) + pd.tseries.frequencies.to_offset(freq)
        if freq == "MS":
            start = start.normalize()
        dates = pd.date_range(start, periods=horizon, freq=freq)
        # Match prepare_series' complete-period boundary, without imputing truth.
        ends = (dates + pd.offsets.MonthEnd(0) if freq == "MS" else dates)
        dates = dates[ends + pd.Timedelta(hours=23) <= data_end]
        horizons[cutoff] = len(dates)
        if not len(dates):
            raise ValueError(f"no complete truth periods after cutoff {cutoff}")
        if frame.empty:
            raise ValueError(f"no ids remain at cutoff {cutoff}; reduce min_share")
        prepared = prepare_series(
            frame,
            freq=freq,
            data_end=service.engine.coverage().end,
            quirks=service.memory.quirks,
            id_col="unique_id",
            time_col="ds",
            value_col="y",
            additive=_metric_additive(service.memory, value_col),
        )
        for uid, train in prepared.df.groupby("unique_id", sort=False):
            train = train.sort_values("ds")
            train_end = train.ds.max()
            if train_end >= dates[0]:
                raise ValueError("training overlaps held-out periods")
            fit_horizon = len(pd.date_range(train_end, dates[-1], freq=freq)) - 1
            y = train.y.to_numpy(dtype=float)
            scale = float(np.abs(y[season:] - y[:-season]).mean()) if len(y) > season else np.nan
            actual = truth.loc[truth.unique_id == uid].set_index("ds").y.reindex(dates)
            if not np.isfinite(actual.to_numpy(dtype=float)).all():
                raise ValueError(f"missing/nonfinite held-out truth for {uid!r} at {cutoff}")
            id_rows = []
            for model in candidate_models(freq):
                name = str(model)
                row = {
                    "cutoff": cutoff, "unique_id": uid, "model": name,
                    "train_end": train_end, "horizon": len(dates), "scored_points": 0,
                    "actual_mean": float(actual.mean()), "forecast_mean": np.nan,
                    **dict.fromkeys(SCORE_COLUMNS, np.nan),
                }
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        predictions = fit_predict(
                            PreparedSeries(train, prepared.adjustments), freq=freq,
                            horizon=fit_horizon, models=[model], levels=(80, 95), n_jobs=n_jobs,
                        ).set_index("ds").reindex(dates)
                    for w in caught:
                        if "possible convergence problem" in str(w.message):
                            arima_warnings += 1
                        else:
                            warnings.warn_explicit(
                                w.message, w.category, w.filename, w.lineno
                            )
                    forecast_columns = [name, *[f"{name}-{side}-{level}"
                                               for level in (80, 95) for side in ("lo", "hi")]]
                    if not np.isfinite(predictions[forecast_columns].to_numpy(dtype=float)).all():
                        raise ValueError("missing/nonfinite forecast or interval")
                    error = float(np.abs(actual.to_numpy() - predictions[name].to_numpy()).mean())
                    row["scored_points"] = len(dates)
                    row["forecast_mean"] = float(predictions[name].mean())
                    row["mae"] = error
                    row["mase"] = _ratio(error, scale, perfect=0.0)
                    for level in (80, 95):
                        row[f"coverage_{level}"] = float((
                            (actual >= predictions[f"{name}-lo-{level}"])
                            & (actual <= predictions[f"{name}-hi-{level}"])
                        ).mean())
                except Exception as exc:  # noqa: BLE001 -- isolate third-party model failures
                    failed_fits.append({
                        "cutoff": cutoff, "unique_id": uid, "model": name,
                        "error": str(exc).splitlines()[0] if str(exc) else type(exc).__name__,
                    })
                id_rows.append(row)
            baseline = next(row["mae"] for row in id_rows if row["model"] == "SeasonalNaive")
            for row in id_rows:
                row["relative_mae"] = _ratio(row["mae"], baseline, perfect=1.0)
            rows.extend(id_rows)
    # Suppressed above to keep output readable; reported here as one adjustment.
    adjustments = [f"AutoARIMA: {arima_warnings} convergence warnings"] if arima_warnings else []
    return ReplayReport(pd.DataFrame(rows, columns=[
        "cutoff", "unique_id", "model", *SCORE_COLUMNS, "train_end", "horizon",
        "scored_points", "forecast_mean", "actual_mean",
    ]), horizons=horizons, dropped_ids=dropped_ids, failed_fits=failed_fits,
        adjustments=adjustments)
