"""Series preparation and honest, rolling-origin model selection.

No data access occurs here. MASE is evaluated separately at every cutoff using
only its training prefix, then scores are averaged equally across windows.
"""

import time
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from typing import Literal

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import MSTL, SeasonalNaive
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import coverage, mae, mase, scaled_crps

from analyst.errors import InvalidRequestError
from analyst.semantic.model import Quirk

from .models import DEFAULTS, candidate_models

Freq = Literal["D", "MS"]
ProgressFn = Callable[[str], None]
MIN_BUDGET_WINDOWS = 4


@dataclass
class PreparedSeries:
    df: pd.DataFrame
    adjustments: list[str]
    imputed_points: int = 0


@dataclass
class BacktestReport:
    scores: pd.DataFrame
    best_model: str
    beats_baseline: bool
    windows: int
    horizon: int
    adjustments: list[str] = field(default_factory=list)
    windows_used: int | None = None

    def __post_init__(self):
        if self.windows_used is None:
            self.windows_used = self.windows


@dataclass
class ForecastRun:
    forecast: pd.DataFrame
    model: str
    origin: datetime
    backtest: BacktestReport
    adjustments: list[str]


def _check_freq(freq):
    if freq not in DEFAULTS:
        raise InvalidRequestError(f"unsupported frequency: {freq!r}")


def _record(adjustments, message):
    if message not in adjustments:
        adjustments.append(message)


def _series_record(series, message):
    ids = series.df.unique_id.unique()
    prefix = f"[{', '.join(map(str, ids))}] " if len(ids) > 1 else ""
    _record(series.adjustments, prefix + message)


def _monthly_gaps(group, quirks, adjustments, *, additive: bool):
    """Adjust months overlapping known gaps: scale additive totals, else interpolate."""
    for month in group.index:
        month_end = month + pd.offsets.MonthEnd(0)
        dates = set()
        for quirk in quirks:
            start = max(month, pd.Timestamp(quirk.start))
            end = min(month_end, pd.Timestamp(quirk.end or quirk.start))
            if start <= end:
                dates.update(pd.date_range(start, end, freq="D"))
        if not dates:
            continue
        dates = sorted(dates)
        ranges = []
        start = end = dates[0]
        for day in dates[1:]:
            if day != end + pd.Timedelta(days=1):
                ranges.append(f"{start.date()}..{end.date()}")
                start = day
            end = day
        ranges.append(f"{start.date()}..{end.date()}")
        description = ", ".join(ranges)
        if len(dates) == month.days_in_month or pd.isna(group.loc[month, "y"]):
            group.loc[month, "y"] = np.nan
            adjustments.append(
                f"imputed {month:%Y-%m} (known gap: {len(dates)} days {description})"
            )
        elif additive:
            group.loc[month, "y"] *= month.days_in_month / (
                month.days_in_month - len(dates)
            )
            adjustments.append(
                f"scaled {month:%Y-%m} for {len(dates)} known-gap days ({description})"
            )
        else:
            group.loc[month, "y"] = np.nan
            adjustments.append(
                f"imputed {month:%Y-%m} by interpolation (known gap; metric not additive)"
            )


def prepare_series(
    df: pd.DataFrame,
    *,
    freq: Freq,
    data_end: datetime,
    quirks: Sequence[Quirk] = (),
    id_col: str | None,
    time_col: str,
    value_col: str,
    additive: bool = False,
) -> PreparedSeries:
    """Normalize, complete the calendar, and interpolate each series independently.

    Boundary gaps use the nearest observed value (no extrapolated trend). An
    entirely missing series cannot be imputed and is rejected. Only approved
    impute quirks apply. Additive monthly metrics are scaled by observed-day fraction;
    non-additive metrics use interpolation across known gaps. Whole missing months are
    interpolated. Input is copied.
    """
    _check_freq(freq)
    data_end = pd.Timestamp(data_end)
    if data_end.tzinfo is not None:
        raise InvalidRequestError("data_end must be timezone-naive")
    quirks = [q for q in quirks if q.status == "approved" and q.treatment == "impute"]
    columns = [time_col, value_col] + ([id_col] if id_col else [])
    frame = df[columns].rename(columns={time_col: "ds", value_col: "y"}).copy()
    if id_col:
        frame = frame.rename(columns={id_col: "unique_id"})
    else:
        frame["unique_id"] = "series"
    frame["ds"] = pd.to_datetime(frame["ds"])
    frame["y"] = pd.to_numeric(frame["y"]).astype(float)
    if frame.empty or frame[["unique_id", "ds"]].isna().any().any():
        raise InvalidRequestError(
            "series must have non-null ids and dates and at least one row"
        )
    if frame.duplicated(["unique_id", "ds"]).any():
        raise InvalidRequestError("duplicate dates within a series")
    if np.isinf(frame.y).any():
        raise InvalidRequestError("series values must be finite")
    all_adjustments, frames = [], []
    imputed_points = 0
    multiple_ids = frame.unique_id.nunique() > 1
    for uid, group in frame.groupby("unique_id", sort=False):
        adjustments = []
        group = group.sort_values("ds").set_index("ds")
        last = group.index.max()
        if freq == "MS":
            boundary = last + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=23)
        else:
            boundary = last.normalize() + pd.Timedelta(hours=23)
        if data_end < boundary:
            group = group.iloc[:-1]
            adjustments.append(f"dropped partial period {last.date()}")
        if group.empty:
            raise InvalidRequestError("series too short after dropping partial period")
        full_index = pd.date_range(group.index.min(), group.index.max(), freq=freq)
        if not group.index.isin(full_index).all():
            raise InvalidRequestError(f"dates must be aligned to {freq}")
        missing = full_index.difference(group.index)
        group = group.reindex(full_index)
        unit = "days" if freq == "D" else "months"
        # Describe each contiguous gap rather than implying nonmissing dates
        # between disjoint gaps were also imputed.
        positions = full_index.get_indexer(missing)
        for chunk in np.split(positions, np.flatnonzero(np.diff(positions) > 1) + 1):
            if len(chunk):
                start, end = full_index[chunk[0]], full_index[chunk[-1]]
                adjustments.append(
                    f"imputed {len(chunk)} missing {unit} {start.date()}..{end.date()}"
                )
        existing_missing = group.y.isna() & ~group.index.isin(missing)
        if existing_missing.any():
            dates = group.index[existing_missing]
            adjustments.append(
                f"imputed {len(dates)} missing values {dates.min().date()}..{dates.max().date()}"
            )
        if freq == "MS":
            _monthly_gaps(group, quirks, adjustments, additive=additive)
        else:
            for quirk in quirks:
                start, end = (
                    pd.Timestamp(quirk.start),
                    pd.Timestamp(quirk.end or quirk.start),
                )
                mask = (group.index >= start) & (group.index <= end)
                if mask.any():
                    dates = group.index[mask]
                    group.loc[mask, "y"] = np.nan
                    adjustments.append(
                        f"imputed {mask.sum()} {unit} {dates.min().date()}..{dates.max().date()} (known gap)"
                    )
        imputed_points += int(group.y.isna().sum())
        group["y"] = group.y.interpolate(method="time", limit_direction="both")
        if group.y.isna().any():
            raise InvalidRequestError(f"cannot impute series {uid!r}: no observed values")
        group["unique_id"] = uid
        group.index.name = "ds"
        frames.append(group.reset_index()[["unique_id", "ds", "y"]])
        prefix = f"[{uid}] " if multiple_ids else ""
        all_adjustments.extend(prefix + message for message in adjustments)
    return PreparedSeries(pd.concat(frames, ignore_index=True), all_adjustments, imputed_points)


def _supported_models(series, models, horizon, n_windows):
    """Copy candidates and record history-driven season changes on the series."""
    history = int(series.df.groupby("unique_id").size().min())
    supported = []
    for original in models:
        model = deepcopy(original)
        season = getattr(model, "season_length", None)
        if season is None:
            supported.append(model)
            continue
        seasons = season if isinstance(season, (list, tuple)) else [season]
        valid = [s for s in seasons if history >= 2 * s + horizon * n_windows]
        for s in seasons:
            if s not in valid:
                label = "yearly" if isinstance(model, MSTL) and s == 365 else str(s)
                _series_record(
                    series,
                    f"{model} {label} season skipped (insufficient history)"
                    if label == "yearly"
                    else f"{model} season {label} skipped (insufficient history)",
                )
        if valid:
            model.season_length = (
                valid if isinstance(season, (list, tuple)) else valid[0]
            )
            supported.append(model)
    if not supported:
        raise InvalidRequestError("series too short for any candidate model")
    return supported


def _backtest_models(series, freq, models, horizon, n_windows):
    candidates = list(candidate_models(freq) if models is None else models)
    # Custom model lists do not waive the honest-baseline requirement.
    season = DEFAULTS[freq]["season"]
    baseline = next(
        (
            m
            for m in candidates
            if isinstance(m, SeasonalNaive)
            and str(m) == "SeasonalNaive"
            and m.season_length == season
        ),
        SeasonalNaive(season_length=season),
    )
    candidates = [m for m in candidates if str(m) != "SeasonalNaive"] + [baseline]
    return _supported_models(series, candidates, horizon, n_windows)


def backtest(
    series: PreparedSeries,
    *,
    freq: Freq,
    models=None,
    horizon=None,
    step=None,
    n_windows=None,
    levels=(80, 95),
    n_jobs=1,
    progress: ProgressFn | None = None,
    budget_s: float | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> BacktestReport:
    """Backtest candidates against SeasonalNaive, with per-cutoff MASE scaling.

    Unsupported seasons, window reductions and failed candidates are recorded
    in the returned report's adjustments; the caller's series is not mutated.
    Scaled CRPS approximates CRPS from the interval quantiles (four by default,
    without a median).
    On a zero-error baseline, relative MAE is 1 for zero error and inf otherwise.
    Zero training scale yields MASE 0 for perfect predictions and inf otherwise.
    Each rolling-origin window is fitted on its own truncated series,
    newest first, so ``progress`` can report "backtest window i/n" and
    ``budget_s`` can stop the loop (keeping the most recent windows) once at
    least four windows are done and the budget is spent.
    """
    _check_freq(freq)
    series = PreparedSeries(series.df, [])
    defaults = DEFAULTS[freq]
    horizon = defaults["horizon"] if horizon is None else horizon
    step = defaults["step"] if step is None else step
    requested = defaults["n_windows"] if n_windows is None else n_windows
    if horizon < 1 or step < 1 or requested < 3:
        raise InvalidRequestError(
            "horizon and step must be positive; n_windows must be at least 3"
        )
    history = series.df.groupby("unique_id").size().min()
    # Both the specified conservative budget and actual rolling-CV training
    # requirement must hold, including when callers choose step > horizon.
    windows = requested
    season = defaults["season"]
    while windows >= 3 and (
        history < 2 * season + horizon * windows
        or history < 2 * season + horizon + step * (windows - 1)
    ):
        windows -= 1
    if windows < 3 or pd.isna(history):
        raise InvalidRequestError("series too short for at least 3 backtest windows")
    if windows != requested:
        _series_record(
            series,
            f"backtest windows reduced from {requested} to {windows} (insufficient history)",
        )
    candidates = _backtest_models(series, freq, models, horizon, windows)
    # Reports always carry the contracted 80/95 columns; callers may add levels.
    levels = sorted(set(levels) | {80, 95})
    keys = ["unique_id", "ds", "cutoff", "y"]
    # Isolate model fits so failed candidates never borrow baseline predictions.
    # Baseline first: its failures must propagate, not trigger a fallback.
    ordered = sorted(candidates, key=lambda m: str(m) != "SeasonalNaive")
    per_model: dict[str, list[pd.DataFrame]] = {str(m): [] for m in ordered}
    failed: set[str] = set()
    sizes = series.df.groupby("unique_id", sort=False).cumcount(ascending=False)
    started = clock()
    windows_used = windows
    for i in range(1, windows + 1):
        if progress is not None:
            progress(f"backtest window {i}/{windows}")
        # Newest first, so a budget stop keeps the most recent windows: window i
        # ends step * (i - 1) periods before the series end.
        window_df = series.df.loc[sizes >= step * (i - 1)]
        for model in ordered:
            name = str(model)
            if name in failed:
                continue
            try:
                result = StatsForecast(
                    models=[model], freq=freq, n_jobs=n_jobs
                ).cross_validation(
                    df=window_df,
                    h=horizon,
                    step_size=step,
                    n_windows=1,
                    level=levels,
                )
            except Exception as error:
                if name == "SeasonalNaive":
                    raise
                _fit_failure(series, model, error)
                failed.add(name)
                continue
            per_model[name].append(result)
        elapsed = clock() - started
        if (
            budget_s is not None
            and MIN_BUDGET_WINDOWS <= i < windows
            and elapsed >= budget_s
        ):
            windows_used = i
            _series_record(
                series,
                f"backtest stopped at {i}/{windows} windows (time budget {budget_s:g} s)",
            )
            break
    cv, names = None, []
    for model in ordered:
        name = str(model)
        if name in failed:
            continue
        result = pd.concat(per_model[name], ignore_index=True)
        cv = result if cv is None else cv.merge(result, on=keys, validate="one_to_one")
        names.append(name)
    evaluated = []
    for cutoff, window in cv.groupby("cutoff"):
        # Dropping cutoff explicitly gives evaluate a single window. In
        # particular no scale may include observations beyond this origin.
        train = series.df.loc[series.df.ds <= cutoff]
        scores = evaluate(
            window.drop(columns="cutoff"),
            models=names,
            train_df=train,
            level=levels,
            metrics=[mae, partial(mase, seasonality=season), coverage, scaled_crps],
        )
        # Resolve only the defined 0/0 MASE case; never silently turn other
        # missing metrics into perfect scores.
        perfect = scores.loc[scores.metric == "mae"].set_index("unique_id")
        for name in names:
            mask = (
                (scores.metric == "mase")
                & scores[name].isna()
                & scores.unique_id.map(perfect[name].eq(0))
            )
            scores.loc[mask, name] = 0.0
        if scores[names].isna().any().any():
            raise InvalidRequestError("undefined backtest scores")
        evaluated.append(scores)
    long = pd.concat(evaluated).melt(
        id_vars=["unique_id", "metric"],
        value_vars=names,
        var_name="model",
        value_name="score",
    )
    scores = (
        long.groupby(["unique_id", "model", "metric"], sort=False)
        .score.mean()
        .unstack("metric")
    )
    scores = scores.rename(
        columns={"coverage_level80": "coverage_80", "coverage_level95": "coverage_95"}
    ).reset_index()
    baseline_mae = (
        scores.loc[scores.model == "SeasonalNaive"].set_index("unique_id").mae
    )
    denom = scores.unique_id.map(baseline_mae)
    scores["relative_mae"] = scores.mae.div(denom)
    scores.loc[(denom == 0) & (scores.mae == 0), "relative_mae"] = 1.0
    columns = [
        "unique_id",
        "model",
        "mae",
        "mase",
        "relative_mae",
        "coverage_80",
        "coverage_95",
        "scaled_crps",
    ]
    scores = scores[columns]
    scores.columns.name = None
    return _report(scores, windows, horizon, series.adjustments, windows_used)


def _fit_failure(series, model, error):
    first_line = str(error).splitlines()[0] if str(error) else type(error).__name__
    _series_record(series, f"{model} skipped (fit failed: {first_line})")


def _report(scores, windows, horizon, adjustments, windows_used=None):
    means = scores.groupby("model")[["relative_mae", "mase"]].mean()
    contenders = means.loc[(means.index != "SeasonalNaive") & (means.relative_mae < 1)]
    best = (
        str(contenders.sort_values(["relative_mae", "mase"], kind="stable").index[0])
        if not contenders.empty else "SeasonalNaive"
    )
    beats = best != "SeasonalNaive"
    return BacktestReport(
        scores, best, beats, windows, horizon, list(adjustments), windows_used
    )


def fit_predict(
    series: PreparedSeries,
    *,
    freq: Freq,
    horizon: int,
    models,
    levels=(80, 95),
    n_jobs=1,
) -> pd.DataFrame:
    """Fit the supplied models on all observations without running a backtest."""
    _check_freq(freq)
    if horizon < 1:
        raise InvalidRequestError("horizon must be positive")
    return StatsForecast(
        models=deepcopy(list(models)), freq=freq, n_jobs=n_jobs
    ).forecast(df=series.df, h=horizon, level=list(levels))


def forecast(
    series: PreparedSeries,
    *,
    freq: Freq,
    horizon: int,
    levels=(80, 95),
    models=None,
    n_jobs=1,
    progress: ProgressFn | None = None,
    budget_s: float | None = None,
) -> ForecastRun:
    """Select by mean relative MAE (MASE tie-break), then refit the winner."""
    report = backtest(
        series,
        freq=freq,
        horizon=horizon,
        models=models,
        levels=levels,
        n_jobs=n_jobs,
        progress=progress,
        budget_s=budget_s,
    )
    working = PreparedSeries(series.df, list(report.adjustments))
    candidates = _backtest_models(working, freq, models, horizon, report.windows)
    # At most one refit attempt per surviving candidate; scores from failed
    # refits are removed before selecting the next best model on the same CV.
    while True:
        winner = next(model for model in candidates if str(model) == report.best_model)
        try:
            predictions = fit_predict(
                series,
                freq=freq,
                horizon=horizon,
                models=[winner],
                levels=sorted(set(levels) | {80, 95}),
                n_jobs=n_jobs,
            )
            break
        except Exception as error:
            if report.best_model == "SeasonalNaive":
                raise
            _fit_failure(working, winner, error)
            remaining = report.scores.loc[
                report.scores.model != report.best_model
            ].copy()
            report = _report(
                remaining,
                report.windows,
                horizon,
                working.adjustments,
                report.windows_used,
            )
    adjustments = list(series.adjustments)
    for message in working.adjustments:
        _record(adjustments, message)
    if not report.beats_baseline:
        _record(adjustments, "no model beat the seasonal baseline")
    names = {report.best_model: "yhat"}
    for level in (80, 95):
        for side in ("lo", "hi"):
            names[f"{report.best_model}-{side}-{level}"] = f"{side}_{level}"
    predictions = predictions.rename(columns=names)
    columns = ["unique_id", "ds", "yhat", "lo_80", "hi_80", "lo_95", "hi_95"]
    return ForecastRun(
        predictions[columns],
        report.best_model,
        pd.Timestamp(series.df.ds.max()).to_pydatetime(),
        report,
        adjustments,
    )
