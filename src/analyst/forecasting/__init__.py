"""Dataset-agnostic, backtested forecasting core."""

from .core import (
    BacktestReport,
    ForecastRun,
    PreparedSeries,
    backtest,
    fit_predict,
    forecast,
    prepare_series,
)
from .models import DEFAULTS, candidate_models

__all__ = [
    "DEFAULTS",
    "BacktestReport",
    "ForecastRun",
    "PreparedSeries",
    "backtest",
    "candidate_models",
    "fit_predict",
    "forecast",
    "prepare_series",
]
