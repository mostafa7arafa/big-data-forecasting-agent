"""Dataset-independent candidate models and evaluation defaults."""

from typing import Literal

from statsforecast.models import (
    MSTL,
    AutoARIMA,
    AutoETS,
    HistoricAverage,
    SeasonalNaive,
)

DEFAULTS = {
    "D": {"horizon": 28, "step": 7, "n_windows": 12, "season": 7},
    "MS": {"horizon": 3, "step": 1, "n_windows": 12, "season": 12},
}


def candidate_models(freq: Literal["D", "MS"]) -> list:
    """Return fresh model instances, including an explicit seasonal baseline."""
    if freq not in DEFAULTS:
        raise ValueError(f"unsupported frequency: {freq!r}")
    season = DEFAULTS[freq]["season"]
    models = [
        AutoETS(season_length=season),
        AutoARIMA(season_length=season, approximation=freq == "D"),
    ]
    if freq == "D":
        models.append(
            MSTL(season_length=[7, 365], trend_forecaster=AutoETS(model="ZZN"))
        )
    return [*models, SeasonalNaive(season_length=season), HistoricAverage()]
