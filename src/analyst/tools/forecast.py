from typing import Literal

from langchain.tools import ToolRuntime

from analyst.config import get_settings
from analyst.context import AnalystContext
from analyst.errors import ForecastTooSlowError
from analyst.tools._common import (
    Result,
    card_result,
    data_tool,
    progress,
    run_tool,
    service_for,
)


@data_tool("forecast")
def forecast(
    handle_id: str,
    value_col: str,
    freq: Literal["D", "MS"],
    runtime: ToolRuntime[AnalystContext],
    id_col: str | None = None,
    horizon: int | None = None,
    time_col: str = "period",
) -> Result:
    """Forecast a column of a result handle with backtested model selection.

    Always compares against SeasonalNaive; origin is the end of the data.
    Slow, uncached forecasts must run as a background task for the worker.
    """

    def body() -> Result:
        service = service_for(runtime)
        settings = get_settings()
        if runtime.context.background and not service.forecast_is_cached(
            handle_id,
            value_col=value_col,
            id_col=id_col,
            freq=freq,
            horizon=horizon,
            time_col=time_col,
        ):
            estimate = service.estimate_forecast_s(handle_id, id_col=id_col, freq=freq)
            if estimate > settings.inline_forecast_limit_s:
                raise ForecastTooSlowError(
                    f"forecast estimated at {estimate:.0f}s, over the "
                    f"{settings.inline_forecast_limit_s:.0f}s inline limit",
                    hint=(
                        "call start_async_task for the worker subagent and pass "
                        "the same forecast arguments"
                    ),
                )
        card = service.forecast(
            handle_id,
            time_col=time_col,
            value_col=value_col,
            id_col=id_col,
            freq=freq,
            horizon=horizon,
            progress=lambda message: progress(runtime, "forecast", message),
            budget_s=settings.forecast_budget_s,
        )
        return card_result(card)

    return run_tool("forecast", runtime, body)
