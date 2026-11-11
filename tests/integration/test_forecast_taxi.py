"""Hero forecast: monthly trips by borough, as of 2026-07-31."""

from datetime import date, datetime

import pytest

from analyst.data.dataset import load_dataset
from analyst.data.service import DataService
from analyst.semantic.compiler import FilterSpec, MetricQuery

pytestmark = [pytest.mark.taxi, pytest.mark.slow]


def _as_date(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return value


def test_hero_monthly_by_borough_cutoff(tmp_path):
    svc = DataService(
        load_dataset("nyc_taxi"),
        cutoff=date(2026, 7, 31),
        store_root=tmp_path / "store",
    )
    card = svc.query_metrics(
        MetricQuery(
            metrics=["trips"],
            dimensions=["pickup_borough"],
            grain="month",
            filters=[
                FilterSpec(
                    dimension="pickup_borough",
                    op="in",
                    values=["Manhattan", "Queens", "Brooklyn"],
                )
            ],
        )
    )
    forecast_card = svc.forecast(
        card.handle_id,
        id_col="pickup_borough",
        value_col="trips",
        freq="MS",
        horizon=2,
    )
    assert forecast_card.rows == 6
    table = svc.store.load(forecast_card.handle_id)
    assert {_as_date(value) for value in table.column("ds").to_pylist()} == {
        date(2026, 8, 1),
        date(2026, 9, 1),
    }
    assert forecast_card.meta["backtest"]
    assert forecast_card.meta["input_handle"] == card.handle_id
    assert forecast_card.meta["freq"] == "MS"
    assert forecast_card.meta["horizon"] == 2
    assert forecast_card.produced_by == "forecast"
    assert forecast_card.governed is card.governed
