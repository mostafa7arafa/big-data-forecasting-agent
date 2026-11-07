from datetime import date

import pytest

from analyst.data.dataset import load_dataset
from analyst.data.engine import Engine
from analyst.memory.files import load_memory
from analyst.semantic.compiler import FilterSpec, MetricQuery, compile_query

pytestmark = pytest.mark.taxi


@pytest.fixture(scope="module")
def taxi():
    dataset = load_dataset("nyc_taxi")
    # Include Q4 2023 for previous-quarter comparisons and 2024/2025 for YoY.
    dataset = dataset.model_copy(
        update={"files": "data/raw/yellow_tripdata_202[345]-*.parquet"}
    )
    return Engine(dataset), load_memory(dataset).semantic


def test_monthly_trips_by_borough_2024(taxi):
    engine, model = taxi
    query = MetricQuery(
        metrics=["trips"],
        dimensions=["pickup_borough"],
        grain="month",
        time_range=(date(2024, 1, 1), date(2025, 1, 1)),
    )
    actual = engine.execute(compile_query(model, query))
    expected = engine.execute(
        "SELECT date_trunc('month', tpep_pickup_datetime)::DATE AS period, "
        "pickup_zone.Borough AS pickup_borough, count(*) AS trips FROM trips "
        "LEFT JOIN zones AS pickup_zone ON trips.PULocationID = pickup_zone.LocationID "
        "WHERE tpep_pickup_datetime >= DATE '2024-01-01' "
        "AND tpep_pickup_datetime < DATE '2025-01-01' GROUP BY 1, 2 ORDER BY 1, 2"
    )
    assert actual.equals(expected)
    assert len(set(actual["period"].to_pylist())) == 12


def test_yoy_manhattan_2025_months(taxi):
    engine, model = taxi
    query = MetricQuery(
        metrics=["trips"],
        dimensions=["pickup_borough"],
        grain="month",
        filters=[FilterSpec(dimension="pickup_borough", op="eq", values=["Manhattan"])],
        time_range=(date(2025, 3, 1), date(2025, 4, 1)),
        compare_to="same_period_last_year",
    )
    actual = engine.execute(compile_query(model, query)).to_pylist()
    expected = engine.execute(
        "SELECT count(*) FROM trips JOIN zones ON trips.PULocationID = zones.LocationID "
        "WHERE zones.Borough = 'Manhattan' AND tpep_pickup_datetime >= DATE '2024-03-01' "
        "AND tpep_pickup_datetime < DATE '2024-04-01'"
    )[0][0].as_py()
    assert len(actual) == 1
    assert actual[0]["period"] == date(2025, 3, 1)
    assert actual[0]["trips_compare"] == expected
    assert expected > 0


def test_previous_period_quarter_vs_prev_quarter_month_grain(taxi):
    engine, model = taxi
    query = MetricQuery(
        metrics=["trips"],
        grain="month",
        time_range=(date(2024, 1, 1), date(2024, 4, 1)),
        compare_to="previous_period",
    )
    sql = compile_query(model, query)
    assert "INTERVAL 3 MONTH" in sql
    actual = engine.execute(sql).to_pylist()
    expected = engine.execute(
        "SELECT date_trunc('month', tpep_pickup_datetime)::DATE AS period, count(*) AS trips "
        "FROM trips WHERE tpep_pickup_datetime >= DATE '2023-10-01' "
        "AND tpep_pickup_datetime < DATE '2024-01-01' GROUP BY 1 ORDER BY 1"
    ).to_pylist()
    assert [row["period"] for row in actual] == [
        date(2024, month, 1) for month in (1, 2, 3)
    ]
    assert [row["period"] for row in expected] == [
        date(2023, month, 1) for month in (10, 11, 12)
    ]
    for current, previous in zip(actual, expected, strict=True):
        assert current["trips_compare"] == previous["trips"]
        assert current["trips_compare"] > 0


def test_previous_period_day_grain_march_vs_february(taxi):
    engine, model = taxi
    query = MetricQuery(
        metrics=["trips"],
        grain="day",
        time_range=(date(2024, 3, 1), date(2024, 4, 1)),
        compare_to="previous_period",
    )
    sql = compile_query(model, query)
    # March's result alone also passes with the old clamp-prone calendar join;
    # assert the positional join so this guards the intended strategy as well.
    assert "cur.period -" in sql and "cmp.period -" in sql
    rows = engine.execute(sql).to_pylist()
    assert len(rows) == 31
    assert len({row["period"] for row in rows}) == 31
    february = engine.execute(
        "SELECT tpep_pickup_datetime::DATE AS period, count(*) AS trips "
        "FROM trips WHERE tpep_pickup_datetime >= DATE '2024-02-01' "
        "AND tpep_pickup_datetime < DATE '2024-03-01' GROUP BY 1 ORDER BY 1"
    ).to_pylist()
    assert len(february) == 29
    for current, previous in zip(rows[:29], february, strict=True):
        assert current["period"].day == previous["period"].day
        assert current["trips_compare"] == previous["trips"]
        assert current["trips_compare"] > 0
    assert [row["period"] for row in rows[29:]] == [
        date(2024, 3, 30),
        date(2024, 3, 31),
    ]
    assert all(row["trips_compare"] is None for row in rows[29:])
    assert all(row["trips_change_pct"] is None for row in rows[29:])


def test_filtered_tip_rate(taxi):
    engine, model = taxi
    query = MetricQuery(
        metrics=["tip_rate", "trips"],
        dimensions=["pickup_borough"],
        grain="month",
        time_range=(date(2024, 3, 1), date(2024, 4, 1)),
    )
    actual = engine.execute(compile_query(model, query))
    expected = engine.execute(
        "SELECT date_trunc('month', tpep_pickup_datetime)::DATE AS period, "
        "pickup_zone.Borough AS pickup_borough, "
        "sum(tip_amount) FILTER (WHERE fare_amount > 0 AND trip_distance > 0) / "
        "nullif(sum(fare_amount) FILTER (WHERE fare_amount > 0 AND trip_distance > 0),0) AS tip_rate, "
        "count(*) AS trips FROM trips LEFT JOIN zones AS pickup_zone "
        "ON trips.PULocationID = pickup_zone.LocationID "
        "WHERE tpep_pickup_datetime >= DATE '2024-03-01' "
        "AND tpep_pickup_datetime < DATE '2024-04-01' GROUP BY 1, 2 ORDER BY 1, 2"
    )
    assert actual.column_names == expected.column_names
    for got, want in zip(actual.to_pylist(), expected.to_pylist(), strict=True):
        assert got["period"] == want["period"]
        assert got["pickup_borough"] == want["pickup_borough"]
        assert got["trips"] == want["trips"]
        assert got["tip_rate"] == pytest.approx(want["tip_rate"])
