from datetime import date, datetime

import pytest
from pydantic import ValidationError

from analyst.data.engine import Engine
from analyst.memory.files import load_memory
from analyst.semantic.compiler import FilterSpec, MetricQuery, compile_query
from analyst.semantic.model import Dimension, Filter, Metric
from analyst.semantic.resolve import resolve_values


@pytest.fixture
def model(synthetic_dataset):
    return load_memory(synthetic_dataset).semantic


@pytest.fixture
def engine(synthetic_dataset):
    return Engine(synthetic_dataset)


def run(engine, model, **kwargs):
    return engine.execute(compile_query(model, MetricQuery(**kwargs)))


def test_total_metric(engine, model):
    assert run(engine, model, metrics=["total_amount"]).equals(
        engine.execute("SELECT sum(amount) AS total_amount FROM events")
    )


def test_grain_and_dimension(engine, model):
    actual = run(
        engine, model, metrics=["total_amount"], dimensions=["zone_group"], grain="day"
    )
    expected = engine.execute(
        "SELECT date_trunc('day', ts)::DATE AS period, region.area AS area, "
        "sum(amount) AS total_amount FROM events "
        "LEFT JOIN regions AS region ON events.region_id = region.region_id "
        "GROUP BY 1, 2 ORDER BY 1, 2"
    )
    assert actual.equals(expected)


def test_metric_filter_applied(engine, model):
    # Fixture qty is positive everywhere; use a selective filter to prove isolation.
    model.filters.append(Filter(name="large_qty", expr="qty > 2"))
    model.metrics.append(
        Metric(name="selective", expr="sum(amount)", filters=["large_qty"])
    )
    actual = run(engine, model, metrics=["positive_amount", "selective", "event_count"])
    assert actual.equals(
        engine.execute(
            "SELECT sum(amount) FILTER (WHERE qty > 0) AS positive_amount, "
            "sum(amount) FILTER (WHERE qty > 2) AS selective, count(*) AS event_count FROM events"
        )
    )


def test_filtered_ratio_and_empty_groups(engine, model):
    model.filters.append(Filter(name="subset", expr="qty > 2 AND region_id = 1"))
    model.metrics.extend(
        [
            Metric(
                name="ratio", expr="sum(amount)/nullif(sum(qty),0)", filters=["subset"]
            ),
            Metric(name="empty_count", expr="count(*) + 1", filters=["subset"]),
        ]
    )
    actual = run(
        engine,
        model,
        metrics=["ratio", "empty_count", "event_count"],
        dimensions=["region_id"],
    )
    expected = engine.execute(
        "SELECT region_id, sum(amount) FILTER (WHERE qty > 2 AND region_id = 1) / "
        "nullif(sum(qty) FILTER (WHERE qty > 2 AND region_id = 1),0) AS ratio, "
        "count(*) FILTER (WHERE qty > 2 AND region_id = 1) + 1 AS empty_count, "
        "count(*) AS event_count FROM events GROUP BY 1 ORDER BY 1"
    )
    assert actual.equals(expected)


def test_compare_previous_period(engine, model):
    actual = run(
        engine,
        model,
        metrics=["total_amount"],
        time_range=(date(2024, 2, 1), date(2024, 3, 1)),
        compare_to="previous_period",
    )
    # Month-aligned totals compare with the complete preceding calendar month.
    feb = engine.execute("SELECT sum(amount) FROM events WHERE ts >= '2024-02-01'")[0][
        0
    ].as_py()
    jan = engine.execute(
        "SELECT sum(amount) FROM events WHERE ts >= '2024-01-01' AND ts < '2024-02-01'"
    )[0][0].as_py()
    assert actual.to_pylist() == [
        {
            "total_amount": feb,
            "total_amount_compare": jan,
            "total_amount_change_pct": round((feb - jan) / jan * 100, 2),
        }
    ]


def test_previous_period_month_grain_feb_vs_jan(engine, model):
    query = MetricQuery(
        metrics=["total_amount"],
        grain="month",
        time_range=(date(2024, 2, 1), date(2024, 3, 1)),
        compare_to="previous_period",
    )
    sql = compile_query(model, query)
    assert "INTERVAL 1 MONTH" in sql
    actual = engine.execute(sql).to_pylist()
    jan = engine.execute(
        "SELECT sum(amount) FROM events WHERE ts >= '2024-01-01' AND ts < '2024-02-01'"
    )[0][0].as_py()
    feb = engine.execute("SELECT sum(amount) FROM events WHERE ts >= '2024-02-01'")[0][
        0
    ].as_py()
    assert actual == [
        {
            "period": date(2024, 2, 1),
            "total_amount": feb,
            "total_amount_compare": jan,
            "total_amount_change_pct": round((feb - jan) / jan * 100, 2),
        }
    ]


def test_previous_period_day_grain_month_range_row_count(engine, model):
    result = run(
        engine,
        model,
        metrics=["total_amount"],
        grain="day",
        time_range=(date(2024, 2, 1), date(2024, 3, 1)),
        compare_to="previous_period",
    )
    rows = result.to_pylist()
    assert len(rows) == 29
    assert len({row["period"] for row in rows}) == 29
    january = engine.execute(
        "SELECT ts::DATE AS period, sum(amount) AS amount FROM events "
        "WHERE ts >= DATE '2024-01-01' AND ts < DATE '2024-01-30' "
        "GROUP BY 1 ORDER BY 1"
    ).to_pylist()
    assert rows[0]["period"] == date(2024, 2, 1)
    for current, previous in zip(rows, january, strict=True):
        assert current["period"].day == previous["period"].day
        assert current["total_amount_compare"] == previous["amount"]


def test_previous_period_hour_grain_month_range_row_count(engine, model):
    rows = run(
        engine,
        model,
        metrics=["total_amount"],
        grain="hour",
        time_range=(date(2024, 2, 1), date(2024, 3, 1)),
        compare_to="previous_period",
    ).to_pylist()
    assert len(rows) == 696
    assert len({row["period"] for row in rows}) == 696
    january = {
        row["ts"]: row["amount"]
        for row in engine.execute(
            "SELECT ts, sum(amount) AS amount FROM events "
            "WHERE ts >= DATE '2024-01-01' AND ts < DATE '2024-01-30' GROUP BY ts"
        ).to_pylist()
    }
    for row in rows:
        stamp = row["period"]
        assert row["total_amount_compare"] == january.get(stamp.replace(month=1))
    assert rows[0]["total_amount_compare"] == january[datetime(2024, 1, 1)]


def test_previous_period_positional_comparison_dimensions(engine, model):
    model.dimensions.append(Dimension(name="nullable", expr="NULL::VARCHAR"))
    rows = run(
        engine,
        model,
        metrics=["total_amount"],
        dimensions=["area", "nullable"],
        grain="day",
        time_range=(date(2024, 2, 1), date(2024, 3, 1)),
        compare_to="previous_period",
    ).to_pylist()
    assert len(rows) == 58
    assert len({(row["period"], row["area"], row["nullable"]) for row in rows}) == 58
    expected = {
        (row["day"], row["area"]): row["amount"]
        for row in engine.execute(
            "SELECT day(ts) AS day, region.area AS area, sum(amount) AS amount "
            "FROM events LEFT JOIN regions AS region ON events.region_id = region.region_id "
            "WHERE ts >= DATE '2024-01-01' AND ts < DATE '2024-01-30' GROUP BY 1, 2"
        ).to_pylist()
    }
    for row in rows:
        assert row["total_amount_compare"] == expected.get((row["period"].day, row["area"]))


def test_previous_period_week_grain_aligned(engine, model):
    query = MetricQuery(
        metrics=["event_count"],
        grain="week",
        time_range=(date(2024, 2, 5), date(2024, 2, 19)),
        compare_to="previous_period",
    )
    sql = compile_query(model, query)
    assert "INTERVAL 2 WEEK" in sql
    rows = engine.execute(sql).to_pylist()
    assert len(rows) == 2
    for row, start, end in zip(
        rows, ["2024-01-22", "2024-01-29"], ["2024-01-29", "2024-02-05"], strict=True
    ):
        expected = engine.execute(
            f"SELECT count(*) FROM events WHERE ts >= DATE '{start}' AND ts < DATE '{end}'"
        )[0][0].as_py()
        assert row["event_count_compare"] == expected


@pytest.mark.parametrize(
    "grain,start,end",
    [
        ("month", date(2024, 2, 2), date(2024, 3, 2)),
        ("week", date(2024, 2, 5), date(2024, 2, 15)),
        ("week", date(2024, 2, 1), date(2024, 3, 1)),
    ],
)
def test_previous_period_misaligned_grain_raises(model, grain, start, end):
    with pytest.raises(ValueError, match=f"range must align to whole {grain}s"):
        compile_query(
            model,
            MetricQuery(
                metrics=["event_count"],
                grain=grain,
                time_range=(start, end),
                compare_to="previous_period",
            ),
        )


def test_previous_period_day_fallback(model):
    sql = compile_query(
        model,
        MetricQuery(
            metrics=["event_count"],
            time_range=(date(2024, 2, 2), date(2024, 2, 12)),
            compare_to="previous_period",
        ),
    )
    assert "INTERVAL 10 DAY" in sql


def test_compare_requires_time_range(model):
    with pytest.raises(ValueError, match="time_range"):
        compile_query(
            model, MetricQuery(metrics=["event_count"], compare_to="previous_period")
        )


@pytest.mark.parametrize(
    "grain,interval",
    [
        ("month", "12 MONTH"),
        ("week", "52 WEEK"),
        ("day", "364 DAY"),
        ("hour", "364 DAY"),
    ],
)
def test_yoy_shift(model, grain, interval):
    sql = compile_query(
        model,
        MetricQuery(
            metrics=["event_count"],
            grain=grain,
            time_range=(date(2024, 2, 1), date(2024, 3, 1)),
            compare_to="same_period_last_year",
        ),
    )
    assert f"INTERVAL {interval}" in sql
    assert "cur.period = cmp.period +" in sql


@pytest.mark.parametrize("grain", [None, "day"])
def test_share_sums_to_100(engine, model, grain):
    rows = run(
        engine,
        model,
        metrics=["total_amount"],
        dimensions=["area"],
        grain=grain,
        share=True,
    ).to_pylist()
    totals = {}
    for row in rows:
        period = row.get("period")
        totals[period] = totals.get(period, 0) + row["total_amount_share_pct"]
    assert all(abs(total - 100) <= 0.05 for total in totals.values())


@pytest.mark.parametrize(
    "expr",
    [
        "sum(amount)/nullif(sum(qty),0)",
        "avg(amount)",
        "sum(amount) + count(*)",
        "sum(amount) * 0",
    ],
)
def test_share_rejects_non_additive_metric(model, expr):
    model.metrics.append(Metric(name="non_additive", expr=expr))
    with pytest.raises(ValueError) as error:
        compile_query(
            model, MetricQuery(metrics=["event_count", "non_additive"], share=True)
        )
    assert (
        str(error.value)
        == "share is only defined for additive metrics (sum/count): non_additive"
    )


@pytest.mark.parametrize(
    "expr",
    [
        "SUM(amount)",
        "count(*)",
        "sum(coalesce(amount, 0))",
        "count(DISTINCT region_id)",
    ],
)
def test_share_accepts_single_sum_or_count(engine, model, expr):
    model.metrics.append(Metric(name="additive", expr=expr))
    result = run(engine, model, metrics=["additive"], share=True)
    assert result["additive_share_pct"][0].as_py() == 100


def test_unknown_dimension_raises(model):
    with pytest.raises(KeyError, match="unknown dimension"):
        compile_query(
            model, MetricQuery(metrics=["event_count"], dimensions=["missing"])
        )


def test_proposed_metric_rejected_unless_allowed(engine, model):
    query = MetricQuery(metrics=["draft_metric"])
    with pytest.raises(ValueError, match="proposed"):
        compile_query(model, query)
    assert (
        engine.execute(compile_query(model, query, allow_proposed=True)).num_rows == 1
    )


@pytest.mark.parametrize("item", ["dimension", "join", "filter", "metric"])
def test_governance_dependencies(model, item):
    query = MetricQuery(metrics=["positive_amount"], dimensions=["area"])
    target = {
        "dimension": model.dimensions[0],
        "join": model.joins[0],
        "filter": model.filters[0],
        "metric": model.metrics[2],
    }[item]
    target.status = "rejected"
    with pytest.raises(ValueError, match="rejected"):
        compile_query(model, query)
    compile_query(model, query, allow_proposed=True)


def test_resolve_case_insensitive(engine, model):
    assert resolve_values(engine, model, "zone_group", ["north", " South "]) == [
        "North",
        "South",
    ]


def test_resolve_unknown_suggests(engine, model):
    with pytest.raises(
        ValueError, match="unknown value 'Nrth' for area; did you mean: North"
    ):
        resolve_values(engine, model, "area", ["Nrth"])


def test_resolve_ints(engine, model):
    assert resolve_values(engine, model, "region_id", [1, 4]) == [1, 4]


@pytest.mark.parametrize(
    "op,values,where",
    [
        ("eq", ["North"], "region.area = 'North'"),
        ("neq", ["North"], "region.area <> 'North'"),
        ("in", ["North", "South"], "region.area IN ('North','South')"),
        ("not_in", ["South"], "region.area NOT IN ('South')"),
        ("in", [], "FALSE"),
        ("not_in", [], "TRUE"),
    ],
)
def test_dimension_filters(engine, model, op, values, where):
    actual = run(
        engine,
        model,
        metrics=["event_count"],
        filters=[FilterSpec(dimension="area", op=op, values=values)],
    )
    assert actual.equals(
        engine.execute(
            "SELECT count(*) AS event_count FROM events LEFT JOIN regions AS region "
            f"ON events.region_id = region.region_id WHERE {where}"
        )
    )


@pytest.mark.parametrize("op", ["neq", "not_in"])
def test_negative_filters_keep_null_dimension(synthetic_dataset, model, tmp_path, op):
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Isolated source with an unmatched lookup ID; do not mutate shared fixture data.
    path = tmp_path / "with_unknown_region.parquet"
    pq.write_table(
        pa.table(
            {
                "ts": [datetime(2024, 2, 1)] * 3,
                "region_id": [1, 3, 99],
                "amount": [10.0, 20.0, 30.0],
                "qty": [1, 1, 1],
            }
        ),
        path,
    )
    lookups = [
        lookup.model_copy(update={"file": synthetic_dataset.root / lookup.file})
        for lookup in synthetic_dataset.lookups
    ]
    dataset = synthetic_dataset.model_copy(
        update={"root": tmp_path, "files": path.name, "lookups": lookups}
    )
    engine = Engine(dataset)
    actual = run(
        engine,
        model,
        metrics=["event_count"],
        dimensions=["area"],
        filters=[FilterSpec(dimension="area", op=op, values=["North"])],
    )
    assert actual.to_pylist() == [
        {"area": "South", "event_count": 1},
        {"area": None, "event_count": 1},
    ]


def test_safety_and_join_selection(engine, model):
    sql = compile_query(model, MetricQuery(metrics=["event_count"]))
    assert "JOIN" not in sql
    injected = "North' OR TRUE --"
    assert (
        run(
            engine,
            model,
            metrics=["event_count"],
            filters=[FilterSpec(dimension="area", op="eq", values=[injected])],
        )[0][0].as_py()
        == 0
    )
    for bad in ["event_count DESC; DROP TABLE events", "missing"]:
        with pytest.raises(ValueError, match="order_by"):
            compile_query(model, MetricQuery(metrics=["event_count"], order_by=bad))
    model.metrics.append(Metric(name="bad;name", expr="count(*)"))
    with pytest.raises(ValueError, match="identifier"):
        compile_query(model, MetricQuery(metrics=["bad;name"]))


def test_order_limit_and_output_columns(engine, model):
    result = run(
        engine,
        model,
        metrics=["event_count", "total_amount"],
        dimensions=["area"],
        grain="day",
        time_range=(date(2024, 2, 1), date(2024, 2, 3)),
        compare_to="previous_period",
        share=True,
        order_by="event_count",
        limit=2,
    )
    assert result.column_names == [
        "period",
        "area",
        "event_count",
        "total_amount",
        "event_count_compare",
        "event_count_change_pct",
        "total_amount_compare",
        "total_amount_change_pct",
        "event_count_share_pct",
        "total_amount_share_pct",
    ]
    assert result.num_rows == 2


def test_hour_period_is_timestamp(engine, model):
    result = run(engine, model, metrics=["event_count"], grain="hour", limit=1)
    assert isinstance(result["period"][0].as_py(), datetime)


def test_null_dimension_comparison(engine, model):
    model.dimensions.append(Dimension(name="nullable", expr="NULL::VARCHAR"))
    result = run(
        engine,
        model,
        metrics=["event_count"],
        dimensions=["nullable"],
        time_range=(date(2024, 2, 1), date(2024, 2, 2)),
        compare_to="previous_period",
    )
    assert result["event_count_compare"][0].as_py() == 24


@pytest.mark.parametrize(
    "kwargs",
    [
        {"metrics": []},
        {"metrics": ["event_count"], "limit": -1},
        {
            "metrics": ["event_count"],
            "time_range": (date(2024, 2, 1), date(2024, 1, 1)),
        },
    ],
)
def test_invalid_query(kwargs):
    with pytest.raises((ValueError, ValidationError)):
        MetricQuery(**kwargs)


@pytest.mark.parametrize(
    "grain,start,end,shift",
    [
        ("hour", date(2024, 2, 1), date(2024, 2, 2), "1 DAY"),
        ("day", date(2024, 2, 1), date(2024, 2, 2), "1 DAY"),
        ("week", date(2024, 2, 5), date(2024, 2, 12), "7 DAY"),
    ],
)
def test_comparison_by_period_and_dimension(engine, model, grain, start, end, shift):
    actual = run(
        engine,
        model,
        metrics=["event_count"],
        dimensions=["area"],
        grain=grain,
        time_range=(start, end),
        compare_to="previous_period",
    )
    period = f"date_trunc('{grain}', ts)" + ("::DATE" if grain != "hour" else "")
    source = (
        "FROM events LEFT JOIN regions AS region ON events.region_id = region.region_id"
    )
    expected = engine.execute(
        f"WITH a AS (SELECT {period} AS period, region.area AS area, count(*) AS n "
        f"{source} WHERE ts >= DATE '{start}' AND ts < DATE '{end}' GROUP BY 1, 2), "
        f"b AS (SELECT {period} AS period, region.area AS area, count(*) AS n "
        f"{source} WHERE ts >= DATE '{start}' - INTERVAL {shift} "
        f"AND ts < DATE '{end}' - INTERVAL {shift} GROUP BY 1, 2) "
        "SELECT a.period, a.area, a.n AS event_count, b.n AS event_count_compare, "
        "round((a.n-b.n)/nullif(b.n,0)*100,2) AS event_count_change_pct FROM a "
        f"LEFT JOIN b ON a.period = b.period + INTERVAL {shift} AND a.area = b.area "
        "ORDER BY 1, 2"
    )
    assert actual.equals(expected)
    expected_count = {"hour": 1, "day": 12, "week": 84}[grain]
    assert all(
        value == expected_count for value in actual["event_count_compare"].to_pylist()
    )


@pytest.mark.parametrize("dimensions", [[], ["area"]])
def test_compound_only_and_empty_input(engine, model, dimensions):
    model.metrics.append(
        Metric(
            name="ratio",
            expr="sum(amount)/nullif(sum(qty),0)",
            filters=["positive_qty"],
        )
    )
    result = run(engine, model, metrics=["ratio"], dimensions=dimensions)
    assert result.num_rows == (2 if dimensions else 1)
    assert all(value > 0 for value in result["ratio"].to_pylist())
    empty = run(
        engine,
        model,
        metrics=["ratio"],
        dimensions=dimensions,
        time_range=(date(2025, 1, 1), date(2025, 1, 2)),
    )
    assert empty.to_pylist() == ([] if dimensions else [{"ratio": None}])


def test_duplicate_outputs_rejected(model):
    model.metrics.append(Metric(name="period", expr="count(*)"))
    for query in [
        MetricQuery(metrics=["event_count", "event_count"]),
        MetricQuery(metrics=["period"], grain="day"),
    ]:
        with pytest.raises(ValueError, match="duplicate"):
            compile_query(model, query)


def test_missing_dependencies_and_unsupported_grain(model):
    model.time_dimension.grains = ["day"]
    with pytest.raises(ValueError, match="grain"):
        compile_query(model, MetricQuery(metrics=["event_count"], grain="month"))
    model.metrics[2].filters = ["missing"]
    with pytest.raises(ValueError, match="unknown metric filter"):
        compile_query(model, MetricQuery(metrics=["positive_amount"]))
    model.dimensions[0].join = "missing"
    with pytest.raises(ValueError, match="unknown join"):
        compile_query(model, MetricQuery(metrics=["event_count"], dimensions=["area"]))


def test_filter_arity():
    for op, values in [("eq", []), ("neq", [1, 2])]:
        with pytest.raises(ValueError, match="exactly one"):
            FilterSpec(dimension="region_id", op=op, values=values)


def test_zero_denominators(engine, model):
    model.metrics.append(Metric(name="zero", expr="sum(amount * 0)"))
    result = run(
        engine,
        model,
        metrics=["zero"],
        share=True,
        time_range=(date(2024, 2, 1), date(2024, 2, 2)),
        compare_to="previous_period",
    )
    assert result["zero_change_pct"][0].as_py() is None
    assert result["zero_share_pct"][0].as_py() is None
