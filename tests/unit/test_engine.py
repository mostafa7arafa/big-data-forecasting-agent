import json
from datetime import date, datetime
from unittest.mock import Mock

import pyarrow as pa
import pytest

from analyst.data.engine import Coverage, Engine, QueryError


def test_view_filters_valid_range(synthetic_dataset):
    engine = Engine(synthetic_dataset)
    for table, expected in (("events", 0), ("events_raw", 3)):
        result = engine.execute(
            f"select count(*) as n from {table} where ts < '2024-01-01'"
        )
        assert isinstance(result, pa.Table)
        assert result["n"][0].as_py() == expected


def test_cutoff_hides_later_rows(synthetic_dataset):
    engine = Engine(synthetic_dataset, cutoff=date(2024, 1, 31))
    coverage = engine.coverage()
    assert engine.cutoff == date(2024, 1, 31)
    assert coverage == Coverage(datetime(2024, 1, 1), datetime(2024, 1, 31, 23))
    assert engine.coverage() is coverage
    assert coverage.end < datetime(2024, 2, 1)
    assert engine.execute("select max(ts) from events_raw")[0][0].as_py() < datetime(
        2024, 2, 1
    )
    assert engine.execute("select count(*) from regions")[0][0].as_py() == 4


def test_union_by_name_schema_drift(synthetic_dataset):
    assert ("note", "VARCHAR") in Engine(synthetic_dataset).columns()


@pytest.mark.parametrize(
    "sql",
    [
        "DROP VIEW events",
        "CREATE TABLE x AS SELECT 1",
        "COPY (SELECT 1) TO 'x.csv'",
        "SELECT 1; SELECT 2",
        "SET threads=1",
        "INSTALL httpfs",
        "ATTACH 'x.db'",
        "",
        "-- just a comment",
    ],
)
def test_rejects_non_select(synthetic_dataset, sql):
    with pytest.raises(QueryError):
        Engine(synthetic_dataset).execute(sql)


def test_blocks_other_files(synthetic_dataset):
    with pytest.raises(QueryError):
        Engine(synthetic_dataset).execute("select * from read_csv('/etc/passwd')")


def test_row_cap(synthetic_dataset):
    engine = Engine(synthetic_dataset, max_rows=10)
    with pytest.raises(QueryError, match="aggregate further"):
        engine.execute("select * from events")
    assert engine.execute("select * from events limit 10").num_rows == 10


def test_timeout(synthetic_dataset):
    engine = Engine(synthetic_dataset, timeout_s=0.5)
    with pytest.raises(QueryError, match="timeout"):
        engine.execute("select count(*) from range(10_000_000_000)")
    assert engine.execute("select 1")[0][0].as_py() == 1


def test_error_is_one_line(synthetic_dataset):
    with pytest.raises(QueryError) as exc:
        Engine(synthetic_dataset).execute("select missing_column\nfrom events")
    assert len(str(exc.value).splitlines()) == 1


def test_parse_error_is_one_line(synthetic_dataset):
    with pytest.raises(QueryError) as exc:
        Engine(synthetic_dataset).execute("select from\n")
    assert len(str(exc.value).splitlines()) == 1


def test_blocks_direct_parquet_read_inside_allowed_dir(synthetic_dataset):
    engine = Engine(synthetic_dataset, cutoff=date(2024, 1, 31))
    path = str(synthetic_dataset.root / "data/events_2024-02.parquet").replace("'", "''")
    with pytest.raises(QueryError, match="query may only read the dataset views"):
        engine.execute(f"select count(*) from read_parquet('{path}')")


def test_blocks_replacement_scan(synthetic_dataset):
    path = str(synthetic_dataset.root / "data/events_2024-01.parquet").replace("'", "''")
    with pytest.raises(QueryError, match="query may only read the dataset views"):
        Engine(synthetic_dataset).execute(f"select * from '{path}'")


def test_blocks_read_csv_lookup_file(synthetic_dataset):
    path = str(synthetic_dataset.root / "data/regions.csv").replace("'", "''")
    with pytest.raises(QueryError, match="query may only read the dataset views"):
        Engine(synthetic_dataset).execute(f"select * from read_csv('{path}')")


def test_cte_and_views_allowed(synthetic_dataset):
    engine = Engine(synthetic_dataset)
    result = engine.execute(
        "with x as (select * from events) "
        "select count(*) from x join regions r on x.region_id = r.region_id"
    )
    assert result[0][0].as_py() == engine.execute("select count(*) from events")[0][0].as_py()


def test_range_allowed(synthetic_dataset):
    assert Engine(synthetic_dataset).execute("select count(*) from range(10)")[0][0].as_py() == 10


def test_getenv_blocked(synthetic_dataset):
    with pytest.raises(QueryError):
        Engine(synthetic_dataset).execute("select getenv('HOME')")


def test_scoped_cte_cannot_whitelist_file_outer(synthetic_dataset):
    path = str(synthetic_dataset.root / "data/events_2024-02.parquet").replace('"', '""')
    engine = Engine(synthetic_dataset, cutoff=date(2024, 1, 31))
    with pytest.raises(QueryError, match="query may only read the dataset views"):
        engine.execute(
            f'select max(ts) from "{path}" '
            f'where 1 in (with "{path}" as (select 1) select 1)'
        )


def test_scoped_cte_cannot_whitelist_file_sibling(synthetic_dataset):
    path = str(synthetic_dataset.root / "data/events_2024-02.parquet").replace('"', '""')
    engine = Engine(synthetic_dataset, cutoff=date(2024, 1, 31))
    with pytest.raises(QueryError, match="query may only read the dataset views"):
        engine.execute(
            f'select * from (with "{path}" as (select 1) select 1), "{path}" limit 1'
        )


@pytest.mark.parametrize(
    "sql",
    [
        'select * from "events x"',
        'with "events x" as (select * from events) select * from "events x"',
    ],
)
def test_non_identifier_table_name_rejected(synthetic_dataset, sql):
    with pytest.raises(QueryError, match="query may only read the dataset views"):
        Engine(synthetic_dataset).execute(sql)


def test_cte_name_not_visible_outside_subquery(synthetic_dataset):
    with pytest.raises(QueryError, match="query may only read the dataset views"):
        Engine(synthetic_dataset).execute(
            "select * from outside_cte where 1 in "
            "(with outside_cte as (select 1) select 1)"
        )


@pytest.mark.parametrize(
    "ast",
    [
        {"type": "BASE_TABLE"},
        {"type": "BASE_TABLE", "table_name": None},
        {"type": "TABLE_FUNCTION", "function": None},
        {"cte_map": None},
    ],
)
def test_malformed_ast_raises_query_error(synthetic_dataset, ast):
    engine = Engine(synthetic_dataset)
    connection = Mock()
    connection.execute.return_value.fetchall.return_value = [(json.dumps(ast),)]
    engine._con = connection
    with pytest.raises(QueryError):
        engine._validate_sql("select 1")
