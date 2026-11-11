import json
from collections import Counter
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import duckdb
import pyarrow as pa
import pytest

from evals.graders.result_match import GradeResult, result_set_match

ROOT = Path(__file__).resolve().parents[2]


def test_exact_match():
    table = pa.table({"x": [1, 2], "label": ["a", "b"]})
    assert result_set_match(table, table).score


def test_row_order_ignored():
    got = pa.table({"x": [2, None, 1, 1], "y": ["b", None, "a", "a"]})
    expected = pa.table({"x": [1, 2, 1, None], "y": ["a", "b", "a", None]})
    assert result_set_match(got, expected).score
    assert not result_set_match(got, expected, ignore_order=False).score


def test_float_tolerance():
    assert result_set_match(pa.table({"x": [1.0000001]}), pa.table({"x": [1.0]})).score
    assert not result_set_match(
        pa.table({"x": [1.0000001]}), pa.table({"x": [1.0]}), rel_tol=0, abs_tol=0
    ).score


def test_float_rounding_before_sort():
    got = pa.table({"x": [1.0000002, 1.0000001], "y": ["a", "b"]})
    expected = pa.table({"x": [1.0, 1.0], "y": ["a", "b"]})
    assert result_set_match(got, expected).score


def test_value_mismatch_comment_names_row():
    grade = result_set_match(pa.table({"x": [1, 3]}), pa.table({"x": [1, 2]}))
    assert not grade.score
    assert "row 1" in grade.comment
    assert "3" in grade.comment and "2" in grade.comment


@pytest.mark.parametrize("got", [pa.table({"x": [1, 2]}), pa.table({"x": [1], "y": [2]})])
def test_shape_mismatch(got):
    grade = result_set_match(got, pa.table({"x": [1]}))
    assert not grade.score and "shape" in grade.comment.lower()


def test_column_names_ignored_by_default():
    got, expected = pa.table({"a": [1]}), pa.table({"b": [1]})
    assert result_set_match(got, expected).score
    grade = result_set_match(got, expected, ignore_column_names=False)
    assert not grade.score and "column" in grade.comment.lower()


def test_columns_matched_by_name_when_requested():
    got = pa.table({"b": [2], "a": [1]})
    expected = pa.table({"a": [1], "b": [2]})
    assert result_set_match(got, expected, ignore_column_names=False).score
    assert not result_set_match(got, expected).score


def test_numeric_normalization():
    assert result_set_match(
        pa.table({"x": [Decimal("1.250000"), None]}),
        pa.table({"x": [1.25, None]}),
    ).score
    assert result_set_match(pa.table({"x": [1]}), pa.table({"x": [1.0]})).score


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_date_equals_timestamp(unit):
    assert result_set_match(
        pa.table({"x": [date(2024, 1, 1)]}),
        pa.table({"x": pa.array([datetime(2024, 1, 1)], type=pa.timestamp(unit))}),
    ).score
    assert result_set_match(
        pa.table({"x": [date(2024, 1, 1), None]}),
        pa.table({"x": [datetime(2024, 1, 1), None]}),
    ).score
    assert not result_set_match(
        pa.table({"x": [date(2024, 1, 1)]}),
        pa.table({"x": [datetime(2024, 1, 1, 1)]}),
    ).score


@pytest.mark.parametrize("got, expected", [("A", "a"), ("1", 1), (None, 0)])
def test_strings_exact_and_null_only_equals_null(got, expected):
    assert not result_set_match(pa.table({"x": [got]}), pa.table({"x": [expected]})).score


def test_duplicates_preserved():
    assert not result_set_match(
        pa.table({"x": [1, 1, 2]}), pa.table({"x": [1, 2, 2]})
    ).score


def test_empty_tables():
    assert result_set_match(pa.table({"x": []}), pa.table({"y": []})).score


def test_as_dict_shape():
    assert GradeResult("result_set_match", True, "match").as_dict() == {
        "key": "result_set_match", "score": True, "comment": "match"
    }


def test_golden_file_valid():
    rows = [json.loads(line) for line in (ROOT / "evals/datasets/sql_golden.jsonl").read_text().splitlines()]
    assert len(rows) == 15
    assert {row["id"] for row in rows} == {f"q{i:03}" for i in range(1, 16)}
    assert Counter(row["tier"] for row in rows) == {1: 5, 2: 5, 3: 5}
    with duckdb.connect() as con:
        for row in rows:
            assert set(row) == {"id", "tier", "question", "gold_sql", "tolerance", "notes"}
            assert row["question"] and row["notes"]
            tolerance = row["tolerance"]
            assert tolerance is None or (set(tolerance) == {"rel"} and tolerance["rel"] >= 0)
            statements = con.extract_statements(row["gold_sql"])
            assert len(statements) == 1
            assert statements[0].type == duckdb.StatementType.SELECT
