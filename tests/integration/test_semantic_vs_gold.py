"""Compare the governed service with frozen, independently written SQL answers.

Unsupported requests remain strict xfails: do not bypass governance with raw SQL
or post-process rows to manufacture an answer the semantic layer cannot express.
"""

import json
from datetime import date
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from analyst.data.dataset import load_dataset
from analyst.data.service import DataService
from analyst.semantic.compiler import FilterSpec, MetricQuery
from evals.graders.result_match import result_set_match

pytestmark = pytest.mark.taxi
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def service(tmp_path_factory):
    return DataService(load_dataset("nyc_taxi"), store_root=tmp_path_factory.mktemp("gold-check"))


@pytest.fixture(scope="module")
def golden():
    return {
        row["id"]: row
        for row in map(json.loads, (ROOT / "evals/datasets/sql_golden.jsonl").read_text().splitlines())
    }


CASES = [
    pytest.param("q001", MetricQuery(
        metrics=["trips"], time_range=(date(2024, 1, 1), date(2024, 2, 1)),
    ), id="q001"),
    pytest.param("q002", MetricQuery(
        metrics=["trips"], time_range=(date(2024, 7, 4), date(2024, 7, 5)),
    ), id="q002"),
    pytest.param("q003", MetricQuery(
        metrics=["avg_fare"], time_range=(date(2024, 1, 1), date(2025, 1, 1)),
    ), id="q003"),
    # Closest available request: order_by is ascending only, so this returns the
    # bottom five instead of the top five. No reversal after LIMIT can fix it.
    pytest.param("q004", MetricQuery(
        metrics=["trips"], dimensions=["pickup_zone_name"],
        time_range=(date(2025, 1, 1), date(2026, 1, 1)), order_by="trips", limit=5,
    ), id="q004", marks=pytest.mark.xfail(
        reason="MetricQuery.order_by supports ascending only; cannot request top-5 descending trips",
        strict=True, raises=AssertionError,
    )),
    pytest.param("q005", MetricQuery(
        metrics=["trips"], dimensions=["payment_type"],
        time_range=(date(2023, 1, 1), date(2024, 1, 1)),
    ), id="q005"),
    pytest.param("q006", MetricQuery(
        metrics=["trips"], dimensions=["pickup_borough"], grain="month",
        time_range=(date(2024, 1, 1), date(2025, 1, 1)),
    ), id="q006"),
    pytest.param("q007", MetricQuery(
        metrics=["trips"], filters=[FilterSpec(
            dimension="pickup_borough", op="eq", values=["Manhattan"],
        )], time_range=(date(2026, 3, 1), date(2026, 4, 1)),
        compare_to="same_period_last_year",
    ), id="q007"),
    pytest.param("q008", MetricQuery(
        metrics=["trips"], dimensions=["hour", "day_type"],
        time_range=(date(2024, 1, 1), date(2025, 1, 1)),
    ), id="q008", marks=pytest.mark.xfail(
        reason="No day_type (weekday/weekend) dimension; weekday names cannot be regrouped by MetricQuery",
        strict=True, raises=KeyError,
    )),
    pytest.param("q009", MetricQuery(
        metrics=["tip_rate"], dimensions=["payment_type"],
        time_range=(date(2024, 1, 1), date(2025, 1, 1)),
    ), id="q009"),
    pytest.param("q010", MetricQuery(
        metrics=["trips"], dimensions=["airport_pickup"], share=True,
        time_range=(date(2024, 1, 1), date(2025, 1, 1)),
    ), id="q010", marks=pytest.mark.xfail(
        reason="No airport_pickup dimension or airport_trips metric; filtering airports loses all-trip denominator",
        strict=True, raises=KeyError,
    )),
]


@pytest.mark.parametrize("question_id, query", CASES)
def test_semantic_vs_gold(service, golden, question_id, query):
    expected = pq.read_table(ROOT / "evals/gold" / f"{question_id}.parquet")
    card = service.query_metrics(query)
    assert card.governed
    actual = service.store.load(card.handle_id)
    # Deliberately align/select service columns before positional grading. Never
    # depend on incidental compiler projection order or extra comparison columns.
    actual = actual.select(expected.column_names)
    tolerance = golden[question_id]["tolerance"]
    grade = result_set_match(actual, expected, rel_tol=tolerance["rel"] if tolerance else 0)
    assert grade.score, grade.comment
