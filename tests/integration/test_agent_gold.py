"""Frozen agent-eval gold equals a fresh run of each row's gold_sql on the full data."""

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from analyst.data.dataset import load_dataset
from analyst.data.engine import Engine
from evals.graders.agent import derive_key_values, load_rows, parse_checks
from evals.graders.result_match import result_set_match

pytestmark = pytest.mark.taxi
ROOT = Path(__file__).resolve().parents[2]
ROWS = load_rows(ROOT / "evals/datasets/agent_single_turn.jsonl")
GOLD_ROWS = [row for row in ROWS if row["gold_sql"] is not None]


@pytest.fixture(scope="module")
def engine():
    # No cutoff: forecast rows freeze the real future.
    return Engine(load_dataset("nyc_taxi"))


@pytest.mark.parametrize("row", GOLD_ROWS, ids=[row["id"] for row in GOLD_ROWS])
def test_frozen_gold_matches_fresh_run(engine, row):
    expected = pq.read_table(ROOT / "evals/gold" / f"{row['id']}.parquet")
    fresh = engine.execute(row["gold_sql"])
    tolerance = row["tolerance"]
    grade = result_set_match(
        fresh,
        expected,
        rel_tol=tolerance["rel"] if tolerance else 1e-9,
        ignore_column_names=False,
    )
    assert grade.score, grade.comment
    stored = row["key_values"]
    derived = derive_key_values(row, fresh)
    assert [k["label"] for k in derived] == [k["label"] for k in stored]
    for got, want in zip(derived, stored, strict=True):
        assert got["value"] == pytest.approx(want["value"], rel=1e-9)
        assert got["unit"] == want["unit"]


def test_every_gold_check_parses():
    for row in ROWS:
        parse_checks(row["gold_check"])


def test_no_gold_parquet_without_gold_sql():
    for row in ROWS:
        if row["gold_sql"] is None:
            assert not (ROOT / "evals/gold" / f"{row['id']}.parquet").exists(), row["id"]


@pytest.mark.parametrize(
    "row_id, dimensions, grain, freq, id_col",
    [
        ("a035", ["pickup_borough"], "month", "MS", "pickup_borough"),
        ("a036", [], "day", "D", None),
    ],
)
def test_real_forecast_passes_forecast_shape(tmp_path, row_id, dimensions, grain, freq, id_col):
    """A real service forecast at the row's cutoff meets forecast_shape, date grid included."""
    from datetime import date

    from langchain_core.messages import ToolMessage

    from analyst.data.service import DataService
    from analyst.semantic.compiler import MetricQuery
    from evals.graders.agent import grade_checks
    from evals.target import _handle_entries

    row = next(r for r in ROWS if r["id"] == row_id)
    _, args = parse_checks(row["gold_check"])[0]
    svc = DataService(load_dataset("nyc_taxi"), cutoff=date.fromisoformat(row["cutoff"]), store_root=tmp_path)
    query = svc.query_metrics(MetricQuery(metrics=["trips"], dimensions=dimensions, grain=grain))
    card = svc.forecast(query.handle_id, value_col="trips", id_col=id_col, freq=freq, horizon=args[2])
    message = ToolMessage(
        content="", tool_call_id="t", artifact={"handle_id": card.handle_id, "produced_by": "forecast", "governed": True}
    )
    outputs = {"answer": {"narrative": "", "claims": []}, "handles": _handle_entries([message], svc.store)}
    result = grade_checks(outputs, row)
    assert result["score"] is True, result["comment"]
