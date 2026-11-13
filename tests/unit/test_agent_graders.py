"""Deterministic agent graders over hand-built outputs (no model, no data)."""

import json
from pathlib import Path

import pytest

from evals.graders.agent import (
    GATES,
    GRADERS,
    grade_answer_values,
    grade_checks,
    grade_forecast_skill,
    grade_grounding,
    grade_label,
    grade_trajectory,
    load_rows,
    parse_checks,
    passed,
)

CUTOFF = "2025-07-31"


def outputs_with(
    narrative: str = "",
    claims: list[dict] | None = None,
    label: str = "verified",
    problems: list[str] | None = None,
    tools: list[str] | None = None,
    handles: dict | None = None,
) -> dict:
    verification = {
        "label": label,
        "reason": "test",
        "off_model_tools": [],
        "problems": problems or [],
    }
    return {
        "answer": {
            "narrative": narrative,
            "claims": claims or [],
            "charts": [],
            "sql": [],
            "verification": verification,
            "background_tasks": [],
        },
        "verification": verification,
        "tools": tools or [],
        "handles": handles or {},
    }


def claim(text: str = "", values=(), type_: str = "descriptive", fact_ids=()) -> dict:
    return {
        "text": text,
        "type": type_,
        "values": list(values),
        "handles": [],
        "fact_ids": list(fact_ids),
    }


def forecast_meta(**overrides) -> dict:
    meta = {
        "produced_by": "forecast",
        "model": "AutoETS",
        "origin": "2025-07-01T00:00:00",
        "horizon": 1,
        "freq": "MS",
        "beats_baseline": True,
        "backtest_handle": "r_0000000001",
        "backtest_summary": {
            "best_model": "AutoETS",
            "mean_relative_mae": 0.8,
            "mean_mase": 0.5,
            "mean_coverage_80": 0.8,
            "mean_coverage_95": 0.95,
            "baseline_mean_mase": 0.7,
        },
        "cutoff": CUTOFF,
        "candidates": ["AutoARIMA", "AutoETS", "SeasonalNaive"],
        "headline_ids": ["Brooklyn", "Manhattan", "Queens"],
    }
    meta.update(overrides)
    return meta


BOROUGH_GOLD = [
    {"period": "2025-08-01", "pickup_borough": "Manhattan", "trips": 1000.0},
    {"period": "2025-08-01", "pickup_borough": "Queens", "trips": 200.0},
    {"period": "2025-08-01", "pickup_borough": "Brooklyn", "trips": 50.0},
    {"period": "2025-08-01", "pickup_borough": "EWR", "trips": 1.0},
]


def forecast_ref(**overrides) -> dict:
    ref = {
        "id": "t-forecast",
        "category": "forecast",
        "cutoff": CUTOFF,
        "key_values": [],
        "tolerance": None,
        "gold_check": ["forecast_shape('monthly_trips_by_borough', 'MS', 1)"],
        "expected_tools": ["query_metrics", "forecast"],
        "expected_label": None,
        "gold": BOROUGH_GOLD,
    }
    ref.update(overrides)
    return ref


# --- answer_values ---------------------------------------------------------


def test_answer_values_gate():
    ref = {"key_values": [{"value": 2894704, "label": "Aug"}], "tolerance": None}
    assert grade_answer_values(outputs_with(narrative="2,894,704 trips"), ref)["score"] is True
    # Rounded prose passes the Task 8 rounding rule only with >= 3 significant
    # digits when |gold| >= 1000: "2.89M" passes, "2.9M" fails.
    assert grade_answer_values(outputs_with(narrative="2.89M trips"), ref)["score"] is True
    assert grade_answer_values(outputs_with(narrative="2.9M trips"), ref)["score"] is False
    assert grade_answer_values(outputs_with(narrative="2,894,705 trips"), ref)["score"] is False


def test_answer_values_from_claim_values():
    ref = {"key_values": [{"value": 19.268123456, "label": "avg"}], "tolerance": None}
    out = outputs_with(narrative="about nineteen dollars", claims=[claim(values=[19.268123456])])
    assert grade_answer_values(out, ref)["score"] is True
    rounded = outputs_with(claims=[claim(values=[19.27])])
    assert grade_answer_values(rounded, ref)["score"] is False


def test_answer_values_small_numbers_round_at_stated_precision():
    ref = {"key_values": [{"value": 19.268, "label": "avg"}], "tolerance": None}
    assert grade_answer_values(outputs_with(narrative="$19.27"), ref)["score"] is True
    assert grade_answer_values(outputs_with(narrative="$19.3"), ref)["score"] is True
    assert grade_answer_values(outputs_with(narrative="$19.26"), ref)["score"] is False


def test_answer_values_percent_of_fraction():
    ref = {"key_values": [{"value": 0.0299, "label": "share"}], "tolerance": None}
    assert grade_answer_values(outputs_with(narrative="3.0% of trips"), ref)["score"] is True


def test_answer_values_relative_tolerance():
    ref = {"key_values": [{"value": 1000.0, "label": "x"}], "tolerance": {"rel": 0.01}}
    assert grade_answer_values(outputs_with(narrative="1,005"), ref)["score"] is True
    assert grade_answer_values(outputs_with(narrative="1,020"), ref)["score"] is False
    assert grade_answer_values(outputs_with(claims=[claim(values=[995.0])]), ref)["score"] is True


def test_answer_values_needs_every_key_value():
    ref = {
        "key_values": [{"value": 10.0, "label": "a"}, {"value": 20.0, "label": "b"}],
        "tolerance": None,
    }
    result = grade_answer_values(outputs_with(narrative="10 and 21"), ref)
    assert result["score"] is False
    assert "b" in result["comment"]
    assert grade_answer_values(outputs_with(narrative="10 and 20"), ref)["score"] is True


def test_answer_values_not_applicable_without_key_values():
    assert grade_answer_values(outputs_with(), {"key_values": [], "tolerance": None})["score"] is None


# --- label and grounding ---------------------------------------------------


def test_label_gate():
    ref = {"expected_label": "verified_off_model"}
    assert grade_label(outputs_with(label="verified_off_model"), ref)["score"] is True
    assert grade_label(outputs_with(label="verified"), ref)["score"] is False
    assert grade_label(outputs_with(), {"expected_label": None})["score"] is None


def test_grounding_gate_for_traps_only():
    ref = {"category": "trap"}
    clean = outputs_with(problems=[])
    dirty = outputs_with(label="not_verified", problems=["unsupported causal claim"])
    assert grade_grounding(clean, ref)["score"] is True
    assert grade_grounding(dirty, ref)["score"] is False
    assert grade_grounding(dirty, {"category": "lookup"})["score"] is None


# --- checks ----------------------------------------------------------------


def test_checks_forecast_shape():
    out = outputs_with(handles={"r_00000000aa": forecast_meta()})
    assert grade_checks(out, forecast_ref())["score"] is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"freq": "D"},
        {"horizon": 3},
        {"cutoff": "2026-01-31"},
        {"cutoff": None},
        {"headline_ids": ["Manhattan", "Queens"]},
        {"candidates": ["AutoETS"]},
        {"backtest_summary": {}},
    ],
)
def test_checks_forecast_shape_rejects(overrides):
    out = outputs_with(handles={"r_00000000aa": forecast_meta(**overrides)})
    assert grade_checks(out, forecast_ref())["score"] is False


def test_checks_forecast_shape_single_series():
    ref = forecast_ref(
        gold_check=["forecast_shape('daily_trips', 'D', 7)"],
        gold=[{"period": "2025-08-01", "trips": 1.0}],
    )
    single = forecast_meta(freq="D", horizon=7, headline_ids=["series"])
    assert grade_checks(outputs_with(handles={"r_1": single}), ref)["score"] is True
    by_id = forecast_meta(freq="D", horizon=7)
    assert grade_checks(outputs_with(handles={"r_1": by_id}), ref)["score"] is False


def test_checks_declined():
    ref = {"gold_check": ["declined()"], "cutoff": None}
    ok = outputs_with(narrative="The data ends on 2026-01-31, so March 2026 is not available.")
    assert grade_checks(ok, ref)["score"] is True
    fabricated = outputs_with(narrative="There were", claims=[claim(values=[3.1e6])])
    assert grade_checks(fabricated, ref)["score"] is False
    assert grade_checks(outputs_with(narrative="  "), ref)["score"] is False


def test_checks_quirk_flagged_and_causal():
    ref = {
        "gold_check": ["quirk_flagged('gap:2023-09-21', 'data:2023-09-21')", "no_causal_without('data:2023-09-21')"],
        "cutoff": None,
    }
    flagged = outputs_with(
        narrative="Counts collapse on these days.",
        claims=[claim("Low counts may reflect the known data gap.", type_="causal", fact_ids=["data:2023-09-21"])],
    )
    assert grade_checks(flagged, ref)["score"] is True
    unflagged = outputs_with(narrative="Counts collapse on these days.")
    assert grade_checks(unflagged, ref)["score"] is False
    wrong_cause = outputs_with(
        narrative="See gap:2023-09-21.",
        claims=[claim("Likely related to a holiday.", type_="causal", fact_ids=["holiday:2023-09-04"])],
    )
    assert grade_checks(wrong_cause, ref)["score"] is False


def test_checks_no_causal_without_any_id_forbids_causal_claims():
    ref = {"gold_check": ["no_causal_without()"], "cutoff": None}
    assert grade_checks(outputs_with(claims=[claim("x", type_="descriptive")]), ref)["score"] is True
    causal = outputs_with(claims=[claim("possibly weather", type_="causal", fact_ids=["policy:2025-01-05"])])
    assert grade_checks(causal, ref)["score"] is False


def test_checks_not_applicable_when_empty():
    assert grade_checks(outputs_with(), {"gold_check": []})["score"] is None


def test_unknown_check_name_rejected():
    with pytest.raises(ValueError, match="unknown check"):
        parse_checks(["must_decline()"])
    with pytest.raises(ValueError):
        parse_checks(["declined(); import os"])
    with pytest.raises(ValueError):
        parse_checks(["quirk_flagged(x)"])  # arguments must be literals
    with pytest.raises(ValueError):
        parse_checks(["forecast_shape('no_such_series', 'MS', 1)"])
    with pytest.raises(ValueError):
        parse_checks(["forecast_shape('monthly_trips_by_borough', 'D', 1)"])  # freq mismatch


def test_load_rows_rejects_unknown_check(tmp_path: Path):
    row = {
        "id": "a999", "tier": 1, "category": "safe", "question": "q", "cutoff": None,
        "gold_sql": None, "key_values": [], "tolerance": None,
        "gold_check": ["nope()"], "expected_tools": [], "expected_label": None, "notes": "",
    }
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="a999"):
        load_rows(path)


# --- trajectory and forecast skill ----------------------------------------


def test_trajectory_is_score_not_gate():
    assert "trajectory" not in GATES
    assert "forecast_skill" not in GATES
    ref = {"expected_tools": ["query_metrics", "forecast"]}
    good = grade_trajectory(outputs_with(tools=["read_knowledge", "query_metrics", "forecast"]), ref)
    assert good["key"] == "trajectory"
    assert good["score"] is True
    assert grade_trajectory(outputs_with(tools=["run_sql"]), ref)["score"] is False
    assert grade_trajectory(outputs_with(tools=[]), {"expected_tools": []})["score"] is True


def test_forecast_skill_scores_error_and_coverage():
    rows = [
        {"unique_id": "Manhattan", "ds": "2025-08-01T00:00:00", "yhat": 900.0,
         "lo_80": 850.0, "hi_80": 990.0, "lo_95": 850.0, "hi_95": 1100.0},
        {"unique_id": "Queens", "ds": "2025-08-01T00:00:00", "yhat": 200.0,
         "lo_80": 190.0, "hi_80": 210.0, "lo_95": 180.0, "hi_95": 220.0},
    ]
    out = outputs_with(handles={"r_1": forecast_meta(forecast=rows)})
    result = grade_forecast_skill(out, forecast_ref())
    assert result["key"] == "forecast_skill"
    # |900-1000| + |200-200| over 1000 + 200
    assert result["score"] == pytest.approx(100 / 1200)
    assert "coverage_80=0.50" in result["comment"]
    assert "coverage_95=1.00" in result["comment"]
    assert grade_forecast_skill(outputs_with(), {"category": "lookup"})["score"] is None
    assert grade_forecast_skill(outputs_with(), forecast_ref())["score"] is None


# --- passed ------------------------------------------------------------------


def run_all(out: dict, ref: dict) -> list[dict]:
    return [grader(out, ref) for grader in GRADERS]


def test_passed_requires_all_applicable_gates():
    ref = {
        "id": "t", "category": "comparison", "cutoff": None,
        "key_values": [{"value": 10.0, "label": "a"}], "tolerance": None,
        "gold_check": [], "expected_tools": ["query_metrics"], "expected_label": "verified",
    }
    good = outputs_with(narrative="10 trips", tools=["run_sql"])
    assert passed(run_all(good, ref)) is True  # trajectory miss is a score only
    wrong_label = outputs_with(narrative="10 trips", label="not_verified")
    assert passed(run_all(wrong_label, ref)) is False
    wrong_value = outputs_with(narrative="11 trips")
    assert passed(run_all(wrong_value, ref)) is False
    missing_gate = [r for r in run_all(good, ref) if r["key"] != "label"]
    assert passed(missing_gate) is False


def test_error_outputs_fail_every_gate():
    ref = {"category": "safe", "key_values": [], "tolerance": None, "gold_check": ["declined()"],
           "expected_tools": [], "expected_label": None, "cutoff": None}
    results = run_all({"error": "RuntimeError('boom')"}, ref)
    gates = [r for r in results if r["key"] in GATES]
    assert len(gates) == len(GATES)
    assert all(r["score"] is False for r in gates)
    assert passed(results) is False


# --- the eval set itself -------------------------------------------------------

EVAL_SET = Path(__file__).resolve().parents[2] / "evals/datasets/agent_single_turn.jsonl"


def test_eval_set_mix():
    rows = load_rows(EVAL_SET)
    assert [r["id"] for r in rows] == [f"a{i:03d}" for i in range(1, 51)]
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    assert counts == {"lookup": 20, "comparison": 8, "forecast": 8, "off_model": 6, "trap": 5, "safe": 3}
    forecast_cutoffs = {r["cutoff"] for r in rows if r["category"] == "forecast"}
    assert forecast_cutoffs == {"2025-07-31", "2026-01-31", "2026-07-31"}


def test_eval_set_rows_have_a_gate():
    for row in load_rows(EVAL_SET):
        applicable = bool(row["key_values"] or row["expected_label"] or row["gold_check"]) or row["category"] == "trap"
        assert applicable, row["id"]
        if row["category"] == "forecast":
            assert row["key_values"] == [] and row["gold_sql"]
        if row["category"] in ("comparison", "off_model"):
            assert row["key_values"], row["id"]


def test_eval_set_carries_sql_gold_rows():
    sql_rows = [json.loads(line) for line in (EVAL_SET.parent / "sql_golden.jsonl").read_text().splitlines()]
    agent = {r["id"]: r for r in load_rows(EVAL_SET)}
    for old in sql_rows:
        new = agent["a" + old["id"][1:]]
        for field in ("tier", "question", "gold_sql", "tolerance", "notes"):
            assert new[field] == old[field], (old["id"], field)
        assert new["category"] == "lookup" and new["cutoff"] is None


def test_derive_key_values():
    import pyarrow as pa

    from evals.graders.agent import derive_key_values

    table = pa.table({"pickup_borough": ["Manhattan", "Queens"], "trips": [10, 20]})
    row = {"id": "x", "category": "lookup"}
    assert derive_key_values(row, table) == [
        {"value": 10.0, "label": "Manhattan trips"},
        {"value": 20.0, "label": "Queens trips"},
    ]
    picked = {**row, "key_select": [{"where": {"pickup_borough": "Queens"}, "col": "trips"}]}
    assert derive_key_values(picked, table) == [{"value": 20.0, "label": "pickup_borough=Queens trips"}]
    assert derive_key_values({**row, "key_select": []}, table) == []
    assert derive_key_values({**row, "category": "forecast"}, table) == []
    big = pa.table({"trips": list(range(25))})
    with pytest.raises(ValueError, match="key_select"):
        derive_key_values(row, big)
    with pytest.raises(ValueError, match="matched 0 rows"):
        derive_key_values({**row, "key_select": [{"where": {"pickup_borough": "Bronx"}, "col": "trips"}]}, table)
