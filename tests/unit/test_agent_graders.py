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
        "input_handle": "r_input",
    }
    meta.update(overrides)
    if "forecast" not in overrides:
        # Rows on the expected grid: the first period after the cutoff, every freq.
        import pandas as pd

        start = "2025-08-01"
        dates = pd.date_range(start, periods=meta["horizon"], freq=meta["freq"])
        meta["forecast"] = [
            {"unique_id": uid, "ds": d.isoformat(), "yhat": 1.0, "lo_80": 0.0,
             "hi_80": 2.0, "lo_95": 0.0, "hi_95": 3.0}
            for uid in meta["headline_ids"] for d in dates
        ]
    return meta


def with_input(handles: dict, metric: str = "trips") -> dict:
    """Add the query_metrics input handle that forecast metas point at."""
    query = {"metrics": [metric], "dimensions": ["pickup_borough"], "grain": "month"}
    return {"r_input": {"produced_by": "query_metrics", "query": query, "cutoff": CUTOFF}, **handles}


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
    ref = {"key_values": [{"value": 0.0299, "label": "share", "unit": "ratio"}], "tolerance": None}
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
    out = outputs_with(handles=with_input({"r_00000000aa": forecast_meta()}))
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
    out = outputs_with(handles=with_input({"r_00000000aa": forecast_meta(**overrides)}))
    assert grade_checks(out, forecast_ref())["score"] is False


def test_checks_forecast_shape_single_series():
    ref = forecast_ref(
        gold_check=["forecast_shape('daily_trips', 'D', 7)"],
        gold=[{"period": "2025-08-01", "trips": 1.0}],
    )
    single = forecast_meta(freq="D", horizon=7, headline_ids=["series"])
    assert grade_checks(outputs_with(handles=with_input({"r_1": single})), ref)["score"] is True
    by_id = forecast_meta(freq="D", horizon=7)
    assert grade_checks(outputs_with(handles=with_input({"r_1": by_id})), ref)["score"] is False


def test_checks_declined():
    ref = {"gold_check": ["declined()"], "cutoff": None}
    ok = outputs_with(narrative="The data ends on 2026-01-31, so March 2026 is not available.")
    assert grade_checks(ok, ref)["score"] is True
    fabricated = outputs_with(narrative="There were", claims=[claim(values=[3.1e6])], label="not_verified")
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
    out = outputs_with(handles=with_input({"r_1": forecast_meta(forecast=rows)}))
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
        {"value": 10.0, "label": "Manhattan trips", "unit": "value"},
        {"value": 20.0, "label": "Queens trips", "unit": "value"},
    ]
    picked = {**row, "key_select": [{"where": {"pickup_borough": "Queens"}, "col": "trips"}]}
    assert derive_key_values(picked, table) == [
        {"value": 20.0, "label": "pickup_borough=Queens trips", "unit": "value"}
    ]
    assert derive_key_values({**row, "key_select": []}, table) == []
    assert derive_key_values({**row, "category": "forecast"}, table) == []
    big = pa.table({"trips": list(range(25))})
    with pytest.raises(ValueError, match="key_select"):
        derive_key_values(row, big)
    with pytest.raises(ValueError, match="matched 0 rows"):
        derive_key_values({**row, "key_select": [{"where": {"pickup_borough": "Bronx"}, "col": "trips"}]}, table)


# --- fix round 1 -------------------------------------------------------------------


@pytest.mark.parametrize(
    "gold, bad, good",
    [
        (1.3339, "1 passenger", "1.33 passengers"),  # a030
        (0.130072, "0", "0.13"),  # a026 tip rate
        (-20.34, "fell 20%", "fell 20.3%"),  # a026 change pct
        (8.8, "9 miles", "8.8 miles"),  # a034
    ],
)
def test_answer_values_significant_digits(gold, bad, good):
    ref = {"key_values": [{"value": gold, "label": "k"}], "tolerance": None}
    assert grade_answer_values(outputs_with(narrative=bad), ref)["score"] is False
    assert grade_answer_values(outputs_with(narrative=good), ref)["score"] is True


def test_answer_values_exact_small_integers_still_pass():
    ref = {"key_values": [{"value": 5.0, "label": "k"}], "tolerance": None}
    assert grade_answer_values(outputs_with(narrative="5 zones"), ref)["score"] is True
    assert grade_answer_values(outputs_with(narrative="6 zones"), ref)["score"] is False


def test_answer_values_one_to_one():
    ref = {
        "key_values": [{"value": 1500.0, "label": "a"}, {"value": 1500.0, "label": "b"}],
        "tolerance": None,
    }
    assert grade_answer_values(outputs_with(narrative="1,500 trips"), ref)["score"] is False
    assert grade_answer_values(outputs_with(narrative="1,500 and 1,500"), ref)["score"] is True
    # Augmenting path: "1,501" fits only b, so "1,500" must go to a.
    ref2 = {
        "key_values": [{"value": 1500.0, "label": "a"}, {"value": 1500.6, "label": "b"}],
        "tolerance": None,
    }
    assert grade_answer_values(outputs_with(narrative="1,501 then 1,500"), ref2)["score"] is True


A048 = {"gold_check": ["declined()"], "cutoff": "2026-01-31", "question": "How many trips were recorded in March 2026?"}


def test_declined_rejects_untraced_narrative_numbers():
    out = outputs_with(narrative="There were 3,952,443 trips in March 2026.", label="not_verified")
    assert grade_checks(out, A048)["score"] is False


def test_declined_allows_traced_helpful_numbers():
    out = outputs_with(
        narrative="Data ends on 2026-01-31, so March 2026 is not available; January 2026 had 2,912,345 trips.",
        label="verified",
    )
    assert grade_checks(out, A048)["score"] is True
    question_only = outputs_with(narrative="March 2026 is after the data ends.", label="not_verified")
    assert grade_checks(question_only, A048)["score"] is True


def test_forecast_shape_requires_series_metric():
    out = outputs_with(handles=with_input({"r_1": forecast_meta()}, metric="revenue"))
    result = grade_checks(out, forecast_ref())
    assert result["score"] is False
    assert "revenue" in result["comment"]


def test_forecast_shape_missing_input_handle_fails():
    out = outputs_with(handles={"r_1": forecast_meta()})
    result = grade_checks(out, forecast_ref())
    assert result["score"] is False
    assert "input handle" in result["comment"]


QUIRK = {"gold_check": ["quirk_flagged('anomaly:2026-02-23')"], "cutoff": None}


def test_quirk_flagged_by_fact_ids():
    out = outputs_with(claims=[claim("Feb 23 was low.", fact_ids=["anomaly:2026-02-23"])])
    assert grade_checks(out, QUIRK)["score"] is True


def test_quirk_flagged_by_narrative():
    out = outputs_with(narrative="Flagged as known anomaly anomaly:2026-02-23.")
    assert grade_checks(out, QUIRK)["score"] is True


def test_quirk_flagged_not_by_claim_text_alone():
    out = outputs_with(claims=[claim("see anomaly:2026-02-23")])
    assert grade_checks(out, QUIRK)["score"] is False


def test_forecast_skill_scores_the_shape_matching_handle():
    def rows(yhat):
        # Every required series; Queens and Brooklyn are exact.
        return [{"unique_id": uid, "ds": "2025-08-01", "yhat": y,
                 "lo_80": 0.0, "hi_80": 2000.0, "lo_95": 0.0, "hi_95": 2000.0}
                for uid, y in (("Manhattan", yhat), ("Queens", 200.0), ("Brooklyn", 50.0))]

    right = forecast_meta(forecast=rows(900.0))  # error 100 / 1250
    wrong = forecast_meta(freq="D", forecast=rows(500.0))  # later, wrong freq
    out = outputs_with(handles=with_input({"r_1": right, "r_2": wrong}))
    result = grade_forecast_skill(out, forecast_ref())
    assert result["score"] == pytest.approx(100 / 1250)
    assert "forecast_shape handle" in result["comment"]
    fallback = outputs_with(handles=with_input({"r_2": wrong}))
    assert grade_forecast_skill(fallback, forecast_ref())["score"] == pytest.approx(500 / 1250)


# --- final fix wave ------------------------------------------------------------


def test_declined_allows_claim_values_when_verified():
    out = outputs_with(
        narrative="Data ends on 2026-01-31, so March 2026 is not available; January 2026 had 2,912,345 trips.",
        claims=[claim("January 2026 had 2,912,345 trips", values=[2_912_345])],
        label="verified",
    )
    assert grade_checks(out, A048)["score"] is True
    off = outputs_with(narrative=out["answer"]["narrative"], claims=out["answer"]["claims"], label="verified_off_model")
    assert grade_checks(off, A048)["score"] is True
    bad = outputs_with(narrative=out["answer"]["narrative"], claims=out["answer"]["claims"], label="not_verified")
    assert grade_checks(bad, A048)["score"] is False


TRAP = {"category": "trap", "cutoff": "2025-07-31"}


def test_grounding_scans_prose_for_causes():
    out = outputs_with(
        narrative="Trips fell to 2.89M because of the snowstorm.",
        claims=[claim("Trips were 2.89M", values=[2_894_704])],
    )
    assert grade_grounding(out, TRAP)["score"] is False
    in_claim = outputs_with(narrative="Trips fell.", claims=[claim("Trips fell due to rain.")])
    assert grade_grounding(in_claim, TRAP)["score"] is False


def test_grounding_allows_hedged_cause_with_valid_id():
    gap = outputs_with(
        narrative="The drop may be due to the known data gap.",
        claims=[claim("Counts collapse", fact_ids=["gap:2023-09-21"])],
    )
    assert grade_grounding(gap, TRAP)["score"] is True
    fact = outputs_with(
        narrative="The drop may be due to congestion pricing.",
        claims=[claim("Counts fell", fact_ids=["policy:2025-01-05"])],
    )
    assert grade_grounding(fact, TRAP)["score"] is True
    # a fact dated after the row's cutoff does not license a cause
    late = outputs_with(
        narrative="The drop may be due to the holiday.",
        claims=[claim("Counts fell", fact_ids=["holiday:2025-12-25"])],
    )
    assert grade_grounding(late, TRAP)["score"] is False
    unhedged = outputs_with(
        narrative="The drop was due to the known data gap.",
        claims=[claim("Counts collapse", fact_ids=["gap:2023-09-21"])],
    )
    assert grade_grounding(unhedged, TRAP)["score"] is False
    plain = outputs_with(
        narrative="The drop coincides with the known data gap.",
        claims=[claim("Counts collapse", type_="causal", fact_ids=["gap:2023-09-21"])],
    )
    assert grade_grounding(plain, TRAP)["score"] is True


def test_no_causal_without_scans_prose():
    ref = {"gold_check": ["no_causal_without('data:2023-09-21')"], "cutoff": None}
    prose = outputs_with(narrative="Counts fell because of a holiday.", claims=[claim("Counts fell")])
    assert grade_checks(prose, ref)["score"] is False
    hedged = outputs_with(
        narrative="Counts fell, possibly due to the known data gap.",
        claims=[claim("Counts fell", fact_ids=["data:2023-09-21"])],
    )
    assert grade_checks(hedged, ref)["score"] is True
    wrong_id = outputs_with(
        narrative="Counts fell, possibly due to a holiday.",
        claims=[claim("Counts fell", fact_ids=["holiday:2023-09-04"])],
    )
    assert grade_checks(wrong_id, ref)["score"] is False
    forbid = {"gold_check": ["no_causal_without()"], "cutoff": None}
    any_cause = outputs_with(narrative="Counts fell, possibly due to rain.", claims=[claim("x")])
    assert grade_checks(any_cause, forbid)["score"] is False


def test_a044_accepts_the_gap_quirk_id_for_a_cause():
    from evals.graders.agent import ROOT

    rows = {r["id"]: r for r in load_rows(ROOT / "evals/datasets/agent_single_turn.jsonl")}
    a044 = rows["a044"]
    for fid in ("gap:2023-09-21", "data:2023-09-21"):
        out = outputs_with(
            narrative="Daily counts collapse on these days.",
            claims=[claim("The drop coincides with the known data gap.", type_="causal", fact_ids=[fid])],
        )
        assert grade_checks(out, a044)["score"] is True, fid


# --- CodeRabbit PR #3 ------------------------------------------------------------


def _change_ref(gold: float, tolerance=None, unit: str = "percent", label: str = "trips_change_pct") -> dict:
    return {"key_values": [{"value": gold, "label": label, "unit": unit}], "tolerance": tolerance}


@pytest.mark.parametrize("tolerance", [None, {"rel": 1e-06}])
def test_answer_values_negative_change_needs_stated_decrease(tolerance):
    ref = _change_ref(-5.23, tolerance)  # a007
    assert grade_answer_values(outputs_with(narrative="Trips increased by 5.23%."), ref)["score"] is False
    assert grade_answer_values(outputs_with(narrative="Trips rose 5.23% year over year."), ref)["score"] is False
    assert grade_answer_values(outputs_with(narrative="Trips changed 5.23%."), ref)["score"] is False
    assert grade_answer_values(outputs_with(narrative="Trips fell 5.23%."), ref)["score"] is True
    assert grade_answer_values(outputs_with(narrative="Trips were 5.23% lower."), ref)["score"] is True
    assert grade_answer_values(outputs_with(narrative="A 5.23% drop in trips."), ref)["score"] is True
    assert grade_answer_values(outputs_with(narrative="Trips changed -5.23%."), ref)["score"] is True


@pytest.mark.parametrize("tolerance", [None, {"rel": 1e-06}])
def test_answer_values_positive_change_rejects_stated_decrease(tolerance):
    ref = _change_ref(16.25, tolerance)  # a022
    assert grade_answer_values(outputs_with(narrative="Trips fell 16.25%."), ref)["score"] is False
    assert grade_answer_values(outputs_with(narrative="Trips declined by 16.25%."), ref)["score"] is False
    assert grade_answer_values(outputs_with(narrative="Trips rose 16.25%."), ref)["score"] is True
    assert grade_answer_values(outputs_with(narrative="Trips changed by 16.25%."), ref)["score"] is True
    assert grade_answer_values(outputs_with(narrative="Trips changed by +16.25%."), ref)["score"] is True


def test_answer_values_direction_is_the_nearest_cue_in_the_sentence():
    ref = {
        "key_values": [
            {"value": 3391937.0, "label": "trips", "unit": "value"},
            {"value": 3578943.0, "label": "trips_compare", "unit": "value"},
            {"value": -5.23, "label": "trips_change_pct", "unit": "percent"},
        ],
        "tolerance": {"rel": 1e-06},
    }
    good = "Trips fell 5.23% to 3,391,937, down from 3,578,943. Revenue rose."
    assert grade_answer_values(outputs_with(narrative=good), ref)["score"] is True
    bad = "Trips fell in Queens. Overall they rose 5.23% to 3,391,937 from 3,578,943."
    assert grade_answer_values(outputs_with(narrative=bad), ref)["score"] is False


@pytest.mark.parametrize("tolerance", [None, {"rel": 1e-06}])
def test_answer_values_percent_unit_never_scaled_by_100(tolerance):
    ref = _change_ref(1.06, tolerance, label="avg_fare_change_pct")  # a024
    assert grade_answer_values(outputs_with(narrative="Average fare rose 106%."), ref)["score"] is False
    assert grade_answer_values(outputs_with(claims=[claim(values=[106.0])]), ref)["score"] is False
    assert grade_answer_values(outputs_with(narrative="Average fare rose 1.06%."), ref)["score"] is True
    assert grade_answer_values(outputs_with(claims=[claim(values=[1.06])]), ref)["score"] is True


@pytest.mark.parametrize("tolerance", [None, {"rel": 1e-03}])
def test_answer_values_ratio_unit_accepts_percent(tolerance):
    ref = _change_ref(0.2206376, tolerance, unit="ratio", label="Credit card tip_rate")  # a009
    assert grade_answer_values(outputs_with(narrative="Card tips were 22.06% of fares."), ref)["score"] is True
    assert grade_answer_values(outputs_with(narrative="Card tip rate 0.2206."), ref)["score"] is True
    assert grade_answer_values(outputs_with(claims=[claim(values=[22.06376])]), ref)["score"] is True
    value = _change_ref(0.2206376, tolerance, unit="value", label="x")
    assert grade_answer_values(outputs_with(narrative="Card tips were 22.06% of fares."), value)["score"] is False
    assert grade_answer_values(outputs_with(claims=[claim(values=[22.06376])]), value)["score"] is False


def test_derive_key_values_units():
    import pyarrow as pa

    from evals.graders.agent import derive_key_values

    table = pa.table({
        "trips": [10], "trips_compare": [12], "trips_change_pct": [-16.67],
        "tip_rate": [0.13], "tip_rate_compare": [0.16], "airport_share_pct": [7.99],
        "share_2023_pct": [87.9], "change_pp": [-1.37], "avg_fare": [19.8],
    })
    units = {kv["label"]: kv["unit"] for kv in derive_key_values({"id": "x", "category": "lookup"}, table)}
    assert units == {
        "trips": "value", "trips_compare": "value", "trips_change_pct": "percent",
        "tip_rate": "ratio", "tip_rate_compare": "ratio", "airport_share_pct": "percent",
        "share_2023_pct": "percent", "change_pp": "value", "avg_fare": "value",
    }


def test_eval_set_key_values_carry_units():
    units = {
        kv["label"]: kv["unit"] for row in load_rows(EVAL_SET) for kv in row["key_values"]
    }
    assert units["Credit card tip_rate"] == "ratio"
    assert units["tip_rate_compare"] == "ratio"
    assert units["avg_fare_change_pct"] == "percent"
    assert units["trips"] == "value"
    assert set(units.values()) <= {"value", "percent", "ratio"}


def test_load_rows_rejects_unknown_unit(tmp_path: Path):
    row = {
        "id": "a998", "tier": 1, "category": "lookup", "question": "q", "cutoff": None,
        "gold_sql": "SELECT 1", "key_values": [{"value": 1.0, "label": "x", "unit": "pct"}],
        "tolerance": None, "gold_check": [], "expected_tools": [], "expected_label": None, "notes": "",
    }
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="a998"):
        load_rows(path)


@pytest.mark.parametrize(
    "narrative",
    [
        "Hello.",
        "Sure, here you go.",
        "I cannot.",
    ],
)
def test_declined_requires_refusal_with_reason(narrative):
    assert grade_checks(outputs_with(narrative=narrative, label="not_verified"), A048)["score"] is False


@pytest.mark.parametrize(
    "narrative",
    [
        "I can't answer that: the data ends on 2026-01-31.",
        "March 2026 is after the data ends on 2026-01-31.",
        "That month is beyond the cutoff of 2026-01-31, so I won't estimate it.",
        "There is no such column in the dataset.",
        "I'm unable to report March 2026 because the data stops in January.",
    ],
)
def test_declined_accepts_refusal_with_reason(narrative):
    assert grade_checks(outputs_with(narrative=narrative, label="not_verified"), A048)["score"] is True


def test_declined_rejects_claim_values_without_trace():
    out = outputs_with(
        narrative="March 2026 is not available because data ends in January.",
        claims=[claim(values=[1.0])],
        label="not_verified",
    )
    assert grade_checks(out, A048)["score"] is False


def _grid(ids, start: str, freq: str, horizon: int) -> list[dict]:
    import pandas as pd

    dates = pd.date_range(start, periods=horizon, freq=freq)
    return [
        {"unique_id": uid, "ds": d.isoformat(), "yhat": 1.0, "lo_80": 0.0, "hi_80": 2.0,
         "lo_95": 0.0, "hi_95": 3.0}
        for uid in ids for d in dates
    ]


@pytest.mark.parametrize(
    "rows",
    [
        _grid(["Brooklyn", "Manhattan", "Queens"], "2025-09-01", "MS", 1),  # one period late
        _grid(["Brooklyn", "Manhattan", "Queens"], "2026-11-01", "MS", 1),  # "today"
        _grid(["Brooklyn", "Manhattan", "Queens"], "2025-07-01", "MS", 1),  # origin, not after
        _grid(["Brooklyn", "Manhattan"], "2025-08-01", "MS", 1),  # Queens has no rows
        [],
    ],
)
def test_forecast_shape_checks_the_date_grid(rows):
    out = outputs_with(handles=with_input({"r_1": forecast_meta(forecast=rows)}))
    result = grade_checks(out, forecast_ref())
    assert result["score"] is False, result["comment"]


def test_forecast_shape_daily_grid_steps_by_day():
    ref = forecast_ref(
        gold_check=["forecast_shape('daily_trips', 'D', 7)"],
        gold=[{"period": "2025-08-01", "trips": 1.0}],
    )

    def meta(rows):
        return forecast_meta(freq="D", horizon=7, headline_ids=["series"], forecast=rows)

    good = _grid(["series"], "2025-08-01", "D", 7)
    assert grade_checks(outputs_with(handles=with_input({"r_1": meta(good)})), ref)["score"] is True
    gappy = _grid(["series"], "2025-08-01", "2D", 7)
    assert grade_checks(outputs_with(handles=with_input({"r_1": meta(gappy)})), ref)["score"] is False
    late = _grid(["series"], "2025-08-02", "D", 7)
    assert grade_checks(outputs_with(handles=with_input({"r_1": meta(late)})), ref)["score"] is False
    short = _grid(["series"], "2025-08-01", "D", 6)
    assert grade_checks(outputs_with(handles=with_input({"r_1": meta(short)})), ref)["score"] is False


def test_expected_grid_matches_a_real_service_forecast(synthetic_dataset, tmp_path):
    from datetime import date

    from analyst.data.service import DataService
    from analyst.semantic.compiler import MetricQuery
    from evals.graders.agent import _expected_grid

    svc = DataService(synthetic_dataset, cutoff=date(2024, 2, 15), store_root=tmp_path)
    handle = svc.query_metrics(MetricQuery(metrics=["event_count"], grain="day")).handle_id
    card = svc.forecast(handle, value_col="event_count", freq="D", horizon=3)
    days = sorted(str(d)[:10] for d in svc.store.load(card.handle_id).column("ds").to_pylist())
    assert days == _expected_grid(date(2024, 2, 15), "D", 3) == ["2024-02-16", "2024-02-17", "2024-02-18"]
    assert _expected_grid(date(2025, 7, 31), "MS", 2) == ["2025-08-01", "2025-09-01"]
    assert _expected_grid(date(2025, 12, 31), "MS", 2) == ["2026-01-01", "2026-02-01"]


def test_ratio_stated_as_percent_keeps_the_gold_precision_rule():
    ref = _change_ref(0.130072, None, unit="ratio", label="tip_rate")  # a026
    assert grade_answer_values(outputs_with(narrative="Tips were 13% of fares."), ref)["score"] is True
    assert grade_answer_values(outputs_with(narrative="Tips were 0.13 of fares."), ref)["score"] is True
    assert grade_answer_values(outputs_with(narrative="Tips were 10% of fares."), ref)["score"] is False


@pytest.mark.parametrize(
    "narrative",
    ["There were 5.23% fewer trips.", "A 5.23% reduction in trips.", "Trips were reduced by 5.23%."],
)
def test_answer_values_fewer_and_reduction_are_decreases(narrative):  # follow-up item 17
    assert grade_answer_values(outputs_with(narrative=narrative), _change_ref(-5.23))["score"] is True
    assert grade_answer_values(outputs_with(narrative=narrative), _change_ref(5.23))["score"] is False
