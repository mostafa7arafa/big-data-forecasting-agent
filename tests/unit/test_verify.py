from datetime import date

import pyarrow as pa
import pytest
from langchain_core.messages import HumanMessage, ToolMessage

from analyst.data.results import ResultStore
from analyst.semantic.model import (
    ContextFact,
    Memory,
    SemanticModel,
    TimeDimension,
)
from analyst.state import Answer, Claim
from analyst.verify import (
    HandleInfo,
    collect_handles,
    extract_numbers,
    matches,
    trace_value,
    verify_answer,
)


def sn(text: str):
    (number,) = extract_numbers(text)
    return number


def info(handle_id, values, produced_by="query_metrics", governed=True):
    return HandleInfo(handle_id, produced_by, governed, list(values))


def make_memory() -> Memory:
    return Memory(
        semantic=SemanticModel(
            table="t",
            time_dimension=TimeDimension(name="d", column="d", grains=["day"]),
        ),
        quirks=[],
        series=[],
        context_facts=[
            ContextFact(
                date=date(2025, 1, 5),
                label="Congestion pricing starts",
                kind="policy",
            ),
            ContextFact(date=date(2026, 6, 1), label="Late event", kind="policy"),
        ],
        recap="",
    )


def save(store, values, produced_by="query_metrics", governed=True, tag=""):
    card = store.save(
        pa.table({"v": pa.array(values, type=pa.float64())}),
        produced_by=produced_by,
        source=f"{produced_by}{values}{tag}",
        governed=governed,
        elapsed_s=0.0,
    )
    return card.handle_id


def run(answer, handles, store, question="", cutoff=date(2026, 3, 1)):
    return verify_answer(
        answer,
        question=question,
        handles=handles,
        memory=make_memory(),
        cutoff=cutoff,
        store=store,
    )


def test_extract_human_formats():
    vals = [
        (n.value, n.scale, n.percent)
        for n in extract_numbers("$2.81M, 2,894,704 trips, −3.0%, 3.4 million, 45k")
    ]
    assert vals == [
        (2.81, 1e6, False),
        (2894704, 1, False),
        (-3.0, 1, True),
        (3.4, 1e6, False),
        (45, 1e3, False),
    ]


def test_calendar_numbers_ignored():
    text = "In Oct 2026 over 12 months since 2023-09-21, a 4-day gap"
    assert extract_numbers(text) == []
    assert extract_numbers("on March 3rd and 21st Oct, the 5th") == []


def test_rounding_rule():
    assert matches(sn("2.81M"), 2_808_154) and not matches(sn("2.82M"), 2_808_154)
    assert matches(sn("-3.0%"), -0.02990) and matches(sn("-3.0%"), -2.99)
    assert not matches(sn("2,894,705"), 2_894_704)
    assert matches(sn("2,894,704"), 2_894_704)


def test_cross_handle_pct_change_traces():
    forecast = info("r_aaaaaaaaaa", [2_808_154], "forecast")
    last_year = info("r_bbbbbbbbbb", [2_894_704], "query_metrics")
    assert trace_value(sn("-3.0%"), [forecast, last_year])
    assert trace_value(sn("86,550"), [forecast, last_year])
    assert not trace_value(sn("-3.5%"), [forecast, last_year])


def test_two_hop_does_not_trace():
    a, b, c, d = 1000.0, 700.0, 330.0, 110.0
    ratio_change = (a / b - c / d) / (c / d) * 100  # -52.4
    handle = info("r_aaaaaaaaaa", [a, b, c, d])
    assert not trace_value(sn(f"{ratio_change:.1f}%"), [handle])


def test_foreign_handle_untraced(tmp_path):
    store = ResultStore(tmp_path)
    answer = Answer(
        narrative="Trips were steady.",
        claims=[Claim(text="steady", type="descriptive", handles=["r_ffffffffff"])],
    )
    result = run(answer, {}, store)
    assert result.label == "not_verified"
    assert result.reason == "unknown handle r_ffffffffff"


def test_store_handle_loaded_when_not_collected(tmp_path):
    store = ResultStore(tmp_path)
    h = save(store, [2_808_154.0])
    answer = Answer(
        narrative="Volume was 2.81M.",
        claims=[Claim(text="v", type="descriptive", values=[2_808_154], handles=[h])],
    )
    assert run(answer, {}, store).label == "verified"


def test_causal_needs_hedge_and_fact(tmp_path):
    store = ResultStore(tmp_path)
    h = save(store, [1.0])
    ok_id = "policy:2025-01-05"

    def answer(text, fact_ids):
        return Answer(
            narrative="Trips fell.",
            claims=[Claim(text=text, type="causal", handles=[h], fact_ids=fact_ids)],
        )

    bad = run(answer("fell because of the storm", []), {}, store)
    assert bad.label == "not_verified" and bad.reason == "unsupported causal claim"
    no_hedge = run(answer("fell because of congestion pricing", [ok_id]), {}, store)
    assert no_hedge.reason == "unsupported causal claim"
    good = run(answer("this coincides with congestion pricing", [ok_id]), {}, store)
    assert good.label == "verified"
    late = run(
        answer("this coincides with a late event", ["policy:2026-06-01"]), {}, store
    )
    assert late.reason == "unsupported causal claim"
    unknown = run(answer("this may reflect it", ["policy:2020-01-01"]), {}, store)
    assert unknown.reason == "unsupported causal claim"


def test_labels_governed_vs_off_model(tmp_path):
    store = ResultStore(tmp_path)
    h1 = save(store, [10.0])
    h2 = save(store, [20.0], "run_sql", False)
    handles = {
        h1: info(h1, [10.0]),
        h2: info(h2, [20.0], "run_sql", False),
    }

    def answer(*cited):
        return Answer(
            narrative="ok",
            claims=[Claim(text="c", type="descriptive", handles=list(cited))],
        )

    assert run(answer(h1), handles, store).label == "verified"
    off = run(answer(h1, h2), handles, store)
    assert off.label == "verified_off_model"
    assert off.off_model_tools == ["run_sql"]


def test_no_numbers_verified_when_cited(tmp_path):
    store = ResultStore(tmp_path)
    h = save(store, [1.0])
    answer = Answer(
        narrative="Nothing notable.",
        claims=[Claim(text="nothing", type="descriptive", handles=[h])],
    )
    result = run(answer, {}, store)
    assert result.label == "verified" and result.problems == []


def test_untraced_number_fails(tmp_path):
    store = ResultStore(tmp_path)
    h = save(store, [100.0])
    answer = Answer(
        narrative="Revenue was 987.",
        claims=[Claim(text="c", type="descriptive", handles=[h])],
    )
    result = run(answer, {}, store)
    assert result.label == "not_verified" and "987" in result.reason


def test_question_numbers_excluded(tmp_path):
    store = ResultStore(tmp_path)
    h = save(store, [100.0])
    answer = Answer(
        narrative="Here are the top 5 zones in 2025.",
        claims=[Claim(text="c", type="descriptive", handles=[h])],
    )
    assert run(answer, {}, store, question="top 5 zones in 2025").label == "verified"
    assert run(answer, {}, store, question="zones").label == "not_verified"


def test_worker_handles_line_collected(tmp_path):
    store = ResultStore(tmp_path)
    h1 = save(store, [1.0])
    h2 = save(store, [2.0], "run_sql", False)
    note = HumanMessage(
        content=f"Task done.\nhandles: {h1}, {h2}",
        additional_kwargs={"source": "async_subagent"},
    )
    plain = HumanMessage(content=f"handles: {h1}")
    tool = ToolMessage(
        content="card",
        tool_call_id="x",
        artifact={"handle_id": h1, "produced_by": "query_metrics", "governed": True},
    )
    chart = ToolMessage(content="c", tool_call_id="y", artifact={"chart": {}})
    got = collect_handles([note], store)
    assert set(got) == {h1, h2}
    assert got[h2].produced_by == "run_sql" and not got[h2].governed
    assert got[h1].values == [1.0]
    assert collect_handles([plain, chart], store) == {}
    meta = HumanMessage(
        content=f"handles: {h1}", response_metadata={"source": "async_subagent"}
    )
    assert set(collect_handles([meta, tool], store)) == {h1}


def test_claim_float_precision_traces(tmp_path):
    store = ResultStore(tmp_path)
    a, b = 2808154.0, 2894704.0
    h = save(store, [a, b])
    value = (a / b - 1) * 100
    answer = Answer(
        narrative="ok",
        claims=[Claim(text="c", type="comparative", values=[value], handles=[h])],
    )
    assert run(answer, {}, store).label == "verified"
    tiny = save(store, [1e-05], tag="tiny")
    answer = Answer(
        narrative="ok",
        claims=[Claim(text="c", type="descriptive", values=[1e-05], handles=[tiny])],
    )
    assert run(answer, {}, store).label == "verified"
    wrong = Answer(
        narrative="ok",
        claims=[
            Claim(text="c", type="descriptive", values=[1.0001e-05], handles=[tiny])
        ],
    )
    assert run(wrong, {}, store).label == "not_verified"


def test_dates_and_times_are_not_numbers():
    for text in [
        "in 2025-10 trips rose",
        "for 10/2025",
        "Jan 1–7 saw",
        "at 10:30",
        "at 10:30 pm",
        "after 1,234 days",
    ]:
        assert extract_numbers(text) == [], text


def test_decimal_durations_are_values():
    assert [n.value for n in extract_numbers("trips took 14.2 minutes")] == [14.2]
    assert [n.value for n in extract_numbers("about 3.5 hours")] == [3.5]
    for text in ["28 days", "3 weeks", "2 years", "24 hours", "12 months", "4-day gap"]:
        assert extract_numbers(text) == [], text


def test_derived_only_between_cited_handles_columns():
    import random

    random.seed(0)
    vals = [float(random.randint(300000, 500000)) for _ in range(300)]
    big = info("r_bbbbbbbbbb", vals)
    for fake in ["12%", "7%", "1.3", "17"]:
        assert not trace_value(sn(fake), [big], derive=False)
    # two columns of one handle are never paired
    two = HandleInfo(
        "r_cccccccccc", "query_metrics", True, [10.0, 20.0], [[10.0], [20.0]]
    )
    assert not trace_value(
        sn("10"), [HandleInfo("r_d", "q", True, [20.0], [[20.0]])], derive=False
    )
    assert not trace_value(sn("-10"), [two]) and trace_value(
        sn("-10"), [info("r_x", [10.0]), info("r_y", [20.0])]
    )


def test_narrative_derived_numbers_need_a_claim(tmp_path):
    import random

    random.seed(0)
    store = ResultStore(tmp_path)
    vals = [float(random.randint(300000, 500000)) for _ in range(300)]
    h = save(store, vals)
    answer = Answer(
        narrative="Trips moved 12% and 7%, a ratio of 1.3 over 17 days.",
        claims=[Claim(text="trips", type="descriptive", handles=[h])],
    )
    result = run(answer, {}, store)
    assert result.label == "not_verified"
    assert {"untraced number 12%", "untraced number 7%", "untraced number 1.3"} <= set(
        result.problems
    )


def test_narrative_derived_number_from_claim_value(tmp_path):
    store = ResultStore(tmp_path)
    h = save(store, [2808154.0, 2894704.0])
    answer = Answer(
        narrative="Volume fell 3.0%.",
        claims=[
            Claim(
                text="drop",
                type="comparative",
                values=[(2808154.0 - 2894704.0) / 2894704.0 * 100],
                handles=[h],
            )
        ],
    )
    assert run(answer, {}, store).label == "verified"


def test_head_and_tail_cells_kept(tmp_path):
    store = ResultStore(tmp_path)
    h = save(store, [float(i) for i in range(5000)])
    from analyst.verify import load_handle_info

    got = load_handle_info(store, h)
    assert 0.0 in got.values and 4999.0 in got.values and len(got.values) <= 2000


def test_hedge_word_boundaries(tmp_path):
    store = ResultStore(tmp_path)
    h = save(store, [1.0])
    answer = Answer(
        narrative="x",
        claims=[
            Claim(
                text="it is impossible to say",
                type="causal",
                handles=[h],
                fact_ids=["policy:2025-01-05"],
            )
        ],
    )
    assert run(answer, {}, store).reason == "unsupported causal claim"


def test_scale_suffix_variants():
    got = [
        (n.value, n.scale)
        for n in extract_numbers("1.2bn, 2.8m, 3 Million, 4 BILLION, 5 Bn")
    ]
    assert got == [(1.2, 1e9), (2.8, 1e6), (3, 1e6), (4, 1e9), (5, 1e9)]


def test_spaced_single_letter_scales():
    got = [
        (n.value, n.scale)
        for n in extract_numbers("2.8 M trips, 5 K, 1.2 B rides, 3 m")
    ]
    assert got == [(2.8, 1e6), (5, 1e3), (1.2, 1e9), (3, 1e6)]


def test_derivation_cap():
    import random

    random.seed(0)
    big = info(
        "r_bbbbbbbbbb", [float(random.randint(300000, 500000)) for _ in range(300)]
    )
    for fake in ["12%", "7%", "1.3", "3.14"]:
        assert not trace_value(sn(fake), [big])
    forecast = info(
        "r_aaaaaaaaaa", [2_808_154.0 + i * 1000 for i in range(8)], "forecast"
    )
    last = info("r_bbbbbbbbbb", [2_894_704.0 + i * 1000 for i in range(8)])
    assert trace_value(sn("-3.0%"), [forecast, last])


def test_untraceable_perf_and_cell_cap(tmp_path):
    import random
    import time

    random.seed(1)
    big = [
        info(f"r_{i:010x}", [random.random() * 1e6 for _ in range(2000)])
        for i in range(5)
    ]
    start = time.time()
    assert not trace_value(sn("-123,456,789.123"), big)
    assert time.time() - start < 0.6
    from analyst.verify.trace import MAX_CELLS, _numeric_columns

    wide = pa.table({f"c{i}": [1.0, 2.0, 3.0, 4.0] for i in range(1500)})
    assert sum(len(c) for c in _numeric_columns(wide)) <= MAX_CELLS


def test_clock_times_with_am_pm():
    for text in ["at 10am", "at 10 am", "at 10 pm", "10 PM"]:
        assert extract_numbers(text) == [], text


# --- final fix wave ------------------------------------------------------------


def _forecast_handle(store):
    import datetime as dt

    card = store.save(
        pa.table(
            {
                "unique_id": ["series"],
                "ds": [dt.date(2025, 8, 1)],
                "yhat": [2_912_345.0],
                "lo_80": [2_701_000.0],
                "hi_80": [3_105_000.0],
                "lo_95": [2_598_000.0],
                "hi_95": [3_220_000.0],
            }
        ),
        produced_by="forecast",
        source="f",
        governed=True,
        elapsed_s=0.0,
        meta={"levels": [80, 95]},
    )
    return card.handle_id


def _forecast_claim(h):
    return Claim(
        text="August forecast",
        type="forecast",
        values=[2_912_345.0, 2_701_000.0, 3_105_000.0],
        handles=[h],
    )


def test_forecast_interval_levels_are_traceable(tmp_path):
    from analyst.verify import load_handle_info

    store = ResultStore(tmp_path)
    h = _forecast_handle(store)
    handles = {h: load_handle_info(store, h)}
    q = "Forecast total trips for August 2025."
    for narrative in [
        "August is forecast at 2.91M trips (80% interval 2.70M to 3.11M).",
        (
            "August is forecast at 2.91M trips, with an 80% interval of 2.70M to "
            "3.11M and a 95% interval of 2.60M to 3.22M."
        ),
        "August is forecast at 2.91M trips; the 80% band is 2.70M to 3.11M.",
        (
            "August is forecast at 2.91M trips, between 2.70M and 3.11M with 80% "
            "probability."
        ),
    ]:
        result = run(
            Answer(narrative=narrative, claims=[_forecast_claim(h)]),
            handles,
            store,
            question=q,
        )
        assert result.label == "verified", (narrative, result.problems)


def test_interval_level_phrases_are_not_claims():
    for text in [
        "80% interval",
        "95% prediction interval",
        "80% band",
        "95% PI",
        "95% confidence",
    ]:
        assert extract_numbers(text) == [], text
    # a level that is not an interval phrase is still a claim
    assert [n.value for n in extract_numbers("80% of trips")] == [80]


def test_levels_do_not_trace_without_a_forecast_handle(tmp_path):
    store = ResultStore(tmp_path)
    h = save(store, [2_912_345.0])
    answer = Answer(
        narrative="Card payments were 95% of trips.",
        claims=[Claim(text="c", type="descriptive", values=[2_912_345.0], handles=[h])],
    )
    assert run(answer, {}, store).label == "not_verified"


def test_question_number_words_excluded():
    got = extract_numbers("The top 5 zones were", exclude="What were the top five zones?")
    assert got == []
    got = extract_numbers("The top 12 zones", exclude="Show the Twelve busiest zones")
    assert got == []
    assert [n.value for n in extract_numbers("The top 5 zones", exclude="top zones")] == [5]


def test_month_name_does_not_swallow_numbers():
    def raws(text):
        return [n.raw for n in extract_numbers(text)]

    assert raws("In March 3.2M trips were recorded") == ["3.2M"]
    assert raws("In March 3,412,005 trips were recorded") == ["3,412,005"]
    assert raws("In March, 3.2M trips") == ["3.2M"]
    assert raws("In May 45k trips") == ["45k"]
    assert raws("In June 12% more trips") == ["12%"]
    assert raws("In March 45 zones were busy") == ["45"]
    assert raws("On Feb 23 25,812 trips") == ["25,812"]
    assert raws("On Feb 23 2894704 trips") == ["2894704"]
    assert raws("March 3, 2025 had 25,812 trips") == ["25,812"]
    assert raws("Jan 1–7 saw 3.1M trips") == ["3.1M"]
    assert raws("Jan 1–7, 2025: 3.1M trips") == ["3.1M"]


def test_answer_without_claims_is_not_verified(tmp_path):
    store = ResultStore(tmp_path)
    plain = run(Answer(narrative="JFK is the busiest pickup zone.", claims=[]), {}, store)
    assert plain.label == "not_verified"
    assert plain.reason == "no claims (decline or acknowledgement)"
    h = save(store, [2_894_704.0])
    handles = {h: info(h, [2_894_704.0])}
    stated = run(Answer(narrative="There were 2,894,704 trips.", claims=[]), handles, store)
    assert stated.label == "not_verified" and stated.reason == "no evidence cited"


def test_decline_is_not_verified_declined(tmp_path):
    store = ResultStore(tmp_path)
    answer = Answer(
        narrative="The data ends on 2026-01-31, so March 2026 is not available.",
        claims=[],
    )
    result = run(answer, {}, store, question="How many trips in March 2026?")
    assert result.label == "not_verified"
    assert result.reason == "no claims (decline or acknowledgement)"


def test_claim_without_handles_is_not_verified(tmp_path):
    store = ResultStore(tmp_path)
    answer = Answer(
        narrative="Manhattan dominates.",
        claims=[Claim(text="Manhattan is busiest", type="descriptive")],
    )
    result = run(answer, {}, store)
    assert result.label == "not_verified" and "claim cites no evidence" in result.problems


def test_collect_handles_skips_artifacts_missing_from_store(tmp_path):
    store = ResultStore(tmp_path)
    tool = ToolMessage(
        content="card",
        tool_call_id="x",
        artifact={"handle_id": "r_0123456789", "produced_by": "query_metrics", "governed": True},
    )
    assert collect_handles([tool], store) == {}


def _memory_with_quirk() -> Memory:
    from analyst.semantic.model import Quirk

    memory = make_memory()
    quirk = Quirk(kind="gap", start=date(2023, 9, 21), end=date(2023, 9, 24))
    late = Quirk(kind="anomaly", start=date(2026, 6, 1))
    return memory.model_copy(update={"quirks": [quirk, late]})


def _run_q(answer, store, cutoff=date(2026, 3, 1)):
    return verify_answer(
        answer,
        question="",
        handles={},
        memory=_memory_with_quirk(),
        cutoff=cutoff,
        store=store,
    )


def test_causal_cue_in_narrative_is_checked(tmp_path):
    store = ResultStore(tmp_path)
    h = save(store, [2_894_704.0])
    answer = Answer(
        narrative="Trips fell to 2.89M because of the snowstorm.",
        claims=[Claim(text="Trips were 2.89M", type="descriptive", values=[2_894_704.0], handles=[h])],
    )
    result = _run_q(answer, store)
    assert result.label == "not_verified"
    assert "unsupported causal claim" in result.problems


@pytest.mark.parametrize(
    "text",
    [
        "Trips fell, due to the storm.",
        "The drop was CAUSED by weather.",
        "Demand was driven by tourism.",
        "This led to fewer trips.",
        "Fewer trips as a result of the storm.",
        "Fewer trips owing to rain.",
        "More trips thanks to the parade.",
        "The storm resulted in fewer trips.",
        "The drop is attributable to rain.",
    ],
)
def test_causal_cues_in_claim_text(tmp_path, text):
    store = ResultStore(tmp_path)
    h = save(store, [1.0])
    answer = Answer(
        narrative="Trips changed.",
        claims=[Claim(text=text, type="descriptive", handles=[h])],
    )
    assert "unsupported causal claim" in _run_q(answer, store).problems


def test_hedged_cause_needs_a_valid_cited_fact(tmp_path):
    store = ResultStore(tmp_path)
    h = save(store, [1.0])

    def answer(narrative, fact_ids):
        return Answer(
            narrative=narrative,
            claims=[Claim(text="Trips changed", type="descriptive", handles=[h], fact_ids=fact_ids)],
        )

    hedged = "The drop may be due to congestion pricing."
    assert _run_q(answer(hedged, []), store).label == "not_verified"
    assert _run_q(answer(hedged, ["policy:2025-01-05"]), store).label == "verified"
    late = answer(hedged, ["policy:2026-06-01"])
    assert _run_q(late, store).label == "not_verified"
    unhedged = answer("The drop was due to congestion pricing.", ["policy:2025-01-05"])
    assert _run_q(unhedged, store).label == "not_verified"


def test_quirk_ids_license_hedged_causes(tmp_path):
    store = ResultStore(tmp_path)
    h = save(store, [1.0])
    answer = Answer(
        narrative="The drop coincides with the known data gap.",
        claims=[
            Claim(
                text="The drop coincides with the known data gap",
                type="causal",
                handles=[h],
                fact_ids=["gap:2023-09-21"],
            )
        ],
    )
    assert _run_q(answer, store).label == "verified"
    late = answer.model_copy(
        update={
            "claims": [
                answer.claims[0].model_copy(update={"fact_ids": ["anomaly:2026-06-01"]})
            ]
        }
    )
    assert _run_q(late, store).reason == "unsupported causal claim"


def test_causal_cue_word_boundaries(tmp_path):
    store = ResultStore(tmp_path)
    h = save(store, [1.0])
    answer = Answer(
        narrative="Becauseless prose; the overdue report is ready.",
        claims=[Claim(text="ok", type="descriptive", handles=[h])],
    )
    assert _run_q(answer, store).label == "verified"


def test_causal_claim_text_is_scanned_per_sentence(tmp_path):
    store = ResultStore(tmp_path)
    h = save(store, [1.0])
    answer = Answer(
        narrative="Trips changed.",
        claims=[
            Claim(
                text="This may reflect the fee. Trips fell because of it.",
                type="causal",
                handles=[h],
                fact_ids=["policy:2025-01-05"],
            )
        ],
    )
    assert _run_q(answer, store).label == "not_verified"


def test_decline_with_a_causal_cue_is_not_a_causal_claim(tmp_path):
    store = ResultStore(tmp_path)
    answer = Answer(
        narrative="I can't answer that because the data ends on 2026-01-31.",
        claims=[],
    )
    result = _run_q(answer, store)
    assert result.label == "not_verified"
    assert result.reason == "no claims (decline or acknowledgement)"
    assert "unsupported causal claim" not in result.problems
    # with a number stated it is an answer, and the scan applies
    stated = Answer(narrative="Trips fell to 2.89M because of the storm.", claims=[])
    assert "unsupported causal claim" in _run_q(stated, store).problems
