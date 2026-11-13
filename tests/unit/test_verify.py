from datetime import date

import pyarrow as pa
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
