"""Deterministic verdict on an Answer: number tracing, causal policy, labels."""

import re
from datetime import date

from analyst.data.results import ResultStore
from analyst.semantic.model import Memory
from analyst.state import Answer, Verification
from analyst.verify.numbers import extract_numbers, matches
from analyst.verify.trace import (
    HandleInfo,
    direct_handles,
    load_handle_info,
    trace_float,
    trace_value,
)

HEDGES: tuple[str, ...] = (
    "coincides with",
    "coincided with",
    "may reflect",
    "may be related to",
    "possibly",
    "possible",
    "perhaps",
    "likely related to",
    "consistent with",
    "could be due to",
    "might be",
    "may be",
)


_HEDGE_RE = re.compile(r"\b(?:" + "|".join(re.escape(h) for h in HEDGES) + r")\b")


def verify_answer(
    answer: Answer,
    *,
    question: str,
    handles: dict[str, HandleInfo],
    memory: Memory,
    cutoff: date | None,
    store: ResultStore,
) -> Verification:
    problems: list[str] = []
    known: dict[str, HandleInfo] = {}

    def resolve(handle_id: str) -> HandleInfo | None:
        if handle_id in known:
            return known[handle_id]
        info = handles.get(handle_id) or load_handle_info(store, handle_id)
        if info is not None:
            known[handle_id] = info
        return info

    claim_handles: list[list[HandleInfo]] = []
    for claim in answer.claims:
        cited = []
        for handle_id in claim.handles:
            info = resolve(handle_id)
            if info is None:
                problems.append(f"unknown handle {handle_id}")
            else:
                cited.append(info)
        claim_handles.append(cited)
    for chart in answer.charts:
        for handle_id in (chart.handle, chart.forecast_handle, chart.last_year_handle):
            if handle_id:
                resolve(handle_id)

    fact_ids = {fact.id for fact in memory.as_of(cutoff).context_facts}
    for claim in answer.claims:
        if claim.type != "causal":
            continue
        text = claim.text.lower()
        hedged = _HEDGE_RE.search(text) is not None
        if not hedged or not any(fid in fact_ids for fid in claim.fact_ids):
            problems.append("unsupported causal claim")

    # Claim values trace (one hop allowed) to the handles that claim cites. Claim
    # text traces the same way, else directly to any collected handle. The
    # narrative traces directly to any collected handle. Either may also equal a
    # traced claim value, which is how a derived number reaches the prose.
    everything = list({**handles, **known}.values())
    used: dict[str, HandleInfo] = {h.handle_id: h for h in known.values()}
    claim_values = [v for claim in answer.claims for v in claim.values]
    for claim, cited in zip(answer.claims, claim_handles, strict=True):
        for value in claim.values:
            if not trace_float(value, cited):
                problems.append(f"untraced number {value:g}")

    def trace_prose(text: str, cited: list[HandleInfo] | None) -> None:
        for number in extract_numbers(text, exclude=question):
            if cited is not None and trace_value(number, cited):
                continue
            direct = direct_handles(number, everything)
            if direct:
                used.update({h.handle_id: h for h in direct})
                continue
            if any(matches(number, v) for v in claim_values):
                continue
            problems.append(f"untraced number {number.raw}")

    for claim, cited in zip(answer.claims, claim_handles, strict=True):
        trace_prose(claim.text, cited)
    trace_prose(answer.narrative, None)

    if problems:
        return Verification(label="not_verified", reason=problems[0], problems=problems)
    off_model = sorted({h.produced_by for h in used.values() if not h.governed})
    if off_model:
        return Verification(
            label="verified_off_model",
            reason="every number traced; results from " + ", ".join(off_model),
            off_model_tools=off_model,
        )
    return Verification(label="verified", reason="every number traced")
