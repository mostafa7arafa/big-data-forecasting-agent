"""Deterministic verdict on an Answer: number tracing, causal policy, labels."""

from datetime import date

from analyst.data.results import ResultStore
from analyst.semantic.model import Memory
from analyst.state import Answer, Verification
from analyst.verify.numbers import StatedNumber, extract_numbers
from analyst.verify.trace import HandleInfo, load_handle_info, trace_value

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


def _claim_value(value: float) -> list[StatedNumber]:
    text = repr(float(value))
    decimals = 0 if "e" in text else len(text.split(".")[1].rstrip("0"))
    return [
        StatedNumber(float(value), decimals, 1.0, percent, text)
        for percent in (False, True)
    ]


def _trace_claim_value(value: float, cited: list[HandleInfo]) -> bool:
    return any(trace_value(variant, cited) for variant in _claim_value(value))


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
        hedged = any(hedge in text for hedge in HEDGES)
        if not hedged or not any(fid in fact_ids for fid in claim.fact_ids):
            problems.append("unsupported causal claim")

    # Narrative and claim text may trace to any collected handle; claim values
    # only to the handles that claim cites.
    everything = list({**handles, **known}.values())
    used: dict[str, HandleInfo] = {h.handle_id: h for h in known.values()}
    for claim, cited in zip(answer.claims, claim_handles, strict=True):
        for value in claim.values:
            if not _trace_claim_value(value, cited):
                problems.append(f"untraced number {value:g}")
    pools = [
        (claim.text, cited)
        for claim, cited in zip(answer.claims, claim_handles, strict=True)
    ] + [(answer.narrative, list(known.values()))]
    for text, preferred in pools:
        for number in extract_numbers(text, exclude=question):
            if trace_value(number, preferred):
                continue
            if trace_value(number, everything):
                used.update({h.handle_id: h for h in everything})
                continue
            problems.append(f"untraced number {number.raw}")

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
