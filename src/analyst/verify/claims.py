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


#: words that state a cause; a sentence with one is a causal statement.
CAUSAL_CUES: tuple[str, ...] = (
    "because",
    "due to",
    "caused",
    "driven by",
    "led to",
    "as a result of",
    "owing to",
    "thanks to",
    "resulted in",
    "attributable to",
)


def _words(phrases: tuple[str, ...]) -> re.Pattern[str]:
    alternatives = "|".join(re.escape(p).replace(r"\ ", r"\s+") for p in phrases)
    return re.compile(r"\b(?:" + alternatives + r")\b", re.IGNORECASE)


#: reason for an answer with no claims and no stated number (a decline, an
#: acknowledgement such as "background task started", a clarifying question).
#: It is never verified, and the graph does not retry it.
NO_CLAIMS = "no claims (decline or acknowledgement)"

_HEDGE_RE = _words(HEDGES)
_CUE_RE = _words(CAUSAL_CUES)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")


def is_hedged(text: str) -> bool:
    return _HEDGE_RE.search(text) is not None


def causal_sentences(text: str) -> list[str]:
    """Sentences of ``text`` that contain a causal cue word."""
    return [s for s in _SENTENCE_END.split(text) if _CUE_RE.search(s)]


def unsupported_causes(
    texts: list[str], cited_ids: set[str], allowed_ids: set[str]
) -> list[str]:
    """Causal sentences that are not hedged, or that no allowed cited id licenses.

    A sentence with a causal cue needs a hedge, and the answer must cite (in
    some claim's ``fact_ids``) at least one id from ``allowed_ids``.
    """
    licensed = bool(cited_ids & allowed_ids)
    return [
        sentence
        for text in texts
        for sentence in causal_sentences(text)
        if not (licensed and is_hedged(sentence))
    ]


def allowed_fact_ids(memory: Memory, cutoff: date | None) -> set[str]:
    """Ids that may license a hedged cause: context facts and known data quirks
    (spec §5.7 lists known data gaps as allowed facts), as of ``cutoff``."""
    known = memory.as_of(cutoff)
    return {f.id for f in known.context_facts} | {q.id for q in known.quirks}


def verify_answer(
    answer: Answer,
    *,
    question: str,
    handles: dict[str, HandleInfo],
    memory: Memory,
    cutoff: date | None,
    store: ResultStore,
) -> Verification:
    if not answer.claims and not extract_numbers(answer.narrative, exclude=question):
        # Nothing claimed or stated: not an answer about the data, so the
        # causal-cue scan does not apply ("because the data ends on ...").
        return Verification(label="not_verified", reason=NO_CLAIMS, problems=[NO_CLAIMS])
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
        if not cited:
            problems.append("claim cites no evidence")
        claim_handles.append(cited)
    for chart in answer.charts:
        for handle_id in (chart.handle, chart.forecast_handle, chart.last_year_handle):
            if handle_id:
                resolve(handle_id)

    fact_ids = allowed_fact_ids(memory, cutoff)
    for claim in answer.claims:
        if claim.type != "causal":
            continue
        if not is_hedged(claim.text) or not fact_ids & set(claim.fact_ids):
            problems.append("unsupported causal claim")
    # Causes stated in prose (the narrative and every claim's text, per
    # sentence) are checked too.
    cited_ids = {fid for claim in answer.claims for fid in claim.fact_ids}
    prose = [answer.narrative] + [c.text for c in answer.claims]
    if (
        unsupported_causes(prose, cited_ids, fact_ids)
        and "unsupported causal claim" not in problems
    ):
        problems.append("unsupported causal claim")

    # Claim values trace (one hop allowed) to the handles that claim cites. Claim
    # text traces the same way, else directly to any collected handle. The
    # narrative traces directly to any collected handle. Either may also equal a
    # traced claim value, which is how a derived number reaches the prose.
    # "Collected" means this turn's handles (the caller's ``handles``) plus the
    # handles this answer's claims cite.
    cited_by_claims = {h.handle_id: h for cited in claim_handles for h in cited}
    everything = list({**handles, **cited_by_claims}.values())
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

    if not answer.claims:
        # Numbers stated but nothing cited: never verified.
        problems.insert(0, "no evidence cited")

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
