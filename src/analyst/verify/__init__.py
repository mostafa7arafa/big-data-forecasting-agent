"""Deterministic answer verification: number tracing, causal policy, labels."""

from analyst.verify.claims import HEDGES, verify_answer
from analyst.verify.numbers import StatedNumber, extract_numbers, matches
from analyst.verify.trace import (
    HandleInfo,
    collect_handles,
    load_handle_info,
    trace_value,
)

__all__ = [
    "HEDGES",
    "HandleInfo",
    "StatedNumber",
    "collect_handles",
    "extract_numbers",
    "load_handle_info",
    "matches",
    "trace_value",
    "verify_answer",
]
