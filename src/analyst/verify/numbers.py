"""Pull human-formatted numbers out of prose and compare them at stated precision."""

import math
import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

_SCALES = {
    "k": 1e3,
    "m": 1e6,
    "mn": 1e6,
    "million": 1e6,
    "b": 1e9,
    "bn": 1e9,
    "billion": 1e9,
}

_MONTH = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?"
    r"|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)

# A day of the month after a month name: 1-31, and not the start of a larger
# number ("March 3,412,005", "March 3.2M", "May 5k", "June 12%").
_DAY = (
    r"(?:3[01]|[12]\d|0?[1-9])(?!\d)(?:st|nd|rd|th)?"
    r"(?![.,]\d)(?!\s?(?:%|(?i:percent|million|billion|mn|bn|k|m|b)\b))"
)

_NUMBER_WORDS = {
    word: str(i)
    for i, word in enumerate(
        [
            "one", "two", "three", "four", "five", "six", "seven", "eight",
            "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
            "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
        ],
        start=1,
    )
}
_NUMBER_WORD = re.compile(r"\b(?:" + "|".join(_NUMBER_WORDS) + r")\b", re.IGNORECASE)

# Calendar-ish tokens that are never claims; blanked out before extraction.
_MASKS = [
    # Interval levels ("80% interval", "95% prediction interval") name the band.
    re.compile(
        r"\b\d{2}%\s+(?:prediction\s+)?(?:interval|band|PI|confidence)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
    re.compile(r"\b\d{4}-\d{1,2}\b"),
    re.compile(r"\b\d{1,2}/\d{4}\b"),
    re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?(?:\s?[ap]m\b)?", re.IGNORECASE),
    re.compile(r"\b\d{1,2}\s?[ap]m\b", re.IGNORECASE),
    re.compile(
        rf"\b{_MONTH}\.? {_DAY}(?:\s?[-\u2013]\s?{_DAY})?(?:,? \d{{4}}(?!\d))?"
    ),
    re.compile(rf"(?<![\d.,])\b\d{{1,2}}(?:st|nd|rd|th)?(?: of)? {_MONTH}\b"),
    re.compile(r"\b\d+(?:st|nd|rd|th)\b"),
    # Integer window phrases only ("12 months", "4-day gap"); decimals such as
    # "14.2 minutes" are metric values and stay.
    re.compile(
        r"(?<![\d.,])(?:\d{1,3}(?:,\d{3})+|\d+)(?![\d.]*\d)[- ]"
        r"(?:hour|day|week|month|year)s?\b",
        re.IGNORECASE,
    ),
]

_NUMBER = re.compile(
    r"(?<![\w.])(?<![A-Za-z]-)"
    r"(?P<sign>[-−–+])?(?P<dollar>\$)?"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:\s?(?:(?P<suf>%|(?i:percent|billion|million|bn|mn)\b)|(?P<suf1>[kKmMB]\b)))?"
)


@dataclass(frozen=True)
class StatedNumber:
    value: float
    decimals: int
    scale: float
    percent: bool
    raw: str


def _mask(text: str) -> str:
    for pattern in _MASKS:
        text = pattern.sub(lambda m: " " * len(m.group(0)), text)
    return text


def _extract(text: str) -> list[StatedNumber]:
    out: list[StatedNumber] = []
    for m in _NUMBER.finditer(_mask(text)):
        digits = m.group("num")
        suffix = m.group("suf") or m.group("suf1")
        if (
            suffix is None
            and m.group("sign") is None
            and m.group("dollar") is None
            and re.fullmatch(r"\d{4}", digits)
            and 1900 <= int(digits) <= 2100
        ):
            continue
        value = float(digits.replace(",", ""))
        if m.group("sign") in ("-", "−", "–"):
            value = -value
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        percent = suffix is not None and suffix.lower() in ("%", "percent")
        out.append(
            StatedNumber(
                value=value,
                decimals=decimals,
                scale=_SCALES.get(suffix.lower(), 1) if suffix else 1,
                percent=percent,
                raw=m.group(0).strip(),
            )
        )
    return out


def extract_numbers(text: str, *, exclude: str = "") -> list[StatedNumber]:
    """Numbers stated in ``text``, minus calendar tokens and those in ``exclude``."""
    numbers = _extract(text)
    if not exclude:
        return numbers
    # "top five zones" in the question exempts "top 5" in the answer.
    exclude = _NUMBER_WORD.sub(lambda m: _NUMBER_WORDS[m.group(0).lower()], exclude)
    skip = {_key(n) for n in _extract(exclude)}
    return [n for n in numbers if _key(n) not in skip]


def _key(n: StatedNumber) -> float:
    return round(abs(n.value * n.scale), 6)


def _round(x: float, decimals: int) -> Decimal:
    return Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-decimals), ROUND_HALF_UP)


def matches(stated: StatedNumber, actual: float) -> bool:
    """Whether ``actual`` rounds to the stated number at the stated precision.

    ``actual`` is divided by the stated scale (k = 1e3, M/million = 1e6,
    B/billion = 1e9), rounded half-up to ``stated.decimals`` and compared. For a
    percent, an ``actual`` that is a fraction (|actual| <= 1.5) is also tried
    times 100, so "-3.0%" matches -0.0299 and -2.99. An unsigned stated number
    also matches a negative actual ("fell 3.0%"). Full integers must match
    exactly.
    """
    if actual is None or not math.isfinite(actual):
        return False
    bases = [actual]
    if stated.percent and abs(actual) <= 1.5:
        bases.append(actual * 100)
    if stated.value > 0:
        bases += [-b for b in bases]
    target = Decimal(repr(float(stated.value)))
    for base in bases:
        scaled = base / stated.scale
        if _round(scaled, stated.decimals) == target:
            return True
    return False


def intervals(stated: StatedNumber) -> list[tuple[float, float]]:
    """Closed intervals of actual values that could match (a pre-filter for matches)."""
    half = 0.5 * 10**-stated.decimals
    spans = [(stated.value - half, stated.value + half)]
    if stated.value > 0:
        spans.append((-stated.value - half, -stated.value + half))
    out = []
    for lo, hi in spans:
        lo, hi = lo * stated.scale, hi * stated.scale
        pad = 1e-9 * max(1.0, abs(lo), abs(hi))
        out.append((lo - pad, hi + pad))
        if stated.percent:
            out.append(((lo - pad) / 100, (hi + pad) / 100))
    return out
