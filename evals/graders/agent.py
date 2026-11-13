"""Deterministic graders for the single-turn agent eval set (spec §9.3).

Every grader is ``(outputs, reference) -> {"key", "score", "comment"}``.

``outputs`` is what the eval target returns from the final graph state::

    {"answer": Answer dict, "verification": Verification dict,
     "tools": [tool names in call order], "handles": {handle_id: entry}}

A handle ``entry`` is the summary card's ``meta`` dict plus ``produced_by``
(a nested ``{"meta": {...}, ...}`` entry is flattened too). Forecast entries
should also carry ``forecast``: the forecast table as a list of row dicts
(``unique_id, ds, yhat, lo_80, hi_80, lo_95, hi_95``), which
``forecast_skill`` scores. A forecast entry's ``input_handle`` must also be in
``handles`` (its ``query`` meta names the metric that ``forecast_shape``
checks). ``outputs = {"error": ...}`` fails every gate.

``reference`` is an eval row (see ``evals/datasets/agent_single_turn.jsonl``).
Gold tables load from ``evals/gold/<id>.parquet``; a ``gold`` list of row
dicts in the reference overrides that (tests).

A score of ``None`` means "not applicable to this row". ``GATES`` decide
``passed``; ``trajectory`` and ``forecast_skill`` are scores only.
"""

import ast
import json
import math
import re
from collections.abc import Callable
from datetime import date, datetime
from functools import cache
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pyarrow.types as pa_types
import yaml
from openevals.trajectory.match import create_trajectory_match_evaluator

from analyst.data.dataset import DatasetConfig
from analyst.memory.files import load_memory
from analyst.semantic.model import Memory
from analyst.verify import allowed_fact_ids, unsupported_causes
from analyst.verify.numbers import StatedNumber, extract_numbers, matches

ROOT = Path(__file__).resolve().parents[2]
GOLD_DIR = ROOT / "evals/gold"
DATASET_DIR = ROOT / "datasets/nyc_taxi"
SERIES_FILE = DATASET_DIR / "memory/series.yaml"

CATEGORIES = ("lookup", "comparison", "forecast", "off_model", "trap", "safe")
LABELS = ("verified", "verified_off_model", "not_verified")
GATES = ("answer_values", "label", "grounding", "checks")
REQUIRED_FIELDS = (
    "id", "tier", "category", "question", "cutoff", "gold_sql", "tolerance",
    "gold_check", "expected_tools", "expected_label", "notes",
)
# Claim values are exact data values; allow float noise only (as verify does).
_CLAIM_REL = 1e-9
_HEADLINE_SHARE = 0.01

Result = dict[str, Any]


def _result(key: str, score: bool | float | None, comment: str) -> Result:
    return {"key": key, "score": score, "comment": comment}


# --- check parsing -----------------------------------------------------------


@cache
def _series_specs() -> dict[str, dict]:
    data = yaml.safe_load(SERIES_FILE.read_text())
    return {spec["name"]: spec for spec in data["series"]}


def _validate_forecast_shape(args: tuple) -> None:
    if len(args) != 3:
        raise ValueError("forecast_shape takes (series, freq, horizon)")
    series, freq, horizon = args
    specs = _series_specs()
    if series not in specs:
        raise ValueError(f"forecast_shape: unknown series {series!r}")
    if freq != specs[series]["freq"]:
        raise ValueError(f"forecast_shape: {series} has freq {specs[series]['freq']}, not {freq!r}")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("forecast_shape: horizon must be a positive integer")


def _validate_declined(args: tuple) -> None:
    if args:
        raise ValueError("declined takes no arguments")


def _validate_ids(name: str, minimum: int) -> Callable[[tuple], None]:
    def validate(args: tuple) -> None:
        if len(args) < minimum or not all(isinstance(a, str) and a for a in args):
            raise ValueError(f"{name} takes {minimum}+ fact or quirk id strings")

    return validate


_VALIDATORS: dict[str, Callable[[tuple], None]] = {
    "forecast_shape": _validate_forecast_shape,
    "declined": _validate_declined,
    "quirk_flagged": _validate_ids("quirk_flagged", 1),
    "no_causal_without": _validate_ids("no_causal_without", 0),
}


def parse_check(text: str) -> tuple[str, tuple]:
    """Parse one check call with literal arguments; unknown names raise."""
    try:
        node = ast.parse(text.strip(), mode="eval").body
    except SyntaxError as exc:
        raise ValueError(f"malformed check {text!r}") from exc
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.keywords:
        raise ValueError(f"malformed check {text!r}: expected name(literal, ...)")
    name = node.func.id
    if name not in _VALIDATORS:
        raise ValueError(f"unknown check {name!r}; known: {', '.join(_VALIDATORS)}")
    if not all(isinstance(arg, ast.Constant) for arg in node.args):
        raise ValueError(f"check {text!r}: arguments must be literals")
    args = tuple(arg.value for arg in node.args)
    _VALIDATORS[name](args)
    return name, args


def parse_checks(checks: list[str]) -> list[tuple[str, tuple]]:
    return [parse_check(check) for check in checks]


def load_rows(path: Path) -> list[dict]:
    """Read and validate eval rows; any bad row raises ValueError naming its id."""
    rows: list[dict] = []
    seen: set[str] = set()
    for number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        row_id = row.get("id", f"line {number}")
        try:
            missing = [f for f in REQUIRED_FIELDS if f not in row]
            if missing:
                raise ValueError(f"missing fields {missing}")
            if row["id"] in seen:
                raise ValueError("duplicate id")
            if row["category"] not in CATEGORIES:
                raise ValueError(f"unknown category {row['category']!r}")
            if row["tier"] not in (1, 2, 3):
                raise ValueError("tier must be 1-3")
            if row["expected_label"] not in (None, *LABELS):
                raise ValueError(f"unknown label {row['expected_label']!r}")
            tolerance = row["tolerance"]
            if tolerance is not None and set(tolerance) != {"rel"}:
                raise ValueError("tolerance must be null or {'rel': r}")
            if row["cutoff"] is not None:
                date.fromisoformat(row["cutoff"])
            parse_checks(row["gold_check"])
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"{row_id}: {exc}") from exc
        seen.add(row["id"])
        rows.append(row)
    return rows


# --- key values (built by scripts/build_gold.py --set agent) ---------------------

MAX_AUTO_KEY_VALUES = 24


def _cell_key(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()[:10]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return float(value)
    return str(value)


def _label(row: dict, numeric: set[str], col: str) -> str:
    keys = [str(_cell_key(row[c])) for c in row if c not in numeric]
    return " ".join([*keys, col])


def derive_key_values(row: dict, table) -> list[dict]:
    """Key values for an eval row from its frozen gold table.

    Forecast rows have none (their gold is the real future, scored only).
    ``key_select`` picks cells by ``where`` (one matching row each); without it
    every numeric cell is a key value when there are at most 24, else it fails.
    """
    if row["category"] == "forecast":
        return []
    numeric = {
        field.name
        for field in table.schema
        if pa_types.is_integer(field.type) or pa_types.is_floating(field.type) or pa_types.is_decimal(field.type)
    }
    records = table.to_pylist()
    out: list[dict] = []
    if "key_select" in row:
        for pick in row["key_select"]:
            where = {k: _cell_key(v) for k, v in pick["where"].items()}
            hits = [r for r in records if all(_cell_key(r.get(k)) == v for k, v in where.items())]
            if len(hits) != 1:
                raise ValueError(f"{row['id']}: key_select {pick} matched {len(hits)} rows")
            value = hits[0].get(pick["col"])
            if pick["col"] not in numeric or not _number(value):
                raise ValueError(f"{row['id']}: key_select {pick} is not a finite number")
            label = " ".join([*(f"{k}={v}" for k, v in pick["where"].items()), pick["col"]])
            out.append({"value": float(value), "label": label})
        return out
    cells = [(r, c) for r in records for c in table.column_names if c in numeric]
    if len(cells) > MAX_AUTO_KEY_VALUES:
        raise ValueError(f"{row['id']}: {len(cells)} numeric cells; add key_select")
    for record, col in cells:
        if not _number(record[col]):
            raise ValueError(f"{row['id']}: {col} is not a finite number ({record[col]!r}); add key_select")
        out.append({"value": float(record[col]), "label": _label(record, numeric, col)})
    return out


# --- shared helpers -----------------------------------------------------------


def _answer(outputs: dict) -> dict:
    return outputs.get("answer") or {}


def _claims(outputs: dict) -> list[dict]:
    return list(_answer(outputs).get("claims") or [])


def _verification(outputs: dict) -> dict | None:
    return outputs.get("verification") or _answer(outputs).get("verification")


def _handle_map(outputs: dict) -> dict[str, dict]:
    """Handle id -> flattened entry (card meta plus produced_by)."""
    out: dict[str, dict] = {}
    for handle_id, entry in (outputs.get("handles") or {}).items():
        if not isinstance(entry, dict):
            continue
        if isinstance(entry.get("meta"), dict):
            entry = {**{k: v for k, v in entry.items() if k != "meta"}, **entry["meta"]}
        out[str(handle_id)] = entry
    return out


def _entries(outputs: dict) -> list[dict]:
    return list(_handle_map(outputs).values())


def _is_forecast(entry: dict) -> bool:
    if "produced_by" in entry:
        return entry["produced_by"] == "forecast"
    return "backtest_summary" in entry and "horizon" in entry


def _gold_rows(reference: dict) -> list[dict] | None:
    if "gold" in reference:
        return list(reference["gold"])
    path = GOLD_DIR / f"{reference.get('id')}.parquet"
    if not path.is_file():
        return None
    return pq.read_table(path).to_pylist()


def _day(value: Any) -> str:
    if isinstance(value, (date, datetime)):
        return value.isoformat()[:10]
    return str(value)[:10]


def _forecast_series(reference: dict) -> dict | None:
    for check in reference.get("gold_check") or []:
        name, args = parse_check(check)
        if name == "forecast_shape":
            return _series_specs()[args[0]]
    return None


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


# --- answer_values -------------------------------------------------------------


def _sig_digits(stated: StatedNumber) -> int:
    found = re.search(r"\d[\d,]*(?:\.\d+)?", stated.raw)
    digits = re.sub(r"\D", "", found.group(0)) if found else ""
    return len(digits.lstrip("0"))


def _exact_integer(stated: StatedNumber, gold: float) -> bool:
    """A whole number stated in full (no scale, decimals or %) for an integer gold."""
    return (
        float(gold).is_integer()
        and stated.scale == 1
        and stated.decimals == 0
        and not stated.percent
    )


def _min_sig_digits(gold: float) -> int:
    return 3 if abs(gold) >= 10 else 2


def _stated_matches(stated: StatedNumber, gold: float, tolerance: dict | None) -> bool:
    if tolerance is None:
        if _exact_integer(stated, gold):
            return matches(stated, gold)  # full integers must match exactly
        if _sig_digits(stated) < _min_sig_digits(gold):
            return False
        return matches(stated, gold)
    bases = [gold]
    if stated.percent and abs(gold) <= 1.5:
        bases.append(gold * 100)
    if stated.value > 0:
        bases += [-b for b in bases]
    value = stated.value * stated.scale
    return any(math.isclose(value, b, rel_tol=tolerance["rel"]) for b in bases)


def _claim_matches(value: float, gold: float, tolerance: dict | None) -> bool:
    rel = _CLAIM_REL if tolerance is None else max(tolerance["rel"], _CLAIM_REL)
    targets = [gold, gold * 100] if abs(gold) <= 1.5 else [gold]
    return any(math.isclose(value, t, rel_tol=rel, abs_tol=1e-12) for t in targets)


def grade_answer_values(outputs: dict, reference: dict) -> Result:
    """Gate: every key value is matched by a claim value or a narrative number.

    tolerance null: claim values match exactly (float noise only); narrative
    numbers use the verify rounding rule and must carry at least 3 significant
    digits when |gold| >= 10, at least 2 below that ("2.89M" passes for
    2,894,704, "2.9M" fails; "1.33" passes for 1.3339, "1" fails). A whole
    number stated in full for an integer gold must match it exactly.
    tolerance {"rel": r}: both match within relative error r.

    Matching is one-to-one: each answer number satisfies at most one key value.
    """
    key = "answer_values"
    if "error" in outputs:
        return _result(key, False, f"target error: {outputs['error']}")
    key_values = reference.get("key_values") or []
    if not key_values:
        return _result(key, None, "no key values")
    tolerance = reference.get("tolerance")
    claim_values = [float(v) for c in _claims(outputs) for v in (c.get("values") or []) if _number(v)]
    stated = extract_numbers(_answer(outputs).get("narrative") or "")
    candidates: list[Callable[[float], bool]] = [
        (lambda gold, v=v: _claim_matches(v, gold, tolerance)) for v in claim_values
    ] + [(lambda gold, n=n: _stated_matches(n, gold, tolerance)) for n in stated]
    options = [
        [j for j, fits in enumerate(candidates) if fits(float(item["value"]))]
        for item in key_values
    ]
    owner: dict[int, int] = {}  # answer-number index -> key-value index

    def assign(i: int, seen: set[int]) -> bool:
        # Augmenting path (bipartite matching): key values are few, so this is cheap.
        for j in options[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owner or assign(owner[j], seen):
                owner[j] = i
                return True
        return False

    matched = {i for i in range(len(key_values)) if assign(i, set())}
    missing = [
        f"{item.get('label', '?')}={float(item['value']):g}"
        for i, item in enumerate(key_values)
        if i not in matched
    ]
    if missing:
        return _result(key, False, "missing: " + "; ".join(missing))
    return _result(key, True, f"matched {len(key_values)} key values")


# --- label / grounding -----------------------------------------------------------


def grade_label(outputs: dict, reference: dict) -> Result:
    key = "label"
    if "error" in outputs:
        return _result(key, False, f"target error: {outputs['error']}")
    expected = reference.get("expected_label")
    if expected is None:
        return _result(key, None, "no expected label")
    verification = _verification(outputs)
    got = verification.get("label") if verification else None
    return _result(key, got == expected, f"label {got!r}, expected {expected!r}")


@cache
def _memory() -> Memory:
    return load_memory(DatasetConfig.model_construct(dir=DATASET_DIR))


def _cited_fact_ids(outputs: dict) -> set[str]:
    return {fid for c in _claims(outputs) for fid in c.get("fact_ids") or []}


def _prose_causes(outputs: dict, allowed: set[str]) -> list[str]:
    """Causal sentences in the narrative or any claim text that are unhedged or
    unlicensed by a cited id in ``allowed`` (the verifier's prose rule)."""
    texts = [_answer(outputs).get("narrative") or ""]
    texts += [c.get("text") or "" for c in _claims(outputs)]
    return unsupported_causes(texts, _cited_fact_ids(outputs), allowed)


def grade_grounding(outputs: dict, reference: dict) -> Result:
    """Gate for traps: no unsupported causal claim, in the verify report or prose.

    A sentence with a causal cue must be hedged, and the answer must cite a
    context fact or quirk id valid as of the row's cutoff.
    """
    key = "grounding"
    if "error" in outputs:
        return _result(key, False, f"target error: {outputs['error']}")
    if reference.get("category") != "trap":
        return _result(key, None, "grounding gates trap rows only")
    verification = _verification(outputs)
    if verification is None:
        return _result(key, False, "no verification report")
    bad = [p for p in verification.get("problems") or [] if p.startswith("unsupported causal claim")]
    cutoff = reference.get("cutoff")
    allowed = allowed_fact_ids(_memory(), date.fromisoformat(cutoff) if cutoff else None)
    bad += _prose_causes(outputs, allowed)
    if bad:
        return _result(key, False, f"{len(bad)} unsupported causal claim(s)")
    return _result(key, True, "no unsupported causal claim")


# --- checks ----------------------------------------------------------------------


def _shape_match(
    outputs: dict, reference: dict, series: str, freq: str, horizon: int
) -> tuple[dict | None, str]:
    """The first forecast handle meeting forecast_shape, or None with the reasons."""
    spec = _series_specs()[series]
    handles = _handle_map(outputs)
    forecasts = [e for e in handles.values() if _is_forecast(e)]
    if not forecasts:
        return None, "no forecast handle"
    required: set[str] | None = None
    if spec["dimensions"]:
        gold = _gold_rows(reference)
        if not gold:
            return None, "no gold actuals to resolve the series ids"
        id_col, value_col = spec["dimensions"][0], spec["metric"]
        totals: dict[str, float] = {}
        for row in gold:
            totals[str(row[id_col])] = totals.get(str(row[id_col]), 0.0) + float(row[value_col] or 0)
        grand = sum(totals.values())
        required = {k for k, v in totals.items() if grand and v / grand >= _HEADLINE_SHARE}
    reasons = []
    for meta in forecasts:
        summary = meta.get("backtest_summary") or {}
        ids = {str(i) for i in meta.get("headline_ids") or []}
        ids |= {str(r.get("unique_id")) for r in meta.get("forecast") or []}
        problems = []
        if meta.get("freq") != freq:
            problems.append(f"freq {meta.get('freq')!r}")
        if meta.get("horizon") != horizon:
            problems.append(f"horizon {meta.get('horizon')!r}")
        if meta.get("cutoff") != reference.get("cutoff"):
            problems.append(f"cutoff {meta.get('cutoff')!r}")
        if "SeasonalNaive" not in (meta.get("candidates") or []) or not _number(summary.get("baseline_mean_mase")):
            problems.append("no SeasonalNaive comparison")
        metrics = ("mean_mase", "mean_relative_mae", "mean_coverage_80", "mean_coverage_95")
        if not all(_number(summary.get(m)) for m in metrics):
            problems.append("backtest metrics missing")
        if required is None:
            if ids != {"series"}:
                problems.append(f"ids {sorted(ids)} for a single series")
        elif not required <= ids or "series" in ids:
            problems.append(f"ids miss {sorted(required - ids)}")
        source = handles.get(str(meta.get("input_handle")))
        if source is None:
            problems.append(f"input handle {meta.get('input_handle')!r} not in outputs")
        else:
            metrics = (source.get("query") or {}).get("metrics") or []
            if spec["metric"] not in metrics:
                problems.append(f"input metrics {metrics}, need {spec['metric']!r}")
        if not problems:
            return meta, f"forecast {series} {freq} h={horizon}"
        reasons.append(", ".join(problems))
    return None, "forecast handle mismatch: " + " | ".join(reasons)


def _check_forecast_shape(outputs: dict, reference: dict, series: str, freq: str, horizon: int) -> tuple[bool, str]:
    meta, comment = _shape_match(outputs, reference, series, freq, horizon)
    return meta is not None, comment


def _check_declined(outputs: dict, reference: dict) -> tuple[bool, str]:
    """Declined with a reason; any number stated (claim value or narrative) must
    be traced, i.e. the label is verified or verified_off_model."""
    values = [v for c in _claims(outputs) for v in (c.get("values") or [])]
    narrative = (_answer(outputs).get("narrative") or "").strip()
    label = (_verification(outputs) or {}).get("label")
    traced = label in ("verified", "verified_off_model")
    if values and not traced:
        return False, f"declined but stated claim values {values[:3]} (label {label!r})"
    if not narrative:
        return False, "declined without saying why"
    numbers = extract_numbers(narrative, exclude=reference.get("question") or "")
    if numbers and not traced:
        return False, f"declined but stated untraced numbers {[n.raw for n in numbers[:3]]} (label {label!r})"
    return True, "declined with a reason"


def _check_quirk_flagged(outputs: dict, reference: dict, *ids: str) -> tuple[bool, str]:
    cited = {fid for c in _claims(outputs) for fid in c.get("fact_ids") or []}
    narrative = _answer(outputs).get("narrative") or ""
    if cited & set(ids) or any(i in narrative for i in ids):
        return True, f"flagged {'/'.join(ids)}"
    return False, f"did not cite {'/'.join(ids)}"


def _check_no_causal_without(outputs: dict, reference: dict, *ids: str) -> tuple[bool, str]:
    allowed = set(ids)
    for claim in _claims(outputs):
        if claim.get("type") != "causal":
            continue
        if not allowed & set(claim.get("fact_ids") or []):
            return False, f"causal claim without {sorted(allowed) or 'any allowed fact'}: {claim.get('text', '')[:80]!r}"
    for sentence in _prose_causes(outputs, allowed):
        return False, f"cause in prose without a hedge and {sorted(allowed) or 'any allowed fact'}: {sentence[:80]!r}"
    return True, "causal claims cite allowed facts" if allowed else "no causal claims"


_CHECKS: dict[str, Callable[..., tuple[bool, str]]] = {
    "forecast_shape": _check_forecast_shape,
    "declined": _check_declined,
    "quirk_flagged": _check_quirk_flagged,
    "no_causal_without": _check_no_causal_without,
}
assert set(_CHECKS) == set(_VALIDATORS)


def grade_checks(outputs: dict, reference: dict) -> Result:
    """Gate when gold_check is non-empty: every named check passes."""
    key = "checks"
    if "error" in outputs:
        return _result(key, False, f"target error: {outputs['error']}")
    checks = parse_checks(reference.get("gold_check") or [])
    if not checks:
        return _result(key, None, "no checks")
    comments, ok = [], True
    for name, args in checks:
        good, comment = _CHECKS[name](outputs, reference, *args)
        ok = ok and good
        comments.append(f"{name}: {'ok' if good else 'FAIL'} ({comment})")
    return _result(key, ok, "; ".join(comments))


# --- scores only -----------------------------------------------------------------


_trajectory_match = create_trajectory_match_evaluator(
    trajectory_match_mode="superset", tool_args_match_mode="ignore"
)


def _trajectory(names: list[str]) -> list[dict]:
    calls = [{"function": {"name": n, "arguments": "{}"}} for n in names]
    return [{"role": "assistant", "content": "", "tool_calls": calls}]


def grade_trajectory(outputs: dict, reference: dict) -> Result:
    """Score only: the called tools are a superset of expected_tools (openevals)."""
    key = "trajectory"
    if "error" in outputs:
        return _result(key, False, f"target error: {outputs['error']}")
    tools = list(outputs.get("tools") or [])
    expected = list(reference.get("expected_tools") or [])
    found = _trajectory_match(outputs=_trajectory(tools), reference_outputs=_trajectory(expected))
    return _result(key, bool(found["score"]), f"called {tools}, expected superset of {expected}")


def grade_forecast_skill(outputs: dict, reference: dict) -> Result:
    """Score only: WAPE of the forecast vs real actuals, with 80/95 coverage."""
    key = "forecast_skill"
    if reference.get("category") != "forecast":
        return _result(key, None, "not a forecast row")
    if "error" in outputs:
        return _result(key, None, f"target error: {outputs['error']}")
    spec = _forecast_series(reference)
    gold = _gold_rows(reference)
    forecasts = [e for e in _entries(outputs) if _is_forecast(e) and e.get("forecast")]
    if spec is None or not gold or not forecasts:
        return _result(key, None, "no forecast rows or no gold actuals")
    chosen, which = forecasts[-1], "last forecast handle"
    for check in reference.get("gold_check") or []:
        name, args = parse_check(check)
        if name == "forecast_shape":
            meta, _ = _shape_match(outputs, reference, *args)
            if meta is not None and meta.get("forecast"):
                chosen, which = meta, "forecast_shape handle"
    id_col = spec["dimensions"][0] if spec["dimensions"] else None
    actual = {
        (str(row[id_col]) if id_col else "series", _day(row["period"])): float(row[spec["metric"]])
        for row in gold
        if _number(row[spec["metric"]])
    }
    errors = scale = 0.0
    inside = {80: 0, 95: 0}
    points = 0
    for row in chosen["forecast"]:
        y = actual.get((str(row["unique_id"]), _day(row["ds"])))
        if y is None:
            continue
        points += 1
        errors += abs(float(row["yhat"]) - y)
        scale += abs(y)
        for level in inside:
            inside[level] += float(row[f"lo_{level}"]) <= y <= float(row[f"hi_{level}"])
    if not points or not scale:
        return _result(key, None, "forecast does not overlap the gold actuals")
    wape = errors / scale
    return _result(
        key,
        wape,
        f"wape={wape:.4f} coverage_80={inside[80] / points:.2f} "
        f"coverage_95={inside[95] / points:.2f} points={points}/{len(actual)} ({which})",
    )


GRADERS: tuple[Callable[[dict, dict], Result], ...] = (
    grade_answer_values,
    grade_label,
    grade_grounding,
    grade_checks,
    grade_trajectory,
    grade_forecast_skill,
)


def passed(results: list[Result]) -> bool:
    """Every gate is present, and every applicable one (score not None) is True."""
    by_key = {r["key"]: r for r in results}
    for gate in GATES:
        if gate not in by_key:
            return False
        score = by_key[gate]["score"]
        if score is not None and score is not True:
            return False
    return True
