"""Collect result handles from a run and trace stated numbers to their values."""

import math
import re
from dataclasses import dataclass, field

import numpy as np
import pyarrow as pa
import pyarrow.types as pat

from analyst.background import is_background_notification
from analyst.data.results import ResultStore
from analyst.errors import UnknownNameError
from analyst.verify.numbers import StatedNumber, intervals, matches

MAX_CELLS = 2000
MAX_DERIVATION_CELLS = 200
_HANDLES_LINE = re.compile(r"^\s*handles:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)
_HANDLE_ID = re.compile(r"r_[0-9a-f]{10}")


@dataclass
class HandleInfo:
    """A result handle's numeric cells. ``columns`` holds them per column.

    Derivations pair cells of one column, or cells of two different handles;
    ``values`` alone (no ``columns``) counts as a single column.
    """

    handle_id: str
    produced_by: str
    governed: bool
    values: list[float] = field(default_factory=list)
    columns: list[list[float]] = field(default_factory=list)

    def column_lists(self) -> list[list[float]]:
        return self.columns or ([self.values] if self.values else [])


def _numeric_columns(table: pa.Table) -> list[list[float]]:
    """Numeric cells per column; head and tail kept within MAX_CELLS overall."""
    cols = [
        col
        for col, field_ in zip(table.columns, table.schema, strict=True)
        if (
            pat.is_integer(field_.type)
            or pat.is_floating(field_.type)
            or pat.is_decimal(field_.type)
        )
    ]
    if not cols:
        return []
    cols = cols[: MAX_CELLS // 2]
    per_col = max(2, MAX_CELLS // len(cols))
    out: list[list[float]] = []
    for col in cols:
        n = len(col)
        if n > per_col:
            half = per_col // 2
            idx = list(range(half)) + list(range(n - half, n))
        else:
            idx = list(range(n))
        cells: list[float] = []
        for i in idx:
            cell = col[i].as_py()
            if cell is None:
                continue
            number = float(cell)
            if math.isfinite(number):
                cells.append(number)
        out.append(cells)
    return out


def load_handle_info(store: ResultStore, handle_id: str) -> HandleInfo | None:
    """HandleInfo from the store, or None when the handle is unknown there."""
    try:
        if not store.exists(handle_id):
            return None
        card = store.card(handle_id)
        columns = _numeric_columns(store.load(handle_id))
    except (UnknownNameError, OSError, ValueError):
        return None
    values = [v for col in columns for v in col]
    return HandleInfo(handle_id, card.produced_by, card.governed, values, columns)


def _get(message: object, name: str):
    if isinstance(message, dict):
        return message.get(name)
    return getattr(message, name, None)


def _text(message: object) -> str:
    content = _get(message, "content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part if isinstance(part, str) else str(part.get("text", ""))
            for part in content
            if isinstance(part, str | dict)
        )
    return ""


def collect_handles(messages, store: ResultStore) -> dict[str, HandleInfo]:
    """Handles produced by tools in this run, plus those reported by workers."""
    found: dict[str, HandleInfo] = {}
    for message in messages:
        artifact = _get(message, "artifact")
        if isinstance(artifact, dict) and isinstance(artifact.get("handle_id"), str):
            hid = artifact["handle_id"]
            loaded = load_handle_info(store, hid)
            found[hid] = HandleInfo(
                hid,
                str(artifact.get("produced_by", "")),
                bool(artifact.get("governed", False)),
                loaded.values if loaded else [],
                loaded.columns if loaded else [],
            )
            continue
        if is_background_notification(message):
            for line in _HANDLES_LINE.findall(_text(message)):
                for hid in _HANDLE_ID.findall(line):
                    if hid in found:
                        continue
                    loaded = load_handle_info(store, hid)
                    if loaded is not None:
                        found[hid] = loaded
    return found


def direct_handles(x: StatedNumber, handles: list[HandleInfo]) -> list[HandleInfo]:
    """Handles holding a cell that matches ``x`` at its stated precision."""
    spans = intervals(x)
    found = []
    for h in handles:
        if not h.values:
            continue
        arr = np.array(h.values, dtype=float)
        mask = np.zeros(arr.shape, dtype=bool)
        for lo, hi in spans:
            mask |= (arr >= lo) & (arr <= hi)
        if any(matches(x, float(v)) for v in arr[mask]):
            found.append(h)
    return found


def _derivable(cited: list[HandleInfo]) -> bool:
    return sum(len(h.values) for h in cited) <= MAX_DERIVATION_CELLS


def _derive_hit(cited: list[HandleInfo], spans, confirm) -> bool:
    """One-hop a-b, (a-b)/b*100, a/b*100, a/b between allowed cell pairs.

    Allowed: two cells of the same column, or cells of two different cited
    handles. Cells of two different columns of one handle are not paired.
    """
    cols = []
    for hi, h in enumerate(cited):
        for col in h.column_lists():
            uniq, counts = np.unique(np.array(col, dtype=float), return_counts=True)
            cols.append((hi, uniq, counts))

    def hit(derived: np.ndarray) -> np.ndarray:
        mask = np.zeros(derived.shape, dtype=bool)
        for lo, hi in spans:
            mask |= (derived >= lo) & (derived <= hi)
        return mask

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for ai, (ah, avals, _) in enumerate(cols):
            for bi, (bh, bvals, bcounts) in enumerate(cols):
                if ah == bh and ai != bi:
                    continue
                nonzero = bvals != 0
                for i, a in enumerate(avals):
                    usable = np.ones(len(bvals), dtype=bool)
                    if ai == bi:
                        usable[i] = bcounts[i] >= 2
                    candidates = (
                        (a - bvals, usable),
                        ((a - bvals) / bvals * 100, usable & nonzero),
                        (a / bvals * 100, usable & nonzero),
                        (a / bvals, usable & nonzero),
                    )
                    for derived, mask in candidates:
                        for idx in np.nonzero(hit(derived) & mask)[0]:
                            if confirm(float(derived[idx])):
                                return True
    return False


def trace_value(
    x: StatedNumber, cited: list[HandleInfo], *, derive: bool = True
) -> bool:
    """A direct match, or (``derive``) a one-hop derivation over ``cited``."""
    if direct_handles(x, cited):
        return True
    if not derive or not _derivable(cited):
        return False
    return _derive_hit(cited, intervals(x), lambda d: matches(x, d))


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def trace_float(value: float, cited: list[HandleInfo]) -> bool:
    """Trace an exact claim value (rel. tolerance 1e-9), directly or one hop.

    A percent given as a number (-2.99) also matches the fraction (-0.0299).
    """
    targets = (value, value / 100)

    def confirm(d: float) -> bool:
        return any(_close(d, t) for t in targets)

    if any(confirm(v) for h in cited for v in h.values):
        return True
    if not _derivable(cited):
        return False
    spans = []
    for t in targets:
        pad = 2e-9 * max(abs(t), 1e-3)
        spans.append((t - pad, t + pad))
    return _derive_hit(cited, spans, confirm)
