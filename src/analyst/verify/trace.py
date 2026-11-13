"""Collect result handles from a run and trace stated numbers to their values."""

import math
import re
from dataclasses import dataclass, field

import numpy as np
import pyarrow as pa
import pyarrow.types as pat

from analyst.data.results import ResultStore
from analyst.errors import UnknownNameError
from analyst.verify.numbers import StatedNumber, intervals, matches

MAX_CELLS = 2000
_HANDLES_LINE = re.compile(r"^\s*handles:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)
_HANDLE_ID = re.compile(r"r_[0-9a-f]{10}")


@dataclass
class HandleInfo:
    handle_id: str
    produced_by: str
    governed: bool
    values: list[float] = field(default_factory=list)


def _numeric_values(table: pa.Table) -> list[float]:
    columns = [
        col
        for col, field_ in zip(table.columns, table.schema, strict=True)
        if (
            pat.is_integer(field_.type)
            or pat.is_floating(field_.type)
            or pat.is_decimal(field_.type)
        )
    ]
    if not columns:
        return []
    rows = max(1, MAX_CELLS // len(columns))
    out: list[float] = []
    for row in range(min(rows, table.num_rows)):
        for col in columns:
            cell = col[row].as_py()
            if cell is None:
                continue
            number = float(cell)
            if math.isfinite(number):
                out.append(number)
    return out[:MAX_CELLS]


def load_handle_info(store: ResultStore, handle_id: str) -> HandleInfo | None:
    """HandleInfo from the store, or None when the handle is unknown there."""
    try:
        if not store.exists(handle_id):
            return None
        card = store.card(handle_id)
        values = _numeric_values(store.load(handle_id))
    except (UnknownNameError, OSError, ValueError):
        return None
    return HandleInfo(handle_id, card.produced_by, card.governed, values)


def _get(message: object, name: str):
    if isinstance(message, dict):
        return message.get(name)
    return getattr(message, name, None)


def _is_worker_note(message: object) -> bool:
    for location in ("additional_kwargs", "response_metadata", "metadata"):
        meta = _get(message, location)
        if isinstance(meta, dict) and meta.get("source") == "async_subagent":
            return True
    return False


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
            )
            continue
        if _is_worker_note(message):
            for line in _HANDLES_LINE.findall(_text(message)):
                for hid in _HANDLE_ID.findall(line):
                    if hid in found:
                        continue
                    loaded = load_handle_info(store, hid)
                    if loaded is not None:
                        found[hid] = loaded
    return found


def trace_value(x: StatedNumber, cited: list[HandleInfo]) -> bool:
    """A direct match, or one hop: a-b, (a-b)/b*100, a/b*100, a/b over ordered pairs."""
    pool = [v for h in cited for v in h.values[:MAX_CELLS]]
    if not pool:
        return False
    if any(matches(x, v) for v in pool):
        return True
    spans = intervals(x)
    values = np.array(pool, dtype=float)
    uniq, counts = np.unique(values, return_counts=True)
    nonzero = uniq != 0

    def hit(derived: np.ndarray) -> np.ndarray:
        mask = np.zeros(derived.shape, dtype=bool)
        for lo, hi in spans:
            mask |= (derived >= lo) & (derived <= hi)
        return mask

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for i, a in enumerate(uniq):
            usable = np.ones(len(uniq), dtype=bool)
            if counts[i] < 2:
                usable[i] = False  # a value cannot pair with its own single cell
            candidates = [a - uniq]
            ok_b = usable & nonzero
            candidates += [(a - uniq) / uniq * 100, a / uniq * 100, a / uniq]
            for n, derived in enumerate(candidates):
                mask = usable if n == 0 else ok_b
                for idx in np.nonzero(hit(derived) & mask)[0]:
                    if matches(x, float(derived[idx])):
                        return True
    return False
