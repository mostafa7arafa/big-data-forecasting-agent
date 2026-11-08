"""Position- or name-aligned, tolerance-aware comparison of scalar SQL tables."""

import math
from dataclasses import asdict, dataclass
from datetime import date, datetime
from decimal import Decimal

import pyarrow as pa


@dataclass(frozen=True)
class GradeResult:
    key: str
    score: bool
    comment: str

    def as_dict(self) -> dict[str, str | bool]:
        return asdict(self)


def _normalize(value):
    if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, date) and not isinstance(value, datetime):
        return datetime.combine(value, datetime.min.time())
    return value


def _sort_value(value):
    # Tagged keys keep NULLs and unlike scalar types sortable. Round only the
    # sort key, never the compared values: custom tolerances still take effect.
    if value is None:
        return (0, "")
    if isinstance(value, float):
        if math.isnan(value):
            return (1, 0, 0.0)
        return (1, 1, float(f"{value:.6g}"))
    if isinstance(value, datetime):
        return (2, value.isoformat())
    if isinstance(value, str):
        return (3, value)
    return (4, type(value).__name__, repr(value))


def _equal(got, expected, rel_tol, abs_tol):
    if got is None or expected is None:
        return got is expected
    if isinstance(got, float) and isinstance(expected, float):
        return (math.isnan(got) and math.isnan(expected)) or math.isclose(
            got, expected, rel_tol=rel_tol, abs_tol=abs_tol
        )
    if isinstance(got, datetime) and isinstance(expected, datetime):
        # Arrow timestamp[ns] yields pandas.Timestamp, a datetime subclass.
        return got == expected
    return type(got) is type(expected) and got == expected


def result_set_match(
    got: pa.Table,
    expected: pa.Table,
    *,
    rel_tol: float = 1e-6,
    abs_tol: float = 1e-9,
    ignore_order: bool = True,
    ignore_column_names: bool = True,
) -> GradeResult:
    """Compare SQL result multisets, retaining duplicates and original precision.

    Row indices in diagnostics are zero-based, after optional canonical sorting.
    Names are positional by default; otherwise columns align by their names.
    NULL equals NULL; NaN equals NaN; infinities must have the same sign.
    """
    if not (math.isfinite(rel_tol) and math.isfinite(abs_tol)) or min(rel_tol, abs_tol) < 0:
        raise ValueError("tolerances must be finite and non-negative")

    def grade(score, comment):
        return GradeResult("result_set_match", score, comment)

    if got.shape != expected.shape:
        return grade(False, f"shape mismatch: got {got.shape}, expected {expected.shape}")
    if not ignore_column_names:
        if (len(set(got.column_names)) != got.num_columns
                or len(set(expected.column_names)) != expected.num_columns):
            return grade(False, "column mismatch: duplicate names cannot align by name")
        if set(got.column_names) != set(expected.column_names):
            return grade(False, f"column mismatch: got {got.column_names}, expected {expected.column_names}")
        got = got.select(expected.column_names)

    def rows(table):
        values = [column.to_pylist() for column in table.columns]
        result = [tuple(_normalize(v) for v in row) for row in zip(*values)]
        if ignore_order:
            result.sort(key=lambda row: tuple(_sort_value(v) for v in row))
        return result

    for index, (actual, wanted) in enumerate(zip(rows(got), rows(expected), strict=True)):
        for column, (left, right) in enumerate(zip(actual, wanted, strict=True)):
            if not _equal(left, right, rel_tol, abs_tol):
                return grade(
                    False,
                    f"row {index}, column {column} ({expected.column_names[column]!r}): "
                    f"got {left!r}, expected {right!r}; rows {actual!r} vs {wanted!r}",
                )
    return grade(True, f"matched {got.num_rows} rows and {got.num_columns} columns")
