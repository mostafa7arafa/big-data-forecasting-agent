"""Resolve user-facing dimension values against the Engine's visible data."""

from difflib import get_close_matches

from analyst.data.engine import Engine
from analyst.errors import InvalidRequestError
from analyst.semantic.compiler import approved, from_sql
from analyst.semantic.model import SemanticModel


def _normalize(value: str) -> str:
    return " ".join(value.split()).casefold()


def _as_integer(value: str) -> int | None:
    """Parse "3" for a dimension whose stored values are integers."""
    try:
        return int(value.strip())
    except ValueError:
        return None


def resolve_values(
    engine: Engine, model: SemanticModel, dimension: str, values: list
) -> list:
    dim = model.dimension(dimension)
    approved(dim, dim.name, False)
    source = from_sql(model, [dim])
    rows = engine.execute(f"SELECT DISTINCT {dim.expr} AS value\n{source}")
    distinct = [value for value in rows["value"].to_pylist() if value is not None]
    # Sorting makes ties/suggestions deterministic, irrespective of scan order.
    text = {
        _normalize(value): value for value in sorted(distinct) if isinstance(value, str)
    }
    integers = {value for value in distinct if isinstance(value, int)}
    resolved = []
    for value in values:
        coerced = _as_integer(value) if isinstance(value, str) else None
        if isinstance(value, int) and value in integers:
            resolved.append(value)
        elif isinstance(value, str) and _normalize(value) in text:
            resolved.append(text[_normalize(value)])
        elif coerced is not None and coerced in integers:
            resolved.append(coerced)
        else:
            labels = text or {str(item): item for item in sorted(integers)}
            needle = _normalize(str(value))
            suggestions = get_close_matches(needle, list(labels), n=5, cutoff=0)
            choices = ", ".join(str(labels[label]) for label in suggestions)
            raise InvalidRequestError(
                f"unknown value {value!r} for {dim.name}; "
                f"did you mean: {choices or '(none)'}",
                hint=f"call describe('dimension:{dim.name}')",
            )
    return resolved
