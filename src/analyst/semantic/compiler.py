"""Compile governed metric requests into dataset-agnostic DuckDB SELECTs.

Semantic expressions are trusted memory, not request-supplied SQL. Request names
are resolved against that memory; all emitted identifiers and values are checked
or quoted. Execution and resource limits remain the Engine's responsibility.
"""

import re
from datetime import date
from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

from analyst.errors import InvalidRequestError
from analyst.semantic.model import Dimension, MemoryItem, Metric, SemanticModel


class FilterSpec(BaseModel):
    dimension: str
    op: Literal["eq", "in", "neq", "not_in"]
    values: list[str | int]

    @model_validator(mode="after")
    def _arity(self) -> Self:
        if self.op in {"eq", "neq"} and len(self.values) != 1:
            raise InvalidRequestError("eq/neq filters require exactly one value")
        return self


class MetricQuery(BaseModel):
    metrics: list[str] = Field(min_length=1)
    dimensions: list[str] = Field(default_factory=list)
    filters: list[FilterSpec] = Field(default_factory=list)
    grain: Literal["hour", "day", "week", "month"] | None = None
    time_range: tuple[date, date] | None = None
    compare_to: Literal["previous_period", "same_period_last_year"] | None = Field(
        default=None,
        description=(
            "previous_period compares with the immediately preceding equal-length "
            "range. same_period_last_year compares with the same range one year "
            "earlier; without a grain, the whole time_range is compared with the "
            "same range one year earlier, as one total."
        ),
    )
    share: bool = False
    order_by: str | None = None
    order_desc: bool = False
    limit: int | None = Field(default=None, ge=0, strict=True)

    @model_validator(mode="after")
    def _range(self) -> Self:
        if self.time_range is not None and self.time_range[0] >= self.time_range[1]:
            raise InvalidRequestError("time_range must have start before end")
        return self


def _identifier(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise InvalidRequestError(f"invalid SQL identifier: {name!r}")
    return f'"{name}"'


def _literal(value: str | int) -> str:
    if isinstance(value, int):
        return str(value)
    return "'" + value.replace("'", "''") + "'"


def approved(item: MemoryItem, name: str, allow_proposed: bool) -> None:
    if item.status != "approved" and not allow_proposed:
        raise InvalidRequestError(
            f"{name!r} is {item.status}; only approved items may be used"
        )


def from_sql(
    model: SemanticModel, dimensions: list[Dimension], *, allow_proposed: bool = False
) -> str:
    # Explicit main avoids a view called cur/cmp being shadowed by our CTEs.
    sql = f"FROM main.{_identifier(model.table)}"
    required = {dimension.join for dimension in dimensions if dimension.join}
    joins = {join.name: join for join in model.joins}
    for name in sorted(required):
        if name not in joins:
            raise InvalidRequestError(f"unknown join {name!r}")
        join = joins[name]
        approved(join, join.name, allow_proposed)
        sql += (
            f"\nLEFT JOIN main.{_identifier(join.table)} AS {_identifier(join.name)}"
            f" ON {join.on}"
        )
    return sql


def _filter_sql(spec: FilterSpec, dimension: Dimension) -> str:
    expr = f"({dimension.expr})"
    values = ", ".join(_literal(value) for value in spec.values)
    if spec.op in {"eq", "neq"}:
        operator = "=" if spec.op == "eq" else "IS DISTINCT FROM"
        return f"{expr} {operator} {values}"
    if not spec.values:
        return "FALSE" if spec.op == "in" else "TRUE"
    if spec.op == "not_in":
        return f"({expr} NOT IN ({values}) OR {expr} IS NULL)"
    return f"{expr} IN ({values})"


def _is_additive(expr: str) -> bool:
    """Recognize a single sum/count call, allowing nested scalar arguments."""
    expr = expr.strip()
    opening = re.match(r"(?:sum|count)\s*\(", expr, re.IGNORECASE)
    if opening is None:
        return False
    depth = 1
    # Skip quoted literals/identifiers and comments when balancing parentheses.
    tokens = re.finditer(
        r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"|--[^\n]*|/\*.*?\*/|[()]",
        expr[opening.end() :],
        re.DOTALL,
    )
    for token in tokens:
        if token.group() == "(":
            depth += 1
        elif token.group() == ")":
            depth -= 1
            if depth == 0:
                return not expr[opening.end() + token.end() :].strip()
    return False


def _previous_period_shift(start: date, end: date, grain: str | None) -> str:
    """Prefer calendar months, then Monday-aligned weeks, else equal-length days.

    A month interval cannot align week-truncated keys; a day interval can align
    them only in multiples of seven. Month grain requires a calendar-month shift.
    """
    days = (end - start).days
    if start.day == end.day == 1:
        amount = (end.year - start.year) * 12 + end.month - start.month
        unit = "MONTH"
    elif grain == "week" and start.weekday() == end.weekday() == 0:
        amount, unit = days // 7, "WEEK"
    else:
        amount, unit = days, "DAY"
    if (grain == "month" and unit != "MONTH") or (
        grain == "week" and (unit == "MONTH" or (unit == "DAY" and days % 7))
    ):
        raise InvalidRequestError(f"previous_period range must align to whole {grain}s")
    return f"INTERVAL {amount} {unit}"


def _metric_predicates(
    model: SemanticModel, metrics: list[Metric], allow_proposed: bool
) -> dict[str, str]:
    filters = {item.name: item for item in model.filters}
    predicates = {}
    for metric in metrics:
        parts = []
        for name in metric.filters:
            if name not in filters:
                raise InvalidRequestError(
                    f"unknown metric filter {name!r}",
                    hint="call describe('metrics')",
                )
            item = filters[name]
            approved(item, name, allow_proposed)
            parts.append(f"({item.expr})")
        predicates[metric.name] = " AND ".join(parts)
    return predicates


def _period_sql(model: SemanticModel, grain: str) -> str:
    expr = f"date_trunc('{grain}', {_identifier(model.time_dimension.column)})"
    return expr if grain == "hour" else f"CAST({expr} AS DATE)"


def _aggregate_sql(
    metrics: list[Metric],
    keys: list[tuple[str, str]],
    predicates: dict[str, str],
    source: str,
    where: list[str],
) -> str:
    """Group simple aggregates together; isolate filtered compound expressions.

    A FILTER cannot attach to a ratio or arithmetic expression. Each such metric
    gets its own CTE over the same group spine, with a correlated scalar aggregate
    over filtered rows. This also preserves empty groups (count(*) + 1 is 1, not
    NULL), and avoids parsing/replacing arbitrary semantic SQL expressions.
    DuckDB decorrelates these scalar aggregates into grouped joins.
    """
    simple, compound = [], []
    for metric in metrics:
        if predicates[metric.name] and not re.fullmatch(
            r"\w+\([^()]*\)", metric.expr.strip()
        ):
            compound.append(metric)
        else:
            simple.append(metric)

    projections = [f"{expr} AS {_identifier(name)}" for name, expr in keys]
    for metric in simple:
        expr = metric.expr
        if predicate := predicates[metric.name]:
            expr += f" FILTER (WHERE {predicate})"
        projections.append(f"{expr} AS {_identifier(metric.name)}")
    # A global aggregate must return one row even on empty input, including when
    # every metric is a filtered compound expression.
    if not keys and not simple:
        projections.append('count(*) AS "_group_spine"')
    grouped = "SELECT " + ", ".join(projections) + "\n" + source
    if where:
        grouped += "\nWHERE " + " AND ".join(where)
    if keys:
        grouped += "\nGROUP BY " + ", ".join(str(i + 1) for i in range(len(keys)))
    if not compound:
        return grouped

    ctes = [f"_semantic_groups AS (\n{grouped}\n)"]
    for index, metric in enumerate(compound):
        conditions = [*where, predicates[metric.name]]
        conditions.extend(
            f"({expr}) IS NOT DISTINCT FROM g.{_identifier(name)}"
            for name, expr in keys
        )
        scalar = (
            f"(SELECT {metric.expr}\n{source}\nWHERE " + " AND ".join(conditions) + ")"
        )
        columns = [f"g.{_identifier(name)}" for name, _ in keys]
        columns.append(f"{scalar} AS {_identifier(metric.name)}")
        ctes.append(
            f"_semantic_metric_{index} AS (\nSELECT "
            + ", ".join(columns)
            + "\nFROM _semantic_groups AS g\n)"
        )
    columns = [f"g.{_identifier(name)}" for name, _ in keys]
    for metric in metrics:
        alias = f"m{compound.index(metric)}" if metric in compound else "g"
        columns.append(f"{alias}.{_identifier(metric.name)}")
    sql = "WITH " + ",\n".join(ctes) + "\nSELECT " + ", ".join(columns)
    sql += "\nFROM _semantic_groups AS g"
    for index, _ in enumerate(compound):
        conditions = [
            f"g.{_identifier(name)} IS NOT DISTINCT FROM m{index}.{_identifier(name)}"
            for name, _ in keys
        ]
        sql += f"\nLEFT JOIN _semantic_metric_{index} AS m{index} ON "
        sql += " AND ".join(conditions) if conditions else "TRUE"
    return sql


def compile_query(
    model: SemanticModel, q: MetricQuery, *, allow_proposed: bool = False
) -> str:
    """Return a single SELECT, with canonical model names as output aliases."""
    if q.compare_to and q.time_range is None:
        raise InvalidRequestError("compare_to requires time_range")
    if q.grain and q.grain not in model.time_dimension.grains:
        raise InvalidRequestError(
            f"unsupported grain {q.grain!r}",
            hint="call describe('dimensions') for the supported grains",
        )
    metrics = [model.metric(name) for name in q.metrics]
    dimensions = [model.dimension(name) for name in q.dimensions]
    filter_dimensions = [model.dimension(spec.dimension) for spec in q.filters]
    for item in [*metrics, *dimensions, *filter_dimensions]:
        approved(item, item.name, allow_proposed)
        _identifier(item.name)
    source = from_sql(
        model, dimensions + filter_dimensions, allow_proposed=allow_proposed
    )
    predicates = _metric_predicates(model, metrics, allow_proposed)
    if q.share:
        for metric in metrics:
            if not _is_additive(metric.expr):
                raise InvalidRequestError(
                    "share is only defined for additive metrics (sum/count): "
                    + metric.name
                )
    keys = [(dimension.name, dimension.expr) for dimension in dimensions]
    if q.grain:
        keys.insert(0, ("period", _period_sql(model, q.grain)))
    output = [name for name, _ in keys] + [metric.name for metric in metrics]
    if q.compare_to:
        for metric in metrics:
            output.extend([f"{metric.name}_compare", f"{metric.name}_change_pct"])
    if q.share:
        output.extend(f"{metric.name}_share_pct" for metric in metrics)
    for name in output:
        _identifier(name)
    if len({name.casefold() for name in output}) != len(output):
        raise InvalidRequestError("duplicate output column names")
    if q.order_by is not None and q.order_by not in output:
        raise InvalidRequestError(f"order_by must be an output column: {', '.join(output)}")

    where = [
        _filter_sql(spec, dim)
        for spec, dim in zip(q.filters, filter_dimensions, strict=True)
    ]
    shift = None
    if q.compare_to == "previous_period":
        start, end = q.time_range
        shift = _previous_period_shift(start, end, q.grain)
    elif q.compare_to:
        shift = "INTERVAL " + {"month": "12 MONTH", "week": "52 WEEK"}.get(
            q.grain, "364 DAY"
        )
        if q.grain is None:
            shift = "INTERVAL 12 MONTH"

    def aggregate(comparison: bool = False) -> str:
        conditions = list(where)
        if q.time_range:
            start, end = q.time_range
            bounds = [f"DATE '{value.isoformat()}'" for value in (start, end)]
            if comparison:
                bounds = [f"({bound} - {shift})" for bound in bounds]
            column = _identifier(model.time_dimension.column)
            conditions.extend([f"{column} >= {bounds[0]}", f"{column} < {bounds[1]}"])
        return _aggregate_sql(metrics, keys, predicates, source, conditions)

    ctes = ["cur AS (\n" + aggregate() + "\n)"]
    if q.compare_to:
        ctes.append("cmp AS (\n" + aggregate(comparison=True) + "\n)")
    columns = [
        f"cur.{_identifier(name)}" for name in output[: len(keys) + len(metrics)]
    ]
    if q.compare_to:
        for metric in metrics:
            name = _identifier(metric.name)
            columns.extend(
                [
                    f"cmp.{name} AS {_identifier(metric.name + '_compare')}",
                    (
                        f"round((cur.{name} - cmp.{name}) / nullif(cmp.{name}, 0) * 100, 2) "
                        f"AS {_identifier(metric.name + '_change_pct')}"
                    ),
                ]
            )
    if q.share:
        partition = "PARTITION BY cur.period" if q.grain else ""
        for metric in metrics:
            name = _identifier(metric.name)
            columns.append(
                f"round(cur.{name} / nullif(sum(cur.{name}) OVER ({partition}), 0) * 100, 2) "
                f"AS {_identifier(metric.name + '_share_pct')}"
            )
    sql = "WITH " + ",\n".join(ctes) + "\nSELECT " + ", ".join(columns) + "\nFROM cur"
    if q.compare_to:
        on = [f"cur.period = cmp.period + {shift}"] if q.grain else []
        if q.compare_to == "previous_period" and q.grain in {"day", "hour"}:
            # Compare elapsed offsets, not calendar addition (which clamps month
            # ends and can match several comparison periods to one current key).
            # TIMESTAMP starts make both DATE and TIMESTAMP periods subtract to
            # intervals, even though DuckDB's DATE - DATE returns integer days.
            start = f"TIMESTAMP '{q.time_range[0].isoformat()}'"
            on = [f"cur.period - {start} = cmp.period - ({start} - {shift})"]
        on.extend(
            f"cur.{_identifier(dim.name)} IS NOT DISTINCT FROM cmp.{_identifier(dim.name)}"
            for dim in dimensions
        )
        sql += "\nLEFT JOIN cmp ON " + (" AND ".join(on) if on else "TRUE")
    order = []
    if q.order_by is not None:
        direction = "DESC" if q.order_desc else "ASC"
        order.append(f"{_identifier(q.order_by)} {direction} NULLS LAST")
    order.extend(
        f"{_identifier(name)} ASC NULLS LAST" for name, _ in keys if name != q.order_by
    )
    if order:
        sql += "\nORDER BY " + ", ".join(order)
    if q.limit is not None:
        sql += f"\nLIMIT {q.limit}"
    return sql
