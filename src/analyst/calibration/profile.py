"""Deterministic dataset profile computed entirely from DuckDB aggregates."""

from __future__ import annotations

import statistics
import time
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from analyst.data.dataset import DatasetConfig
from analyst.data.engine import Coverage, Engine


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


class ColumnProfile(BaseModel):
    name: str
    type: str
    null_rate: float
    distinct_approx: int
    role: Literal["time", "dimension", "measure", "id", "other"]
    top_values: list[tuple[str, int]] | None
    min: Any | None
    max: Any | None
    nonpositive_rate: float | None
    first_non_null_month: str | None
    first_material_month: str | None


class Dependency(BaseModel):
    determinant: str
    dependent: str
    via_lookup: str | None


class SeriesProfile(BaseModel):
    grain: Literal["day", "month"]
    points: int
    gaps: list[tuple[date, date]]
    anomalies: list[tuple[date, float]]
    acf: dict[int, float]
    seasonal_lags: list[int]


class Profile(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset: str
    rows: int
    raw_rows: int
    implausible_timestamps: int
    coverage: Coverage
    columns: list[ColumnProfile]
    dependencies: list[Dependency]
    daily: SeriesProfile
    monthly: SeriesProfile
    elapsed_s: float


def profile_dataset(dataset: DatasetConfig) -> Profile:
    started = time.perf_counter()
    engine = Engine(dataset, timeout_s=120)
    table = _quote(dataset.table)
    raw = _quote(dataset.table + "_raw")
    time_col = _quote(dataset.time_column)
    valid_start, valid_end = dataset.valid_time_range

    rows = _scalar(
        engine,
        f"SELECT COUNT(*)::BIGINT FROM {table}",
    )
    raw_rows = _scalar(engine, f"SELECT COUNT(*)::BIGINT FROM {raw}")
    implausible = _scalar(
        engine,
        "SELECT COUNT(*)::BIGINT FROM "
        f"{raw} WHERE {time_col} < DATE {_literal(valid_start.isoformat())} "
        f"OR {time_col} >= DATE {_literal(valid_end.isoformat())}",
    )
    coverage = engine.coverage()
    columns = _profile_columns(engine, dataset, rows)
    dependencies = _profile_dependencies(engine, dataset, columns)
    daily = _daily_series(engine, dataset)
    monthly = _monthly_series(engine, dataset)
    elapsed = time.perf_counter() - started
    return Profile(
        dataset=dataset.name,
        rows=rows,
        raw_rows=raw_rows,
        implausible_timestamps=implausible,
        coverage=coverage,
        columns=columns,
        dependencies=dependencies,
        daily=daily,
        monthly=monthly,
        elapsed_s=elapsed,
    )


def write_profile(profile: Profile, dataset: DatasetConfig) -> Path:
    directory = dataset.memory_dir
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "profile.json"
    path.write_text(profile.model_dump_json(indent=2))
    return path


def _scalar(engine: Engine, sql: str) -> int:
    return int(engine.execute(sql)[0][0].as_py())


def _profile_columns(
    engine: Engine, dataset: DatasetConfig, rows: int,
) -> list[ColumnProfile]:
    table = _quote(dataset.table)
    specs = engine.columns()
    if not specs:
        return []
    month_fields = _month_presence(engine, dataset, [name for name, _ in specs])
    parts: list[str] = []
    for name, dtype in specs:
        col = _quote(name)
        nonpositive = ""
        if _is_numeric(dtype):
            nonpositive = (
                f", AVG(CASE WHEN {col} IS NOT NULL AND {col} <= 0 THEN 1.0 ELSE 0.0 END) "
                "AS nonpositive_rate"
            )
        else:
            nonpositive = ", CAST(NULL AS DOUBLE) AS nonpositive_rate"
        parts.append(
            f"SELECT {_literal(name)} AS name, "
            f"CAST({_literal(dtype)} AS VARCHAR) AS dtype, "
            f"AVG(CASE WHEN {col} IS NULL THEN 1.0 ELSE 0.0 END) AS null_rate, "
            f"approx_count_distinct({col})::BIGINT AS distinct_approx, "
            f"CAST(MIN({col}) AS VARCHAR) AS min_value, "
            f"CAST(MAX({col}) AS VARCHAR) AS max_value "
            f"{nonpositive} "
            f"FROM {table}"
        )
    result = engine.execute(" UNION ALL ".join(parts))
    profiles: list[ColumnProfile] = []
    for row in zip(*[result[column].to_pylist() for column in result.column_names]):
        name, dtype, null_rate, distinct_approx, min_v, max_v, nonpos = row
        first_month, material_month = month_fields.get(name, (None, None))
        role = _infer_role(
            name, dtype, int(distinct_approx), rows, dataset.time_column,
        )
        top_values = None
        if role == "dimension":
            top_values = _top_values(engine, dataset.table, name)
        nonpositive_rate = float(nonpos) if nonpos is not None and _is_numeric(dtype) else None
        profiles.append(
            ColumnProfile(
                name=name,
                type=dtype,
                null_rate=float(null_rate),
                distinct_approx=int(distinct_approx),
                role=role,
                top_values=top_values,
                min=_parse_bound(dtype, min_v),
                max=_parse_bound(dtype, max_v),
                nonpositive_rate=nonpositive_rate,
                first_non_null_month=first_month,
                first_material_month=material_month,
            )
        )
    return profiles


def _month_presence(
    engine: Engine, dataset: DatasetConfig, columns: list[str],
) -> dict[str, tuple[str | None, str | None]]:
    if not columns:
        return {}
    table = _quote(dataset.table)
    time_col = _quote(dataset.time_column)
    selects = [
        f"COUNT(*) FILTER (WHERE {_quote(name)} IS NOT NULL)::BIGINT AS {_quote(name)}"
        for name in columns
    ]
    sql = (
        f"SELECT strftime(DATE_TRUNC('month', {time_col}), '%Y-%m') AS month, "
        f"COUNT(*)::BIGINT AS month_rows, {', '.join(selects)} "
        f"FROM {table} GROUP BY DATE_TRUNC('month', {time_col}) ORDER BY 1"
    )
    result = engine.execute(sql)
    months = result["month"].to_pylist()
    month_rows = result["month_rows"].to_pylist()
    non_null: dict[str, list[tuple[str, int, int]]] = {name: [] for name in columns}
    for row_index, month in enumerate(months):
        total = int(month_rows[row_index])
        for name in columns:
            count = int(result[name][row_index].as_py())
            if count > 0:
                non_null[name].append((month, count, total))
    out: dict[str, tuple[str | None, str | None]] = {}
    for name, buckets in non_null.items():
        if not buckets:
            out[name] = (None, None)
            continue
        literal_first = buckets[0][0]
        material = next(
            (month for month, count, total in buckets if count / total >= 0.01),
            None,
        )
        out[name] = (literal_first, material or literal_first)
    return out


def _top_values(engine: Engine, table: str, column: str) -> list[tuple[str, int]]:
    ident = _quote(column)
    rel = _quote(table)
    rows = engine.execute(
        f"SELECT CAST({ident} AS VARCHAR) AS value, COUNT(*)::BIGINT AS n "
        f"FROM {rel} WHERE {ident} IS NOT NULL "
        "GROUP BY 1 ORDER BY n DESC LIMIT 10"
    )
    return [(row[0].as_py(), int(row[1].as_py())) for row in zip(rows[0], rows[1])]


def _infer_role(
    name: str,
    dtype: str,
    distinct: int,
    rows: int,
    time_column: str,
) -> Literal["time", "dimension", "measure", "id", "other"]:
    if name == time_column or _is_timestamp(dtype):
        return "time"
    if _name_marks_id_column(name, dtype):
        return "id"
    if rows > 0 and distinct > rows * 0.5:
        return "id"
    if _is_numeric(dtype) and distinct > 50:
        return "measure"
    if _is_string(dtype) and distinct <= 1000:
        return "dimension"
    if _is_integer(dtype) and distinct <= 1000:
        return "dimension"
    return "other"


def _profile_dependencies(
    engine: Engine,
    dataset: DatasetConfig,
    columns: list[ColumnProfile],
) -> list[Dependency]:
    if not dataset.lookups:
        return []
    main_cols = [
        column.name
        for column in columns
        if column.role == "id"
        and (_is_integer(column.type) or _is_string(column.type))
    ]
    deps: list[Dependency] = []
    for lookup in dataset.lookups:
        lookup_cols = _lookup_columns(engine, lookup.name)
        join_keys: dict[str, list[str]] = {}
        for lk_col, lk_dtype in lookup_cols:
            if not (_is_integer(lk_dtype) or _is_string(lk_dtype)):
                continue
            if not _name_marks_id_column(lk_col, lk_dtype):
                continue
            matched = _matching_main_columns(
                engine, dataset, lookup.name, lk_col, main_cols,
            )
            if matched:
                join_keys[lk_col] = matched
        for lk_col, main_columns in join_keys.items():
            for main_col in main_columns:
                for other_col, _ in lookup_cols:
                    if other_col == lk_col:
                        continue
                    if _functional_dependency(engine, lookup.name, lk_col, other_col):
                        deps.append(
                            Dependency(
                                determinant=main_col,
                                dependent=other_col,
                                via_lookup=lookup.name,
                            )
                        )
        for col_a, _ in lookup_cols:
            for col_b, _ in lookup_cols:
                if col_a == col_b:
                    continue
                if _functional_dependency(engine, lookup.name, col_a, col_b):
                    deps.append(
                        Dependency(
                            determinant=col_a,
                            dependent=col_b,
                            via_lookup=lookup.name,
                        )
                    )
    return _dedupe_dependencies(deps)


def _lookup_columns(engine: Engine, lookup: str) -> list[tuple[str, str]]:
    empty = engine.execute(f"SELECT * FROM {_quote(lookup)} WHERE FALSE")
    return [
        (name, str(empty.field(i).type))
        for i, name in enumerate(empty.column_names)
    ]


def _matching_main_columns(
    engine: Engine,
    dataset: DatasetConfig,
    lookup: str,
    lookup_col: str,
    main_cols: list[str],
) -> list[str]:
    lk = _quote(lookup)
    lk_col = _quote(lookup_col)
    main = _quote(dataset.table)
    matched: list[str] = []
    for main_col in main_cols:
        mcol = _quote(main_col)
        row = engine.execute(
            f"WITH lk AS (SELECT DISTINCT CAST({lk_col} AS VARCHAR) AS v FROM {lk}), "
            f"present AS ("
            f"SELECT DISTINCT CAST({mcol} AS VARCHAR) AS v FROM {main} WHERE {mcol} IS NOT NULL"
            "), "
            "lk_n AS (SELECT COUNT(*)::DOUBLE AS n FROM lk), "
            "main_n AS (SELECT COUNT(*)::DOUBLE AS n FROM present) "
            "SELECT "
            "COALESCE("
            "(SELECT COUNT(*)::DOUBLE FROM lk WHERE v IN (SELECT v FROM present)) "
            "/ NULLIF((SELECT n FROM lk_n), 0), 0) AS lk_in_main, "
            "COALESCE("
            "(SELECT COUNT(*)::DOUBLE FROM present WHERE v IN (SELECT v FROM lk)) "
            "/ NULLIF((SELECT n FROM main_n), 0), 0) AS main_in_lk"
        )
        lk_in_main, main_in_lk = row[0][0].as_py(), row[1][0].as_py()
        if lk_in_main >= 0.95 and main_in_lk >= 0.95:
            matched.append(main_col)
    return matched


def _functional_dependency(
    engine: Engine, lookup: str, determinant: str, dependent: str,
) -> bool:
    lk = _quote(lookup)
    det = _quote(determinant)
    dep = _quote(dependent)
    violations = _scalar(
        engine,
        "SELECT COUNT(*)::BIGINT FROM ("
        f"SELECT {det}, COUNT(DISTINCT {dep})::BIGINT AS n "
        f"FROM {lk} GROUP BY 1"
        ") WHERE n > 1",
    )
    return violations == 0


def _dedupe_dependencies(deps: list[Dependency]) -> list[Dependency]:
    seen: set[tuple[str, str, str | None]] = set()
    out: list[Dependency] = []
    for dep in deps:
        key = (dep.determinant, dep.dependent, dep.via_lookup)
        if key not in seen:
            seen.add(key)
            out.append(dep)
    return out


def _daily_series(engine: Engine, dataset: DatasetConfig) -> SeriesProfile:
    table = _quote(dataset.table)
    time_col = _quote(dataset.time_column)
    sql = (
        "WITH counted AS ("
        f"  SELECT CAST(DATE_TRUNC('day', {time_col}) AS DATE) AS bucket, "
        "COUNT(*)::DOUBLE AS value FROM "
        f"{table} GROUP BY 1"
        "), bounds AS (SELECT MIN(bucket) AS min_day, MAX(bucket) AS max_day FROM counted), "
        "spine AS ("
        "  SELECT CAST(UNNEST(GENERATE_SERIES("
        "    (SELECT min_day FROM bounds), (SELECT max_day FROM bounds), INTERVAL 1 DAY"
        "  )) AS DATE) AS bucket"
        ") "
        "SELECT spine.bucket, COALESCE(counted.value, 0.0) AS value "
        "FROM spine LEFT JOIN counted USING (bucket) ORDER BY 1"
    )
    dates, values = _series_rows(engine, sql)
    return _build_series_profile(dates, values, "day")


def _monthly_series(engine: Engine, dataset: DatasetConfig) -> SeriesProfile:
    table = _quote(dataset.table)
    time_col = _quote(dataset.time_column)
    sql = (
        "WITH counted AS ("
        f"  SELECT DATE_TRUNC('month', {time_col})::DATE AS bucket, "
        "COUNT(*)::DOUBLE AS value FROM "
        f"{table} GROUP BY 1"
        "), bounds AS (SELECT MIN(bucket) AS min_day, MAX(bucket) AS max_day FROM counted), "
        "spine AS ("
        "  SELECT CAST(UNNEST(GENERATE_SERIES("
        "    (SELECT min_day FROM bounds), (SELECT max_day FROM bounds), INTERVAL 1 MONTH"
        "  )) AS DATE) AS bucket"
        ") "
        "SELECT spine.bucket, COALESCE(counted.value, 0.0) AS value "
        "FROM spine LEFT JOIN counted USING (bucket) ORDER BY 1"
    )
    dates, values = _series_rows(engine, sql)
    acf, seasonal = _compute_acf(values, dates, "month")
    return SeriesProfile(
        grain="month",
        points=len(values),
        gaps=[],
        anomalies=[],
        acf=acf,
        seasonal_lags=seasonal,
    )


def _series_rows(engine: Engine, sql: str) -> tuple[list[date], list[float]]:
    table = engine.execute(sql)
    dates = [row.as_py() for row in table[0]]
    values = [float(row.as_py()) for row in table[1]]
    return dates, values


def _build_series_profile(
    dates: list[date], values: list[float], grain: Literal["day", "month"],
) -> SeriesProfile:
    gaps = _detect_gaps(dates, values)
    gap_days = _expand_gap_days(gaps)
    anomalies = _detect_anomalies(dates, values, gap_days)
    acf, seasonal = _compute_acf(values, dates, grain)
    return SeriesProfile(
        grain=grain,
        points=len(values),
        gaps=gaps,
        anomalies=anomalies,
        acf=acf,
        seasonal_lags=seasonal,
    )


def _detect_gaps(dates: list[date], values: list[float]) -> list[tuple[date, date]]:
    window = 28 if len(values) >= 7 else max(3, len(values))
    medians = _centered_rolling_median(values, window)
    hard = [
        medians[index] > 0 and values[index] < 0.1 * medians[index]
        for index in range(len(values))
    ]
    soft = [
        medians[index] > 0 and values[index] < 0.5 * medians[index]
        for index in range(len(values))
    ]
    flags = list(hard)
    for index, is_hard in enumerate(hard):
        if not is_hard:
            continue
        offset = index - 1
        while offset >= 0 and soft[offset]:
            flags[offset] = True
            offset -= 1
        offset = index + 1
        while offset < len(flags) and soft[offset]:
            flags[offset] = True
            offset += 1
    return _merge_ranges(dates, flags)


def _centered_rolling_median(values: list[float], window: int) -> list[float]:
    half = window // 2
    out: list[float] = []
    for index in range(len(values)):
        start = max(0, index - half)
        end = min(len(values), index + half + 1)
        out.append(statistics.median(values[start:end]))
    return out


def _merge_ranges(dates: list[date], flags: list[bool]) -> list[tuple[date, date]]:
    ranges: list[tuple[date, date]] = []
    start: date | None = None
    for index, flag in enumerate(flags):
        if flag and start is None:
            start = dates[index]
        elif not flag and start is not None:
            ranges.append((start, dates[index - 1]))
            start = None
    if start is not None:
        ranges.append((start, dates[-1]))
    return ranges


def _expand_gap_days(gaps: list[tuple[date, date]]) -> set[date]:
    days: set[date] = set()
    for start, end in gaps:
        day = start
        while day <= end:
            days.add(day)
            day += timedelta(days=1)
    return days


def _detect_anomalies(
    dates: list[date], values: list[float], gap_days: set[date],
) -> list[tuple[date, float]]:
    dow_buckets: dict[int, list[float]] = defaultdict(list)
    for day, value in zip(dates, values, strict=True):
        dow_buckets[day.weekday()].append(value)
    dow_median = {dow: statistics.median(bucket) for dow, bucket in dow_buckets.items()}
    residual_points: list[tuple[date, float]] = []
    for day, value in zip(dates, values, strict=True):
        if day in gap_days:
            continue
        residual_points.append((day, value - dow_median[day.weekday()]))
    if len(residual_points) < 3:
        return []
    residuals = [point[1] for point in residual_points]
    med = statistics.median(residuals)
    mad = statistics.median([abs(value - med) for value in residuals])
    if mad == 0:
        return []
    scale = 1.4826 * mad
    out: list[tuple[date, float]] = []
    for day, residual in residual_points:
        z = (residual - med) / scale
        if abs(z) > 4:
            out.append((day, z))
    return out


def _compute_acf(
    values: list[float],
    dates: list[date],
    grain: Literal["day", "month"],
) -> tuple[dict[int, float], list[int]]:
    if grain == "day":
        span_days = (dates[-1] - dates[0]).days + 1 if dates else 0
        lags = [1, 7, 14, 30]
        if span_days >= 730:
            lags.append(365)
        seasonal_candidates = [7, 14, 30, 365]
    else:
        lags = [1]
        if len(values) >= 24:
            lags.append(12)
        seasonal_candidates = [12]
    acf = {lag: _acf_at_lag(values, lag) for lag in lags}
    seasonal = [lag for lag in seasonal_candidates if lag in acf and acf[lag] > 0.3]
    return acf, seasonal


def _acf_at_lag(values: list[float], lag: int) -> float:
    if lag <= 0 or lag >= len(values):
        return 0.0
    n = len(values)
    mean = sum(values) / n
    denom = sum((value - mean) ** 2 for value in values)
    if denom == 0:
        return 0.0
    numer = sum(
        (values[index] - mean) * (values[index + lag] - mean)
        for index in range(n - lag)
    )
    return numer / denom


def _name_marks_id_column(name: str, dtype: str) -> bool:
    if not _is_integer(dtype):
        return False
    return name.endswith("ID") or name.lower().endswith("_id")


def _is_numeric(dtype: str) -> bool:
    upper = dtype.upper()
    if "INTERVAL" in upper or "POINT" in upper:
        return False
    return any(token in upper for token in ("INT", "DOUBLE", "FLOAT", "DECIMAL", "HUGEINT"))


def _is_integer(dtype: str) -> bool:
    upper = dtype.upper()
    return "INT" in upper and "INTERVAL" not in upper and "POINT" not in upper


def _is_string(dtype: str) -> bool:
    upper = dtype.upper()
    return "VARCHAR" in upper or upper == "STRING" or "TEXT" in upper


def _is_timestamp(dtype: str) -> bool:
    upper = dtype.upper()
    return "TIMESTAMP" in upper or upper == "DATE"


def _parse_bound(dtype: str, value: Any) -> Any:
    if value is None:
        return None
    if _is_numeric(dtype):
        try:
            number = float(value)
            if number.is_integer():
                return int(number)
            return number
        except (TypeError, ValueError):
            return value
    if _is_timestamp(dtype):
        text = str(value)
        try:
            if "T" in text or " " in text:
                stamp = datetime.fromisoformat(text.replace(" ", "T", 1))
            else:
                stamp = datetime.combine(date.fromisoformat(text), datetime.min.time())
            return stamp.isoformat(timespec="seconds")
        except ValueError:
            return text
    return value
