"""Deterministic dataset profiler on the synthetic events fixture."""

import shutil
from datetime import date, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from analyst.calibration.profile import Profile, profile_dataset, write_profile
from analyst.config import Settings
from analyst.data.dataset import DatasetConfig, load_dataset


def _weekly_acf_dataset(tmp_path: Path, synthetic_dataset: DatasetConfig) -> DatasetConfig:
    root = tmp_path / "weekly-acf"
    data = root / "data"
    data.mkdir(parents=True)
    start = date(2024, 1, 1)
    days = 120
    rows: list[tuple] = []
    for offset in range(days):
        day = start + timedelta(days=offset)
        is_weekend = day.weekday() >= 5
        hours = range(0, 24, 2) if is_weekend else range(24)
        for hour in hours:
            rows.append(
                (
                    datetime(day.year, day.month, day.day, hour),
                    (hour % 4) + 1,
                    float(100 + day.weekday()),
                    1,
                )
            )
    columns = list(zip(*rows, strict=True))
    pq.write_table(
        pa.table(
            {
                "ts": pa.array(columns[0], type=pa.timestamp("us")),
                "region_id": pa.array(columns[1], type=pa.int32()),
                "amount": pa.array(columns[2], type=pa.float64()),
                "qty": pa.array(columns[3], type=pa.int32()),
            }
        ),
        data / "events_weekly.parquet",
    )
    (data / "regions.csv").write_text(
        "region_id,area\n1,North\n2,North\n3,South\n4,South\n"
    )
    dataset_dir = root / "datasets" / "events"
    shutil.copytree(synthetic_dataset.dir, dataset_dir)
    end = start + timedelta(days=days)
    (dataset_dir / "dataset.yaml").write_text(
        f"""\
name: events
description: Weekend-sparse hourly events
files: data/events_weekly.parquet
table: events
time_column: ts
valid_time_range: [{start.isoformat()}, {end.isoformat()}]
lookups:
  - {{name: regions, file: data/regions.csv}}
"""
    )
    settings = Settings(repo_root=root, datasets_dir=root / "datasets")
    return load_dataset("events", settings)


def _numeric_minmax_dataset(tmp_path: Path, synthetic_dataset: DatasetConfig) -> DatasetConfig:
    root = tmp_path / "minmax"
    data = root / "data"
    data.mkdir(parents=True)
    rows = [
        (datetime(2024, 1, 15, 12), 1, 9.0, 1),
        (datetime(2024, 1, 15, 13), 1, 10.0, 1),
        (datetime(2024, 1, 15, 14), 1, -5.0, 1),
        (datetime(2024, 1, 15, 15), 1, -50.0, 1),
    ]
    pq.write_table(
        pa.table(
            {
                "ts": pa.array([row[0] for row in rows], type=pa.timestamp("us")),
                "region_id": pa.array([row[1] for row in rows], type=pa.int32()),
                "amount": pa.array([row[2] for row in rows], type=pa.float64()),
                "qty": pa.array([row[3] for row in rows], type=pa.int32()),
            }
        ),
        data / "events.parquet",
    )
    (data / "regions.csv").write_text("region_id,area\n1,North\n")
    dataset_dir = root / "datasets" / "events"
    shutil.copytree(synthetic_dataset.dir, dataset_dir)
    (dataset_dir / "dataset.yaml").write_text(
        """\
name: events
description: Min/max ordering check
files: data/events.parquet
table: events
time_column: ts
valid_time_range: [2024-01-01, 2024-02-01]
lookups:
  - {name: regions, file: data/regions.csv}
"""
    )
    settings = Settings(repo_root=root, datasets_dir=root / "datasets")
    return load_dataset("events", settings)


def _sparse_material_dataset(tmp_path: Path, synthetic_dataset: DatasetConfig) -> DatasetConfig:
    root = tmp_path / "sparse"
    data = root / "data"
    data.mkdir(parents=True)
    rows = [(datetime(2024, 1, 1, 12), 1, 1.0, 1) for _ in range(1000)]
    rare = [None] * 995 + ["x"] * 5
    pq.write_table(
        pa.table(
            {
                "ts": pa.array([row[0] for row in rows], type=pa.timestamp("us")),
                "region_id": pa.array([row[1] for row in rows], type=pa.int32()),
                "amount": pa.array([row[2] for row in rows], type=pa.float64()),
                "qty": pa.array([row[3] for row in rows], type=pa.int32()),
                "rare_flag": pa.array(rare, type=pa.string()),
            }
        ),
        data / "events.parquet",
    )
    (data / "regions.csv").write_text("region_id,area\n1,North\n")
    dataset_dir = root / "datasets" / "events"
    shutil.copytree(synthetic_dataset.dir, dataset_dir)
    (dataset_dir / "dataset.yaml").write_text(
        """\
name: events
description: Sparse non-null months
files: data/events.parquet
table: events
time_column: ts
valid_time_range: [2024-01-01, 2024-02-01]
lookups:
  - {name: regions, file: data/regions.csv}
"""
    )
    settings = Settings(repo_root=root, datasets_dir=root / "datasets")
    return load_dataset("events", settings)


def _dual_join_key_dataset(tmp_path: Path, synthetic_dataset: DatasetConfig) -> DatasetConfig:
    root = tmp_path / "dual-join"
    data = root / "data"
    data.mkdir(parents=True)
    rows = [
        (datetime(2024, 1, day, 12), day, day, 1.0)
        for day in range(1, 6)
    ]
    pq.write_table(
        pa.table(
            {
                "ts": pa.array([row[0] for row in rows], type=pa.timestamp("us")),
                "pickup_id": pa.array([row[1] for row in rows], type=pa.int32()),
                "dropoff_id": pa.array([row[2] for row in rows], type=pa.int32()),
                "amount": pa.array([row[3] for row in rows], type=pa.float64()),
            }
        ),
        data / "events.parquet",
    )
    (data / "zones.csv").write_text(
        "LocationID,Borough,Zone\n"
        "1,B1,Z1\n2,B1,Z2\n3,B2,Z3\n4,B2,Z4\n5,B3,Z5\n"
    )
    dataset_dir = root / "datasets" / "events"
    shutil.copytree(synthetic_dataset.dir, dataset_dir)
    (dataset_dir / "dataset.yaml").write_text(
        """\
name: events
description: Two join keys to one lookup
files: data/events.parquet
table: events
time_column: ts
valid_time_range: [2024-01-01, 2024-02-01]
lookups:
  - {name: zones, file: data/zones.csv}
"""
    )
    settings = Settings(repo_root=root, datasets_dir=root / "datasets")
    return load_dataset("events", settings)


def _false_join_dataset(tmp_path: Path, synthetic_dataset: DatasetConfig) -> DatasetConfig:
    root = tmp_path / "false-join"
    data = root / "data"
    data.mkdir(parents=True)
    start = date(2024, 1, 1)
    rows = [
        (datetime.combine(start + timedelta(days=offset), datetime.min.time().replace(hour=12)), offset + 1, float(offset + 1))
        for offset in range(50)
    ]
    pq.write_table(
        pa.table(
            {
                "ts": pa.array([row[0] for row in rows], type=pa.timestamp("us")),
                "small_code": pa.array([row[1] for row in rows], type=pa.int32()),
                "amount": pa.array([row[2] for row in rows], type=pa.float64()),
            }
        ),
        data / "events.parquet",
    )
    (data / "zones.csv").write_text(
        "LocationID,Borough,Zone\n"
        + "\n".join(f"{index},B,Z{index}" for index in range(1, 6))
        + "\n"
    )
    dataset_dir = root / "datasets" / "events"
    shutil.copytree(synthetic_dataset.dir, dataset_dir)
    (dataset_dir / "dataset.yaml").write_text(
        """\
name: events
description: False join key
files: data/events.parquet
table: events
time_column: ts
valid_time_range: [2024-01-01, 2024-02-01]
lookups:
  - {name: zones, file: data/zones.csv}
"""
    )
    settings = Settings(repo_root=root, datasets_dir=root / "datasets")
    return load_dataset("events", settings)


def _roles_probe_dataset(tmp_path: Path, synthetic_dataset: DatasetConfig) -> DatasetConfig:
    root = tmp_path / "roles"
    data = root / "data"
    data.mkdir(parents=True)
    row_count = 8000
    labels = [f"label-{index % 2000}" for index in range(row_count)]
    pq.write_table(
        pa.table(
            {
                "ts": pa.array(
                    [datetime(2024, 1, 1, index % 24) for index in range(row_count)],
                    type=pa.timestamp("us"),
                ),
                "valid": pa.array([1.5] * row_count, type=pa.float64()),
                "big_label": pa.array(labels, type=pa.string()),
                "amount": pa.array([float(index) for index in range(row_count)], type=pa.float64()),
            }
        ),
        data / "events.parquet",
    )
    dataset_dir = root / "datasets" / "events"
    shutil.copytree(synthetic_dataset.dir, dataset_dir)
    (dataset_dir / "dataset.yaml").write_text(
        """\
name: events
description: Role inference probes
files: data/events.parquet
table: events
time_column: ts
valid_time_range: [2024-01-01, 2024-02-01]
lookups: []
"""
    )
    settings = Settings(repo_root=root, datasets_dir=root / "datasets")
    return load_dataset("events", settings)


def _anomaly_dataset(tmp_path: Path, synthetic_dataset: DatasetConfig) -> DatasetConfig:
    root = tmp_path / "anomaly"
    data = root / "data"
    data.mkdir(parents=True)
    rows: list[tuple] = []
    start = date(2024, 1, 1)
    for offset in range(70):
        day = start + timedelta(days=offset)
        if day == date(2024, 2, 20):
            for hour in range(2000):
                rows.append((datetime(day.year, day.month, day.day, hour % 24), 1, 1.0, 1))
            continue
        if day in {date(2024, 1, 20), date(2024, 1, 21)}:
            hours = (0, 12)
        else:
            hours = range(24 + (offset % 3))
        for hour in hours:
            rows.append((datetime(day.year, day.month, day.day, hour % 24), 1, 1.0, 1))
    pq.write_table(
        pa.table(
            {
                "ts": pa.array([row[0] for row in rows], type=pa.timestamp("us")),
                "region_id": pa.array([row[1] for row in rows], type=pa.int32()),
                "amount": pa.array([row[2] for row in rows], type=pa.float64()),
                "qty": pa.array([row[3] for row in rows], type=pa.int32()),
            }
        ),
        data / "events.parquet",
    )
    (data / "regions.csv").write_text("region_id,area\n1,North\n")
    dataset_dir = root / "datasets" / "events"
    shutil.copytree(synthetic_dataset.dir, dataset_dir)
    (dataset_dir / "dataset.yaml").write_text(
        """\
name: events
description: Spike and gap anomaly check
files: data/events.parquet
table: events
time_column: ts
valid_time_range: [2024-01-01, 2024-03-01]
lookups:
  - {name: regions, file: data/regions.csv}
"""
    )
    settings = Settings(repo_root=root, datasets_dir=root / "datasets")
    return load_dataset("events", settings)


def _soft_gap_dataset(tmp_path: Path, synthetic_dataset: DatasetConfig) -> DatasetConfig:
    root = tmp_path / "soft-gap"
    data = root / "data"
    data.mkdir(parents=True)
    rows: list[tuple] = []
    start = date(2024, 1, 1)
    for offset in range(35):
        day = start + timedelta(days=offset)
        if day == date(2024, 1, 15):
            hours = range(2)
        elif day == date(2024, 1, 16):
            hours = range(1)
        elif day in {date(2024, 1, 17), date(2024, 1, 28)}:
            hours = range(10)
        else:
            hours = range(24)
        for hour in hours:
            rows.append((datetime(day.year, day.month, day.day, hour), 1, 1.0, 1))
    pq.write_table(
        pa.table(
            {
                "ts": pa.array([row[0] for row in rows], type=pa.timestamp("us")),
                "region_id": pa.array([row[1] for row in rows], type=pa.int32()),
                "amount": pa.array([row[2] for row in rows], type=pa.float64()),
                "qty": pa.array([row[3] for row in rows], type=pa.int32()),
            }
        ),
        data / "events.parquet",
    )
    dataset_dir = root / "datasets" / "events"
    shutil.copytree(synthetic_dataset.dir, dataset_dir)
    (dataset_dir / "dataset.yaml").write_text(
        """\
name: events
description: Soft gap edge vs isolated dip
files: data/events.parquet
table: events
time_column: ts
valid_time_range: [2024-01-01, 2024-03-01]
lookups: []
"""
    )
    settings = Settings(repo_root=root, datasets_dir=root / "datasets")
    return load_dataset("events", settings)


def test_roles(synthetic_dataset):
    profile = profile_dataset(synthetic_dataset)
    by_name = {column.name: column for column in profile.columns}
    assert by_name["ts"].role == "time"
    assert by_name["amount"].role == "measure"
    assert by_name["region_id"].role in {"id", "dimension"}


def test_gap_detected(synthetic_dataset):
    profile = profile_dataset(synthetic_dataset)
    assert any(
        start <= date(2024, 1, 20) and end >= date(2024, 1, 21)
        for start, end in profile.daily.gaps
    )


def test_implausible_timestamps(synthetic_dataset):
    profile = profile_dataset(synthetic_dataset)
    assert profile.implausible_timestamps == 3


def test_schema_drift_first_non_null(synthetic_dataset):
    profile = profile_dataset(synthetic_dataset)
    by_name = {column.name: column for column in profile.columns}
    assert by_name["note"].first_non_null_month == "2024-02"
    assert by_name["note"].first_material_month == "2024-02"


def test_weekly_acf_detected(synthetic_dataset, tmp_path):
    dataset = _weekly_acf_dataset(tmp_path, synthetic_dataset)
    profile = profile_dataset(dataset)
    assert 7 in profile.daily.seasonal_lags


def test_lookup_dependency(synthetic_dataset):
    profile = profile_dataset(synthetic_dataset)
    assert any(
        dep.determinant == "region_id"
        and dep.dependent == "area"
        and dep.via_lookup == "regions"
        for dep in profile.dependencies
    )


def test_profile_json_roundtrip(synthetic_dataset, tmp_path):
    profile = profile_dataset(synthetic_dataset)
    path = write_profile(profile, synthetic_dataset.model_copy(update={"root": tmp_path}))
    restored = Profile.model_validate_json(path.read_text())
    assert restored == profile


def test_numeric_min_max_not_lexicographic(tmp_path, synthetic_dataset):
    dataset = _numeric_minmax_dataset(tmp_path, synthetic_dataset)
    amount = next(c for c in profile_dataset(dataset).columns if c.name == "amount")
    assert amount.min == -50
    assert amount.max == 10


def test_sparse_first_material_month_fallback(tmp_path, synthetic_dataset):
    dataset = _sparse_material_dataset(tmp_path, synthetic_dataset)
    rare = next(c for c in profile_dataset(dataset).columns if c.name == "rare_flag")
    assert rare.first_non_null_month == "2024-01"
    assert rare.first_material_month == "2024-01"


def test_dual_lookup_join_keys(tmp_path, synthetic_dataset):
    dataset = _dual_join_key_dataset(tmp_path, synthetic_dataset)
    profile = profile_dataset(dataset)
    main_det = {d.determinant for d in profile.dependencies if d.via_lookup == "zones"}
    assert "pickup_id" in main_det and "dropoff_id" in main_det


def test_false_join_not_detected(tmp_path, synthetic_dataset):
    dataset = _false_join_dataset(tmp_path, synthetic_dataset)
    profile = profile_dataset(dataset)
    assert not any(d.determinant == "small_code" for d in profile.dependencies)


def test_monthly_series_has_no_gaps_or_anomalies(synthetic_dataset):
    profile = profile_dataset(synthetic_dataset)
    assert profile.monthly.gaps == []
    assert profile.monthly.anomalies == []


def test_role_valid_measure_not_id(tmp_path, synthetic_dataset):
    dataset = _roles_probe_dataset(tmp_path, synthetic_dataset)
    by_name = {c.name: c for c in profile_dataset(dataset).columns}
    assert by_name["valid"].role != "id"
    assert by_name["big_label"].role == "other"


def test_spike_anomaly_gap_days_excluded(tmp_path, synthetic_dataset):
    dataset = _anomaly_dataset(tmp_path, synthetic_dataset)
    profile = profile_dataset(dataset)
    assert any(d == date(2024, 2, 20) for d, _ in profile.daily.anomalies)
    gap_days = {
        day
        for start, end in profile.daily.gaps
        for day in (
            start + timedelta(days=offset)
            for offset in range((end - start).days + 1)
        )
    }
    assert not any(d in gap_days for d, _ in profile.daily.anomalies)


def test_soft_gap_extension_vs_isolated_dip(tmp_path, synthetic_dataset):
    dataset = _soft_gap_dataset(tmp_path, synthetic_dataset)
    profile = profile_dataset(dataset)
    anchor = date(2024, 1, 15)
    gap = next(
        (start, end)
        for start, end in profile.daily.gaps
        if start <= anchor <= end
    )
    assert gap[1] == date(2024, 1, 17)
    isolated = date(2024, 1, 28)
    assert not any(start <= isolated <= end for start, end in profile.daily.gaps)
