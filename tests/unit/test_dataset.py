from collections import Counter
from datetime import date, datetime, timedelta

import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from analyst.config import Settings
from analyst.data.dataset import load_dataset


def test_load_taxi_dataset():
    ds = load_dataset("nyc_taxi")
    assert ds.table == "trips"
    assert ds.valid_time_range == (date(2023, 1, 1), date(2026, 10, 1))
    assert ds.lookups[0].name == "zones"


def test_data_dirs_unique():
    ds = load_dataset("nyc_taxi")
    assert ds.data_dirs == [ds.root / "data" / "raw"]


def test_memory_dir_under_dataset_folder():
    ds = load_dataset("nyc_taxi")
    assert ds.dir == ds.root / "datasets" / "nyc_taxi"
    assert ds.memory_dir == ds.root / "datasets" / "nyc_taxi" / "memory"


def test_missing_dataset_raises(tmp_path):
    settings = Settings(repo_root=tmp_path, datasets_dir=tmp_path / "datasets")
    missing = tmp_path / "datasets" / "nope" / "dataset.yaml"
    with pytest.raises(FileNotFoundError) as exc_info:
        load_dataset("nope", settings)
    assert str(missing) in str(exc_info.value)


def test_bad_yaml_field_raises(tmp_path):
    dataset_dir = tmp_path / "datasets" / "broken"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "dataset.yaml").write_text(
        """\
name: broken
description: incomplete registration
files: data/*.parquet
table: events
valid_time_range: [2024-01-01, 2024-02-01]
lookups: []
"""
    )
    settings = Settings(repo_root=tmp_path, datasets_dir=tmp_path / "datasets")
    with pytest.raises(ValidationError, match="time_column"):
        load_dataset("broken", settings)


def test_non_mapping_yaml_raises(tmp_path):
    dataset_dir = tmp_path / "datasets" / "broken"
    dataset_dir.mkdir(parents=True)
    path = dataset_dir / "dataset.yaml"
    path.write_text("- not\n- a\n- mapping\n")
    settings = Settings(repo_root=tmp_path, datasets_dir=tmp_path / "datasets")
    with pytest.raises(ValueError) as exc_info:
        load_dataset("broken", settings)
    assert str(exc_info.value) == f"{path}: expected a mapping"


def test_synthetic_fixture_loads(synthetic_dataset):
    root = synthetic_dataset.root
    files = sorted(root.glob(synthetic_dataset.files))
    assert [path.name for path in files] == [
        "events_2024-01.parquet",
        "events_2024-02.parquet",
    ]
    assert synthetic_dataset.table == "events"
    assert synthetic_dataset.time_column == "ts"
    assert synthetic_dataset.valid_time_range == (date(2024, 1, 1), date(2024, 3, 1))
    assert synthetic_dataset.lookups[0].name == "regions"
    assert synthetic_dataset.dir == root / "datasets" / "events"
    assert synthetic_dataset.memory_dir == synthetic_dataset.dir / "memory"
    assert not synthetic_dataset.memory_dir.exists()

    january = pq.read_table(root / "data" / "events_2024-01.parquet")
    february = pq.read_table(root / "data" / "events_2024-02.parquet")
    assert january.column_names == ["ts", "region_id", "amount", "qty"]
    assert february.column_names == ["ts", "region_id", "amount", "qty", "note"]

    jan_ts = january.column("ts").to_pylist()
    feb_ts = february.column("ts").to_pylist()
    assert sum(ts.year == 2001 for ts in jan_ts) == 3
    assert all(ts.year == 2024 for ts in feb_ts)

    jan_days = Counter(ts.date() for ts in jan_ts if ts.year == 2024)
    feb_days = Counter(ts.date() for ts in feb_ts)
    assert jan_days[date(2024, 1, 20)] == 2
    assert jan_days[date(2024, 1, 21)] == 2
    assert jan_days[date(2024, 1, 1)] == 24
    assert feb_days[date(2024, 2, 1)] == 24
    assert set(jan_days) | set(feb_days) == {
        date(2024, 1, 1) + timedelta(days=offset) for offset in range(60)
    }

    regions = set(january.column("region_id").to_pylist()) | set(
        february.column("region_id").to_pylist()
    )
    assert regions == {1, 2, 3, 4}

    amount_by_stamp = {
        ts: amount
        for ts, amount in zip(
            jan_ts, january.column("amount").to_pylist(), strict=True
        )
        if ts.year == 2024
    }
    monday = datetime(2024, 1, 1)
    assert amount_by_stamp[monday] == amount_by_stamp[datetime(2024, 1, 8)]
    assert amount_by_stamp[monday] != amount_by_stamp[datetime(2024, 1, 2)]

    lookup = (root / "data" / "regions.csv").read_text().splitlines()
    assert lookup == [
        "region_id,area",
        "1,North",
        "2,North",
        "3,South",
        "4,South",
    ]
