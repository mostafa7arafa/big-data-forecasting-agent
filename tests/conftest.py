from datetime import date, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from analyst.config import Settings
from analyst.data.dataset import DatasetConfig, load_dataset

_TAXI_SAMPLE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "yellow_tripdata_2023-01.parquet"
)


def pytest_collection_modifyitems(items):
    if _TAXI_SAMPLE.is_file():
        return
    skip = pytest.mark.skip(reason=f"missing {_TAXI_SAMPLE.name}")
    for item in items:
        if "taxi" in item.keywords:
            item.add_marker(skip)


def _hourly_rows() -> tuple[list[tuple], list[tuple]]:
    january: list[tuple] = []
    february: list[tuple] = []
    gap_days = {date(2024, 1, 20), date(2024, 1, 21)}
    day = date(2024, 1, 1)
    while day < date(2024, 3, 1):
        hours = (0, 12) if day in gap_days else range(24)
        bucket = january if day.month == 1 else february
        for hour in hours:
            bucket.append(
                (
                    datetime(day.year, day.month, day.day, hour),
                    (hour % 4) + 1,
                    float(100 + 10 * day.weekday() + hour),
                    1 + hour % 3,
                )
            )
        day += timedelta(days=1)
    for index, stamp in enumerate(
        (
            datetime(2001, 1, 1),
            datetime(2001, 6, 15, 12),
            datetime(2001, 12, 31, 23),
        )
    ):
        january.append((stamp, (index % 4) + 1, 1.0, 1))
    return january, february


def _write_parquet(path: Path, rows: list[tuple], *, note: bool) -> None:
    columns = list(zip(*rows, strict=True))
    arrays = [
        pa.array(columns[0], type=pa.timestamp("us")),
        pa.array(columns[1], type=pa.int32()),
        pa.array(columns[2], type=pa.float64()),
        pa.array(columns[3], type=pa.int32()),
    ]
    names = ["ts", "region_id", "amount", "qty"]
    if note:
        arrays.append(pa.array(["drift"] * len(rows), type=pa.string()))
        names.append("note")
    pq.write_table(pa.Table.from_arrays(arrays, names=names), path)


@pytest.fixture(scope="session")
def synthetic_dataset(tmp_path_factory) -> DatasetConfig:
    root = tmp_path_factory.mktemp("synthetic")
    data = root / "data"
    data.mkdir()
    dataset_dir = root / "datasets" / "events"
    dataset_dir.mkdir(parents=True)
    january, february = _hourly_rows()
    _write_parquet(data / "events_2024-01.parquet", january, note=False)
    _write_parquet(data / "events_2024-02.parquet", february, note=True)
    (data / "regions.csv").write_text(
        "region_id,area\n1,North\n2,North\n3,South\n4,South\n"
    )
    (dataset_dir / "dataset.yaml").write_text(
        """\
name: events
description: Synthetic hourly events
files: data/events_2024-*.parquet
table: events
time_column: ts
valid_time_range: [2024-01-01, 2024-03-01]
lookups:
  - {name: regions, file: data/regions.csv}
"""
    )
    settings = Settings(repo_root=root, datasets_dir=root / "datasets")
    return load_dataset("events", settings)
