from datetime import datetime

import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from analyst.data.dataset import load_dataset
from analyst.data.engine import Engine


@pytest.mark.taxi
def test_month_count_matches_pyarrow():
    dataset = load_dataset("nyc_taxi")
    timestamps = pq.read_table(
        dataset.root / "data/raw/yellow_tripdata_2024-03.parquet",
        columns=[dataset.time_column],
    )[dataset.time_column]
    start, end = datetime(2024, 3, 1), datetime(2024, 4, 1)
    expected = pc.sum(
        pc.cast(pc.and_(pc.greater_equal(timestamps, start), pc.less(timestamps, end)), "int64")
    ).as_py()
    # Restrict the engine to this file too: no scan of the full archive.
    dataset = dataset.model_copy(
        update={"files": "data/raw/yellow_tripdata_2024-03.parquet"}
    )
    actual = Engine(dataset).execute(
        "select count(*) from trips "
        "where tpep_pickup_datetime >= '2024-03-01' "
        "and tpep_pickup_datetime < '2024-04-01'"
    )[0][0].as_py()
    assert actual == expected
