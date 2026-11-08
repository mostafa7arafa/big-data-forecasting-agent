from datetime import date

import pytest

from analyst.calibration.profile import profile_dataset
from analyst.data.dataset import load_dataset


@pytest.mark.taxi
@pytest.mark.slow
def test_taxi_profile():
    dataset = load_dataset("nyc_taxi")
    profile = profile_dataset(dataset)
    assert any(
        start <= date(2023, 9, 21) and end >= date(2023, 9, 24)
        for start, end in profile.daily.gaps
    )
    assert profile.implausible_timestamps == 176
    by_name = {column.name: column for column in profile.columns}
    cbd = by_name["cbd_congestion_fee"]
    assert cbd.first_non_null_month is not None
    assert cbd.first_material_month == "2025-01"
    assert profile.elapsed_s < 60
