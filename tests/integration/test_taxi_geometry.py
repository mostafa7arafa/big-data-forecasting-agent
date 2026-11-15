import csv
import json

import pytest

from analyst.data.dataset import load_dataset


@pytest.mark.taxi
def test_taxi_zones_geojson_covers_zone_names():
    ds = load_dataset("nyc_taxi")
    spec = ds.geometry("taxi_zones")
    data = json.loads(ds.geometry_path("taxi_zones").read_text())
    names = {f["properties"][spec.key_property] for f in data["features"]}
    with (ds.root / "data" / "raw" / "taxi_zone_lookup.csv").open() as fh:
        zones = {r["Zone"] for r in csv.DictReader(fh)} - {"N/A", "Outside of NYC"}
    assert zones - names == set()
    assert data["metadata"]["source_url"].startswith("https://")
    assert ds.geometry_path("taxi_zones").stat().st_size <= 1_500_000
