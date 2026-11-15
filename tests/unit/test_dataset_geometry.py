import json
from datetime import UTC, date, datetime

import pyarrow as pa
import pytest
from pydantic import ValidationError

from analyst.charts.payload import build_payload
from analyst.charts.spec import MapChart
from analyst.charts.store import StoredChart
from analyst.config import Settings
from analyst.data.dataset import GeometrySpec, load_dataset
from analyst.data.results import ResultStore
from analyst.errors import InvalidRequestError

YAML = """\
name: geo
description: d
files: data/*.parquet
table: t
time_column: ts
valid_time_range: [2024-01-01, 2024-02-01]
lookups: []
geometries:
  - {{name: areas, file: {file}, key_property: area, dimension: area_name}}
"""

FC = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"area": "A"},
            "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
        },
        {
            "type": "Feature",
            "properties": {"area": "B"},
            "geometry": {"type": "Polygon", "coordinates": [[[2, 0], [3, 0], [3, 1], [2, 0]]]},
        },
    ],
}


def _make(tmp_path, file="geo/areas.geojson"):
    d = tmp_path / "datasets" / "geo"
    (d / "geo").mkdir(parents=True)
    (d / "geo" / "areas.geojson").write_text(json.dumps(FC))
    (d / "dataset.yaml").write_text(YAML.format(file=file))
    return Settings(repo_root=tmp_path, datasets_dir=tmp_path / "datasets")


def test_geometry_declared_and_resolved(tmp_path):
    ds = load_dataset("geo", _make(tmp_path))
    spec = ds.geometry("areas")
    assert spec == GeometrySpec(
        name="areas", file="geo/areas.geojson", key_property="area", dimension="area_name"
    )
    assert ds.geometry("nope") is None
    assert ds.geometry_path("areas") == (ds.dir / "geo" / "areas.geojson").resolve()
    assert load_dataset("nyc_taxi").geometry("taxi_zones").dimension == "pickup_zone_name"


@pytest.mark.parametrize("bad", ["../outside.geojson", "/etc/passwd", "geo/../../x.json"])
def test_geometry_path_cannot_escape_dataset_dir(tmp_path, bad):
    settings = _make(tmp_path, file=bad)
    with pytest.raises(ValidationError):
        load_dataset("geo", settings)


def test_geometry_symlink_cannot_escape_dataset_dir(tmp_path):
    settings = _make(tmp_path)
    ds_dir = tmp_path / "datasets" / "geo"
    outside = tmp_path / "secret.geojson"
    outside.write_text("{}")
    (ds_dir / "geo" / "link.geojson").symlink_to(outside)
    (ds_dir / "dataset.yaml").write_text(YAML.format(file="geo/link.geojson"))
    ds = load_dataset("geo", settings)
    with pytest.raises(ValueError, match="outside"):
        ds.geometry_path("areas")


def test_map_payload_matches_keys_and_reports_unmatched(tmp_path):
    ds = load_dataset("geo", _make(tmp_path))
    store = ResultStore(tmp_path / "r")
    h = store.save(
        pa.table({"k": ["A", "Z", "Y"], "v": [1.0, 2.0, 3.0]}),
        produced_by="query", source="s", governed=True, elapsed_s=0.1, meta={},
    ).handle_id
    stored = StoredChart(
        chart_id="c_0123456789", dataset="geo", cutoff=date(2024, 1, 31),
        spec=MapChart(title="m", handle=h, key="k", value="v", geometry="areas"),
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    p = build_payload(
        stored, store, geometries={g.name: g for g in ds.geometries}, geometry_dir=ds.dir
    )
    d = p["data"]
    assert d["geometry"] == "areas" and d["key_property"] == "area"
    assert d["keys"] == ["A", "Z", "Y"] and d["values"] == [1.0, 2.0, 3.0]
    assert d["unmatched"] == ["Y", "Z"]
    with pytest.raises(InvalidRequestError):
        build_payload(stored, store, geometries={}, geometry_dir=ds.dir)


def test_map_payload_matches_numeric_key_property(tmp_path):
    settings = _make(tmp_path)
    ds_dir = tmp_path / "datasets" / "geo"
    fc = json.loads(json.dumps(FC))
    for i, f in enumerate(fc["features"], start=1):
        f["properties"] = {"area": i}
    (ds_dir / "geo" / "areas.geojson").write_text(json.dumps(fc))
    ds = load_dataset("geo", settings)
    store = ResultStore(tmp_path / "r")
    h = store.save(
        pa.table({"k": [1.0, 2.0, 3.0], "v": [1.0, 2.0, 3.0]}),
        produced_by="query", source="s", governed=True, elapsed_s=0.1, meta={},
    ).handle_id
    stored = StoredChart(
        chart_id="c_0123456789", dataset="geo", cutoff=date(2024, 1, 31),
        spec=MapChart(title="m", handle=h, key="k", value="v", geometry="areas"),
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    d = build_payload(
        stored, store, geometries={g.name: g for g in ds.geometries}, geometry_dir=ds.dir
    )["data"]
    assert d["keys"] == ["1", "2", "3"]
    assert d["unmatched"] == ["3"]


@pytest.mark.parametrize("bad", ["Areas", "a-b", "", "x" * 41, "../x"])
def test_geometry_name_pattern_rejected(bad):
    with pytest.raises(ValidationError):
        GeometrySpec(name=bad, file="geo/a.geojson", key_property="k")
