"""Custom HTTP routes served next to the Agent Server (meta, charts, geo, reveal, feedback)."""

import json
import os
import shutil
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from analyst.charts.reveal import clear_engine_cache
from analyst.charts.spec import (
    ForecastFanChart,
    LineChart,
    MapChart,
    SmallMultiplesChart,
)
from analyst.config import get_settings
from analyst.data.dataset import load_dataset
from analyst.data.registry import clear_services, get_service
from analyst.semantic.compiler import MetricQuery

CUT = date(2024, 2, 23)
FC_DAYS = [datetime(2024, 2, 24) + timedelta(days=i) for i in range(8)]  # .. Mar 2

FC = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"area": "North"},
            "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
        }
    ],
}

GEOMETRIES = """\
geometries:
  - {name: areas, file: geo/areas.geojson, key_property: area, dimension: area}
  - {name: missing, file: geo/missing.geojson, key_property: area}
  - {name: broken, file: geo/broken.geojson, key_property: area}
  - {name: escape, file: geo/link.geojson, key_property: area}
"""


def _clear():
    get_settings.cache_clear()
    clear_services()
    clear_engine_cache()


@pytest.fixture(autouse=True)
def _isolated_caches():
    _clear()
    yield
    _clear()


def _forecast_table(ids: list[str], *, lo_80=0.0, hi_80=1e12, lo_95=-2.0, hi_95=-1.0):
    rows = []
    for uid in ids:
        for ds in FC_DAYS:
            rows.append(
                {
                    "unique_id": uid,
                    "ds": ds,
                    "yhat": 1000.0,
                    "lo_80": lo_80,
                    "hi_80": hi_80,
                    "lo_95": lo_95,
                    "hi_95": hi_95,
                    "baseline": 900.0,
                }
            )
    return pa.Table.from_pylist(rows)


def _save_forecast(svc, input_card, *, id_col, ids, value_col="total_amount"):
    return svc.store.save(
        _forecast_table(ids),
        produced_by="forecast",
        source=f"fake forecast {input_card.handle_id}",
        governed=input_card.governed,
        elapsed_s=0.0,
        meta={
            "time_col": "period",
            "value_col": value_col,
            "id_col": id_col,
            "freq": "D",
            "horizon": len(FC_DAYS),
            "origin": "2024-02-23T00:00:00",
            "input_handle": input_card.handle_id,
            "headline_ids": ids,
            "model": "AutoETS",
            "cutoff": CUT.isoformat(),
        },
    )


@pytest.fixture
def env(synthetic_dataset, tmp_path, monkeypatch):
    root = tmp_path / "repo"
    shutil.copytree(synthetic_dataset.root / "data", root / "data")
    ds_dir = root / "datasets" / "events"
    shutil.copytree(synthetic_dataset.dir, ds_dir)
    (ds_dir / "dataset.yaml").write_text((ds_dir / "dataset.yaml").read_text() + GEOMETRIES)
    geo = ds_dir / "geo"
    geo.mkdir()
    (geo / "areas.geojson").write_text(json.dumps(FC))
    (geo / "broken.geojson").write_text("{not json")
    secret = tmp_path / "secret.geojson"
    secret.write_text(json.dumps(FC))
    (geo / "link.geojson").symlink_to(secret)
    # Junk next to the real dataset: never listed, never loaded.
    (root / "datasets" / "Bad-Name").mkdir()
    (root / "datasets" / "broken").mkdir()
    (root / "datasets" / "broken" / "dataset.yaml").write_text("- not a mapping\n")
    artifacts = tmp_path / "artifacts"
    monkeypatch.setenv("ANALYST_REPO_ROOT", str(root))
    monkeypatch.setenv("ANALYST_DATASETS_DIR", str(root / "datasets"))
    monkeypatch.setenv("ANALYST_ARTIFACTS_DIR", str(artifacts))
    _clear()

    svc = get_service("events", CUT)
    by_area = svc.query_metrics(
        MetricQuery(metrics=["total_amount"], dimensions=["area"], grain="day")
    )
    hourly = svc.query_metrics(MetricQuery(metrics=["total_amount"], grain="hour"))
    area_totals = svc.query_metrics(MetricQuery(metrics=["total_amount"], dimensions=["area"]))
    sql_card = svc.run_sql(
        "SELECT CAST(ts AS DATE) AS period, sum(amount) AS total_amount FROM events GROUP BY 1"
    )
    py_card = svc.store.save(
        pa.table({"period": [date(2024, 2, 23)], "total_amount": [1.0]}),
        produced_by="run_python",
        source="print('x')",
        governed=False,
        elapsed_s=0.0,
    )
    fc_area = _save_forecast(svc, by_area, id_col="area", ids=["North", "South"])
    fc_sql = _save_forecast(svc, sql_card, id_col=None, ids=["series"])
    fc_py = _save_forecast(svc, py_card, id_col=None, ids=["series"])

    def chart(spec):
        return svc.charts.save(spec, dataset="events", cutoff=CUT).chart_id

    charts = SimpleNamespace(
        fan=chart(ForecastFanChart(title="fan", forecast_handle=fc_area.handle_id)),
        multiples=chart(
            SmallMultiplesChart(
                title="sm",
                facet="unique_id",
                inner={"kind": "forecast_fan", "forecast_handle": fc_area.handle_id},
            )
        ),
        sql_fan=chart(ForecastFanChart(title="sql fan", forecast_handle=fc_sql.handle_id)),
        py_fan=chart(ForecastFanChart(title="py fan", forecast_handle=fc_py.handle_id)),
        line=chart(LineChart(title="hourly", handle=hourly.handle_id, x="period", y="total_amount")),
        map=chart(
            MapChart(
                title="map", handle=area_totals.handle_id, key="area", value="total_amount",
                geometry="areas",
            )
        ),
    )
    full = get_service("events", None)
    full_line = full.charts.save(
        LineChart(
            title="full hourly",
            handle=full.query_metrics(MetricQuery(metrics=["total_amount"], grain="hour")).handle_id,
            x="period",
            y="total_amount",
        ),
        dataset="events",
        cutoff=None,
    ).chart_id
    _clear()

    from analyst.http.app import app

    return SimpleNamespace(
        client=TestClient(app),
        root=root,
        artifacts=artifacts,
        charts=charts,
        full_line=full_line,
        segment=CUT.isoformat(),
    )


def _error(response, status):
    assert response.status_code == status, response.text
    body = response.json()
    assert set(body) == {"error", "hint"}
    assert isinstance(body["error"], str) and body["error"]
    assert "Traceback" not in response.text
    return body


def test_meta_lists_dataset_and_coverage(env):
    response = env.client.get("/meta")
    assert response.status_code == 200
    body = response.json()
    assert [d["name"] for d in body["datasets"]] == ["events"]
    events = body["datasets"][0]
    assert events["description"] == "Synthetic hourly events"
    # Coverage is the uncut data: the valid range clips the 2001 rows.
    assert events["coverage"] == {"start": "2024-01-01T00:00:00", "end": "2024-02-29T23:00:00"}
    assert [g["name"] for g in events["geometries"]] == ["areas", "missing", "broken", "escape"]
    assert events["geometries"][0] == {"name": "areas", "key_property": "area", "dimension": "area"}
    assert body["models"] == []
    assert isinstance(body["default_model"], str)


def test_meta_does_not_create_segments(env):
    shutil.rmtree(env.artifacts / "events" / "full")
    assert env.client.get("/meta").status_code == 200
    assert not (env.artifacts / "events" / "full").exists()


def test_chart_route_returns_payload(env):
    response = env.client.get(f"/charts/events/{env.segment}/{env.charts.fan}")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["chart_id"] == env.charts.fan
    assert body["kind"] == "forecast_fan"
    assert body["cutoff"] == env.segment
    assert body["data"]["series_id"] == "North"
    assert len(body["data"]["forecast"]["x"]) == len(FC_DAYS)

    response = env.client.get(f"/charts/events/{env.segment}/{env.charts.map}")
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["geometry"] == "areas"
    assert data["unmatched"] == ["South"]

    full = env.client.get(f"/charts/events/full/{env.full_line}")
    assert full.status_code == 200
    assert full.json()["cutoff"] is None


def test_chart_route_from_to(env):
    response = env.client.get(
        f"/charts/events/{env.segment}/{env.charts.line}",
        params={"from": "2024-02-01T00:00:00", "to": "2024-02-02T00:00:00"},
    )
    assert response.status_code == 200, response.text
    xs = response.json()["data"]["series"][0]["x"]
    assert xs and all("2024-02-01" <= x[:10] <= "2024-02-02" for x in xs)
    # A browser's toISOString() is UTC with a Z; the payload compares it as naive UTC.
    aware = env.client.get(
        f"/charts/events/{env.segment}/{env.charts.line}",
        params={"from": "2024-02-01T00:00:00.000Z", "to": "2024-02-02T00:00:00.000Z"},
    )
    assert aware.status_code == 200, aware.text
    assert aware.json()["data"] == response.json()["data"]
    _error(
        env.client.get(f"/charts/events/{env.segment}/{env.charts.line}", params={"from": "soon"}),
        400,
    )


@pytest.mark.parametrize(
    ("dataset", "segment", "chart"),
    [
        ("nope", "full", "c_0000000000"),
        ("Events", "full", "c_0000000000"),
        ("broken", "full", "c_0000000000"),
        ("events", "2026-13-01", "c_0000000000"),
        ("events", "20240223", "c_0000000000"),
        ("events", "2024-2-23", "c_0000000000"),
        ("events", "latest", "c_0000000000"),
        ("events", "2024-02-22", "c_0000000000"),
        ("events", "full", "c_XYZ"),
        ("events", "full", "..%2F..%2Fsecret"),
    ],
)
def test_chart_route_rejects_bad_paths(env, dataset, segment, chart):
    response = env.client.get(f"/charts/{dataset}/{segment}/{chart}")
    assert response.status_code == 404, response.text
    assert "Traceback" not in response.text


def test_chart_route_rejects_cross_segment_and_bad_ids(env):
    # A valid chart id that lives in the other segment.
    _error(env.client.get(f"/charts/events/full/{env.charts.fan}"), 404)
    _error(env.client.get(f"/charts/events/{env.segment}/{env.full_line}"), 404)
    _error(env.client.get(f"/charts/events/{env.segment}/c_0123456789"), 404)
    _error(env.client.get("/charts/nope/full/c_0123456789"), 404)
    _error(env.client.get("/charts/events/2026-13-01/c_0123456789"), 404)
    # Path segments with a slash in them never reach the handler (router 404).
    assert env.client.get("/charts/events/..%2Fx/c_0123456789").status_code == 404
    assert env.client.get("/charts/events/../x/c_0123456789").status_code == 404
    # An unknown valid-looking date never creates a segment directory.
    env.client.get("/charts/events/2024-02-22/c_0123456789")
    assert not (env.artifacts / "events" / "2024-02-22").exists()


def test_width_clamped(env):
    url = f"/charts/events/{env.segment}/{env.charts.line}"

    def data(width):
        response = env.client.get(url, params={"width": width})
        assert response.status_code == 200, response.text
        return response.json()

    tiny, smallest = data(1), data(100)
    assert tiny["data"] == smallest["data"]
    assert tiny["downsampled"]["to"] <= 4 * 100 < tiny["downsampled"]["from"]
    huge, widest = data(10**9), data(4000)
    assert huge["data"] == widest["data"]
    assert huge["downsampled"] is None
    _error(env.client.get(url, params={"width": "wide"}), 400)


def test_geo_route_serves_declared_geometry_only(env):
    response = env.client.get("/geo/events/areas")
    assert response.status_code == 200, response.text
    assert response.json() == FC
    assert response.headers["cache-control"] == "max-age=86400"
    assert "geo+json" in response.headers["content-type"]

    for name in ("nope", "AREAS", "missing", "escape", "a" * 41):
        _error(env.client.get(f"/geo/events/{name}"), 404)
    assert env.client.get("/geo/events/..%2Fsecret").status_code == 404
    _error(env.client.get("/geo/nope/areas"), 404)
    _error(env.client.get("/geo/broken/areas"), 404)
    body = _error(env.client.get("/geo/events/broken"), 500)
    assert "broken" in body["error"]


def _snapshot(root: Path) -> dict:
    out = {}
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            path = Path(dirpath) / name
            stat = path.stat()
            out[str(path)] = (stat.st_size, stat.st_mtime_ns)
    return out


def test_reveal_writes_nothing_and_needs_past_cutoff(env):
    shutil.rmtree(env.artifacts / "events" / "full")
    before = _snapshot(env.artifacts)
    for chart in (env.charts.fan, env.charts.multiples, env.charts.sql_fan):
        response = env.client.get(f"/charts/events/{env.segment}/{chart}/reveal")
        assert response.status_code == 200, response.text
    assert _snapshot(env.artifacts) == before
    assert not (env.artifacts / "events" / "full").exists()
    assert sorted(os.listdir(env.artifacts / "events")) == [env.segment]

    # No cutoff, nothing to reveal.
    _error(env.client.get(f"/charts/events/full/{env.full_line}/reveal"), 404)
    _error(env.client.get(f"/charts/events/full/{env.charts.fan}/reveal"), 404)
    # Not a forecast chart.
    _error(env.client.get(f"/charts/events/{env.segment}/{env.charts.line}/reveal"), 404)
    # Arbitrary code produced the history: no governed way to re-run it.
    _error(env.client.get(f"/charts/events/{env.segment}/{env.charts.py_fan}/reveal"), 404)
    _error(env.client.get(f"/charts/events/{env.segment}/c_0123456789/reveal"), 404)
    _error(env.client.get(f"/charts/events/2024-02-22/{env.charts.fan}/reveal"), 404)
    assert _snapshot(env.artifacts) == before


def _expected_daily(root: Path) -> pd.DataFrame:
    frames = [pq.read_table(p).select(["ts", "region_id", "amount"]).to_pandas()
              for p in sorted((root / "data").glob("events_2024-*.parquet"))]
    df = pd.concat(frames)
    df = df.loc[(df.ts >= "2024-01-01") & (df.ts < "2024-03-01")]
    df["area"] = df.region_id.map({1: "North", 2: "North", 3: "South", 4: "South"})
    df["day"] = df.ts.dt.normalize()
    return df


def test_reveal_returns_actuals_after_cutoff(env):
    raw = _expected_daily(env.root)
    known = [d for d in FC_DAYS if d < datetime(2024, 3, 1)]
    xs = [d.isoformat() for d in FC_DAYS]

    fan = env.client.get(f"/charts/events/{env.segment}/{env.charts.fan}/reveal").json()
    assert [s["series_id"] for s in fan["series"]] == ["North"]
    north = fan["series"][0]
    assert north["x"] == xs
    by_day = raw.loc[raw.area == "North"].groupby("day").amount.sum()
    assert north["actual"][: len(known)] == pytest.approx([by_day[d] for d in known])
    # Past the end of the data there is nothing to reveal.
    assert north["actual"][len(known):] == [None, None]
    assert north["yhat"] == [1000.0] * len(FC_DAYS)
    assert north["inside_80"] == [True] * len(known) + [None, None]
    assert north["inside_95"] == [False] * len(known) + [None, None]

    multiples = env.client.get(
        f"/charts/events/{env.segment}/{env.charts.multiples}/reveal"
    ).json()
    assert sorted(s["series_id"] for s in multiples["series"]) == ["North", "South"]
    south = next(s for s in multiples["series"] if s["series_id"] == "South")
    by_day = raw.loc[raw.area == "South"].groupby("day").amount.sum()
    assert south["actual"][: len(known)] == pytest.approx([by_day[d] for d in known])

    sql = env.client.get(f"/charts/events/{env.segment}/{env.charts.sql_fan}/reveal").json()
    (series,) = sql["series"]
    assert series["series_id"] == "series"
    by_day = raw.groupby("day").amount.sum()
    assert series["actual"][: len(known)] == pytest.approx([by_day[d] for d in known])
    assert series["actual"][len(known):] == [None, None]
    # Nothing before the forecast horizon leaks into the response.
    assert series["x"] == xs


def test_feedback_appends_line_and_validates(env):
    path = env.artifacts / "feedback" / "feedback.jsonl"
    good = {"thread_id": "t1", "run_id": "r1", "rating": "up", "correction": None}
    response = env.client.post("/feedback", json=good)
    assert response.status_code == 204, response.text
    assert response.content == b""
    response = env.client.post(
        "/feedback", json={"thread_id": "t1", "run_id": "r2", "rating": "down", "correction": "x" * 2000}
    )
    assert response.status_code == 204, response.text

    text = path.read_text()
    assert text.endswith("\n")
    lines = [json.loads(line) for line in text.splitlines()]
    assert len(lines) == 2
    assert lines[0]["thread_id"] == "t1" and lines[0]["rating"] == "up"
    assert lines[0]["user_id"] == "local"
    assert datetime.fromisoformat(lines[0]["at"]).tzinfo is not None
    assert datetime.fromisoformat(lines[0]["at"]) <= datetime.now(UTC)
    assert lines[1]["correction"] == "x" * 2000

    bad = [
        {**good, "rating": "meh"},
        {**good, "extra": 1},
        {**good, "user_id": "someone"},
        {**good, "correction": "x" * 2001},
        {**good, "thread_id": "t" * 129},
        {**good, "run_id": "r" * 129},
        {"thread_id": "t1", "rating": "up"},
    ]
    for body in bad:
        _error(env.client.post("/feedback", json=body), 400)
    _error(env.client.post("/feedback", content=b"not json", headers={"content-type": "application/json"}), 400)
    assert len(path.read_text().splitlines()) == 2


def test_dataset_must_match_its_folder(env):
    ds_dir = env.root / "datasets" / "events"
    shutil.copytree(ds_dir, env.root / "datasets" / "alias")
    assert load_dataset("alias", get_settings()).name == "events"
    _error(env.client.get("/geo/alias/areas"), 404)
    _error(env.client.get(f"/charts/alias/{env.segment}/{env.charts.fan}"), 404)
    names = [d["name"] for d in env.client.get("/meta").json()["datasets"]]
    assert names == ["events"]


def test_map_chart_with_unreadable_geometry_is_a_clean_error(env):
    svc = get_service("events", CUT)
    handle = svc.query_metrics(MetricQuery(metrics=["total_amount"], dimensions=["area"]))
    for name in ("broken", "missing", "escape"):
        chart_id = svc.charts.save(
            MapChart(title=name, handle=handle.handle_id, key="area", value="total_amount", geometry=name),
            dataset="events",
            cutoff=CUT,
        ).chart_id
        _error(env.client.get(f"/charts/events/{env.segment}/{chart_id}"), 400)


@pytest.mark.parametrize(
    ("freq", "periods", "data_end", "kept"),
    [
        # Daily: Feb 29 is complete once data reaches 23:00; Mar 1 at 11:00 is partial.
        ("D", ["2024-02-29", "2024-03-01"], "2024-03-01 11:00", ["2024-02-29"]),
        ("D", ["2024-02-29", "2024-03-01"], "2024-03-01 23:00", ["2024-02-29", "2024-03-01"]),
        # Monthly: one calendar month per period; mid-March data leaves March partial.
        ("MS", ["2024-02-01", "2024-03-01"], "2024-03-15 00:00", ["2024-02-01"]),
        ("MS", ["2024-02-01", "2024-03-01"], "2024-03-31 23:00", ["2024-02-01", "2024-03-01"]),
    ],
)
def test_reveal_drops_partial_periods(freq, periods, data_end, kept):
    from analyst.charts.reveal import _actuals, _complete_until
    from analyst.data.engine import Coverage

    stamps = [pd.Timestamp(p) for p in periods]
    frame = pd.DataFrame({"period": [s.date() for s in stamps], "v": [1.0, float("nan")]})
    until = _complete_until(Coverage(datetime(2024, 1, 1), pd.Timestamp(data_end).to_pydatetime()))
    out = _actuals(
        frame, time_col="period", value_col="v", id_col=None,
        horizon=set(stamps), complete_until=until, freq=freq,
    )
    assert sorted(ts.strftime("%Y-%m-%d") for _sid, ts in out) == kept
    assert out[("series", stamps[0])] == 1.0
    if len(kept) == 2:
        assert out[("series", stamps[1])] is None  # NaN becomes null
