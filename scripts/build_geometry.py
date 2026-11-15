"""Build the taxi-zone GeoJSON from the TLC shapefile zip (EPSG:2263 -> EPSG:4326).

The zip is read as data only (members are parsed by pyshp); nothing in it is executed.

    uv run python scripts/build_geometry.py --src <taxi_zones.zip> --out datasets/nyc_taxi/geo/taxi_zones.geojson
"""

from __future__ import annotations

import argparse
import io
import itertools
import json
import math
import zipfile
from pathlib import Path

import shapefile
from pyproj import Transformer

SOURCE_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
DOWNLOAD_DATE = "2026-11-15"
DECIMALS = 5
MAX_BYTES = 1_500_000
# Tolerance in degrees (~1e-4 deg is roughly 9 m at NYC latitude); doubled until the size fits.
START_TOLERANCE = 0.00005


def _dp(points: list[tuple[float, float]], tol: float) -> list[tuple[float, float]]:
    """Iterative Douglas-Peucker on an open polyline; keeps both endpoints."""
    n = len(points)
    if n < 3:
        return list(points)
    keep = [False] * n
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        (x1, y1), (x2, y2) = points[i], points[j]
        dx, dy = x2 - x1, y2 - y1
        norm = math.hypot(dx, dy)
        best, idx = -1.0, -1
        for k in range(i + 1, j):
            px, py = points[k]
            d = (
                math.hypot(px - x1, py - y1)
                if norm == 0
                else abs(dy * (px - x1) - dx * (py - y1)) / norm
            )
            if d > best:
                best, idx = d, k
        if best > tol:
            keep[idx] = True
            stack += [(i, idx), (idx, j)]
    return [p for p, k in zip(points, keep, strict=True) if k]


def _ring(points: list[tuple[float, float]], tol: float) -> list[list[float]]:
    """Simplify a closed ring, never below 4 points (a closed triangle)."""
    open_pts = points[:-1] if points[0] == points[-1] else list(points)
    simple = _dp(open_pts + [open_pts[0]], tol)[:-1]
    if len(simple) < 3:
        # collapsed: fall back to the first, middle and last vertex of the original
        simple = [open_pts[0], open_pts[len(open_pts) // 2], open_pts[-1]]
        if len({tuple(p) for p in simple}) < 3:
            simple = open_pts[:3]
    rounded = [[round(x, DECIMALS), round(y, DECIMALS)] for x, y in simple]
    rounded.append(rounded[0])
    return rounded


def _geometry(shape: shapefile.Shape, tf: Transformer, tol: float) -> dict:
    pts = [tf.transform(x, y) for x, y in shape.points]
    starts = [*shape.parts, len(pts)]
    rings = [pts[a:b] for a, b in itertools.pairwise(starts)]
    # Shapefile outer rings are clockwise, holes counter-clockwise (signed area test).
    polys: list[list[list[list[float]]]] = []
    for ring in rings:
        area = sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1], strict=True)
        )
        simplified = _ring(ring, tol)
        if area <= 0 or not polys:  # clockwise (negative area in x/y) = outer ring
            polys.append([simplified])
        else:
            polys[-1].append(simplified)
    if len(polys) == 1:
        return {"type": "Polygon", "coordinates": polys[0]}
    return {"type": "MultiPolygon", "coordinates": polys}


def build(src: Path, tol: float) -> dict:
    with zipfile.ZipFile(src) as zf:
        members = {Path(n).suffix: n for n in zf.namelist() if not n.endswith("/")}
        reader = shapefile.Reader(
            shp=io.BytesIO(zf.read(members[".shp"])),
            shx=io.BytesIO(zf.read(members[".shx"])),
            dbf=io.BytesIO(zf.read(members[".dbf"])),
        )
    tf = Transformer.from_crs("EPSG:2263", "EPSG:4326", always_xy=True)
    features = []
    for sr in reader.iterShapeRecords():
        rec = sr.record.as_dict()
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "location_id": int(rec["LocationID"]),
                    "zone": rec["zone"],
                    "borough": rec["borough"],
                },
                "geometry": _geometry(sr.shape, tf, tol),
            }
        )
    return {
        "type": "FeatureCollection",
        "metadata": {
            "source_url": SOURCE_URL,
            "downloaded": DOWNLOAD_DATE,
            "crs_conversion": "EPSG:2263 -> EPSG:4326 (pyproj, always_xy)",
            "simplification": f"Douglas-Peucker, tolerance {tol} degrees, "
            f"coordinates rounded to {DECIMALS} decimals",
        },
        "features": features,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    tol = START_TOLERANCE
    while True:
        text = json.dumps(build(args.src, tol), separators=(",", ":"), allow_nan=False)
        if len(text.encode()) <= MAX_BYTES:
            break
        tol *= 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(text)} bytes, tolerance {tol})")


if __name__ == "__main__":
    main()
