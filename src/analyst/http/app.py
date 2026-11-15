"""Dashboard routes mounted on the LangGraph Agent Server (``http.app`` in langgraph.json).

A security boundary: path parameters reach the filesystem and ``/reveal``
reads uncut data. Every parameter is checked against a strict pattern before
any filesystem access; files are only opened through the result/chart stores
(which re-validate their ids) or :meth:`DatasetConfig.geometry_path`.
GET routes never create state: a segment must already exist on disk.

Errors are JSON ``{"error", "hint"}`` from these routes only. This app becomes
the whole server's app, so no app-level exception handlers or lifespan hooks
are registered here (they would apply to the Agent Server's own routes).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Callable, Coroutine
from datetime import date, datetime
from typing import Annotated, Any

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.routing import APIRoute

from analyst.charts.payload import DEFAULT_WIDTH, MAX_WIDTH, MIN_WIDTH, build_payload
from analyst.charts.reveal import reveal_actuals, uncut_engine
from analyst.charts.store import CHART_ID_RE, StoredChart
from analyst.config import get_settings
from analyst.data.dataset import DatasetConfig, load_dataset
from analyst.data.engine import QueryError
from analyst.data.registry import get_service
from analyst.data.service import DataService, segment_root
from analyst.errors import AnalystError, InvalidRequestError, UnknownNameError
from analyst.feedback import Feedback, append_feedback
from analyst.providers.catalog import load_catalog

logger = logging.getLogger(__name__)

DATASET_RE = re.compile(r"^[a-z0-9_]{1,40}$")
GEOMETRY_RE = re.compile(r"^[a-z0-9_]{1,40}$")
GEO_CACHE = "max-age=86400"
FULL = "full"


class HttpError(Exception):
    def __init__(self, status: int, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.hint = hint


def _error(status: int, message: str, hint: str | None = None) -> JSONResponse:
    return JSONResponse({"error": message, "hint": hint or None}, status_code=status)


def _validation_message(exc: RequestValidationError) -> str:
    # Field locations and messages only, never input values; an unknown field's
    # name is caller-supplied, so each location part is capped.
    parts = []
    for item in exc.errors()[:5]:
        loc = ".".join(
            str(p)[:40] for p in item.get("loc", ()) if p not in {"body", "query", "path"}
        )
        parts.append(f"{loc or 'request'}: {item.get('msg', 'invalid')}")
    return "invalid request: " + "; ".join(parts)


class _GuardedRoute(APIRoute):
    """Maps errors from these routes to clean JSON; never a traceback."""

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        handler = super().get_route_handler()

        async def guarded(request: Request) -> Response:
            try:
                return await handler(request)
            except HttpError as exc:
                return _error(exc.status, exc.message, exc.hint)
            except RequestValidationError as exc:
                return _error(400, _validation_message(exc))
            except UnknownNameError as exc:
                return _error(404, exc.message, exc.hint)
            except InvalidRequestError as exc:
                return _error(400, exc.message, exc.hint)
            except QueryError:
                logger.exception("query failed in %s", request.url.path)
                return _error(500, "the query behind this chart failed")
            except AnalystError as exc:
                return _error(400, exc.message, exc.hint)
            except Exception:
                logger.exception("unhandled error in %s", request.url.path)
                return _error(500, "internal error")

        return guarded


app = FastAPI(
    title="analyst dashboard routes",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
app.router.route_class = _GuardedRoute


# ------------------------------------------------------------ path params


def _dataset(name: str) -> DatasetConfig:
    """A dataset loadable from the datasets dir whose folder name is ``name``."""
    if not DATASET_RE.fullmatch(name):
        raise HttpError(404, "unknown dataset", "use a dataset name listed by /meta")
    try:
        dataset = load_dataset(name)
    except FileNotFoundError:
        raise HttpError(404, f"unknown dataset {name}", "use a dataset name listed by /meta") from None
    except Exception:
        logger.exception("dataset %s failed to load", name)
        raise HttpError(404, f"unknown dataset {name}", "use a dataset name listed by /meta") from None
    if dataset.name != name:
        raise HttpError(404, f"unknown dataset {name}", "use a dataset name listed by /meta")
    return dataset


def _cutoff(segment: str) -> date | None:
    if segment == FULL:
        return None
    try:
        cutoff = date.fromisoformat(segment)
    except ValueError:
        cutoff = None
    if cutoff is None or cutoff.isoformat() != segment:
        raise HttpError(404, "unknown segment", "use 'full' or a cutoff date YYYY-MM-DD")
    return cutoff


def _service(dataset: DatasetConfig, cutoff: date | None) -> DataService:
    # Only segments that already hold charts: a GET must never create one.
    if not (segment_root(dataset.name, cutoff) / "charts").is_dir():
        raise HttpError(404, "no charts in this segment", "use the segment a chart was made in")
    return get_service(dataset.name, cutoff)


def _chart(service: DataService, dataset: DatasetConfig, cutoff: date | None, chart_id: str) -> StoredChart:
    if not CHART_ID_RE.fullmatch(chart_id):
        raise HttpError(404, "unknown chart", "use a chart id returned by make_chart")
    stored = service.charts.get(chart_id)
    if stored.dataset != dataset.name or stored.cutoff != cutoff:
        raise HttpError(404, f"unknown chart {chart_id}", "use a chart id returned by make_chart")
    return stored


# ------------------------------------------------------------ meta


def _coverage(dataset: DatasetConfig) -> dict | None:
    try:
        cov = uncut_engine(dataset).coverage()
    except Exception:
        logger.exception("coverage failed for dataset %s", dataset.name)
        return None
    return {"start": cov.start.isoformat(), "end": cov.end.isoformat()}


def _meta() -> dict:
    settings = get_settings()
    datasets = []
    root = settings.datasets_dir
    children = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    for name in children:
        if not DATASET_RE.fullmatch(name):
            continue
        try:
            dataset = _dataset(name)
        except HttpError:
            continue
        datasets.append(
            {
                "name": dataset.name,
                "description": dataset.description,
                "coverage": _coverage(dataset),
                "geometries": [
                    {"name": g.name, "key_property": g.key_property, "dimension": g.dimension}
                    for g in dataset.geometries
                ],
            }
        )
    models = [
        {
            "spec": m.spec,
            "label": m.label,
            "context_window": m.context_window,
            "has_prices": m.has_prices,
            "reasoning_summary": m.reasoning_summary,
        }
        for m in load_catalog(settings)
    ]
    return {"datasets": datasets, "models": models, "default_model": settings.model}


@app.get("/meta")
async def meta() -> JSONResponse:
    return JSONResponse(await asyncio.to_thread(_meta))


# ------------------------------------------------------------ charts


def _payload(dataset_name: str, segment: str, chart_id: str, width: int, x_from, x_to) -> dict:
    dataset = _dataset(dataset_name)
    cutoff = _cutoff(segment)
    service = _service(dataset, cutoff)
    stored = _chart(service, dataset, cutoff, chart_id)
    return build_payload(
        stored,
        service.store,
        width=max(MIN_WIDTH, min(MAX_WIDTH, width)),
        x_from=x_from,
        x_to=x_to,
        geometries={g.name: g for g in dataset.geometries},
        geometry_dir=dataset.dir,
    )


@app.get("/charts/{dataset}/{segment}/{chart_id}")
async def chart(
    dataset: str,
    segment: str,
    chart_id: str,
    width: int = DEFAULT_WIDTH,
    x_from: Annotated[datetime | None, Query(alias="from")] = None,
    x_to: Annotated[datetime | None, Query(alias="to")] = None,
) -> JSONResponse:
    payload = await asyncio.to_thread(_payload, dataset, segment, chart_id, width, x_from, x_to)
    return JSONResponse(payload)


def _reveal(dataset_name: str, segment: str, chart_id: str) -> dict:
    dataset = _dataset(dataset_name)
    cutoff = _cutoff(segment)
    if cutoff is None:
        raise HttpError(404, "the full segment has no cutoff; there is nothing to reveal")
    service = _service(dataset, cutoff)
    stored = _chart(service, dataset, cutoff, chart_id)
    return reveal_actuals(stored, service, dataset)


@app.get("/charts/{dataset}/{segment}/{chart_id}/reveal")
async def reveal(dataset: str, segment: str, chart_id: str) -> JSONResponse:
    return JSONResponse(await asyncio.to_thread(_reveal, dataset, segment, chart_id))


# ------------------------------------------------------------ geometry


def _geometry(dataset_name: str, name: str) -> bytes:
    dataset = _dataset(dataset_name)
    if not GEOMETRY_RE.fullmatch(name) or dataset.geometry(name) is None:
        raise HttpError(404, "unknown geometry", "use a geometry listed by /meta")
    try:
        path = dataset.geometry_path(name)
    except (KeyError, ValueError):
        raise HttpError(404, f"unknown geometry {name}", "use a geometry listed by /meta") from None
    if not path.is_file():
        raise HttpError(404, f"geometry {name} has no file", "the dataset declares a file that is missing")
    try:
        content = path.read_bytes()
        parsed = json.loads(content)
    except (OSError, ValueError):
        logger.exception("geometry %s/%s is unreadable", dataset_name, name)
        raise HttpError(500, f"geometry {name} is not valid GeoJSON") from None
    if not isinstance(parsed, dict) or not isinstance(parsed.get("features"), list):
        raise HttpError(500, f"geometry {name} is not a GeoJSON FeatureCollection")
    return content


@app.get("/geo/{dataset}/{name}")
async def geometry(dataset: str, name: str) -> Response:
    content = await asyncio.to_thread(_geometry, dataset, name)
    return Response(content, media_type="application/geo+json", headers={"Cache-Control": GEO_CACHE})


# ------------------------------------------------------------ feedback


@app.post("/feedback", status_code=204)
async def feedback(item: Feedback) -> Response:
    await asyncio.to_thread(append_feedback, item)
    return Response(status_code=204)
