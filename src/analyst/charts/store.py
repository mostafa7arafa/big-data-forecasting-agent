from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from datetime import UTC, date, datetime
from pathlib import Path

from pydantic import BaseModel

from analyst.charts.spec import ChartSpec
from analyst.errors import UnknownNameError

CHART_ID_RE = re.compile(r"^c_[0-9a-f]{10}$")


class StoredChart(BaseModel):
    chart_id: str
    dataset: str
    cutoff: date | None
    spec: ChartSpec
    created_at: datetime


class ChartStore:
    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._charts_dir = self._root / "charts"
        self._charts_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        spec: ChartSpec,
        *,
        dataset: str,
        cutoff: date | None,
    ) -> StoredChart:
        chart_id = _chart_id(spec, dataset=dataset, cutoff=cutoff)
        path = self._charts_dir / f"{chart_id}.json"
        if path.is_file():
            return self.get(chart_id)
        created_at = datetime.now(UTC)
        stored = StoredChart(
            chart_id=chart_id,
            dataset=dataset,
            cutoff=cutoff,
            spec=spec,
            created_at=created_at,
        )
        tag = uuid.uuid4().hex
        tmp = self._charts_dir / f"{chart_id}.json.{tag}.tmp"
        try:
            tmp.write_text(
                json.dumps(stored.model_dump(mode="json"), indent=2),
                encoding="utf-8",
            )
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
        return stored

    def get(self, chart_id: str) -> StoredChart:
        self._validate_chart_id(chart_id)
        path = self._charts_dir / f"{chart_id}.json"
        if not path.is_file():
            raise _unknown_chart(chart_id)
        data = json.loads(path.read_text(encoding="utf-8"))
        return StoredChart.model_validate(data)

    def exists(self, chart_id: str) -> bool:
        self._validate_chart_id(chart_id)
        return (self._charts_dir / f"{chart_id}.json").is_file()

    def _validate_chart_id(self, chart_id: str) -> None:
        if not CHART_ID_RE.fullmatch(chart_id):
            raise _unknown_chart(chart_id)


def _chart_id(spec: ChartSpec, *, dataset: str, cutoff: date | None) -> str:
    spec_json = json.dumps(spec.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    cutoff_part = cutoff.isoformat() if cutoff is not None else ""
    digest = hashlib.sha256((spec_json + dataset + cutoff_part).encode()).hexdigest()[:10]
    return f"c_{digest}"


def _unknown_chart(chart_id: str) -> UnknownNameError:
    return UnknownNameError(
        f"unknown chart {chart_id}",
        hint="use a chart id returned by make_chart",
    )