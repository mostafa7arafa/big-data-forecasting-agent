"""Thread-safe cache of DataService instances, one per (dataset, cutoff)."""

from __future__ import annotations

import threading
from datetime import date

from analyst.data.dataset import load_dataset
from analyst.data.service import DataService

_services: dict[tuple[str, date | None], DataService] = {}
_lock = threading.Lock()


def get_service(dataset: str, cutoff: date | None) -> DataService:
    key = (dataset, cutoff)
    with _lock:
        service = _services.get(key)
        if service is None:
            service = DataService(load_dataset(dataset), cutoff=cutoff)
            _services[key] = service
        return service


def clear_services() -> None:
    with _lock:
        _services.clear()
