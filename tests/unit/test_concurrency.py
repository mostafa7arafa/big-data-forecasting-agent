import os
import threading
import time
from datetime import date

import pyarrow as pa
import pytest

from analyst.config import get_settings
from analyst.data.gate import QueryGate, get_gate
from analyst.data.registry import clear_services, get_service
from analyst.data.results import ResultStore
from analyst.data.service import DataService


@pytest.fixture
def synthetic_service(synthetic_dataset, tmp_path):
    return DataService(synthetic_dataset, store_root=tmp_path)


def _run_threads(fn, n):
    results, errors = [None] * n, []

    def work(i):
        try:
            results[i] = fn()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    return results


def test_gate_caps_concurrency():
    gate = QueryGate(2)

    def hold():
        with gate.slot():
            time.sleep(0.05)

    _run_threads(hold, 8)
    assert gate.peak == 2
    assert gate.in_use == 0


def test_engine_queries_go_through_gate(synthetic_service):
    gate = get_gate()
    gate.reset_peak()
    cards = _run_threads(
        lambda: synthetic_service.run_sql("SELECT count(*) AS n FROM events"), 6
    )
    assert 1 <= gate.peak <= get_settings().max_concurrent_queries
    assert len({c.preview for c in cards}) == 1


def test_save_is_atomic(tmp_path, monkeypatch):
    store = ResultStore(tmp_path)
    real_replace = os.replace

    def flaky(src, dst):
        if str(dst).endswith(".card.json"):
            raise OSError("boom")
        return real_replace(src, dst)

    monkeypatch.setattr("analyst.data.results.os.replace", flaky)
    table = pa.table({"x": [1, 2]})
    with pytest.raises(OSError):
        store.save(table, produced_by="p", source="s", governed=True, elapsed_s=0.0)
    handles = {p.name.split(".")[0] for p in tmp_path.glob("r_*")}
    assert all(not store.exists(h) for h in handles)
    assert not list(tmp_path.rglob("*.tmp"))


def test_lookup_finds_saved_by_source(tmp_path):
    store = ResultStore(tmp_path)
    table = pa.table({"x": [1, 2]})
    card = store.save(
        table, produced_by="forecast", source="S", governed=True, elapsed_s=0.0
    )
    assert store.exists(card.handle_id)
    assert store.lookup("forecast", "S").handle_id == card.handle_id
    assert store.lookup("forecast", "other") is None


def test_registry_returns_same_service_per_key(synthetic_dataset, monkeypatch, tmp_path):
    monkeypatch.setattr(
        "analyst.data.registry.load_dataset", lambda name: synthetic_dataset
    )
    monkeypatch.setenv("ANALYST_ARTIFACTS_DIR", str(tmp_path))
    get_settings.cache_clear()
    clear_services()
    try:
        name = synthetic_dataset.name
        assert get_service(name, None) is get_service(name, None)
        assert get_service(name, date(2024, 1, 31)) is not get_service(name, None)
    finally:
        clear_services()
        get_settings.cache_clear()


def test_reset_gate_rereads_settings(monkeypatch):
    from analyst.data.gate import reset_gate

    get_settings.cache_clear()
    monkeypatch.setenv("ANALYST_MAX_CONCURRENT_QUERIES", "1")
    reset_gate()
    try:
        assert get_gate().max_concurrent == 1
    finally:
        monkeypatch.delenv("ANALYST_MAX_CONCURRENT_QUERIES")
        get_settings.cache_clear()
        reset_gate()
