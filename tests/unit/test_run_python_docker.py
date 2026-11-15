import subprocess

import pyarrow as pa
import pytest
from langchain.tools import ToolRuntime

from analyst.config import get_settings
from analyst.context import AnalystContext
from analyst.data import registry
from analyst.data.results import ResultStore
from analyst.data.service import DataService
from analyst.sandbox import close_all
from analyst.tools.run_python import run_python

pytestmark = pytest.mark.docker


def _need_image():
    image = get_settings().sandbox_image
    try:
        ok = (
            subprocess.run(
                ["docker", "image", "inspect", image], check=False, capture_output=True
            ).returncode
            == 0
        )
    except FileNotFoundError:
        ok = False
    if not ok:
        pytest.skip(f"docker image {image} unavailable (run make sandbox-image)")


@pytest.fixture
def env(synthetic_dataset, tmp_path, monkeypatch):
    _need_image()
    monkeypatch.setenv("ANALYST_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    get_settings.cache_clear() if hasattr(get_settings, "cache_clear") else None
    monkeypatch.setattr(registry, "load_dataset", lambda name: synthetic_dataset)
    root = tmp_path / "store"
    original = DataService.__init__

    def init(self, dataset, *, cutoff=None, store_root=None):
        original(self, dataset, cutoff=cutoff, store_root=root)

    monkeypatch.setattr(DataService, "__init__", init)
    monkeypatch.setattr("analyst.sandbox.docker.segment_root", lambda d, c: root)
    registry.clear_services()
    yield root
    close_all()
    registry.clear_services()
    if hasattr(get_settings, "cache_clear"):
        get_settings.cache_clear()


def _call(code, thread="docker-t"):
    runtime = ToolRuntime(
        state={},
        context=AnalystContext(dataset="events"),
        config={"configurable": {"thread_id": thread}},
        stream_writer=lambda e: None,
        tool_call_id="t",
        store=None,
    )
    return run_python.invoke(
        {
            "name": "run_python",
            "args": {"code": code, "runtime": runtime},
            "id": "t",
            "type": "tool_call",
        }
    )


def test_end_to_end_load_compute_save(env):
    store = ResultStore(env)
    src = store.save(
        pa.table({"d": list(range(10)), "v": [float(i * i) for i in range(10)]}),
        produced_by="query_metrics",
        source="seed",
        governed=True,
        elapsed_s=0.0,
    )
    msg = _call(
        f"df = load('{src.handle_id}')\n"
        "df['roll'] = df['v'].rolling(3).mean()\n"
        "save(df, 'rolled')\n"
        "print('done')"
    )
    assert msg.status != "error", msg.content
    assert "done" in msg.content
    art = msg.artifact
    assert art["produced_by"] == "run_python" and art["governed"] is False
    table = store.load(art["handle_id"])
    assert "roll" in table.column_names and table.num_rows == 10
