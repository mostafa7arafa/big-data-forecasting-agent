import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage

from analyst.context import AnalystContext
from analyst.data import registry
from analyst.data.results import ResultStore
from analyst.data.service import DataService
from analyst.sandbox.base import ExecResult, SavedFrame
from analyst.state import Answer, Claim
from analyst.tools import DATA_TOOLS
from analyst.verify.claims import verify_answer
from analyst.verify.trace import collect_handles

rp_mod = __import__("analyst.tools.run_python", fromlist=["x"])


class FakeSandbox:
    def __init__(self, results):
        self.results = list(results)
        self.codes = []

    def run(self, code):
        self.codes.append(code)
        return self.results.pop(0)

    def close(self):
        pass


@pytest.fixture
def svc(synthetic_dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "load_dataset", lambda name: synthetic_dataset)
    original = DataService.__init__

    def init(self, dataset, *, cutoff=None, store_root=None):
        original(self, dataset, cutoff=cutoff, store_root=tmp_path / "store")

    monkeypatch.setattr(DataService, "__init__", init)
    registry.clear_services()
    yield tmp_path
    registry.clear_services()


def install(monkeypatch, results, seen=None):
    box = FakeSandbox(results)

    def fake_get(thread_id, dataset, cutoff):
        if seen is not None:
            seen.append((thread_id, dataset, cutoff))
        return box

    monkeypatch.setattr(rp_mod, "get_sandbox", fake_get)
    return box


def invoke(code, config=None):
    runtime = ToolRuntime(
        state={},
        context=AnalystContext(dataset="events"),
        config=config if config is not None else {},
        stream_writer=lambda e: None,
        tool_call_id="t1",
        store=None,
    )
    return rp_mod.run_python.invoke(
        {
            "name": "run_python",
            "args": {"code": code, "runtime": runtime},
            "id": "t1",
            "type": "tool_call",
        }
    )


def frame(tmp_path, name, table, fname=None):
    p = tmp_path / (fname or f"{name}.parquet")
    pq.write_table(table, p)
    return SavedFrame(name=name, path=p, rows=table.num_rows)


def test_guard_rejection_is_tool_error(svc):
    msg = invoke("import socket")
    assert msg.status == "error"
    assert "Allowed imports" in msg.content


def test_code_length_cap(svc, monkeypatch):
    install(monkeypatch, [])
    msg = invoke("x = 1\n" * 5000)
    assert msg.status == "error" and "split the work into smaller cells" in msg.content


def test_saved_frame_becomes_handle(svc, monkeypatch):
    f = frame(svc, "corr", pa.table({"a": [1.0, 2.0], "b": [3.0, 4.0]}))
    install(monkeypatch, [ExecResult(True, "hello\n", None, [f])])
    msg = invoke("print('hello')")
    assert msg.status != "error"
    art = msg.artifact
    assert art["produced_by"] == "run_python" and art["governed"] is False
    assert f"handle: {art['handle_id']}  rows: 2  columns: a, b" in msg.content
    assert msg.content.startswith("hello")
    assert not f.path.exists()
    store = ResultStore(svc / "store")
    assert store.exists(art["handle_id"])
    assert "off-model" in " ".join(store.card(art["handle_id"]).notes)


def test_several_frames_and_unreadable_frame(svc, monkeypatch):
    f1 = frame(svc, "a", pa.table({"x": [1]}))
    f2 = frame(svc, "b", pa.table({"y": [2]}))
    bad = SavedFrame("bad", svc / "bad.parquet", 1)
    bad.path.write_bytes(b"not parquet")
    install(monkeypatch, [ExecResult(True, "", None, [f1, bad, f2])])
    msg = invoke("pass")
    assert len(msg.artifact["handles"]) == 2
    assert "bad" in msg.content and "could not be read" in msg.content
    assert not bad.path.exists() and not f1.path.exists() and not f2.path.exists()


def test_many_columns_truncated_and_content_capped(svc, monkeypatch):
    t = pa.table({f"c{i}": [i] for i in range(30)})
    f = frame(svc, "wide", t)
    install(monkeypatch, [ExecResult(True, "z" * 9000, None, [f])])
    msg = invoke("pass")
    assert len(msg.content) <= 4096
    assert "(+22 more)" in msg.content and "handle: r_" in msg.content


def test_user_exception_is_error_with_traceback(svc, monkeypatch):
    install(monkeypatch, [ExecResult(False, "", "Traceback ...\nNameError: x", [])])
    msg = invoke("x")
    assert (
        msg.status == "error"
        and "NameError: x" in msg.content
        and "Python error" in msg.content
    )


def test_timeout_is_error_with_lost_hint(svc, monkeypatch):
    install(
        monkeypatch,
        [
            ExecResult(
                False,
                "",
                "timeout after 30 s; sandbox restarted, variables were lost",
                [],
                restarted=True,
            )
        ],
    )
    msg = invoke("while True: pass")
    assert msg.status == "error"
    assert (
        "timeout after 30 s" in msg.content
        and "reload handles with load()" in msg.content
    )


def test_thread_id_scopes_sandbox(svc, monkeypatch):
    seen = []
    install(
        monkeypatch,
        [
            ExecResult(True, "a", None),
            ExecResult(True, "b", None),
            ExecResult(True, "c", None),
        ],
        seen,
    )
    invoke("pass", {"configurable": {"thread_id": "T1"}})
    invoke("pass")
    invoke("pass", {"configurable": {}})
    assert [s[0] for s in seen] == ["T1", "default", "default"]
    assert seen[0][1] == "events"


def test_run_python_handles_are_off_model(svc, monkeypatch):
    f = frame(svc, "r", pa.table({"v": [10.0, 20.0]}))
    install(monkeypatch, [ExecResult(True, "", None, [f])])
    msg = invoke("pass")
    store = ResultStore(svc / "store")
    handles = collect_handles([msg], store)
    hid = msg.artifact["handle_id"]
    assert handles[hid].governed is False and handles[hid].produced_by == "run_python"
    # multi-handle artifact form
    multi = ToolMessage(
        content="x", tool_call_id="1", artifact={"handles": [msg.artifact]}
    )
    assert hid in collect_handles([multi], store)
    answer = Answer(
        narrative="ok",
        claims=[Claim(text="c", type="descriptive", handles=[hid], values=[10.0])],
    )
    v = verify_answer(
        answer,
        question="q",
        handles=handles,
        memory=_memory(),
        cutoff=None,
        store=store,
    )
    assert v.label == "verified_off_model" and v.off_model_tools == ["run_python"]


def _memory():
    from analyst.semantic.model import Memory, SemanticModel, TimeDimension

    return Memory(
        semantic=SemanticModel(
            table="t",
            time_dimension=TimeDimension(name="d", column="d", grains=["day"]),
        ),
        quirks=[],
        series=[],
        context_facts=[],
        recap="",
    )


def test_tool_in_worker_allowlist():
    assert "run_python" in [t.name for t in DATA_TOOLS]
    from analyst import worker

    assert "run_python" in worker.WORKER_PROMPT


def test_small_frame_rows_echoed_full_precision(svc, monkeypatch):
    v = 0.08283779943279351
    f = frame(svc, "c", pa.table({"measure": ["m"], "correlation": [v]}))
    install(monkeypatch, [ExecResult(True, "", None, [f])])
    msg = invoke("pass")
    assert f'  {{"measure": "m", "correlation": {v!r}}}' in msg.content
    big = frame(svc, "big", pa.table({"x": list(range(6))}), "big.parquet")
    install(monkeypatch, [ExecResult(True, "", None, [big])])
    assert '{"x"' not in invoke("pass.").content


@pytest.mark.parametrize("restarted", [False, True])
def test_failed_result_unlinks_staged_frames(svc, monkeypatch, restarted):
    f = frame(svc, "s", pa.table({"x": [1]}))
    install(monkeypatch, [ExecResult(False, "", "boom", [f], restarted=restarted)])
    assert invoke("pass").status == "error"
    assert not f.path.exists()
