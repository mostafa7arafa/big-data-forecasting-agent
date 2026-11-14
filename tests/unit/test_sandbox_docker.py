import io
import os
import subprocess
import threading
from datetime import date
from pathlib import Path
from typing import ClassVar

import pandas as pd
import pytest

from analyst.config import get_settings
from analyst.sandbox import docker as sbx
from analyst.sandbox.docker import DockerSandbox, docker_argv, get_sandbox

IMAGE = "analyst-sandbox:1"


def _need_image():
    try:
        ok = subprocess.run(["docker", "image", "inspect", IMAGE], check=False, capture_output=True).returncode == 0
    except FileNotFoundError:
        ok = False
    if not ok:
        pytest.skip(f"docker image {IMAGE} unavailable (run make sandbox-image)")


@pytest.fixture
def tmp_dirs(tmp_path):
    _need_image()
    i, o = tmp_path / "in", tmp_path / "out"
    sb = DockerSandbox(in_dir=i, out_dir=o, image=IMAGE, timeout_s=20)
    yield sb
    sb.close()


def test_argv_has_every_hardening_flag():
    argv = docker_argv(Path("/a"), Path("/b"), "img", "analyst-sbx-x")
    for flag in ["--network", "none", "--cap-drop", "ALL", "--read-only", "--pids-limit", "128",
                 "--memory", "2g", "--memory-swap", "--cpus", "2", "no-new-privileges", "--user", "--tmpfs",
                 "--rm", "-i", "analyst-sbx-x"]:
        assert flag in argv
    joined = " ".join(argv)
    assert "/a:/artifacts/in:ro" in joined and "/b:/artifacts/out:rw" in joined
    assert not any("/data" in a for a in argv)
    assert "-e" not in argv and "--env" not in argv and "--env-file" not in argv
    assert not argv[argv.index("--user") + 1].startswith("0:")
    assert argv[-1] == "img"
    assert argv[argv.index("--log-driver") + 1] == "none"
    assert argv[argv.index("--ulimit") + 1] == "fsize=268435456"
    labels = [argv[i + 1] for i, a in enumerate(argv) if a == "--label"]
    assert "analyst.sandbox=1" in labels
    assert f"analyst.sandbox.owner={sbx._owner()}" in labels


def test_thread_dir_cannot_escape_or_collide():
    a, b = sbx._safe_thread_dir("../x"), sbx._safe_thread_dir("..-x")
    assert "/" not in a and ".." not in a and a != b


def _resp(*names):
    return {"ok": True, "stdout": "", "error": None, "truncated": False, "saved": [
        {"name": n, "path": f"/artifacts/out/{n}.parquet", "rows": 1} for n in names]}


def _box(tmp_path):
    sb = DockerSandbox(in_dir=tmp_path / "i", out_dir=tmp_path / "o", image="x", timeout_s=1)
    sb.out_dir.mkdir(parents=True)
    return sb


def test_symlink_in_scratch_rejected(tmp_path):
    secret = tmp_path / "secret.parquet"
    secret.write_bytes(b"beyond-cutoff")
    sb = _box(tmp_path)
    (sb.out_dir / "x.parquet").symlink_to(secret)
    res = sb._to_result(_resp("x"))
    assert not res.ok and not res.saved and "rejected" in res.error
    assert secret.read_bytes() == b"beyond-cutoff"


def test_hardlink_in_scratch_rejected(tmp_path):
    secret = tmp_path / "secret.parquet"
    secret.write_bytes(b"data")
    sb = _box(tmp_path)
    os.link(secret, sb.out_dir / "h.parquet")
    res = sb._to_result(_resp("h"))
    assert not res.ok and not res.saved and "hard links" in res.error


def test_normal_file_is_staged_outside_out_dir(tmp_path):
    sb = _box(tmp_path)
    (sb.out_dir / "n.parquet").write_bytes(b"abc")
    res = sb._to_result(_resp("n"))
    assert res.ok
    (f,) = res.saved
    assert sb.out_dir not in f.path.parents and f.path.read_bytes() == b"abc"
    assert not (sb.out_dir / "n.parquet").exists()


def test_fifo_in_scratch_rejected_without_hang(tmp_path):
    sb = _box(tmp_path)
    os.mkfifo(sb.out_dir / "f.parquet")
    out = []
    t = threading.Thread(target=lambda: out.append(sb._to_result(_resp("f"))), daemon=True)
    t.start()
    t.join(5)
    assert not t.is_alive(), "FIFO open blocked"
    assert not out[0].ok and not out[0].saved and "regular" in out[0].error


def test_nul_in_name_rejected(tmp_path):
    sb = _box(tmp_path)
    resp = _resp("a")
    resp["saved"][0]["path"] = "/artifacts/out/a\x00.parquet"
    res = sb._to_result(resp)
    assert not res.ok and not res.saved and "rejected" in res.error


def test_exited_between_calls_reports_restart(tmp_path, monkeypatch):
    sb = _box(tmp_path)

    class Dead:
        def poll(self):
            return 1

    sb._proc = Dead()
    monkeypatch.setattr(sb, "_kill", lambda: setattr(sb, "_proc", None))

    def fake_start():
        sb._proc = Live()
        sb._lines.put(b'{"id": "1", "ok": true, "stdout": "1\\n", "error": null, "saved": [], "truncated": false}\n')

    monkeypatch.setattr(sb, "_start", fake_start)
    res = sb.run("print(1)")
    assert res.ok and res.restarted and res.error == sbx.BETWEEN_CALLS_NOTE


class Live:
    def __init__(self):
        self.stdin = io.BytesIO()

    def poll(self):
        return None


def test_saved_path_rewrite_and_reject(tmp_path):
    sb = _box(tmp_path)
    (sb.out_dir / "a-1.parquet").write_bytes(b"x")
    res = sb._to_result({"ok": True, "stdout": "", "error": None, "truncated": False, "saved": [
        {"name": "a-1", "path": "/artifacts/out/a-1.parquet", "rows": 3},
        {"name": "b", "path": "/artifacts/out/../etc/b.parquet", "rows": 1},
        {"name": "c", "path": "/tmp/c.parquet", "rows": 1},
    ]})
    # any rejected item rejects the whole response; staged copies are removed
    assert res.saved == [] and not res.ok and "rejected" in res.error
    assert not sb.staging_dir.exists() or not any(sb.staging_dir.iterdir())


@pytest.mark.docker
def test_state_persists(tmp_dirs):
    assert tmp_dirs.run("x = 41").ok
    r = tmp_dirs.run("x + 1")
    assert r.ok and r.stdout.strip() == "42" and not r.restarted


@pytest.mark.docker
def test_save_returns_host_path(tmp_dirs):
    r = tmp_dirs.run("import pandas as pd\nsave(pd.DataFrame({'a': [1, 2]}), 'demo')")
    assert r.ok, r.error
    (f,) = r.saved
    assert tmp_dirs.out_dir not in f.path.parents and f.rows == 2
    assert len(pd.read_parquet(f.path)) == 2


@pytest.mark.docker
def test_timeout_restarts_and_reports_state_loss(tmp_path):
    _need_image()
    sb = DockerSandbox(in_dir=tmp_path / "i", out_dir=tmp_path / "o", image=IMAGE, timeout_s=2)
    name = ""
    try:
        assert sb.run("x = 5").ok
        name = sb._name
        r = sb.run("while True: pass")
        assert not r.ok and r.restarted
        assert r.error == "timeout after 2 s; sandbox restarted, variables were lost"
        assert sb.run("print(1)").ok
        assert not sb.run("x").ok
        name = sb._name
    finally:
        sb.close()
    ps = subprocess.run(["docker", "ps", "-q", "--filter", f"name={name}"],
                        check=False, capture_output=True, text=True).stdout
    assert ps.strip() == ""


@pytest.mark.docker
def test_container_exit_restarts(tmp_dirs):
    r = tmp_dirs.run("import os\nos._exit(1)")
    assert not r.ok and r.restarted and "variables were lost" in r.error
    assert tmp_dirs.run("print(2)").ok


@pytest.mark.docker
def test_output_capped(tmp_dirs):
    r = tmp_dirs.run("print('x' * 100000)")
    assert r.ok and r.truncated and len(r.stdout) <= 4096


@pytest.mark.docker
def test_no_network(tmp_dirs):
    r = tmp_dirs.run("import socket\nsocket.create_connection(('1.1.1.1', 53), 2)")
    assert not r.ok


@pytest.mark.docker
def test_mount_is_segment_scoped(tmp_path, monkeypatch):
    _need_image()
    monkeypatch.setattr(get_settings(), "artifacts_dir", tmp_path)
    sbx.close_all()
    try:
        d1, d2 = date(2024, 1, 1), date(2024, 2, 1)
        for d, name in [(d1, "one"), (d2, "two")]:
            root = tmp_path / "ds" / d.isoformat()
            root.mkdir(parents=True)
            (root / f"{name}.txt").write_text(name)
        s1, s2 = get_sandbox("t", "ds", d1), get_sandbox("t", "ds", d2)
        assert s1 is get_sandbox("t", "ds", d1) and s1 is not s2
        assert s1.run("import os\nprint(sorted(os.listdir('/artifacts/in')))").stdout.strip() == "['one.txt']"
        assert s2.run("import os\nprint(sorted(os.listdir('/artifacts/in')))").stdout.strip() == "['two.txt']"
        assert not s1.run("open('/artifacts/in/x', 'w')").ok
    finally:
        sbx.close_all()


# --- host-side bounds (I2) --------------------------------------------------------
def _pump(data: bytes, limit: int = sbx.MAX_LINE_BYTES) -> list:
    import queue

    q: queue.Queue = queue.Queue(maxsize=8)
    t = threading.Thread(target=sbx.pump_lines, args=(io.BytesIO(data), q, threading.Event(), limit))
    t.start()
    t.join(5)
    assert not t.is_alive()
    out = []
    while not q.empty():
        out.append(q.get())
    return out


def test_pump_lines_splits_and_ends_with_none():
    assert _pump(b'{"a": 1}\n{"b": 2}\n') == [b'{"a": 1}\n', b'{"b": 2}\n', None]


def test_pump_lines_overflow_without_newline_is_bounded():
    out = _pump(b"x" * (3 * sbx.MAX_LINE_BYTES))
    assert out == [sbx._OVERFLOW]


def test_pump_lines_overlong_line_with_newline_overflows():
    assert _pump(b"y" * 100 + b"\n", limit=50) == [sbx._OVERFLOW]


def test_pump_lines_blocks_instead_of_growing_queue():
    import queue

    q: queue.Queue = queue.Queue(maxsize=2)
    stop = threading.Event()
    t = threading.Thread(target=sbx.pump_lines, args=(io.BytesIO(b"l\n" * 1000), q, stop), daemon=True)
    t.start()
    t.join(0.5)
    assert t.is_alive() and q.qsize() == 2
    stop.set()
    t.join(5)
    assert not t.is_alive()


def test_pump_tail_keeps_only_the_tail():
    tail = bytearray()
    sbx.pump_tail(io.BytesIO(b"a" * 1_000_000 + b"END"), tail)
    assert len(tail) == sbx.STDERR_KEEP_BYTES and tail.endswith(b"END")


def test_overflow_restarts_with_protocol_error(tmp_path, monkeypatch):
    sb = _box(tmp_path)
    killed = []
    monkeypatch.setattr(sb, "_kill", lambda: (killed.append(1), setattr(sb, "_proc", None)))

    def fake_start():
        sb._proc = Live()
        sb._lines.put(sbx._OVERFLOW)

    monkeypatch.setattr(sb, "_start", fake_start)
    res = sb.run("print(1)")
    assert not res.ok and res.restarted and "protocol error" in res.error
    assert "variables were lost" in res.error and killed


def test_stage_rejects_file_over_limit(tmp_path):
    sb = DockerSandbox(in_dir=tmp_path / "i", out_dir=tmp_path / "o", image="x", timeout_s=1,
                       max_stage_bytes=1000)
    sb.out_dir.mkdir(parents=True)
    with open(sb.out_dir / "s.parquet", "wb") as f:
        f.truncate(5000)  # sparse: big size, no blocks
    res = sb._to_result(_resp("s"))
    assert not res.ok and not res.saved and "staging limit" in res.error
    assert not sb.staging_dir.exists() or not any(sb.staging_dir.iterdir())


def test_stage_default_limit_rejects_sparse_256mb(tmp_path):
    sb = _box(tmp_path)
    assert sb.max_stage_bytes == 256 * 1024 * 1024
    with open(sb.out_dir / "s.parquet", "wb") as f:
        f.truncate(sb.max_stage_bytes + 1)
    res = sb._to_result(_resp("s"))
    assert not res.ok and "staging limit" in res.error


def test_stage_budget_is_per_response(tmp_path):
    sb = DockerSandbox(in_dir=tmp_path / "i", out_dir=tmp_path / "o", image="x", timeout_s=1,
                       max_stage_bytes=1000)
    sb.out_dir.mkdir(parents=True)
    (sb.out_dir / "a.parquet").write_bytes(b"a" * 600)
    (sb.out_dir / "b.parquet").write_bytes(b"b" * 600)
    res = sb._to_result(_resp("a", "b"))
    assert not res.ok and not res.saved and "staging limit" in res.error
    assert not any(sb.staging_dir.iterdir())


def test_wipe_out_removes_files_dirs_and_links_not_targets(tmp_path):
    sb = _box(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep")
    (sb.out_dir / "f.txt").write_text("x")
    (sb.out_dir / "link").symlink_to(outside)
    (sb.out_dir / "sub").mkdir()
    (sb.out_dir / "sub" / "g").write_text("y")
    os.mkfifo(sb.out_dir / "fifo")
    sb._wipe_out()
    assert list(sb.out_dir.iterdir()) == []
    assert (outside / "keep.txt").read_text() == "keep"


def test_run_wipes_out_dir_after_the_call(tmp_path, monkeypatch):
    sb = _box(tmp_path)
    monkeypatch.setattr(sb, "_kill", lambda: setattr(sb, "_proc", None))

    def fake_start():
        sb._proc = Live()
        (sb.out_dir / "leftover.parquet").write_bytes(b"junk")
        sb._lines.put(b'{"id": "1", "ok": true, "stdout": "", "error": null, "saved": [], "truncated": false}\n')

    monkeypatch.setattr(sb, "_start", fake_start)
    assert sb.run("pass").ok
    assert list(sb.out_dir.iterdir()) == []


# --- malformed responses (I6) -----------------------------------------------------
@pytest.mark.parametrize("bad", [
    {"name": "b", "path": "/artifacts/out/b.parquet", "rows": "many"},
    {"path": "/artifacts/out/b.parquet", "rows": 1},
    {"name": "b", "rows": 1},
    "not-a-dict",
    None,
    {"name": "b", "path": ["/artifacts/out/b.parquet"], "rows": 1},
])
def test_malformed_saved_item_rejected_and_staged_unlinked(tmp_path, bad):
    sb = _box(tmp_path)
    (sb.out_dir / "a.parquet").write_bytes(b"x")
    resp = _resp("a")
    resp["saved"].append(bad)
    res = sb._to_result(resp)
    assert not res.ok and res.saved == [] and "rejected" in res.error
    assert not any(sb.staging_dir.iterdir())


def test_non_list_saved_and_odd_fields(tmp_path):
    sb = _box(tmp_path)
    res = sb._to_result({"ok": True, "stdout": None, "error": 5, "saved": {"x": 1}})
    assert not res.ok and res.saved == [] and res.stdout == ""
    assert res.error.startswith("5\n") and "malformed" in res.error
    assert not sb._to_result({"ok": "yes", "saved": []}).ok


# --- lifecycle (I3) ---------------------------------------------------------------
def test_kill_falls_back_to_rm_when_docker_kill_fails(tmp_path, monkeypatch):
    sb = _box(tmp_path)
    calls = []

    def fake_docker(*args, timeout=15):
        calls.append(args)
        return subprocess.CompletedProcess(args, 1 if args[0] == "kill" else 0, "", "")

    monkeypatch.setattr(sbx, "_docker", fake_docker)

    class P:
        stdin = stdout = stderr = None

        def kill(self):
            pass

        def wait(self, timeout=None):
            return 0

    sb._proc, sb._name = P(), "analyst-sbx-abc"
    sb._kill()
    assert calls == [("kill", "analyst-sbx-abc"), ("rm", "-f", "analyst-sbx-abc")]
    assert sb._proc is None


def test_reap_removes_only_dead_owners_on_this_host(monkeypatch):
    host, me = sbx.socket.gethostname(), os.getpid()
    rows = "\n".join([
        f"c_dead {host}:111",
        f"c_live {host}:222",
        f"c_mine {host}:{me}",
        "c_other otherhost:111",
        "c_nolabel ",
    ])
    calls = []

    def fake_docker(*args, timeout=15):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, rows if args[0] == "ps" else "", "")

    monkeypatch.setattr(sbx, "_docker", fake_docker)
    monkeypatch.setattr(sbx, "_pid_alive", lambda pid: pid == 222)
    assert sbx.reap() == ["c_dead"]
    assert calls[-1] == ("rm", "-f", "c_dead")
    assert sbx.reap(own=True) == ["c_dead", "c_mine"]


def test_reap_tolerates_missing_docker(monkeypatch):
    monkeypatch.setattr(sbx, "_docker", lambda *a, **k: None)
    assert sbx.reap() == []


class _Closing:
    closed: ClassVar[list] = []

    def __init__(self, **kw):
        self.kw = kw

    def close(self):
        _Closing.closed.append(self)


@pytest.fixture
def fake_registry(monkeypatch, tmp_path):
    monkeypatch.setattr(get_settings(), "artifacts_dir", tmp_path)
    monkeypatch.setattr(get_settings(), "sandbox_max_live", 2)
    monkeypatch.setattr(sbx, "DockerSandbox", _Closing)
    monkeypatch.setattr(sbx, "_reaped", True)
    monkeypatch.setattr(sbx, "_registry", sbx.OrderedDict())
    _Closing.closed = []
    return sbx._registry


def _wait_closed(n):
    import time

    for _ in range(50):
        if len(_Closing.closed) >= n:
            return
        time.sleep(0.02)


def test_registry_evicts_least_recently_used(fake_registry):
    a = get_sandbox("a", "ds", None)
    b = get_sandbox("b", "ds", None)
    assert get_sandbox("a", "ds", None) is a  # a is now most recent
    get_sandbox("c", "ds", None)
    _wait_closed(1)
    assert _Closing.closed == [b]
    assert [k[0] for k in fake_registry] == ["a", "c"]


def test_close_sandbox_closes_every_entry_of_a_thread(fake_registry, monkeypatch):
    monkeypatch.setattr(get_settings(), "sandbox_max_live", 8)
    x1 = get_sandbox("x", "ds", None)
    x2 = get_sandbox("x", "ds", date(2024, 1, 1))
    y = get_sandbox("y", "ds", None)
    sbx.close_sandbox("x")
    assert set(map(id, _Closing.closed)) == {id(x1), id(x2)}
    assert list(fake_registry.values()) == [y]


def test_first_get_sandbox_reaps_orphans(monkeypatch, tmp_path):
    monkeypatch.setattr(get_settings(), "artifacts_dir", tmp_path)
    monkeypatch.setattr(sbx, "DockerSandbox", _Closing)
    monkeypatch.setattr(sbx, "_registry", sbx.OrderedDict())
    monkeypatch.setattr(sbx, "_reaped", False)
    reaps, hooks = [], []
    monkeypatch.setattr(sbx, "reap", lambda **k: reaps.append(k))
    monkeypatch.setattr(sbx.atexit, "register", hooks.append)
    get_sandbox("t", "ds", None)
    get_sandbox("u", "ds", None)
    assert reaps == [{}] and hooks == [sbx._at_exit]


@pytest.mark.parametrize("rows", [float("inf"), float("nan"), 1e999])
def test_non_finite_rows_rejected(tmp_path, rows):
    sb = _box(tmp_path)
    (sb.out_dir / "a.parquet").write_bytes(b"x")
    (sb.out_dir / "b.parquet").write_bytes(b"y")
    resp = _resp("a", "b")
    resp["saved"][1]["rows"] = rows
    res = sb._to_result(resp)
    assert not res.ok and res.saved == [] and "rejected" in res.error
    assert not any(sb.staging_dir.iterdir())


def test_deeply_nested_response_restarts(tmp_path, monkeypatch):
    sb = _box(tmp_path)
    monkeypatch.setattr(sb, "_kill", lambda: setattr(sb, "_proc", None))

    def fake_start():
        sb._proc = Live()
        sb._lines.put(b"[" * 100_000 + b"]" * 100_000 + b"\n")

    monkeypatch.setattr(sb, "_start", fake_start)
    res = sb.run("pass")
    assert not res.ok and res.restarted


def test_instances_for_same_key_get_distinct_scratch_dirs_and_close_removes_own(tmp_path):
    a = DockerSandbox(in_dir=tmp_path / "i", out_dir=tmp_path / "o", image="x", timeout_s=1)
    b = DockerSandbox(in_dir=tmp_path / "i", out_dir=tmp_path / "o", image="x", timeout_s=1)
    assert a.out_dir != b.out_dir
    assert a.out_dir.parent == b.out_dir.parent == tmp_path / "o"
    a.out_dir.mkdir(parents=True)
    b.out_dir.mkdir(parents=True)
    (b.out_dir / "keep.parquet").write_bytes(b"x")
    a.close()
    assert not a.out_dir.exists()
    assert (b.out_dir / "keep.parquet").exists()
