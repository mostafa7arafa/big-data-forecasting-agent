import io
import os
import subprocess
import threading
from datetime import date
from pathlib import Path

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

    class Live:
        def __init__(self):
            self.stdin = io.StringIO()

        def poll(self):
            return None

    def fake_start():
        sb._proc = Live()
        sb._lines.put('{"id": "1", "ok": true, "stdout": "1\\n", "error": null, "saved": [], "truncated": false}\n')

    monkeypatch.setattr(sb, "_start", fake_start)
    res = sb.run("print(1)")
    assert res.ok and res.restarted and "variables were lost" in res.error


def test_saved_path_rewrite_and_reject(tmp_path):
    sb = _box(tmp_path)
    (sb.out_dir / "a-1.parquet").write_bytes(b"x")
    res = sb._to_result({"ok": True, "stdout": "", "error": None, "truncated": False, "saved": [
        {"name": "a-1", "path": "/artifacts/out/a-1.parquet", "rows": 3},
        {"name": "b", "path": "/artifacts/out/../etc/b.parquet", "rows": 1},
        {"name": "c", "path": "/tmp/c.parquet", "rows": 1},
    ]})
    assert len(res.saved) == 1 and res.saved[0].name == "a-1"
    assert not res.ok and "rejected" in res.error


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
    try:
        assert sb.run("x = 5").ok
        r = sb.run("while True: pass")
        assert not r.ok and r.restarted
        assert r.error == "timeout after 2 s; sandbox restarted, variables were lost"
        assert sb.run("print(1)").ok
        assert not sb.run("x").ok
    finally:
        sb.close()
    ps = subprocess.run(["docker", "ps", "-q", "--filter", "name=analyst-sbx-"],
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
