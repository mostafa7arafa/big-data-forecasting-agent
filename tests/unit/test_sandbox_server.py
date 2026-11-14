import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

SERVER = Path(__file__).parents[2] / "docker" / "sandbox" / "server.py"


class Proc:
    def __init__(self, tmp_path):
        self.inp, self.out = tmp_path / "in", tmp_path / "out"
        self.inp.mkdir()
        self.out.mkdir()
        env = {**os.environ, "ANALYST_SANDBOX_IN": str(self.inp), "ANALYST_SANDBOX_OUT": str(self.out)}
        self.p = subprocess.Popen(
            [sys.executable, "-u", str(SERVER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, env=env,
        )
        self.n = 0

    def run(self, code):
        self.n += 1
        self.p.stdin.write(json.dumps({"id": str(self.n), "code": code}) + "\n")
        self.p.stdin.flush()
        resp = json.loads(self.p.stdout.readline())
        assert resp["id"] == str(self.n)
        return resp


@pytest.fixture
def server(tmp_path):
    s = Proc(tmp_path)
    yield s
    s.p.stdin.close()
    s.p.wait(timeout=10)


@pytest.fixture
def handle_parquet(server):
    pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]}).to_parquet(server.inp / "r_0123456789.parquet")
    return "r_0123456789"


def test_state_persists_between_requests(server):
    server.run("x = 41")
    assert server.run("print(x + 1)")["stdout"] == "42\n"


def test_load_and_save_roundtrip(server, handle_parquet):
    r = server.run(f"df = load('{handle_parquet}')\nprint(save(df, 'My Result!'))")
    assert r["ok"], r["error"]
    assert len(r["saved"]) == 1
    s = r["saved"][0]
    assert s["rows"] == 3 and s["path"] == r["stdout"].strip()
    assert Path(s["path"]).parent == server.out.resolve()
    assert Path(s["path"]).name == "my-result-1.parquet"
    assert len(pd.read_parquet(s["path"])) == 3
    r2 = server.run("print(save(df, 'My Result!'))")
    assert r2["saved"][0]["path"].endswith("my-result-2.parquet")


@pytest.mark.parametrize("bad", ["../etc/passwd", "r_ABCDEF0123", "r_012345678", "r_0123456789\\n", "r_0123456789/../x"])
def test_load_rejects_bad_ids(server, bad):
    r = server.run(f"load({bad!r})")
    assert not r["ok"] and "invalid handle id" in r["error"]


def test_save_cannot_escape_out_dir(server):
    r = server.run("import pandas as pd\nprint(save(pd.DataFrame({'a': [1]}), '../../evil'))")
    assert r["ok"]
    assert Path(r["saved"][0]["path"]).parent == server.out.resolve()
    assert not (server.out.parent / "evil-1.parquet").exists()


def test_output_capped(server):
    r = server.run("print('x'*50000)")
    assert len(r["stdout"]) <= 4096 and r["truncated"] and r["stdout"].endswith("(truncated)")


def test_traceback_is_error_not_crash(server):
    r = server.run("1/0")
    assert not r["ok"] and "ZeroDivisionError" in r["error"]
    assert server.run("print('alive')")["stdout"] == "alive\n"


def test_syntax_error_and_sys_exit_do_not_crash(server):
    assert not server.run("def (")["ok"]
    assert not server.run("import sys; sys.exit(3)")["ok"]
    assert server.run("print(1)")["ok"]


def test_bare_dataframe_prints_head_and_shape(server):
    r = server.run("import pandas as pd\npd.DataFrame({'a': range(25)})")
    assert r["ok"] and "shape: (25, 1)" in r["stdout"]
    assert r["stdout"].count("\n") <= 13


def test_bare_expression_prints_repr(server):
    assert server.run("1 + 1")["stdout"] == "2\n"
    assert server.run("None")["stdout"] == ""


def test_stderr_and_raw_fd_writes_do_not_corrupt_protocol(server):
    r = server.run("import sys, os\nprint('out')\nprint('err', file=sys.stderr)\nos.write(1, b'RAW\\n')")
    assert r["ok"] and "out" in r["stdout"] and "err" in r["stdout"]
    assert server.run("print(2)")["stdout"] == "2\n"


def test_malformed_request_gets_one_error_line(server):
    server.p.stdin.write("not json\n")
    server.p.stdin.flush()
    resp = json.loads(server.p.stdout.readline())
    assert resp["ok"] is False and "bad request" in resp["error"]
    assert server.run("print(1)")["ok"]


def test_stdin_eof_exits_despite_non_daemon_thread(tmp_path):
    s = Proc(tmp_path)
    r = s.run("import threading, time\nthreading.Thread(target=time.sleep, args=(600,)).start()")
    assert r["ok"], r["error"]
    s.p.stdin.close()
    assert s.p.wait(timeout=10) == 0


def test_save_count_is_capped_per_cell(server):
    r = server.run("df = pd.DataFrame({'a': [1]})\nfor i in range(65):\n    save(df, 'x')")
    assert not r["ok"] and "at most 64 save() calls per cell" in r["error"]
    assert len(r["saved"]) == 64
    r = server.run("save(pd.DataFrame({'a': [1]}), 'y')")
    assert r["ok"] and len(r["saved"]) == 1


def _server_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("sandbox_server_mod", SERVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_bounded_writer_never_holds_more_than_limit():
    mod = _server_module()
    w = mod.BoundedWriter(limit=100)
    for _ in range(200_000):
        w.write("line of output\n")
    assert w.size == 100 and len(w.getvalue()) == 100 and w.truncated


def test_bounded_writer_small_output_not_truncated():
    mod = _server_module()
    w = mod.BoundedWriter(limit=100)
    w.write("hi")
    assert w.getvalue() == "hi" and not w.truncated


def test_repeated_prints_are_capped(server):
    r = server.run("for _ in range(200000):\n    print('spam line')")
    assert r["ok"] and r["truncated"] and len(r["stdout"]) <= 4096
    assert r["stdout"].endswith("… (truncated)")
