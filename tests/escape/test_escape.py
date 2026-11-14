"""Sandbox escape-attempt suite.

Every case runs its attack with the AST guard bypassed (``DockerSandbox.run``
directly) and asserts the container contains it. Unless the guard legitimately
allows the code (ruling R4: huge stdout, infinite loop, memory bomb, load()
traversal, sparse files, output floods), the case also asserts the guard
rejects it at the tool level.

"Contains" is not always "fails": ctypes, subprocess, importlib, eval/exec and
the ``__subclasses__`` walk are guard-blocked; with the guard bypassed the
container runs them, unprivileged (uid != 0, no capabilities, no network, no
secrets in the environment), and the cases assert exactly that.
Probes print secret env var NAMES only, never values.
"""

import subprocess
import textwrap
import time
from datetime import date

import pytest

from analyst.config import get_settings
from analyst.errors import InvalidRequestError
from analyst.sandbox import docker as sbx
from analyst.sandbox.ast_guard import check_code
from analyst.sandbox.docker import DockerSandbox, get_sandbox

pytestmark = pytest.mark.docker

IMAGE = "analyst-sandbox:1"
SECRET_NAMES = ("OPENAI_API_KEY", "LANGSMITH_API_KEY", "OPENROUTER", "ANTHROPIC_API_KEY")


def _need_image():
    try:
        ok = subprocess.run(["docker", "image", "inspect", IMAGE], check=False,
                            capture_output=True).returncode == 0
    except FileNotFoundError:
        ok = False
    if not ok:
        pytest.skip(f"docker image {IMAGE} unavailable (run make sandbox-image)")


def _containers(*names: str) -> str:
    """Ids of sandbox containers still present (waits briefly for --rm cleanup)."""
    out = ""
    for _ in range(20):
        ps = subprocess.run(["docker", "ps", "-a", "--format", "{{.Names}}", "--filter", "name=analyst-sbx-"],
                            check=False, capture_output=True, text=True).stdout.split()
        out = " ".join(n for n in ps if not names or n in names)
        if not out:
            return ""
        time.sleep(0.25)
    return out


def _track(box, names: list[str]) -> None:
    """Record every container name the sandbox starts (restarts included)."""
    orig_start = box._start

    def start() -> None:
        orig_start()
        names.append(box._name)

    box._start = start


@pytest.fixture(scope="module")
def sb(tmp_path_factory):
    _need_image()
    root = tmp_path_factory.mktemp("escape")
    box = DockerSandbox(in_dir=root / "in", out_dir=root / "out", image=IMAGE, timeout_s=30)
    names: list[str] = []
    _track(box, names)
    yield box
    box.close()
    assert _containers(*names) == "", "sandbox containers left running"


def _run(sb, code: str):
    return sb.run(textwrap.dedent(code))


def _guard_rejects(code: str, fragment: str) -> None:
    with pytest.raises(InvalidRequestError) as exc:
        check_code(textwrap.dedent(code))
    assert fragment.lower() in exc.value.message.lower(), exc.value.message
    assert exc.value.hint


def _alive(sb) -> None:
    r = sb.run("print('alive')")
    assert r.ok and r.stdout.strip() == "alive", r.error


# 1
def test_network_connect_refused(sb):
    code = "import socket\nsocket.create_connection(('1.1.1.1', 53), 2)"
    _guard_rejects(code, "import of 'socket'")
    r = _run(sb, code)
    assert not r.ok and ("OSError" in r.error or "unreachable" in r.error)


# 2
def test_dns_lookup_refused(sb):
    code = "import socket\nsocket.getaddrinfo('example.com', 443)"
    _guard_rejects(code, "import of 'socket'")
    r = _run(sb, code)
    assert not r.ok and "gaierror" in r.error


# 3
def test_read_etc_shadow_refused(sb):
    code = "print(open('/etc/shadow').read())"
    _guard_rejects(code, "under /artifacts/")
    r = _run(sb, code)
    assert not r.ok and "PermissionError" in r.error and r.stdout == ""


# 4
def test_write_usr_read_only_root(sb):
    code = "open('/usr/x', 'w').write('x')"
    _guard_rejects(code, "under /artifacts/")
    r = _run(sb, code)
    assert not r.ok and "Read-only file system" in r.error


# 5
def test_write_artifacts_in_read_only(sb):
    code = "open('/artifacts/in/x', 'w').write('x')"
    _guard_rejects(code, "may only write under /artifacts/out/")
    r = _run(sb, code)
    assert not r.ok and "Read-only file system" in r.error


# 6
def test_no_data_mount(sb):
    _guard_rejects("import os\nprint(os.listdir('/'))", "import of 'os'")
    r = _run(sb, """
        import os
        print(os.path.exists('/data'), '/data' in os.listdir('/'))
        targets = sorted(line.split()[4] for line in open('/proc/self/mountinfo'))
        print([t for t in targets if t.startswith(('/artifacts', '/data', '/home', '/srv', '/mnt'))])
    """)
    assert r.ok, r.error
    lines = r.stdout.strip().splitlines()
    assert lines[0] == "False False"
    assert lines[1] == "['/artifacts/in', '/artifacts/out']"


# 7
def test_fork_bomb_hits_pids_limit(sb):
    code = """
        import os, signal, time
        pids, err = [], None
        try:
            for _ in range(400):
                pid = os.fork()
                if pid == 0:
                    time.sleep(60)
                    os._exit(0)
                pids.append(pid)
        except OSError as e:
            err = type(e).__name__
        for p in pids:
            os.kill(p, signal.SIGKILL)
            os.waitpid(p, 0)
        print(err, len(pids))
    """
    _guard_rejects(code, "import of 'os'")
    r = _run(sb, code)
    assert r.ok, r.error
    err, n = r.stdout.split()
    assert err == "BlockingIOError" and int(n) < 128
    _alive(sb)


# 8 (guard allows: R4)
def test_memory_bomb_contained(sb):
    code = "import numpy as np\nx = np.ones(400_000_000)\nprint(x.sum())"  # 3.2 GB > 2 GB limit
    check_code(code)
    r = _run(sb, code)
    assert not r.ok
    assert r.restarted or "MemoryError" in r.error
    assert "timeout" not in r.error, "a hang must not pass as OOM"
    _alive(sb)


# 9 (guard allows: R4)
def test_infinite_loop_times_out(tmp_path):
    _need_image()
    code = "while True:\n    pass"
    check_code(code)
    box = DockerSandbox(in_dir=tmp_path / "in", out_dir=tmp_path / "out", image=IMAGE, timeout_s=5)
    try:
        r = box.run(code)
        assert not r.ok and r.restarted and r.error.startswith("timeout after 5 s")
        _alive(box)
    finally:
        box.close()


# 10
def test_ctypes(sb):
    _guard_rejects("import ctypes\nctypes.CDLL(None).system(b'id')", "import of 'ctypes'")
    # Container side: libc is reachable without the guard; what it can do is not.
    r = _run(sb, """
        import ctypes
        libc = ctypes.CDLL(None)
        print(libc.getuid(), libc.setuid(0), ctypes.get_errno())
    """)
    assert r.ok, r.error
    uid, rc, _ = r.stdout.split()
    assert uid != "0" and rc == "-1"


# 11
def test_subprocess_and_os_system(sb):
    _guard_rejects("import subprocess\nsubprocess.run(['id'])", "import of 'subprocess'")
    _guard_rejects("import os\nos.system('id')", "import of 'os'")
    _guard_rejects("pd.io.common.os.system('id')", "os.system")
    r = _run(sb, """
        import subprocess, sys
        print(subprocess.run(['id', '-u'], capture_output=True, text=True).stdout.strip())
        probe = "import socket; socket.create_connection(('1.1.1.1', 53), 2)"
        print(subprocess.run([sys.executable, '-c', probe], capture_output=True).returncode)
    """)
    assert r.ok, r.error
    uid, rc = r.stdout.split()
    assert uid != "0" and rc != "0"


# 12
def test_subclasses_walk(sb):
    code = """
        wrap = [c for c in ().__class__.__bases__[0].__subclasses__() if c.__name__ == '_wrap_close'][0]
        osmod = wrap.__init__.__globals__
        print(osmod['getuid'](), sorted(k for k in osmod['environ'] if any(s in k for s in SECRETS)))
    """
    _guard_rejects(code, "dunder attribute")
    r = _run(sb, f"SECRETS = {SECRET_NAMES!r}\n" + textwrap.dedent(code))
    assert r.ok, r.error
    assert r.stdout.split()[0] != "0" and r.stdout.strip().endswith("[]")


# 13
def test_importlib(sb):
    code = "import importlib\nos = importlib.import_module('os')\nprint(os.getuid())"
    _guard_rejects(code, "import of 'importlib'")
    _guard_rejects("from importlib import import_module", "import of 'importlib'")
    r = _run(sb, code)
    assert r.ok and r.stdout.strip() != "0", r.error


# 14
@pytest.mark.parametrize("code", [
    "eval(\"__import__('os').getuid()\")",
    "exec(\"import os; print(os.getuid())\")",
    "exec(compile(\"import os; print(os.getuid())\", 's', 'exec'))",
])
def test_eval_exec_compile(sb, code):
    _guard_rejects(code, f"'{code.split('(', 1)[0]}' is not allowed")
    r = _run(sb, f"print({code})" if code.startswith("eval") else code)
    assert r.ok and r.stdout.strip().splitlines()[0] != "0", r.error


# 15 (guard allows: R4)
def test_huge_stdout_capped(sb):
    code = "print('x' * 10_000_000)"
    check_code(code)
    r = _run(sb, code)
    assert r.ok and r.truncated and len(r.stdout) <= 4096


# 16
def test_setuid_and_capabilities(sb):
    _guard_rejects("import os\nos.setuid(0)", "import of 'os'")
    _guard_rejects("print(open('/proc/self/status').read())", "under /artifacts/")
    r = _run(sb, "import os\nos.setuid(0)")
    assert not r.ok and "PermissionError" in r.error
    r = _run(sb, """
        caps = dict(line.split(':\\t') for line in open('/proc/self/status') if line.startswith('Cap'))
        print(int(caps['CapEff'], 16), int(caps['CapPrm'], 16), int(caps['CapBnd'], 16))
        print([l.split()[1] for l in open('/proc/self/status') if l.startswith('NoNewPrivs')][0])
    """)
    assert r.ok, r.error
    eff, prm, bnd = r.stdout.splitlines()[0].split()
    assert eff == prm == bnd == "0"
    assert r.stdout.splitlines()[1] == "1"


# 17
def test_no_host_secrets_in_environ(sb):
    _guard_rejects("print(open('/proc/1/environ').read())", "under /artifacts/")
    r = _run(sb, f"""
        import os
        names = [kv.split(b'=', 1)[0].decode() for kv in open('/proc/1/environ', 'rb').read().split(b'\\0') if kv]
        names += list(os.environ)
        print(sorted({{n for n in names if any(s in n for s in {SECRET_NAMES!r})}}))
        print(' '.join(sorted(set(names))))
    """)
    assert r.ok, r.error
    hits, names = r.stdout.splitlines()
    assert hits == "[]"
    # Only the image's own ENV (plus Docker's HOSTNAME): nothing from the host leaks in.
    image_env = subprocess.run(
        ["docker", "image", "inspect", "-f", "{{range .Config.Env}}{{println .}}{{end}}", IMAGE],
        check=True, capture_output=True, text=True).stdout.split()
    allowed = {kv.split("=", 1)[0] for kv in image_env} | {"HOSTNAME"}
    assert set(names.split()) <= allowed


# 18 (guard allows: R4)
@pytest.mark.parametrize("arg", ["'../../etc/passwd'", "'/etc/passwd'", "'r_0123456789/../../x'", "5"])
def test_load_traversal_rejected(sb, arg):
    code = f"load({arg})"
    check_code(code)
    r = _run(sb, code)
    assert not r.ok and "invalid handle id" in r.error


# 19
def test_other_cutoff_segment_not_mounted(tmp_path, monkeypatch):
    _need_image()
    _guard_rejects("open('/artifacts/../2024-02-01/two.txt')", "'..'")
    monkeypatch.setattr(get_settings(), "artifacts_dir", tmp_path)
    sbx.close_all()
    started: list[str] = []
    try:
        d1, d2 = date(2024, 1, 1), date(2024, 2, 1)
        for d, name in [(d1, "one"), (d2, "two")]:
            root = tmp_path / "ds" / d.isoformat()
            root.mkdir(parents=True)
            (root / f"{name}.txt").write_text(name)
        s1, s2 = get_sandbox("escape", "ds", d1), get_sandbox("escape", "ds", d2)
        _track(s1, started)
        _track(s2, started)
        assert s2.run("import os\nprint(sorted(os.listdir('/artifacts/in')))").stdout.strip() == "['two.txt']"
        r = s1.run(textwrap.dedent("""
            import os
            found = [os.path.join(d, f) for top in ('/artifacts', '/tmp', '/srv', '/mnt', '/home')
                     for d, _, fs in os.walk(top) for f in fs if f == 'two.txt']
            mounts = open('/proc/self/mountinfo').read()
            print(found, '2024-02-01' in mounts, '2024-01-01' in mounts)
            print(sorted(os.listdir('/artifacts/in')))
        """))
        assert r.ok, r.error
        assert r.stdout.splitlines() == ["[] False True", "['one.txt']"]
    finally:
        sbx.close_all()
    assert len(started) == 2 and _containers(*started) == ""


# 20
@pytest.mark.parametrize("path", ["/x", "/srv/x", "/home/x", "/etc/x", "/artifacts/x"])
def test_write_outside_artifacts_out_fails(sb, path):
    code = f"open({path!r}, 'w').write('x')"
    _guard_rejects(code, "under /artifacts/")
    r = _run(sb, code)
    assert not r.ok and ("Read-only file system" in r.error or "PermissionError" in r.error)


def test_write_inside_scratch_still_works(sb):
    code = "open('/artifacts/out/ok.txt', 'w').write('x')"
    check_code(code)
    r = _run(sb, code + "\nopen('/tmp/ok.txt', 'w').write('y')")
    assert r.ok, r.error


# --- host-side resource bounds and lifecycle (final review I2, I3) -------------------
def _box(tmp_path, **kw) -> DockerSandbox:
    _need_image()
    return DockerSandbox(in_dir=tmp_path / "in", out_dir=tmp_path / "out", image=IMAGE,
                         timeout_s=kw.pop("timeout_s", 30), **kw)


_SPARSE = """
    save(pd.DataFrame({'a': [1]}), 'sparse')
    f = open('/artifacts/out/sparse-1.parquet', 'r+b')
    f.seek({offset})
    f.write(b'1')
    f.close()
"""


# 21 (guard allows: R4) sparse file past RLIMIT_FSIZE
def test_sparse_file_past_fsize_blocked_in_container(tmp_path):
    code = textwrap.dedent(_SPARSE).replace("{offset}", str(300 * 1024 * 1024))
    check_code(code)
    box = _box(tmp_path)
    try:
        r = box.run(code)
        assert not r.ok and ("File too large" in r.error or "Errno 27" in r.error), r.error
        for f in r.saved:
            f.path.unlink(missing_ok=True)
        _alive(box)
    finally:
        box.close()


# 22 (guard allows: R4) sparse file under fsize but over the host staging limit
def test_sparse_file_over_staging_limit_rejected_on_host(tmp_path):
    code = textwrap.dedent(_SPARSE).replace("{offset}", str(100 * 1024 * 1024))
    check_code(code)
    box = _box(tmp_path, max_stage_bytes=50 * 1024 * 1024)
    try:
        r = box.run(code)
        assert not r.ok and not r.saved and "staging limit" in r.error, r.error
        assert not box.staging_dir.exists() or not any(box.staging_dir.iterdir())
        assert list(box.out_dir.iterdir()) == []  # scratch wiped after the call
    finally:
        box.close()


# 23 stderr flood without newline: host keeps only a bounded tail
def test_stderr_flood_is_bounded(sb):
    code = "import os\nchunk = b'e' * (1 << 20)\nfor _ in range(300):\n    os.write(2, chunk)\nprint('done')"
    _guard_rejects(code, "import of 'os'")
    r = sb.run(code)
    assert r.ok and r.stdout.strip() == "done", r.error
    assert len(sb._stderr) <= 8 * 1024
    _alive(sb)


# 24 protocol-fd flood without newline: protocol error, restart, recovery
def test_protocol_fd_flood_restarts(tmp_path):
    code = textwrap.dedent("""
        import os, stat
        skip = {os.fstat(0).st_ino, os.fstat(2).st_ino}
        chunk = b'p' * (1 << 20)
        for fd in range(3, 64):
            try:
                st = os.fstat(fd)
            except OSError:
                continue
            if stat.S_ISFIFO(st.st_mode) and st.st_ino not in skip:
                for _ in range(300):
                    os.write(fd, chunk)
    """)
    _guard_rejects(code, "import of 'os'")
    box = _box(tmp_path)
    try:
        t0 = time.monotonic()
        r = box.run(code)
        assert time.monotonic() - t0 < 30
        assert not r.ok and r.restarted and "protocol error" in r.error, r.error
        _alive(box)
    finally:
        box.close()


# 25 orphan on exit: a non-daemon thread must not keep the container alive
def test_stdin_eof_exits_container_despite_non_daemon_thread(tmp_path):
    box = _box(tmp_path)
    try:
        r = box.run("import threading, time\nthreading.Thread(target=time.sleep, args=(600,)).start()")
        assert r.ok, r.error
        name = box._name
        labelled = subprocess.run(
            ["docker", "ps", "-q", "--filter", "label=analyst.sandbox=1", "--filter", f"name={name}"],
            check=False, capture_output=True, text=True).stdout
        assert labelled.strip(), "sandbox container is not labelled"
        box._proc.stdin.close()  # what the host's death looks like to the container
        box._proc.wait(timeout=20)
        assert _containers(name) == "", "container outlived its host connection"
    finally:
        box.close()
