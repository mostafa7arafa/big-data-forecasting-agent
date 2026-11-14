"""DockerSandbox: a persistent, hardened container running the exec server.

The container is the security boundary: no network, no capabilities, read-only
root, resource limits, non-root user, no environment and no raw-data mount.
It sees only the (dataset, cutoff) result-store segment (read-only) and a
per-thread scratch directory (read-write).

The host side is bounded too: protocol lines and the stderr tail are read in
fixed-size chunks, staged files have a size cap, container logs are off, and
every container carries labels so orphans of dead processes can be reaped.
"""

import atexit
import hashlib
import json
import os
import queue
import re
import shutil
import socket
import stat
import subprocess
import threading
import uuid
from collections import OrderedDict
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import BinaryIO

from analyst.config import get_settings
from analyst.data.service import segment_root
from analyst.sandbox.base import ExecResult, Sandbox, SavedFrame

CONTAINER_IN = "/artifacts/in"
CONTAINER_OUT = "/artifacts/out"
LABEL = "analyst.sandbox"
OWNER_LABEL = "analyst.sandbox.owner"
MAX_LINE_BYTES = 256 * 1024  # one response is ~2 x 4,096 capped chars + <= 64 saved items
STDERR_KEEP_BYTES = 8 * 1024
MAX_STAGE_BYTES = 256 * 1024 * 1024  # per run; also the container's RLIMIT_FSIZE
_CHUNK = 64 * 1024
_RESTART_ERROR = "sandbox process exited; sandbox restarted, variables were lost"
_PROTOCOL_ERROR = (
    f"sandbox protocol error (response line over {MAX_LINE_BYTES} bytes); "
    "sandbox restarted, variables were lost"
)
BETWEEN_CALLS_NOTE = "sandbox had restarted before this cell; earlier variables were lost"
_STDERR_TAIL = 500
_LINE_QUEUE = 8


class _Overflow:
    """Queue sentinel: the container sent a protocol line over the limit."""


_OVERFLOW = _Overflow()


def _user() -> str:
    uid, gid = os.getuid(), os.getgid()
    if uid == 0:
        uid, gid = 10001, 10001
    return f"{uid}:{gid}"


def _owner() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


def docker_argv(in_dir: Path, out_dir: Path, image: str, name: str | None = None) -> list[str]:
    argv = ["docker", "run", "-i", "--rm"]
    if name:
        argv += ["--name", name]
    argv += [
        "--label", f"{LABEL}=1",
        "--label", f"{OWNER_LABEL}={_owner()}",
        "--log-driver", "none",
        "--network", "none",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--read-only",
        "--pids-limit", "128",
        "--memory", "2g",
        "--memory-swap", "2g",
        "--cpus", "2",
        "--ulimit", f"fsize={MAX_STAGE_BYTES}",
        "--user", _user(),
        "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
        "-v", f"{in_dir}:{CONTAINER_IN}:ro",
        "-v", f"{out_dir}:{CONTAINER_OUT}:rw",
        image,
    ]
    return argv


# -- bounded readers -----------------------------------------------------------
def _put(q: queue.Queue, item: object, stop: threading.Event) -> bool:
    """Put without growing the queue past its bound; give up once ``stop`` is set."""
    while not stop.is_set():
        try:
            q.put(item, timeout=0.2)
            return True
        except queue.Full:
            continue
    return False


def pump_lines(
    stream: BinaryIO, lines: queue.Queue, stop: threading.Event, limit: int = MAX_LINE_BYTES
) -> None:
    """Forward newline-terminated lines (bytes); never buffer more than ``limit``.

    A line over ``limit`` puts the overflow sentinel; end of stream puts
    ``None``. Whenever forwarding stops early (overflow, or ``stop`` set while
    the bounded queue is full) the rest of the stream is drained and discarded:
    a blocked pipe would stall the docker CLI, and with it ``docker kill``.
    """
    if _forward(stream, lines, stop, limit):
        _put(lines, None, stop)
    _drain(stream)


def _forward(stream: BinaryIO, lines: queue.Queue, stop: threading.Event, limit: int) -> bool:
    """True at end of stream; False when forwarding stopped early."""
    buf = bytearray()
    try:
        while not stop.is_set():
            chunk = stream.read(_CHUNK)
            if not chunk:
                return True
            buf += chunk
            while (i := buf.find(b"\n")) >= 0:
                line = bytes(buf[: i + 1])
                del buf[: i + 1]
                if len(line) > limit:
                    _put(lines, _OVERFLOW, stop)
                    return False
                if not _put(lines, line, stop):
                    return False
            if len(buf) > limit:
                _put(lines, _OVERFLOW, stop)
                return False
    except (OSError, ValueError):
        return True
    return False


def _drain(stream: BinaryIO) -> None:
    try:
        while stream.read(_CHUNK):
            pass
    except (OSError, ValueError):
        pass


def pump_tail(stream: BinaryIO, tail: bytearray, keep: int = STDERR_KEEP_BYTES) -> None:
    """Drain ``stream`` keeping only its last ``keep`` bytes in ``tail``."""
    try:
        while True:
            chunk = stream.read(_CHUNK)
            if not chunk:
                return
            tail += chunk
            if len(tail) > keep:
                del tail[:-keep]
    except (OSError, ValueError):
        return


def _docker(*args: str, timeout: float = 15) -> subprocess.CompletedProcess | None:
    """Best-effort docker CLI call; None when docker is missing or hangs."""
    try:
        return subprocess.run(
            ["docker", *args], capture_output=True, text=True, check=False, timeout=timeout
        )
    except (subprocess.TimeoutExpired, OSError):
        return None


class DockerSandbox:
    def __init__(
        self, *, in_dir: Path, out_dir: Path, image: str, timeout_s: float,
        staging_dir: Path | None = None, max_stage_bytes: int = MAX_STAGE_BYTES,
    ) -> None:
        self.in_dir = Path(in_dir)
        self.out_dir = Path(out_dir)
        # Host-only (never mounted): saved files are copied here after validation.
        self.staging_dir = (
            Path(staging_dir) if staging_dir is not None
            else self.out_dir.with_name(self.out_dir.name + ".staged")
        )
        self.image = image
        self.timeout_s = timeout_s
        self.max_stage_bytes = max_stage_bytes
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._lines: queue.Queue = queue.Queue(maxsize=_LINE_QUEUE)
        self._stderr = bytearray()
        self._stop = threading.Event()
        self._name = ""
        self._n = 0

    # -- process management -------------------------------------------------
    def _start(self) -> None:
        self.in_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._wipe_out()
        self._name = f"analyst-sbx-{uuid.uuid4().hex[:12]}"
        argv = docker_argv(self.in_dir, self.out_dir, self.image, self._name)
        proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=0,
        )
        lines: queue.Queue = queue.Queue(maxsize=_LINE_QUEUE)
        stderr = bytearray()
        stop = threading.Event()
        threading.Thread(target=pump_lines, args=(proc.stdout, lines, stop), daemon=True).start()
        threading.Thread(target=pump_tail, args=(proc.stderr, stderr), daemon=True).start()
        self._proc, self._lines, self._stderr, self._stop = proc, lines, stderr, stop

    def _kill(self) -> None:
        proc, name = self._proc, self._name
        self._proc = None
        self._stop.set()
        if proc is None:
            return
        if name:
            done = _docker("kill", name)
            if done is None or done.returncode != 0:
                _docker("rm", "-f", name)
        try:
            proc.kill()
        except OSError:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            try:
                if stream:
                    stream.close()
            except (OSError, ValueError):
                pass

    def _restarted(self, error: str) -> ExecResult:
        self._kill()
        return ExecResult(ok=False, stdout="", error=error, restarted=True)

    def _wipe_out(self) -> None:
        """Remove everything in the scratch dir without following links."""
        try:
            entries = list(os.scandir(self.out_dir))
        except OSError:
            return
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    shutil.rmtree(entry.path)
                else:
                    os.unlink(entry.path)
            except OSError:
                pass

    def _send(self, data: bytes) -> None:
        view = memoryview(data)
        while view:
            n = self._proc.stdin.write(view)
            if not n:
                raise OSError("short write to sandbox stdin")
            view = view[n:]

    # -- protocol -------------------------------------------------------------
    def run(self, code: str) -> ExecResult:
        with self._lock:
            notice = ""
            if self._proc is not None and self._proc.poll() is not None:
                # Died between calls: tell the model its variables are gone.
                notice = BETWEEN_CALLS_NOTE
            if self._proc is None or self._proc.poll() is not None:
                self._kill()
                self._start()
            try:
                return self._exchange(code, notice)
            finally:
                self._wipe_out()

    def _exchange(self, code: str, notice: str) -> ExecResult:
        self._n += 1
        rid = str(self._n)
        try:
            self._send((json.dumps({"id": rid, "code": code}) + "\n").encode("utf-8"))
        except (OSError, ValueError):
            return self._restarted(self._exit_error())
        try:
            line = self._lines.get(timeout=self.timeout_s)
        except queue.Empty:
            t = f"{self.timeout_s:g}"
            return self._restarted(f"timeout after {t} s; sandbox restarted, variables were lost")
        if line is None:
            return self._restarted(self._exit_error())
        if line is _OVERFLOW:
            return self._restarted(_PROTOCOL_ERROR)
        try:
            resp = json.loads(line)
            if not isinstance(resp, dict) or resp.get("id") != rid:
                raise ValueError("bad response")
        except (ValueError, RecursionError):
            return self._restarted(self._exit_error())
        res = self._to_result(resp)
        if notice:
            res = replace(res, error=f"{notice}\n{res.error}" if res.error else notice, restarted=True)
        return res

    def _exit_error(self) -> str:
        tail = bytes(self._stderr).decode("utf-8", "replace")[-_STDERR_TAIL:].strip()
        return f"{_RESTART_ERROR} ({tail})" if tail else _RESTART_ERROR

    def _to_result(self, resp: dict) -> ExecResult:
        """Validate and stage every saved item; any bad item rejects them all."""
        saved: list[SavedFrame] = []
        problems: list[str] = []
        error = resp.get("error")
        error = None if error is None else str(error)
        items = resp.get("saved") or []
        if not isinstance(items, list):
            problems.append("rejected malformed saved list")
            items = []
        budget = self.max_stage_bytes
        for item in items:
            try:
                if not isinstance(item, dict):
                    raise TypeError("saved item is not an object")
                name, rows, cpath = str(item["name"]), int(item["rows"]), str(item["path"])
                parent, _, fname = cpath.rpartition("/")
                if parent != CONTAINER_OUT or not fname or fname in (".", ".."):
                    raise ValueError(f"saved path outside {CONTAINER_OUT}: {cpath!r}")
                staged, size = self._stage(fname, budget)
                budget -= size
                saved.append(SavedFrame(name, staged, rows))
            except Exception as exc:  # noqa: BLE001 - a forged item must never escape run()
                problems.append(f"rejected saved item {repr(item)[:200]}: {exc}")
        if problems:
            for frame in saved:
                frame.path.unlink(missing_ok=True)
            saved = []
            error = "\n".join([error, *problems] if error else problems)
        return ExecResult(
            ok=resp.get("ok") is True and not problems,
            stdout=str(resp.get("stdout") or ""),
            error=error,
            saved=saved,
            truncated=bool(resp.get("truncated")),
        )

    def _stage(self, fname: str, budget: int | None = None) -> tuple[Path, int]:
        """Copy a container-written file to host-only staging, refusing links.

        The scratch dir is container-writable, so the file may be a symlink to a
        host path (cutoff bypass), a hardlink, or a huge sparse file; open with
        O_NOFOLLOW, require a regular single-link file within the size budget,
        copy from the open fd, then remove the scratch file.
        """
        limit = self.max_stage_bytes if budget is None else min(budget, self.max_stage_bytes)
        src = self.out_dir / fname
        if src.parent.resolve() != self.out_dir.resolve():
            raise OSError("path escapes the output directory")
        fd = os.open(src, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0))
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode):
                raise OSError("not a regular file")
            if st.st_nlink != 1:
                raise OSError("file has multiple hard links")
            if st.st_size > limit:
                raise OSError(f"file is {st.st_size} bytes; the staging limit is {limit}")
            self.staging_dir.mkdir(parents=True, exist_ok=True)
            dest = self.staging_dir / f"{uuid.uuid4().hex}.parquet"
            copied = 0
            try:
                with os.fdopen(fd, "rb", closefd=False) as fin, open(dest, "xb") as fout:
                    while copied < st.st_size:
                        chunk = fin.read(min(1 << 20, st.st_size - copied))
                        if not chunk:
                            break
                        fout.write(chunk)
                        copied += len(chunk)
            except BaseException:
                dest.unlink(missing_ok=True)
                raise
        finally:
            os.close(fd)
        try:
            os.unlink(src)
        except OSError:
            pass
        return dest, copied

    def close(self) -> None:
        with self._lock:
            self._kill()


# -- orphan reaping ----------------------------------------------------------------
def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def reap(*, own: bool = False) -> list[str]:
    """Remove labelled sandbox containers whose owner process on this host is gone.

    ``own=True`` also removes this process's own containers (used at exit).
    Containers of other live processes, or of other hosts, are left alone.
    """
    listed = _docker(
        "ps", "-a", "--filter", f"label={LABEL}=1",
        "--format", f'{{{{.ID}}}} {{{{.Label "{OWNER_LABEL}"}}}}',
    )
    if listed is None or listed.returncode != 0:
        return []
    host, me = socket.gethostname(), os.getpid()
    doomed = []
    for row in listed.stdout.splitlines():
        cid, _, owner = row.strip().partition(" ")
        o_host, _, o_pid = owner.rpartition(":")
        if not cid or o_host != host or not o_pid.isdigit():
            continue
        pid = int(o_pid)
        if (pid == me and own) or (pid != me and not _pid_alive(pid)):
            doomed.append(cid)
    if doomed:
        _docker("rm", "-f", *doomed, timeout=60)
    return doomed


# -- registry ---------------------------------------------------------------
_registry: "OrderedDict[tuple[str, str, date | None], DockerSandbox]" = OrderedDict()
_registry_lock = threading.Lock()
_reaped = False


def _safe_thread_dir(thread_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", thread_id).strip("-")[:40] or "thread"
    return f"{slug}-{hashlib.sha256(thread_id.encode()).hexdigest()[:10]}"


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip(".-") or "dataset"


def _close_quietly(sb: DockerSandbox) -> None:
    try:
        sb.close()
    except Exception:  # noqa: BLE001, S110 - closing is best effort
        pass


def _at_exit() -> None:
    close_all()
    reap(own=True)


def get_sandbox(thread_id: str, dataset: str, cutoff: date | None) -> Sandbox:
    global _reaped
    key = (thread_id, dataset, cutoff)
    evicted: list[DockerSandbox] = []
    with _registry_lock:
        if not _reaped:
            _reaped = True
            reap()
            atexit.register(_at_exit)
        settings = get_settings()
        sb = _registry.get(key)
        if sb is None:
            segment = cutoff.isoformat() if cutoff is not None else "full"
            rel = Path(_safe_thread_dir(thread_id)) / _safe_name(dataset) / segment
            sb = DockerSandbox(
                in_dir=segment_root(dataset, cutoff),
                out_dir=settings.artifacts_dir / "sandbox" / rel,
                staging_dir=settings.artifacts_dir / "sandbox-staging" / rel,
                image=settings.sandbox_image,
                timeout_s=settings.sandbox_timeout_s,
            )
            _registry[key] = sb
        _registry.move_to_end(key)
        while len(_registry) > settings.sandbox_max_live:
            evicted.append(_registry.popitem(last=False)[1])
    # Outside the lock and off this caller: close() waits for an in-flight run.
    for old in evicted:
        threading.Thread(target=_close_quietly, args=(old,), daemon=True).start()
    return sb


def close_sandbox(thread_id: str) -> None:
    """Close every sandbox of ``thread_id`` (all datasets and cutoffs)."""
    with _registry_lock:
        keys = [k for k in _registry if k[0] == thread_id]
        boxes = [_registry.pop(k) for k in keys]
    for sb in boxes:
        _close_quietly(sb)


def close_all() -> None:
    with _registry_lock:
        boxes = list(_registry.values())
        _registry.clear()
    for sb in boxes:
        _close_quietly(sb)
