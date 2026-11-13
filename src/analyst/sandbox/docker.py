"""DockerSandbox: a persistent, hardened container running the exec server.

The container is the security boundary: no network, no capabilities, read-only
root, resource limits, non-root user, no environment and no raw-data mount.
It sees only the (dataset, cutoff) result-store segment (read-only) and a
per-thread scratch directory (read-write).
"""

import hashlib
import json
import os
import queue
import re
import shutil
import stat
import subprocess
import threading
import uuid
from dataclasses import replace
from datetime import date
from pathlib import Path

from analyst.config import get_settings
from analyst.data.service import segment_root
from analyst.sandbox.base import ExecResult, Sandbox, SavedFrame

CONTAINER_IN = "/artifacts/in"
CONTAINER_OUT = "/artifacts/out"
_RESTART_ERROR = "sandbox process exited; sandbox restarted, variables were lost"
_STDERR_TAIL = 500


def _user() -> str:
    uid, gid = os.getuid(), os.getgid()
    if uid == 0:
        uid, gid = 10001, 10001
    return f"{uid}:{gid}"


def docker_argv(in_dir: Path, out_dir: Path, image: str, name: str | None = None) -> list[str]:
    argv = ["docker", "run", "-i", "--rm"]
    if name:
        argv += ["--name", name]
    argv += [
        "--network", "none",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--read-only",
        "--pids-limit", "128",
        "--memory", "2g",
        "--memory-swap", "2g",
        "--cpus", "2",
        "--user", _user(),
        "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
        "-v", f"{in_dir}:{CONTAINER_IN}:ro",
        "-v", f"{out_dir}:{CONTAINER_OUT}:rw",
        image,
    ]
    return argv


class DockerSandbox:
    def __init__(
        self, *, in_dir: Path, out_dir: Path, image: str, timeout_s: float,
        staging_dir: Path | None = None,
    ) -> None:
        self.in_dir = Path(in_dir)
        self.out_dir = Path(out_dir)
        # Host-only (never mounted): saved files are copied here after validation.
        self.staging_dir = (
            Path(staging_dir) if staging_dir is not None
            else self.out_dir.with_name(self.out_dir.name + ".staged")
        )
        self._notice = ""
        self.image = image
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._stderr: list[str] = []
        self._name = ""
        self._n = 0

    # -- process management -------------------------------------------------
    def _start(self) -> None:
        self.in_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._name = f"analyst-sbx-{uuid.uuid4().hex[:12]}"
        argv = docker_argv(self.in_dir, self.out_dir, self.image, self._name)
        proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1,
        )
        lines: queue.Queue[str | None] = queue.Queue()
        stderr: list[str] = []

        def read_out() -> None:
            for line in proc.stdout:
                lines.put(line)
            lines.put(None)

        def read_err() -> None:
            for line in proc.stderr:
                stderr.append(line)
                del stderr[:-20]

        threading.Thread(target=read_out, daemon=True).start()
        threading.Thread(target=read_err, daemon=True).start()
        self._proc, self._lines, self._stderr = proc, lines, stderr

    def _kill(self) -> None:
        proc, name = self._proc, self._name
        self._proc = None
        if proc is None:
            return
        if name:
            try:
                subprocess.run(["docker", "kill", name], capture_output=True, check=False, timeout=15)
            except (subprocess.TimeoutExpired, OSError):
                pass
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

    # -- protocol -------------------------------------------------------------
    def run(self, code: str) -> ExecResult:
        with self._lock:
            if self._proc is not None and self._proc.poll() is not None:
                # Died between calls: tell the model its variables are gone.
                self._notice = "sandbox had exited; restarted, variables were lost"
            if self._proc is None or self._proc.poll() is not None:
                self._kill()
                self._start()
            self._n += 1
            rid = str(self._n)
            try:
                self._proc.stdin.write(json.dumps({"id": rid, "code": code}) + "\n")
                self._proc.stdin.flush()
            except (OSError, ValueError):
                return self._restarted(self._exit_error())
            try:
                line = self._lines.get(timeout=self.timeout_s)
            except queue.Empty:
                t = f"{self.timeout_s:g}"
                return self._restarted(f"timeout after {t} s; sandbox restarted, variables were lost")
            if line is None:
                return self._restarted(self._exit_error())
            try:
                resp = json.loads(line)
                if not isinstance(resp, dict) or resp.get("id") != rid:
                    raise ValueError("bad response")
            except ValueError:
                return self._restarted(self._exit_error())
            res = self._to_result(resp)
            if self._notice:
                note, self._notice = self._notice, ""
                res = replace(res, error=f"{note}\n{res.error}" if res.error else note, restarted=True)
            return res

    def _exit_error(self) -> str:
        tail = "".join(self._stderr)[-_STDERR_TAIL:].strip()
        return f"{_RESTART_ERROR} ({tail})" if tail else _RESTART_ERROR

    def _to_result(self, resp: dict) -> ExecResult:
        saved: list[SavedFrame] = []
        error = resp.get("error")
        for item in resp.get("saved") or []:
            cpath = str(item.get("path", ""))
            parent, _, fname = cpath.rpartition("/")
            if parent != CONTAINER_OUT or not fname or fname in (".", ".."):
                msg = f"rejected saved path outside {CONTAINER_OUT}: {cpath!r}"
                error = f"{error}\n{msg}" if error else msg
                continue
            try:
                staged = self._stage(fname)
            except OSError as exc:
                msg = f"rejected saved file {fname!r}: {exc}"
                error = f"{error}\n{msg}" if error else msg
                continue
            saved.append(SavedFrame(str(item["name"]), staged, int(item["rows"])))
        return ExecResult(
            ok=bool(resp.get("ok")) and error == resp.get("error"),
            stdout=str(resp.get("stdout", "")),
            error=error,
            saved=saved,
            truncated=bool(resp.get("truncated")),
        )

    def _stage(self, fname: str) -> Path:
        """Copy a container-written file to host-only staging, refusing links.

        The scratch dir is container-writable, so the file may be a symlink to a
        host path (cutoff bypass) or a hardlink; open with O_NOFOLLOW, require a
        regular single-link file, copy from the open fd, then remove the scratch file.
        """
        src = self.out_dir / fname
        if src.parent.resolve() != self.out_dir.resolve():
            raise OSError("path escapes the output directory")
        fd = os.open(src, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0))
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode):
                raise OSError("not a regular file")
            if st.st_nlink != 1:
                raise OSError("file has multiple hard links")
            self.staging_dir.mkdir(parents=True, exist_ok=True)
            dest = self.staging_dir / f"{uuid.uuid4().hex}.parquet"
            with os.fdopen(fd, "rb", closefd=False) as fin, open(dest, "xb") as fout:
                shutil.copyfileobj(fin, fout)
        finally:
            os.close(fd)
        try:
            os.unlink(src)
        except OSError:
            pass
        return dest

    def close(self) -> None:
        with self._lock:
            self._kill()


# -- registry ---------------------------------------------------------------
_registry: dict[tuple[str, str, date | None], DockerSandbox] = {}
_registry_lock = threading.Lock()


def _safe_thread_dir(thread_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", thread_id).strip("-")[:40] or "thread"
    return f"{slug}-{hashlib.sha256(thread_id.encode()).hexdigest()[:10]}"


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip(".-") or "dataset"


def get_sandbox(thread_id: str, dataset: str, cutoff: date | None) -> Sandbox:
    key = (thread_id, dataset, cutoff)
    with _registry_lock:
        sb = _registry.get(key)
        if sb is None:
            settings = get_settings()
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
        return sb


def close_all() -> None:
    with _registry_lock:
        boxes = list(_registry.values())
        _registry.clear()
    for sb in boxes:
        try:
            sb.close()
        except Exception:  # noqa: BLE001, S110 - keep closing the rest
            pass
