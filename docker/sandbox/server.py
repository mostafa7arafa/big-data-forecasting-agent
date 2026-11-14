"""Persistent Python exec server for the analyst CodeAct sandbox.

Protocol (newline-delimited JSON over stdin/stdout):
  request : {"id": str, "code": str}
  response: {"id", "ok", "stdout", "error", "saved": [{"name","path","rows"}], "truncated"}
Exactly one response line per request line, even when user code raises, exits or
writes to stderr. User stdout/stderr are captured and never reach the protocol stream.

Globals persist between requests. Pre-bound: pd, np, load(handle_id), save(df, name).
A bare final expression behaves like a REPL: DataFrames print head(10) plus shape,
any other non-None value prints its repr.

Standard library + pandas only at import time.
"""

import ast
import contextlib
import json
import os
import re
import sys
import traceback

import numpy as np
import pandas as pd

CAP = 4096
MAX_SAVES = 64  # per cell: keeps one response line far below the host's line limit
MARK = "… (truncated)"
ID_RE = re.compile(r"^r_[0-9a-f]{10}$")


def _dirs() -> tuple[str, str]:
    return (
        os.environ.get("ANALYST_SANDBOX_IN", "/artifacts/in"),
        os.environ.get("ANALYST_SANDBOX_OUT", "/artifacts/out"),
    )


def _cap(text: str) -> tuple[str, bool]:
    if len(text) <= CAP:
        return text, False
    return text[: CAP - len(MARK)] + MARK, True


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9_-]+", "-", str(name).lower()).strip("-")[:40].strip("-")
    return s or "result"


class BoundedWriter:
    """Text sink keeping at most ``limit`` chars; later writes are discarded."""

    def __init__(self, limit: int = CAP + 64) -> None:
        self.limit = limit
        self.size = 0
        self.truncated = False
        self._parts: list[str] = []

    def write(self, text: str) -> int:
        n = len(text)
        room = self.limit - self.size
        if room > 0:
            self._parts.append(text[:room])
            self.size += min(n, room)
        if n > room:
            self.truncated = True
        return n

    def flush(self) -> None:
        pass

    def getvalue(self) -> str:
        return "".join(self._parts)


class Session:
    def __init__(self) -> None:
        self.saved: list[dict] = []
        self._counts: dict[str, int] = {}
        self.ns: dict = {
            "__name__": "__sandbox__",
            "pd": pd,
            "np": np,
            "load": self.load,
            "save": self.save,
        }

    def load(self, handle_id: str) -> pd.DataFrame:
        if not isinstance(handle_id, str) or not ID_RE.fullmatch(handle_id):
            raise ValueError(f"invalid handle id {handle_id!r}; expected r_ + 10 hex chars")
        return pd.read_parquet(os.path.join(_dirs()[0], f"{handle_id}.parquet"))

    def save(self, df: pd.DataFrame, name: str) -> str:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("save() expects a pandas DataFrame")
        if len(self.saved) >= MAX_SAVES:
            raise ValueError(f"at most {MAX_SAVES} save() calls per cell")
        out = os.path.realpath(_dirs()[1])
        slug = _slug(name)
        while True:
            n = self._counts.get(slug, 0) + 1
            self._counts[slug] = n
            path = os.path.realpath(os.path.join(out, f"{slug}-{n}.parquet"))
            if os.path.dirname(path) != out:
                raise ValueError("save() may only write inside the output directory")
            if not os.path.exists(path):
                break
        df.to_parquet(path)
        self.saved.append({"name": f"{slug}-{n}", "path": path, "rows": len(df)})
        return path

    def _run(self, code: str) -> None:
        tree = ast.parse(code, "<sandbox>", "exec")
        last = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last = ast.Expression(tree.body.pop().value)
        exec(compile(tree, "<sandbox>", "exec"), self.ns)
        if last is not None:
            val = eval(compile(last, "<sandbox>", "eval"), self.ns)
            if isinstance(val, pd.DataFrame):
                print(val.head(10).to_string())
                print(f"shape: {val.shape}")
            elif val is not None:
                print(repr(val))

    def handle(self, req: dict) -> dict:
        self.saved = []
        buf = BoundedWriter()
        error = None
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                self._run(str(req.get("code", "")))
        except BaseException as exc:  # noqa: BLE001 - user code must never crash the server
            tb = exc.__traceback__
            error = "".join(traceback.format_exception(type(exc), exc, tb.tb_next if tb else None))
        stdout, t1 = _cap(buf.getvalue())
        t1 = t1 or buf.truncated
        err, t2 = _cap(error) if error else (None, False)
        return {
            "id": req.get("id"),
            "ok": error is None,
            "stdout": stdout,
            "error": err,
            "saved": list(self.saved),
            "truncated": t1 or t2,
        }


def main() -> None:
    proto = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)  # stray raw writes to fd 1 land on stderr, not the protocol stream
    session = Session()
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            if not isinstance(req, dict):
                raise ValueError("request must be a JSON object")
            resp = session.handle(req)
        except Exception as exc:  # noqa: BLE001
            resp = {"id": None, "ok": False, "stdout": "", "error": f"bad request: {exc}",
                    "saved": [], "truncated": False}
        proto.write(json.dumps(resp) + "\n")
        proto.flush()
    # stdin closed: the host is gone. Exit now, even if user code left
    # non-daemon threads running (they would keep the container alive).
    with contextlib.suppress(Exception):
        proto.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
