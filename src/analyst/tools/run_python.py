"""CodeAct tool: run model-written Python in the per-thread sandbox."""

import hashlib
import json
import time

import pyarrow.parquet as pq
from langchain.tools import ToolRuntime

from analyst.context import AnalystContext
from analyst.errors import AnalystError, InvalidRequestError
from analyst.sandbox import ExecResult, SavedFrame, get_sandbox
from analyst.sandbox.ast_guard import check_code
from analyst.sandbox.docker import BETWEEN_CALLS_NOTE
from analyst.tools._common import (
    MAX_CONTENT_CHARS,
    Result,
    cap,
    data_tool,
    progress,
    run_tool,
    service_for,
)

MAX_CODE_CHARS = 20_000
MAX_CARD_COLUMNS = 8
MAX_ECHO_ROWS = 5
MAX_ECHO_ROW_CHARS = 300
ECHO_BUDGET = 1500
MAX_COLUMNS_CHARS = 200
_OMITTED = "(rows omitted; use describe)"
_LOST = "variables were lost; reload handles with load()"
_CODE_ERROR_HINT = (
    "fix the code and run it again; variables from earlier cells are still defined"
)
_OFF_MODEL_NOTE = "computed in the Python sandbox (off-model)"


class SandboxError(AnalystError):
    """The sandbox session failed (timeout, crash); its state is gone."""


def _thread_id(runtime: ToolRuntime[AnalystContext]) -> str:
    config = getattr(runtime, "config", None) or {}
    configurable = config.get("configurable") or {}
    return str(configurable.get("thread_id") or "default")


def _columns(names: list[str]) -> str:
    shown = ", ".join(names[:MAX_CARD_COLUMNS])
    extra = len(names) - MAX_CARD_COLUMNS
    text = f"{shown}, … (+{extra} more)" if extra > 0 else shown
    return (
        text if len(text) <= MAX_COLUMNS_CHARS else text[: MAX_COLUMNS_CHARS - 1] + "…"
    )


def _echo_row(row: dict) -> str:
    text = "  " + json.dumps(row, default=str)
    if len(text) > MAX_ECHO_ROW_CHARS:
        text = text[: MAX_ECHO_ROW_CHARS - 1] + "…"
    return text


def _echo_lines(echoes: list[tuple[str, list[str]]], budget: int) -> list[str]:
    """Echoed rows within ``budget`` chars; a marker when some were left out."""
    out: list[str] = []
    used = 0
    omitted = False
    label = len(echoes) > 1
    for handle_id, rows in echoes:
        block = ([f"rows of {handle_id}:"] if label else []) + rows
        for line in block:
            if used + len(line) + 1 > min(budget, ECHO_BUDGET):
                omitted = True
                break
            out.append(line)
            used += len(line) + 1
        else:
            continue
        break
    if omitted:
        out.append(_OMITTED)
    return out


def _content_sha(path) -> str:
    """sha256 of the saved parquet file's bytes (first 16 hex chars)."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def _ingest(
    runtime: ToolRuntime[AnalystContext],
    code: str,
    frames: list[SavedFrame],
    elapsed_s: float,
) -> tuple[list[str], list[dict], list[tuple[str, list[str]]]]:
    store = service_for(runtime).store
    lines: list[str] = []
    artifacts: list[dict] = []
    echoes: list[tuple[str, list[str]]] = []
    for frame in frames:
        try:
            try:
                table = pq.read_table(frame.path)
                # Content-addressed: the same cell over different kernel state
                # (new thread, restart) must never overwrite an earlier handle.
                card = store.save(
                    table,
                    produced_by="run_python",
                    source=code,
                    governed=False,
                    elapsed_s=elapsed_s,
                    notes=[_OFF_MODEL_NOTE],
                    meta={"name": frame.name, "content_sha": _content_sha(frame.path)},
                    index=False,
                )
            finally:
                frame.path.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001 - a bad frame must not lose the rest
            lines.append(
                f"note: saved frame {frame.name!r} could not be read ({type(exc).__name__})"
            )
            continue
        lines.append(
            f"handle: {card.handle_id}  rows: {table.num_rows}  "
            f"columns: {_columns(table.column_names)}"
        )
        if table.num_rows <= MAX_ECHO_ROWS:
            # Full precision, so the model can cite values exactly as stored.
            try:
                rows = [_echo_row(row) for row in table.to_pylist()]
            except Exception:  # noqa: BLE001 - an odd cell only loses the echo
                rows = []
            echoes.append((card.handle_id, rows))
        artifacts.append(
            {
                "handle_id": card.handle_id,
                "produced_by": "run_python",
                "governed": False,
            }
        )
    return lines, artifacts, echoes


def _discard(frames: list[SavedFrame]) -> None:
    for frame in frames:
        frame.path.unlink(missing_ok=True)


def _run(code: str, runtime: ToolRuntime[AnalystContext]) -> Result:
    if len(code) > MAX_CODE_CHARS:
        raise InvalidRequestError(
            f"code is {len(code)} chars; the limit is {MAX_CODE_CHARS}",
            hint="split the work into smaller cells",
        )
    check_code(code)
    ctx = runtime.context
    sandbox = get_sandbox(_thread_id(runtime), ctx.dataset, ctx.cutoff)
    progress(runtime, "run_python", "running Python in the sandbox")
    start = time.monotonic()
    res: ExecResult = sandbox.run(code)
    elapsed = time.monotonic() - start
    # restarted + ok: the container had died BEFORE this cell, which then ran
    # fine in a fresh one. Keep the result and say earlier variables are gone.
    restarted_before = res.restarted and res.ok
    if res.restarted and not res.ok:
        _discard(res.saved)
        raise SandboxError(res.error or "sandbox restarted", hint=_LOST)
    if not res.ok:
        _discard(res.saved)
        trace = res.error or "unknown error"
        raise InvalidRequestError(
            "Python error:\n" + trace[-(MAX_CONTENT_CHARS - 600) :],
            hint=_CODE_ERROR_HINT,
        )
    lines, artifacts, echoes = _ingest(runtime, code, res.saved, elapsed)
    parts = [f"note: {BETWEEN_CALLS_NOTE}"] if restarted_before else []
    if res.stdout.strip():
        parts.append(res.stdout.rstrip())
    if res.truncated:
        parts.append("(stdout truncated)")
    # Cards must survive the cap: bound stdout so the handle lines always fit.
    card_text = "\n".join(lines)
    room = MAX_CONTENT_CHARS - len(card_text) - 2
    echo = _echo_lines(echoes, max(0, room - len(_OMITTED) - 1))
    if echo:
        card_text = "\n".join([card_text, *echo]) if card_text else "\n".join(echo)
    budget = max(0, MAX_CONTENT_CHARS - len(card_text) - 2)
    head = cap("\n".join(parts)) if parts else ""
    if len(head) > budget:
        head = head[: max(0, budget - 14)] + "… (truncated)"
    content = "\n".join(x for x in (head, card_text) if x) or "ok (no output)"
    if not artifacts:
        return content, None
    if len(artifacts) == 1:
        return content, artifacts[0]
    return content, {"handles": artifacts}


@data_tool("run_python")
def run_python(code: str, runtime: ToolRuntime[AnalystContext]) -> Result:
    """Run Python in a persistent sandbox for analysis the typed tools cannot do.

    Use for correlations, rolling windows, custom statistics and reshaping.
    Read earlier results with load('r_…') (a pandas DataFrame) and keep any
    table you will cite with save(df, 'name'): that returns a new handle.
    Variables persist between calls. No network; numbers you cite must come
    from saved handles. Results are off-model.
    """
    return run_tool("run_python", runtime, lambda: _run(code, runtime))
