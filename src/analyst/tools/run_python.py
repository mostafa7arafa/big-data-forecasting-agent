"""CodeAct tool: run model-written Python in the per-thread sandbox."""

import hashlib
import time

import pyarrow.parquet as pq
from langchain.tools import ToolRuntime

from analyst.context import AnalystContext
from analyst.errors import AnalystError, InvalidRequestError
from analyst.sandbox import ExecResult, SavedFrame, get_sandbox
from analyst.sandbox.ast_guard import check_code
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
    return f"{shown}, … (+{extra} more)" if extra > 0 else shown


def _ingest(
    runtime: ToolRuntime[AnalystContext],
    code_sha: str,
    frames: list[SavedFrame],
    elapsed_s: float,
) -> tuple[list[str], list[dict]]:
    store = service_for(runtime).store
    lines: list[str] = []
    artifacts: list[dict] = []
    for frame in frames:
        try:
            try:
                table = pq.read_table(frame.path)
                card = store.save(
                    table,
                    produced_by="run_python",
                    source=f"{code_sha}:{frame.name}",
                    governed=False,
                    elapsed_s=elapsed_s,
                    notes=[_OFF_MODEL_NOTE],
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
        artifacts.append(
            {
                "handle_id": card.handle_id,
                "produced_by": "run_python",
                "governed": False,
            }
        )
    return lines, artifacts


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
    if res.restarted:
        _discard(res.saved)
        raise SandboxError(res.error or "sandbox restarted", hint=_LOST)
    if not res.ok:
        _discard(res.saved)
        trace = res.error or "unknown error"
        raise InvalidRequestError(
            "Python error:\n" + trace[-(MAX_CONTENT_CHARS - 600) :],
            hint=_CODE_ERROR_HINT,
        )
    code_sha = hashlib.sha256(code.encode()).hexdigest()[:16]
    lines, artifacts = _ingest(runtime, code_sha, res.saved, elapsed)
    parts = [res.stdout.rstrip()] if res.stdout.strip() else []
    if res.truncated:
        parts.append("(stdout truncated)")
    # Cards must survive the cap: bound stdout so the handle lines always fit.
    card_text = "\n".join(lines)
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
