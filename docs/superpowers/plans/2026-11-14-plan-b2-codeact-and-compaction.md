# Plan B2: CodeAct Sandbox and Compaction, Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The analyst can run model-written Python against its own results in a locked-down Docker sandbox (`run_python`), proven by an escape-attempt suite, and long chats are compacted into an analytics digest without losing handles.

**Architecture:**
- **Sandbox:** a `Sandbox` protocol with a `DockerSandbox` implementation. Each thread gets one persistent container (`--network none`, `cap_drop ALL`, read-only root, CPU/memory/pid limits) running a small stdin/stdout exec server that keeps its variables between calls. The container sees only the result store of the run's (dataset, cutoff) as read-only, plus a scratch output directory.
- **Guard:** an AST guard rejects disallowed imports and calls before any code reaches the container.
- **Tool:** `run_python` returns stdout capped at 4 KB, and new handles via `save(df, name)`. Its answers are labelled Verified · off-model.
- **Compaction:** a compaction middleware replaces old history with a structured digest when the context nears a token threshold.

**Tech Stack:** Docker 29 (CLI via `subprocess`, no new Python SDK dependency); image `analyst-sandbox` built from `python:3.12-slim` with pinned duckdb 2.0.0, pandas, polars, numpy, scipy, statsmodels, statsforecast, plotly, pyarrow; LangChain v1 middleware; the existing pytest/LangSmith stack.

**Spec:**
- `docs/superpowers/specs/2026-11-12-plan-b-agent-harness-design.md` §11 (B2 outline);
- `docs/superpowers/specs/2026-11-07-taxi-analytics-agent-design.md` §5.3 (`run_python`), §5.5 (context management, compaction), §5.6 (sandbox and guards);
- B1 decisions: `docs/superpowers/plans/2026-11-14-plan-b1-followups.md`.

## Global Constraints

- **`src/analyst/` stays dataset-agnostic.** The dataset-agnostic test applies to the sandbox code too.
- **The sandbox is a security boundary on top of the engine boundary, and never loosens it:**
  - `--network none`, `--cap-drop ALL`, `--read-only`, `--security-opt no-new-privileges`;
  - limits: `--pids-limit 128`, `--memory 2g`, `--cpus 2`;
  - a non-root user;
  - a `tmpfs` at `/tmp` (64 MB).
- **The cutoff is honoured by construction** [call, refines spec §11 "/data:ro honouring the cutoff"]:
  - the container gets **no raw-data mount**;
  - it mounts only the result-store directory of the run's (dataset, cutoff) read-only at `/artifacts/in`, plus a per-thread scratch directory read-write at `/artifacts/out`;
  - every value the sandbox can see therefore already passed through the cutoff-enforcing engine.
- **Per-call timeout `ANALYST_SANDBOX_TIMEOUT_S=30`.** On timeout the container is killed and restarted, and the tool says the session state was lost.
- **Output caps:** stdout and tracebacks are capped at 4,096 chars; DataFrames print as compact summaries; tool content is ≤ 4,096 chars (the B1 `run_tool` cap).
- **Handles saved from Python** have `produced_by="run_python"` and `governed=False`, so answers that rely on them are labelled **Verified · off-model**.
- **AST guard:**
  - import allowlist: `duckdb, pandas, polars, numpy, scipy, statsmodels, statsforecast, plotly, math, statistics, datetime, json, re, collections, itertools, functools`;
  - blocklist, rejected with a clear error message: `subprocess`, `socket`, `os.system`, `ctypes`, `importlib`, `__import__`, `eval`, `exec`, `compile`, `open` outside `/artifacts`, `sys.modules`, dunder attribute access (`__subclasses__`, `__globals__`, `__builtins__`, `__code__`);
  - the guard is a usability layer; **the container is the real boundary** (spec §5.6). Both are tested.
- **Docker-dependent tests** carry the marker `docker`. `make test` excludes it; `make test-docker` runs it.
- **Compaction digest:** goal; confirmed findings with their numbers and handles; live handles; open questions; user preferences (project spec §5.5.6). Handle ids survive compaction verbatim.
- **Delegates never commit.** The controller gates, reviews and commits.

## Review Focus

1. **A Python result that is wrong but looks like evidence.** The model saves a DataFrame it computed itself. The handle is real, so verify traces the number, and the label must say **off-model**, never plain Verified. Owned by Task 4 (`test_run_python_handles_are_off_model`).
2. **A long-running or hung cell** (an infinite loop, or `time.sleep(999)`). The tool returns a timeout error within ~30 s, and the next call works in a fresh container with a "state lost" note. Owned by Task 2 (`test_timeout_restarts_and_reports_state_loss`).
3. **Huge output** (`print(df)` on 1M rows, or a 50 KB traceback). The model sees ≤ 4,096 chars with a truncation marker, never the raw dump. Owned by Task 2 (`test_output_capped`).
4. **A thread that runs under one cutoff and later under another, or two threads at once.** Each thread's container mounts only its own (dataset, cutoff) segment, so one cutoff never reads another's files. Owned by Task 2 (`test_mount_is_segment_scoped`).
5. **Compaction drops a handle a later claim needs.** After compaction, verify still resolves handles named in the digest, and the answer to a follow-up question can cite them. Owned by Task 5 (`test_handles_survive_compaction`).

---

## File map

| Path | Responsibility | Task |
|---|---|---|
| `docker/sandbox/Dockerfile`, `docker/sandbox/server.py` | sandbox image; stdin/stdout exec server with persistent globals, `load`/`save`, output capping | 1 |
| `Makefile`, `pyproject.toml` | `sandbox-image`, `test-docker` targets; `docker` marker | 1 |
| `src/analyst/sandbox/base.py` | `Sandbox` protocol, `ExecResult` | 2 |
| `src/analyst/sandbox/docker.py` | `DockerSandbox` (one container per thread, restart on timeout), `get_sandbox()` | 2 |
| `src/analyst/sandbox/ast_guard.py` | `check_code()` | 3 |
| `src/analyst/tools/run_python.py`, `tools/__init__.py`, `prompts.py` | the tool, its ingest of saved frames into the result store, prompt policy | 4 |
| `src/analyst/middleware/compaction.py`, `middleware/__init__.py` | `CompactionMiddleware` + `Digest` | 5 |
| `tests/escape/test_escape.py` | escape-attempt suite (marker `docker`) | 3 |
| `evals/datasets/agent_single_turn.jsonl`, `evals/gold/a05*.parquet`, `docs/results/agent-baseline.md` | Python eval rows, re-baselined | 6 |

---

### Task 1: Sandbox image and exec server

**Files:**
- Create: `docker/sandbox/Dockerfile`, `docker/sandbox/server.py`
- Modify: `Makefile`, `pyproject.toml` (marker `docker`), `.env.example` (`ANALYST_SANDBOX_IMAGE=analyst-sandbox:1`, `ANALYST_SANDBOX_TIMEOUT_S=30`)
- Test: `tests/unit/test_sandbox_server.py` (runs `server.py` in-process on the host, no Docker)

**Interfaces:**
- **Produces**, the server protocol:
  - one JSON object per line on stdin: `{"id": str, "code": str}`;
  - one JSON line on stdout per request: `{"id": str, "ok": bool, "stdout": str, "error": str | null, "saved": [{"name": str, "path": str, "rows": int}], "truncated": bool}`.
- **Globals persist between requests.** Two helpers are pre-bound:
  - `load(handle_id: str) -> pandas.DataFrame` reads `/artifacts/in/<handle_id>.parquet`, and only ids matching `^r_[0-9a-f]{10}$`;
  - `save(df, name: str) -> str` writes `/artifacts/out/<slug(name)>-<n>.parquet` and returns that path; `name` is slugged to `[a-z0-9_-]{1,40}`.
- **Output capping:** stdout and traceback are capped at 4,096 chars with `"… (truncated)"`. A bare DataFrame expression prints `df.head(10)` plus its shape.
- **Paths:** `ANALYST_SANDBOX_IN` and `ANALYST_SANDBOX_OUT` environment variables override `/artifacts/in` and `/artifacts/out`, for host tests.
- **Image:**
  - `FROM python:3.12-slim`;
  - pinned package versions taken from `uv.lock` (`duckdb==2.0.0` and the lock's pandas, polars, numpy, scipy, statsmodels, statsforecast, pyarrow), plus plotly;
  - `USER 10001`, `WORKDIR /tmp`;
  - `ENTRYPOINT ["python", "-u", "/srv/server.py"]`.
- **Make targets:**
  - `make sandbox-image` = `docker build -t analyst-sandbox:1 docker/sandbox`;
  - `make test-docker` = `uv run pytest -m docker`;
  - `make test` adds `and not docker`.

- [ ] **Step 1: Write failing tests**

```python
def test_state_persists_between_requests(server): run(server, "x = 41"); assert run(server, "print(x + 1)")["stdout"] == "42\n"
def test_load_and_save_roundtrip(server, handle_parquet):   # load(h) → df; save(df, "My Result!") → path under OUT, rows reported
def test_load_rejects_bad_ids(server): assert not run(server, "load('../etc/passwd')")["ok"]
def test_output_capped(server): r = run(server, "print('x'*50000)"); assert len(r["stdout"]) <= 4096 and r["truncated"]
def test_traceback_is_error_not_crash(server): r = run(server, "1/0"); assert not r["ok"] and "ZeroDivisionError" in r["error"]
def test_bare_dataframe_prints_head_and_shape(server): ...
```

- [ ] **Step 2: Run** `uv run pytest tests/unit/test_sandbox_server.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement `server.py`** (standard library + pandas only; `exec` against one persistent dict) and the Dockerfile.
- [ ] **Step 4: Run** `make test && make lint`, then `make sandbox-image`. Expected: tests pass, and the image builds.
- [ ] **Step 5: Commit** `feat: sandbox image and persistent exec server`.

### Task 2: `Sandbox` protocol and `DockerSandbox`

**Files:**
- Create: `src/analyst/sandbox/__init__.py`, `base.py`, `docker.py`
- Modify: `src/analyst/config.py` (`sandbox_image: str = "analyst-sandbox:1"`, `sandbox_timeout_s: float = 30.0`)
- Test: `tests/unit/test_sandbox_docker.py`, with marker `docker` except the argv test

**Interfaces:**
- **`ExecResult`** (a frozen dataclass): `ok: bool`, `stdout: str`, `error: str | None`, `saved: list[SavedFrame]`, `truncated: bool`, `restarted: bool`.
- **`SavedFrame`**: `name: str`, `path: Path` (host path), `rows: int`.
- **`class Sandbox(Protocol)`**:
  - `def run(self, code: str) -> ExecResult`
  - `def close(self) -> None`
- **`DockerSandbox(*, in_dir: Path, out_dir: Path, image: str, timeout_s: float)`**: `run` and `close` per the protocol.
  - It starts `docker run -i --rm` lazily on first `run`, with exactly the flags in Global Constraints.
  - Mounts: `-v {in_dir}:/artifacts/in:ro` and `-v {out_dir}:/artifacts/out:rw`.
  - It keeps the process; each `run` writes one request line and reads one response line.
  - On timeout it kills the process, restarts lazily, and returns `ok=False, error="timeout after 30 s; sandbox restarted, variables were lost", restarted=True`.
- **`docker_argv(in_dir, out_dir, image) -> list[str]`**: a pure function that builds the argv, unit-testable without Docker.
- **`get_sandbox(thread_id: str, dataset: str, cutoff: date | None) -> Sandbox`**:
  - a process-wide registry with a lock, keyed by `(thread_id, dataset, cutoff)`;
  - `in_dir` = that segment's `ResultStore` root;
  - `out_dir` = `settings.artifacts_dir / "sandbox" / <thread_id>`;
  - `close_all()` closes every sandbox.
- **Blocking:** `run` is blocking. Async callers use `asyncio.to_thread` (the B1 rule).

- [ ] **Step 1: Write failing tests**

```python
def test_argv_has_every_hardening_flag():   # no Docker needed
    argv = docker_argv(Path("/a"), Path("/b"), "img")
    for flag in ["--network", "none", "--cap-drop", "ALL", "--read-only", "--pids-limit", "128",
                 "--memory", "2g", "--cpus", "2", "no-new-privileges"]: assert flag in argv
    assert "/a:/artifacts/in:ro" in " ".join(argv) and not any("/data" in a for a in argv)

@pytest.mark.docker
def test_state_persists(tmp_dirs): ...
@pytest.mark.docker
def test_timeout_restarts_and_reports_state_loss(tmp_dirs):   # timeout_s=2, "while True: pass" → ok False, restarted; next run "print(1)" ok; earlier vars gone
@pytest.mark.docker
def test_output_capped(tmp_dirs): ...
@pytest.mark.docker
def test_mount_is_segment_scoped(tmp_path):   # two get_sandbox with different cutoffs → each sees only its own in_dir files
```

- [ ] **Step 2: Run** `uv run pytest tests/unit/test_sandbox_docker.py -v -m "docker or not docker"`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `make test && make lint && make test-docker`. Expected: pass.
- [ ] **Step 5: Commit** `feat: DockerSandbox with per-thread persistent kernel`.

### Task 3: AST guard and escape-attempt suite

**Files:**
- Create: `src/analyst/sandbox/ast_guard.py`, `tests/escape/__init__.py`, `tests/escape/test_escape.py`
- Test: `tests/unit/test_ast_guard.py`

**Interfaces:**
- **`check_code(code: str) -> None`**:
  - raises `InvalidRequestError(message, hint)` on a syntax error, a disallowed import (the hint lists the allowlist), a blocked call or name, or dunder attribute access;
  - `open(...)` is allowed only with a literal path starting `/artifacts/`.
- **Escape suite** (marker `docker`):
  - each case **bypasses the guard** (it calls `DockerSandbox.run` directly) and asserts the container still stops it;
  - each case also asserts the guard rejects it at the tool level;
  - at least these 20 cases:

| Case | Expected outcome |
|---|---|
| network connect | refused |
| DNS lookup | refused |
| read `/etc/shadow` | refused |
| write `/usr` | fails (read-only root) |
| write `/artifacts/in` | fails (read-only) |
| list `/` for a data mount | no `/data` |
| fork bomb | pids-limit hit, container survives or restarts |
| memory bomb | OOM → error, sandbox restarts |
| infinite loop | timeout |
| `ctypes` | blocked |
| `subprocess` / `os.system` | blocked |
| `__subclasses__` walk to builtins | blocked |
| `importlib` | blocked |
| `eval` / `exec` / `compile` strings | blocked |
| huge stdout | capped |
| setuid / capability probe (`os.setuid(0)`) | fails |
| `/proc/1/environ` holding host secrets | absent (no env passed) |
| path traversal in `load()` | rejected |
| reading another cutoff's segment | not mounted |
| writing outside `/artifacts/out` | fails |

- [ ] **Step 1: Write failing tests**: unit guard cases (allowed: pandas/duckdb on `load()` handles, plotting; rejected: each blocklist item with its hint) and the escape suite.
- [ ] **Step 2: Run** `uv run pytest tests/unit/test_ast_guard.py -v` and `make test-docker`. Expected: FAIL.
- [ ] **Step 3: Implement `check_code`** by walking the AST, with no regex on source text.
- [ ] **Step 4: Run** `make test && make lint && make test-docker`. Expected: pass. Record the suite's case count in the report.
- [ ] **Step 5: Commit** `feat: AST guard and sandbox escape-attempt suite`.

### Task 4: `run_python` tool

**Files:**
- Create: `src/analyst/tools/run_python.py`
- Modify: `src/analyst/tools/__init__.py` (add to `DATA_TOOLS`, which therefore reaches the worker allowlist too), `src/analyst/prompts.py` (when to use Python), `src/analyst/worker.py` (worker prompt mention)
- Test: `tests/unit/test_run_python.py` (fake `Sandbox`), `tests/unit/test_run_python_docker.py` (marker `docker`)

**Interfaces:**
- **Consumes:**
  - `check_code`, `get_sandbox`, `ExecResult`;
  - the B1 `run_tool`/`data_tool` wrapper (errors, 4,096 cap, progress);
  - `ResultStore.save(table, produced_by="run_python", source=<code sha + name>, governed=False, elapsed_s=..., notes=[...])`.
- **Tool `run_python(code: str, runtime)`:**
  1. runs `check_code`;
  2. runs `get_sandbox(thread_id, ctx.dataset, ctx.cutoff).run(code)` via the wrapper; the thread id comes from `runtime.config["configurable"]["thread_id"]`, falling back to `"default"`;
  3. for every saved frame, reads the parquet from the host `out_dir`, saves it to the run's `ResultStore` as above, and deletes the scratch file;
  4. returns content = capped stdout, plus one card line per new handle (`handle: r_…  rows: N  columns: …`).
- **Artifacts:**
  - one new handle → `{"handle_id", "produced_by": "run_python", "governed": False}`;
  - several handles → `{"handles": [...]}`. Extend `collect_handles` in `src/analyst/verify/trace.py` to read the `handles` list form.
- **Errors:** guard and timeout failures become tool errors with hints.
- **Prompt policy:**
  - use `run_python` only for analysis the typed tools can't express (correlations, rolling windows, custom statistics, reshaping);
  - always `load()` handles from earlier tools;
  - `save()` any table you will cite;
  - numbers you cite must come from saved handles.

- [ ] **Step 1: Write failing tests**

```python
def test_guard_rejection_is_tool_error(): ...            # "import socket" → status error, HINT lists allowlist
def test_saved_frame_becomes_handle(fake_sandbox): ...   # artifact produced_by run_python, governed False, card in content
def test_run_python_handles_are_off_model(): ...          # verify on a claim citing that handle → verified_off_model, off_model_tools == ["run_python"]
def test_thread_id_scopes_sandbox(): ...
def test_tool_in_worker_allowlist(): ...
@pytest.mark.docker
def test_end_to_end_load_compute_save(real_store):        # query_metrics handle → load → rolling mean → save → new handle readable
```

- [ ] **Step 2: Run** `uv run pytest tests/unit/test_run_python.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `make test && make lint && make test-docker`. Expected: pass. The OpenAI schema test (B1) still passes for every tool.
- [ ] **Step 5: Controller live check.** Run `make sandbox-image`, then `make dev`, then ask "What is the correlation between daily trips and average fare in 2025?". Expected: `query_metrics` → `run_python` (load, corr, save), with the answer labelled **Verified · off-model**.
- [ ] **Step 6: Commit** `feat: run_python CodeAct tool over the sandbox`.

### Task 5: Compaction middleware

**Files:**
- Create: `src/analyst/middleware/compaction.py`
- Modify: `src/analyst/middleware/__init__.py` (insert first in `analyst_middleware`), `src/analyst/config.py` (`compaction_trigger_tokens: int = 60_000`, `compaction_keep_messages: int = 8`)
- Test: `tests/unit/test_compaction.py`

**Interfaces:**
- **`Digest`** (pydantic):
  - `goal: str`;
  - `findings: list[Finding]`, where `Finding` has `text: str`, `values: list[float]`, `handles: list[str]`;
  - `live_handles: list[str]`;
  - `open_questions: list[str]`;
  - `preferences: list[str]`.
- **`CompactionMiddleware(trigger_tokens: int, keep_messages: int)`**:
  - in `before_model` (and `abefore_model`), when the approximate token count of `state["messages"]` exceeds `trigger_tokens`, it summarises everything except the last `keep_messages`;
  - the summary uses the run's model with `with_structured_output(Digest)`; this extra model call is the only one in B2;
  - it replaces the old messages with one `HumanMessage` built by `add_notification_metadata(..., source="compaction")` whose content renders the digest as Markdown;
  - the summarised slice must end on a complete AI/tool pair, so no `tool_call` is left without its ToolMessage;
  - `live_handles` is the union of the handles the model listed and every handle id regex-found in the removed messages, so ids survive verbatim even if the model omits one;
  - LangSmith metadata `compacted=True` and `compacted_messages=N`.
- **Verify:** `collect_handles` treats handle ids in a `compaction`-sourced message as known, resolved through the store like worker handles.

- [ ] **Step 1: Write failing tests**

```python
def test_no_compaction_below_trigger(): ...
def test_compacts_above_trigger_keeps_tail_and_pairs(): ...   # scripted model returns a Digest; tail of 8 kept; no orphan tool_calls
def test_handles_survive_compaction(): ...                     # a handle only mentioned in removed messages appears in the digest message; verify resolves it
def test_digest_failure_falls_back_to_no_compaction(): ...    # model error → messages unchanged, run continues
```

- [ ] **Step 2: Run** `uv run pytest tests/unit/test_compaction.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** against the installed langchain 1.5 middleware hooks; check how `before_model` replaces messages (`RemoveMessage` / `REMOVE_ALL_MESSAGES`).
- [ ] **Step 4: Run** `make test && make lint`. Expected: pass.
- [ ] **Step 5: Commit** `feat: compaction middleware with analytics digest`.

### Task 6: Python eval rows, baseline, docs

**Files:**
- Modify: `evals/datasets/agent_single_turn.jsonl` (add `a051`–`a055`, category `python`), `scripts/build_gold.py` (the category is accepted), `evals/graders/agent.py` (only if a new check name is needed), `docs/results/agent-baseline.md`, `README.md` (roadmap: B2 checked, plus one line in "How it's built" about the sandbox)
- Test: `tests/unit/test_agent_graders.py` (category plumbing), `tests/integration/test_agent_gold.py` (new rows' gold)

**Interfaces:**
- **Five rows** (gold from `gold_sql` on the full data; `expected_tools` include `run_python`; `expected_label` `verified_off_model`):
  - the Pearson correlation of daily trips and daily average fare in 2025;
  - the peak 7-day rolling average of daily trips in 2025 and its end date;
  - the coefficient of variation of monthly trips by borough over 2024, for the top 3 boroughs;
  - the median day-over-day % change in daily trips, January–March 2026;
  - the share of 2025 days whose trips exceeded the 2025 daily mean plus one standard deviation.
- **Total rows:** 55. The dataset upload creates a new version tag.

- [ ] **Step 1: Author the 5 rows and gold SQL.** The controller shows the owner the 5 new questions before freezing.
- [ ] **Step 2: Run** `uv run python scripts/build_gold.py --set agent` and `make test-taxi`. Expected: 55 rows OK, 0 orphans, gold tests pass.
- [ ] **Step 3: Controller run** `make eval-agent` (needs the sandbox image). Update `docs/results/agent-baseline.md` with a B2 section: overall plus the `python` category, label distribution, and whether compaction ever triggered (the LangSmith `compacted` metadata).
- [ ] **Step 4: Commit** `feat: Python eval rows and B2 baseline`.

---

## After the last task

- Run a whole-branch review on Opus, focused on security: sandbox flags, mounts, the guard, and cutoff by construction. Then one fix wave and a re-review.
- Push, open a PR, run CodeRabbit once, and merge on the owner's word.
- File the rulings into `docs/superpowers/plans/<date>-plan-b2-followups.md` and delete the ledger.
