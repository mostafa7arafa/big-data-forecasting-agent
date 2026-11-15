# Plan B2 rulings and deferred items

_Plan B2 (Docker CodeAct sandbox and compaction) was built on branch `feat/plan-b2-codeact`. This file keeps every decision made while executing it, the sandbox's known limits, and the deferred minors for later plans. The source is the SDD ledger, which is deleted after merge._

## Rulings

- Ruling R1: docker_argv adds `--user <host uid>:<host gid>` (non-root; fallback 10001 when host is root); image keeps USER 10001 default; Dockerfile sets HOME=/tmp, NUMBA_CACHE_DIR=/tmp, MPLCONFIGDIR=/tmp — read-only root + bind-mount ownership — if wrong, out_dir writes fail visibly in docker tests.
- Ruling R2: DockerSandbox rewrites saved paths from container /artifacts/out to host out_dir and rejects anything outside — tool needs host paths — cheap to change.
- Ruling R3: EOF/crash of the container process → ok=False, error "sandbox process exited (…); restarted, variables were lost", restarted=True — same contract as timeout — low.
- Ruling R4: escape cases the guard legitimately allows (huge stdout, infinite loop, numpy memory bomb, load() traversal) assert container/server behavior only; every other case asserts both — plan intent "both are tested" kept — low.
- Ruling R5: get_sandbox/DockerSandbox mkdir in_dir/out_dir before docker run (else docker creates root-owned dirs) — low.
- Ruling R6: polars 2.1.0 / plotly 7.1.0 pinned in Dockerfile though absent from uv.lock — not host deps; image-only — low.
- Ruling R7: reviewer minor "trailing-newline test uses literal backslash" rejected — the string is spliced into executed code so '\\n' becomes a real newline there — none.
- Ruling R8: sandbox out_dir = artifacts/sandbox/<slug+sha8 of thread_id>/<segment>; container gets --name analyst-sbx-<hex>, docker kill on timeout; no env passed — thread ids untrusted, no orphan containers — low.
- Ruling R9: Task 2 review minors 2 (uncaught docker-kill timeout / close_all partial), 3 (silent restart after dead container), 4 (dataset missing from out_dir), 5 (--memory-swap 2g) enter fix round 1 with Important #1 — they sit on the security boundary and are cheap — low.
- Ruling R10: import allowlist gains warnings, typing, dataclasses (harmless, models write warnings.filterwarnings before statsmodels) — guard is a usability layer, container is the boundary — if wrong, remove 3 strings.
- Ruling R11: Task 3 minors 1(input/help only blocked as calls/loads), 2 (RecursionError → InvalidRequestError), 4, 5, 6 go into one polish round — cheap, test honesty — low.
- Ruling R12: run_python content prints saved frames of ≤5 rows in full precision (repr), and the prompt says copy cited numbers exactly from that output; server pre-imports pandas as pd and numpy as np (model forgot import) — fixes observed live failure — if wrong, label gate stays low for python rows.
- Ruling R13: compaction moves to the OUTER graph: a node (or the top of the analyst node) that compacts state["messages"] before the analyst runs, only over prior turns (the current question and everything after stay in the tail); CompactionMiddleware logic is reused as a function, removed from analyst_middleware. In-turn growth stays with ClearOldToolResults — plan said "first in analyst_middleware", but the spec goal (long chats compacted without losing handles) needs the thread state to change — if wrong, very long single turns are not compacted.
- Ruling R14: a052 end date stays in the question, ungraded (peak average identifies the window uniquely); date checked by human/judge — low.
- Ruling R15: B2 baseline numbers in c60a721 are provisional — rerun after the fix wave (per-row thread ids); doc updated then — if not rerun, published python 5/5 may rest on shared-kernel state.
- Ruling R16: final-review minors 2 (chip names forecast not run_python), 5 (container runs as host uid), 6 (image not hash/digest pinned), 7 (compaction.py lives in middleware/), 8 (no cutoff python eval row) → followups doc, not fix wave — low.
- Ruling R17: final re-review minors parked to followups (no second fix wave): evicted-while-held sandbox starts untracked container until process exit (fix: closed flag, refuse run); replacement after LRU eviction lacks the variables-lost note; shared out_dir race in that case; evals/target.py reads inputs['id'] before try; in-call disk fill through bind mount (needs quota'd volume or bytes-over-protocol) — narrow, reaped at exit — low.

- Ruling R18: CodeRabbit "row echo returns raw rows" rejected — the ≤5-row echo is the model's own computed result saved via save(), capped and budgeted, not source records; it is what lets claims verify exactly (R12) — cost if wrong: a model could save() ≤5 raw source rows and see them (it already can print them in stdout).
- Ruling R19: CodeRabbit "regenerate uv.lock" rejected — uv.lock was updated in 81b55b5 and `uv lock --check` passes. "Dockerfile from lockfile" deferred (already in followups, R16).

## Security model: what the guard and the container each stop

The AST guard is a usability layer. It rejects disallowed imports, dunder access, `eval`/`exec`/`compile`, `open()` outside `/artifacts` and similar, with a hint the model can act on. **The container is the boundary.**

The container runs with:
- `--network none`;
- `--cap-drop ALL` and `no-new-privileges`;
- a read-only root;
- `--pids-limit 128`, 2 GB memory with no extra swap, and 2 CPUs;
- `--ulimit fsize` of 256 MB;
- `--log-driver none`;
- a non-root user;
- no environment variables passed in;
- no raw-data mount. It sees only the run's (dataset, cutoff) result segment, read-only, plus a per-thread scratch directory.

The escape suite runs 25 cases with the guard bypassed.

Attacks only the guard stops, which run unprivileged if the guard is bypassed: `ctypes`, `subprocess`/`os.system`, `importlib`, `eval`/`exec`, and walks over `__subclasses__`. With the guard bypassed they run as a non-root uid with no capabilities, no network and no secrets.

Known gaps in the guard, all accepted because the container bounds them:
- module attribute chains such as `pd.io.common.os.popen`, and aliasing such as `m = warnings.sys`;
- library file readers such as `pd.read_csv('/etc/passwd')` and duckdb `read_csv`;
- `str.format` attribute traversal;
- three-argument `type()` and `.mro()`;
- `typing.get_type_hints` and `ForwardRef`, and `dataclasses.make_dataclass`, which execute code indirectly.

Host-side handling of container output (`src/analyst/sandbox/docker.py`):
- Reads are bounded: protocol lines are limited to 256 KB, and only the last 8 KB of stderr is kept.
- Saved files are staged with `O_NOFOLLOW|O_NONBLOCK`, must be regular files with `nlink == 1`, and total at most 256 MB per response, copied into a host-only directory.
- The scratch directory is wiped on start and after every call.
- Containers are labelled with their owner, and orphans whose owner process has died are reaped.

Residual risks:
- Within one call, which lasts at most 30 s, a cell can write many files of up to 256 MB each into the scratch mount, because the mount has no quota. The fix is a quota'd volume, or returning `save()` bytes over the protocol and dropping the read-write mount.
- Every thread's container can `load()` any handle in the same (dataset, cutoff) segment. This is the B1 shared-store design: single user, public data.
- The container runs as the host user's uid (R1). A namespace escape would land as that uid. Rootless Docker or userns-remap is the follow-up.

## Deferred items

- (Task 2) stdin write unbounded for huge code (cap code size in run_python, T4)
- (Task 2) stderr tail read before drain → often empty crash message
- (Task 2) no disk quota on rw out_dir (host disk fill) → follow-ups
- (Task 2) staged files never cleaned up; reserve dataset names sandbox/sandbox-staging
- (Task 2) _to_result int(item["rows"])/str(item["name"]) outside try — malformed container line raises out of run() (fix in final wave)
- (Task 2) rejected non-regular scratch files stay in out_dir; staged copy not checked == st_size
- (Task 3) document guard gaps (module chains like pd.io.common.os.popen, library file readers, str.format attr traversal, 3-arg type) in followups.
- (Task 3) typing.get_type_hints/ForwardRef, dataclasses.make_dataclass, <mod>.sys aliasing are dunder-free guard gaps — document as guard limitation (container bounds them)
- (Task 4) claim text carried the 17-digit value verbatim ("0.08283779943279351") — prompt could let claim text round while values stay exact (verify already matches prose at stated precision).
- (Task 4) orphan "rows of X:" label when no row fits
- (Task 5) compaction trigger counts messages only (not system prompt/tool schemas) — document; metadata on compact node run unverified under real LangSmith; digest structured-output streams in stream_mode=messages (dashboard)
- (Final review) The off-model chip names `forecast`, not `run_python`, for a forecast over a run_python handle. Carry the input's off-model origin through.
- (Final review) The image build is not reproducible: pip has no `--require-hashes` and the base image is not pinned by digest.
- (Final review) `src/analyst/middleware/compaction.py` is no longer middleware after R13. Move it next to the graph nodes.
- (Final review) No Python eval row uses a cutoff, so cutoff-by-construction is not exercised end to end in the eval. Add one in Plan D.
- (Final re-review) A sandbox evicted from the LRU cap while a caller still holds it starts an untracked container on its next `run()`, which lives until process exit or the next reap. Fix: a closed flag, and refuse to run.
- (Final re-review) After an LRU eviction, the thread's next sandbox lacks the "variables were lost" note.
- (Final re-review) `evals/target.py` reads `inputs['id']` before its `try`.
- (CodeRabbit re-review) Per-instance scratch directories are removed only by `close()`. After a hard crash they persist, because `reap()` cleans containers, not directories.
- (CodeRabbit re-review) `pq.ParquetFile` in the footer checks is not closed explicitly.
- (CodeRabbit) The parquet footer caps (uncompressed bytes 512 MB, 5M rows, 20M cells) bound decoding, but not every pathological encoding.
