# 02 — LangChain / LangGraph + CodeAct stack research

_Researched 2026-11-07. Versions are from PyPI on that date. Items marked **[unverified]** were not confirmed against a primary source._

| Package | Version (released) | Status |
|---|---|---|
| `langchain` | 1.4.9 (2026-11-04) | active |
| `langchain-core` | 1.7.6 | active |
| `langgraph` | 1.2.20 (2026-11-05) | active |
| `langgraph-cli` | 0.4.42 | active |
| `langchain-openai` | 1.7.4 | active |
| `langchain-openrouter` | 0.2.9 (2026-09-22) | active, first-party (lives in the `langchain-ai/langchain` monorepo) |
| `langsmith` | 0.14.17 | active |
| `openevals` | 0.2.2 (2026-10-22) | active |
| `agentevals` | 0.0.9 (2025-07-24) | low activity; repo last pushed 2026-07 |
| `langgraph-codeact` | 0.1.3 (2025-04-25) | **ARCHIVED** (GitHub, read-only) |
| `langchain-sandbox` | 0.0.6 (2025-05-21) | **ARCHIVED, "no longer maintained"** |
| `deepagents` | 0.7.34 | very active |
| `smolagents` | 1.26.0 (2026-05-29) | active |
| `e2b-code-interpreter` | 2.11.9 | active |
| `duckdb` | 2.0.0 (2026-10-21) | new major version, so check for breaking changes |
| `statsforecast` | 2.1.1 | active |

---

## 1. LangChain v1 `create_agent`

`langchain.agents.create_agent` is the standard agent in v1. It is a tool-calling (ReAct-style) loop that **compiles to a LangGraph graph**, so you get checkpointing, streaming, interrupts, Studio and LangSmith tracing for free. It replaces `langgraph.prebuilt.create_react_agent`. You customise it through **middleware** rather than by editing graph nodes. Docs: https://docs.langchain.com/oss/python/langchain/agents

**Middleware** (https://docs.langchain.com/oss/python/langchain/middleware/built-in). Built-in options include tool error handling, tool retry, model retry, model fallback, `SummarizationMiddleware`, `HumanInTheLoopMiddleware`, model call limit, tool call limit, `PIIMiddleware`, to-do list, LLM tool selector, provider tool search, shell tool, filesystem, subagent, rubric grading (beta), file search, context editing and LLM tool emulator.

**Custom middleware** (https://docs.langchain.com/oss/python/langchain/middleware/custom). There are two forms:
- **Decorators.** Node-style: `@before_agent`, `@before_model`, `@after_model`, `@after_agent`. Wrap-style: `@wrap_model_call`, `@wrap_tool_call`. There is also `@dynamic_prompt`.
- **Classes.** Subclass `AgentMiddleware` with sync and async hooks. A middleware can also add state keys and tools.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import (SummarizationMiddleware, HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware, ToolCallLimitMiddleware, wrap_tool_call)
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model=llm, tools=[run_sql, describe_table, forecast, make_chart],
    system_prompt=SYSTEM,
    response_format=AnalystAnswer,           # pydantic → ToolStrategy / ProviderStrategy auto-picked
    middleware=[
        SummarizationMiddleware(model=cheap_llm, trigger=("tokens", 40_000), keep=("messages", 20)),
        ModelCallLimitMiddleware(run_limit=25),   # param names: check reference [unverified]
        HumanInTheLoopMiddleware(interrupt_on={"run_python": {"allowed_decisions": ["approve", "reject"]}}),
    ],
    checkpointer=InMemorySaver(),
)
```

**Structured output** (https://docs.langchain.com/oss/python/langchain/structured-output). `response_format` accepts `ToolStrategy[T]`, `ProviderStrategy[T]` (the provider's native JSON schema output), or a bare type, in which case the strategy is picked automatically. The result ends up in `result["structured_response"]`.

**When to use a custom `StateGraph` instead.** Write your own graph when you need explicit, deterministic topology. For this project that means:
- a router that sends work to a SQL path, a code path or a forecast path;
- a planner/executor split;
- parallel branches (`Send`);
- typed state carrying artifacts such as `result_ids`, `charts` and `forecast_metrics`;
- subgraphs.

A good pattern is a **custom StateGraph whose nodes are `create_agent` subagents**. You keep the middleware and also get a graph that is visible in Studio.

## 2. CodeAct in the LangChain ecosystem

**The paper.** Wang et al., *Executable Code Actions Elicit Better LLM Agents* (ICML 2024), https://arxiv.org/abs/2402.01030. It claims that using Python as the unified action space beats JSON/text tool calls. On its M3ToolEval benchmark it reports **up to ~20% higher success rate** and **up to ~30% fewer actions** across 17 LLMs. The paper also released the CodeActInstruct dataset and the fine-tuned CodeActAgent model. The gains come from composition (loops, variables, chaining tools in a single step) and from self-debugging on tracebacks.

**`langgraph-codeact`** (https://github.com/langchain-ai/langgraph-codeact) offers `create_codeact(model, tools, eval_fn, prompt=None)`, which returns a `StateGraph` that you `.compile(checkpointer=...)`. `eval_fn(code: str, _locals: dict) -> (output: str, new_vars: dict)` is a sandbox you supply yourself. Variables persist between turns. `create_default_prompt(tools)` renders tools as Python function signatures. **Status: archived, with the last commit in May 2025.** The README says it was tested only with Claude 3.7, and its example `exec` sandbox is "not safe". **Do not depend on it.** It is about 300 lines, though, so vendoring or re-implementing the pattern is easy.

**LangChain's 2026 direction.** LangChain now runs code inside the agent loop as **middleware**. Deep Agents 0.6 (May 2026) added a code interpreter: `CodeInterpreterMiddleware` from `langchain_quickjs`, installed with `deepagents[quickjs]`. It also added **programmatic tool calling (PTC)**, where allowlisted tools appear as async functions inside the interpreter. LangChain reports up to 35% fewer tokens on some tasks (https://www.langchain.com/blog/give-your-agents-an-interpreter). **Caveat: the interpreter runs JavaScript/TypeScript in QuickJS, not Python.** It cannot use DuckDB, pandas or StatsForecast, so it is a poor fit for a data agent.

**Combining ReAct and CodeAct in one graph.** Two options:
1. **Code as one tool (recommended).** A `run_python(code)` tool sits next to typed tools such as `run_sql`, `describe_schema`, `forecast` and `plot`. A single `create_agent` handles both styles: typed tools cover common paths cheaply and reliably, and `run_python` is the escape hatch for multi-step analysis. Wrap it in HITL middleware or a tool call limit. Persist interpreter state in the sandbox session, not in LLM context.
2. **Router node.** `classify → {react_agent | codeact_subgraph} → synthesize`. The CodeAct subgraph is a hand-rolled loop: the model emits code, the sandbox runs it, and stdout or errors go back to the model. This is easier to show off in Studio and to evaluate per branch, but it adds an extra LLM call and routing errors.

Option 1 is simpler and is what most 2026 agents converge on (Anthropic, Cloudflare Code Mode and Deep Agents PTC are all "code as a tool or middleware"). For the portfolio, option 2 can be layered on top as a visible StateGraph.

**smolagents `CodeAgent`** (https://huggingface.co/docs/smolagents/tutorials/secure_code_execution) is the most mature CodeAct-first framework. The model writes Python every step, and tools are Python functions. `executor_type` can be `"local"` (an AST interpreter with an import allowlist and an op-count cap), `"e2b"`, `"modal"`, `"docker"`, `"blaxel"` or `"wasm"`. Its own docs warn that "no local python sandbox can ever be completely secure." It is a good comparison point and a source of ideas (the AST executor). Mixing it into LangGraph gives you two agent runtimes and two tracing models, though, so don't adopt it as the core.

## 3. Sandboxing model-written Python

| Option | Isolation | Local Parquet/DuckDB? | Notes |
|---|---|---|---|
| In-process `exec` | none | yes | Demo only. |
| smolagents `LocalPythonExecutor` (AST) | weak | yes, if `duckdb` is allowlisted | Usable as a library on its own; DuckDB can still read or write any file it can reach. |
| `langchain-sandbox` (Pyodide/Deno) | WASM | **no**: no native DuckDB wheel, and file access needs Deno permissions | **Unmaintained** (README: "no longer maintained… not recommended for production"). |
| **Docker / Podman container** (or a persistent Jupyter kernel inside it) | good (namespaces; add `--network none`, `cap_drop ALL`, mem/cpu/pids limits, `--read-only`) | **yes**: bind-mount `data/` **read-only** | Local-first, free, reproducible. Best fit. |
| E2B (`e2b-code-interpreter` 2.11.9) | Firecracker microVM | must upload data, and 140M rows of Parquet is impractical | Good for demos with small extracts; paid, cloud. |
| Daytona / Modal | container or VM, cloud | Modal Volumes / Daytona volumes could hold the Parquet | Cloud dependency; Modal is good if you want GPU or scale. |
| LangSmith Sandboxes (GA May 2026) | microVM, snapshots, pause-when-idle | cloud, so data must be shipped | https://www.langchain.com/blog/langsmith-sandboxes-generally-available; tightly integrated with Deep Agents. Pricing and limits: [unverified]. |

**Recommendation for a local-first project.** Use a **long-lived Docker sandbox** per session: `python:3.12-slim` plus duckdb, polars/pandas, statsforecast and matplotlib/plotly. Mount the Parquet at `/data:ro` and a scratch dir at `/artifacts:rw`. Run code through a small exec server or a Jupyter kernel (`jupyter_client`) so variables persist between calls, as CodeAct needs. Set `--network none` and a timeout of about 60 s. Hide this behind a `Sandbox` protocol (`run(code) -> ExecResult`) so E2B or LangSmith Sandboxes can be swapped in for a hosted demo. The typed `run_sql` tool can run DuckDB in-process but must be **read-only**: open the database with `duckdb.connect(read_only=True)`, set `SET enable_external_access` / allowed directories, and reject DDL/COPY. Check the exact DuckDB 2.0 lock-down settings [unverified].

**Keeping 140M rows out of the context window:**
- Tools return **handles plus previews**, not data. For example, `run_sql` writes the result to `/artifacts/r_7.parquet` and returns `{result_id, row_count, columns+dtypes, first 10 rows (markdown), summary stats}`.
- Later tools take `result_id` (for example `forecast(result_id, ts_col, y_col, h=28)` or `plot(result_id, spec)`), and code in the sandbox reads `/artifacts/r_7.parquet`.
- Charts are saved to files, and the state stores paths. Only a caption and the path go back to the LLM.
- Truncate stdout (around 4 KB) and tracebacks. Enforce `LIMIT` on previews and push `SUMMARIZE`/aggregation down into DuckDB.
- Give the model the schema and a data dictionary in the system prompt, plus a `describe_table` tool, instead of sample dumps.

## 4. OpenAI and OpenRouter models

- **OpenAI:** `ChatOpenAI(model=...)` from `langchain-openai`, or `init_chat_model("openai:...")`.
- **OpenRouter (first-party package):** `from langchain_openrouter import ChatOpenRouter`, which reads `OPENROUTER_API_KEY`. Parameters include `openrouter_provider={...}` (`order`, `only`, `sort`, `data_collection`, `allow_fallbacks`, …), `route`, `reasoning={"effort": ...}`, `app_url`/`app_title`, and `usage_metadata` with reasoning and cache breakdowns (https://docs.langchain.com/oss/python/integrations/chat/openrouter). This is preferable to the older workaround of `ChatOpenAI(base_url="https://openrouter.ai/api/v1", api_key=...)`. That workaround still works for basic chat and tools, but it drops OpenRouter-specific fields and reasoning details.
- **Cost in LangSmith** (https://docs.langchain.com/langsmith/cost-tracking). LangSmith computes cost from token counts × a **model pricing map**, matched by regex on `ls_model_name` and optionally `ls_provider`. Built-in prices cover OpenAI and other major providers. **OpenRouter model IDs such as `openai/gpt-…` will probably have no price entry.** You can either:
  - (a) add pricing-map rows for the models you use, or
  - (b) send exact costs per run with `usage_metadata` fields `input_cost`, `output_cost` (and optionally `total_cost`). Option (b) fits OpenRouter, which reports the actual charge in its response `usage` (field: `usage.cost` [unverified for `ChatOpenRouter` passthrough]). One way to attach it is a small `@wrap_model_call` middleware.

  Price changes are **not backfilled** to traces already logged. Tool runs can also carry `total_cost`, which is useful for E2B or sandbox seconds.

## 5. openevals (and agentevals)

The README is at https://github.com/langchain-ai/openevals. Every evaluator takes `(inputs, outputs, reference_outputs)` and returns `{key, score, comment}`.
- **LLM-as-judge:** `create_llm_as_judge(prompt=..., model="openai:...", feedback_key=..., continuous/choices/output_schema)`. Prebuilt prompts include `CORRECTNESS_PROMPT`, `CONCISENESS_PROMPT`, hallucination, the RAG prompts (`RAG_GROUNDEDNESS_PROMPT`, helpfulness, retrieval relevance), safety and security prompts (`PII_LEAKAGE_PROMPT`, …), and multimodal (image) prompts.
- **Deterministic:** exact match, Levenshtein, embedding similarity, and `create_json_match_evaluator` (per-key, with an aggregator, optionally an LLM per key).
- **Code:** `create_pyright_evaluator`, `create_mypy_evaluator`, `create_code_llm_as_judge`, plus E2B-sandboxed `create_e2b_pyright_evaluator` / `create_e2b_execution_evaluator`.
- **Agent trajectory (now in openevals):** `create_trajectory_match_evaluator(trajectory_match_mode="strict"|"unordered"|"subset"|"superset", tool_args_match_mode=...)`, `create_trajectory_llm_as_judge(prompt=TRAJECTORY_ACCURACY_PROMPT[_WITH_REFERENCE])`, and conversation and trajectory prompts.
- **Multi-turn simulation:** `run_multiturn_simulation` with prebuilt or custom simulated users, including LangGraph integration.
- **agentevals** (https://github.com/langchain-ai/agentevals) still holds the **LangGraph graph-trajectory** evaluators (node-level paths). Its last release was 2025-07, so treat it as low-maintenance.
- **No SQL-specific evaluator exists.** Write custom ones in the openevals interface.

**LangSmith integration.** There are two routes:
- **pytest:** `@pytest.mark.langsmith`, with `langsmith.testing as t` providing `t.log_inputs/log_outputs/log_reference_outputs`. Evaluators called inside the test log feedback automatically.
- **`Client().evaluate(target, data="dataset", evaluators=[...])`.**

Custom evaluators should wrap their logic in `_run_evaluator` so results log correctly.

**SQL answer by result set (custom evaluator):**
```python
def sql_result_match(*, inputs, outputs, reference_outputs, **_):
    got = con.sql(outputs["sql"]).df(); exp = con.sql(reference_outputs["gold_sql"]).df()
    got, exp = normalize(got), normalize(exp)  # sort rows/cols, round floats, ignore col names
    return {"key": "result_set_match", "score": got.equals(exp)}
```
Compute the gold results once and cache them as Parquet, because re-running against 140M rows on every eval is slow. Add a relaxed variant that checks numbers within a tolerance, a `create_trajectory_match_evaluator(mode="superset")` check that `run_sql` was called, and an LLM judge for the narrative.

**Forecast by metrics.** Use a fixed holdout (for example the last 4 weeks). The evaluator recomputes MASE/sMAPE/RMSE of the agent's forecast (read via `result_id`) against actuals. Score `mase < 1` (beats seasonal naive) and report `skill = 1 - MASE_model / MASE_snaive`. Also check that the forecast respects the requested horizon and frequency, and log continuous scores so experiments can be compared. `utilsforecast.losses` / `evaluate` provide the metrics (Nixtla).

## 6. LangSmith

- **Tracing:** set `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, and `LANGSMITH_PROJECT`. LangChain and LangGraph trace automatically. Use `@traceable` for non-LangChain code such as the sandbox and DuckDB, and add metadata and tags (`model`, `git_sha`, `prompt_version`).
- **Datasets and experiments:** create datasets in the UI or SDK, run `client.evaluate` or pytest, then compare experiments side by side. Repetitions help, because agents are noisy.
- **Latency and cost:** every run records latency, tokens and cost (see §4 for OpenRouter). Dashboards and monitors show p50/p99 latency, cost and error rate per project, and you can filter by metadata.
- **Online evals:** online evaluators (LLM-as-judge or code rules) run on sampled production traces or threads. Automation rules can route traces into datasets or annotation queues.
- **New in 2026** (Interrupt 2026, https://langchain.com/blog/interrupt-2026-overview):
  - LangSmith Sandboxes (GA)
  - **LangSmith Engine** (beta), which clusters trace failures and opens PRs with fixes and evals
  - SmithDB, a faster trace store
  - Messages View for multi-turn traces
  - Context Hub, for versioned prompts and policies
  - **LLM Gateway** (private beta), with spend caps and real-time spend
  - Managed Deep Agents
  - Fleet

## 7. Repo layout and recommendation

LangChain's official template (https://github.com/langchain-ai/new-langgraph-project, updated today) uses `src/agent/graph.py` (which exports `graph`), `langgraph.json`, `pyproject.toml` with uv, `tests/unit_tests` and `tests/integration_tests`, a `Makefile` and `.env.example`. Its `langgraph.json` is:
```json
{"$schema":"https://langgra.ph/schema.json","dependencies":["."],
 "graphs":{"analyst":"./src/taxi_agent/graph.py:graph"},"env":".env","image_distro":"wolfi"}
```
`langgraph dev` (from `langgraph-cli[inmem]`) opens **LangGraph Studio** in the browser, which is great for a portfolio GIF.

**Agent Chat UI** (https://github.com/langchain-ai/agent-chat-ui) is a Next.js chat UI for any graph with a `messages` key. **The GitHub API reports it as `archived: true` as of 2026-11-07, with the last push on 2026-11-03.** The docs still reference it, so I could not find a stated replacement [unverified]. Treat it as frozen, or use Streamlit or Chainlit as a simple front end.

**Suggested layout:**
```
src/taxi_agent/
  graph.py            # StateGraph: route → analyst(create_agent) / codeact loop → answer
  state.py            # typed state: messages, result_ids, charts, forecasts
  prompts.py
  models.py           # get_llm(): ChatOpenAI | ChatOpenRouter from config
  tools/ sql.py  schema.py  forecast.py  charts.py  python_exec.py
  sandbox/ base.py (Protocol)  docker.py  e2b.py
  middleware/ cost.py (OpenRouter cost → usage_metadata)  guards.py
  data/ catalog.py (views over Parquet), ingest.py
evals/ datasets/*.jsonl  evaluators.py (result-set, forecast MASE)  test_agent.py (pytest+langsmith)
docker/sandbox.Dockerfile   langgraph.json   pyproject.toml   Makefile   .env.example   README (arch diagram, eval table)
```

**Recommended stack:**
- `langchain` 1.4 `create_agent` plus middleware, inside a small custom `langgraph` 1.2 StateGraph.
- Typed tools first: `run_sql`, `describe`, `forecast` and `plot`.
- `run_python` as a CodeAct tool running in a **persistent Docker sandbox** with Parquet mounted read-only.
- Write your own ~150-line CodeAct loop or tool rather than using `langgraph-codeact`.
- `ChatOpenRouter` and `ChatOpenAI` behind a config switch.
- LangSmith for tracing and experiments.
- openevals for the LLM judge and trajectory checks, plus custom result-set and MASE evaluators run through pytest.

This shows real engineering (graph design, sandboxing, handle-based data flow and quantitative evals) without depending on archived code.

**Risks:**
1. `langgraph-codeact` and `langchain-sandbox` are archived, and agent-chat-ui appears archived. Don't build on them.
2. LangChain's own code-execution push (QuickJS interpreter, PTC) is JS-only, so Python CodeAct stays DIY.
3. LangSmith cost for OpenRouter needs manual pricing or a cost-injection middleware.
4. DuckDB 2.0 is weeks old, so pin it and test the read-only and sandbox settings.
5. Docker isolation is not a hard security boundary. That is fine locally, but use microVMs (E2B or LangSmith Sandboxes) for a public demo.
6. LangChain APIs move fast, so pin versions in `uv.lock`.
7. `agentevals` is lightly maintained, so prefer the trajectory evaluators in openevals.
