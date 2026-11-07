# 01 — Analytics agent products: Omni and peers

_Researched 2026-11-07. Vendor claims are marked as such. ⚠️ = could not verify._

## 1. Omni (omni.co)

### Agent architecture
- **Named persona → product name.** The assistant is **Blobby**. In 2025 it went from "autonomously perform analysis inside a Topic" to an "agentic harness" that was "rearchitected… to plan, think, and act" ([2025 highlights](https://omni.co/blog/2025-product-highlights)). Current docs call the surfaces **Omni Agent** (chat), **Workbook Agent**, **Modeling Agent**, and **Chat with Content** (dashboards) ([docs](https://docs.omni.co/ai)).
- **Coordinator loop and specialised tools** ([architecture post, Dec 2025](https://omni.co/blog/building-omnis-architecture-for-agentic-analytics)):
  - A coordinator picks the next tool from the question, prior results and what it has already tried. It retries on failure and stops when the answer is useful.
  - **Topic picker + query tool** looks at metadata across curated datasets ("Topics").
  - **Field-values tool** fixes filter mismatches, e.g. "NYC" vs "New York".
  - **Summarization sub-agent** reads results in its own context window, so the main context stays lean.
  - Results come back as Arrow, are converted to CSV, and large results stay out of the prompt. This keeps prompt caching intact.
- **Plan, then a task list.** For open questions the agent writes a task list, ticks steps off, checks results, tries another approach if they look wrong, and writes a summary. The reasoning stays visible to the user ([multi-step post](https://omni.co/blog/moving-beyond-simple-questions-with-agentic-ai)).
- **The agent selects; the engine compiles.** The 2026 benchmark post says the agent "selects fields and filters, but does not write SQL". A query engine compiles that selection through the semantic model, and a **validation agent** checks the answer against what the user meant ([benchmark, Jul 2026](https://omni.co/blog/benchmarking-omnis-agentic-analytics-harness)). The Dec 2025 post loosely says the agent "writes SQL". Treat the field-selection-and-compile design as the current one.
- **Semantic layer as AI context** ([AI optimization docs](https://docs.omni.co/modeling/develop/ai-optimization)). The YAML keys are:
  - `ai_context`: guidance set at model, topic, view or field level.
  - `ai_fields`: a curated subset of fields the agent can see.
  - `sample_queries`: example questions with known-good answers.
  - `synonyms`, `sample_values`, `all_values`.
  - Character budgets (`ai_prune_characters`) that prune in a fixed order: `all_values` first, `synonyms` last.
  - `ai_chat_topics`: a topic allowlist.
- **Other tools** ([chat docs](https://docs.omni.co/ai/chat)): search dashboards, build charts and dashboards, routines (scheduled AI), personal memory, docs/web search, and integrations (Drive, Slack, GitHub). Also available: [Skills](https://docs.omni.co/ai/skills.md), [forecasting](https://docs.omni.co/ai/forecasting.md), an [MCP server](https://docs.omni.co/ai/mcp), and Python ("Blobby learns Python", [Feb 2026 demo](https://docs.omni.co/demos/2026/20260206.md)).
- **How it shows its work** ([chat docs](https://docs.omni.co/ai/chat)):
  - Every answer is a **live, re-runnable query** that can go onto a dashboard.
  - "Query details" shows each field's definition.
  - Answers carry a **Verified / Not verified chip**. The hover reason is one of *Off topic*, *Not modeled*, *New fields* or *Review suggested*.
  - When an existing dashboard tile already answers the question, it appears in a highlighted card.
- **Guardrails** ([architecture](https://omni.co/blog/building-omnis-architecture-for-agentic-analytics), [AI page](https://omni.co/ai)): strict validation of tool inputs and outputs, detectors for malformed state, queries run with the user's own permissions and row-level security, and a topic/field allowlist.
- **Evals as a product feature** ([evals docs](https://docs.omni.co/ai/evals)):
  - Prompt sets of up to 25 questions, each with optional *expected behavior* that the agent never sees.
  - Runs execute on a model **branch**.
  - An LLM accuracy judge checks queries, results and the final answer for hallucinated claims, wrong date filters, truncated row limits and bad period comparisons.
  - Branch-vs-main comparisons show which prompts regressed before a change is promoted.
  - Related: [learn-from-conversation](https://docs.omni.co/ai/learn-from-conversation.md) and [suggestions](https://docs.omni.co/ai/suggestions.md) feed improvements back into the model.
- **Benchmark (vendor-run)** ([benchmark](https://omni.co/blog/benchmarking-omnis-agentic-analytics-harness)):
  - 100 real questions with rubrics, including "close but wrong" answers. Graded by Opus, then audited by a second Opus. The full matrix ran twice.
  - Omni scored 95/100 (90% on hard questions; next best: 60%). It gave the same answer on 96/100 questions across runs, while alternatives changed 30.
  - Cost was about $1.04 per correct answer, using about 670K tokens vs 1–6M for alternatives.

### Brand identity
- **Mascot: Blobby**, a soft pink blob character. It is an inline SVG with fill `#FFA6DD` ([site asset](https://omni.co/assets/blobby-CSOCXmt2.js)) and the CSS has a 7 s ease-in-out "bounce" loop, a mirrored `blob-facing-left` variant, a larger `blob-awww` variant, and staggered delays for groups of blobs ([CSS](https://omni.co/assets/blobby-DoJR3eew.css)).
- **Blobby is the AI's name, not just decoration.** Demo videos are titled "Blobby Deliveries" and "Blobby Content Downloads" ([homepage](https://omni.co/)), and there was a "Blobby Debugger" demo ([Mar 2026](https://omni.co/demos/20260306)).
- ⚠️ I could not confirm that Blobby appears as an in-app avatar. Product docs now say "Omni Agent".
- **Visual style:** hot-pink brand color `--color-brand:#ff5789` (dark: `#f91f68`) on light backgrounds, very large headings, and real product screenshots rather than abstract art ([homepage CSS/HTML](https://omni.co/)).
- **Tone:** plain and confident ("The AI analytics platform… answers they trust"). The blog publishes frank numbers along with caveats ("directional, not exact").
- **Worth copying in spirit:**
  - A friendly named character that *is* the agent.
  - One loud accent color.
  - Trust language ("verified", "governed") next to the playful character.
  - A gentle idle animation.
  - Engineering posts with real eval numbers.

## 2. Comparable agents

| Product | Semantic vs raw SQL | Planning | Execution | Verification and anti-hallucination |
|---|---|---|---|---|
| **Databricks Genie** (Agent mode, formerly "Research Agent") | Unity Catalog tables + instructions. **Trusted assets** are parameterized SQL/functions verified by the author ([concepts](https://docs.databricks.com/aws/en/genie-agents/concepts)) | Writes a research plan, runs multiple SQL queries, tests hypotheses, returns a report **with citations** + charts. A fast "Flash" mode for quick questions ([agent mode](https://docs.databricks.com/aws/en/genie-agents/agent-mode)) | SQL only, read-only, on the author's warehouse | "Inspect" re-reviews the generated SQL (preview). **Benchmarks** are held out from context ([concepts](https://docs.databricks.com/aws/en/genie-agents/concepts)) |
| **Snowflake Cortex Analyst** | **Semantic views** + a **Verified Query Repository** (question→SQL pairs, retrieved by similarity) ([VQR](https://docs.snowflake.com/en/user-guide/views-semantic/verified-query-repository), [Apr 2026 note](https://docs.snowflake.com/en/release-notes/2026/other/2026-04-05-semantic-views-verified-queries)) | Mostly single-shot text-to-SQL ⚠️ (multi-step lives in Cortex Agents; not verified here) | SQL | Built-in evals run the generated SQL and **compare result sets** with verified queries, temporarily removing the matched query so it can't just be copied ([evals](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-analyst-evaluations)) |
| **Hex Threads / Notebook Agent** | Prefers semantic models (dbt MetricFlow, Cube, Snowflake) and endorsed tables. A "Governed" mode restricts it to those ([threads docs](https://learn.hex.tech/docs/explore-data/threads), [semantic models](https://hex.tech/product/semantic-models)) | Plans and iterates. For open questions it may **offer several approaches or ask which direction to take** ([Threads, Oct 2025](https://hex.tech/blog/introducing-threads/)) | SQL + chart cells. Notebooks add Python. Threads Python ⚠️ | Step-by-step reasoning shown. A thread can **become a notebook** for auditing. Admins review the agent's reasoning and adjust rules ([AI overview](https://learn.hex.tech/docs/getting-started/ai-overview)) |
| **Looker Conversational Analytics (Gemini)** | LookML. The agent picks **fields, filters, sorts and limits**, and Looker composes the SQL ([overview](https://docs.cloud.google.com/looker/docs/2606/conversational-analytics-overview)) | Agent instructions + schema fetched from connected Explores | Optional **Code Interpreter** (Python, ≤5,000 rows, Preview), with a warning that it varies more than the core path ([code interpreter](https://cloud.google.com/looker/docs/studio/conversational-analytics-code-interpreter)) | Metrics can only come from LookML in core mode |
| **ClickHouse DWAINE** (internal) | Raw SQL via the ClickHouse MCP server + dbt models and an **MDBook glossary** over GitHub/filesystem MCP (LibreChat + Claude on Bedrock) ([blog, Nov 2025](https://clickhouse.com/blog/ai-first-data-warehouse)) | The model's own loop | SQL | The glossary "reduced hallucinations dramatically". Self-corrects on DB errors. Users ask for **commented SQL** to check by hand. Scoped to a few marts. dbt quality checks. About 70% of internal analytics questions handled |
| _Julius (aside)_ | Uploaded files, no semantic layer | — | Python/R in a per-user sandbox ([FAQ](https://hbs.julius.ai/docs/faqs)) | A third-party review reports hallucinated columns on complex joins ([review](https://guptadeepak.com/tools/top-5-ai-data-analysis-tools-2026/)) ⚠️ secondary source |

## 3. Synthesis: what separates a harness from "chat with a DB"

Difficulty is for a solo engineer building an MVP on DuckDB + Parquet.

1. **Semantic layer the agent selects from instead of writing free SQL.** The LLM emits `{measures, dimensions, filters, time_grain}` and a compiler builds the SQL. Who does it: Omni, Looker, Cortex. **Medium**: a YAML of ~15 taxi metrics plus a small compiler, or reuse dbt MetricFlow/Cube.
2. **AI context in the model**: descriptions, synonyms, sample values, curated field lists, and a character budget. Who does it: Omni `ai_context`/`ai_fields`, Hex workspace rules. **Low**.
3. **Field-value lookup tool** to resolve "JFK", "Manhattan", "yellow cab" to real values. Who does it: Omni. **Low**: `SELECT DISTINCT` cached per dimension.
4. **Verified/golden queries retrieved by similarity.** Who does it: Cortex VQR, Genie trusted assets, Omni `sample_queries`. **Low–Medium**: embed about 30 Q→SQL pairs.
5. **Coordinator + visible plan/task list + retry on failure.** Who does it: Omni, Genie Agent mode, Hex. **Medium**.
6. **Results stay out of the prompt.** The agent sees schema, row counts and a small sample, and a sub-agent summarizes with its own context window. Who does it: Omni (Arrow→CSV, summarization sub-agent). **Medium**. This is essential at 140M rows.
7. **Validation pass + Verified/Not-verified chip with a reason.** Who does it: Omni validation agent, Genie Inspect. **Medium**: a second LLM call with a rubric plus deterministic checks (empty result, row-limit hit, date range matches the question).
8. **Grounded narrative.** Every number in the explanation must trace back to a query result cell; reject or flag numbers that don't. Who does it: Genie (report with citations), Omni's judge (looks for "hallucinated claims"). **Medium**: regex the numbers and match them against result frames.
9. **Inspectable, re-runnable artifacts.** Show SQL and field definitions, and offer "open as notebook" or "pin to dashboard". Who does it: Omni live queries, Hex thread→notebook, DWAINE commented SQL. **Low**.
10. **Two lanes for computation: governed SQL by default, a sandboxed Python lane for advanced analysis, with a stated variance warning and row cap.** Who does it: Looker Code Interpreter, Omni Python, Julius. **Medium**. Forecasting with StatsForecast belongs in this lane, behind a typed tool rather than free Python.
11. **Evals as a feature**: a golden question set with expected behavior, an LLM judge using a "close but wrong" rubric, run twice for consistency, comparing branches. Who does it: Omni evals, Cortex evals, Genie benchmarks. **Medium**. This is the highest-signal item for a portfolio project.
12. **A named, characterful agent with a trust-first tone.** Who does it: Omni's Blobby. **Low** (design work only).

**Caveat:** every accuracy number above is self-reported by the vendor.
