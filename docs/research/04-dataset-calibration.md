# 04 — Dataset calibration: how agents learn a new dataset

_Researched 2026-11-07. Vendor claims are marked as such. ⚠️ means I could not verify it against a primary source. Report 01 covers how these products answer questions; this report covers only **onboarding**: how they learn a dataset and store what they learn._

## 1. How competitors onboard a dataset

| Product | Auto-generated | Human-curated | Stored artifacts | Review | Refresh |
|---|---|---|---|---|---|
| **Omni** | After a new connection, a **quickstart skill** drafts a Topic from the business questions you describe ([quickstart](https://docs.omni.co/modeling/topics/quickstart)). The **Modeling Agent**, grounded in the model and connection, edits the model directly ([model assistant](https://docs.omni.co/ai/model-assistant.md)). **Auto-learn** (on by default) spots metric definitions, filters and synonyms in chats ([learn](https://docs.omni.co/ai/learn-from-conversation.md)). | Topics, `ai_context`, `ai_fields` | Model YAML: descriptions, `ai_context`, synonyms, `sample_values`, `sample_queries`, new measures and dimensions ([AI optimization](https://docs.omni.co/modeling/develop/ai-optimization)) | Each learning is shown as a **YAML diff in a modal** to accept, edit or reject. Evals run on branches (see report 01). | Driven by conversations. No scheduled re-profiling is documented. |
| **Snowflake** | **Semantic View Autopilot** (GA Feb 2026): AI-written table and column descriptions based on names and **sample values**; primary keys found by counting distinct values; joins and **verified queries** mined from **query history** and from Tableau/Power BI files ([docs](https://docs.snowflake.com/user-guide/views-semantic/autopilot)) | Name, description, table and column choice (docs suggest <10 tables and ≤50 columns), example question→SQL pairs (CSV) | Semantic view (a schema object; YAML is legacy): dimensions, facts, metrics, relationships, synonyms, verified queries, custom instructions | Invalid SQL is dropped. Edits happen in Semantic Studio or YAML, with inline accept/reject diffs. | After creation, Cortex Analyst posts **improvement suggestions**. An optimisation loop compares generated SQL with up to 50 verified queries, scores it with an LLM judge, and proposes one diff ([eng blog](https://www.snowflake.com/en/engineering-blog/using-ai-improving-ai/), vendor claim). |
| **Databricks Genie** | Unity Catalog **AI-generated comments** ([AI features](https://docs.databricks.com/aws/en/databricks-ai/)); primary/foreign keys turned into joins; **entity matching** (≤120 columns × 1,024 values) and format assistance turned on automatically ([knowledge store](https://docs.databricks.com/aws/en/genie/knowledge-store)) | Space-local descriptions, synonyms, SQL expressions (measures, filters, dimensions), example SQL, SQL functions, general instructions ([tune quality](https://docs.databricks.com/aws/en/genie/tune-quality)) | Knowledge store (≤200 snippets), **trusted assets** (answers that use them are labelled "Trusted"), benchmarks | "Knowledge mining" proposes new SQL expressions and joins when an **author** gives a thumbs-up or downloads results. The author approves them. | Values are refreshed **manually** with "Refresh prompt matching". Benchmarks are only used for evaluation and are never learned from. The 2026 "Genie Ontology" is described as self-maintaining ([SiliconANGLE](https://siliconangle.com/2026/06/16/databricks-new-agentic-coworker-genie-one-brings-ai-automation-every-part-business/), vendor claim ⚠️). |
| **Hex** | A Modeling Agent drafts and validates semantic models, and can read an existing project's code as context ([blog](https://hex.tech/blog/introducing-semantic-authoring/)) | **Endorsements** on schemas, tables, projects and models. A **workspace context** markdown file of ≤300 lines is sent with every prompt ([endorsements](https://learn.hex.tech/docs/agent-management/context-management/endorsements-in-context-studio), [best practices](https://learn.hex.tech/tutorials/ai-best-practices/workspace-context-best-practices)). | Semantic models, endorsement status, guides | Admins and Managers set endorsement status. Model **version history** includes checkpoints taken before the agent edits ([fall 2025](https://hex.tech/blog/fall-2025-launch/)). | Manual. |
| **Looker / Gemini** | The LookML Assistant suggests LookML from a natural-language prompt (Preview) ([docs](https://docs.cloud.google.com/looker/docs/gemini-overview-looker)) | LookML, data-agent instructions for Conversational Analytics | LookML | Git-based LookML development mode | ⚠️ I found no auto-profiling onboarding. |
| **dbt SL / MetricFlow** | dbt Copilot can generate semantic models, but **not yet in the new Fusion spec** ([docs](https://docs.getdbt.com/docs/build/metrics-spec-fusion)). There is also a `building-dbt-semantic-layer` agent skill ([listing](https://mcpservers.org/de/agent-skills/dbt-labs/building-dbt-semantic-layer), ⚠️ third-party mirror). | Everything else | YAML semantic models and metrics, exposed through the MCP server | Git PRs | dbt builds |
| **ThoughtSpot Spotter** | "SpotterModel" suggests descriptions, relationships and synonyms (⚠️ vendor summary). Spotter can be asked which columns confuse it. | AI context per column, **reference questions** (coaching), business terms | Model metadata and coaching memory | Docs say to **fix the model first and coach second** ([coaching](https://docs.thoughtspot.com/cloud/26.5.0.cl/spotter-coach-not-coach)) | Reference questions are revised as user queries come in ([after coaching](https://docs.thoughtspot.com/cloud/26.3.1.cl/spotter-coach-after)) |
| **Julius** | A **learning sub-agent** builds a "what AI has learned" table per connector. Progress caps at 85%. | Custom instructions per connection | Learned schema knowledge | The user can view and edit it, **freeze the schema**, or add knowledge ([docs](https://hbs.julius.ai/docs/data-connectors/customize-connections); learning page returned 403 ⚠️) | Grows with usage, and can be refreshed manually |
| **MotherDuck** | — | `COMMENT ON` for tables and columns. **Guides** are versioned markdown business-context docs stored in the warehouse; the MCP server reads them before writing SQL ([agents guide](https://motherduck.com/docs/key-tasks/ai-and-motherduck/building-analytics-agents.md)). | Comments and guides | Manual | Manual |

**Patterns across products**
1. Every product uses the same **inputs**: catalog metadata, **sample and distinct values**, key cardinality, and **query history or BI files**.
2. Each one stores its output as a **declarative artifact** (YAML or markdown), in a size-capped form (Genie's 200 snippets, Hex's 300 lines, Omni's character budgets).
3. A human approves every change through a **diff**. None of them writes changes silently.
4. Value lists and suggestions are **refreshed manually or through feedback**, not on a schedule.
5. Learning from feedback uses only *author* signals (Genie, Omni), never anonymous thumbs-up.
6. A practical warning from the field: comments only helped once the agent was **explicitly told to read them** ([codecentric](https://www.codecentric.de/en/knowledge-hub/blog/talk-to-your-data-part-2-limits-and-performance-enhancements)). The data memory must therefore be injected into the prompt, not merely made available.

## 2. Research and open source

- **AutoDDG** (NYU, [arXiv 2502.01050](https://arxiv.org/html/2502.01050v3)): first build a **content profile** and a **semantic profile**, then have the LLM write descriptions *from those statistics*. The authors show that LLM-only descriptions hallucinate facts. This supports a profile-then-describe design.
- **Metadata description generation for catalogs** (Fidelity, [arXiv 2503.09003](https://pdf.arxiv.org/pdf/2503.09003)): retrieval-based few-shot prompting plus an **abbreviation-expansion** step that leaves a term unexpanded when it stays ambiguous. **TACO** follows a similar route: expand abbreviations, then write descriptions with synonyms and keywords ([summary](https://www.roboticscenter.ai/research/papers/taco-task-aware-column-description-generation-using-llms-2606), ⚠️ venue unverified).
- **Towards Agentic Schema Refinement** ([arXiv 2412.07786](https://arxiv.org/html/2412.07786v1)): analyst and critic agents propose a semantic layer as views, and a **verifier executes them**. This is the generate → execute → critique loop to copy.
- **Semantic-layer benchmark** ([arXiv 2604.25149](https://arxiv.org/pdf/2604.25149)): when a semantic layer fails, the agent **refuses**; when raw text-to-SQL fails, it gives **confident wrong numbers**. It also cites **AskData**, which profiles the database, mines query logs and generates SQL-to-text metadata.
- **Per-dataset memory from feedback.** One paper ([arXiv 2511.10674](https://arxiv.org/abs/2511.10674), NeurIPS 2025 workshop) stores question, SQL, expert feedback and final correct SQL, then **distils** them into several memory stores. A "procedural" memory gave the biggest BIRD gains. **MERIT** ([REALM 2026](https://aclanthology.org/2026.realm-1.70/)) keeps memories of **verified corrections and failed directions**.
- **Boring Semantic Layer** ([PyPI](https://pypi.org/project/boring-semantic-layer/), [blog](https://juhache.substack.com/p/boring-semantic-layer-v2)): a YAML semantic layer built on Ibis that runs on DuckDB and has an optional MCP extra. Its author argues an LLM can be given the semantic layer as nothing more than a YAML file. It is a good target format, or a model to copy.
- **Profilers**: ydata-profiling `tsmode` detects gaps from timestamp intervals, plots ACF and PACF, and raises NON_STATIONARY and SEASONAL alerts based on ADF tests ([docs](https://docs.profiling.ydata.ai/4.5/features/time_series_datasets)). Nixtla **tsfeatures** provides `entropy` (spectral entropy as a forecastability proxy), `stability`, `lumpiness` and STL features ([PyPI](https://pypi.org/project/tsfeatures/), [R vignette](https://archive.linux.duke.edu/cran/web/packages/tsfeatures/vignettes/tsfeatures.html)). ⚠️ It is unconfirmed whether `seasonal_strength` comes under that name in Python. If it doesn't, it can be computed from the STL output.

## 3. What calibration should produce for time-series data

All of these are deterministic SQL or Python run in DuckDB unless marked **[LLM]**.

1. **Time column detection.** Look for temporal types, then for strings or integers that parse as dates (epoch ranges, ISO patterns). Rank candidates by non-null share and monotonicity. If several remain (pickup vs dropoff), **[LLM]** picks the canonical one and records the others as alternates.
2. **Grain.** Find the modal Δt between distinct timestamps and check whether the key (time × entity columns) is unique, so each row is classed as an *event* or a *pre-aggregated* row. This decides whether to use `COUNT(*)` or `SUM(x)`.
3. **Coverage and gaps.** Record min and max time per series. Build a calendar spine and count missing buckets. Flag partial first and last periods, and catch **future or implausible timestamps** (taxi data has rows from 2002 and 2090).
4. **Seasonality.** Run ACF at candidate lags (24 and 168 for hourly data; 7 and 365 for daily) and measure STL seasonal strength. Store the detected `season_length` per series for StatsForecast.
5. **Anomalies.** Use robust z-scores (MAD) on the aggregated series, and look for level shifts and zero runs. **[LLM]** annotates likely causes (holidays, COVID, outages) as *hypotheses* only.
6. **Entity hierarchies.** For each pair of low-cardinality columns, check functional dependency: if `COUNT(DISTINCT parent) per child = 1`, the child sits under the parent (zone → borough). Lookup tables can be joined in on key-overlap ratio. **[LLM]** names the levels.
7. **Measures vs dimensions.** Numeric columns with high cardinality that make sense to sum are measures. Integer codes with low cardinality are dimensions (e.g. `payment_type`). **[LLM]** settles ambiguous cases and decodes codes using sample values or a data dictionary.
8. **Dirty-row rules.** Look for negative fares and amounts, zero distances, durations ≤0 or of several days, nulls in the key, and outliers outside the p0.1–p99.9 range. Each rule is stored as a SQL predicate together with the share of rows it hits, and the rules are applied as a default `clean` view.
9. **Candidate metrics.** **[LLM]** proposes them from the measures and the grain (`trips = COUNT(*)`, `revenue = SUM(total_amount)`, `avg_fare = revenue/trips`). Each proposal is executed and sanity-checked: not null, plausible magnitude, and additive components summing to the total.
10. **Forecastability per series.** Report length in periods, the zero share (ADI/CV² intermittency class, a standard Syntetos–Boylan heuristic), spectral entropy and seasonal strength. From these, recommend a model family (AutoETS/AutoARIMA, CrostonOptimized/ADIDA for intermittent series, or "don't forecast" for series that are too short or too noisy). Optionally add a quick backtest MASE against SeasonalNaive.

## 4. Recommended MVP pipeline

```
connect → [A] profile (deterministic, DuckDB SUMMARIZE + custom SQL/py) → profile.json
        → [B] LLM draft (reads profile.json + samples only) → semantic.yaml, quirks.yaml, recap.md (status: draft)
        → [C] verify (execute every metric/rule/join; critic LLM checks contradictions with profile)
        → [D] human review gate (CLI/UI diff: accept/edit/reject per item) → status: approved, version++
        → [E] runtime: inject recap.md + relevant YAML slices; log Q/SQL/feedback → learned_queries.yaml (proposals → gate D)
        → [F] drift check on each new Parquet partition
```

- **Artifacts**, kept in `data_memory/<dataset_id>/`:
  - `profile.json` (machine facts plus a schema fingerprint)
  - `semantic.yaml` (entities, time dimension and grain, dimensions with synonyms and sample values, measures, metrics, joins, hierarchies)
  - `quirks.yaml` (rule id, SQL predicate, affected share, severity, evidence)
  - `series.yaml` (season length, coverage, forecastability class)
  - `verified_queries.yaml`
  - `recap.md` (≤300 lines, following Hex's budget)

  Every item carries `source: profiler|llm|human|feedback`, `status` and `confidence`. YAML is a good fit because Omni, Snowflake, dbt and BSL all store this kind of artifact as YAML.
- **Split the work.** Deterministic code computes anything that is a fact. The LLM only names things, describes them, resolves ambiguity and proposes metrics, and it must cite profile evidence for each (the AutoDDG lesson). Nothing the LLM writes reaches `approved` without first being executed (step C) and then reviewed (step D).
- **Refresh and drift.** For each new file or partition, compute the schema hash, row counts per period, null rates, new categorical values not yet in `sample_values`, and the time range, and compare them with `profile.json`.
  - A schema change triggers a **re-calibration diff**.
  - A statistical shift appends a quirk proposal.
  - New values refresh entity matching (Genie's "Refresh prompt matching" is the model for this).
- **Learning loop.** A thumbs-up from the owner (only the author, as in Genie) promotes the question and SQL to a *proposed* verified query. Corrections are kept as negative and procedural notes, as in arXiv 2511.10674 and MERIT.

### Evaluating calibration quality

1. **Gold layers.** Hand-write `semantic.gold.yaml` for 2–3 datasets: NYC TLC trips with a zone lookup, plus a dataset without a time column and one with an intermittent series. Score element by element:
   - time-column and grain accuracy
   - measure/dimension classification F1
   - hierarchy-edge precision and recall
   - quirk-rule recall, measured by the overlap of flagged row sets (Jaccard)
   - metric matching **by execution result** rather than by string (does the proposed `revenue` give the gold number for 10 random slices?)
   - error in detected season length
   - forecastability-class agreement
2. **Downstream A/B.** Run the question eval set from report 03 with no memory, with the auto draft, and with the approved memory. The difference shows how much the calibration step itself contributes, and how much human review adds on top.
3. **Perturbation tests.** Rename columns to cryptic names, inject dirty rows or a gap, and append a new month, then check that calibration and drift detection notice each change.
4. **Stability.** Run calibration twice and diff the outputs. Snowflake and Omni both stress repeatability; the target is for nothing outside descriptions to change between runs.
