# 03 — Evaluation Design

_Researched 2026-11-07. "(unverified)" means I could not open a primary source and am relying on search snippets or prior knowledge._

## 1. Data-agent benchmarks: what to borrow

**DAB — Data Agent Benchmark** ([arXiv 2603.20576](https://arxiv.org/abs/2603.20576), [repo](https://github.com/ucbepic/DataAgentBench)). 104 queries over 17 datasets in 4 DBMSs. It is based on interviews with companies building data agents, and its data includes heterogeneous sources, inconsistent duplicate records and facts hidden in free text. Grading ([HTML paper](https://arxiv.org/html/2603.20576)):
- Each query has its own **deterministic Python validator**.
- Numeric answers pass if they fall within a predefined tolerance. Single values use substring match. Fuzzy strings use Levenshtein distance (about 20% of the answer's length). Sets require every element. Dicts require each key with its value nearby.
- Extra wrong content is **not** penalised. No LLM judge is used for grading because it would hurt scalability and determinism.
- 5 trials per query, reported as unbiased pass@k with stratified averaging. The best agent reaches 57% pass@1 and none exceeds 70% pass@5.
- A GPT-5 judge labels failures only: FM1 no attempt, FM2 wrong plan, FM3 wrong data source, FM4 wrong implementation, FM5 runtime/timeout. **FM2 and FM4 together make up about 90%** of failures. The judge agreed exactly with a human on 67% of 30 trajectories.

**ClickHouse data-agent-mnist** ([blog](https://clickhouse.com/blog/agentic-analytics-benchmark-data-agent-mnist), [repo](https://github.com/ClickHouse/data-agent-mnist)). 201 real questions from their internal agent (DWAINE), run over a synthetic copy of their warehouse. The agent finds the schema itself and can take multiple turns. **There is no gold SQL: the result set is graded.** Ground truth comes from a "committee" agreeing, and the post admits that agreement is not verification. It reports a correctness-vs-cost frontier: about $1 vs about $52 per full run for roughly 7 points of correctness. _(Unverified: I could not fetch the blog or repo, so the details come from search snippets, and versions of the post disagree on the model count, 28 vs 29.)_ A related ClickHouse post on agentic analytics benchmarks ([JEV](https://clickhouse.com/ko/blog/agentic-analytics-benchmark-jev)) found judges marking correct answers as failures. It also found a comparator bug that failed a time-series answer when the reference was a single snapshot. Their fix was a "correct-despite-fail" audit. _(Unverified, from snippets.)_

**BIRD / Spider 2.0.** Both use execution accuracy, meaning the predicted SQL's result set is compared with the gold SQL's result set. Spider 2.0 adds enterprise and multi-dialect workflows. The main 2026 lesson comes from a UIUC audit ([arXiv 2601.08778](https://arxiv.org/abs/2601.08778), [blog](https://ddkang.substack.com/p/pervasive-annotation-errors-break)):
- **52.8% of BIRD Mini-Dev and 62.8% of Spider 2.0-Snow examples contain annotation errors.**
- After correction, agent scores shift by −7% to +31% (relative) and ranks move by up to 9 places. The Spearman correlation between rankings before and after correction is only 0.32.

**Others.**
- A semantic-layer study ([arXiv 2604.25149](https://arxiv.org/pdf/2604.25149)) runs the reference and candidate SQL and compares rows. Its "strict" check ignores column order and allows under 1% numeric drift. Its "analytical" check uses an LLM judge that also credits grouped breakdowns. The paper reports hallucination separately from accuracy.
- Tinybird ([post](https://www.tinybird.co/blog-posts/we-graded-19-llms-on-sql-you-graded-us)) found that exact row matching penalised valid answers. Its suggested alternatives are comparing sorted-result hashes or using EXCEPT/INTERSECT.

**Borrow for our 50–100-question set:**
1. Write a **per-question deterministic validator** with explicit tolerance (e.g. relative 0.5% for sums and averages, exact for counts) and order-insensitive set matching. Use an LLM judge only for prose and failure triage, as DAB does.
2. **Grade answers, not SQL.** Pin the reference result in a frozen snapshot, because TLC data gets revised.
3. **Audit the golden set.** Two people (or you plus a different model) should independently re-derive each answer, and you should re-check every question where all systems "fail". Annotation error is the biggest threat to validity.
4. Report **pass@1 over k=3–5 runs** with stratification by tier.
5. Use the **DAB FM1–FM5 taxonomy** for error analysis, and report how well the judge agrees with your own labels.

## 2. Forecast evaluation done right

**Protocol: rolling-origin backtests.** Use `StatsForecast.cross_validation(df, h=28, step_size=7, n_windows=12+)` to get many forecast cutoffs spanning all seasons. Use an expanding window, and check that no feature or holiday information leaks past each cutoff. A single train/test split on daily taxi data is noise, because a single month dominates it.

**Point metrics.** Use scale-free metrics so boroughs can be pooled (Manhattan is about 100× Staten Island):
- **MASE with seasonality=7**, scaled by in-sample seasonal-naive error.
- **Relative MAE vs SeasonalNaive** (`rmae`, where <1 beats the baseline).
- sMAPE only as a secondary metric. It is unstable near zero, which affects EWR and Staten Island on low days.

**Probabilistic metrics.**
- **Coverage** of the 80% and 95% intervals. A well-calibrated 95% interval covers about 95% of actuals. Report the gap, not just the number.
- **Winkler (interval) score**, which rewards narrow intervals and penalises misses by 2/α × the miss distance.
- **Scaled CRPS** from multiple quantile levels.

**Support in Nixtla's utilsforecast** ([losses](https://nixtlaverse.nixtla.io/utilsforecast/losses.html), [evaluation](https://nixtlaverse.nixtla.io/utilsforecast/evaluation)):
- `losses` provides `mae, rmse, smape, mase(seasonality, train_df), rmae(baseline), msse/rmsse, quantile_loss, mqloss, scaled_mqloss, scaled_crps, coverage(level), winkler_score(level), calibration`.
- `evaluate(df, metrics=[...], train_df=..., level=[80,95])` runs over the cross-validation output grouped by `cutoff`, so you get metrics per cutoff and can show their spread. Wrap level-dependent metrics with `functools.partial`.

**Taxi-specific pitfalls.**
- **Multiple seasonality.** Taxi demand has weekly and yearly cycles. AutoETS/AutoARIMA take one season length (7), so a yearly pattern bleeds into errors. Include MSTL (`season_length=[7,365]`) as a candidate, or at least report errors by month.
- **Holidays.** Thanksgiving, Christmas–New Year and July 4 produce huge drops. Report metrics **with and without holiday windows**, and add a holiday-only slice. Consider holiday exogenous features with care to avoid leakage.
- **Outliers and regime shifts.** Storms and blizzards (e.g. Jan 2024 snow), plus the **Jan 5 2025 congestion pricing start** in the Manhattan CBD, which is a level shift relevant to the explanation evals. Do not delete these from the test set. Report them as a "shock" slice, and keep outliers out of training-scale denominators.
- **Data artifacts.** Late-arriving months, out-of-range `tpep_pickup_datetime` values, and partial final months. Assert complete days before using them as actuals.
- **Baselines.** Always include **SeasonalNaive(7)** and a **Naive/HistoricAverage** baseline. If AutoETS doesn't beat SeasonalNaive by a margin larger than the spread across cutoffs, say so in the results.

## 3. Evaluating LLM forecasting agents and grounded explanations

**Related benchmarks:**
- **Context is Key (CiK)** ([arXiv 2410.18959](https://arxiv.org/abs/2410.18959)) pairs series with text that is *essential* to forecast them. It scores forecasts with RCRPS, a region-of-interest-weighted CRPS that also penalises constraint violations _(metric details unverified, from memory)_.
- **GIFT-CTX** (Salesforce, [MoiraiAgent blog](https://www.salesforce.com/blog/moiraiagent/)) curates 120 CiK samples plus 125 synthetic ones, so that a forecast needs both history and context. It is scored with weighted NMAE/NRMSE. All results are vendor-reported.
- **TemporalBench** ([arXiv 2602.13272](https://arxiv.org/abs/2602.13272)) uses four tiers: interpreting historical structure, context-free forecasting, contextual reasoning, and event-conditioned prediction. Its key finding is that **good forecasting accuracy does not imply correct temporal reasoning**.
- **Dr-CiK** ([arXiv 2605.27904](https://arxiv.org/abs/2605.27904)) makes the agent *retrieve* context. Distractors mislead agents, and retrieved context can make forecasts **worse than no context**. The lesson for us is that our agent should not pull in outside "reasons" by default.

**Detecting invented causes.** Use a pipeline of claim extraction followed by verification, in the spirit of FActScore/RAGAS faithfulness _(general technique, no single paper verified here)_:
1. **Make the agent emit structured evidence.** It should return its explanation together with an `evidence` JSON of the computed facts it relied on: STL trend slope, weekly and yearly seasonal amplitude, YoY %, anomaly dates, and the model chosen with its MASE.
2. **Extract claims.** An LLM splits the explanation into atomic claims and labels each one as *descriptive* ("weekday trips 18% above weekend"), *comparative*, *forecast*, or *causal* ("due to congestion pricing", "because of weather").
3. **Verify claims.**
   - Descriptive and comparative numbers are checked **deterministically** against the evidence and against a re-run SQL query, with a tolerance.
   - Causal claims pass only if they are hedged and labelled as hypotheses, or if they cite a fact present in a small **allow-listed context file** (e.g. holiday calendar, congestion-pricing start date). Anything else counts as an **unsupported causal claim**.
4. **Metrics.**
   - *Claim precision*: supported claims ÷ all claims.
   - *Unsupported-causal rate*: the share of explanations with ≥1 unsupported causal claim. **This is the headline number.**
   - *Numeric-claim accuracy*.
   - *Coverage of key facts*: did it mention the dominant seasonality?
5. **Judge with a rubric**, using openevals `create_llm_as_judge` with a custom prompt and binary per-criterion scores (grounded / hedged / no new entities / interval mentioned). Calibrate it against about 40 hand-labelled explanations and report the agreement rate (Cohen's κ). DAB's 67% judge–human agreement is a reminder that judges must be validated.
6. **Adversarial traps.** Write prompts that invite a cause, e.g. "Why did Bronx trips drop in March 2024?" when nothing notable happened. The correct behaviour is to describe the pattern and say the data cannot establish a cause.

## 4. Agent-level metrics and credible reporting

- **Task success** comes from the per-tier validators.
- **Trajectory quality.** Use agentevals trajectory match (strict, unordered, subset or superset over tool calls) for cases where the path matters, e.g. the agent must call `forecast` and must not hand-roll a forecast in SQL. For other cases use an LLM trajectory judge _(openevals/agentevals API from memory; verify against the current docs)_.
- **Tool-call errors.**
  - SQL that fails to execute, as a rate per question and as the share of runs that recovered after retry.
  - Schema hallucinations, i.e. nonexistent columns.
  - Forecast calls with invalid arguments.
- **Efficiency.** p50/p95 latency, mean tool calls, input/output tokens and **$ per question** (LangSmith captures latency and tokens per trace).

**README table conventions a senior engineer will trust:**
- Report n, k runs and **95% CIs** (bootstrap or Wilson).
- Report per-tier breakdowns, not only an aggregate.
- Put the baseline in the same table as the agent.
- Pin model, prompt and dataset versions and the date, and link the LangSmith experiment.
- Include a "known failures" section with FM-coded examples.
- Never report a single run on a self-written test set without caveats.

| Eval | n | Metric | Agent (model X) | Baseline | Notes |
|---|---|---|---|---|---|
| SQL golden, T1–T3 | 80 × 3 runs | pass@1 [95% CI] | … | plain text-to-SQL | per tier below |
| Forecast backtest | 5 boroughs × 12 cutoffs | MASE / rMAE vs SN7 / 95% cov / Winkler | … | SeasonalNaive(7) | holiday & shock slices |
| Explanation grounding | 40 | unsupported-causal rate ↓, claim precision ↑ | … | — | judge κ vs human |
| Adversarial | 20 | correct-refusal rate | … | — | |
| Cost/latency | all | $/q, p50/p95 s | … | | |

## 5. Proposed eval plan for this project

**A. SQL golden set (~70 questions) in tiers:**
- **T1 lookup/aggregate (25).** E.g. "total trips Jul 2024", "median fare JFK". Graded with numeric tolerance.
- **T2 multi-step (25).** Joins with the zone lookup, window functions, YoY, percentiles, top-k by borough. Graded by order-insensitive result-set match with tolerance.
- **T3 analytical/ambiguous (20).** E.g. "busiest hour on rainy-looking weekdays" or definitional questions like "trips" vs "rides with fare>0". The rubric accepts multiple valid interpretations if the agent **states its assumption**.

Freeze the answers on a dated Parquet snapshot and have every answer double-derived. Graders are DAB-style Python validators wrapped as LangSmith evaluators. Run k=3.

**B. Forecast backtests.** Daily trips for each of 5 boroughs (plus airports). Use `cross_validation` with h=28, step 7, 12+ windows covering 2025-2026. Compare AutoETS, AutoARIMA, MSTL, SeasonalNaive(7) and HistoricAverage. Score with `evaluate` using MASE(7), rMAE vs SN7, sMAPE, coverage@80/95, Winkler@95 and scaled CRPS. Slice by holiday, shock and normal days. This tests the forecasting tool itself.

**C. Agent forecasting tasks (15).** E.g. "forecast Queens trips for the next 4 weeks". The graders check four things:
- The agent called the forecast tool with correct arguments (h, series, filters) and included the baseline. Checked by trajectory match.
- The chart contains intervals. Checked from the artifact JSON.
- The reported numbers match the tool output exactly.
- The stated model comparison is correct.

**D. Explanation grounding (40).** Taken from B and C outputs plus historical "why/what changed" questions. Graded with the claim-extraction and verification pipeline (§3) and a calibrated rubric judge.

**E. Adversarial and unanswerable (20).**
- Questions outside the data range (e.g. "trips in 2019", "Dec 2026 actuals").
- Columns that don't exist (driver names, weather).
- Causal bait.
- Prompt injection through question text.
- Destructive SQL requests (DROP or writes must be refused or blocked).

The grader is the correct-refusal or clarification rate, plus a false-refusal rate on matched answerable controls.

**Headline numbers to publish:**
1. SQL pass@1 by tier with CIs.
2. Forecast MASE and rMAE vs SeasonalNaive, with 95% interval coverage.
3. **Unsupported-causal-claim rate.**
4. Adversarial correct-refusal rate alongside the false-refusal rate.
5. $/question and p95 latency.

Also publish the FM1–FM5 failure breakdown and the judge-vs-human κ.
