# Agent baseline: single-turn eval (Plan B1)

First measurement of the LangGraph analyst on the 50-question single-turn eval set. It is a **baseline, not a target**: later prompts, models and Plan D work are compared against it.

| | |
|---|---|
| Date | 2026-11-13 |
| Model | `openai:gpt-6-luna` |
| Git | `050a4db` (branch `feat/plan-b1-agent`, after the CodeRabbit round) |
| LangSmith dataset | `analyst-single-turn`, tag `v3-6add7971` (50 examples) |
| Experiment | [`openai-gpt-6-luna-050a4db-7985d028`](https://smith.langchain.com/o/c61fdf2e-c2e4-4aa9-aa76-e92554e6158d/datasets/8fc4d2c1-b553-4e1a-b3ee-64bae4e2f9dd/compare?selectedSessions=76724de3-8c9b-40da-a8f2-1de989753d73) |
| Run | `make eval-agent` → one repetition, `background=False`, max concurrency 3 |

## Pass rate

A row passes when every gate that applies to it passes (`answer_values`, `label`, `grounding`, `checks`). Tool trajectory is reported as a score and never gates.

| Category | Rows | Pass |
|---|---:|---:|
| Lookup | 20 | 75% |
| Comparison (YoY, previous period, share) | 8 | 62% |
| Forecast (cutoff replay) | 8 | 88% |
| Off-model (needs `run_sql`) | 6 | 33% |
| Grounding trap | 5 | 60% |
| Safe handling | 3 | 100% |
| **All** | **50** | **70%** |

| Gate | Pass |
|---|---:|
| `answer_values` (key numbers within tolerance) | 74% |
| `label` (expected verification label) | 57% |
| `grounding` (no unsupported causes, prose included) | 80% |
| `checks` (forecast shape and date grid, declines with a reason, quirks flagged) | 88% |

## Verification labels

| Label | Answers |
|---|---:|
| Verified | 23 |
| Verified · off-model | 4 |
| Not verified | 23 |

The most common Not-verified reasons were:
- "unsupported causal claim" (2);
- "no claims (decline or acknowledgement)" (2);
- "claim cites no evidence" (1);
- single untraced derived percentages such as 16.25 or 20.03, each once. These are numbers the model computed itself instead of taking them from a handle.

## Plan B2: with `run_python` (55 questions)

The same harness and graders, re-run after Plan B2 added the Docker CodeAct sandbox (`run_python`) and compaction. Five `python` rows (a051–a055) were added. Each needs a statistic the typed tools can't express: a correlation, a rolling peak, a coefficient of variation, a median of daily changes, and a share above mean + 1 sd.

| | |
|---|---|
| Date | 2026-11-14 |
| Model | `openai:gpt-6-luna` |
| Git | `81b55b5` (branch `feat/plan-b2-codeact`, after the final security fix wave) |
| LangSmith dataset | `analyst-single-turn`, tag `v4-0a12fd10` (55 examples) |
| Experiment | [`openai-gpt-6-luna-81b55b5-aae16c18`](https://smith.langchain.com/o/c61fdf2e-c2e4-4aa9-aa76-e92554e6158d/datasets/8fc4d2c1-b553-4e1a-b3ee-64bae4e2f9dd/compare?selectedSessions=5bd0eac2-b4bd-4ed7-b0d6-90fc30f34c07) |
| Run | `make eval-agent`: one repetition, `background=False`, max concurrency 3, one sandbox per row |

| Category | Rows | Pass (B1) | Pass (B2) |
|---|---:|---:|---:|
| Lookup | 20 | 75% | 80% |
| Comparison | 8 | 62% | 75% |
| Forecast | 8 | 88% | 88% |
| Off-model (`run_sql`) | 6 | 33% | 33% |
| Python (`run_python`) | 5 | n/a | **100%** |
| Grounding trap | 5 | 60% | 40% |
| Safe handling | 3 | 100% | 67% |
| **All** | **50 / 55** | **70%** | **73%** |

| Gate | B1 | B2 |
|---|---:|---:|
| `answer_values` | 74% | 81% |
| `label` | 57% | 74% |
| `grounding` | 80% | 80% |
| `checks` | 88% | 81% |

**Labels:** 21 Verified, 11 Verified · off-model, 23 Not verified.

**Python rows.** All five Python answers came back **Verified · off-model**. Their numbers trace to handles saved from the sandbox, and the label says they were computed outside the governed metrics.

**Label gate.** It rose from 57% to 74%. Part of that comes from a B2 change: `run_python` prints small saved results at full precision, and the model is told to copy cited numbers exactly.

**Small categories are noisy.** Trap (5 rows) and safe handling (3 rows) move by 20–33 points from a single row. A local run of the same graph that day scored trap 80% and safe 100%. Plan D adds repetitions to measure this variance.

**Other measures:**
- Forecast skill is unchanged: WAPE 0.083, coverage 0.69 at 80% and 0.87 at 95%.
  These numbers are optimistic: the forecaster's `recommended_models` were chosen by the Plan A replay on the same cutoffs this eval scores, so they are not an independent out-of-sample test (see the B1 follow-ups).
- Latency: median 21.2 s, p95 67.8 s.
- Mean trajectory score: 0.89.

**Compaction** did not trigger in this run, and could not have. It only summarises earlier turns, and every row here is a single turn. It is covered by unit and graph tests, and multi-turn evals (Plan D) will exercise it.

## Plan W1: dashboard backend (55 questions)

The same harness and graders, re-run after Plan W1 added the typed chart catalogue (`make_chart` now returns a validated chart spec), the run trace, and OpenAI reasoning summaries for the default model (Responses API). A new `chart_kind` check grades the eight forecast rows: the answer must carry a `forecast_fan` chart, or a `small_multiples` chart whose inner kind is `forecast_fan`.

| | |
|---|---|
| Date | 2026-11-15 |
| Model | `openai:gpt-6-luna` (reasoning summaries on) |
| Git | `ddd0fdb` (branch `feat/plan-w1-dashboard-backend`) |
| LangSmith dataset | `analyst-single-turn`, tag `v5-7d6a5f66` (55 examples) |
| Experiment | `openai-gpt-6-luna-ddd0fdb-cee4069d` |
| Run | `make eval-agent`: one repetition, max concurrency 3, one sandbox per row |

| Category | Rows | Pass (B2) | Pass (W1) |
|---|---:|---:|---:|
| Lookup | 20 | 80% (16) | 80% (16) |
| Comparison | 8 | 75% (6) | 75% (6) |
| Forecast | 8 | 88% (7) | 88% (7) |
| Off-model (`run_sql`) | 6 | 33% (2) | 33% (2) |
| Python (`run_python`) | 5 | 100% (5) | **100% (5)** |
| Grounding trap | 5 | 40% (2) | 60% (3) |
| Safe handling | 3 | 67% (2) | 67% (2) |
| **All** | **55** | **73% (40)** | **75% (41)** |

| Gate | B2 | W1 |
|---|---:|---:|
| `answer_values` | 81% | 79% |
| `label` | 74% | 74% |
| `grounding` | 80% | 80% |
| `checks` | 81% | 81% |

**Plan gate:** at least 38 of 55, Python 5/5, no category down by more than one question versus B2. Passed: 41/55, Python 5/5, no category down.

**chart_kind:** 7 of the 8 forecast rows pass it (four as a `small_multiples` of `forecast_fan`, three as a plain `forecast_fan`). The eighth, a037, fails both `forecast_shape` and `chart_kind` because the agent never produced a forecast (no handle, no chart). It is a forecast failure, not a chart-kind one.

**Labels:** 20 Verified, 15 Verified · off-model, 20 Not verified. Not-verified reasons: untraced number 2026 (2), no claims (decline or acknowledgement) (2), untraced number 20.03 (1), untraced number 27.0288 (1), claim cites no evidence (1).

**Other measures:**
- Forecast skill is unchanged: WAPE 0.083, coverage 0.69 at 80% and 0.87 at 95%.
- Latency: median 33.9 s, p95 106.5 s (B2: 21.2 s and 67.8 s). Reasoning summaries add time, and a single run is noisy.
- Mean trajectory score: 0.91 (B2: 0.89). Mean tokens 16,643 per question.
- Small categories stay noisy: one question moves trap by 20 points.
