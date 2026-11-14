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
- Latency: median 21.2 s, p95 67.8 s.
- Mean trajectory score: 0.89.

**Compaction** did not trigger in this run, and could not have. It only summarises earlier turns, and every row here is a single turn. It is covered by unit and graph tests, and multi-turn evals (Plan D) will exercise it.
