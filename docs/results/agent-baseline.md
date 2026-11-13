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
