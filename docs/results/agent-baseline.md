# Agent baseline: single-turn eval (Plan B1)

First measurement of the LangGraph analyst on the 50-question single-turn eval set. It is a **baseline, not a target**: later prompts, models and Plan D work are compared against it.

| | |
|---|---|
| Date | 2026-11-13 |
| Model | `openai:gpt-6-luna` |
| Git | `36870c8` (branch `feat/plan-b1-agent`) |
| LangSmith dataset | `analyst-single-turn`, tag `v2-4dd6dbb5` (50 examples) |
| Experiment | [`openai-gpt-6-luna-36870c8-7262e3ed`](https://smith.langchain.com/o/c61fdf2e-c2e4-4aa9-aa76-e92554e6158d/datasets/8fc4d2c1-b553-4e1a-b3ee-64bae4e2f9dd/compare?selectedSessions=2e1e0f7e-7260-40dd-a50f-9ad8fac98cad) |
| Run | `make eval-agent` → one repetition, `background=False`, max concurrency 3 |

## Pass rate

A row passes when every gate that applies to it passes (`answer_values`, `label`, `grounding`, `checks`). Tool trajectory is reported as a score and never gates.

| Category | Rows | Pass |
|---|---:|---:|
| Lookup | 20 | 70% |
| Comparison (YoY, previous period, share) | 8 | 62% |
| Forecast (cutoff replay) | 8 | 88% |
| Off-model (needs `run_sql`) | 6 | 33% |
| Grounding trap | 5 | 80% |
| Safe handling | 3 | 100% |
| **All** | **50** | **70%** |

| Gate | Pass |
|---|---:|
| `answer_values` (key numbers within tolerance) | 68% |
| `label` (expected verification label) | 64% |
| `grounding` (no unsupported causes) | 100% |
| `checks` (forecast shape, declines, quirks flagged) | 94% |

## Verification labels

| Label | Answers |
|---|---:|
| Verified | 24 |
| Verified · off-model | 4 |
| Not verified | 22 |

The most common Not-verified reasons were:
- "no claims (decline or acknowledgement)" (3);
- "unsupported causal claim" (2);
- single untraced derived percentages such as 20.03 or 27.27, each once. These are numbers the model computed itself instead of taking them from a handle.

## Forecasts and cost

| Metric | Value |
|---|---|
| Mean trajectory score (superset match) | 0.96 |
| Forecast WAPE vs the real future (mean of forecast rows) | 0.083 |
| 80% interval coverage | 0.69 |
| 95% interval coverage | 0.87 |
| Latency, median / p95 | 20.8 s / 77.6 s |
| Mean tokens per question | 12,635 |
| Tool-output bytes per question, max (sum of tool messages in the run) | 70,092 |
| Cost per question | n/a: LangSmith prices OpenAI runs server-side; see the experiment |

## Read this before quoting the numbers

- **Forecast skill is optimistic.** The forecasting lineup (`recommended_models` in `datasets/nyc_taxi/memory/series.yaml`) was settled by the Plan A replay on the same cutoffs these forecast rows use: 2025-07-31, 2026-01-31 and 2026-07-31. That is look-ahead. An honest out-of-sample number needs cutoffs the lineup never saw (Plan D).
- **Interval coverage is below nominal.** 80% intervals covered 69% of outcomes, and 95% intervals covered 87%, over 8 forecast rows. The sample is small, but this is worth watching.
- **Tool bytes are not context bytes.** The 70 KB figure sums every tool message in a run. What the model actually sees per call is smaller, because older tool outputs are cleared to one-line stubs. The precise per-call numbers are the `context_tool_bytes` metadata on the `ContextMeter` spans in LangSmith; the peak in the live Studio runs was about 8 KB.
- **One repetition.** Pass rates will move a few points between runs. Plan D adds repetitions and confidence intervals.
