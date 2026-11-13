# Model replay: daily_trips

Run date: 2026-11-13
Dataset: nyc_taxi
Series: daily_trips
Cutoffs: 2025-07-31, 2026-01-31, 2026-07-31
Wall-clock time: 10.509 seconds

## How to read this

Each cutoff is a single forecast origin, not multiple independent backtests.
Daily evaluation: 1 series × 28 points per cutoff.
Sample sizes below count id-period observations once, not once per candidate model; points scored are per successful model (failed fits score zero points).
Relative MAE compares to SeasonalNaive, which repeats the last season (last 7 days). For daily forecasts this is a weak baseline when the last week before a cutoff is atypical.
MASE > 1 means worse than an in-sample seasonal-naive one-step forecast.
Results with this few origins are indicative, not conclusive; beating SeasonalNaive alone does not establish that a model is good.
Each id has equal weight. A model wins only with complete fits and mean relative MAE < 1.
Lineup requires wins at ≥ 2 of 3 cutoffs; SeasonalNaive is always retained.
MASE uses only training seasonal differences. Truth is observed, not imputed.
Zero baseline MAE: relative MAE is 1 for perfect predictions, infinity otherwise.
Zero training scale: MASE is 0 for perfect predictions, infinity otherwise.

## Cutoff 2025-07-31

Horizon: 28 periods
Retained ids: 1 (series)
Points scored per id: 28
Total scored points: 28

| Id | Points scored | Forecast origin (last training period) |
| --- | ---: | --- |
| series | 28 | 2025-07-31 |

Sample size: 1 series × 28 points per cutoff.

Baseline note: SeasonalNaive mean 134,417.000 vs actual mean 115,423.250 (+16.456%); pooled over retained ids and scored points.

| Model | Mean relative_mae | Mean MASE | coverage_80 | coverage_95 |
| --- | ---: | ---: | ---: | ---: |
| AutoARIMA | 0.946 | 1.655 | 0.536 | 1.000 |
| AutoETS | 0.796 | 1.392 | 0.964 | 1.000 |
| HistoricAverage | 0.399 | 0.699 | 1.000 | 1.000 |
| MSTL | 0.631 | 1.105 | 0.607 | 0.893 |
| SeasonalNaive | 1.000 | 1.750 | 0.964 | 1.000 |

## Cutoff 2026-01-31

Horizon: 28 periods
Retained ids: 1 (series)
Points scored per id: 28
Total scored points: 28

| Id | Points scored | Forecast origin (last training period) |
| --- | ---: | --- |
| series | 28 | 2026-01-31 |

Sample size: 1 series × 28 points per cutoff.

Baseline note: SeasonalNaive mean 115,814.143 vs actual mean 121,423.786 (-4.620%); pooled over retained ids and scored points.

| Model | Mean relative_mae | Mean MASE | coverage_80 | coverage_95 |
| --- | ---: | ---: | ---: | ---: |
| AutoARIMA | 0.576 | 1.122 | 0.821 | 0.964 |
| AutoETS | 0.709 | 1.382 | 0.964 | 0.964 |
| HistoricAverage | 0.789 | 1.538 | 0.857 | 0.964 |
| MSTL | 0.824 | 1.606 | 0.679 | 0.929 |
| SeasonalNaive | 1.000 | 1.949 | 0.821 | 0.857 |

## Cutoff 2026-07-31

Horizon: 28 periods
Retained ids: 1 (series)
Points scored per id: 28
Total scored points: 28

| Id | Points scored | Forecast origin (last training period) |
| --- | ---: | --- |
| series | 28 | 2026-07-31 |

Sample size: 1 series × 28 points per cutoff.

Baseline note: SeasonalNaive mean 117,640.143 vs actual mean 107,781.821 (+9.147%); pooled over retained ids and scored points.

| Model | Mean relative_mae | Mean MASE | coverage_80 | coverage_95 |
| --- | ---: | ---: | ---: | ---: |
| AutoARIMA | 1.336 | 1.283 | 0.679 | 1.000 |
| AutoETS | 0.813 | 0.781 | 1.000 | 1.000 |
| HistoricAverage | 1.253 | 1.203 | 0.929 | 1.000 |
| MSTL | 0.449 | 0.431 | 0.964 | 1.000 |
| SeasonalNaive | 1.000 | 0.960 | 1.000 | 1.000 |

## Wins per model

| Model | Wins |
| --- | ---: |
| AutoARIMA | 2 |
| AutoETS | 3 |
| HistoricAverage | 2 |
| MSTL | 3 |
| SeasonalNaive | 0 |

## Dropped ids

Minimum training share: 0.010

None.

## Failed fits

None.

## Adjustments

- AutoARIMA: 99 convergence warnings

## Final lineup

AutoARIMA, AutoETS, HistoricAverage, MSTL, SeasonalNaive
