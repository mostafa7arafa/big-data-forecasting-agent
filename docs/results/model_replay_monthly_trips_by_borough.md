# Model replay: monthly_trips_by_borough

Run date: 2026-11-08
Dataset: nyc_taxi
Series: monthly_trips_by_borough
Cutoffs: 2025-07-31, 2026-01-31, 2026-07-31
Wall-clock time: 8.064 seconds

## How to read this

Each cutoff is a single forecast origin, not multiple independent backtests.
Monthly sample sizes vary by cutoff; see the counts below.
Sample sizes below count id-period observations once, not once per candidate model; points scored are per successful model (failed fits score zero points).
Relative MAE compares to SeasonalNaive, which repeats the last season (last 12 months). For daily forecasts this is a weak baseline when the last week before a cutoff is atypical.
MASE > 1 means worse than an in-sample seasonal-naive one-step forecast.
Results with this few origins are indicative, not conclusive; beating SeasonalNaive alone does not establish that a model is good.
Each id has equal weight. A model wins only with complete fits and mean relative MAE < 1.
Lineup requires wins at ≥ 2 of 3 cutoffs; SeasonalNaive is always retained.
MASE uses only training seasonal differences. Truth is observed, not imputed.
Zero baseline MAE: relative MAE is 1 for perfect predictions, infinity otherwise.
Zero training scale: MASE is 0 for perfect predictions, infinity otherwise.

## Cutoff 2025-07-31

Horizon: 3 periods
Retained ids: 3 (Brooklyn, Manhattan, Queens)
Points scored per id: 3
Total scored points: 9

| Id | Points scored | Forecast origin (last training period) |
| --- | ---: | --- |
| Brooklyn | 3 | 2025-07-01 |
| Manhattan | 3 | 2025-07-01 |
| Queens | 3 | 2025-07-01 |

Sample size: 3 series × 3 points per cutoff.

Baseline note: SeasonalNaive mean 1,152,762.889 vs actual mean 1,347,992.667 (-14.483%); pooled over retained ids and scored points.

| Model | Mean relative_mae | Mean MASE | coverage_80 | coverage_95 |
| --- | ---: | ---: | ---: | ---: |
| AutoARIMA | 0.346 | 0.568 | 0.889 | 1.000 |
| AutoETS | 0.191 | 0.315 | 1.000 | 1.000 |
| HistoricAverage | 1.171 | 1.980 | 0.111 | 0.667 |
| SeasonalNaive | 1.000 | 1.702 | 0.333 | 1.000 |

## Cutoff 2026-01-31

Horizon: 3 periods
Retained ids: 3 (Brooklyn, Manhattan, Queens)
Points scored per id: 3
Total scored points: 9

| Id | Points scored | Forecast origin (last training period) |
| --- | ---: | --- |
| Brooklyn | 3 | 2026-01-01 |
| Manhattan | 3 | 2026-01-01 |
| Queens | 3 | 2026-01-01 |

Sample size: 3 series × 3 points per cutoff.

Baseline note: SeasonalNaive mean 1,286,520.444 vs actual mean 1,228,115.111 (+4.756%); pooled over retained ids and scored points.

| Model | Mean relative_mae | Mean MASE | coverage_80 | coverage_95 |
| --- | ---: | ---: | ---: | ---: |
| AutoARIMA | 1.945 | 1.050 | 0.333 | 0.333 |
| AutoETS | 1.331 | 0.724 | 0.444 | 0.556 |
| HistoricAverage | 1.600 | 0.794 | 0.889 | 1.000 |
| SeasonalNaive | 1.000 | 0.519 | 1.000 | 1.000 |

## Cutoff 2026-07-31

Horizon: 2 periods
Retained ids: 3 (Brooklyn, Manhattan, Queens)
Points scored per id: 2
Total scored points: 6

| Id | Points scored | Forecast origin (last training period) |
| --- | ---: | --- |
| Brooklyn | 2 | 2026-07-01 |
| Manhattan | 2 | 2026-07-01 |
| Queens | 2 | 2026-07-01 |

Sample size: 3 series × 2 points per cutoff.

Baseline note: SeasonalNaive mean 1,290,876.333 vs actual mean 1,193,932.000 (+8.120%); pooled over retained ids and scored points.

| Model | Mean relative_mae | Mean MASE | coverage_80 | coverage_95 |
| --- | ---: | ---: | ---: | ---: |
| AutoARIMA | 0.214 | 0.140 | 1.000 | 1.000 |
| AutoETS | 0.204 | 0.162 | 1.000 | 1.000 |
| HistoricAverage | 0.755 | 0.541 | 1.000 | 1.000 |
| SeasonalNaive | 1.000 | 1.099 | 0.833 | 1.000 |

## Wins per model

| Model | Wins |
| --- | ---: |
| AutoARIMA | 2 |
| AutoETS | 2 |
| HistoricAverage | 1 |
| SeasonalNaive | 0 |

## Dropped ids

Minimum training share: 0.010

- 2025-07-31: Bronx (share 0.004)
- 2025-07-31: EWR (share 0.000)
- 2025-07-31: N/A (share 0.001)
- 2025-07-31: Staten Island (share 0.000)
- 2025-07-31: Unknown (share 0.005)
- 2026-01-31: Bronx (share 0.005)
- 2026-01-31: EWR (share 0.000)
- 2026-01-31: N/A (share 0.001)
- 2026-01-31: Staten Island (share 0.000)
- 2026-01-31: Unknown (share 0.004)
- 2026-07-31: Bronx (share 0.005)
- 2026-07-31: EWR (share 0.000)
- 2026-07-31: N/A (share 0.001)
- 2026-07-31: Staten Island (share 0.000)
- 2026-07-31: Unknown (share 0.004)

## Failed fits

None.

## Final lineup

AutoARIMA, AutoETS, SeasonalNaive
