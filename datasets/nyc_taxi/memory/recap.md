A row is one NYC yellow taxi trip: pickup, dropoff, fare components, and payment.

The trips view covers 2023-01-01 through 2026-09-30 (the valid window ends 2026-10-01). It excludes 176 rows whose timestamps fall outside their file's month.

Volume is skewed by borough: Manhattan about 3.3M trips a month, Queens about 350k, Brooklyn about 150k. Bronx and Staten Island are tiny.

Gaps and anomalies: 2023-09-21 through 2023-09-24 is a known gap (impute). 2026-02-23 is unusually low, about 26k trips; flag it, the cause is not in the data.

Schema: cbd_congestion_fee was added on 2025-01-05, when Manhattan congestion pricing starts. request_source was added in 2026-06, the first month the column is non-null (earliest value 2026-06-01).

Dirty rows: about 2.5% have fare or distance at or below zero, and 0.4% use an unknown zone (PULocationID 264 or 265). The trips metric includes those rows. revenue, avg_fare, avg_distance, and tip_rate apply valid_trip (fare and distance above zero). known_zone drops the unknown zones.
