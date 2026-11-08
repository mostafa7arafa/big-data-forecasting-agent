"""Replay a governed series, publish evidence, and propose its model lineup."""

import argparse
import time
from datetime import date
from pathlib import Path

import yaml

from analyst.data.dataset import load_dataset
from analyst.forecasting.models import DEFAULTS
from analyst.forecasting.replay import cutoff_scores, lineup, replay, wins
from analyst.memory.files import load_memory
from analyst.semantic.compiler import MetricQuery

CUTOFFS = [date(2025, 7, 31), date(2026, 1, 31), date(2026, 7, 31)]


def _baseline_note(window):
    baseline = window.loc[window.model == "SeasonalNaive"]
    if (len(baseline) != window.unique_id.nunique() or baseline.empty
            or baseline.forecast_mean.isna().any() or (baseline.scored_points == 0).any()):
        return "Baseline note: unavailable (incomplete SeasonalNaive forecasts)."
    points = baseline.scored_points
    predicted = float((baseline.forecast_mean * points).sum() / points.sum())
    actual = float((baseline.actual_mean * points).sum() / points.sum())
    if actual == 0:
        difference = "percentage difference undefined (actual mean is zero)"
    else:
        difference = f"{(predicted / actual - 1) * 100:+.3f}%"
    return (f"Baseline note: SeasonalNaive mean {predicted:,.3f} vs actual mean "
            f"{actual:,.3f} ({difference}); pooled over retained ids and scored points.")


def render_report(report, *, dataset, series, elapsed, min_share, min_wins=2, freq="D"):
    selected = lineup(report, min_wins=min_wins)
    samples = {
        (report.rows.loc[report.rows.cutoff == cutoff, "unique_id"].nunique(), horizon)
        for cutoff, horizon in report.horizons.items()
    }
    frequency = "Daily" if freq == "D" else "Monthly"
    if len(samples) == 1:
        n_ids, points = next(iter(samples))
        sample_note = f"{frequency} evaluation: {n_ids} series × {points} points per cutoff."
    else:
        sample_note = f"{frequency} sample sizes vary by cutoff; see the counts below."
    lines = [
        f"# Model replay: {series}", "",
        f"Run date: {date.today().isoformat()}",
        f"Dataset: {dataset}", f"Series: {series}",
        f"Cutoffs: {', '.join(map(str, report.horizons))}",
        f"Wall-clock time: {elapsed:.3f} seconds", "",
        "## How to read this", "",
        "Each cutoff is a single forecast origin, not multiple independent backtests.",
        sample_note,
        ("Sample sizes below count id-period observations once, not once per candidate model; "
         "points scored are per successful model (failed fits score zero points)."),
        ("Relative MAE compares to SeasonalNaive, which repeats the last season "
         f"(last {DEFAULTS[freq]['season']} {'days' if freq == 'D' else 'months'}). "
         "For daily forecasts this is a weak baseline when the last week before a cutoff is atypical."),
        "MASE > 1 means worse than an in-sample seasonal-naive one-step forecast.",
        ("Results with this few origins are indicative, not conclusive; "
         "beating SeasonalNaive alone does not establish that a model is good."),
        "Each id has equal weight. A model wins only with complete fits and mean relative MAE < 1.",
        (f"Lineup requires wins at ≥ {min_wins} of {len(report.horizons)} cutoffs; "
         "SeasonalNaive is always retained."),
        "MASE uses only training seasonal differences. Truth is observed, not imputed.",
        "Zero baseline MAE: relative MAE is 1 for perfect predictions, infinity otherwise.",
        "Zero training scale: MASE is 0 for perfect predictions, infinity otherwise.", "",
    ]
    scores = cutoff_scores(report)
    for cutoff, horizon in report.horizons.items():
        window = report.rows.loc[report.rows.cutoff == cutoff]
        ids = window.groupby("unique_id", sort=True).agg(
            scored_points=("scored_points", "max"), train_end=("train_end", "first"),
        )
        counts = ids.scored_points
        per_id = str(int(counts.iloc[0])) if counts.nunique() == 1 else ", ".join(
            f"{uid}: {int(count)}" for uid, count in counts.items()
        )
        lines.extend([
            f"## Cutoff {cutoff}", "", f"Horizon: {horizon} periods",
            f"Retained ids: {len(ids)} ({', '.join(map(str, ids.index))})",
            f"Points scored per id: {per_id}",
            f"Total scored points: {int(counts.sum())}", "",
            "| Id | Points scored | Forecast origin (last training period) |",
            "| --- | ---: | --- |",
        ])
        lines.extend(
            f"| {uid} | {int(row.scored_points)} | {row.train_end.date()} |"
            for uid, row in ids.iterrows()
        )
        if counts.nunique() == 1:
            lines.extend(["", f"Sample size: {len(ids)} series × {per_id} points per cutoff."])
        lines.extend([
            "", _baseline_note(window), "",
            "| Model | Mean relative_mae | Mean MASE | coverage_80 | coverage_95 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ])
        for row in scores.loc[scores.cutoff == cutoff].itertuples():
            lines.append(
                f"| {row.model} | {row.relative_mae:.3f} | {row.mase:.3f} "
                f"| {row.coverage_80:.3f} | {row.coverage_95:.3f} |"
            )
        lines.append("")
    lines.extend(["## Wins per model", "", "| Model | Wins |", "| --- | ---: |"])
    lines.extend(f"| {name} | {count} |" for name, count in wins(report).items())
    lines.extend(["", "## Dropped ids", "", f"Minimum training share: {min_share:.3f}", ""])
    lines.extend(
        f"- {item['cutoff']}: {item['unique_id']} (share {item['share']:.3f})"
        for item in report.dropped_ids
    )
    if not report.dropped_ids:
        lines.append("None.")
    lines.extend(["", "## Failed fits", ""])
    lines.extend(
        f"- {item['cutoff']}: {item['unique_id']} / {item['model']}: {item['error']}"
        for item in report.failed_fits
    )
    if not report.failed_fits:
        lines.append("None.")
    lines.extend(["", "## Final lineup", "", ", ".join(selected), ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--series", required=True)
    parser.add_argument("--min-share", type=float, default=0.01)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()
    started = time.perf_counter()
    dataset = load_dataset(args.dataset)
    memory = load_memory(dataset)
    spec = next((spec for spec in memory.series if spec.name == args.series), None)
    if spec is None:
        parser.error(f"unknown series: {args.series}")
    if len(spec.dimensions) > 1:
        parser.error("replay currently requires at most one id dimension")
    query = MetricQuery(
        metrics=[spec.metric], dimensions=spec.dimensions,
        grain={"D": "day", "MS": "month"}[spec.freq],
    )
    report = replay(
        dataset, query, cutoffs=CUTOFFS, freq=spec.freq,
        horizon=DEFAULTS[spec.freq]["horizon"],
        id_col=spec.dimensions[0] if spec.dimensions else None,
        value_col=spec.metric, min_share=args.min_share, n_jobs=args.n_jobs,
    )
    elapsed = time.perf_counter() - started
    output = dataset.root / "docs" / "results" / f"model_replay_{spec.name}.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_report(
        report, dataset=dataset.name, series=spec.name, elapsed=elapsed,
        min_share=args.min_share, freq=spec.freq,
    ))
    path = Path(dataset.memory_dir) / "series.yaml"
    mapping = yaml.safe_load(path.read_text())
    selected = lineup(report)
    for item in mapping["series"]:
        if item["name"] == spec.name:
            item.update(recommended_models=selected, source="profile", status="proposed")
            break
    path.write_text(yaml.safe_dump(mapping, sort_keys=False))
    print(f"{spec.name}: {', '.join(selected)} ({elapsed:.3f}s)", flush=True)
    print(f"Report: {output}", flush=True)


if __name__ == "__main__":
    main()
