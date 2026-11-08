# Big-Data Forecasting Agent

An AI data analyst over ~162M NYC taxi trips (DuckDB over Parquet) that answers questions, forecasts with StatsForecast, and explains results without inventing causes. Portfolio flagship for GitHub/LinkedIn: architecture and measured results must impress a senior AI engineer.

## Read first
- Spec (binding): `docs/superpowers/specs/2026-11-07-taxi-analytics-agent-design.md`
- Current plan: `docs/superpowers/plans/2026-11-07-plan-a-data-and-forecasting-core.md` (Plan A; Plans B–D get written after A lands)
- Progress ledger (gitignored, the recovery map): `.superpowers/sdd/2026-11-07-plan-a-data-and-forecasting-core/progress.md` — tasks with `Task N: complete` are DONE; every `Ruling:` line is a decision to report to the owner
- Research: `docs/research/01-04`; web dashboard brief (separate later session): `docs/superpowers/briefs/web-dashboard-brief.md`

## How we work
- Superpowers process: brainstorm → spec → plan → subagent-driven development. Don't build before a spec/plan is approved.
- Implementation is delegated to external agents, visible in Herdr panes next to Claude: Cursor (`composer-2.5` or `grok-4.7-medium`, normal speed, never `-fast`) and Pi (`openai-codex/gpt-6.1-sol`). Start with `herdr agent start <name> --kind cursor|pi --pane <id> -- --model <m> [--force --trust]`; send a dispatch file via `herdr agent prompt`.
- Delegates never commit. Claude re-runs gates, gets an independent Claude-subagent review (sonnet; opus for security/statistics), runs fix rounds, then commits with the attribution trailer.
- One implementer at a time. Briefs, dispatches, reports, reviews live in the ledger directory.

## Commands
- `make test` (unit, synthetic data) · `make test-taxi` (integration, needs `data/raw`) · `make lint` · `make replay` (model cutoff replay → `docs/results/`)
- Data: `scripts/download_data.sh` (45 monthly files 2023-01..2026-09 into `data/raw/`, resumable)

## Rules that bite
- `src/analyst/` is dataset-agnostic: a test fails if `taxi`, `borough`, `tpep_`, `PULocationID`, `DOLocationID` appear there. Dataset specifics live in `datasets/<name>/` YAML/Markdown.
- The engine is a security boundary: single SELECT, AST-validated to reference only dataset views/CTEs with plain identifiers; cutoff enforced in the engine. Don't weaken it.
- Tools return summary cards (≤ 4,096 chars) and handles, never raw rows.
- Forecasts always carry a SeasonalNaive comparison; origin = end of data (respecting cutoff), never today.
- Python 3.12 via `uv`; `duckdb==2.0.0` pinned; everything locked in `uv.lock`.
