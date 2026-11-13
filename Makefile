.PHONY: test test-taxi lint replay dev

test:
	uv run pytest -m "not taxi"

test-taxi:
	uv run pytest -m taxi

lint:
	uv run ruff check src tests

replay:
	uv run python scripts/replay_models.py --dataset nyc_taxi --series daily_trips && \
	uv run python scripts/replay_models.py --dataset nyc_taxi --series monthly_trips_by_borough

dev:
	uv run langgraph dev --n-jobs-per-worker 10
