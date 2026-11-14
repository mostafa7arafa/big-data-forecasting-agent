.PHONY: test test-docker sandbox-image test-taxi lint replay dev eval-agent

test:
	uv run pytest -m "not taxi and not live and not docker"

test-docker:
	uv run pytest -m docker

sandbox-image:
	docker build -t analyst-sandbox:1 docker/sandbox

test-taxi:
	uv run pytest -m taxi

lint:
	uv run ruff check src tests

replay:
	uv run python scripts/replay_models.py --dataset nyc_taxi --series daily_trips && \
	uv run python scripts/replay_models.py --dataset nyc_taxi --series monthly_trips_by_borough

dev:
	uv run langgraph dev --n-jobs-per-worker 10

eval-agent:
	uv run python scripts/eval_agent.py
