"""Rebuild the frozen SQL answers. Run with: uv run python scripts/build_gold.py."""

import json
import time
from pathlib import Path

import pyarrow.parquet as pq

from analyst.data.dataset import load_dataset
from analyst.data.engine import Engine

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    rows = [json.loads(line) for line in (ROOT / "evals/datasets/sql_golden.jsonl").read_text().splitlines()]
    destination = ROOT / "evals/gold"
    destination.mkdir(parents=True, exist_ok=True)
    engine = Engine(load_dataset("nyc_taxi"))
    for row in rows:
        started = time.perf_counter()
        try:
            table = engine.execute(row["gold_sql"])
            pq.write_table(table, destination / f"{row['id']}.parquet")
        except Exception:
            print(f"{row['id']}: FAILED after {time.perf_counter() - started:.3f}s", flush=True)
            raise
        print(f"{row['id']}: {table.num_rows} rows, {time.perf_counter() - started:.3f}s", flush=True)


if __name__ == "__main__":
    main()
