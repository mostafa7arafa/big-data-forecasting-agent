"""Rebuild the frozen gold answers.

    uv run python scripts/build_gold.py              # SQL gold set (q*.parquet)
    uv run python scripts/build_gold.py --set agent  # agent eval set (a*.parquet + key_values)

Gold always runs on the full data (no cutoff): forecast rows freeze the real
future. Parquets that belong to neither set are removed as orphans.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analyst.data.dataset import load_dataset
from analyst.data.engine import Engine
from evals.graders.agent import derive_key_values, load_rows

SQL_SET = ROOT / "evals/datasets/sql_golden.jsonl"
AGENT_SET = ROOT / "evals/datasets/agent_single_turn.jsonl"
GOLD = ROOT / "evals/gold"


def _sql_rows() -> list[dict]:
    return [json.loads(line) for line in SQL_SET.read_text().splitlines() if line.strip()]


def _freeze(engine: Engine, row: dict):
    started = time.perf_counter()
    try:
        table = engine.execute(row["gold_sql"])
        pq.write_table(table, GOLD / f"{row['id']}.parquet")
    except Exception:
        print(f"{row['id']}: FAILED after {time.perf_counter() - started:.3f}s", flush=True)
        raise
    print(f"{row['id']}: {table.num_rows} rows, {time.perf_counter() - started:.3f}s", flush=True)
    return table


def build_sql(engine: Engine) -> int:
    rows = _sql_rows()
    for row in rows:
        _freeze(engine, row)
    return len(rows)


def build_agent(engine: Engine) -> int:
    rows = load_rows(AGENT_SET)
    for row in rows:
        if row["gold_sql"] is None:
            row["key_values"] = []
            continue
        table = _freeze(engine, row)
        row["key_values"] = derive_key_values(row, table)
    AGENT_SET.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return len(rows)


def remove_orphans() -> list[str]:
    owned = {row["id"] for row in _sql_rows()}
    if AGENT_SET.is_file():
        owned |= {row["id"] for row in load_rows(AGENT_SET) if row["gold_sql"] is not None}
    orphans = sorted(p for p in GOLD.glob("*.parquet") if p.stem not in owned)
    for path in orphans:
        path.unlink()
        print(f"removed orphan {path.name}", flush=True)
    return [p.name for p in orphans]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", choices=("sql", "agent"), default="sql", dest="which")
    args = parser.parse_args()
    GOLD.mkdir(parents=True, exist_ok=True)
    engine = Engine(load_dataset("nyc_taxi"))
    count = build_agent(engine) if args.which == "agent" else build_sql(engine)
    orphans = remove_orphans()
    print(f"{count} rows OK, {len(orphans)} orphans", flush=True)


if __name__ == "__main__":
    main()
