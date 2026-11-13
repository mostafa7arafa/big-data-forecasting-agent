"""Run the single-turn eval set through the agent graph.

``--dry-run`` runs and grades locally (no LangSmith). Otherwise the rows are
synced to the LangSmith dataset ``analyst-single-turn`` and run as an experiment.
"""

import argparse
import asyncio
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from evals.graders.agent import load_rows
from evals.target import (
    DATASET_NAME,
    evaluators,
    example_id,
    format_summary,
    run_case,
    run_local,
    summarize,
    sync_dataset,
)

DATASET_FILE = ROOT / "evals/datasets/agent_single_turn.jsonl"


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "nogit"


def slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")


def resolved_model(model: str | None) -> str:
    """The model a run uses: ``--model``, else the configured ``Settings.model``."""
    if model:
        return model
    from analyst.config import get_settings

    return get_settings().model


def experiment_prefix(model: str | None, sha: str) -> str:
    """``<model slug>-<git sha>``, naming the resolved model, never "default"."""
    return f"{slug(resolved_model(model))}-{sha}"


def positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=None, help="model spec (default: provider default)")
    p.add_argument("--reps", type=positive_int, default=1, help="repetitions (dry runs too)")
    p.add_argument("--limit", type=positive_int, default=None, help="first N rows (>= 1)")
    p.add_argument("--ids", nargs="*", default=None, help="only these row ids")
    p.add_argument("--dry-run", action="store_true", help="run locally, no LangSmith")
    return p.parse_args(argv)


def select(rows: list[dict], ids: list[str] | None, limit: int | None) -> list[dict]:
    if ids:
        unknown = set(ids) - {r["id"] for r in rows}
        if unknown:
            raise SystemExit(f"unknown ids: {sorted(unknown)}")
        rows = [r for r in rows if r["id"] in set(ids)]
    return rows if limit is None else rows[:limit]


def run_dry(rows: list[dict], model: str | None, reps: int = 1) -> None:
    """Run and grade locally ``reps`` times; the summary covers every repetition."""
    results: list[dict] = []
    for rep in range(1, reps + 1):
        batch = asyncio.run(run_local(rows, model=model))
        for r in batch:
            failed = [x["key"] for x in r["results"] if x["score"] is False]
            error = r["outputs"].get("error")
            tag = f" rep {rep}/{reps}" if reps > 1 else ""
            print(f"{r['id']}{tag} {r['category']:<10} {'PASS' if r['passed'] else 'FAIL'} {error or ','.join(failed)}")
        results.extend(batch)
    print("\n" + format_summary(summarize(results)))


async def run_experiment(rows: list[dict], args: argparse.Namespace) -> None:
    from langsmith import Client, aevaluate

    client = Client()
    all_rows = load_rows(DATASET_FILE)
    tag = sync_dataset(client, all_rows)
    prefix = experiment_prefix(args.model, git_sha())
    if len(rows) == len(all_rows):
        data = DATASET_NAME
    else:
        data = list(client.list_examples(dataset_name=DATASET_NAME, example_ids=[example_id(r["id"]) for r in rows]))

    async def target(inputs: dict) -> dict:
        return await run_case(inputs, model=args.model)

    experiment = await aevaluate(
        target,
        data=data,
        evaluators=evaluators(),
        experiment_prefix=prefix,
        num_repetitions=args.reps,
        max_concurrency=3,
        metadata={"dataset_tag": tag, "model": resolved_model(args.model), "git_sha": git_sha()},
        client=client,
    )
    results = []
    async for item in experiment:
        reference = item["example"].outputs or {}
        scored = [
            {"key": r.key, "score": r.score, "comment": r.comment}
            for r in item["evaluation_results"]["results"]
        ]
        by_key = {r["key"]: r for r in scored}
        results.append(
            {
                "id": reference.get("id"),
                "category": reference.get("category", "?"),
                "results": scored,
                "outputs": item["run"].outputs or {},
                "passed": by_key.get("passed", {}).get("score") is True,
            }
        )
    print(f"dataset tag: {tag}; experiment prefix: {prefix}")
    name = getattr(experiment, "experiment_name", None)
    url = getattr(getattr(experiment, "_manager", None), "_experiment", None)
    url = getattr(url, "url", None)
    print(format_summary(summarize(results), experiment=name or prefix, url=url))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rows = select(load_rows(DATASET_FILE), args.ids, args.limit)
    if args.dry_run:
        run_dry(rows, args.model, reps=args.reps)
    else:
        asyncio.run(run_experiment(rows, args))


if __name__ == "__main__":
    main()
