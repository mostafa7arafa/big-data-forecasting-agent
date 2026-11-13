"""Eval target for the single-turn agent set: run a case, sync the dataset, summarize.

``run_case`` returns ``outputs`` in the shape ``evals/graders/agent.py`` expects.
Everything here is local and network-free except ``sync_dataset``, which only
calls the LangSmith client it is given.
"""

import asyncio
import hashlib
import json
import re
import time
from collections import defaultdict
from collections.abc import Callable
from datetime import date, datetime
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from analyst.context import AnalystContext
from analyst.data import registry
from analyst.verify.trace import collect_handles
from evals.graders.agent import GATES, GRADERS, passed

DATASET_NAME = "analyst-single-turn"
_TAG = re.compile(r"v(\d+)-([0-9a-f]{8})")


def example_id(row_id: str) -> UUID:
    return uuid5(NAMESPACE_URL, "analyst-single-turn/" + row_id)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _handle_entries(messages: list, store) -> dict[str, dict]:
    """Every collected handle (plus each forecast's input handle) as a grader entry."""
    entries: dict[str, dict] = {}

    def add(handle_id: str, produced_by: str | None = None, governed: bool | None = None) -> None:
        if handle_id in entries or not store.exists(handle_id):
            return
        card = store.card(handle_id)
        entry = {
            **card.meta,
            "handle_id": handle_id,
            "produced_by": produced_by or card.produced_by,
            "governed": card.governed if governed is None else governed,
        }
        if entry["produced_by"] == "forecast":
            entry["forecast"] = _jsonable(store.load(handle_id).to_pylist())
        entries[handle_id] = _jsonable(entry)
        source = card.meta.get("input_handle")
        if entry["produced_by"] == "forecast" and isinstance(source, str):
            add(source)

    for info in collect_handles(messages, store).values():
        add(info.handle_id, info.produced_by, info.governed)
    return entries


def _tools(messages: list) -> list[str]:
    return [
        tc["name"]
        for m in messages
        if isinstance(m, AIMessage)
        for tc in (m.tool_calls or [])
    ]


def _usage(messages: list) -> dict[str, int]:
    total = {"input_tokens": 0, "output_tokens": 0}
    for m in messages:
        if isinstance(m, AIMessage) and m.usage_metadata:
            for key in total:
                total[key] += int(m.usage_metadata.get(key) or 0)
    return total


async def run_case(inputs: dict, *, model: str | None, graph=None) -> dict:
    """Run one eval row through the graph; an exception becomes ``{"error": ...}``."""
    if graph is None:
        from analyst.graph import graph as default_graph

        graph = default_graph
    started = time.perf_counter()
    try:
        context = AnalystContext.from_settings(
            cutoff=inputs.get("cutoff"),
            model=model,
            background=False,
            question_id=inputs["id"],
        )
        state = await graph.ainvoke(
            {"messages": [HumanMessage(inputs["question"])]}, context=context
        )
        messages = list(state.get("messages") or [])
        store = registry.get_service(context.dataset, context.cutoff).store
        handles = await asyncio.to_thread(_handle_entries, messages, store)
        answer = state.get("answer")
        verification = state.get("verification")
        if verification is None and answer is not None:
            verification = answer.verification
        return {
            "answer": answer.model_dump(mode="json") if answer is not None else {},
            "verification": verification.model_dump(mode="json") if verification else None,
            "tools": _tools(messages),
            "handles": handles,
            "latency_s": time.perf_counter() - started,
            "usage": _usage(messages),
            "tool_bytes": sum(len(str(m.content)) for m in messages if isinstance(m, ToolMessage)),
        }
    except Exception as exc:  # noqa: BLE001 - every failure is a failed case
        return {"error": repr(exc)}


# --- LangSmith dataset sync ------------------------------------------------------


def content_hash(rows: list[dict]) -> str:
    lines = sorted(json.dumps(r, sort_keys=True, ensure_ascii=False) for r in rows)
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def _example(row: dict) -> dict:
    return {
        "id": example_id(row["id"]),
        "inputs": {"id": row["id"], "question": row["question"], "cutoff": row["cutoff"]},
        "outputs": row,
        "metadata": {"category": row["category"], "tier": row["tier"]},
    }


def sync_dataset(client, rows: list[dict]) -> str:
    """Upsert the rows as examples; tag a new dataset version ``v<n>-<hash8>`` on change."""
    if not client.has_dataset(dataset_name=DATASET_NAME):
        client.create_dataset(DATASET_NAME, description="Single-turn analyst eval set")
    have = {e.id for e in client.list_examples(dataset_name=DATASET_NAME)}
    examples = [_example(r) for r in rows]
    new = [e for e in examples if e["id"] not in have]
    old = [e for e in examples if e["id"] in have]
    if new:
        client.create_examples(dataset_name=DATASET_NAME, examples=new)
    if old:
        client.update_examples(dataset_name=DATASET_NAME, updates=old)
    digest = content_hash(rows)[:8]
    tags = [
        t
        for v in client.list_dataset_versions(dataset_name=DATASET_NAME)
        for t in (v.tags or [])
        if _TAG.fullmatch(t)
    ]
    for tag in tags:
        if tag.endswith("-" + digest):
            return tag
    number = max((int(_TAG.fullmatch(t).group(1)) for t in tags), default=0) + 1
    tag = f"v{number}-{digest}"
    latest = client.read_dataset_version(dataset_name=DATASET_NAME, tag="latest")
    client.update_dataset_tag(dataset_name=DATASET_NAME, as_of=latest.as_of, tag=tag)
    return tag


# --- evaluators and local runs ---------------------------------------------------


def evaluators() -> list[Callable[[dict, dict], dict]]:
    """The six graders plus ``passed`` as LangSmith ``(outputs, reference_outputs)`` evaluators."""
    out: list[Callable[[dict, dict], dict]] = []
    for grader in GRADERS:

        def ev(outputs: dict, reference_outputs: dict, _g=grader) -> dict:
            return _g(outputs or {}, reference_outputs)

        ev.__name__ = grader.__name__.removeprefix("grade_")
        out.append(ev)

    def passed_ev(outputs: dict, reference_outputs: dict) -> dict:
        results = [g(outputs or {}, reference_outputs) for g in GRADERS]
        failed = [r["key"] for r in results if r["key"] in GATES and r["score"] is False]
        return {
            "key": "passed",
            "score": passed(results),
            "comment": "all gates pass" if not failed else "failed: " + ", ".join(failed),
        }

    passed_ev.__name__ = "passed"
    out.append(passed_ev)
    return out


async def run_local(
    rows: list[dict],
    *,
    model: str | None,
    graph_factory: Callable[[], Any] | None = None,
    concurrency: int = 3,
) -> list[dict]:
    """Run and grade rows locally; each result has id, category, outputs and grader results."""
    sem = asyncio.Semaphore(concurrency)

    async def one(row: dict) -> dict:
        async with sem:
            graph = graph_factory() if graph_factory else None
            outputs = await run_case(row, model=model, graph=graph)
        results = [g(outputs, row) for g in GRADERS]
        return {
            "id": row["id"],
            "category": row["category"],
            "outputs": outputs,
            "results": results,
            "passed": passed(results),
        }

    return list(await asyncio.gather(*(one(r) for r in rows)))


def summarize(results: list[dict]) -> dict:
    """Pass rate overall, per category, per gate (over rows where it applies), forecast skill."""
    def rate(flags: list[bool]) -> float | None:
        return sum(flags) / len(flags) if flags else None

    by_cat: dict[str, list[bool]] = defaultdict(list)
    by_gate: dict[str, list[bool]] = defaultdict(list)
    skill: list[float] = []
    for r in results:
        by_cat[r["category"]].append(r["passed"])
        for res in r["results"]:
            if res["key"] in GATES and res["score"] is not None:
                by_gate[res["key"]].append(res["score"] is True)
            if res["key"] == "forecast_skill" and isinstance(res["score"], float):
                skill.append(res["score"])
    return {
        "n": len(results),
        "overall": rate([r["passed"] for r in results]),
        "by_category": {k: rate(v) for k, v in sorted(by_cat.items())},
        "by_gate": {g: rate(by_gate[g]) for g in GATES if g in by_gate},
        "mean_forecast_wape": sum(skill) / len(skill) if skill else None,
    }
