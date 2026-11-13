"""The agent eval runner: target shape, dataset sync, local summary (no network)."""

import asyncio
import json
import re
from datetime import datetime
from types import SimpleNamespace

from analyst.graph import build_graph
from evals.graders.agent import GATES, GRADERS, passed
from evals.target import (
    DATASET_NAME,
    content_hash,
    evaluators,
    example_id,
    run_case,
    run_local,
    summarize,
    sync_dataset,
)
from tests.unit.test_graph import (  # noqa: F401  (events_service is an autouse fixture)
    COUNT,
    ScriptedModel,
    call,
    cite_total,
    events_service,
    no_plan,
    tool_turn,
)

ROW = {
    "id": "x001", "tier": 1, "category": "lookup", "question": "how many events?",
    "cutoff": None, "gold_sql": "", "key_values": [], "tolerance": None,
    "gold_check": [], "expected_tools": ["query_metrics"], "expected_label": "verified",
    "notes": "",
}


def scripted_graph(*script):
    return build_graph(model=ScriptedModel(script=list(script)))


def test_example_id_stable():
    assert example_id("a001") == example_id("a001") != example_id("a002")


def test_run_case_with_scripted_model():
    graph = scripted_graph(no_plan(), tool_turn(call("query_metrics", COUNT, "c1")), cite_total)
    out = asyncio.run(run_case(ROW, model=None, graph=graph))
    assert {"answer", "verification", "tools", "handles"} <= set(out)
    assert out["verification"]["label"] == "verified"
    assert "query_metrics" in out["tools"]
    assert out["handles"]
    (entry,) = [e for e in out["handles"].values() if e["produced_by"] == "query_metrics"]
    assert entry["query"]["metrics"] == ["event_count"] and entry["governed"] is True
    json.dumps(out)  # LangSmith needs JSON-serialisable outputs
    assert out["latency_s"] >= 0
    results = [g(out, ROW) for g in GRADERS]
    assert passed(results)


def test_run_case_exception_fails_all_gates():
    class Boom:
        async def ainvoke(self, *a, **k):
            raise RuntimeError("boom")

    out = asyncio.run(run_case(ROW, model=None, graph=Boom()))
    assert out == {"error": "RuntimeError('boom')"}
    row = {**ROW, "key_values": [{"value": 1.0, "label": "n"}], "gold_check": ["declined()"]}
    results = [g(out, row) for g in GRADERS]
    assert not passed(results)
    assert all(r["score"] is False for r in results if r["key"] in GATES and r["key"] != "grounding")


class FakeClient:
    def __init__(self):
        self.exists = False
        self.examples: dict = {}
        self.tags: list[tuple[datetime, str]] = []
        self.versions = 0

    def has_dataset(self, *, dataset_name):
        return self.exists

    def create_dataset(self, dataset_name, **kw):
        self.exists = True
        return SimpleNamespace(id="ds")

    def list_examples(self, *, dataset_name, **kw):
        return [SimpleNamespace(id=i) for i in self.examples]

    def create_examples(self, *, dataset_name, examples):
        for e in examples:
            self.examples[e["id"]] = e
        self.versions += 1

    def update_examples(self, *, dataset_name, updates):
        for e in updates:
            self.examples[e["id"]] = e
        self.versions += 1

    def list_dataset_versions(self, *, dataset_name):
        return [SimpleNamespace(tags=[t], as_of=a) for a, t in self.tags]

    def read_dataset_version(self, *, dataset_name, tag):
        return SimpleNamespace(tags=[tag], as_of=datetime(2026, 1, 1, 0, 0, self.versions))

    def update_dataset_tag(self, *, dataset_name, as_of, tag):
        self.tags.append((as_of, tag))


def test_sync_dataset_tags_only_on_change():
    client = FakeClient()
    rows = [ROW, {**ROW, "id": "x002"}]
    tag = sync_dataset(client, rows)
    assert re.fullmatch(r"v1-[0-9a-f]{8}", tag)
    assert sync_dataset(client, rows) == tag
    assert len(client.tags) == 1 and len(client.examples) == 2
    changed = [ROW, {**ROW, "id": "x002", "question": "other?"}]
    tag2 = sync_dataset(client, changed)
    assert tag2.startswith("v2-") and tag2 != tag
    assert len(client.tags) == 2
    assert DATASET_NAME == "analyst-single-turn"


def test_content_hash_order_independent():
    assert content_hash([ROW, {**ROW, "id": "b"}]) == content_hash([{**ROW, "id": "b"}, ROW])


def test_evaluators_return_langsmith_results():
    evs = evaluators()
    assert [e.__name__ for e in evs][-1] == "passed"
    out = {"error": "x"}
    res = [e(out, ROW) for e in evs]
    assert all({"key", "score", "comment"} <= set(r) for r in res)
    assert res[-1] == {"key": "passed", "score": False, "comment": res[-1]["comment"]}


def test_run_local_and_summarize():
    def graph_for():
        return scripted_graph(no_plan(), tool_turn(call("query_metrics", COUNT, "c1")), cite_total)

    cases = [{**ROW, "id": "x001"}, {**ROW, "id": "x002"}]
    graphs = iter([graph_for(), graph_for()])
    results = asyncio.run(
        run_local(cases, model=None, graph_factory=lambda: next(graphs), concurrency=1)
    )
    assert [r["id"] for r in results] == ["x001", "x002"]
    summary = summarize(results)
    assert summary["overall"] == 1.0
    assert summary["by_category"] == {"lookup": 1.0}
    assert summary["by_gate"]["label"] == 1.0
    assert summary["n"] == 2
