"""The agent eval runner: target shape, dataset sync, local summary (no network)."""

import asyncio
import json
import re
from datetime import datetime
from types import SimpleNamespace

import pytest

from analyst.graph import build_graph
from evals.graders.agent import GATES, GRADERS, passed
from evals.target import (
    DATASET_NAME,
    content_hash,
    evaluators,
    example_id,
    format_summary,
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
        self.writes = 0

    def has_dataset(self, *, dataset_name):
        return self.exists

    def create_dataset(self, dataset_name, **kw):
        self.exists = True
        return SimpleNamespace(id="ds")

    def list_examples(self, *, dataset_name, **kw):
        return [SimpleNamespace(id=i, **{k: e[k] for k in ("inputs", "outputs", "metadata")}) for i, e in self.examples.items()]

    def create_examples(self, *, dataset_name, examples):
        for e in examples:
            self.examples[e["id"]] = e
        self.versions += 1
        self.writes += 1

    def update_examples(self, *, dataset_name, updates):
        for e in updates:
            self.examples[e["id"]] = e
        self.versions += 1
        self.writes += 1

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


def test_sync_dataset_skips_unchanged_and_lists_stale(capsys):
    client = FakeClient()
    rows = [ROW, {**ROW, "id": "x002"}]
    sync_dataset(client, rows)
    writes = client.writes
    sync_dataset(client, rows)
    assert client.writes == writes  # identical rows: no create, no update
    sync_dataset(client, [ROW])
    assert client.writes == writes
    assert f"stale ids: {example_id('x002')}" in capsys.readouterr().out
    sync_dataset(client, [{**ROW, "question": "changed?"}])
    assert client.writes == writes + 1


def test_summary_metrics_and_format():
    def res(key, score, comment=""):
        return {"key": key, "score": score, "comment": comment}

    def row(cat, ok, latency, tool_bytes, usage, forecast=None):
        results = [res("label", ok), res("trajectory", ok)]
        if forecast is not None:
            results.append(res("forecast_skill", forecast, "wape=0.1 coverage_80=0.80 coverage_95=0.90"))
        return {
            "category": cat, "passed": ok, "results": results,
            "outputs": {"latency_s": latency, "tool_bytes": tool_bytes, "usage": usage},
        }

    rows = [
        row("lookup", True, 1.0, 100, {"input_tokens": 10, "output_tokens": 5, "cost": 0.01}),
        row("lookup", False, 3.0, 900, {"input_tokens": 20, "output_tokens": 15, "cost": 0.03}),
        row("forecast", True, 5.0, 300, {"input_tokens": 30, "output_tokens": 20}, forecast=0.1),
    ]
    s = summarize(rows)
    assert s["median_latency_s"] == 3.0 and s["p95_latency_s"] == 5.0
    assert s["mean_cost_usd"] == pytest.approx(0.02)
    assert s["mean_tokens"] == pytest.approx((15 + 35 + 50) / 3)
    assert s["peak_tool_bytes"] == 900
    assert s["mean_trajectory"] == pytest.approx(2 / 3)
    assert s["mean_forecast_wape"] == pytest.approx(0.1)
    assert s["mean_coverage_80"] == pytest.approx(0.8) and s["mean_coverage_95"] == pytest.approx(0.9)
    text = format_summary(s, experiment="exp-1", url="https://x/y")
    assert "exp-1 (https://x/y)" in text and "p95 5.0s" in text and "$0.0200" in text
    assert "n/a" in format_summary(summarize([{"category": "a", "passed": True, "results": []}]))


# --- final fix wave ------------------------------------------------------------


def _verdict(label, reason="r"):
    return {"category": "lookup", "passed": label != "not_verified", "results": [],
            "outputs": {"verification": {"label": label, "reason": reason}}}


def test_label_distribution_summary():
    rows = [
        _verdict("verified"),
        _verdict("verified"),
        _verdict("verified_off_model"),
        _verdict("not_verified", "untraced number 12%"),
        _verdict("not_verified", "untraced number 12%"),
        _verdict("not_verified", "no evidence cited"),
        {"category": "lookup", "passed": False, "results": [], "outputs": {"error": "boom"}},
    ]
    s = summarize(rows)
    assert s["label_distribution"] == {
        "verified": 2, "verified_off_model": 1, "not_verified": 3, "no_label": 1,
    }
    assert s["not_verified_reasons"] == [("untraced number 12%", 2), ("no evidence cited", 1)]
    text = format_summary(s)
    line = next(x for x in text.splitlines() if x.startswith("Labels:"))
    assert "verified 2" in line and "verified_off_model 1" in line and "not_verified 3" in line
    assert "no_label 1" in line
    assert "untraced number 12% (2)" in text


def _eval_script():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "scripts/eval_agent.py"
    spec = importlib.util.spec_from_file_location("eval_agent_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_experiment_prefix_names_the_resolved_model(monkeypatch):
    from analyst.config import get_settings

    script = _eval_script()
    monkeypatch.setenv("ANALYST_MODEL", "openai:gpt-6-luna")
    get_settings.cache_clear()
    try:
        assert script.experiment_prefix(None, "abc1234") == "openai-gpt-6-luna-abc1234"
        assert script.experiment_prefix("anthropic:claude-x", "abc1234") == "anthropic-claude-x-abc1234"
    finally:
        get_settings.cache_clear()


def test_limit_must_be_positive():
    script = _eval_script()
    for bad in ("0", "-1"):
        with pytest.raises(SystemExit):
            script.parse_args(["--limit", bad])
    assert script.parse_args(["--limit", "2"]).limit == 2
    assert script.parse_args([]).limit is None


def test_dry_run_repeats_reps_times(monkeypatch, capsys):
    script = _eval_script()
    calls = []

    async def fake_run_local(rows, *, model):
        calls.append([r["id"] for r in rows])
        return [
            {"id": r["id"], "category": r["category"], "outputs": {}, "results": [], "passed": True}
            for r in rows
        ]

    monkeypatch.setattr(script, "run_local", fake_run_local)
    rows = [{"id": "a001", "category": "lookup"}, {"id": "a002", "category": "lookup"}]
    script.run_dry(rows, None, reps=3)
    assert calls == [["a001", "a002"]] * 3
    out = capsys.readouterr().out
    assert "over 6 runs" in out
    assert "rep 3/3" in out
    monkeypatch.setattr(script, "load_rows", lambda path: rows)
    calls.clear()
    script.main(["--dry-run", "--reps", "2"])
    assert len(calls) == 2
