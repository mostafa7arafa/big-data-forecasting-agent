from datetime import date

import pytest
from langchain.tools import ToolRuntime

from analyst.context import AnalystContext
from analyst.data import registry
from analyst.data.service import DataService
from analyst.errors import InvalidRequestError
from analyst.state import Answer, Claim
from analyst.tools import (
    DATA_TOOLS,
    describe,
    forecast,
    make_chart,
    query_metrics,
    read_knowledge,
    run_sql,
)

MONTH = {"metrics": ["total_amount"], "grain": "month"}


@pytest.fixture(autouse=True)
def events_service(synthetic_dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "load_dataset", lambda name: synthetic_dataset)
    original = DataService.__init__

    def init(self, dataset, *, cutoff=None, store_root=None):
        original(self, dataset, cutoff=cutoff, store_root=tmp_path / f"s-{cutoff}")

    monkeypatch.setattr(DataService, "__init__", init)
    registry.clear_services()
    yield
    registry.clear_services()


def make_runtime(events=None, **ctx):
    sink = events if events is not None else []
    return ToolRuntime(
        state={},
        context=AnalystContext(dataset="events", **ctx),
        config={},
        stream_writer=sink.append,
        tool_call_id="t1",
        store=None,
    )


def invoke(tool, args, events=None, **ctx):
    call = {
        "name": tool.name,
        "args": {**args, "runtime": make_runtime(events, **ctx)},
        "id": "t1",
        "type": "tool_call",
    }
    return tool.invoke(call)


def test_tool_names():
    assert [t.name for t in DATA_TOOLS] == [
        "query_metrics",
        "run_sql",
        "describe",
        "forecast",
        "make_chart",
        "read_knowledge",
    ]


def test_runtime_hidden_from_model_schema():
    for t in DATA_TOOLS:
        assert "runtime" not in t.tool_call_schema.model_json_schema().get(
            "properties", {}
        )


def test_query_metrics_returns_card_and_artifact():
    msg = invoke(query_metrics, MONTH)
    assert msg.status == "success"
    assert (
        msg.artifact["handle_id"].startswith("r_") and msg.artifact["governed"] is True
    )
    assert msg.artifact["produced_by"] == "query_metrics"
    assert len(msg.content) <= 4096 and "r_" in msg.content


def test_run_sql_returns_ungoverned_artifact():
    msg = invoke(run_sql, {"sql": "SELECT count(*) AS n FROM events"})
    assert msg.artifact["governed"] is False


def test_run_sql_rejected_query_is_error_not_raise():
    msg = invoke(run_sql, {"sql": "DROP TABLE x"})
    assert msg.status == "error" and "HINT:" in msg.content
    assert msg.content.startswith("ERROR: ")


def test_unknown_handle_is_tool_error():
    msg = invoke(describe, {"target": "r_0000000000"})
    assert msg.status == "error"


def test_describe_text_and_handle():
    msg = invoke(describe, {"target": "metrics"})
    assert msg.status == "success" and msg.artifact is None
    handle = invoke(query_metrics, MONTH).artifact["handle_id"]
    msg = invoke(describe, {"target": handle})
    assert msg.artifact["handle_id"] == handle


def test_internal_error_is_masked(monkeypatch):
    def boom(self, sql):
        raise RuntimeError("secret path /x")

    monkeypatch.setattr(DataService, "run_sql", boom)
    msg = invoke(run_sql, {"sql": "SELECT 1"})
    assert msg.status == "error"
    assert msg.content == "ERROR: internal error in run_sql"


FC = {"value_col": "total_amount", "freq": "MS"}


def _handle():
    return invoke(query_metrics, MONTH).artifact["handle_id"]


def test_forecast_slow_refused_when_background_enabled(monkeypatch):
    monkeypatch.setattr(DataService, "estimate_forecast_s", lambda *a, **k: 60.0)
    monkeypatch.setattr(DataService, "forecast_is_cached", lambda *a, **k: False)

    def fail(*a, **k):
        raise AssertionError("must not compute")

    monkeypatch.setattr(DataService, "forecast", fail)
    msg = invoke(forecast, {"handle_id": _handle(), **FC}, background=True)
    assert msg.status == "error"
    assert "start_async_task" in msg.content and "worker" in msg.content


def test_forecast_runs_inline_when_background_disabled(monkeypatch):
    from analyst.data.results import SummaryCard

    handle = _handle()
    monkeypatch.setattr(DataService, "estimate_forecast_s", lambda *a, **k: 60.0)
    monkeypatch.setattr(DataService, "forecast_is_cached", lambda *a, **k: False)
    service = registry.get_service("events", None)
    card = service.store.card(handle)
    assert isinstance(card, SummaryCard)
    monkeypatch.setattr(DataService, "forecast", lambda self, *a, **k: card)
    msg = invoke(forecast, {"handle_id": handle, **FC}, background=False)
    assert msg.status == "success" and msg.artifact["handle_id"] == handle


def test_forecast_emits_progress_events(monkeypatch):
    handle = _handle()
    service = registry.get_service("events", None)
    card = service.store.card(handle)

    def fake(self, *a, progress=None, **k):
        progress("fitting")
        return card

    monkeypatch.setattr(DataService, "forecast", fake)
    monkeypatch.setattr(DataService, "forecast_is_cached", lambda *a, **k: True)
    events: list = []
    invoke(forecast, {"handle_id": handle, **FC}, events=events)
    assert any(e["type"] == "progress" and e["tool"] == "forecast" for e in events)
    assert any(e["message"] == "fitting" for e in events)


def test_make_chart_validates_handles():
    args = {
        "kind": "line",
        "title": "T",
        "handle": "r_0000000000",
        "x": "period",
        "y": "total_amount",
    }
    assert invoke(make_chart, args).status == "error"
    handle = _handle()
    msg = invoke(make_chart, {**args, "handle": handle})
    assert msg.status == "success"
    assert msg.artifact["chart"]["handle"] == handle
    assert "\n" not in msg.content
    bad = invoke(make_chart, {**args, "handle": handle, "y": "nope"})
    assert bad.status == "error"


def test_read_knowledge_respects_cutoff():
    msg = invoke(read_knowledge, {"topic": "recap"}, cutoff=date(2024, 2, 1))
    assert msg.content == "recap unavailable under a cutoff"
    assert "Synthetic" in invoke(read_knowledge, {"topic": "recap"}).content
    assert invoke(read_knowledge, {"topic": "nope"}).status == "error"


def test_context_from_settings(monkeypatch):
    import analyst.context as ctx_mod
    from analyst.config import Settings

    monkeypatch.setattr(
        ctx_mod, "get_settings", lambda: Settings(dataset=None, _env_file=None)
    )
    with pytest.raises(InvalidRequestError):
        AnalystContext.from_settings()
    monkeypatch.setattr(
        ctx_mod, "get_settings", lambda: Settings(dataset="x", _env_file=None)
    )
    assert AnalystContext.from_settings(background=False).dataset == "x"
    assert AnalystContext.from_settings(dataset="y").dataset == "y"


def test_answer_schema_defaults():
    a = Answer(narrative="n", claims=[Claim(text="t", type="descriptive")])
    assert a.verification is None and a.charts == [] and a.claims[0].values == []


def _array_nodes(node):
    if isinstance(node, dict):
        if node.get("type") == "array" or "prefixItems" in node:
            yield node
        for value in node.values():
            yield from _array_nodes(value)
    elif isinstance(node, list):
        for value in node:
            yield from _array_nodes(value)


@pytest.mark.parametrize("tool", DATA_TOOLS, ids=lambda t: t.name)
def test_openai_tool_schema_arrays_have_items(tool):
    from langchain_core.utils.function_calling import convert_to_openai_tool

    schema = convert_to_openai_tool(tool)
    for node in _array_nodes(schema):
        assert "items" in node and "prefixItems" not in node, (tool.name, node)


def test_query_metrics_time_range_as_list():
    args = {**MONTH, "time_range": ["2024-01-01", "2024-02-01"]}
    msg = invoke(query_metrics, args)
    assert msg.status == "success", msg.content


def test_query_metrics_time_range_wrong_length_is_rejected():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        invoke(query_metrics, {**MONTH, "time_range": ["2024-01-01"]})
