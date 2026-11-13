import hashlib
import json
import subprocess
import sys

import pyarrow as pa
import pytest

from analyst.data.results import ResultStore


def _handle(source: str, produced_by: str, meta: dict | None = None) -> str:
    meta = meta or {}
    payload = source + produced_by + json.dumps(meta, sort_keys=True)
    return "r_" + hashlib.sha256(payload.encode()).hexdigest()[:10]


def test_roundtrip(tmp_path):
    table = pa.table({"a": [1, 2], "b": ["x", "y"]})
    store = ResultStore(tmp_path)
    card = store.save(
        table,
        produced_by="run_sql",
        source="select 1",
        governed=True,
        elapsed_s=0.1,
    )
    loaded = store.load(card.handle_id)
    assert loaded.equals(table)
    assert store.card(card.handle_id) == card


def test_preview_max_10_rows(tmp_path):
    table = pa.table({"n": list(range(1000))})
    store = ResultStore(tmp_path)
    card = store.save(
        table,
        produced_by="run_sql",
        source="select n from generate_series",
        governed=False,
        elapsed_s=0.05,
    )
    assert str(9) in card.preview
    assert "| 10 |" not in card.preview and "| 10\n" not in card.preview
    body_lines = card.preview.strip().splitlines()
    sep_idx = next(i for i, ln in enumerate(body_lines) if ln.startswith("|---"))
    data_lines = [ln for ln in body_lines[sep_idx + 1 :] if ln.startswith("|")]
    assert len(data_lines) == 10


def test_stats(tmp_path):
    table = pa.table({"v": pa.array([1, 2, None, 5], type=pa.float64())})
    store = ResultStore(tmp_path)
    card = store.save(
        table,
        produced_by="run_sql",
        source="select v",
        governed=True,
        elapsed_s=0.01,
    )
    stats = card.stats[0]
    assert stats.name == "v"
    assert stats.nulls == 1
    assert stats.min == 1.0
    assert stats.max == 5.0
    assert stats.mean == pytest.approx((1 + 2 + 5) / 3)


def test_render_meta_capped():
    from analyst.data.results import SummaryCard

    card = SummaryCard(
        handle_id="r_meta",
        rows=1,
        columns=[("a", "int64")],
        preview="| a |\n|---|\n| 1 |",
        stats=[],
        produced_by="forecast",
        source="x",
        governed=True,
        elapsed_s=0.1,
        meta={"blob": "x" * 5000},
    )
    rendered = card.render()
    assert len(rendered) <= 4096
    assert "preview:" in rendered
    assert "| 1 |" in rendered
    meta_line = next(line for line in rendered.splitlines() if line.startswith("meta:"))
    suffix = " …(meta truncated)"
    assert meta_line.endswith(suffix)
    assert len(meta_line) == 1000 + len(suffix)


def test_render_meta_short_keys_survive_cap():
    from analyst.data.results import SummaryCard

    card = SummaryCard(
        handle_id="r_meta_order",
        rows=1,
        columns=[("a", "int64")],
        preview="| a |\n|---|\n| 1 |",
        stats=[],
        produced_by="forecast",
        source="x",
        governed=True,
        elapsed_s=0.1,
        meta={"a_long": ["row"] * 400, "z_short": 1},
    )
    rendered = card.render()
    meta_line = next(line for line in rendered.splitlines() if line.startswith("meta:"))
    assert '"z_short": 1' in meta_line
    assert meta_line.endswith(" …(meta truncated)")


def test_display_rounding_preview_and_stats(tmp_path):
    table = pa.table(
        {
            "n": pa.array([2, 3], type=pa.int64()),
            "big": pa.array([1.236, 8.1], type=pa.float64()),
            "small": pa.array([0.00123456, 0.000123456], type=pa.float64()),
        }
    )
    store = ResultStore(tmp_path)
    card = store.save(
        table,
        produced_by="run_sql",
        source="select rounded",
        governed=True,
        elapsed_s=0.0,
    )
    rendered = card.render()
    assert "1.24" in rendered
    assert "8.10" in rendered
    assert "0.001235" in rendered
    assert "0.0001235" in rendered
    assert "1.236" not in rendered
    assert "0.00123456" not in rendered
    assert "0.000123456" not in rendered
    assert "2.00" not in rendered
    loaded = store.load(card.handle_id)
    assert loaded.column("big").to_pylist() == pytest.approx([1.236, 8.1])
    assert loaded.column("n").to_pylist() == [2, 3]


def test_elapsed_display_rounding_preserves_storage(tmp_path):
    store = ResultStore(tmp_path)
    card = store.save(pa.table({"x": [1]}), produced_by="run_sql", source="elapsed",
                      governed=False, elapsed_s=1.236789)
    assert "elapsed_s: 1.24\n" in card.render()
    assert store.card(card.handle_id).elapsed_s == 1.236789


def test_render_capped(tmp_path):
    cols = {f"c{i}": [float(i)] * 10 for i in range(200)}
    table = pa.table(cols)
    store = ResultStore(tmp_path)
    card = store.save(
        table,
        produced_by="run_sql",
        source="select wide",
        governed=False,
        elapsed_s=1.0,
    )
    rendered = card.render()
    assert len(rendered) <= 4096
    assert rendered.splitlines()[-1] == "…(truncated)"
    assert "preview:" in rendered


def test_same_source_same_handle(tmp_path):
    table = pa.table({"a": [1]})
    store = ResultStore(tmp_path)
    kw = {
        "produced_by": "run_sql",
        "source": "select 1",
        "governed": True,
        "elapsed_s": 0.0,
        "meta": {"k": "v"},
    }
    c1 = store.save(table, **kw)
    c2 = store.save(table, **kw)
    assert c1.handle_id == c2.handle_id
    assert c1.handle_id == _handle("select 1", "run_sql", {"k": "v"})


def test_unknown_handle(tmp_path):
    store = ResultStore(tmp_path)
    handle = "r_deadbeef00"
    with pytest.raises(KeyError, match=f"unknown result handle {handle}"):
        store.load(handle)
    with pytest.raises(KeyError, match=f"unknown result handle {handle}"):
        store.card(handle)


@pytest.mark.parametrize("method", ["load", "card", "exists"])
@pytest.mark.parametrize("handle", ["../x", "absolute", "r_ABCDEF1234", "r_1234567890\n"])
def test_invalid_handle_rejected(tmp_path, method, handle):
    store = ResultStore(tmp_path / "store")
    if handle == "absolute":
        handle = str(tmp_path / "outside")
    # Put real artifacts at the escaped location: rejecting missing files alone
    # would not prove that the store enforces its identifier boundary.
    if handle == "../x" or handle.startswith("/"):
        import pyarrow.parquet as pq

        escaped = tmp_path / "store" / f"{handle}.parquet"
        pq.write_table(pa.table({"secret": [42]}), escaped)
        (escaped.with_suffix(".card.json")).write_text("{}")
    with pytest.raises(KeyError, match="unknown result handle"):
        getattr(store, method)(handle)


def test_unknown_valid_handle_does_not_exist(tmp_path):
    assert not ResultStore(tmp_path).exists("r_deadbeef00")


def test_empty_table_saves(tmp_path):
    table = pa.table({"a": pa.array([], type=pa.int64())})
    store = ResultStore(tmp_path)
    card = store.save(
        table,
        produced_by="run_sql",
        source="select where false",
        governed=True,
        elapsed_s=0.0,
    )
    assert card.rows == 0
    assert card.stats[0].nulls == 0
    assert store.load(card.handle_id).equals(table)


def test_nested_types_save(tmp_path):
    table = pa.table(
        {
            "lst": pa.array([[1, 2], [3]], type=pa.list_(pa.int64())),
            "st": pa.array(
                [{"a": 1}, {"a": 2}],
                type=pa.struct([pa.field("a", pa.int64())]),
            ),
        }
    )
    store = ResultStore(tmp_path)
    card = store.save(
        table,
        produced_by="run_sql",
        source="select nested",
        governed=False,
        elapsed_s=0.0,
    )
    assert card.stats[0].min is None and card.stats[0].max is None
    assert card.stats[1].min is None and card.stats[1].max is None
    assert store.load(card.handle_id).equals(table)


def test_null_typed_column(tmp_path):
    table = pa.table({"n": pa.array([None, None], type=pa.null())})
    store = ResultStore(tmp_path)
    card = store.save(
        table,
        produced_by="run_sql",
        source="select nulls",
        governed=True,
        elapsed_s=0.0,
    )
    assert card.stats[0].nulls == 2
    assert card.stats[0].min is None
    assert card.stats[0].max is None
    assert card.stats[0].mean is None


def test_save_in_fresh_process(tmp_path):
    root = tmp_path / "fresh"
    script = f"""
import pyarrow as pa
from analyst.data.results import ResultStore
store = ResultStore({str(root)!r})
table = pa.table({{"x": [1, 2]}})
card = store.save(
    table,
    produced_by="run_sql",
    source="fresh",
    governed=False,
    elapsed_s=0.0,
)
assert store.load(card.handle_id).equals(table)
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_orphan_parquet_without_card_is_unknown(tmp_path):
    import pyarrow.parquet as pq

    pq.write_table(pa.table({"a": [1]}), tmp_path / "r_deadbeef00.parquet")
    store = ResultStore(tmp_path)
    with pytest.raises(KeyError, match="unknown result handle r_deadbeef00"):
        store.load("r_deadbeef00")
