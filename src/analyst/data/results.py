from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pydantic import BaseModel, Field, field_validator

from analyst.errors import UnknownNameError

_RENDER_BUDGET = 4096
_PREVIEW_ROWS = 10
_TRUNC_SUFFIX = "\n…(truncated)"
_PREVIEW_LABEL = "preview:\n"
_META_LINE_LIMIT = 1000
_META_SUFFIX = " …(meta truncated)"


class ColumnStats(BaseModel):
    name: str
    type: str
    nulls: int
    min: float | str | None
    max: float | str | None
    mean: float | None


class SummaryCard(BaseModel):
    handle_id: str
    rows: int
    columns: list[tuple[str, str]]
    preview: str
    stats: list[ColumnStats]
    produced_by: str
    source: str
    governed: bool
    elapsed_s: float
    notes: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("columns", mode="before")
    @classmethod
    def _columns_to_tuples(cls, value: object) -> list[tuple[str, str]]:
        if not value:
            return []
        if isinstance(value[0], (list, tuple)):
            return [(str(a), str(b)) for a, b in value]
        return value  # type: ignore[return-value]

    def render(self) -> str:
        header = self._render_header()
        stats = self._render_stats(self.stats)
        preview = self.preview
        preview_block = _PREVIEW_LABEL + preview
        truncated = False

        if len(header) + len(stats) + len(preview_block) > _RENDER_BUDGET:
            truncated = True
            preview = _truncate_preview_lines(
                preview, header, stats, extra=len(_PREVIEW_LABEL)
            )
            preview_block = _PREVIEW_LABEL + preview
            while (
                len(header) + len(stats) + len(preview_block) + len(_TRUNC_SUFFIX)
                > _RENDER_BUDGET
                and _preview_line_count(preview) > 0
            ):
                preview = _truncate_preview_lines(
                    preview,
                    header,
                    stats,
                    extra=len(_PREVIEW_LABEL),
                    max_rows=_preview_line_count(preview) - 1,
                )
                preview_block = _PREVIEW_LABEL + preview

        if len(header) + len(stats) + len(preview_block) > _RENDER_BUDGET:
            truncated = True
            kept_stats = list(self.stats)
            while kept_stats and len(header) + len(self._render_stats(kept_stats)) + len(
                preview_block
            ) + len(_TRUNC_SUFFIX) > _RENDER_BUDGET:
                kept_stats.pop()
            stats = self._render_stats(kept_stats)

        body = header + stats + preview_block
        if truncated:
            if len(body) + len(_TRUNC_SUFFIX) > _RENDER_BUDGET:
                body = body[: _RENDER_BUDGET - len(_TRUNC_SUFFIX)]
            body += _TRUNC_SUFFIX
        return body[:_RENDER_BUDGET]

    def _render_header(self) -> str:
        col_summary = ", ".join(f"{n}:{t}" for n, t in self.columns)
        lines = [
            f"handle: {self.handle_id}",
            f"rows: {self.rows}",
            f"columns: {col_summary}",
            f"produced_by: {self.produced_by}",
            f"governed: {self.governed}",
            f"elapsed_s: {self.elapsed_s:.2f}",
        ]
        if self.notes:
            lines.append(f"notes: {', '.join(self.notes)}")
        if self.meta:
            meta_line = f"meta: {json.dumps(_meta_by_value_length(self.meta))}"
            if len(meta_line) > _META_LINE_LIMIT:
                meta_line = meta_line[:_META_LINE_LIMIT] + _META_SUFFIX
            lines.append(meta_line)
        return "\n".join(lines) + "\n\n"

    def _render_stats(self, stats: list[ColumnStats]) -> str:
        if not stats:
            return ""
        out = ["stats:"]
        for s in stats:
            numeric = _is_numeric_type_str(s.type)
            integer = _is_integer_type_str(s.type)
            if numeric:
                out.append(
                    f"  {s.name} nulls={s.nulls} "
                    f"min={_format_stat_value(s.min, as_int=integer)} "
                    f"max={_format_stat_value(s.max, as_int=integer)} "
                    f"mean={_format_stat_value(s.mean, as_int=False)}"
                )
            else:
                out.append(
                    f"  {s.name} nulls={s.nulls} "
                    f"min={_format_stat_value(s.min, as_int=False)} "
                    f"max={_format_stat_value(s.max, as_int=False)}"
                )
        return "\n".join(out) + "\n\n"


class ResultStore:
    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        table: pa.Table,
        *,
        produced_by: str,
        source: str,
        governed: bool,
        elapsed_s: float,
        notes: list[str] = (),
        meta: dict | None = None,
    ) -> SummaryCard:
        meta = meta or {}
        handle_id = _make_handle(source, produced_by, meta)
        card = _build_card(
            handle_id,
            table,
            produced_by=produced_by,
            source=source,
            governed=governed,
            elapsed_s=elapsed_s,
            notes=list(notes),
            meta=meta,
        )
        pq_path = self._root / f"{handle_id}.parquet"
        card_path = self._root / f"{handle_id}.card.json"
        tag = uuid.uuid4().hex
        pq_tmp = self._root / f"{handle_id}.parquet.{tag}.tmp"
        card_tmp = self._root / f"{handle_id}.card.json.{tag}.tmp"
        index_path = self._index_path(produced_by, source)
        index_tmp = self._root / "index" / f"{index_path.name}.{tag}.tmp"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            pq.write_table(table, pq_tmp)
            card_tmp.write_text(
                json.dumps(_card_to_json(card), indent=2), encoding="utf-8"
            )
            index_tmp.write_text(json.dumps({"handle_id": handle_id}), encoding="utf-8")
            # Card last: its presence means the parquet is complete.
            os.replace(pq_tmp, pq_path)
            os.replace(card_tmp, card_path)
            os.replace(index_tmp, index_path)
        finally:
            for tmp in (pq_tmp, card_tmp, index_tmp):
                tmp.unlink(missing_ok=True)
        return card

    def lookup(self, produced_by: str, source: str) -> SummaryCard | None:
        try:
            data = json.loads(self._index_path(produced_by, source).read_text("utf-8"))
            handle_id = data["handle_id"]
            if not self.exists(handle_id):
                return None
            return self.card(handle_id)
        except (OSError, ValueError, KeyError):
            return None

    def _index_path(self, produced_by: str, source: str) -> Path:
        key = hashlib.sha256((produced_by + "\0" + source).encode()).hexdigest()
        return self._root / "index" / f"{key}.json"

    def load(self, handle_id: str) -> pa.Table:
        path = self._parquet_path(handle_id)
        return pq.read_table(path)

    def card(self, handle_id: str) -> SummaryCard:
        path = self._card_path(handle_id)
        data = json.loads(path.read_text(encoding="utf-8"))
        return SummaryCard.model_validate(data)

    def exists(self, handle_id: str) -> bool:
        return self._card_path(handle_id, fail=False).is_file()

    def _validate_handle_id(self, handle_id: str) -> None:
        if not is_result_handle_id(handle_id):
            raise _unknown(handle_id)

    def _parquet_path(self, handle_id: str, *, fail: bool = True) -> Path:
        self._validate_handle_id(handle_id)
        path = self._root / f"{handle_id}.parquet"
        if fail and not path.is_file():
            raise _unknown(handle_id)
        return path

    def _card_path(self, handle_id: str, *, fail: bool = True) -> Path:
        self._validate_handle_id(handle_id)
        path = self._root / f"{handle_id}.card.json"
        if fail and not path.is_file():
            raise _unknown(handle_id)
        return path


def _unknown(handle_id: str) -> UnknownNameError:
    return UnknownNameError(
        f"unknown result handle {handle_id}",
        hint="use a handle id returned by an earlier tool call",
    )


def is_result_handle_id(value: str) -> bool:
    return re.fullmatch(r"r_[0-9a-f]{10}", value) is not None


def _make_handle(source: str, produced_by: str, meta: dict) -> str:
    payload = source + produced_by + json.dumps(meta, sort_keys=True)
    return "r_" + hashlib.sha256(payload.encode()).hexdigest()[:10]


def _build_card(
    handle_id: str,
    table: pa.Table,
    *,
    produced_by: str,
    source: str,
    governed: bool,
    elapsed_s: float,
    notes: list[str],
    meta: dict,
) -> SummaryCard:
    columns = [(name, str(field.type)) for name, field in zip(table.schema.names, table.schema)]
    stats = [_column_stats(table, name) for name in table.schema.names]
    preview = _preview_markdown(table, _PREVIEW_ROWS)
    return SummaryCard(
        handle_id=handle_id,
        rows=table.num_rows,
        columns=columns,
        preview=preview,
        stats=stats,
        produced_by=produced_by,
        source=source,
        governed=governed,
        elapsed_s=elapsed_s,
        notes=notes,
        meta=meta,
    )


def _is_numeric_type(dtype: pa.DataType) -> bool:
    return (
        pa.types.is_integer(dtype)
        or pa.types.is_floating(dtype)
        or pa.types.is_decimal(dtype)
    )


def _is_orderable_type(dtype: pa.DataType) -> bool:
    if _is_numeric_type(dtype) or pa.types.is_boolean(dtype):
        return True
    if pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
        return True
    if pa.types.is_binary(dtype) or pa.types.is_large_binary(dtype):
        return True
    return pa.types.is_temporal(dtype)


def _is_numeric_type_str(type_str: str) -> bool:
    lowered = type_str.lower()
    return any(
        token in lowered
        for token in ("int", "float", "double", "decimal", "numeric")
    )


def _column_stats(table: pa.Table, name: str) -> ColumnStats:
    col = table.column(name)
    field = table.schema.field(name)
    nulls = int(col.null_count)
    dtype = field.type
    if not _is_orderable_type(dtype):
        return ColumnStats(
            name=name,
            type=str(dtype),
            nulls=nulls,
            min=None,
            max=None,
            mean=None,
        )
    valid = pc.drop_null(col)
    if valid.length() == 0:
        return ColumnStats(
            name=name,
            type=str(dtype),
            nulls=nulls,
            min=None,
            max=None,
            mean=None,
        )
    if _is_numeric_type(dtype):
        vmin = pc.min(valid).as_py()
        vmax = pc.max(valid).as_py()
        vmean = pc.mean(valid).as_py()
        return ColumnStats(
            name=name,
            type=str(dtype),
            nulls=nulls,
            min=float(vmin) if vmin is not None else None,
            max=float(vmax) if vmax is not None else None,
            mean=float(vmean) if vmean is not None else None,
        )
    vmin = pc.min(valid).as_py()
    vmax = pc.max(valid).as_py()
    return ColumnStats(
        name=name,
        type=str(dtype),
        nulls=nulls,
        min=str(vmin),
        max=str(vmax),
        mean=None,
    )


def _preview_markdown(table: pa.Table, max_rows: int) -> str:
    slice_table = table.slice(0, min(max_rows, table.num_rows))
    df = _format_preview_frame(slice_table.to_pandas())
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return _pipe_markdown(df)


def _meta_by_value_length(meta: dict) -> dict:
    """Shortest serialized values first so the render cap keeps them."""
    return {
        key: meta[key]
        for key in sorted(meta, key=lambda key: (len(json.dumps(meta[key])), key))
    }


def _format_preview_frame(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()
    for name in formatted.columns:
        series = formatted[name]
        if pd.api.types.is_float_dtype(series):
            formatted[name] = series.map(_format_display_float)
    return formatted


def _format_display_float(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        return str(value)
    number = float(value)
    if math.isnan(number):
        return ""
    if abs(number) >= 1:
        return f"{number:.2f}"
    return f"{number:.4g}"


def _format_stat_value(value: object, *, as_int: bool) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if as_int and value.is_integer():
            return str(int(value))
        return _format_display_float(value)
    return str(value)


def _is_integer_type_str(type_str: str) -> bool:
    head = type_str.lower().split("[", 1)[0]
    return head.startswith(("int", "uint"))


def _pipe_markdown(df: pd.DataFrame) -> str:
    headers = [str(c) for c in df.columns]
    rows = [[_cell(v) for v in row] for row in df.itertuples(index=False, name=None)]
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    fmt = lambda parts: "| " + " | ".join(p.ljust(widths[i]) for i, p in enumerate(parts)) + " |"

    lines = [fmt(headers), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    lines.extend(fmt(row) for row in rows)
    return "\n".join(lines)


def _cell(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)


def _card_to_json(card: SummaryCard) -> dict[str, Any]:
    data = card.model_dump()
    data["columns"] = [[name, typ] for name, typ in card.columns]
    return data


def _preview_line_count(preview: str) -> int:
    body_lines = preview.strip().splitlines()
    try:
        sep_idx = next(i for i, ln in enumerate(body_lines) if "|---" in ln)
    except StopIteration:
        return 0
    return sum(1 for ln in body_lines[sep_idx + 1 :] if ln.startswith("|"))


def _truncate_preview_lines(
    preview: str,
    header: str,
    stats: str,
    *,
    extra: int = 0,
    max_rows: int | None = None,
) -> str:
    body_lines = preview.strip().splitlines()
    try:
        sep_idx = next(i for i, ln in enumerate(body_lines) if "|---" in ln)
    except StopIteration:
        return preview
    header_part = body_lines[: sep_idx + 1]
    data_lines = [ln for ln in body_lines[sep_idx + 1 :] if ln.startswith("|")]
    if max_rows is None:
        budget = (
            _RENDER_BUDGET
            - len(header)
            - len(stats)
            - extra
            - len(_TRUNC_SUFFIX)
        )
        while data_lines and len("\n".join(header_part + data_lines)) > budget:
            data_lines.pop()
    else:
        data_lines = data_lines[:max_rows]
    return "\n".join(header_part + data_lines)
