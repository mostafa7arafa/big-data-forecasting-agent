"""The model catalogue: labels, context windows, prices and the reasoning opt-in.

Read from ``<repo_root>/config/models.yaml``: a list of entries keyed by model
spec. Numbers there are copied from provider pages or ``null``; a ``null``
stays ``None`` here so a trace reports "unknown" instead of a made-up figure.
A missing file is an empty catalogue, and an unknown spec has no entry.

The file is read once per path and cached; the first read touches the disk,
so async callers go through ``asyncio.to_thread``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from analyst.config import Settings, get_settings

__all__ = [
    "MODEL_SPEC_KEY",
    "ModelInfo",
    "Prices",
    "cached_model_info",
    "catalog_path",
    "clear_catalog_cache",
    "load_catalog",
    "model_info",
    "model_spec_of",
]

#: ``BaseChatModel.metadata`` key holding the spec ``get_model`` built a model
#: from, so the live trace can price its calls with the catalogue entry.
MODEL_SPEC_KEY = "analyst_model_spec"


class Prices(BaseModel):
    """USD per 1M tokens; ``None`` when the provider page does not state it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input: float | None = Field(default=None, gt=0)
    cached_input: float | None = Field(default=None, gt=0)
    output: float | None = Field(default=None, gt=0)


class ModelInfo(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    spec: str
    label: str
    context_window: int | None = Field(default=None, gt=0)
    prices: Prices | None = None
    reasoning_summary: bool = False

    @property
    def has_prices(self) -> bool:
        """Whether input and output prices are both known."""
        return (
            self.prices is not None
            and self.prices.input is not None
            and self.prices.output is not None
        )


def catalog_path(settings: Settings | None = None) -> Path:
    settings = get_settings() if settings is None else settings
    return settings.repo_root / "config" / "models.yaml"


@lru_cache(maxsize=8)
def _load(path: Path) -> tuple[ModelInfo, ...]:
    if not path.is_file():
        return ()
    raw = yaml.safe_load(path.read_text()) or []
    if not isinstance(raw, list):
        raise TypeError(f"{path}: the model catalogue must be a list of entries")
    entries = tuple(ModelInfo.model_validate(item) for item in raw)
    specs = [e.spec for e in entries]
    if len(set(specs)) != len(specs):
        raise ValueError(f"{path}: duplicate model spec in the catalogue")
    return entries


# The catalogue most recently loaded, by spec: read without any I/O.
_warm: dict[str, ModelInfo] = {}


def load_catalog(settings: Settings | None = None) -> tuple[ModelInfo, ...]:
    """Every catalogue entry, in file order."""
    global _warm
    entries = _load(catalog_path(settings))
    _warm = {e.spec: e for e in entries}
    return entries


def model_info(spec: str, *, settings: Settings | None = None) -> ModelInfo | None:
    """The catalogue entry for ``spec``, or ``None`` when it is not listed."""
    return next((e for e in load_catalog(settings) if e.spec == spec), None)


def cached_model_info(spec: str) -> ModelInfo | None:
    """The entry for ``spec`` in the catalogue already loaded, never reading it.

    Safe on the event loop: it touches neither settings nor the disk. ``None``
    when no catalogue has been loaded yet or ``spec`` is not listed.
    ``get_model`` loads it (off the loop) before any model runs.
    """
    return _warm.get(spec)


def model_spec_of(model: object) -> str | None:
    """The spec ``get_model`` built ``model`` from, or ``None`` (e.g. a test double)."""
    meta = getattr(model, "metadata", None)
    spec = meta.get(MODEL_SPEC_KEY) if isinstance(meta, dict) else None
    return spec if isinstance(spec, str) and spec else None


def clear_catalog_cache() -> None:
    global _warm
    _load.cache_clear()
    _warm = {}
