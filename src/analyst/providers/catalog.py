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
    "ModelInfo",
    "Prices",
    "catalog_path",
    "clear_catalog_cache",
    "load_catalog",
    "model_info",
]


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


def load_catalog(settings: Settings | None = None) -> tuple[ModelInfo, ...]:
    """Every catalogue entry, in file order."""
    return _load(catalog_path(settings))


def model_info(spec: str, *, settings: Settings | None = None) -> ModelInfo | None:
    """The catalogue entry for ``spec``, or ``None`` when it is not listed."""
    return next((e for e in load_catalog(settings) if e.spec == spec), None)


def clear_catalog_cache() -> None:
    _load.cache_clear()
