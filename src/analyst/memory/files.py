"""Load a dataset's hand-written memory files."""

from pathlib import Path

import yaml

from analyst.data.dataset import DatasetConfig
from analyst.semantic.model import ContextFact, Memory, Quirk, SemanticModel, SeriesSpec


def load_memory(dataset: DatasetConfig) -> Memory:
    directory = dataset.memory_dir
    semantic = SemanticModel.model_validate(_mapping(directory / "semantic.yaml"))
    quirks = _mapping(directory / "quirks.yaml")["quirks"]
    series = _mapping(directory / "series.yaml")["series"]
    facts = _mapping(directory / "context_facts.yaml")["context_facts"]
    recap = (directory / "recap.md").read_text()
    return Memory(
        semantic=semantic,
        quirks=[Quirk.model_validate(item) for item in quirks],
        series=[SeriesSpec.model_validate(item) for item in series],
        context_facts=[ContextFact.model_validate(item) for item in facts],
        recap=recap,
    )


def _mapping(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping")  # noqa: TRY004
    return raw
