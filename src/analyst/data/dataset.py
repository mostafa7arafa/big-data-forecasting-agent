from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel

from analyst.config import Settings, get_settings


class Lookup(BaseModel):
    name: str
    file: Path


class DatasetConfig(BaseModel):
    name: str
    description: str
    files: str
    table: str
    time_column: str
    valid_time_range: tuple[date, date]
    lookups: list[Lookup]
    root: Path
    dir: Path

    @property
    def data_dirs(self) -> list[Path]:
        parents: list[Path] = []
        paths = list(self.root.glob(self.files))
        paths.extend(self.root / lookup.file for lookup in self.lookups)
        for path in paths:
            parent = path.parent
            if parent not in parents:
                parents.append(parent)
        return parents

    @property
    def memory_dir(self) -> Path:
        return self.dir / "memory"


def load_dataset(name: str, settings: Settings | None = None) -> DatasetConfig:
    resolved = settings if settings is not None else get_settings()
    datasets_dir = resolved.datasets_dir or (resolved.repo_root / "datasets")
    path = datasets_dir / name / "dataset.yaml"
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping")  # noqa: TRY004
    raw["root"] = resolved.repo_root
    raw["dir"] = datasets_dir / name
    return DatasetConfig.model_validate(raw)
