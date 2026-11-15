from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

from analyst.config import Settings, get_settings


class Lookup(BaseModel):
    name: str
    file: Path


class GeometrySpec(BaseModel):
    """A GeoJSON file in the dataset folder that map charts can colour by key."""

    name: str = Field(pattern=r"^[a-z0-9_]{1,40}$")
    file: Path
    key_property: str
    dimension: str | None = None

    @field_validator("file")
    @classmethod
    def _file_is_relative(cls, value: Path) -> Path:
        if value.is_absolute() or ".." in value.parts:
            raise ValueError(f"geometry file {value} must be a relative path inside the dataset folder")
        return value


def contained_geometry_path(dataset_dir: Path, spec: GeometrySpec) -> Path:
    """The one containment check for geometry files (symlinks resolved)."""
    base = dataset_dir.resolve()
    path = (base / spec.file).resolve()
    if not path.is_relative_to(base):
        raise ValueError(f"geometry {spec.name!r} resolves outside the dataset folder")
    return path


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
    geometries: list[GeometrySpec] = Field(default_factory=list)

    def geometry(self, name: str) -> GeometrySpec | None:
        return next((g for g in self.geometries if g.name == name), None)

    def geometry_path(self, name: str) -> Path:
        """Resolved geometry file; must stay inside ``dir`` (symlinks included)."""
        spec = self.geometry(name)
        if spec is None:
            raise KeyError(name)
        return contained_geometry_path(self.dir, spec)

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
