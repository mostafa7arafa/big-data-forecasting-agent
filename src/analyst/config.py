from functools import lru_cache
from pathlib import Path
from typing import Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ANALYST_", extra="ignore")

    repo_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2])
    datasets_dir: Path | None = None
    artifacts_dir: Path | None = None

    @model_validator(mode="after")
    def _default_path_dirs(self) -> Self:
        if self.datasets_dir is None:
            self.datasets_dir = self.repo_root / "datasets"
        if self.artifacts_dir is None:
            self.artifacts_dir = self.repo_root / "data" / "artifacts"
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
