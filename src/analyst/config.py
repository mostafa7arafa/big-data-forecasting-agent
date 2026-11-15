from functools import lru_cache
from pathlib import Path
from typing import Self

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ANALYST_",
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )

    repo_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2])
    datasets_dir: Path | None = None
    artifacts_dir: Path | None = None

    # Dataset and agent runtime.
    dataset: str | None = None
    model: str = "openai:gpt-6-luna"
    max_background_tasks: int = 3
    compaction_trigger_tokens: int = Field(default=60_000, gt=0)
    compaction_keep_messages: int = Field(default=8, ge=1)
    max_concurrent_queries: int = 3
    forecast_budget_s: float = 120.0
    inline_forecast_limit_s: float = 15.0
    nanogpt_base_url: str = "https://nano-gpt.com/api/v1"
    sandbox_image: str = "analyst-sandbox:1"
    sandbox_timeout_s: float = 30.0
    sandbox_max_live: int = Field(default=8, gt=0)
    sandbox_max_frame_bytes: int = Field(default=512 * 1024 * 1024, gt=0)
    sandbox_max_frame_rows: int = Field(default=5_000_000, gt=0)
    sandbox_max_frame_cells: int = Field(default=20_000_000, gt=0)

    # Provider keys keep their ecosystem-standard names, so the LangChain,
    # LangSmith and OpenAI SDKs read the same environment this does.
    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY", "ANALYST_OPENAI_API_KEY"),
    )
    openrouter_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "OPENROUTER_API_KEY", "ANALYST_OPENROUTER_API_KEY"
        ),
    )
    nanogpt_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("NANOGPT_API_KEY", "ANALYST_NANOGPT_API_KEY"),
    )
    langsmith_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "LANGSMITH_API_KEY", "ANALYST_LANGSMITH_API_KEY"
        ),
    )

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
