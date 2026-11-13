import os

from analyst.config import Settings

KEY_VARS = (
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "NANOGPT_API_KEY",
    "LANGSMITH_API_KEY",
)


def _clear_env(monkeypatch):
    for name in [key for key in os.environ if key.startswith("ANALYST_")] + list(
        KEY_VARS
    ):
        monkeypatch.delenv(name, raising=False)


def test_settings_defaults(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    s = Settings()
    assert s.datasets_dir == s.repo_root / "datasets"
    assert s.artifacts_dir == s.repo_root / "data" / "artifacts"


def test_settings_env_override(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANALYST_ARTIFACTS_DIR", str(tmp_path))
    assert Settings().artifacts_dir == tmp_path


def test_settings_empty_env_uses_default(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANALYST_ARTIFACTS_DIR", "")
    s = Settings()
    assert s.artifacts_dir == s.repo_root / "data" / "artifacts"


def test_settings_new_defaults(monkeypatch):
    _clear_env(monkeypatch)
    s = Settings(_env_file=None)
    assert s.dataset is None and s.model == "openai:gpt-6-luna"
    assert (s.max_background_tasks, s.max_concurrent_queries) == (3, 3)


def test_settings_new_numeric_defaults(monkeypatch):
    _clear_env(monkeypatch)
    s = Settings(_env_file=None)
    assert s.forecast_budget_s == 120.0
    assert s.inline_forecast_limit_s == 15.0
    assert s.nanogpt_base_url == "https://nano-gpt.com/api/v1"


def test_settings_reads_unprefixed_keys(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert Settings(_env_file=None).openai_api_key.get_secret_value() == "sk-test"


def test_settings_other_unprefixed_keys(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    monkeypatch.setenv("NANOGPT_API_KEY", "nano-test")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test")
    s = Settings(_env_file=None)
    assert s.openrouter_api_key.get_secret_value() == "or-test"
    assert s.nanogpt_api_key.get_secret_value() == "nano-test"
    assert s.langsmith_api_key.get_secret_value() == "ls-test"
    assert s.openai_api_key is None


def test_settings_reads_prefixed_overrides(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("ANALYST_MODEL", "openrouter:some-model")
    monkeypatch.setenv("ANALYST_DATASET", "some_dataset")
    monkeypatch.setenv("ANALYST_MAX_CONCURRENT_QUERIES", "7")
    s = Settings(_env_file=None)
    assert s.model == "openrouter:some-model"
    assert s.dataset == "some_dataset"
    assert s.max_concurrent_queries == 7
