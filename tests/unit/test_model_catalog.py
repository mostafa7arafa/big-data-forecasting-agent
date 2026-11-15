"""The model catalogue (config/models.yaml), the reasoning-summary opt-in, /meta models."""

import os

import pytest
from fastapi.testclient import TestClient
from langchain_openai import ChatOpenAI

from analyst.config import Settings, get_settings
from analyst.providers import get_model
from analyst.providers.catalog import clear_catalog_cache, load_catalog, model_info

KEY_VARS = ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "NANOGPT_API_KEY")


@pytest.fixture(autouse=True)
def _fresh():
    clear_catalog_cache()
    get_settings.cache_clear()
    yield
    clear_catalog_cache()
    get_settings.cache_clear()


def _settings(monkeypatch, **env) -> Settings:
    for name in [n for n in os.environ if n.startswith(("ANALYST_", "LANGSMITH_"))]:
        monkeypatch.delenv(name, raising=False)
    for name in KEY_VARS:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return Settings(_env_file=None)


def _write_catalog(root, text: str) -> None:
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "config" / "models.yaml").write_text(text)


def test_catalog_loads_and_unknown_is_none(monkeypatch):
    settings = _settings(monkeypatch)
    info = model_info("openai:gpt-6-luna", settings=settings)
    assert info is not None
    assert info.spec == "openai:gpt-6-luna"
    assert info.label
    assert info.reasoning_summary is True
    # Every shipped entry: numbers are either copied or null, never 0.
    for entry in load_catalog(settings):
        assert entry.context_window is None or entry.context_window > 0
        if entry.prices is not None:
            for value in (entry.prices.input, entry.prices.cached_input, entry.prices.output):
                assert value is None or value > 0
    assert model_info("openai:no-such-model", settings=settings) is None
    assert model_info("nope", settings=settings) is None


def test_catalog_missing_file_is_empty(monkeypatch, tmp_path):
    settings = _settings(monkeypatch, ANALYST_REPO_ROOT=str(tmp_path))
    assert load_catalog(settings) == ()
    assert model_info("openai:gpt-6-luna", settings=settings) is None


def test_openai_reasoning_summary_opt_in(monkeypatch, tmp_path):
    _write_catalog(
        tmp_path,
        """\
- {spec: "openai:thinker", label: Thinker, context_window: null, prices: null, reasoning_summary: true}
- {spec: "openai:plain", label: Plain, context_window: 1000, prices: null, reasoning_summary: false}
- {spec: "nanogpt:thinker", label: Nano, context_window: null, prices: null, reasoning_summary: true}
""",
    )
    settings = _settings(
        monkeypatch,
        ANALYST_REPO_ROOT=str(tmp_path),
        OPENAI_API_KEY="sk-test",
        NANOGPT_API_KEY="nano-test",
    )
    thinker = get_model("openai:thinker", settings=settings)
    assert isinstance(thinker, ChatOpenAI)
    assert thinker.use_responses_api is True
    assert thinker.reasoning == {"summary": "auto"}

    for spec in ("openai:plain", "openai:unlisted", "nanogpt:thinker"):
        model = get_model(spec, settings=settings)
        assert isinstance(model, ChatOpenAI)
        assert not model.use_responses_api, spec
        assert model.reasoning is None, spec


def test_meta_lists_models(monkeypatch, tmp_path):
    _write_catalog(
        tmp_path,
        """\
- spec: "openai:a"
  label: A
  context_window: 2000
  prices: {input: 1.0, cached_input: null, output: 2.0}
  reasoning_summary: true
- {spec: "openrouter:b", label: B, context_window: null, prices: null, reasoning_summary: false}
""",
    )
    (tmp_path / "datasets").mkdir()
    _settings(monkeypatch, ANALYST_REPO_ROOT=str(tmp_path), ANALYST_MODEL="openai:a")
    from analyst.http.app import app

    body = TestClient(app).get("/meta").json()
    assert body["default_model"] == "openai:a"
    assert body["models"] == [
        {
            "spec": "openai:a",
            "label": "A",
            "context_window": 2000,
            "has_prices": True,
            "reasoning_summary": True,
        },
        {
            "spec": "openrouter:b",
            "label": "B",
            "context_window": None,
            "has_prices": False,
            "reasoning_summary": False,
        },
    ]
