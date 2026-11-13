import os
import socket

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_openrouter import ChatOpenRouter

from analyst.config import Settings
from analyst.errors import InvalidRequestError
from analyst.providers import get_model

KEY_VARS = ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "NANOGPT_API_KEY")


def _clean_env(monkeypatch):
    """Clear every provider key and ANALYST_* override: never read the dev .env."""
    names = [
        name
        for name in os.environ
        if name.startswith(("ANALYST_", "LANGSMITH_"))
    ]
    for name in [*names, *KEY_VARS]:
        monkeypatch.delenv(name, raising=False)


def _settings(monkeypatch, **env):
    _clean_env(monkeypatch)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return Settings(_env_file=None)


def test_openai_default(monkeypatch):
    settings = _settings(monkeypatch, OPENAI_API_KEY="sk-test")
    model = get_model(settings=settings)
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "gpt-6-luna"


def test_openai_uses_settings_default_spec(monkeypatch):
    settings = _settings(monkeypatch, OPENAI_API_KEY="sk-test")
    settings.model = "openai:gpt-6-sol"
    assert get_model(settings=settings).model_name == "gpt-6-sol"


def test_spec_overrides_settings_model(monkeypatch):
    settings = _settings(monkeypatch, OPENAI_API_KEY="sk-test")
    assert get_model("openai:gpt-6-terra", settings=settings).model_name == (
        "gpt-6-terra"
    )


def test_openrouter_spec(monkeypatch):
    settings = _settings(monkeypatch, OPENROUTER_API_KEY="or-test")
    model = get_model("openrouter:deepseek/deepseek-v4.1-flash-1017", settings=settings)
    assert isinstance(model, ChatOpenRouter)
    assert model.model_name == "deepseek/deepseek-v4.1-flash-1017"


def test_nanogpt_uses_base_url(monkeypatch):
    settings = _settings(monkeypatch, NANOGPT_API_KEY="nano-test")
    model = get_model("nanogpt:some-model", settings=settings)
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "some-model"
    assert model.openai_api_base == settings.nanogpt_base_url
    assert model.openai_api_key.get_secret_value() == "nano-test"


def test_missing_key_hint(monkeypatch):
    settings = _settings(monkeypatch)
    with pytest.raises(InvalidRequestError) as e:
        get_model("openrouter:x", settings=settings)
    assert "OPENROUTER_API_KEY" in e.value.hint


@pytest.mark.parametrize(
    ("spec", "env_var"),
    [("openai:x", "OPENAI_API_KEY"), ("nanogpt:x", "NANOGPT_API_KEY")],
)
def test_missing_key_hint_per_provider(monkeypatch, spec, env_var):
    settings = _settings(monkeypatch)
    with pytest.raises(InvalidRequestError) as e:
        get_model(spec, settings=settings)
    assert env_var in e.value.hint


def test_default_spec_missing_key_hints_openai(monkeypatch):
    settings = _settings(monkeypatch)
    with pytest.raises(InvalidRequestError) as e:
        get_model(settings=settings)
    assert "OPENAI_API_KEY" in e.value.hint


def test_unknown_provider_names_the_choices(monkeypatch):
    settings = _settings(monkeypatch, OPENAI_API_KEY="sk-test")
    with pytest.raises(InvalidRequestError) as e:
        get_model("anthropic:claude-sonnet-5", settings=settings)
    assert "anthropic" in str(e.value)
    for provider in ("openai", "openrouter", "nanogpt"):
        assert provider in e.value.hint


@pytest.mark.parametrize("spec", ["openai", "openai:", ":gpt-6-luna", ""])
def test_malformed_spec_is_invalid_request(monkeypatch, spec):
    settings = _settings(monkeypatch, OPENAI_API_KEY="sk-test")
    with pytest.raises(InvalidRequestError):
        get_model(spec, settings=settings)


def test_construction_makes_no_network_call(monkeypatch):
    settings = _settings(
        monkeypatch,
        OPENAI_API_KEY="sk-test",
        OPENROUTER_API_KEY="or-test",
        NANOGPT_API_KEY="nano-test",
    )

    def _boom(*args, **kwargs):
        raise AssertionError("get_model opened a socket")

    monkeypatch.setattr(socket.socket, "connect", _boom)
    monkeypatch.setattr(socket, "create_connection", _boom)

    models = [
        get_model("openai:gpt-6-luna", settings=settings),
        get_model("openrouter:some/model", settings=settings),
        get_model("nanogpt:some-model", settings=settings),
    ]
    assert all(isinstance(model, BaseChatModel) for model in models)
