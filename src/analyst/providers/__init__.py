"""One factory for every chat model the analyst can run on.

A model spec is ``"<provider>:<model id>"`` with provider in ``openai``,
``openrouter`` or ``nanogpt``. The spec defaults to ``ANALYST_MODEL``; keys
come from the environment through :class:`analyst.config.Settings`. Building a
model never touches the network — only the first call does.
"""

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_openrouter import ChatOpenRouter

from analyst.config import Settings, get_settings
from analyst.errors import InvalidRequestError

__all__ = ["PROVIDERS", "get_model"]

#: provider name -> the environment variable that carries its key.
_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "nanogpt": "NANOGPT_API_KEY",
}

PROVIDERS = tuple(_KEY_ENV)


def _parse(spec: str) -> tuple[str, str]:
    provider, _, model_id = spec.partition(":")
    provider = provider.strip().lower()
    model_id = model_id.strip()
    if provider not in _KEY_ENV:
        raise InvalidRequestError(
            f"unknown model provider {provider!r} in spec {spec!r}",
            hint=(
                "use '<provider>:<model id>' with provider one of "
                + ", ".join(PROVIDERS)
            ),
        )
    if not model_id:
        raise InvalidRequestError(
            f"missing model id in spec {spec!r}",
            hint="use '<provider>:<model id>', e.g. 'openai:gpt-6-luna'",
        )
    return provider, model_id


def _api_key(settings: Settings, provider: str) -> str:
    secret = getattr(settings, f"{provider}_api_key")
    if secret is None or not secret.get_secret_value():
        env_var = _KEY_ENV[provider]
        raise InvalidRequestError(
            f"no API key configured for provider {provider!r}",
            hint=f"set {env_var} in the environment (or .env) and retry",
        )
    return secret.get_secret_value()


def get_model(
    spec: str | None = None, *, settings: Settings | None = None
) -> BaseChatModel:
    """Build the chat model named by ``spec`` (default: ``settings.model``)."""
    settings = get_settings() if settings is None else settings
    provider, model_id = _parse(settings.model if spec is None else spec)
    api_key = _api_key(settings, provider)

    if provider == "openrouter":
        return ChatOpenRouter(model=model_id, api_key=api_key)
    if provider == "nanogpt":
        return ChatOpenAI(
            model=model_id, base_url=settings.nanogpt_base_url, api_key=api_key
        )
    return ChatOpenAI(model=model_id, api_key=api_key)
