"""LLM provider factory supporting Ollama (default, free, local) plus optional
cloud providers (Groq, OpenAI, Anthropic) selectable via config/env."""

import logging
import sys
from pathlib import Path

from langchain_core.language_models import BaseChatModel

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = {"ollama", "groq", "openai", "anthropic"}


class LLMProviderError(Exception):
    """Raised when a provider can't be constructed (missing key, unreachable, etc.)."""


def get_llm(provider: str | None = None) -> BaseChatModel:
    """Return a LangChain chat model for the requested provider.

    Defaults to Ollama, which requires no API key and runs fully locally.
    Raises LLMProviderError with a human-readable message if the provider
    is misconfigured (e.g. missing API key) — callers should catch this
    and surface it as a clean error rather than a stack trace.
    """
    provider = (provider or settings.llm_provider).lower()

    if provider not in SUPPORTED_PROVIDERS:
        raise LLMProviderError(
            f"Unknown provider '{provider}'. Supported: {sorted(SUPPORTED_PROVIDERS)}"
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0.1,
        )

    if provider == "groq":
        if not settings.groq_api_key:
            raise LLMProviderError("GROQ_API_KEY is not set. Add it to .env or choose provider=ollama.")
        from langchain_groq import ChatGroq

        return ChatGroq(model=settings.groq_model, api_key=settings.groq_api_key, temperature=0.1)

    if provider == "openai":
        if not settings.openai_api_key:
            raise LLMProviderError("OPENAI_API_KEY is not set. Add it to .env or choose provider=ollama.")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=settings.openai_model, api_key=settings.openai_api_key, temperature=0.1)

    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise LLMProviderError("ANTHROPIC_API_KEY is not set. Add it to .env or choose provider=ollama.")
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=settings.anthropic_model, api_key=settings.anthropic_api_key, temperature=0.1)

    raise LLMProviderError(f"Provider '{provider}' is not implemented.")


def check_ollama_available() -> bool:
    """Ping the local Ollama server to check it's running before we try to use it."""
    import requests

    try:
        response = requests.get(f"{settings.ollama_base_url}/api/tags", timeout=3)
        return response.status_code == 200
    except requests.RequestException:
        return False


if __name__ == "__main__":
    if not check_ollama_available():
        logger.error("Ollama does not appear to be running at %s", settings.ollama_base_url)
    else:
        llm = get_llm("ollama")
        response = llm.invoke("Say 'MedAssist RAG is working' and nothing else.")
        logger.info("Response: %s", response.content)
