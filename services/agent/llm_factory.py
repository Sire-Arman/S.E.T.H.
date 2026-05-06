"""LLM provider factory for the LangGraph agent.

Returns LangChain BaseChatModel instances that support `bind_tools()`.
Each provider is lazily imported — only the selected one is loaded.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from config import Settings


def create_llm(provider: str, settings: "Settings") -> "BaseChatModel":
    """Create and return a LangChain chat model for *provider*.

    Supported providers:
        - ``cohere``  — uses ``langchain-cohere`` (``ChatCohere``)
        - ``ollama``  — uses ``langchain-ollama`` (``ChatOllama``)

    The returned model supports ``.bind_tools()`` for LangGraph tool calling.

    Raises:
        ValueError: If *provider* is unknown or missing required config.
    """
    provider = provider.lower().strip()

    if provider == "cohere":
        return _create_cohere(settings)
    elif provider == "ollama":
        return _create_ollama(settings)
    else:
        raise ValueError(
            f"Unknown agent LLM provider '{provider}'. "
            f"Supported: cohere, ollama"
        )


# ── Provider constructors ─────────────────────────────────────────


def _create_cohere(settings: "Settings") -> "BaseChatModel":
    if not settings.COHERE_API_KEY:
        raise ValueError(
            "AGENT_LLM is set to 'cohere' but COHERE_API_KEY is not configured."
        )
    from langchain_cohere import ChatCohere

    llm = ChatCohere(
        cohere_api_key=settings.COHERE_API_KEY,
        model=settings.COHERE_MODEL,
        temperature=settings.LLM_TEMPERATURE,
    )
    logger.info(f"Agent LLM: Cohere ({settings.COHERE_MODEL})")
    return llm


def _create_ollama(settings: "Settings") -> "BaseChatModel":
    from langchain_ollama import ChatOllama

    # Strip /v1 suffix if present — ChatOllama uses native Ollama API, not OpenAI compat
    base_url = settings.OLLAMA_BASE_URL.rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]

    llm = ChatOllama(
        base_url=base_url,
        model=settings.OLLAMA_AGENT_MODEL,
        temperature=settings.LLM_TEMPERATURE,
    )
    logger.info(f"Agent LLM: Ollama ({settings.OLLAMA_AGENT_MODEL} @ {base_url})")
    return llm
