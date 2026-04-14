"""Factory that returns (LLMProvider, EmbeddingProvider) pairs given
provider-name strings.

Used by the bench agents so callers can say
``--llm-provider openai --embedder-provider openai --llm-model gpt-4.1-mini``
without the agents hardcoding Gemini imports.

Lazy imports per-provider: a bench run that only touches OpenAI
doesn't need Anthropic's SDK on PYTHONPATH.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neuromem.providers import EmbeddingProvider, LLMProvider

# Default models per provider — pick something modern + stable.
_LLM_DEFAULTS: dict[str, str] = {
    "gemini": "gemini-2.0-flash-001",
    "openai": "gpt-4.1-mini",
    "anthropic": "claude-sonnet-4-6",
    "gemma": "gemma3",
}

_EMBEDDER_DEFAULTS: dict[str, str] = {
    "gemini": "gemini-embedding-001",
    "openai": "text-embedding-3-small",
    "gemma": "embeddinggemma",
}

# Provider names that support each role.
_LLM_PROVIDERS = frozenset({"gemini", "openai", "anthropic", "gemma"})
_EMBEDDER_PROVIDERS = frozenset({"gemini", "openai", "gemma"})


@dataclass
class ProviderPair:
    """Instantiated LLM + Embedder for a single bench agent.

    Only ``llm`` is guaranteed non-None; ``embedder`` can be None
    when the bench agent doesn't need one (e.g. NullAgent baselines).
    """

    llm: LLMProvider
    embedder: EmbeddingProvider | None


def build_llm(
    provider: str,
    api_key: str,
    model: str | None = None,
) -> LLMProvider:
    """Instantiate an LLMProvider by short name."""
    provider = provider.lower()
    model = model or _LLM_DEFAULTS[provider]

    if provider == "gemini":
        from neuromem_gemini import GeminiLLMProvider  # noqa: PLC0415

        return GeminiLLMProvider(api_key=api_key, model=model)
    if provider == "openai":
        from neuromem_openai import OpenAILLMProvider  # noqa: PLC0415

        return OpenAILLMProvider(api_key=api_key, model=model)
    if provider == "anthropic":
        from neuromem_anthropic import AnthropicLLMProvider  # noqa: PLC0415

        return AnthropicLLMProvider(api_key=api_key, model=model)
    if provider == "gemma":
        from neuromem_gemma import GemmaLLMProvider  # noqa: PLC0415

        # api_key unused by Ollama but the ctor requires non-empty;
        # GemmaLLMProvider defaults it to "ollama" if the caller
        # passes an empty string.
        return GemmaLLMProvider(model=model, api_key=api_key or "ollama")
    raise ValueError(f"Unknown LLM provider {provider!r}. Valid: {sorted(_LLM_PROVIDERS)}")


def build_embedder(
    provider: str,
    api_key: str,
    model: str | None = None,
) -> EmbeddingProvider:
    """Instantiate an EmbeddingProvider by short name.

    Anthropic doesn't ship an embedder; callers pairing Claude with
    a local embedder should use ``openai`` (with a fresh OPENAI_API_KEY)
    or ``gemma`` (local Ollama).
    """
    provider = provider.lower()
    model = model or _EMBEDDER_DEFAULTS[provider]

    if provider == "gemini":
        from neuromem_gemini import GeminiEmbeddingProvider  # noqa: PLC0415

        return GeminiEmbeddingProvider(api_key=api_key, model=model)
    if provider == "openai":
        from neuromem_openai import OpenAIEmbeddingProvider  # noqa: PLC0415

        return OpenAIEmbeddingProvider(api_key=api_key, model=model)
    if provider == "gemma":
        from neuromem_gemma import GemmaEmbeddingProvider  # noqa: PLC0415

        return GemmaEmbeddingProvider(model=model, api_key=api_key or "ollama")
    raise ValueError(
        f"Unknown embedder provider {provider!r}. Valid: {sorted(_EMBEDDER_PROVIDERS)} "
        f"(note: anthropic has no native embedder)"
    )


def build_pair(
    llm_provider: str,
    llm_api_key: str,
    llm_model: str | None = None,
    *,
    embedder_provider: str | None = None,
    embedder_api_key: str | None = None,
    embedder_model: str | None = None,
) -> ProviderPair:
    """Convenience: build both halves. Embedder defaults to the same
    provider as the LLM unless overridden; for anthropic LLM, the
    default embedder is openai (Anthropic has no native embedder)."""
    llm = build_llm(llm_provider, llm_api_key, llm_model)

    eff_embedder_provider = embedder_provider
    if eff_embedder_provider is None:
        # Anthropic LLM → OpenAI embedder default (uses the same API
        # key if one wasn't supplied separately).
        eff_embedder_provider = "openai" if llm_provider.lower() == "anthropic" else llm_provider

    eff_embedder_api_key = embedder_api_key if embedder_api_key is not None else llm_api_key
    embedder = build_embedder(eff_embedder_provider, eff_embedder_api_key, embedder_model)
    return ProviderPair(llm=llm, embedder=embedder)


__all__ = [
    "ProviderPair",
    "build_embedder",
    "build_llm",
    "build_pair",
]
