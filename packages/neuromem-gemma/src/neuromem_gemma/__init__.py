"""neuromem-gemma — local Gemma provider pair served via Ollama.

Thin convenience layer over neuromem-openai: Ollama exposes an
OpenAI-compatible endpoint at ``http://localhost:11434/v1``, so the
OpenAI SDK drives it fine. This package sets that base_url as the
default plus picks Gemma-friendly model names.

Example::

    from neuromem import NeuroMemory, SQLiteAdapter
    from neuromem_gemma import GemmaLLMProvider, GemmaEmbeddingProvider

    memory = NeuroMemory(
        storage=SQLiteAdapter("memory.db"),
        llm=GemmaLLMProvider(),          # gemma3 via localhost Ollama
        embedder=GemmaEmbeddingProvider(),  # embeddinggemma
    )

Prerequisite::

    ollama pull gemma3
    ollama pull embeddinggemma
    # leave `ollama serve` running

Requires no API key. The constructors set a dummy ``OPENAI_API_KEY``
because the openai SDK requires one, but Ollama ignores it.
"""

from __future__ import annotations

from neuromem_openai import OpenAIEmbeddingProvider, OpenAILLMProvider

from neuromem_gemma.prompts import load_prompt  # noqa: F401 — exported for callers

__version__ = "0.1.0"

_DEFAULT_BASE_URL = "http://localhost:11434/v1"
_DEFAULT_LLM_MODEL = "gemma3"
_DEFAULT_EMBEDDING_MODEL = "embeddinggemma"


class GemmaLLMProvider(OpenAILLMProvider):
    """Gemma LLM served via Ollama's OpenAI-compatible endpoint.

    Defaults are the shipped-with-Ollama Gemma 3 tag and the local
    Ollama endpoint. Pass ``base_url`` / ``model`` to override
    (e.g. to hit a remote GPU box or use a different Gemma size like
    ``gemma3:27b``).

    Pass ``api_key`` if you've put Ollama behind an auth proxy; by
    default we pass the literal string ``"ollama"`` which Ollama
    ignores but the openai SDK requires non-empty.
    """

    # Local Ollama serves a single model instance with a small
    # (default 1–4) parallelism budget; flooding it with the parent
    # class's 10-way threadpool causes queued requests to time out.
    # Cap at 2 concurrent in-flight calls — matches Ollama's default
    # OLLAMA_NUM_PARALLEL without starving it.
    _BATCH_WORKERS = 2

    def __init__(
        self,
        model: str = _DEFAULT_LLM_MODEL,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        api_key: str = "ollama",
        request_timeout_s: float = 600.0,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
            base_url=base_url,
            request_timeout_s=request_timeout_s,
        )


class GemmaEmbeddingProvider(OpenAIEmbeddingProvider):
    """EmbeddingGemma served via Ollama. Default model is
    ``embeddinggemma`` (Google's 308M open-weight embedding model,
    768-dim).

    Swap ``model`` to ``nomic-embed-text`` / ``mxbai-embed-large`` /
    etc. if you've already pulled an alternative local embedder.
    """

    def __init__(
        self,
        model: str = _DEFAULT_EMBEDDING_MODEL,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        api_key: str = "ollama",
        request_timeout_s: float = 600.0,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
            base_url=base_url,
            request_timeout_s=request_timeout_s,
        )


__all__ = [
    "GemmaEmbeddingProvider",
    "GemmaLLMProvider",
    "__version__",
]
