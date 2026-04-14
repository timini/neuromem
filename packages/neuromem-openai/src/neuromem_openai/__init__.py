"""neuromem-openai — OpenAI provider pair for ``neuromem-core``.

Implements the ``LLMProvider`` and ``EmbeddingProvider`` abstract base
classes from ``neuromem.providers`` against the ``openai`` Python SDK.

Also usable against any OpenAI-compatible endpoint (Ollama, vLLM, LM
Studio, OpenRouter, Azure) by passing ``base_url`` to the constructor.
"""

from __future__ import annotations

from neuromem_openai.embedder import OpenAIEmbeddingProvider
from neuromem_openai.llm import OpenAILLMProvider

__version__ = "0.1.0"

__all__ = [
    "OpenAIEmbeddingProvider",
    "OpenAILLMProvider",
    "__version__",
]
