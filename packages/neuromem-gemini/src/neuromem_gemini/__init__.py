"""neuromem-gemini — Google Gemini provider pair for ``neuromem-core``.

Implements the ``LLMProvider`` and ``EmbeddingProvider`` abstract base
classes from ``neuromem.providers`` against the ``google-genai`` SDK.

Example::

    import os
    from neuromem import NeuroMemory, SQLiteAdapter
    from neuromem_gemini import GeminiLLMProvider, GeminiEmbeddingProvider

    api_key = os.environ["GEMINI_API_KEY"]
    memory = NeuroMemory(
        storage=SQLiteAdapter("memory.db"),
        llm=GeminiLLMProvider(api_key=api_key),
        embedder=GeminiEmbeddingProvider(api_key=api_key),
    )

See the package README for full usage, default model names, and
integration test invocation.
"""

from __future__ import annotations

from neuromem_gemini.embedder import GeminiEmbeddingProvider
from neuromem_gemini.llm import GeminiLLMProvider

__version__ = "0.1.0"

__all__ = [
    "GeminiEmbeddingProvider",
    "GeminiLLMProvider",
    "__version__",
]
