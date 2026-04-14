"""OpenAI implementation of ``neuromem.providers.EmbeddingProvider``.

Wraps ``openai.embeddings.create``. Default model is
``text-embedding-3-small`` (1536-dim); pass ``model=
"text-embedding-3-large"`` for the 3072-dim variant.

Dimension lock: ``neuromem-core`` locks the embedding dimension on
the first ``upsert_node`` call (data-model.md I-N3) — one embedder
instance per ``NeuroMemory`` lifetime. Don't swap models mid-run.
"""

from __future__ import annotations

import numpy as np
from neuromem.providers import EmbeddingProvider
from numpy.typing import NDArray
from openai import OpenAI


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI-backed embedding provider.

    Batches the full input list into a single HTTP call (OpenAI's
    embeddings endpoint accepts up to 2048 inputs per call for
    text-embedding-3 models). For very large batches the SDK paginates
    internally; we pass the whole list and let the SDK decide.

    Usage::

        from neuromem_openai import OpenAIEmbeddingProvider

        emb = OpenAIEmbeddingProvider(
            api_key=os.environ["OPENAI_API_KEY"],
            model="text-embedding-3-small",
        )
        vectors = emb.get_embeddings(["hello", "world"])
        # vectors.shape == (2, 1536)

    ``base_url`` is exposed so this provider can drive any
    OpenAI-compatible endpoint — Ollama, vLLM, Azure, OpenRouter, LM
    Studio, etc. — by pointing at e.g. ``http://localhost:11434/v1``.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        *,
        base_url: str | None = None,
        request_timeout_s: float = 60.0,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must be non-empty")
        if not model:
            raise ValueError("model must be non-empty")
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=request_timeout_s)
        self._model = model

    def get_embeddings(self, texts: list[str]) -> NDArray[np.floating]:
        if not texts:
            # Match GeminiEmbeddingProvider's behaviour: return a
            # (0, D) array when the caller passes an empty list. We
            # don't know D without at least one call, so fall back to
            # (0, 1).
            return np.zeros((0, 1), dtype=np.float32)
        resp = self._client.embeddings.create(model=self._model, input=texts)
        # resp.data is a list of Embedding objects in input order.
        vectors = [np.asarray(item.embedding, dtype=np.float32) for item in resp.data]
        return np.stack(vectors)
