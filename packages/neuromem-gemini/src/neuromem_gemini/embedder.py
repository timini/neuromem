"""Google Gemini implementation of ``neuromem.providers.EmbeddingProvider``.

Wraps the ``google-genai`` SDK's ``embed_content`` endpoint. The
default model is ``gemini-embedding-001``, Google's current stable
embedding model — it returns a 3072-dimensional vector per input
text. Pass a different model name to the constructor if you want
to use the preview line (``gemini-embedding-2-preview``) or an
older endpoint once they're available.

The ``neuromem`` core library locks the embedding dimension on the
first ``upsert_node`` call and refuses to ingest vectors of a
different shape afterwards (data-model.md I-N3). That means you
cannot switch embedding models mid-run without rebuilding the whole
concept graph — one ``GeminiEmbeddingProvider`` instance per
``NeuroMemory`` lifetime.
"""

from __future__ import annotations

import numpy as np
from google import genai
from neuromem.providers import EmbeddingProvider
from numpy.typing import NDArray


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Google Gemini-backed ``EmbeddingProvider``.

    Takes a list of strings, hands them to ``genai.Client.models.
    embed_content`` in a single batched call, and returns a
    ``(len(texts), D)`` float32 numpy array preserving input order.

    The ``google-genai`` SDK uses the ``models.embed_content`` entry
    point (unified with the chat-completions path at the API level).
    Response shape is ``resp.embeddings[i].values`` — a list[float]
    per input text. We stack them into a contiguous float32 matrix
    because downstream ``neuromem.vectors`` functions (cosine
    similarity, centroid, agglomerative clustering) prefer numeric
    numpy input.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-001",
    ) -> None:
        if not api_key:
            raise ValueError("api_key must be non-empty")
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def get_embeddings(self, texts: list[str]) -> NDArray[np.floating]:
        """Return one embedding vector per input text, as a 2-D numpy array."""
        if not texts:
            # Match the conventional "empty in, empty out" behaviour.
            # Use dim=0 so the caller sees shape (0, 0) — the dreaming
            # pipeline never calls us on an empty list, but being
            # defensive is cheap.
            return np.empty((0, 0), dtype=np.float32)

        resp = self._client.models.embed_content(
            model=self._model,
            contents=texts,
        )
        # ``resp.embeddings`` is a list of ContentEmbedding objects,
        # each with a ``.values`` attribute holding the raw float list.
        # Stack into (N, D) float32 preserving the input order.
        rows = [np.asarray(e.values, dtype=np.float32) for e in resp.embeddings]
        return np.stack(rows, axis=0)
