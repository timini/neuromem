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

import time

import numpy as np
from google import genai
from google.genai.errors import ServerError
from neuromem.providers import EmbeddingProvider
from numpy.typing import NDArray

# Same retryable-exceptions tuple as the LLM provider — transport-
# level connection drops + Gemini 5xx.
_RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (ServerError,)
try:
    import httpx  # noqa: PLC0415

    _RETRYABLE_EXCEPTIONS = (
        ServerError,
        httpx.RemoteProtocolError,
        httpx.ConnectError,
        httpx.ReadTimeout,
        httpx.WriteTimeout,
        httpx.PoolTimeout,
    )
except ImportError:
    pass


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

    # Gemini's ``embed_content`` endpoint caps at 100 texts per batch.
    # Callers that pass more than 100 texts in a single call get a
    # ``400 INVALID_ARGUMENT: BatchEmbedContentsRequest.requests: at
    # most 100 requests can be in one batch``. Rather than push that
    # chunking requirement onto every caller, we handle it transparently
    # inside this provider by sub-batching internally.
    _MAX_BATCH = 100

    def get_embeddings(self, texts: list[str]) -> NDArray[np.floating]:
        """Return one embedding vector per input text, as a 2-D numpy array.

        Transparently chunks large inputs into ``_MAX_BATCH``-sized
        sub-batches to work around Gemini's 100-items-per-request
        limit. Row order in the returned array always matches input
        order across chunk boundaries.
        """
        if not texts:
            # Match the conventional "empty in, empty out" behaviour.
            # Use dim=0 so the caller sees shape (0, 0) — the dreaming
            # pipeline never calls us on an empty list, but being
            # defensive is cheap.
            return np.empty((0, 0), dtype=np.float32)

        all_rows: list[np.ndarray] = []
        for start in range(0, len(texts), self._MAX_BATCH):
            chunk = texts[start : start + self._MAX_BATCH]
            resp = self._embed_chunk_with_retry(chunk)
            # ``resp.embeddings`` is a list of ContentEmbedding objects,
            # each with a ``.values`` attribute holding the raw float list.
            all_rows.extend(np.asarray(e.values, dtype=np.float32) for e in resp.embeddings)

        # Stack into (N, D) float32 preserving input order.
        return np.stack(all_rows, axis=0)

    def _embed_chunk_with_retry(self, chunk: list[str]) -> object:
        """Call ``embed_content`` on one ≤100-item chunk with retry on
        transient 5xx server errors.

        Gemini's embed_content endpoint occasionally returns
        ``503 UNAVAILABLE`` under load. The underlying
        ``google-genai`` SDK has its own tenacity-based retry loop,
        but under sustained load it sometimes exhausts the default
        retries. We add a second layer here with exponential backoff
        and a higher ceiling so benchmark workloads that hit the
        endpoint hundreds of times don't fail on a single transient
        hiccup.

        Retries on ``ServerError`` (5xx) only. ``ClientError`` (4xx)
        is not retried — a 400/403/404 won't fix itself on retry.
        """
        max_attempts = 5
        base_delay = 2.0  # seconds
        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            try:
                return self._client.models.embed_content(
                    model=self._model,
                    contents=chunk,
                )
            except _RETRYABLE_EXCEPTIONS as exc:
                last_exc = exc
                if attempt == max_attempts - 1:
                    break
                delay = base_delay * (2**attempt)
                time.sleep(delay)
        # Exhausted retries — re-raise the last ServerError so the
        # caller sees a real Gemini error, not a generic wrapped one.
        assert last_exc is not None
        raise last_exc
