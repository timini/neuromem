"""Internal shared Gemini answering client.

Private to ``neuromem_bench`` — the leading underscore on the
module name signals it's an implementation detail, not a stable
public API. Used by both ``agent.py`` (for each agent's answer-
generation path) and ``metrics.py`` (for the ``llm_judge``
metric's verdict call).

Lives in its own module so the two call sites don't have to
cross-import private symbols from each other (``metrics.py``
used to import ``_GeminiAnsweringClient`` from ``agent.py``,
which made ``metrics`` implicitly coupled to ``agent`` through
a private symbol — bad shape).

Not in ``neuromem-gemini`` because that package's public API is
the provider pair (``GeminiLLMProvider`` / ``GeminiEmbeddingProvider``)
— a bare "call Gemini once with a prompt" helper is a bench-
internal convenience, not a library primitive.
"""

from __future__ import annotations

import time as _time

from google import genai
from google.genai import types as genai_types

# Reuse the same retryable-exception set as the memory-layer provider
# (neuromem_gemini.llm._RETRYABLE_EXCEPTIONS) so the answer/judge path
# recovers from the same transient failures (5xx, httpx TimeoutException,
# RemoteProtocolError, ConnectError). Prior to this the answer client
# had no retries at all, and a single httpx.ReadTimeout killed an
# overnight run at instance 17/200.
from neuromem_gemini.llm import _RETRYABLE_EXCEPTIONS  # noqa: PLC2701

# Exponential backoff schedule for retries: 2s, 4s, 8s, 16s, 32s (sum 62s).
# Matches the memory-layer provider so ingestion + answer behave
# consistently under API flakiness.
_RETRY_MAX_ATTEMPTS = 5
_RETRY_BASE_DELAY_S = 2.0


class GeminiAnsweringClient:
    """Thin wrapper around ``google.genai.Client`` for one-shot
    answer generation.

    Takes a ``(system_instruction, user_message)`` pair and
    returns the model's text response. Used by all three agent
    backends for the final answer call, and by the ``llm_judge``
    metric for the verdict call.

    This is intentionally a one-shot convenience — the agents
    manage their own conversation state before the answer call;
    this client just does the final LLM turn.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash-001",
        *,
        request_timeout_ms: int = 60_000,
        rate_per_minute: int = 60,
    ) -> None:
        """``request_timeout_ms`` bounds every call; prevents a hung
        response from blocking the benchmark indefinitely. Same
        rationale as the provider pair in ``neuromem-gemini``.

        ``rate_per_minute`` — participates in the same token-bucket
        budget as any ``GeminiLLMProvider`` / ``GeminiEmbeddingProvider``
        constructed with the same ``api_key``. The answer LLM (this
        client) is typically called once per benchmark instance so
        it doesn't add much pressure, but sharing the budget means
        the llm_judge metric's verdict call also gets throttled
        when a whole run is in progress.
        """
        from neuromem_gemini._rate_limit import get_bucket  # noqa: PLC0415

        self._client = genai.Client(
            api_key=api_key,
            http_options=genai_types.HttpOptions(timeout=request_timeout_ms),
        )
        self._model = model
        self._bucket = get_bucket(api_key, rate_per_minute)

    def generate(
        self,
        *,
        system_instruction: str | None,
        user_message: str,
    ) -> str:
        config = (
            genai_types.GenerateContentConfig(
                system_instruction=system_instruction,
            )
            if system_instruction
            else None
        )
        last_exc: Exception | None = None
        for attempt in range(_RETRY_MAX_ATTEMPTS):
            self._bucket.acquire()
            try:
                resp = self._client.models.generate_content(
                    model=self._model,
                    contents=user_message,
                    config=config,
                )
                return (resp.text or "").strip()
            except _RETRYABLE_EXCEPTIONS as exc:
                last_exc = exc
                if attempt == _RETRY_MAX_ATTEMPTS - 1:
                    break
                _time.sleep(_RETRY_BASE_DELAY_S * (2**attempt))
        assert last_exc is not None
        raise last_exc
