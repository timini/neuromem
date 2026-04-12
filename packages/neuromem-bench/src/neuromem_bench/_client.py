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

from google import genai
from google.genai import types as genai_types


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

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash-001") -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model

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
        resp = self._client.models.generate_content(
            model=self._model,
            contents=user_message,
            config=config,
        )
        return (resp.text or "").strip()
