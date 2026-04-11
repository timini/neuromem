"""Google Gemini implementation of ``neuromem.providers.LLMProvider``.

Wraps the ``google-genai`` SDK (the current ``google.genai`` namespace,
formerly ``google-generativeai``) to satisfy the three LLM call-sites
in the ``neuromem`` cognitive pipeline:

- ``generate_summary`` — called on every ``enqueue()`` on the caller's
  thread. Hot path. Uses a short system instruction to get 1–2
  sentences back and no preamble.
- ``extract_tags`` — called per-memory from the dreaming thread. Asks
  for 3–5 comma-separated concepts and strips stopwords. A real LLM
  fixes the "`is` / `a` became tags" issue the mock provider exhibits.
- ``generate_category_name`` — called per-cluster merge from the
  dreaming thread. Asks for exactly one word, and defensively takes
  the first token of whatever comes back just in case.

Model names default to the current stable Gemini flash line. Both
are overridable via constructor args.
"""

from __future__ import annotations

from google import genai
from neuromem.providers import LLMProvider


class GeminiLLMProvider(LLMProvider):
    """Google Gemini-backed ``LLMProvider``.

    Construct with an API key; pass a different model name if you
    want to override the default ``gemini-2.0-flash-001`` (e.g., for
    cheaper ``gemini-2.0-flash-lite`` or more capable
    ``gemini-2.5-pro``).

    The underlying ``genai.Client`` is thread-safe for concurrent
    ``generate_content`` calls, so a single instance can serve both
    the caller thread (``generate_summary`` via ``enqueue``) and the
    background dream thread (``extract_tags`` + ``generate_category_name``)
    at the same time. This is the concurrency guarantee the
    ``LLMProvider`` ABC asks for.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash-001",
    ) -> None:
        if not api_key:
            raise ValueError("api_key must be non-empty")
        self._client = genai.Client(api_key=api_key)
        self._model = model

    # ------------------------------------------------------------------
    # Summary — caller-thread hot path
    # ------------------------------------------------------------------

    def generate_summary(self, raw_text: str) -> str:
        """Return a 1–2 sentence episodic summary of ``raw_text``.

        Uses a tight prompt that suppresses preambles like "Here's a
        summary:" — the docstring on the ABC says this is on the hot
        path inside ``enqueue``, so fewer tokens is always better.
        """
        prompt = (
            "Summarise the following text in one or two sentences. "
            "Respond with ONLY the summary — no preamble, no markdown, "
            "no quotation marks.\n\n"
            f"Text: {raw_text}"
        )
        resp = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
        )
        return (resp.text or "").strip()

    # ------------------------------------------------------------------
    # Tag extraction — dream-thread
    # ------------------------------------------------------------------

    def extract_tags(self, summary: str) -> list[str]:
        """Extract 3–5 key concepts from a memory summary.

        Prompt explicitly excludes stopwords and articles — this is
        the direct fix for the "how can `is` / `a` be a tag?" issue
        the deterministic MockLLMProvider shows in unit tests.
        """
        prompt = (
            "Extract between 3 and 5 key concepts from the following text "
            "and return them as a comma-separated list. "
            "Exclude stopwords, articles, prepositions, and generic words "
            '(e.g. "the", "is", "a", "of", "to", "for"). '
            "Respond with ONLY the concepts, no numbering, no explanation, "
            "no bullet points, no markdown.\n\n"
            f"Text: {summary}"
        )
        resp = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
        )
        raw = (resp.text or "").strip()
        # Robust split: strip each token, drop empties, cap at 5.
        tags = [t.strip().strip("\"'").strip() for t in raw.split(",")]
        return [t for t in tags if t][:5]

    # ------------------------------------------------------------------
    # Category naming — dream-thread
    # ------------------------------------------------------------------

    def generate_category_name(self, concepts: list[str]) -> str:
        """Return a single one-word category name for a cluster of concepts.

        Asks Gemini for exactly one word. Defensively takes only the
        first whitespace-separated token and strips common punctuation
        so the returned value satisfies the one-word contract even if
        the model hedges with a "category: databases." style reply.
        """
        if not concepts:
            raise ValueError("concepts must be non-empty")
        prompt = (
            "Reply with EXACTLY ONE English noun (lowercase) that names "
            "the category encompassing the following concepts. "
            "No explanation, no punctuation, no markdown. One word only.\n\n"
            f"Concepts: {', '.join(concepts)}"
        )
        resp = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
        )
        raw = (resp.text or "").strip()
        if not raw:
            # Fall back to a deterministic synthesis so the dream cycle
            # never crashes on an empty model response.
            return concepts[0].split()[0]
        return raw.split()[0].strip(".,!?:;\"'").lower()
