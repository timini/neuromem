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

import json
import time as _time

from google import genai
from google.genai.errors import ServerError
from neuromem.providers import LLMProvider

# Transient errors we retry on. ServerError covers 5xx from the Gemini API.
# The remaining entries cover transport-level connection issues (server
# dropped the TCP connection, SSL handshake failure, DNS timeout, etc.)
# that surface as httpx exceptions through the google-genai SDK. We import
# them defensively — if httpx isn't installed (shouldn't happen, since
# google-genai depends on it) we just catch fewer exception types.
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


def _generate_with_retry(
    client: genai.Client,
    model: str,
    contents: str,
    *,
    max_attempts: int = 5,
    base_delay: float = 2.0,
) -> object:
    """Call ``models.generate_content`` with retry on transient errors.

    Catches both Gemini-level ``ServerError`` (5xx) and transport-
    level connection drops (``httpx.RemoteProtocolError``, etc.).
    Exponential backoff: 2/4/8/16/32s. ``ClientError`` (4xx) is
    never retried — those are deterministic failures.
    """
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
            )
        except _RETRYABLE_EXCEPTIONS as exc:
            last_exc = exc
            if attempt == max_attempts - 1:
                break
            delay = base_delay * (2**attempt)
            _time.sleep(delay)
    assert last_exc is not None
    raise last_exc


def _strip_markdown_fence(text: str) -> str:
    """Strip a leading/trailing triple-backtick fence from an LLM
    response if present.

    Gemini often wraps JSON output in ``\\`\\`\\`json ... \\`\\`\\``` fences
    despite being told not to. This helper strips both the leading
    fence (with optional language tag on the first line) and the
    trailing fence so ``json.loads`` sees clean input.

    Pure-function, no state, shared by multiple LLMProvider
    methods (currently ``extract_tags_batch``; future methods
    that request JSON output will reuse it).
    """
    text = text.strip()
    if not text.startswith("```"):
        return text
    # Drop the opening fence line (which may include a lang tag
    # like "```json") up to the first newline.
    if "\n" in text:
        text = text.split("\n", 1)[1]
    else:
        text = text[3:]
    # Drop the trailing fence if present.
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    return text.strip()


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
        resp = _generate_with_retry(self._client, self._model, prompt)
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
        resp = _generate_with_retry(self._client, self._model, prompt)
        raw = (resp.text or "").strip()
        # Robust split: strip each token, drop empties, cap at 5.
        tags = [t.strip().strip("\"'").strip() for t in raw.split(",")]
        return [t for t in tags if t][:5]

    def extract_tags_batch(self, summaries: list[str]) -> list[list[str]]:
        """Extract tags for many summaries in ONE LLM call.

        Override of the ABC's default loop implementation (which
        calls :meth:`extract_tags` once per summary). For a typical
        dream-cycle batch of ~100 memories this collapses ~100
        serial Gemini calls into one — 50–100× faster.

        Fixes issue #44: LongMemEval_s's 160-memory dream cycles were
        taking 16+ minutes against the serial implementation,
        blocking the whole benchmark package's ability to run
        ``NeuromemAdkAgent`` at any meaningful scale.

        **Protocol**:

        - Constructs a single prompt with all summaries numbered
          ``[1]``, ``[2]``, ... and asks for a JSON array of arrays
          where each inner array is the tag list for the
          corresponding numbered text.
        - Parses the response as JSON. Strips common markdown-fence
          patterns (``\\`\\`\\`json``, etc.) defensively.
        - **Falls back to the per-memory loop** on any of:
            * JSON parse failure
            * Response is not a list
            * Response length doesn't match ``len(summaries)``
            * An inner element is not a list
          This keeps the dream cycle running even when Gemini
          hiccups on the batch format — correctness first, speed
          second.
        - Caps each tag list at 5 entries (same as the single-call
          version) and strips quote characters from each tag.

        Edge cases:
        - Empty input → empty output, no LLM call.
        - Single-item input → delegates to single-call
          ``extract_tags`` so the prompt isn't wasted on a
          batch-shape for one item.
        """
        if not summaries:
            return []
        if len(summaries) == 1:
            return [self.extract_tags(summaries[0])]

        numbered = "\n".join(f"[{i + 1}] {s}" for i, s in enumerate(summaries))
        prompt = (
            f"For each of the {len(summaries)} numbered texts below, "
            "extract between 3 and 5 key concepts. Return ONLY a JSON "
            "array containing one inner array per numbered text, in "
            "order: the first inner array for text [1], the second "
            "for text [2], and so on. Each inner array contains only "
            "the concept strings. Exclude stopwords, articles, "
            "prepositions, and generic words. No preamble, no "
            "markdown fences, no explanation — ONLY the JSON.\n\n"
            f"{numbered}"
        )
        resp = _generate_with_retry(self._client, self._model, prompt)
        raw = (resp.text or "").strip()

        # Strip common markdown fence patterns. Gemini usually
        # ignores the "no markdown" instruction when the output
        # is JSON.
        raw = _strip_markdown_fence(raw)

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return [self.extract_tags(s) for s in summaries]

        if not isinstance(parsed, list) or len(parsed) != len(summaries):
            return [self.extract_tags(s) for s in summaries]

        result: list[list[str]] = []
        for i, inner in enumerate(parsed):
            if not isinstance(inner, list):
                result.append(self.extract_tags(summaries[i]))
                continue
            tags = [
                str(t).strip().strip("\"'").strip()
                for t in inner
                if t is not None and str(t).strip()
            ]
            result.append(tags[:5])
        return result

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
        resp = _generate_with_retry(self._client, self._model, prompt)
        raw = (resp.text or "").strip()
        if not raw:
            # Fall back to a deterministic synthesis so the dream cycle
            # never crashes on an empty model response. Guard against
            # the pathological case where ``concepts[0]`` is whitespace-
            # only (unlikely in practice — the tag extractor filters
            # empty strings — but ``"   ".split()`` is ``[]`` and would
            # otherwise raise IndexError here).
            tokens = concepts[0].split() if concepts[0] else []
            return tokens[0] if tokens else "concept"
        return raw.split()[0].strip(".,!?:;\"'").lower()
