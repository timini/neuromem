"""Unit tests for GeminiLLMProvider.extract_tags_batch (issue #44).

These tests run against a mocked ``google.genai.Client`` — no
network calls, no API cost. The integration test in
``test_cognitive_loop_real_llm.py`` covers the real-API path
for end-to-end confidence.

What we verify:
- Empty input → empty output, no LLM call fired
- Single-item input → delegates to single-call ``extract_tags``
- Multi-item input → ONE batched LLM call, result length matches input
- Markdown-fenced JSON response → parsed correctly
- Malformed JSON → falls back to per-call loop
- Wrong result length → falls back to per-call loop
- Non-list inner element → individual fallback for that item only
- Module-level ``_strip_markdown_fence`` helper covers its edge cases
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from neuromem_gemini.llm import GeminiLLMProvider, _strip_markdown_fence


def _make_llm_with_stub_client(
    batch_responses: list[str] | None = None,
    single_responses: list[str] | None = None,
) -> tuple[GeminiLLMProvider, MagicMock]:
    """Build a GeminiLLMProvider with the internal client replaced by
    a mock. Returns the provider and the mock's generate_content
    method so the caller can inspect call history.

    ``batch_responses`` / ``single_responses`` are returned in order
    on successive calls — useful for tests that exercise fallback
    paths (first call returns bad JSON, second call gets the
    fallback per-memory response).
    """
    # Prevent the real google.genai.Client constructor from firing —
    # it tries to configure credentials at import time.
    with patch("neuromem_gemini.llm.genai.Client") as client_cls:
        client = MagicMock()
        client_cls.return_value = client
        provider = GeminiLLMProvider(api_key="test-key")

    generate_content = client.models.generate_content

    # Build a side-effect list: batch responses first, then single
    # responses. We rely on the test structure to know the order.
    responses: list[SimpleNamespace] = []
    for text in (batch_responses or []) + (single_responses or []):
        responses.append(SimpleNamespace(text=text))

    if responses:
        generate_content.side_effect = responses

    return provider, generate_content


# ---------------------------------------------------------------------------
# _strip_markdown_fence — the helper
# ---------------------------------------------------------------------------


class TestStripMarkdownFence:
    def test_passes_unfenced_through_unchanged(self) -> None:
        assert _strip_markdown_fence('[["a"]]') == '[["a"]]'

    def test_strips_plain_triple_backtick_fence(self) -> None:
        assert _strip_markdown_fence('```\n[["a"]]\n```') == '[["a"]]'

    def test_strips_json_language_tagged_fence(self) -> None:
        assert _strip_markdown_fence('```json\n[["a"]]\n```') == '[["a"]]'

    def test_strips_fence_with_leading_trailing_whitespace(self) -> None:
        assert _strip_markdown_fence('  \n```json\n[["a"]]\n```  \n') == '[["a"]]'

    def test_handles_no_trailing_fence(self) -> None:
        # Broken/truncated response — at least strip what we can.
        assert _strip_markdown_fence('```json\n[["a"]]') == '[["a"]]'


# ---------------------------------------------------------------------------
# extract_tags_batch happy path
# ---------------------------------------------------------------------------


class TestExtractTagsBatchHappyPath:
    def test_empty_input_returns_empty_and_makes_no_api_call(self) -> None:
        provider, generate = _make_llm_with_stub_client()
        assert provider.extract_tags_batch([]) == []
        generate.assert_not_called()

    def test_single_item_delegates_to_single_call(self) -> None:
        """One-item batches skip the JSON-array prompt and go
        through the single ``extract_tags`` path instead — cleaner
        prompt, no JSON parsing risk."""
        provider, generate = _make_llm_with_stub_client(
            single_responses=["alpha, beta, gamma"],
        )
        result = provider.extract_tags_batch(["Some text about alpha beta"])
        assert result == [["alpha", "beta", "gamma"]]
        assert generate.call_count == 1

    def test_multi_item_batch_parses_json_array_of_arrays(self) -> None:
        """The happy path: Gemini returns a JSON array of arrays with
        the right shape, we parse it and return tag lists in input
        order."""
        provider, generate = _make_llm_with_stub_client(
            batch_responses=[
                '[["python", "sqlite"], ["async", "event loop"], ["numpy", "broadcasting"]]'
            ],
        )
        result = provider.extract_tags_batch(
            [
                "Python SQLite discussion",
                "Asyncio event loops",
                "Numpy broadcasting example",
            ]
        )
        assert result == [
            ["python", "sqlite"],
            ["async", "event loop"],
            ["numpy", "broadcasting"],
        ]
        # Exactly ONE batched call, not three per-memory calls.
        assert generate.call_count == 1

    def test_multi_item_batch_strips_markdown_fences(self) -> None:
        """Gemini ignores the 'no markdown' instruction sometimes —
        the parser strips fences defensively."""
        provider, generate = _make_llm_with_stub_client(
            batch_responses=['```json\n[["a"], ["b"]]\n```'],
        )
        result = provider.extract_tags_batch(["one", "two"])
        assert result == [["a"], ["b"]]
        assert generate.call_count == 1

    def test_caps_each_tag_list_at_5_entries(self) -> None:
        """Per the original ``extract_tags`` contract, we cap at
        5 tags per item. The batch path enforces the same."""
        provider, generate = _make_llm_with_stub_client(
            batch_responses=['[["t1","t2","t3","t4","t5","t6","t7"]]'],
        )
        result = provider.extract_tags_batch(["text"])
        # Wait — single-item batches use the single-call path,
        # which has its own cap. Use 2 items to exercise the
        # batch path with oversize inner lists.
        # Rebuild the provider for a fresh call history.
        provider, generate = _make_llm_with_stub_client(
            batch_responses=['[["a","b","c","d","e","f","g"],["x","y","z"]]'],
        )
        result = provider.extract_tags_batch(["t1", "t2"])
        assert result[0] == ["a", "b", "c", "d", "e"]  # capped
        assert result[1] == ["x", "y", "z"]  # under cap, unchanged

    def test_strips_quote_characters_from_tags(self) -> None:
        """Gemini sometimes wraps individual strings in extra quotes
        inside the JSON. The parser strips them."""
        provider, generate = _make_llm_with_stub_client(
            batch_responses=['[["\\"python\\"", "\'sqlite\'"], ["async"]]'],
        )
        result = provider.extract_tags_batch(["a", "b"])
        assert result == [["python", "sqlite"], ["async"]]


# ---------------------------------------------------------------------------
# extract_tags_batch fallback paths — robustness
# ---------------------------------------------------------------------------


class TestExtractTagsBatchFallbacks:
    def test_malformed_json_falls_back_to_per_call(self) -> None:
        """When Gemini returns something un-parseable, we fall back
        to one ``extract_tags`` call per input so the dream cycle
        still completes correctly, just slower."""
        provider, generate = _make_llm_with_stub_client(
            batch_responses=["not a json response at all"],
            single_responses=[
                "per, call, one",
                "per, call, two",
            ],
        )
        result = provider.extract_tags_batch(["text one", "text two"])
        assert result == [
            ["per", "call", "one"],
            ["per", "call", "two"],
        ]
        # 1 batch call + 2 per-call fallbacks = 3 total.
        assert generate.call_count == 3

    def test_wrong_length_response_falls_back(self) -> None:
        """If Gemini returns a JSON array whose length doesn't match
        the input, we fall back entirely. Length mismatch is a
        correctness red flag — we can't guess which text maps to
        which inner list."""
        provider, generate = _make_llm_with_stub_client(
            batch_responses=['[["a"], ["b"]]'],  # 2 items but we asked for 3
            single_responses=[
                "x, y",
                "p, q",
                "m, n",
            ],
        )
        result = provider.extract_tags_batch(["t1", "t2", "t3"])
        assert result == [["x", "y"], ["p", "q"], ["m", "n"]]
        assert generate.call_count == 4  # 1 batch + 3 per-call fallbacks

    def test_non_list_inner_falls_back_for_that_item_only(self) -> None:
        """If one inner element isn't a list (e.g. Gemini returned a
        string instead of a list for one entry), we fall back for
        THAT item only, not the whole batch. Preserves the speedup
        for the well-formed items."""
        provider, generate = _make_llm_with_stub_client(
            batch_responses=['[["good", "list"], "oops this is a string", ["fine", "too"]]'],
            single_responses=["recovery, tags"],
        )
        result = provider.extract_tags_batch(["t1", "t2", "t3"])
        assert result[0] == ["good", "list"]
        assert result[1] == ["recovery", "tags"]  # fallback for item 2
        assert result[2] == ["fine", "too"]
        # 1 batch call + 1 per-call fallback for the broken item.
        assert generate.call_count == 2

    def test_top_level_not_a_list_falls_back(self) -> None:
        """If Gemini returns valid JSON but the top-level value
        isn't a list (e.g. returned a dict), we fall back entirely."""
        provider, generate = _make_llm_with_stub_client(
            batch_responses=['{"tags": ["a", "b"]}'],
            single_responses=["alpha", "beta"],
        )
        result = provider.extract_tags_batch(["x", "y"])
        assert result == [["alpha"], ["beta"]]
        assert generate.call_count == 3
