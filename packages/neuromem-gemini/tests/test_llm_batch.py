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

    def test_caps_each_tag_list_at_12_entries(self) -> None:
        """The cap was bumped from 5 → 12 to accommodate dense
        session-level summaries with many fact-bearing topics.
        Below the cap → unchanged. Above the cap → truncated."""
        # Use 2 items to exercise the batch path (single-item batches
        # delegate to the single-call extract_tags).
        provider, _ = _make_llm_with_stub_client(
            batch_responses=[
                '[["a","b","c","d","e","f","g","h","i","j","k","l","m","n"],["x","y","z"]]'
            ],
        )
        result = provider.extract_tags_batch(["t1", "t2"])
        # First item has 14 entries → capped at 12.
        assert result[0] == [
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "h",
            "i",
            "j",
            "k",
            "l",
        ]
        # Second item under cap → unchanged.
        assert result[1] == ["x", "y", "z"]

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


# ---------------------------------------------------------------------------
# extract_named_entities — single-call path
# ---------------------------------------------------------------------------


class TestExtractNamedEntitiesSingle:
    """Mocked unit tests for the per-summary NER path.

    The production behaviour we need to lock in:
    1. A Gemini response of exactly ``NONE`` (possibly with trailing
       punctuation) → empty list. This is the "no entities" signal
       the prompt asks for.
    2. A comma-separated response → parsed into a list with quote/
       whitespace stripping, capped at 8 entries.
    3. Empty response → empty list (defensive: don't return [""]).
    """

    def test_none_response_returns_empty_list(self) -> None:
        provider, _ = _make_llm_with_stub_client(single_responses=["NONE"])
        assert provider.extract_named_entities("summary") == []

    def test_none_with_trailing_punct_returns_empty_list(self) -> None:
        """Defensive: Gemini sometimes adds punctuation even when told
        not to. 'NONE.' or 'NONE!' must still be recognised."""
        provider, _ = _make_llm_with_stub_client(single_responses=["NONE."])
        assert provider.extract_named_entities("summary") == []

    def test_empty_response_returns_empty_list(self) -> None:
        provider, _ = _make_llm_with_stub_client(single_responses=[""])
        assert provider.extract_named_entities("summary") == []

    def test_comma_separated_response_parsed_and_stripped(self) -> None:
        provider, _ = _make_llm_with_stub_client(single_responses=["Target, Cartwheel, RedCard"])
        result = provider.extract_named_entities("summary")
        assert result == ["Target", "Cartwheel", "RedCard"]

    def test_quotes_stripped_from_entities(self) -> None:
        provider, _ = _make_llm_with_stub_client(single_responses=['"Target", "Cartwheel"'])
        result = provider.extract_named_entities("summary")
        assert result == ["Target", "Cartwheel"]

    def test_result_capped_at_eight(self) -> None:
        provider, _ = _make_llm_with_stub_client(
            single_responses=["A, B, C, D, E, F, G, H, I, J, K"]
        )
        result = provider.extract_named_entities("summary")
        assert len(result) == 8
        assert result == ["A", "B", "C", "D", "E", "F", "G", "H"]


# ---------------------------------------------------------------------------
# extract_named_entities_batch — batched path + fallbacks
# ---------------------------------------------------------------------------


class TestExtractNamedEntitiesBatch:
    """Mirrors the structure of TestExtractTagsBatch — if the prompt
    contract breaks, the fallback path must keep the dream cycle
    running with correctness-preserving per-item calls."""

    def test_empty_input_no_llm_call(self) -> None:
        provider, generate = _make_llm_with_stub_client()
        assert provider.extract_named_entities_batch([]) == []
        assert generate.call_count == 0

    def test_single_input_delegates_to_per_call(self) -> None:
        """Batched prompt would be wasted on one item; the single-call
        path returns the same result with less prompt overhead."""
        provider, generate = _make_llm_with_stub_client(single_responses=["Target, Cartwheel"])
        result = provider.extract_named_entities_batch(["summary text"])
        assert result == [["Target", "Cartwheel"]]
        assert generate.call_count == 1

    def test_batch_happy_path(self) -> None:
        """Well-formed JSON array-of-arrays → parsed directly, no
        fallback calls. Empty inner arrays are valid (summary with
        no entities)."""
        provider, generate = _make_llm_with_stub_client(
            batch_responses=[
                '[["Target"], [], ["Cartwheel", "RedCard"]]',
            ]
        )
        result = provider.extract_named_entities_batch(["s1", "s2", "s3"])
        assert result == [["Target"], [], ["Cartwheel", "RedCard"]]
        assert generate.call_count == 1

    def test_markdown_fenced_batch_response(self) -> None:
        """Gemini often wraps JSON in ``\\`\\`\\`json ... \\`\\`\\```` despite
        instructions. The strip helper covers this."""
        provider, _ = _make_llm_with_stub_client(
            batch_responses=[
                '```json\n[["Target"], ["Cartwheel"]]\n```',
            ]
        )
        result = provider.extract_named_entities_batch(["s1", "s2"])
        assert result == [["Target"], ["Cartwheel"]]

    def test_malformed_json_falls_back_to_per_call(self) -> None:
        """Garbage-JSON batch response → N per-item calls."""
        provider, generate = _make_llm_with_stub_client(
            batch_responses=["not even close to json"],
            single_responses=["NONE", "Beta"],
        )
        result = provider.extract_named_entities_batch(["alpha", "beta"])
        assert result == [[], ["Beta"]]
        # 1 failed batch + 2 per-item fallbacks.
        assert generate.call_count == 3

    def test_wrong_length_response_falls_back(self) -> None:
        provider, generate = _make_llm_with_stub_client(
            batch_responses=['[["only-one"]]'],
            single_responses=["One", "Two"],
        )
        result = provider.extract_named_entities_batch(["a", "b"])
        assert result == [["One"], ["Two"]]
        assert generate.call_count == 3

    def test_non_list_inner_element_fallback_per_item(self) -> None:
        """A batch that's mostly well-formed but has one bad inner
        element falls back for only that item, not the whole batch."""
        provider, generate = _make_llm_with_stub_client(
            batch_responses=['[["OK"], "bad", ["Fine"]]'],
            single_responses=["Recovered"],
        )
        result = provider.extract_named_entities_batch(["s1", "s2", "s3"])
        assert result == [["OK"], ["Recovered"], ["Fine"]]
        # 1 batch + 1 per-item fallback for the broken middle entry.
        assert generate.call_count == 2

    def test_inner_list_capped_at_eight(self) -> None:
        """Two-summary batch with well-formed JSON and a 10-entity
        inner list → the inner list is capped at 8, matching the
        single-call cap."""
        provider, _ = _make_llm_with_stub_client(
            batch_responses=[
                '[["A","B","C","D","E","F","G","H","I","J"], []]',
            ]
        )
        result = provider.extract_named_entities_batch(["x", "y"])
        assert len(result) == 2
        assert len(result[0]) == 8
        assert result[0] == ["A", "B", "C", "D", "E", "F", "G", "H"]
        assert result[1] == []


# ---------------------------------------------------------------------------
# generate_category_names_batch — ADR-002 lazy-naming Gemini override
# ---------------------------------------------------------------------------


class TestGenerateCategoryNamesBatch:
    """Gemini override of the batched naming method used by
    NeuroMemory.resolve_centroid_names. Same chunking + parallelism
    + per-chunk fallback shape as extract_tags_batch.

    Mocked client; no Gemini API calls.
    """

    def test_empty_input_no_llm_call(self) -> None:
        provider, generate = _make_llm_with_stub_client()
        assert provider.generate_category_names_batch([]) == []
        assert generate.call_count == 0

    def test_single_pair_delegates_to_per_call(self) -> None:
        """One-pair input takes the per-call generate_category_name
        path; no batched prompt is wasted on a single-item batch."""
        provider, generate = _make_llm_with_stub_client(single_responses=["voucher"])
        result = provider.generate_category_names_batch([["coupon", "discount"]])
        assert result == ["voucher"]
        assert generate.call_count == 1

    def test_batch_happy_path(self) -> None:
        """Well-formed JSON array → parsed, lowercased, first-word-only."""
        provider, _ = _make_llm_with_stub_client(
            batch_responses=['["voucher", "canine", "country"]']
        )
        result = provider.generate_category_names_batch(
            [["coupon", "discount"], ["dog", "puppy"], ["paris", "france"]]
        )
        assert result == ["voucher", "canine", "country"]

    def test_markdown_fenced_response_is_stripped(self) -> None:
        provider, _ = _make_llm_with_stub_client(batch_responses=['```json\n["one", "two"]\n```'])
        result = provider.generate_category_names_batch([["a", "b"], ["c", "d"]])
        assert result == ["one", "two"]

    def test_first_word_only_extracted(self) -> None:
        """Multi-word reply → first word only, lowercased, punctuation stripped."""
        provider, _ = _make_llm_with_stub_client(
            batch_responses=['["voucher type", "Canine animals."]']
        )
        result = provider.generate_category_names_batch([["coupon", "discount"], ["dog", "puppy"]])
        assert result == ["voucher", "canine"]

    def test_malformed_json_falls_back_per_pair(self) -> None:
        provider, generate = _make_llm_with_stub_client(
            batch_responses=["not even json"],
            single_responses=["alpha", "beta"],
        )
        result = provider.generate_category_names_batch([["a", "b"], ["c", "d"]])
        assert result == ["alpha", "beta"]
        # 1 failed batch + 2 per-pair fallback.
        assert generate.call_count == 3

    def test_wrong_length_response_falls_back(self) -> None:
        provider, generate = _make_llm_with_stub_client(
            batch_responses=['["only-one"]'],
            single_responses=["x", "y"],
        )
        result = provider.generate_category_names_batch([["a", "b"], ["c", "d"]])
        assert result == ["x", "y"]
        assert generate.call_count == 3

    def test_non_string_inner_element_per_pair_fallback(self) -> None:
        """One bad inner element falls back per-pair just for THAT
        slot, not the whole batch."""
        provider, generate = _make_llm_with_stub_client(
            batch_responses=['["good", 42, "fine"]'],
            single_responses=["recovered"],
        )
        result = provider.generate_category_names_batch([["a", "b"], ["c", "d"], ["e", "f"]])
        assert result == ["good", "recovered", "fine"]
        # 1 batch + 1 per-pair fallback for the bad middle slot.
        assert generate.call_count == 2
