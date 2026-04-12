"""Unit tests for neuromem.providers — ABC instantiation behaviour.

Per contracts/providers.md, the two ABCs (EmbeddingProvider,
LLMProvider) must refuse direct instantiation and refuse subclasses
that are missing required abstract methods. These tests enforce that
contract at the ABC level, independent of any concrete provider.
"""

from __future__ import annotations

import numpy as np
import pytest
from neuromem.providers import EmbeddingProvider, LLMProvider

# ---------------------------------------------------------------------------
# EmbeddingProvider
# ---------------------------------------------------------------------------


class TestEmbeddingProvider:
    def test_direct_instantiation_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            EmbeddingProvider()  # type: ignore[abstract]

    def test_subclass_missing_method_cannot_instantiate(self) -> None:
        class Incomplete(EmbeddingProvider):
            pass

        with pytest.raises(TypeError, match="abstract"):
            Incomplete()  # type: ignore[abstract]

    def test_fully_implemented_subclass_instantiates(self) -> None:
        class Stub(EmbeddingProvider):
            def get_embeddings(self, texts: list[str]) -> np.ndarray:
                return np.zeros((len(texts), 4), dtype=np.float32)

        instance = Stub()
        result = instance.get_embeddings(["a", "b"])
        assert result.shape == (2, 4)


# ---------------------------------------------------------------------------
# LLMProvider
# ---------------------------------------------------------------------------


class TestLLMProvider:
    def test_direct_instantiation_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            LLMProvider()  # type: ignore[abstract]

    def test_subclass_missing_all_methods_cannot_instantiate(self) -> None:
        class Incomplete(LLMProvider):
            pass

        with pytest.raises(TypeError, match="abstract"):
            Incomplete()  # type: ignore[abstract]

    def test_subclass_missing_one_method_cannot_instantiate(self) -> None:
        class PartialOne(LLMProvider):
            def generate_summary(self, raw_text: str) -> str:
                return raw_text[:40]

            def extract_tags(self, summary: str) -> list[str]:
                return summary.split()[:3]

            # generate_category_name missing

        with pytest.raises(TypeError, match="abstract"):
            PartialOne()  # type: ignore[abstract]

    def test_subclass_missing_different_method_cannot_instantiate(self) -> None:
        class PartialTwo(LLMProvider):
            def generate_summary(self, raw_text: str) -> str:
                return raw_text[:40]

            # extract_tags missing

            def generate_category_name(self, concepts: list[str]) -> str:
                return "X"

        with pytest.raises(TypeError, match="abstract"):
            PartialTwo()  # type: ignore[abstract]

    def test_fully_implemented_subclass_instantiates(self) -> None:
        class Stub(LLMProvider):
            def generate_summary(self, raw_text: str) -> str:
                return raw_text[:40]

            def extract_tags(self, summary: str) -> list[str]:
                return summary.split()[:3]

            def generate_category_name(self, concepts: list[str]) -> str:
                return "Category"

        instance = Stub()
        assert instance.generate_summary("hello world") == "hello world"
        assert instance.extract_tags("one two three four") == ["one", "two", "three"]
        assert instance.generate_category_name(["a", "b"]) == "Category"


# ---------------------------------------------------------------------------
# extract_tags_batch default fallback (issue #44)
# ---------------------------------------------------------------------------


class TestExtractTagsBatchDefault:
    """The default ``extract_tags_batch`` is a correctness fallback
    that loops over ``extract_tags``. Providers may override with a
    real batched LLM call for performance. These tests lock in the
    contract: the default must produce IDENTICAL output to the
    per-call loop, in the same order, with the same length."""

    class _CountingStub(LLMProvider):
        """Tracks every call to ``extract_tags`` so we can assert the
        default ``extract_tags_batch`` is a faithful wrapper, not a
        silent no-op."""

        def __init__(self) -> None:
            self.calls: list[str] = []

        def generate_summary(self, raw_text: str) -> str:
            return raw_text

        def extract_tags(self, summary: str) -> list[str]:
            self.calls.append(summary)
            return summary.split()[:2]

        def generate_category_name(self, concepts: list[str]) -> str:
            return "x"

    def test_empty_batch_returns_empty_list(self) -> None:
        stub = self._CountingStub()
        assert stub.extract_tags_batch([]) == []
        assert stub.calls == []

    def test_single_item_batch_matches_per_call(self) -> None:
        stub = self._CountingStub()
        result = stub.extract_tags_batch(["alpha beta gamma"])
        assert result == [["alpha", "beta"]]
        assert stub.calls == ["alpha beta gamma"]

    def test_multi_item_batch_preserves_order_and_length(self) -> None:
        """The default fallback calls extract_tags once per input in
        order. Return value length must equal input length exactly —
        this is the invariant the dream cycle's zip(strict=True)
        relies on."""
        stub = self._CountingStub()
        summaries = [
            "alpha beta gamma delta",
            "one two three",
            "x y z q r",
        ]
        result = stub.extract_tags_batch(summaries)
        assert len(result) == len(summaries)
        assert result[0] == ["alpha", "beta"]
        assert result[1] == ["one", "two"]
        assert result[2] == ["x", "y"]
        # And the per-call calls happened in input order.
        assert stub.calls == summaries

    def test_overridden_batch_is_used_directly(self) -> None:
        """A provider that overrides ``extract_tags_batch`` with a
        real batched implementation should NOT trigger the per-call
        path. Verified by counting extract_tags invocations — a
        correct batched override never touches them."""

        class BatchedStub(LLMProvider):
            def __init__(self) -> None:
                self.per_call_count = 0
                self.batch_call_count = 0

            def generate_summary(self, raw_text: str) -> str:
                return raw_text

            def extract_tags(self, summary: str) -> list[str]:
                self.per_call_count += 1
                return ["from-per-call"]

            def extract_tags_batch(self, summaries: list[str]) -> list[list[str]]:
                self.batch_call_count += 1
                return [["batched", f"tag{i}"] for i in range(len(summaries))]

            def generate_category_name(self, concepts: list[str]) -> str:
                return "x"

        stub = BatchedStub()
        result = stub.extract_tags_batch(["a", "b", "c"])
        assert len(result) == 3
        assert result[0] == ["batched", "tag0"]
        assert result[2] == ["batched", "tag2"]
        assert stub.batch_call_count == 1
        assert stub.per_call_count == 0  # override bypassed per-call entirely


# ---------------------------------------------------------------------------
# extract_named_entities default implementations
# ---------------------------------------------------------------------------


class TestExtractNamedEntitiesDefaults:
    """``extract_named_entities`` and its batch variant are NOT
    abstract — they ship with default implementations so providers
    that don't care about NER (mocks, simple offline stubs)
    degrade gracefully to empty-list output rather than forcing
    every downstream to know about them.

    These tests lock in:
    1. A minimal provider (implementing only the abstract methods)
       can still be instantiated — the new methods are non-abstract.
    2. The default ``extract_named_entities`` returns ``[]`` — the
       "no entities" signal the dream cycle accepts.
    3. ``extract_named_entities_batch`` default = per-call loop,
       preserves length invariant.
    4. A provider that overrides the batch method gets its override
       called directly (same pattern as extract_tags_batch).
    """

    class _MinimalStub(LLMProvider):
        """Implements ONLY the abstract methods.

        Deliberately does NOT implement extract_named_entities or
        its batch variant — so we're exercising the ABC's default
        implementations directly.
        """

        def generate_summary(self, raw_text: str) -> str:
            return raw_text

        def extract_tags(self, summary: str) -> list[str]:
            return ["tag"]

        def generate_category_name(self, concepts: list[str]) -> str:
            return "cat"

    def test_minimal_provider_instantiates_without_implementing_ner(self) -> None:
        """Providers that don't know about named entities MUST NOT
        be forced to implement them. Instantiation alone proves the
        methods have default bodies (non-abstract)."""
        stub = self._MinimalStub()
        assert isinstance(stub, LLMProvider)

    def test_default_extract_named_entities_returns_empty_list(self) -> None:
        """The graceful-degradation contract. The dream cycle stores
        whatever this returns — empty list ≡ 'this memory has no
        entities', which is a valid state."""
        stub = self._MinimalStub()
        assert stub.extract_named_entities("Some summary text.") == []

    def test_default_batch_returns_empty_lists_per_input(self) -> None:
        """Length invariant applies just as strictly as the tags
        variant — the dream cycle uses zip(strict=True)."""
        stub = self._MinimalStub()
        result = stub.extract_named_entities_batch(["a", "b", "c"])
        assert result == [[], [], []]

    def test_empty_batch_input_returns_empty_batch_output(self) -> None:
        stub = self._MinimalStub()
        assert stub.extract_named_entities_batch([]) == []

    def test_overridden_single_method_is_picked_up_by_batch_loop(self) -> None:
        """A provider that overrides extract_named_entities but NOT
        the batch variant gets its override called per-item via the
        default loop."""

        class NerStub(self._MinimalStub):
            def extract_named_entities(self, summary: str) -> list[str]:
                # Capitalised words = "entities" for this stub.
                return [w for w in summary.split() if w and w[0].isupper()]

        stub = NerStub()
        result = stub.extract_named_entities_batch(
            ["Target sells Cartwheel coupons", "no entities here", "just Paris"]
        )
        assert result == [["Target", "Cartwheel"], [], ["Paris"]]

    def test_overridden_batch_is_used_directly(self) -> None:
        """Mirror of TestExtractTagsBatchDefault — a batched override
        must not fall through to per-item calls."""

        class BatchedNerStub(self._MinimalStub):
            def __init__(self) -> None:
                self.per_call_count = 0
                self.batch_call_count = 0

            def extract_named_entities(self, summary: str) -> list[str]:
                self.per_call_count += 1
                return ["from-per-call"]

            def extract_named_entities_batch(self, summaries: list[str]) -> list[list[str]]:
                self.batch_call_count += 1
                return [[f"batched{i}"] for i in range(len(summaries))]

        stub = BatchedNerStub()
        result = stub.extract_named_entities_batch(["a", "b"])
        assert result == [["batched0"], ["batched1"]]
        assert stub.batch_call_count == 1
        assert stub.per_call_count == 0
