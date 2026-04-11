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
