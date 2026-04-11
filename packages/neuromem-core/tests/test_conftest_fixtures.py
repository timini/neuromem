"""Smoke tests for conftest.py fixtures.

Sanity-checks that MockEmbeddingProvider and MockLLMProvider are
usable via the shared fixtures. The deeper behavioural tests for
the provider ABCs themselves live in test_providers.py.
"""

from __future__ import annotations

import numpy as np

from tests.conftest import MockEmbeddingProvider, MockLLMProvider


class TestMockEmbeddingProviderFixture:
    def test_returns_correct_shape(self, mock_embedder: MockEmbeddingProvider) -> None:
        result = mock_embedder.get_embeddings(["alpha", "beta", "gamma"])
        assert result.shape == (3, 16)
        assert result.dtype == np.float32

    def test_deterministic_across_calls(
        self,
        mock_embedder: MockEmbeddingProvider,
    ) -> None:
        a = mock_embedder.get_embeddings(["hello"])
        b = mock_embedder.get_embeddings(["hello"])
        np.testing.assert_array_equal(a, b)

    def test_different_inputs_yield_different_vectors(
        self,
        mock_embedder: MockEmbeddingProvider,
    ) -> None:
        result = mock_embedder.get_embeddings(["one", "two"])
        assert not np.allclose(result[0], result[1])

    def test_respects_custom_dim(self) -> None:
        provider = MockEmbeddingProvider(dim=32)
        result = provider.get_embeddings(["x"])
        assert result.shape == (1, 32)


class TestMockLLMProviderFixture:
    def test_summary_truncates_to_80_chars(self, mock_llm: MockLLMProvider) -> None:
        long_text = "a" * 200
        assert mock_llm.generate_summary(long_text) == "a" * 80

    def test_summary_short_text_unchanged(self, mock_llm: MockLLMProvider) -> None:
        assert mock_llm.generate_summary("hello") == "hello"

    def test_extract_tags_takes_first_three_alpha_words(
        self,
        mock_llm: MockLLMProvider,
    ) -> None:
        tags = mock_llm.extract_tags("Python SQLite 42 Redis MongoDB")
        assert tags == ["Python", "SQLite", "Redis"]

    def test_extract_tags_empty_summary(self, mock_llm: MockLLMProvider) -> None:
        assert mock_llm.extract_tags("") == []

    def test_extract_tags_no_alpha(self, mock_llm: MockLLMProvider) -> None:
        assert mock_llm.extract_tags("123 456 789") == []

    def test_generate_category_name_first_letters(
        self,
        mock_llm: MockLLMProvider,
    ) -> None:
        assert mock_llm.generate_category_name(["SQLite", "Neo4j"]) == "CatSN"

    def test_generate_category_name_single_concept(
        self,
        mock_llm: MockLLMProvider,
    ) -> None:
        assert mock_llm.generate_category_name(["Python"]) == "CatP"
