"""Mocked-client unit tests for OpenAILLMProvider.

Validates the abstract contract and the batched paths' JSON
round-tripping. No real API calls — the openai.OpenAI client is
patched with a MagicMock.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from neuromem_openai.llm import OpenAILLMProvider


def _make_llm_with_stub_client(
    responses: list[str] | None = None,
) -> tuple[OpenAILLMProvider, MagicMock]:
    """Build an OpenAILLMProvider with ``openai.OpenAI`` patched out.

    ``responses`` is the sequence of strings returned by successive
    ``chat.completions.create`` calls (useful for tests that exercise
    fallback paths). Returns (provider, completions_create_mock).
    """
    with patch("neuromem_openai.llm.OpenAI") as client_cls:
        client = MagicMock()
        client_cls.return_value = client
        provider = OpenAILLMProvider(api_key="sk-test")

    create_mock = client.chat.completions.create
    if responses is not None:
        # Wrap each response string in the shape ``resp.choices[0].message.content``.
        create_mock.side_effect = [
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])
            for text in responses
        ]
    return provider, create_mock


class TestGenerateSummary:
    def test_single_call_passes_prompt_and_returns_text(self) -> None:
        provider, create = _make_llm_with_stub_client(responses=["Short summary of the raw text."])
        out = provider.generate_summary("I graduated with a Business Administration degree.")
        assert out == "Short summary of the raw text."
        assert create.call_count == 1


class TestGenerateSummaryBatch:
    def test_empty_input_returns_empty(self) -> None:
        provider, create = _make_llm_with_stub_client()
        assert provider.generate_summary_batch([]) == []
        assert create.call_count == 0

    def test_single_input_delegates_to_single_call(self) -> None:
        provider, create = _make_llm_with_stub_client(responses=["single"])
        out = provider.generate_summary_batch(["hello"])
        assert out == ["single"]
        assert create.call_count == 1

    def test_batch_json_roundtrip(self) -> None:
        provider, create = _make_llm_with_stub_client(
            responses=['{"summaries": ["s1", "s2", "s3"]}']
        )
        out = provider.generate_summary_batch(["a", "b", "c"])
        assert out == ["s1", "s2", "s3"]
        assert create.call_count == 1

    def test_batch_json_parse_failure_falls_back_per_item(self) -> None:
        # First call (batched) returns garbage; next 3 calls provide
        # per-item fallbacks.
        provider, create = _make_llm_with_stub_client(
            responses=["not json at all", "fa", "fb", "fc"]
        )
        out = provider.generate_summary_batch(["a", "b", "c"])
        assert out == ["fa", "fb", "fc"]
        # 1 failed batch + 3 per-item calls = 4.
        assert create.call_count == 4


class TestExtractTagsBatch:
    def test_batch_json_roundtrip(self) -> None:
        provider, create = _make_llm_with_stub_client(
            responses=['{"tags": [["alpha", "beta"], ["gamma"]]}']
        )
        out = provider.extract_tags_batch(["first", "second"])
        assert out == [["alpha", "beta"], ["gamma"]]
        assert create.call_count == 1

    def test_length_mismatch_falls_back(self) -> None:
        provider, create = _make_llm_with_stub_client(
            responses=['{"tags": [["only one"]]}', "t1", "t2"]
        )
        out = provider.extract_tags_batch(["a", "b"])
        # Fell back to per-item extract_tags (comma-split).
        assert out == [["t1"], ["t2"]]
        assert create.call_count == 3


class TestGenerateCategoryName:
    def test_blocked_generic_falls_back_to_first_concept(self) -> None:
        """ADR-003 F3: 'thing' → fallback to first concept's first word."""
        provider, create = _make_llm_with_stub_client(responses=["thing"])
        out = provider.generate_category_name(["coupon", "voucher"])
        assert out == "coupon"
        assert create.call_count == 1

    def test_valid_name_passes_through_lowercased(self) -> None:
        provider, create = _make_llm_with_stub_client(responses=["Shopping."])
        out = provider.generate_category_name(["coupon", "voucher"])
        assert out == "shopping"
        assert create.call_count == 1


class TestGenerateJunctionSummary:
    def test_empty_children_returns_empty(self) -> None:
        provider, create = _make_llm_with_stub_client()
        out = provider.generate_junction_summary([])
        assert out == ""
        assert create.call_count == 0

    def test_single_child_single_call(self) -> None:
        provider, create = _make_llm_with_stub_client(responses=["The branch covers X and Y."])
        out = provider.generate_junction_summary(["child summary a"])
        assert out == "The branch covers X and Y."
        assert create.call_count == 1


class TestExtractNamedEntities:
    def test_none_returns_empty_list(self) -> None:
        provider, create = _make_llm_with_stub_client(responses=["NONE"])
        out = provider.extract_named_entities("Generic text with no proper nouns.")
        assert out == []

    def test_comma_list_parsed(self) -> None:
        provider, create = _make_llm_with_stub_client(responses=["Target, Sarah, Shoreditch"])
        out = provider.extract_named_entities("Some text.")
        assert out == ["Target", "Sarah", "Shoreditch"]


class TestConstructorValidation:
    def test_empty_api_key_raises(self) -> None:
        with pytest.raises(ValueError, match="api_key"):
            OpenAILLMProvider(api_key="")

    def test_empty_model_raises(self) -> None:
        with pytest.raises(ValueError, match="model"):
            OpenAILLMProvider(api_key="sk-test", model="")
