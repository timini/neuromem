"""Mocked-client unit tests for AnthropicLLMProvider."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from neuromem_anthropic.llm import AnthropicLLMProvider


def _make_llm_with_stub_client(
    responses: list[str] | None = None,
) -> tuple[AnthropicLLMProvider, MagicMock]:
    with patch("neuromem_anthropic.llm.Anthropic") as client_cls:
        client = MagicMock()
        client_cls.return_value = client
        provider = AnthropicLLMProvider(api_key="sk-ant-test")

    create_mock = client.messages.create
    if responses is not None:
        # Response shape: resp.content[0].text
        create_mock.side_effect = [
            SimpleNamespace(content=[SimpleNamespace(text=text)]) for text in responses
        ]
    return provider, create_mock


class TestBasicCalls:
    def test_generate_summary_returns_text(self) -> None:
        provider, create = _make_llm_with_stub_client(responses=["A 2-sentence summary."])
        out = provider.generate_summary("Some raw text to summarise.")
        assert out == "A 2-sentence summary."
        assert create.call_count == 1
        # max_tokens is required by Anthropic SDK.
        _args, kwargs = create.call_args
        assert "max_tokens" in kwargs

    def test_generate_category_name_strips_and_lowercases(self) -> None:
        provider, create = _make_llm_with_stub_client(responses=["Finance."])
        out = provider.generate_category_name(["mortgage", "loan"])
        assert out == "finance"

    def test_generate_category_name_blocks_generic(self) -> None:
        provider, create = _make_llm_with_stub_client(responses=["thing"])
        out = provider.generate_category_name(["coupon", "voucher"])
        # Fell back to first concept.
        assert out == "coupon"


class TestBatched:
    def test_summary_batch_json_roundtrip(self) -> None:
        provider, create = _make_llm_with_stub_client(responses=['["s1", "s2", "s3"]'])
        out = provider.generate_summary_batch(["a", "b", "c"])
        assert out == ["s1", "s2", "s3"]
        assert create.call_count == 1

    def test_summary_batch_parse_failure_falls_back(self) -> None:
        provider, create = _make_llm_with_stub_client(responses=["not json", "fa", "fb"])
        out = provider.generate_summary_batch(["a", "b"])
        assert out == ["fa", "fb"]
        assert create.call_count == 3

    def test_tags_batch_roundtrip(self) -> None:
        provider, create = _make_llm_with_stub_client(responses=['[["a", "b"], ["c"]]'])
        out = provider.extract_tags_batch(["first", "second"])
        assert out == [["a", "b"], ["c"]]
        assert create.call_count == 1

    def test_category_names_batch_roundtrip(self) -> None:
        provider, create = _make_llm_with_stub_client(
            responses=['["shopping", "finance", "family"]']
        )
        out = provider.generate_category_names_batch([["a", "b"], ["c", "d"], ["e", "f"]])
        assert out == ["shopping", "finance", "family"]

    def test_category_names_batch_blocks_generic_slot(self) -> None:
        provider, create = _make_llm_with_stub_client(
            responses=['["shopping", "aspect", "finance"]', "education"]
        )
        out = provider.generate_category_names_batch([["a", "b"], ["c", "d"], ["e", "f"]])
        # Middle slot was blocklisted → re-rolled via single call.
        assert out == ["shopping", "education", "finance"]
        # 1 batched + 1 per-slot reroll.
        assert create.call_count == 2


class TestConstructorValidation:
    def test_empty_api_key_raises(self) -> None:
        with pytest.raises(ValueError, match="api_key"):
            AnthropicLLMProvider(api_key="")

    def test_empty_model_raises(self) -> None:
        with pytest.raises(ValueError, match="model"):
            AnthropicLLMProvider(api_key="sk-ant-test", model="")
