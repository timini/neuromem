"""Smoke tests for GemmaLLMProvider / GemmaEmbeddingProvider.

Validates the thin-wrapper shape: defaults land on localhost Ollama,
subclass relationship is intact so neuromem-core treats them as
LLMProvider / EmbeddingProvider instances.
"""

from __future__ import annotations

from neuromem.providers import EmbeddingProvider, LLMProvider
from neuromem_gemma import GemmaEmbeddingProvider, GemmaLLMProvider
from neuromem_openai import OpenAIEmbeddingProvider, OpenAILLMProvider


class TestGemmaDefaults:
    def test_llm_subclasses_openai_provider(self) -> None:
        llm = GemmaLLMProvider()
        assert isinstance(llm, OpenAILLMProvider)
        assert isinstance(llm, LLMProvider)

    def test_embedder_subclasses_openai_provider(self) -> None:
        emb = GemmaEmbeddingProvider()
        assert isinstance(emb, OpenAIEmbeddingProvider)
        assert isinstance(emb, EmbeddingProvider)

    def test_llm_default_model_and_base_url(self) -> None:
        llm = GemmaLLMProvider()
        assert llm._model == "gemma3"
        # openai SDK stores base_url on the client
        assert "localhost:11434" in str(llm._client.base_url)

    def test_embedder_default_model(self) -> None:
        emb = GemmaEmbeddingProvider()
        assert emb._model == "embeddinggemma"

    def test_override_model_and_base_url(self) -> None:
        llm = GemmaLLMProvider(
            model="gemma3:27b",
            base_url="http://gpu-box.local:11434/v1",
        )
        assert llm._model == "gemma3:27b"
        assert "gpu-box.local" in str(llm._client.base_url)
