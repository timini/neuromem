"""Shared pytest fixtures for neuromem-core.

Populated incrementally by:
  - T011: MockEmbeddingProvider, MockLLMProvider fixtures
  - T012: storage_adapter parametrised fixture (initially empty params)
  - T013: SQLiteAdapter added to storage_adapter params
  - T025: DictStorageAdapter added to storage_adapter params
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import pytest
from neuromem.providers import EmbeddingProvider, LLMProvider

if TYPE_CHECKING:
    from neuromem.storage.base import StorageAdapter


# ---------------------------------------------------------------------------
# Mock providers (T011)
# ---------------------------------------------------------------------------


class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic seeded-RNG embedding for tests.

    For any input string, produces a fixed-dimension float32 vector
    derived from the md5 hash of the string. Two identical input
    strings always yield the same vector. Different input strings
    yield statistically uncorrelated vectors. Zero network access.
    """

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim

    def get_embeddings(self, texts: list[str]) -> np.ndarray:
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            digest = hashlib.md5(text.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:8], "big")
            rng = np.random.default_rng(seed)
            out[i] = rng.uniform(-1.0, 1.0, size=self.dim).astype(np.float32)
        return out


class MockLLMProvider(LLMProvider):
    """Deterministic canned-response LLM for tests.

    - generate_summary: first 80 chars of raw_text
    - extract_tags: first 3 alpha tokens from the summary
    - generate_category_name: 'Cat' + first letter of each concept

    Fast, deterministic, zero network access. The real test of the
    cognitive pipeline is that it calls these methods in the right
    order with the right arguments, which MockLLMProvider captures
    structurally (just by being a valid provider).
    """

    def generate_summary(self, raw_text: str) -> str:
        return raw_text[:80]

    def extract_tags(self, summary: str) -> list[str]:
        return [w for w in summary.split() if w.isalpha()][:3]

    def generate_category_name(self, concepts: list[str]) -> str:
        return "Cat" + "".join(c[:1].upper() for c in concepts)


@pytest.fixture
def mock_embedder() -> MockEmbeddingProvider:
    """A freshly-constructed 16-dim MockEmbeddingProvider."""
    return MockEmbeddingProvider(dim=16)


@pytest.fixture
def mock_llm() -> MockLLMProvider:
    """A freshly-constructed MockLLMProvider."""
    return MockLLMProvider()


# ---------------------------------------------------------------------------
# StorageAdapter parametrised fixture (T012)
# ---------------------------------------------------------------------------

# List of ``() -> StorageAdapter`` callables. Each entry produces a
# fresh adapter instance per test. Starts empty; T013 appends a
# SQLiteAdapter factory; T025 appends a DictStorageAdapter factory.
STORAGE_ADAPTER_FACTORIES: list[Callable[[], StorageAdapter]] = []


@pytest.fixture(
    params=STORAGE_ADAPTER_FACTORIES,
    ids=lambda f: f.__name__.replace("_factory", ""),
)
def storage_adapter(request: pytest.FixtureRequest) -> StorageAdapter:
    """Parametrised fixture over every concrete StorageAdapter.

    Used by test_storage_adapter_contract.py to run the 13-item
    contract test suite against every implementation in a single
    pytest run. Starts empty (no adapters registered); tasks T013
    and T025 append SQLiteAdapter and DictStorageAdapter factories
    respectively. With an empty params list, every contract test
    is collected with 0 parameterisations and skipped silently.
    """
    factory: Callable[[], StorageAdapter] = request.param
    return factory()
