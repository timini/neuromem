"""Unit tests for neuromem.system.NeuroMemory.

Grown incrementally:
  - T019: constructor validation + sync enqueue path (this file's
    initial content)
  - T020: _run_dream_cycle pipeline tests
  - T021: threading + force_dream concurrency tests
  - T024: US3 decay/archival end-to-end tests
  - T026: US5 force_dream acceptance tests
"""

from __future__ import annotations

import time

import pytest
from neuromem.providers import EmbeddingProvider, LLMProvider
from neuromem.storage.sqlite import SQLiteAdapter
from neuromem.system import NeuroMemory

from tests.conftest import MockEmbeddingProvider, MockLLMProvider


@pytest.fixture
def memory_system(
    mock_embedder: MockEmbeddingProvider,
    mock_llm: MockLLMProvider,
) -> NeuroMemory:
    """A freshly-constructed NeuroMemory with an in-memory SQLite backend."""
    return NeuroMemory(
        storage=SQLiteAdapter(":memory:"),
        llm=mock_llm,
        embedder=mock_embedder,
    )


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_accepts_valid_dependencies(
        self,
        mock_embedder: MockEmbeddingProvider,
        mock_llm: MockLLMProvider,
    ) -> None:
        system = NeuroMemory(
            storage=SQLiteAdapter(":memory:"),
            llm=mock_llm,
            embedder=mock_embedder,
        )
        assert system.dream_threshold == 10
        assert system.is_dreaming is False

    def test_accepts_custom_thresholds(
        self,
        mock_embedder: MockEmbeddingProvider,
        mock_llm: MockLLMProvider,
    ) -> None:
        system = NeuroMemory(
            storage=SQLiteAdapter(":memory:"),
            llm=mock_llm,
            embedder=mock_embedder,
            dream_threshold=5,
            decay_lambda=1e-6,
            archive_threshold=0.2,
            cluster_threshold=0.75,
        )
        assert system.dream_threshold == 5
        assert system.decay_lambda == pytest.approx(1e-6)
        assert system.archive_threshold == pytest.approx(0.2)
        assert system.cluster_threshold == pytest.approx(0.75)

    def test_rejects_non_storage_adapter(
        self,
        mock_embedder: MockEmbeddingProvider,
        mock_llm: MockLLMProvider,
    ) -> None:
        with pytest.raises(TypeError, match="StorageAdapter"):
            NeuroMemory(
                storage="not an adapter",  # type: ignore[arg-type]
                llm=mock_llm,
                embedder=mock_embedder,
            )

    def test_rejects_non_llm_provider(
        self,
        mock_embedder: MockEmbeddingProvider,
    ) -> None:
        with pytest.raises(TypeError, match="LLMProvider"):
            NeuroMemory(
                storage=SQLiteAdapter(":memory:"),
                llm=object(),  # type: ignore[arg-type]
                embedder=mock_embedder,
            )

    def test_rejects_non_embedding_provider(
        self,
        mock_llm: MockLLMProvider,
    ) -> None:
        with pytest.raises(TypeError, match="EmbeddingProvider"):
            NeuroMemory(
                storage=SQLiteAdapter(":memory:"),
                llm=mock_llm,
                embedder=object(),  # type: ignore[arg-type]
            )

    def test_rejects_zero_dream_threshold(
        self,
        mock_embedder: MockEmbeddingProvider,
        mock_llm: MockLLMProvider,
    ) -> None:
        with pytest.raises(ValueError, match="dream_threshold"):
            NeuroMemory(
                storage=SQLiteAdapter(":memory:"),
                llm=mock_llm,
                embedder=mock_embedder,
                dream_threshold=0,
            )

    def test_rejects_negative_decay_lambda(
        self,
        mock_embedder: MockEmbeddingProvider,
        mock_llm: MockLLMProvider,
    ) -> None:
        with pytest.raises(ValueError, match="decay_lambda"):
            NeuroMemory(
                storage=SQLiteAdapter(":memory:"),
                llm=mock_llm,
                embedder=mock_embedder,
                decay_lambda=-1e-6,
            )

    def test_rejects_out_of_range_archive_threshold(
        self,
        mock_embedder: MockEmbeddingProvider,
        mock_llm: MockLLMProvider,
    ) -> None:
        with pytest.raises(ValueError, match="archive_threshold"):
            NeuroMemory(
                storage=SQLiteAdapter(":memory:"),
                llm=mock_llm,
                embedder=mock_embedder,
                archive_threshold=1.5,
            )

    def test_rejects_out_of_range_cluster_threshold(
        self,
        mock_embedder: MockEmbeddingProvider,
        mock_llm: MockLLMProvider,
    ) -> None:
        with pytest.raises(ValueError, match="cluster_threshold"):
            NeuroMemory(
                storage=SQLiteAdapter(":memory:"),
                llm=mock_llm,
                embedder=mock_embedder,
                cluster_threshold=0.0,
            )


# ---------------------------------------------------------------------------
# enqueue (sync path)
# ---------------------------------------------------------------------------


class TestEnqueueSync:
    def test_returns_memory_id(self, memory_system: NeuroMemory) -> None:
        mem_id = memory_system.enqueue("hello world")
        assert isinstance(mem_id, str)
        assert mem_id.startswith("mem_")

    def test_memory_lands_in_inbox_status(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        mem_id = memory_system.enqueue("raw content")
        row = memory_system.storage.get_memory_by_id(mem_id)
        assert row is not None
        assert row["status"] == "inbox"
        assert row["access_weight"] == pytest.approx(1.0)
        assert row["raw_content"] == "raw content"

    def test_summary_populated_from_llm(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        # MockLLMProvider returns raw_text[:80] as the summary.
        long_text = "a" * 200
        mem_id = memory_system.enqueue(long_text)
        row = memory_system.storage.get_memory_by_id(mem_id)
        assert row["summary"] == "a" * 80

    def test_metadata_round_trip(self, memory_system: NeuroMemory) -> None:
        mem_id = memory_system.enqueue("hi", metadata={"role": "user", "turn": 3})
        row = memory_system.storage.get_memory_by_id(mem_id)
        assert row["metadata"] == {"role": "user", "turn": 3}

    def test_empty_raw_text_raises(self, memory_system: NeuroMemory) -> None:
        with pytest.raises(ValueError, match="raw_text"):
            memory_system.enqueue("")

    def test_enqueue_latency_sc002(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        """SC-002: enqueue must return in <50 ms excluding generate_summary.

        Since MockLLMProvider.generate_summary is essentially free (a
        string slice), total enqueue latency here is dominated by the
        SQLite write. A 50 ms budget with MockLLMProvider is a very
        loose bound — a real failure would show latency in the
        hundreds of ms, indicating a serious issue.
        """
        start = time.perf_counter()
        memory_system.enqueue("hello")
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        assert elapsed_ms < 50.0, f"enqueue took {elapsed_ms:.1f} ms (budget 50)"

    def test_multiple_enqueues_all_in_inbox(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        for i in range(5):
            memory_system.enqueue(f"memory {i}")
        assert memory_system.storage.count_memories_by_status("inbox") == 5

    def test_enqueue_below_threshold_does_not_spawn_dream(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        # Default threshold is 10; 5 enqueues should leave is_dreaming False.
        for i in range(5):
            memory_system.enqueue(f"memory {i}")
        assert memory_system.is_dreaming is False

    def test_concrete_provider_types_accepted(
        self,
        mock_embedder: MockEmbeddingProvider,
    ) -> None:
        """Custom subclasses of the ABCs must work."""

        class TerseLLM(LLMProvider):
            def generate_summary(self, raw_text: str) -> str:
                return raw_text[:20]

            def extract_tags(self, summary: str) -> list[str]:
                return []

            def generate_category_name(self, concepts: list[str]) -> str:
                return "Misc"

        system = NeuroMemory(
            storage=SQLiteAdapter(":memory:"),
            llm=TerseLLM(),
            embedder=mock_embedder,
        )
        mem_id = system.enqueue("hello world this is a long sentence")
        row = system.storage.get_memory_by_id(mem_id)
        # TerseLLM returns raw_text[:20]. Index 20 of the input is the
        # space character AFTER "is", so the slice is 20 chars with a
        # trailing space — exactly what Python produces. No off-by-one
        # adjustment in the assertion.
        assert row["summary"] == "hello world this is "


# ---------------------------------------------------------------------------
# force_dream (T020/T021 placeholder — tests land in those tasks)
# ---------------------------------------------------------------------------


class TestForceDreamPlaceholder:
    def test_force_dream_raises_until_t020(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        with pytest.raises(NotImplementedError, match="T020"):
            memory_system.force_dream()


def test_explicit_ignore_unused_embedding_provider_import() -> None:
    """Referenced for type-checker compatibility."""
    # EmbeddingProvider is referenced by type annotations in test fixtures.
    _ = EmbeddingProvider
