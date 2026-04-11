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

import numpy as np
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
# force_dream (T020 Half A: block=True runs sync; block=False lands in T021)
# ---------------------------------------------------------------------------


class TestForceDream:
    def test_block_true_runs_synchronously(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        """force_dream(block=True) runs _run_dream_cycle directly."""
        for i in range(3):
            memory_system.enqueue(f"memory {i}")
        memory_system.force_dream(block=True)
        # After dreaming, all memories must be consolidated.
        assert memory_system.storage.count_memories_by_status("inbox") == 0
        assert memory_system.storage.count_memories_by_status("consolidated") == 3

    def test_block_false_not_yet_implemented(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        with pytest.raises(NotImplementedError, match="T021"):
            memory_system.force_dream(block=False)


# ---------------------------------------------------------------------------
# _run_dream_cycle (T020 Half A — pipeline skeleton, no clustering)
# ---------------------------------------------------------------------------


class TestRunDreamCycleHappyPath:
    def test_three_memories_consolidate_through_pipeline(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        # MockLLMProvider.extract_tags returns the first 3 alpha tokens
        # of the summary. MockEmbeddingProvider produces deterministic
        # 16-dim float32 vectors.
        ids = [
            memory_system.enqueue("alpha beta gamma"),
            memory_system.enqueue("delta epsilon zeta"),
            memory_system.enqueue("eta theta iota"),
        ]
        memory_system.force_dream()

        # 1. All 3 memories are now consolidated.
        for mid in ids:
            row = memory_system.storage.get_memory_by_id(mid)
            assert row is not None
            assert row["status"] == "consolidated"

        # 2. Nine distinct tag nodes (one per unique token) exist and
        #    are leaf tags, not centroids (clustering lands in Half B).
        nodes = memory_system.storage.get_all_nodes()
        labels = {n["label"] for n in nodes}
        expected = {
            "alpha",
            "beta",
            "gamma",
            "delta",
            "epsilon",
            "zeta",
            "eta",
            "theta",
            "iota",
        }
        assert expected <= labels
        for node in nodes:
            assert node["is_centroid"] is False

        # 3. has_tag edges wired from each memory to its 3 tags.
        #    get_subgraph traverses from a node and collects reachable
        #    memories — use alpha's node as the root.
        alpha_id = next(n["id"] for n in nodes if n["label"] == "alpha")
        sub = memory_system.storage.get_subgraph([alpha_id], depth=1)
        reachable_memory_ids = {m["id"] for m in sub["memories"]}
        assert ids[0] in reachable_memory_ids

    def test_empty_inbox_is_no_op(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        """_run_dream_cycle on empty inbox must not raise."""
        memory_system.force_dream()
        assert memory_system.storage.count_memories_by_status("inbox") == 0
        assert memory_system.storage.count_memories_by_status("consolidated") == 0
        assert memory_system.is_dreaming is False

    def test_is_dreaming_false_after_cycle(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        memory_system.enqueue("hello")
        memory_system.force_dream()
        assert memory_system.is_dreaming is False

    def test_inbox_emptied_after_cycle(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        for i in range(5):
            memory_system.enqueue(f"raw {i}")
        assert memory_system.storage.count_memories_by_status("inbox") == 5
        memory_system.force_dream()
        assert memory_system.storage.count_memories_by_status("inbox") == 0


class TestRunDreamCycleRollback:
    def test_extract_tags_exception_rolls_back_batch(
        self,
        mock_embedder: MockEmbeddingProvider,
    ) -> None:
        class BrokenLLM(LLMProvider):
            def generate_summary(self, raw_text: str) -> str:
                return raw_text[:40]

            def extract_tags(self, summary: str) -> list[str]:
                raise RuntimeError("provider boom")

            def generate_category_name(self, concepts: list[str]) -> str:
                return "Misc"

        system = NeuroMemory(
            storage=SQLiteAdapter(":memory:"),
            llm=BrokenLLM(),
            embedder=mock_embedder,
        )
        mem_ids = [system.enqueue(f"raw {i}") for i in range(3)]

        with pytest.raises(RuntimeError, match="provider boom"):
            system.force_dream()

        # All 3 memories should be BACK in inbox, not stuck in dreaming.
        assert system.storage.count_memories_by_status("inbox") == 3
        assert system.storage.count_memories_by_status("dreaming") == 0
        assert system.storage.count_memories_by_status("consolidated") == 0
        for mid in mem_ids:
            assert system.storage.get_memory_by_id(mid)["status"] == "inbox"
        # And is_dreaming is cleared even though we raised.
        assert system.is_dreaming is False

    def test_embedder_exception_rolls_back_batch(
        self,
        mock_llm: MockLLMProvider,
    ) -> None:
        class BrokenEmbedder(EmbeddingProvider):
            def get_embeddings(self, texts):
                raise RuntimeError("embed boom")

        system = NeuroMemory(
            storage=SQLiteAdapter(":memory:"),
            llm=mock_llm,
            embedder=BrokenEmbedder(),
        )
        mem_id = system.enqueue("hello there")

        with pytest.raises(RuntimeError, match="embed boom"):
            system.force_dream()

        assert system.storage.get_memory_by_id(mem_id)["status"] == "inbox"
        assert system.is_dreaming is False


class TestRunDreamCycleLocking:
    def test_held_lock_causes_immediate_return(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        """Validates the lock-skip branch of ``_run_dream_cycle``.

        Acquires ``_dream_lock`` on the test thread, then calls
        ``_run_dream_cycle`` from the SAME thread. Because
        ``threading.Lock`` is not re-entrant, the non-blocking
        ``acquire(blocking=False)`` inside ``_run_dream_cycle``
        returns ``False`` and the method returns immediately without
        touching any inbox memories.

        Note: this test exercises the lock-skip **logic branch**, not
        actual cross-thread concurrency. A genuine two-thread test
        (spawning two workers hitting ``_run_dream_cycle`` simultaneously
        with a ``threading.Barrier``) is a T021 deliverable and will
        cover the cross-thread exclusion path. See PR #36 review
        finding I-3 for the reasoning behind the split.
        """
        mem_id = memory_system.enqueue("stays in inbox")
        assert memory_system._dream_lock.acquire(blocking=False)
        try:
            memory_system._run_dream_cycle()
            # Memory was NOT processed.
            assert memory_system.storage.get_memory_by_id(mem_id)["status"] == "inbox"
        finally:
            memory_system._dream_lock.release()


class TestEnsureTagNodes:
    def test_reuses_existing_nodes_by_label(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        """Second dream cycle reuses tag nodes created in the first.

        After two dream cycles that both produce the tag 'shared',
        there should be exactly ONE 'shared' node in storage, not two.
        """
        memory_system.enqueue("shared common tokens")
        memory_system.force_dream()
        nodes_after_first = memory_system.storage.get_all_nodes()
        shared_count_1 = sum(1 for n in nodes_after_first if n["label"] == "shared")
        assert shared_count_1 == 1

        memory_system.enqueue("shared another thing")
        memory_system.force_dream()
        nodes_after_second = memory_system.storage.get_all_nodes()
        shared_count_2 = sum(1 for n in nodes_after_second if n["label"] == "shared")
        assert shared_count_2 == 1


# ---------------------------------------------------------------------------
# Agglomerative clustering (T020 Half B)
# ---------------------------------------------------------------------------


class ControlledEmbedder(EmbeddingProvider):
    """Deterministic embedder for clustering tests.

    Returns a fixed 4-dim vector for every known label. Unknown
    labels raise KeyError so bugs in the tag-to-node wiring surface
    loudly rather than silently returning arbitrary vectors.
    """

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = {label: np.asarray(vec, dtype=np.float32) for label, vec in vectors.items()}
        self.dim = next(iter(self._vectors.values())).shape[0]

    def get_embeddings(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vectors[t] for t in texts])


class TestAgglomerativeClustering:
    def test_clustering_creates_centroid_when_close(
        self,
        mock_llm: MockLLMProvider,
    ) -> None:
        """Two tags with cosine ~0.995 merge into a centroid."""
        ctrl = ControlledEmbedder(
            {
                # alpha and beta are nearly identical (cos ≈ 0.995)
                "alpha": [1.0, 0.0, 0.0, 0.0],
                "beta": [0.995, 0.1, 0.0, 0.0],
                # gamma is orthogonal to both
                "gamma": [0.0, 0.0, 1.0, 0.0],
            }
        )
        system = NeuroMemory(
            storage=SQLiteAdapter(":memory:"),
            llm=mock_llm,
            embedder=ctrl,
            cluster_threshold=0.9,
        )
        # MockLLMProvider.extract_tags returns first 3 alpha tokens.
        system.enqueue("alpha beta gamma")
        system.force_dream()

        nodes = system.storage.get_all_nodes()
        leaves = [n for n in nodes if not n["is_centroid"]]
        centroids = [n for n in nodes if n["is_centroid"]]

        # 3 leaves (alpha, beta, gamma) + at least 1 centroid (alpha+beta)
        leaf_labels = {n["label"] for n in leaves}
        assert leaf_labels >= {"alpha", "beta", "gamma"}
        assert len(centroids) >= 1

        # MockLLMProvider.generate_category_name(["alpha","beta"]) → "CatAB"
        # or "CatBA" depending on argmax ordering; both are valid.
        centroid_labels = {n["label"] for n in centroids}
        assert centroid_labels & {"CatAB", "CatBA"}

    def test_clustering_does_not_merge_below_threshold(
        self,
        mock_llm: MockLLMProvider,
    ) -> None:
        """With cluster_threshold=0.99, cos=0.5 pairs are not merged."""
        ctrl = ControlledEmbedder(
            {
                "alpha": [1.0, 0.0, 0.0, 0.0],
                "beta": [0.5, 0.866, 0.0, 0.0],  # cos(alpha, beta) = 0.5
                "gamma": [0.0, 0.5, 0.866, 0.0],
            }
        )
        system = NeuroMemory(
            storage=SQLiteAdapter(":memory:"),
            llm=mock_llm,
            embedder=ctrl,
            cluster_threshold=0.99,
        )
        system.enqueue("alpha beta gamma")
        system.force_dream()

        nodes = system.storage.get_all_nodes()
        centroids = [n for n in nodes if n["is_centroid"]]
        assert len(centroids) == 0

    def test_has_tag_edges_still_point_to_leaves_after_clustering(
        self,
        mock_llm: MockLLMProvider,
    ) -> None:
        """A memory's has_tag edges must land on the LEAF tag nodes,
        never on their centroid parents, so contextual recall can
        still reach the original concept."""
        ctrl = ControlledEmbedder(
            {
                "alpha": [1.0, 0.0, 0.0, 0.0],
                "beta": [0.999, 0.01, 0.0, 0.0],
                "gamma": [0.0, 0.0, 1.0, 0.0],
            }
        )
        system = NeuroMemory(
            storage=SQLiteAdapter(":memory:"),
            llm=mock_llm,
            embedder=ctrl,
            cluster_threshold=0.9,
        )
        mem_id = system.enqueue("alpha beta gamma")
        system.force_dream()

        # Pull alpha's leaf node id and walk the subgraph from it.
        nodes = system.storage.get_all_nodes()
        alpha_leaf = next(n for n in nodes if n["label"] == "alpha" and not n["is_centroid"])
        sub = system.storage.get_subgraph([alpha_leaf["id"]], depth=2)

        # The memory must be reachable via the alpha leaf (has_tag edge).
        reachable = {m["id"] for m in sub["memories"]}
        assert mem_id in reachable

        # Verify directly: the has_tag edge source is the memory, target
        # is the leaf, NOT the centroid.
        has_tag_edges = [e for e in sub["edges"] if e["relationship"] == "has_tag"]
        assert any(
            e["source_id"] == mem_id and e["target_id"] == alpha_leaf["id"] for e in has_tag_edges
        )

    def test_child_of_edges_wired_centroid_to_members(
        self,
        mock_llm: MockLLMProvider,
    ) -> None:
        """After clustering, centroid → member edges exist with
        relationship='child_of' and weight equal to the merge similarity."""
        ctrl = ControlledEmbedder(
            {
                "alpha": [1.0, 0.0, 0.0, 0.0],
                "beta": [0.99, 0.0, 0.0, 0.0],  # cos = 0.99 exactly
            }
        )
        system = NeuroMemory(
            storage=SQLiteAdapter(":memory:"),
            llm=mock_llm,
            embedder=ctrl,
            cluster_threshold=0.9,
        )
        system.enqueue("alpha beta")
        system.force_dream()

        nodes = system.storage.get_all_nodes()
        centroids = [n for n in nodes if n["is_centroid"]]
        assert len(centroids) == 1
        centroid = centroids[0]

        sub = system.storage.get_subgraph([centroid["id"]], depth=1)
        child_of_edges = [
            e
            for e in sub["edges"]
            if e["relationship"] == "child_of" and e["source_id"] == centroid["id"]
        ]
        assert len(child_of_edges) == 2
        for edge in child_of_edges:
            assert edge["weight"] >= 0.9  # above the cluster threshold
            assert edge["weight"] <= 1.0


class TestSanitiseCategoryName:
    def test_single_word_passthrough(self) -> None:
        from neuromem.system import _sanitise_category_name

        assert _sanitise_category_name("Databases") == "Databases"

    def test_multi_word_takes_first(self) -> None:
        from neuromem.system import _sanitise_category_name

        assert _sanitise_category_name("Large Databases") == "Large"

    def test_newline_separated_takes_first(self) -> None:
        from neuromem.system import _sanitise_category_name

        assert _sanitise_category_name("Databases\nStorage") == "Databases"

    def test_empty_returns_fallback(self) -> None:
        from neuromem.system import _sanitise_category_name

        assert _sanitise_category_name("") == "Category"

    def test_whitespace_only_returns_fallback(self) -> None:
        from neuromem.system import _sanitise_category_name

        assert _sanitise_category_name("   ") == "Category"

    def test_leading_trailing_whitespace_stripped(self) -> None:
        from neuromem.system import _sanitise_category_name

        assert _sanitise_category_name("  Databases  ") == "Databases"


def test_explicit_ignore_unused_embedding_provider_import() -> None:
    """Referenced for type-checker compatibility."""
    # EmbeddingProvider is referenced by type annotations in test fixtures.
    _ = EmbeddingProvider
