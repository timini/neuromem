"""Unit tests for neuromem.tools — search_memory + retrieve_memories.

Both functions are tested end-to-end against a live NeuroMemory
with an in-memory SQLiteAdapter. The search_memory tests are
intentionally minimal because the heavy lifting is covered by
test_context.py — search_memory is a thin wrapper over
ContextHelper.build_prompt_context. The retrieve_memories tests
are comprehensive because that's where the LTP spike logic lives.
"""

from __future__ import annotations

import time

import pytest
from neuromem.storage.sqlite import SQLiteAdapter
from neuromem.system import NeuroMemory
from neuromem.tools import expand_node, retrieve_memories, search_memory

from tests.conftest import MockEmbeddingProvider, MockLLMProvider


@pytest.fixture
def memory_system(
    mock_embedder: MockEmbeddingProvider,
    mock_llm: MockLLMProvider,
) -> NeuroMemory:
    return NeuroMemory(
        storage=SQLiteAdapter(":memory:"),
        llm=mock_llm,
        embedder=mock_embedder,
    )


# ---------------------------------------------------------------------------
# search_memory — thin wrapper over ContextHelper
# ---------------------------------------------------------------------------


class TestSearchMemory:
    def test_empty_graph_returns_empty_string(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        assert search_memory("anything", memory_system) == ""

    def test_populated_graph_returns_tree_with_memory_ids(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        for i in range(3):
            memory_system.enqueue(f"python sqlite topic {i}")
        memory_system.force_dream(block=True)

        result = search_memory("python", memory_system)
        assert result != ""
        assert "📁" in result
        assert "📄" in result

    def test_same_output_as_context_helper(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        """search_memory and ContextHelper.build_prompt_context must
        produce identical output for the same inputs."""
        from neuromem.context import ContextHelper

        memory_system.enqueue("alpha beta gamma")
        memory_system.enqueue("delta epsilon")
        memory_system.force_dream(block=True)

        tool_result = search_memory("alpha", memory_system)
        helper_result = ContextHelper(memory_system).build_prompt_context("alpha")
        assert tool_result == helper_result

    def test_top_k_parameter_honoured(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        for i in range(5):
            memory_system.enqueue(f"topic{i} example content")
        memory_system.force_dream(block=True)

        small = search_memory("example", memory_system, top_k=1)
        large = search_memory("example", memory_system, top_k=10)
        assert small != ""
        assert large != ""

    def test_empty_query_raises(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            search_memory("", memory_system)

    def test_invalid_top_k_raises(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        with pytest.raises(ValueError, match="top_k"):
            search_memory("hello", memory_system, top_k=0)

    def test_invalid_depth_raises(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        with pytest.raises(ValueError, match="depth"):
            search_memory("hello", memory_system, depth=-1)


# ---------------------------------------------------------------------------
# retrieve_memories — fetch + LTP spike
# ---------------------------------------------------------------------------


class TestRetrieveMemories:
    def test_empty_list_returns_empty_list(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        assert retrieve_memories([], memory_system) == []

    def test_single_consolidated_memory_round_trip(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        mem_id = memory_system.enqueue("hello world")
        memory_system.force_dream(block=True)

        results = retrieve_memories([mem_id], memory_system)
        assert len(results) == 1
        row = results[0]
        assert row["id"] == mem_id
        assert row["raw_content"] == "hello world"
        assert row["status"] == "consolidated"

    def test_missing_id_silently_skipped(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        """Missing IDs must not raise and must not appear in results."""
        mem_id = memory_system.enqueue("real memory")
        memory_system.force_dream(block=True)

        results = retrieve_memories(
            [mem_id, "nonexistent-id", "also-nonexistent"],
            memory_system,
        )
        assert len(results) == 1
        assert results[0]["id"] == mem_id

    def test_all_missing_ids_returns_empty_list(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        results = retrieve_memories(
            ["ghost-1", "ghost-2", "ghost-3"],
            memory_system,
        )
        assert results == []

    def test_multiple_consolidated_memories_returned_in_order(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        ids = [memory_system.enqueue(f"memory {i}") for i in range(3)]
        memory_system.force_dream(block=True)

        # Retrieve in a specific order and verify the result order matches.
        results = retrieve_memories([ids[1], ids[0], ids[2]], memory_system)
        assert len(results) == 3
        returned_ids = [r["id"] for r in results]
        assert returned_ids == [ids[1], ids[0], ids[2]]


class TestRetrieveMemoriesLTP:
    """Long-Term Potentiation: access_weight spike on retrieval."""

    def test_consolidated_memory_gets_ltp_spike(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        mem_id = memory_system.enqueue("will be spiked")
        memory_system.force_dream(block=True)

        # Artificially lower access_weight to simulate prior decay.
        memory_system.storage._conn.execute(
            "UPDATE memories SET access_weight = 0.3, last_accessed = 0 WHERE id = ?",
            (mem_id,),
        )

        # Use int(time.time()) to match retrieve_memories' precision —
        # float comparison would fail because int() truncates downward
        # and the truncated `now` can be < the float `before`.
        before = int(time.time())
        results = retrieve_memories([mem_id], memory_system)
        after = int(time.time())

        assert len(results) == 1
        # The returned dict must reflect the POST-spike values.
        assert results[0]["access_weight"] == pytest.approx(1.0)
        assert before <= results[0]["last_accessed"] <= after

        # And the database row must also be updated.
        stored = memory_system.storage.get_memory_by_id(mem_id)
        assert stored["access_weight"] == pytest.approx(1.0)
        assert before <= stored["last_accessed"] <= after

    def test_inbox_memory_returned_but_not_spiked(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        """Inbox (not yet consolidated) memories are returned as-is —
        LTP is reserved for consolidated memories per spec V-M4."""
        mem_id = memory_system.enqueue("stays in inbox")
        # No force_dream — the memory is in inbox status.

        results = retrieve_memories([mem_id], memory_system)
        assert len(results) == 1
        row = results[0]
        assert row["status"] == "inbox"
        # last_accessed stays None (not touched by LTP).
        assert row["last_accessed"] is None
        # The database row should also be untouched.
        stored = memory_system.storage.get_memory_by_id(mem_id)
        assert stored["last_accessed"] is None

    def test_archived_memory_returned_but_not_resurrected(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        """Archived memories are returned but NOT flipped back to
        consolidated (per User Story 3 acceptance #2) and do not
        receive LTP."""
        mem_id = memory_system.enqueue("will be archived")
        memory_system.force_dream(block=True)

        # Force archival by directly flipping the status.
        memory_system.storage._conn.execute(
            "UPDATE memories SET status = 'archived', access_weight = 0.05 WHERE id = ?",
            (mem_id,),
        )

        results = retrieve_memories([mem_id], memory_system)
        assert len(results) == 1
        row = results[0]
        assert row["status"] == "archived"  # NOT resurrected
        assert row["access_weight"] == pytest.approx(0.05)  # NOT spiked

        # Database row should also stay archived.
        stored = memory_system.storage.get_memory_by_id(mem_id)
        assert stored["status"] == "archived"
        assert stored["access_weight"] == pytest.approx(0.05)

    def test_batched_spike_covers_all_consolidated(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        """A single retrieve_memories call spikes every consolidated
        memory in the batch via ONE storage.spike_access_weight call."""
        ids = [memory_system.enqueue(f"memory {i}") for i in range(4)]
        memory_system.force_dream(block=True)

        # Lower all access_weights to simulate decay.
        memory_system.storage._conn.execute(
            "UPDATE memories SET access_weight = 0.4 WHERE status = 'consolidated'"
        )

        results = retrieve_memories(ids, memory_system)
        assert len(results) == 4
        # Every returned memory has access_weight back to 1.0.
        for row in results:
            assert row["access_weight"] == pytest.approx(1.0)
            assert row["last_accessed"] is not None

    def test_mixed_status_batch_only_spikes_consolidated(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        """A batch containing consolidated + inbox memories only spikes
        the consolidated ones."""
        consolidated_id = memory_system.enqueue("will be consolidated")
        memory_system.force_dream(block=True)
        inbox_id = memory_system.enqueue("stays in inbox")

        # Drop consolidated's weight to make the spike observable.
        memory_system.storage._conn.execute(
            "UPDATE memories SET access_weight = 0.2 WHERE id = ?",
            (consolidated_id,),
        )

        results = retrieve_memories([consolidated_id, inbox_id], memory_system)
        assert len(results) == 2

        by_id = {r["id"]: r for r in results}
        # Consolidated got spiked.
        assert by_id[consolidated_id]["access_weight"] == pytest.approx(1.0)
        assert by_id[consolidated_id]["last_accessed"] is not None
        # Inbox did NOT get spiked.
        assert by_id[inbox_id]["status"] == "inbox"
        assert by_id[inbox_id]["last_accessed"] is None

    def test_retrieve_preserves_metadata(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        mem_id = memory_system.enqueue(
            "hello",
            metadata={"role": "user", "turn": 42},
        )
        memory_system.force_dream(block=True)

        results = retrieve_memories([mem_id], memory_system)
        assert results[0]["metadata"] == {"role": "user", "turn": 42}

    def test_full_memory_dict_shape(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        """Every returned dict has the 8 keys documented in the contract."""
        mem_id = memory_system.enqueue("shape test", metadata={"x": 1})
        memory_system.force_dream(block=True)

        results = retrieve_memories([mem_id], memory_system)
        assert len(results) == 1
        row = results[0]

        # Per contracts/public-api.md §retrieve_memories
        expected_keys = {
            "id",
            "raw_content",
            "summary",
            "status",
            "access_weight",
            "created_at",
            "last_accessed",
            "metadata",
        }
        assert expected_keys.issubset(row.keys())


class TestRetrieveMemoriesDoesNotLeakBetweenCalls:
    def test_repeated_retrieval_is_idempotent(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        """Calling retrieve_memories twice with the same IDs returns
        the same data (the second call re-spikes, which is a no-op
        on an already-spiked memory)."""
        mem_id = memory_system.enqueue("idempotent")
        memory_system.force_dream(block=True)

        first = retrieve_memories([mem_id], memory_system)
        second = retrieve_memories([mem_id], memory_system)

        assert len(first) == 1
        assert len(second) == 1
        assert first[0]["id"] == second[0]["id"]
        assert first[0]["raw_content"] == second[0]["raw_content"]
        # Both calls spike to 1.0, so both see the spiked value.
        assert first[0]["access_weight"] == pytest.approx(1.0)
        assert second[0]["access_weight"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# expand_node — ADR-003 F6 agent "zoom in" tool
# ---------------------------------------------------------------------------


class TestExpandNode:
    """expand_node renders a subtree rooted at a given node id,
    resolving centroid names and paragraph summaries lazily. It
    is NOT LTP-spiking — browsing the ontology is a different action
    from recalling a specific memory.
    """

    def test_empty_node_id_raises(self, memory_system: NeuroMemory) -> None:
        with pytest.raises(ValueError, match="node_id must be non-empty"):
            expand_node("", memory_system)

    def test_negative_depth_raises(self, memory_system: NeuroMemory) -> None:
        with pytest.raises(ValueError, match="depth must be >= 0"):
            expand_node("anything", memory_system, depth=-1)

    def test_unknown_id_returns_empty_string(self, memory_system: NeuroMemory) -> None:
        assert expand_node("node_does_not_exist", memory_system) == ""

    def test_renders_subtree(self, memory_system: NeuroMemory) -> None:
        """After some memories are ingested + consolidated, expand_node
        on any existing tag or centroid returns a non-empty tree."""
        memory_system.enqueue("alpha beta gamma")
        memory_system.force_dream(block=True)

        nodes = memory_system.storage.get_all_nodes()
        # Pick any node that actually has descendants — a centroid
        # preferred, else the first leaf.
        centroids = [n for n in nodes if n["is_centroid"]]
        target_id = centroids[0]["id"] if centroids else nodes[0]["id"]

        tree = expand_node(target_id, memory_system, depth=2)
        assert tree, "expand_node on an existing node should return non-empty tree"
        assert "Relevant Memory Context" in tree

    def test_does_not_spike_memory_access_weight(self, memory_system: NeuroMemory) -> None:
        """LTP only fires on retrieve_memories, not on browsing the
        ontology via expand_node."""
        mem_id = memory_system.enqueue("alpha beta gamma")
        memory_system.force_dream(block=True)
        # Apply decay so access_weight drops below 1.0.
        nodes = memory_system.storage.get_all_nodes()
        target_id = next((n["id"] for n in nodes if n["is_centroid"]), nodes[0]["id"])
        # Fast-forward time and decay so the memory's access_weight is
        # measurably below 1.0. A week of decay at default lambda is
        # enough to move the needle.
        future = int(time.time()) + 7 * 24 * 3600
        memory_system.storage.apply_decay_and_archive(
            decay_lambda=memory_system.decay_lambda,
            archive_threshold=0.0,  # never archive during this test
            current_timestamp=future,
        )
        before = memory_system.storage.get_memory_by_id(mem_id)
        assert before is not None
        weight_before = float(before["access_weight"])
        assert weight_before < 1.0

        expand_node(target_id, memory_system, depth=2)
        after = memory_system.storage.get_memory_by_id(mem_id)
        assert after is not None
        assert float(after["access_weight"]) == weight_before, (
            "expand_node must NOT spike memory access_weight (ADR-003 F6)"
        )
