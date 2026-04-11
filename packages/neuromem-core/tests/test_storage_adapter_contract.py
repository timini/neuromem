"""Contract tests for StorageAdapter implementations.

These 13 test functions exercise the full StorageAdapter contract
against every concrete adapter registered in
``conftest.STORAGE_ADAPTER_FACTORIES``. The ``storage_adapter``
fixture is parametrised over that list, so each test runs once per
registered adapter in a single pytest session.

The contract itself is defined in
``specs/001-neuromem-core/contracts/storage-adapter.md §Contract tests``.
Test numbers below match the numbered list there.

State machine reference (data-model.md §State Machine):
    inbox → dreaming → consolidated → archived
      ^                      |
      └── (rollback on error)

Task dependencies:
  - T013-T018 register SQLiteAdapter in conftest.STORAGE_ADAPTER_FACTORIES
    as they land; each task makes progressively more tests pass.
  - T025 registers DictStorageAdapter and the whole suite runs again
    against it, proving Principle III (adapter swap with zero
    orchestration-layer changes).
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from neuromem.storage.base import StorageAdapter

# ---------------------------------------------------------------------------
# 1. Construction
# ---------------------------------------------------------------------------


def test_01_construction(storage_adapter: StorageAdapter) -> None:
    """Construction: adapter schema initialises without raising."""
    # The fixture itself exercises __init__. If we got here, that's
    # the test. Just sanity-check we got a StorageAdapter instance.
    assert isinstance(storage_adapter, StorageAdapter)


# ---------------------------------------------------------------------------
# 2. insert_memory round-trip
# ---------------------------------------------------------------------------


def test_02_insert_memory_round_trip(storage_adapter: StorageAdapter) -> None:
    memory_id = storage_adapter.insert_memory(
        raw_content="hello world",
        summary="hello summary",
        metadata={"role": "user"},
    )
    fetched = storage_adapter.get_memory_by_id(memory_id)
    assert fetched is not None
    assert fetched["id"] == memory_id
    assert fetched["raw_content"] == "hello world"
    assert fetched["summary"] == "hello summary"
    assert fetched["status"] == "inbox"
    assert fetched["access_weight"] == pytest.approx(1.0)
    assert fetched["metadata"] == {"role": "user"}


# ---------------------------------------------------------------------------
# 3. Status state machine round-trip
# ---------------------------------------------------------------------------


def test_03_status_state_machine(storage_adapter: StorageAdapter) -> None:
    memory_id = storage_adapter.insert_memory("raw", "summary")
    assert storage_adapter.get_memory_by_id(memory_id)["status"] == "inbox"

    storage_adapter.update_memory_status([memory_id], "dreaming")
    assert storage_adapter.get_memory_by_id(memory_id)["status"] == "dreaming"

    storage_adapter.update_memory_status([memory_id], "consolidated")
    assert storage_adapter.get_memory_by_id(memory_id)["status"] == "consolidated"


# ---------------------------------------------------------------------------
# 4. count_memories_by_status
# ---------------------------------------------------------------------------


def test_04_count_memories_by_status(storage_adapter: StorageAdapter) -> None:
    assert storage_adapter.count_memories_by_status("inbox") == 0
    ids = [storage_adapter.insert_memory(f"r{i}", f"s{i}") for i in range(3)]
    assert storage_adapter.count_memories_by_status("inbox") == 3
    assert storage_adapter.count_memories_by_status("consolidated") == 0

    storage_adapter.update_memory_status(ids[:2], "dreaming")
    assert storage_adapter.count_memories_by_status("inbox") == 1
    assert storage_adapter.count_memories_by_status("dreaming") == 2


# ---------------------------------------------------------------------------
# 5. upsert_node round-trip
# ---------------------------------------------------------------------------


def test_05_upsert_node_round_trip(storage_adapter: StorageAdapter) -> None:
    embedding = np.arange(16, dtype=np.float32)
    storage_adapter.upsert_node("node_1", "Python", embedding, is_centroid=False)

    nodes = storage_adapter.get_all_nodes()
    assert len(nodes) == 1
    assert nodes[0]["id"] == "node_1"
    assert nodes[0]["label"] == "Python"
    assert nodes[0]["is_centroid"] is False
    np.testing.assert_allclose(nodes[0]["embedding"], embedding)


# ---------------------------------------------------------------------------
# 6. Dimension mismatch rejection
# ---------------------------------------------------------------------------


def test_06_embedding_dimension_mismatch(
    storage_adapter: StorageAdapter,
) -> None:
    storage_adapter.upsert_node("n16", "First", np.zeros(16, dtype=np.float32), False)
    with pytest.raises(ValueError):
        storage_adapter.upsert_node("n32", "Second", np.zeros(32, dtype=np.float32), False)


# ---------------------------------------------------------------------------
# 7. Edge insert idempotency
# ---------------------------------------------------------------------------


def test_07_edge_insert_idempotent(storage_adapter: StorageAdapter) -> None:
    emb = np.zeros(16, dtype=np.float32)
    storage_adapter.upsert_node("a", "A", emb, False)
    storage_adapter.upsert_node("b", "B", emb, False)

    storage_adapter.insert_edge("a", "b", 0.9, "child_of")
    # Second insert with same triple must NOT raise; must NOT duplicate.
    storage_adapter.insert_edge("a", "b", 0.9, "child_of")

    subgraph = storage_adapter.get_subgraph(["a"], depth=1)
    edges = [
        e
        for e in subgraph["edges"]
        if e["relationship"] == "child_of" and e["source_id"] == "a" and e["target_id"] == "b"
    ]
    assert len(edges) == 1


# ---------------------------------------------------------------------------
# 7b. remove_edges_for_memory preserves child_of edges
# ---------------------------------------------------------------------------


def test_07b_remove_edges_preserves_child_of(
    storage_adapter: StorageAdapter,
) -> None:
    """``remove_edges_for_memory`` must only remove ``has_tag`` edges.

    The docstring on ``StorageAdapter.remove_edges_for_memory``
    explicitly says: *"Does NOT remove child_of edges."* A concrete
    adapter that naively deletes all edges from a memory would pass
    the other 13 contract tests while silently corrupting the concept
    graph during the dreaming cycle's archival step.

    This test sets up the exact topology the dreaming cycle produces
    (``memory --has_tag--> tag <--child_of-- centroid``), archives
    the memory, and asserts the ``child_of`` edge between the tag and
    its centroid parent is still there.
    """
    emb = np.zeros(16, dtype=np.float32)
    mem_id = storage_adapter.insert_memory("raw", "summary")

    storage_adapter.upsert_node("tag_sqlite", "SQLite", emb, is_centroid=False)
    storage_adapter.upsert_node("centroid_db", "Databases", emb, is_centroid=True)

    # memory --has_tag--> tag
    storage_adapter.insert_edge(mem_id, "tag_sqlite", 1.0, "has_tag")
    # centroid --child_of--> tag
    storage_adapter.insert_edge("centroid_db", "tag_sqlite", 0.9, "child_of")

    # Pre-condition sanity: both edges exist.
    pre = storage_adapter.get_subgraph(["centroid_db"], depth=2)
    pre_keys = {(e["source_id"], e["target_id"], e["relationship"]) for e in pre["edges"]}
    assert (mem_id, "tag_sqlite", "has_tag") in pre_keys
    assert ("centroid_db", "tag_sqlite", "child_of") in pre_keys

    # Action under test.
    storage_adapter.remove_edges_for_memory(mem_id)

    # Post-condition: child_of is preserved, has_tag is gone.
    post = storage_adapter.get_subgraph(["centroid_db"], depth=2)
    post_keys = {(e["source_id"], e["target_id"], e["relationship"]) for e in post["edges"]}
    assert ("centroid_db", "tag_sqlite", "child_of") in post_keys, (
        "remove_edges_for_memory must NOT delete child_of edges"
    )
    assert (mem_id, "tag_sqlite", "has_tag") not in post_keys, (
        "remove_edges_for_memory must delete the memory's has_tag edge"
    )


# ---------------------------------------------------------------------------
# 8. get_nearest_nodes ranking
# ---------------------------------------------------------------------------


def test_08_get_nearest_nodes_ranking(storage_adapter: StorageAdapter) -> None:
    # Insert 5 nodes. The first should be the closest to the query.
    base = np.zeros(16, dtype=np.float32)
    base[0] = 1.0
    storage_adapter.upsert_node("closest", "Exact", base.copy(), False)

    off1 = base.copy()
    off1[1] = 0.5
    storage_adapter.upsert_node("near", "Near", off1, False)

    off2 = np.zeros(16, dtype=np.float32)
    off2[5] = 1.0
    storage_adapter.upsert_node("far", "Far", off2, False)

    results = storage_adapter.get_nearest_nodes(base, top_k=3)
    assert len(results) == 3
    assert results[0]["id"] == "closest"
    assert "similarity" in results[0]
    assert results[0]["similarity"] >= results[1]["similarity"] >= results[2]["similarity"]


# ---------------------------------------------------------------------------
# 9. get_subgraph includes memories
# ---------------------------------------------------------------------------


def test_09_get_subgraph(storage_adapter: StorageAdapter) -> None:
    emb = np.zeros(16, dtype=np.float32)
    # centroid → child; memory → child (has_tag)
    storage_adapter.upsert_node("centroid", "Databases", emb, is_centroid=True)
    storage_adapter.upsert_node("child", "SQLite", emb, is_centroid=False)
    storage_adapter.insert_edge("centroid", "child", 0.9, "child_of")

    mem_id = storage_adapter.insert_memory("Memory about SQLite", "SQLite summary")
    storage_adapter.update_memory_status([mem_id], "consolidated")
    storage_adapter.insert_edge(mem_id, "child", 1.0, "has_tag")

    subgraph = storage_adapter.get_subgraph(["centroid"], depth=2)
    node_ids = {n["id"] for n in subgraph["nodes"]}
    assert "centroid" in node_ids
    assert "child" in node_ids
    memory_ids = {m["id"] for m in subgraph["memories"]}
    assert mem_id in memory_ids


# ---------------------------------------------------------------------------
# 10. Decay + archive
# ---------------------------------------------------------------------------


def test_10_decay_and_archive(storage_adapter: StorageAdapter) -> None:
    mem_id = storage_adapter.insert_memory("old", "old summary")
    storage_adapter.update_memory_status([mem_id], "consolidated")
    # Spike to set last_accessed, then simulate old access by running
    # decay with a future timestamp far past last_accessed.
    now = int(time.time())
    storage_adapter.spike_access_weight([mem_id], now)
    future = now + 60 * 86400  # 60 days later
    # Aggressive decay lambda so the memory is archived in one pass.
    archived = storage_adapter.apply_decay_and_archive(
        decay_lambda=1e-5, archive_threshold=0.1, current_timestamp=future
    )
    assert mem_id in archived
    assert storage_adapter.get_memory_by_id(mem_id)["status"] == "archived"


# ---------------------------------------------------------------------------
# 11. LTP ignores non-consolidated memories
# ---------------------------------------------------------------------------


def test_11_ltp_ignores_non_consolidated(
    storage_adapter: StorageAdapter,
) -> None:
    mem_id = storage_adapter.insert_memory("raw", "summary")
    # Still in 'inbox' state. Spiking should be a no-op — access_weight
    # stays at its default 1.0 (and last_accessed should remain NULL).
    storage_adapter.spike_access_weight([mem_id], int(time.time()))
    mem = storage_adapter.get_memory_by_id(mem_id)
    assert mem["status"] == "inbox"
    assert mem["access_weight"] == pytest.approx(1.0)
    assert mem["last_accessed"] is None


# ---------------------------------------------------------------------------
# 12. Archival preserves content
# ---------------------------------------------------------------------------


def test_12_archival_preserves_content(storage_adapter: StorageAdapter) -> None:
    mem_id = storage_adapter.insert_memory("important raw", "important summary")
    storage_adapter.update_memory_status([mem_id], "consolidated")
    now = int(time.time())
    storage_adapter.spike_access_weight([mem_id], now)
    archived = storage_adapter.apply_decay_and_archive(
        decay_lambda=1e-4,
        archive_threshold=0.1,
        current_timestamp=now + 365 * 86400,  # one year later
    )
    assert mem_id in archived
    fetched = storage_adapter.get_memory_by_id(mem_id)
    assert fetched is not None
    assert fetched["raw_content"] == "important raw"
    assert fetched["summary"] == "important summary"
    assert fetched["status"] == "archived"


# ---------------------------------------------------------------------------
# 13. Missing ID silent skip
# ---------------------------------------------------------------------------


def test_13_missing_id_silent_skip(storage_adapter: StorageAdapter) -> None:
    assert storage_adapter.get_memory_by_id("nonexistent-id") is None
    # spike_access_weight on a missing ID must not raise.
    storage_adapter.spike_access_weight(["missing-1", "missing-2"], int(time.time()))
