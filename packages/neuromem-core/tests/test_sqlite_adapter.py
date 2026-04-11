"""SQLiteAdapter-specific regression tests.

The 14 parametrised contract tests in ``test_storage_adapter_contract.py``
cover every behavioural requirement of the ``StorageAdapter`` ABC. This
file holds tests for bugs and edge cases that are specific to the SQLite
implementation (thread-of-origin handling, blob corruption detection,
etc.) — things that wouldn't necessarily apply to a dict-backed or
postgres-backed adapter.

Each test here references the PR #36 review finding it regresses:
  - B-1: check_same_thread=False missing from sqlite3.connect
  - B-2: blob length corruption check used integer division
"""

from __future__ import annotations

import sqlite3
import threading
import uuid

import numpy as np
import pytest
from neuromem.storage.base import StorageError
from neuromem.storage.sqlite import SQLiteAdapter

# ---------------------------------------------------------------------------
# B-1 regression: cross-thread connection use
# ---------------------------------------------------------------------------


class TestCrossThreadAccess:
    """Regression for PR #36 review finding B-1.

    Before the fix, SQLiteAdapter.__init__ called sqlite3.connect
    without check_same_thread=False. The connection defaults to
    check_same_thread=True, which makes Python's sqlite3 module
    raise ProgrammingError on any cross-thread call. This broke
    the moment T021 added background-thread dreaming.
    """

    def test_insert_memory_from_background_thread_succeeds(self) -> None:
        """The whole point of the fix: spawning a thread and calling
        the adapter from it must not raise ProgrammingError."""
        adapter = SQLiteAdapter(":memory:")

        errors: list[Exception] = []
        memory_ids: list[str] = []

        def worker() -> None:
            try:
                mem_id = adapter.insert_memory(
                    raw_content="from background thread",
                    summary="bg",
                )
                memory_ids.append(mem_id)
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=2.0)

        assert not thread.is_alive(), "worker thread hung"
        assert errors == [], f"worker raised: {errors}"
        assert len(memory_ids) == 1

        # And the main thread can still see the written row.
        fetched = adapter.get_memory_by_id(memory_ids[0])
        assert fetched is not None
        assert fetched["raw_content"] == "from background thread"

    def test_full_method_surface_works_from_background_thread(self) -> None:
        """Exercise every method group from a non-origin thread.

        If check_same_thread were True, ANY of these would raise
        sqlite3.ProgrammingError. This test is a belt-and-braces
        smoke check that the entire adapter surface is cross-thread
        safe.
        """
        adapter = SQLiteAdapter(":memory:")
        errors: list[Exception] = []

        def worker() -> None:
            try:
                # Acquisition
                mem_id = adapter.insert_memory("raw", "summary")
                assert adapter.count_memories_by_status("inbox") == 1
                adapter.update_memory_status([mem_id], "consolidated")
                assert adapter.get_memory_by_id(mem_id)["status"] == "consolidated"

                # Consolidation
                node_id = str(uuid.uuid4())
                adapter.upsert_node(
                    node_id=node_id,
                    label="Python",
                    embedding=np.zeros(16, dtype=np.float32),
                    is_centroid=False,
                )
                assert len(adapter.get_all_nodes()) == 1
                adapter.insert_edge(mem_id, node_id, 1.0, "has_tag")

                # Recall
                nearest = adapter.get_nearest_nodes(np.zeros(16, dtype=np.float32), top_k=1)
                assert len(nearest) == 1

                # Forgetting
                adapter.spike_access_weight([mem_id], 12345)
                adapter.apply_decay_and_archive(1e-9, 0.01, 99999)
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=2.0)

        assert not thread.is_alive(), "worker thread hung"
        assert errors == [], f"worker raised: {errors}"


# ---------------------------------------------------------------------------
# B-2 regression: exact-length blob corruption check
# ---------------------------------------------------------------------------


class TestBlobCorruptionDetection:
    """Regression for PR #36 review finding B-2.

    Before the fix, the blob length check used integer division
    (``len(blob) // 4 != expected_dim``), which rounds down and
    misses corruption where the blob length is
    ``expected_dim * 4 + r`` for r ∈ {1, 2, 3}. The exact check
    (``!= expected_dim * 4``) now catches every byte of drift.
    """

    def _corrupt_blob(
        self,
        adapter: SQLiteAdapter,
        node_id: str,
        new_blob: bytes,
    ) -> None:
        """Directly overwrite the embedding BLOB to simulate on-disk corruption."""
        adapter._conn.execute(
            "UPDATE nodes SET embedding = ? WHERE id = ?",
            (new_blob, node_id),
        )

    @pytest.mark.parametrize("extra_bytes", [1, 2, 3])
    def test_trailing_garbage_below_4_bytes_is_detected(self, extra_bytes: int) -> None:
        """1, 2, or 3 trailing garbage bytes must raise StorageError.

        This is the exact case the old ``// 4`` check missed.
        """
        adapter = SQLiteAdapter(":memory:")
        dim = 16
        node_id = str(uuid.uuid4())
        adapter.upsert_node(
            node_id=node_id,
            label="Python",
            embedding=np.zeros(dim, dtype=np.float32),
            is_centroid=False,
        )

        # Original blob is dim * 4 = 64 bytes. Append garbage.
        corrupt = b"\x00" * (dim * 4) + b"\x00" * extra_bytes
        assert len(corrupt) == dim * 4 + extra_bytes
        self._corrupt_blob(adapter, node_id, corrupt)

        with pytest.raises(StorageError, match="corrupt embedding"):
            adapter.get_all_nodes()

    def test_truncated_blob_detected(self) -> None:
        """A blob shorter than expected_dim * 4 is also caught."""
        adapter = SQLiteAdapter(":memory:")
        dim = 16
        node_id = str(uuid.uuid4())
        adapter.upsert_node(
            node_id=node_id,
            label="Python",
            embedding=np.zeros(dim, dtype=np.float32),
            is_centroid=False,
        )

        # Too short: 62 bytes instead of 64.
        corrupt = b"\x00" * (dim * 4 - 2)
        self._corrupt_blob(adapter, node_id, corrupt)

        with pytest.raises(StorageError, match="corrupt embedding"):
            adapter.get_all_nodes()

    def test_exact_length_blob_passes(self) -> None:
        """Sanity: a correctly-sized blob must NOT raise."""
        adapter = SQLiteAdapter(":memory:")
        dim = 16
        adapter.upsert_node(
            node_id=str(uuid.uuid4()),
            label="Python",
            embedding=np.arange(dim, dtype=np.float32),
            is_centroid=False,
        )
        nodes = adapter.get_all_nodes()
        assert len(nodes) == 1
        assert nodes[0]["embedding"].shape == (dim,)

    def test_error_message_includes_node_id_and_byte_counts(self) -> None:
        """The error message should be actionable for debugging."""
        adapter = SQLiteAdapter(":memory:")
        dim = 8
        node_id = "node_abc"
        adapter.upsert_node(
            node_id=node_id,
            label="test",
            embedding=np.zeros(dim, dtype=np.float32),
            is_centroid=False,
        )
        self._corrupt_blob(adapter, node_id, b"\x00" * (dim * 4 + 1))

        with pytest.raises(StorageError) as exc_info:
            adapter.get_all_nodes()

        msg = str(exc_info.value)
        assert node_id in msg
        assert str(dim * 4 + 1) in msg  # actual byte count
        assert "32" in msg  # expected byte count (8 * 4)


# Silence unused-import lint rule on sqlite3 — it's imported above for
# the type context but only used transitively through SQLiteAdapter.
_ = sqlite3
