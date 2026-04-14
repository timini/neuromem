"""SQLite implementation of the ``StorageAdapter`` contract.

Default storage backend for ``neuromem-core``. Uses only the Python
stdlib ``sqlite3`` module plus numpy for embedding (de)serialisation.
No vector extension required.

Schema (canonical, from specs/001-neuromem-core/data-model.md):

    CREATE TABLE memories (
        id           TEXT PRIMARY KEY,
        raw_content  TEXT NOT NULL,
        summary      TEXT,
        status       TEXT NOT NULL DEFAULT 'inbox'
                      CHECK (status IN ('inbox','dreaming','consolidated','archived')),
        access_weight REAL NOT NULL DEFAULT 1.0,
        last_accessed INTEGER,
        created_at    INTEGER NOT NULL,
        metadata      TEXT
    );

    CREATE TABLE nodes (
        id            TEXT PRIMARY KEY,
        label         TEXT NOT NULL,
        embedding     BLOB NOT NULL,       -- raw float32 bytes
        embedding_dim INTEGER NOT NULL,    -- corruption check
        is_centroid   INTEGER NOT NULL DEFAULT 0
    );

    CREATE TABLE edges (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        source_id    TEXT NOT NULL,
        target_id    TEXT NOT NULL,
        weight       REAL NOT NULL DEFAULT 1.0,
        relationship TEXT NOT NULL
                      CHECK (relationship IN ('has_tag','child_of')),
        UNIQUE(source_id, target_id, relationship)
    );

Indexes: idx_memories_status, idx_edges_source, idx_edges_target.

Implementation is incremental: T013 lands __init__, schema, and
basic memory CRUD (insert_memory, get_memory_by_id,
count_memories_by_status). Tasks T014–T018 complete the remaining
abstract methods.
"""

from __future__ import annotations

import contextlib
import json
import math
import sqlite3
import time
import uuid
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..vectors import batch_cosine_similarity
from .base import StorageAdapter, StorageError

# Allowed status values — enforced both at the ORM level and by a
# CHECK constraint in the DDL.
_VALID_STATUSES = frozenset(("inbox", "dreaming", "consolidated", "archived"))
_VALID_RELATIONSHIPS = frozenset(("has_tag", "child_of"))

# DDL statements, kept as a module-level constant so tests can
# reference them and so initialisation is a straight script.
_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS memories (
    id              TEXT PRIMARY KEY,
    raw_content     TEXT NOT NULL,
    summary         TEXT,
    status          TEXT NOT NULL DEFAULT 'inbox'
                      CHECK (status IN ('inbox','dreaming','consolidated','archived')),
    access_weight   REAL NOT NULL DEFAULT 1.0,
    last_accessed   INTEGER,
    created_at      INTEGER NOT NULL,
    metadata        TEXT,
    named_entities  TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS nodes (
    id                TEXT PRIMARY KEY,
    label             TEXT NOT NULL,
    embedding         BLOB NOT NULL,
    embedding_dim     INTEGER NOT NULL,
    is_centroid       INTEGER NOT NULL DEFAULT 0,
    paragraph_summary TEXT
);

CREATE TABLE IF NOT EXISTS edges (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id    TEXT NOT NULL,
    target_id    TEXT NOT NULL,
    weight       REAL NOT NULL DEFAULT 1.0,
    relationship TEXT NOT NULL
                    CHECK (relationship IN ('has_tag','child_of')),
    UNIQUE(source_id, target_id, relationship)
);

CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status);
CREATE INDEX IF NOT EXISTS idx_edges_source    ON edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target    ON edges(target_id);
"""


# Additive migrations for columns added after the initial v0.1.0 schema.
# Each entry is (table, column, DDL for ADD COLUMN). Running against a
# fresh database is a no-op because CREATE TABLE in _SCHEMA_DDL already
# includes these columns; running against a pre-migration database
# applies the missing column and sets the default for existing rows.
# The OperationalError "duplicate column name" is the only exception we
# silently tolerate — every other error propagates as a real failure.
_ADDITIVE_MIGRATIONS: list[tuple[str, str]] = [
    # (column, full ADD COLUMN statement)
    ("named_entities", "ALTER TABLE memories ADD COLUMN named_entities TEXT NOT NULL DEFAULT '[]'"),
    # ADR-003: per-centroid paragraph summary (2-4 sentences, LLM-generated,
    # cached). NULL means "not yet summarised" — resolve_junction_summaries
    # populates lazily on first render, the dream cycle populates eagerly
    # for the trunk (direct-above-leaf centroids + their parents).
    ("paragraph_summary", "ALTER TABLE nodes ADD COLUMN paragraph_summary TEXT"),
]


def _run_additive_migrations(conn: sqlite3.Connection) -> None:
    """Apply ``ALTER TABLE ADD COLUMN`` statements that were introduced
    after v0.1.0, idempotently.

    SQLite has no ``IF NOT EXISTS`` clause for ``ADD COLUMN``, so we
    attempt each and suppress the specific "duplicate column name"
    OperationalError. Any other error surfaces normally — a corrupt
    database is a real problem, not something to mask.
    """
    for _column, ddl in _ADDITIVE_MIGRATIONS:
        try:
            conn.execute(ddl)
        except sqlite3.OperationalError as exc:
            # The message text varies slightly between SQLite versions,
            # but always includes "duplicate column name" for this case.
            if "duplicate column name" in str(exc).lower():
                continue
            raise


class SQLiteAdapter(StorageAdapter):
    """Stdlib sqlite3-backed ``StorageAdapter``.

    Constructor takes a path to a database file, or ``":memory:"``
    for a transient in-process database (tests use this). The schema
    is created on ``__init__`` if it does not already exist; calling
    the constructor against an existing database file is a no-op on
    the schema.
    """

    def __init__(self, db_path: str) -> None:
        try:
            # isolation_level=None → autocommit; we manage transactions
            # explicitly with `with conn:` blocks.
            # check_same_thread=False lets the background dreaming
            # thread (spawned by T021) touch this connection. The
            # real write serialisation primitive is
            # NeuroMemory._dream_lock at the orchestration layer —
            # check_same_thread would duplicate that lock's purpose
            # and crash on every cross-thread call.
            self._conn = sqlite3.connect(
                db_path,
                isolation_level=None,
                detect_types=sqlite3.PARSE_DECLTYPES,
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
        except sqlite3.Error as exc:
            raise StorageError(f"failed to open SQLite at {db_path!r}: {exc}") from exc

        try:
            # WAL is the sensible default for single-writer concurrent
            # reads. Set before any write.
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=OFF")
            self._conn.executescript(_SCHEMA_DDL)
            # Idempotent migrations for columns added after v0.1.0.
            # Each ADD COLUMN raises OperationalError when the column
            # already exists — we swallow exactly that one case.
            _run_additive_migrations(self._conn)
        except sqlite3.Error as exc:
            raise StorageError(f"schema init failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Acquisition (Hippocampus)
    # ------------------------------------------------------------------

    def insert_memory(
        self,
        raw_content: str,
        summary: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        self._check_open()
        if raw_content is None or raw_content == "":
            raise ValueError("raw_content must be non-empty")

        metadata_json: str | None
        if metadata is None:
            metadata_json = None
        else:
            try:
                metadata_json = json.dumps(metadata)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"metadata is not JSON-serialisable: {exc}") from exc

        memory_id = f"mem_{uuid.uuid4().hex}"
        created_at = int(time.time())
        try:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO memories
                        (id, raw_content, summary, status,
                         access_weight, last_accessed, created_at, metadata)
                    VALUES (?, ?, ?, 'inbox', 1.0, NULL, ?, ?)
                    """,
                    (memory_id, raw_content, summary, created_at, metadata_json),
                )
        except sqlite3.Error as exc:
            raise StorageError(f"insert_memory failed: {exc}") from exc
        return memory_id

    def count_memories_by_status(self, status: str) -> int:
        self._check_open()
        if status not in _VALID_STATUSES:
            raise ValueError(f"invalid status: {status!r}")
        try:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM memories WHERE status = ?",
                (status,),
            ).fetchone()
        except sqlite3.Error as exc:
            raise StorageError(f"count_memories_by_status failed: {exc}") from exc
        return int(row["n"]) if row is not None else 0

    def get_memories_by_status(
        self,
        status: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        self._check_open()
        if status not in _VALID_STATUSES:
            raise ValueError(f"invalid status: {status!r}")
        if limit is not None and limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")

        sql = """
            SELECT id, raw_content, summary, status, access_weight,
                   last_accessed, created_at, metadata, named_entities
            FROM memories
            WHERE status = ?
            ORDER BY created_at ASC, id ASC
        """
        params: tuple[Any, ...] = (status,)
        if limit is not None:
            sql += " LIMIT ?"
            params = (status, limit)

        try:
            rows = self._conn.execute(sql, params).fetchall()
        except sqlite3.Error as exc:
            raise StorageError(f"get_memories_by_status failed: {exc}") from exc
        return [_row_to_memory_dict(row) for row in rows]

    def update_memory_status(
        self,
        memory_ids: list[str],
        new_status: str,
    ) -> None:
        self._check_open()
        if new_status not in _VALID_STATUSES:
            raise ValueError(f"invalid status: {new_status!r}")
        if not memory_ids:
            return  # no-op

        # Single SQL statement for atomicity — this is the double-buffer
        # primitive used by NeuroMemory._run_dream_cycle.
        placeholders = ",".join("?" * len(memory_ids))
        sql = f"UPDATE memories SET status = ? WHERE id IN ({placeholders})"
        try:
            with self._conn:
                self._conn.execute(sql, (new_status, *memory_ids))
        except sqlite3.Error as exc:
            raise StorageError(f"update_memory_status failed: {exc}") from exc

    def get_memory_by_id(self, memory_id: str) -> dict[str, Any] | None:
        self._check_open()
        try:
            row = self._conn.execute(
                """
                SELECT id, raw_content, summary, status, access_weight,
                       last_accessed, created_at, metadata, named_entities
                FROM memories WHERE id = ?
                """,
                (memory_id,),
            ).fetchone()
        except sqlite3.Error as exc:
            raise StorageError(f"get_memory_by_id failed: {exc}") from exc
        if row is None:
            return None
        return _row_to_memory_dict(row)

    def set_named_entities(self, updates: dict[str, list[str]]) -> None:
        """Batch-write named-entity lists onto existing memory rows.

        ``updates`` maps ``memory_id`` → list of entity strings. The
        dream cycle calls this once per cycle with every memory that
        was just consolidated, avoiding per-memory round-trips.

        Contract:
        - Empty ``updates`` → no-op (returns immediately).
        - IDs not present in the ``memories`` table are silently
          skipped (SQLite ``UPDATE`` against a missing id is a zero-
          row no-op — matches ``update_memory_status`` semantics).
        - Each list is JSON-serialised and stored in the
          ``named_entities`` TEXT column. Non-string elements are
          coerced via ``str()`` to stay consistent with
          ``_row_to_memory_dict``'s read-side coercion.
        - Runs all updates in a single transaction.
        """
        self._check_open()
        if not updates:
            return
        try:
            with self._conn:
                for mem_id, entities in updates.items():
                    serialised = json.dumps([str(e) for e in entities])
                    self._conn.execute(
                        "UPDATE memories SET named_entities = ? WHERE id = ?",
                        (serialised, mem_id),
                    )
        except sqlite3.Error as exc:
            raise StorageError(f"set_named_entities failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Consolidation / Recall / Pruning — stubs land in T015–T018.
    # ------------------------------------------------------------------

    def upsert_node(
        self,
        node_id: str,
        label: str,
        embedding: NDArray[np.floating] | list[float],
        is_centroid: bool,
    ) -> None:
        self._check_open()
        arr = np.asarray(embedding, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError(f"embedding must be 1-D, got shape {arr.shape}")
        if arr.size == 0:
            raise ValueError("embedding must be non-empty")

        # Enforce the dimension-match invariant (data-model.md I-N3):
        # the first node's dimension is the lock-in for the whole table.
        existing_dim = self._get_first_node_dim()
        if existing_dim is not None and arr.size != existing_dim:
            raise ValueError(f"embedding dim mismatch: expected {existing_dim}, got {arr.size}")

        # Performance warning above 4096 dims (data-model.md V-N3).
        if arr.size > 4096:
            import logging

            logging.getLogger("neuromem").warning(
                "node %s has embedding dim %d > 4096 — SQLiteAdapter BLOB "
                "storage may become the bottleneck",
                node_id,
                arr.size,
            )

        blob = arr.tobytes()
        try:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO nodes (id, label, embedding, embedding_dim, is_centroid)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        label = excluded.label,
                        embedding = excluded.embedding,
                        embedding_dim = excluded.embedding_dim,
                        is_centroid = excluded.is_centroid,
                        -- ADR-003 D2: a re-upsert signals the centroid's
                        -- membership or embedding changed, so the cached
                        -- paragraph_summary no longer reflects the subtree.
                        -- Clear it so the next render (or dream-cycle
                        -- trunk pass) regenerates a fresh summary.
                        paragraph_summary = NULL
                    """,
                    (node_id, label, blob, int(arr.size), int(is_centroid)),
                )
        except sqlite3.Error as exc:
            raise StorageError(f"upsert_node failed: {exc}") from exc

    def update_node_labels(self, updates: dict[str, str]) -> None:
        """Batch-rename node labels in a single transaction.

        Used by lazy centroid naming (ADR-002) to persist render-time-
        generated labels back to storage. Matches ``set_named_entities``
        semantics: empty dict no-op, missing IDs silently skipped.
        """
        self._check_open()
        if not updates:
            return
        try:
            with self._conn:
                for node_id, label in updates.items():
                    self._conn.execute(
                        "UPDATE nodes SET label = ? WHERE id = ?",
                        (label, node_id),
                    )
        except sqlite3.Error as exc:
            raise StorageError(f"update_node_labels failed: {exc}") from exc

    def update_junction_summaries(self, updates: dict[str, str]) -> None:
        """Batch-write per-centroid paragraph summaries (ADR-003).

        Used both eagerly by the dream cycle's trunk-summarisation step
        and lazily at render time by ``resolve_junction_summaries``.
        Mirrors ``update_node_labels``: empty dict no-op, missing IDs
        silently skipped, single transaction. Overwrites any prior
        summary for the same node (regeneration case).
        """
        self._check_open()
        if not updates:
            return
        try:
            with self._conn:
                for node_id, summary in updates.items():
                    self._conn.execute(
                        "UPDATE nodes SET paragraph_summary = ? WHERE id = ?",
                        (summary, node_id),
                    )
        except sqlite3.Error as exc:
            raise StorageError(f"update_junction_summaries failed: {exc}") from exc

    def get_all_nodes(self) -> list[dict[str, Any]]:
        self._check_open()
        try:
            rows = self._conn.execute(
                "SELECT id, label, embedding, embedding_dim, is_centroid, paragraph_summary FROM nodes ORDER BY id ASC"
            ).fetchall()
        except sqlite3.Error as exc:
            raise StorageError(f"get_all_nodes failed: {exc}") from exc
        return [_row_to_node_dict(row) for row in rows]

    def insert_edge(
        self,
        source_id: str,
        target_id: str,
        weight: float,
        relationship: str,
    ) -> None:
        self._check_open()
        if relationship not in _VALID_RELATIONSHIPS:
            raise ValueError(f"invalid relationship: {relationship!r}")
        try:
            with self._conn:
                # INSERT OR IGNORE makes the call idempotent on the
                # (source, target, relationship) UNIQUE constraint.
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO edges
                        (source_id, target_id, weight, relationship)
                    VALUES (?, ?, ?, ?)
                    """,
                    (source_id, target_id, float(weight), relationship),
                )
        except sqlite3.Error as exc:
            raise StorageError(f"insert_edge failed: {exc}") from exc

    def _delete_has_tag_edges(self, memory_id: str) -> None:
        """Raw DELETE of all ``has_tag`` edges originating from ``memory_id``.

        Private helper shared by ``remove_edges_for_memory`` and
        ``apply_decay_and_archive``'s archival step. Extracted so both
        call sites go through a single implementation and future changes
        to the edge-deletion SQL (logging, additional relationship types,
        audit events) land in one place instead of drifting apart.

        Does NOT wrap itself in a ``with self._conn:`` block — the caller
        is responsible for transaction context. This avoids nested
        SAVEPOINTs when the helper is called from inside another
        ``with self._conn:`` block (as ``apply_decay_and_archive`` does).
        """
        self._conn.execute(
            "DELETE FROM edges WHERE relationship = 'has_tag' AND source_id = ?",
            (memory_id,),
        )

    def remove_edges_for_memory(self, memory_id: str) -> None:
        self._check_open()
        try:
            with self._conn:
                # Only has_tag edges originate from memories.
                # child_of edges are between nodes and MUST be preserved.
                self._delete_has_tag_edges(memory_id)
        except sqlite3.Error as exc:
            raise StorageError(f"remove_edges_for_memory failed: {exc}") from exc

    def get_nearest_nodes(
        self,
        query_embedding: NDArray[np.floating] | list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        self._check_open()
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        query = np.asarray(query_embedding, dtype=np.float64)
        if query.ndim != 1 or query.size == 0:
            raise ValueError(
                f"query_embedding must be a non-empty 1-D vector, got shape {query.shape}"
            )

        nodes = self.get_all_nodes()
        if not nodes:
            return []

        # Stack all embeddings into a single (N, D) matrix and compute
        # cosine similarities in one BLAS call via numpy.
        matrix = np.stack([n["embedding"] for n in nodes]).astype(np.float64)
        if matrix.shape[1] != query.size:
            raise ValueError(f"query dim {query.size} != stored node dim {matrix.shape[1]}")

        sims = batch_cosine_similarity(query, matrix)
        effective_k = min(top_k, len(nodes))
        # argpartition is O(N) vs argsort's O(N log N) for top-k.
        top_idx = np.argpartition(-sims, effective_k - 1)[:effective_k]
        # Sort the top-k slice by similarity descending.
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        results: list[dict[str, Any]] = []
        for idx in top_idx:
            node = dict(nodes[int(idx)])
            node["similarity"] = float(sims[int(idx)])
            results.append(node)
        return results

    def get_subgraph(
        self,
        root_node_ids: list[str],
        depth: int = 2,
    ) -> dict[str, Any]:
        self._check_open()
        if depth < 0:
            raise ValueError(f"depth must be >= 0, got {depth}")
        if not root_node_ids:
            return {"nodes": [], "edges": [], "memories": []}

        try:
            return self._traverse_subgraph(root_node_ids, depth)
        except sqlite3.Error as exc:
            raise StorageError(f"get_subgraph failed: {exc}") from exc

    def _traverse_subgraph(
        self,
        root_node_ids: list[str],
        depth: int,
    ) -> dict[str, Any]:
        # BFS over child_of edges in BOTH directions from each root up to
        # `depth` hops. Then collect has_tag edges from every reached node
        # to gather non-archived memories.
        reached_node_ids: set[str] = set(root_node_ids)
        frontier: set[str] = set(root_node_ids)
        node_edges: list[sqlite3.Row] = []

        for _ in range(depth):
            if not frontier:
                break
            placeholders = ",".join("?" * len(frontier))
            rows = self._conn.execute(
                f"""
                SELECT source_id, target_id, weight, relationship
                FROM edges
                WHERE relationship = 'child_of'
                  AND (source_id IN ({placeholders}) OR target_id IN ({placeholders}))
                """,
                (*frontier, *frontier),
            ).fetchall()
            node_edges.extend(rows)
            new_frontier: set[str] = set()
            for row in rows:
                for endpoint in (row["source_id"], row["target_id"]):
                    if endpoint not in reached_node_ids:
                        new_frontier.add(endpoint)
                        reached_node_ids.add(endpoint)
            frontier = new_frontier

        # Fetch full node records for every reached id (including roots
        # that might not exist — silently dropped).
        if reached_node_ids:
            placeholders = ",".join("?" * len(reached_node_ids))
            node_rows = self._conn.execute(
                f"""
                SELECT id, label, embedding, embedding_dim, is_centroid, paragraph_summary
                FROM nodes WHERE id IN ({placeholders})
                """,
                tuple(reached_node_ids),
            ).fetchall()
        else:
            node_rows = []

        # Collect has_tag edges where target is a reached node and the
        # source (memory) is non-archived.
        if reached_node_ids:
            placeholders = ",".join("?" * len(reached_node_ids))
            mem_edge_rows = self._conn.execute(
                f"""
                SELECT e.source_id, e.target_id, e.weight, e.relationship
                FROM edges e
                JOIN memories m ON m.id = e.source_id
                WHERE e.relationship = 'has_tag'
                  AND e.target_id IN ({placeholders})
                  AND m.status != 'archived'
                """,
                tuple(reached_node_ids),
            ).fetchall()
        else:
            mem_edge_rows = []

        # Pull memory records for the non-archived has_tag sources.
        memory_ids = {row["source_id"] for row in mem_edge_rows}
        if memory_ids:
            placeholders = ",".join("?" * len(memory_ids))
            mem_rows = self._conn.execute(
                f"""
                SELECT id, raw_content, summary, status, access_weight,
                       last_accessed, created_at, metadata, named_entities
                FROM memories WHERE id IN ({placeholders})
                """,
                tuple(memory_ids),
            ).fetchall()
        else:
            mem_rows = []

        all_edges = [_edge_row_to_dict(r) for r in node_edges] + [
            _edge_row_to_dict(r) for r in mem_edge_rows
        ]

        return {
            "nodes": [_row_to_node_dict(r) for r in node_rows],
            "edges": all_edges,
            "memories": [_row_to_memory_dict(r) for r in mem_rows],
        }

    def apply_decay_and_archive(
        self,
        decay_lambda: float,
        archive_threshold: float,
        current_timestamp: int,
    ) -> list[str]:
        self._check_open()
        if decay_lambda <= 0:
            raise ValueError(f"decay_lambda must be positive, got {decay_lambda}")
        if not (0 <= archive_threshold < 1.0):
            raise ValueError(f"archive_threshold must be in [0, 1.0), got {archive_threshold}")

        archived: list[str] = []
        try:
            with self._conn:
                rows = self._conn.execute(
                    """
                    SELECT id, access_weight, last_accessed, created_at
                    FROM memories WHERE status = 'consolidated'
                    """
                ).fetchall()
                for row in rows:
                    anchor = row["last_accessed"]
                    if anchor is None:
                        anchor = row["created_at"]
                    t = max(0, current_timestamp - int(anchor))
                    w_new = float(row["access_weight"]) * math.exp(-decay_lambda * t)
                    mem_id = row["id"]
                    if w_new < archive_threshold:
                        self._conn.execute(
                            "UPDATE memories SET status = 'archived', access_weight = ? "
                            "WHERE id = ?",
                            (w_new, mem_id),
                        )
                        # Drop has_tag edges through the shared helper so
                        # this path honours the base class contract
                        # (StorageAdapter.apply_decay_and_archive docstring
                        # says: "call remove_edges_for_memory(id)").
                        self._delete_has_tag_edges(mem_id)
                        archived.append(mem_id)
                    else:
                        self._conn.execute(
                            "UPDATE memories SET access_weight = ? WHERE id = ?",
                            (w_new, mem_id),
                        )
        except sqlite3.Error as exc:
            raise StorageError(f"apply_decay_and_archive failed: {exc}") from exc
        return archived

    def spike_access_weight(
        self,
        memory_ids: list[str],
        timestamp: int,
    ) -> None:
        self._check_open()
        if not memory_ids:
            return
        placeholders = ",".join("?" * len(memory_ids))
        try:
            with self._conn:
                self._conn.execute(
                    f"""
                    UPDATE memories
                    SET access_weight = 1.0, last_accessed = ?
                    WHERE status = 'consolidated'
                      AND id IN ({placeholders})
                    """,
                    (int(timestamp), *memory_ids),
                )
        except sqlite3.Error as exc:
            raise StorageError(f"spike_access_weight failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_first_node_dim(self) -> int | None:
        """Return the embedding_dim of any existing node, or None if empty."""
        try:
            row = self._conn.execute("SELECT embedding_dim FROM nodes LIMIT 1").fetchone()
        except sqlite3.Error as exc:
            raise StorageError(f"_get_first_node_dim failed: {exc}") from exc
        if row is None:
            return None
        return int(row["embedding_dim"])

    # ------------------------------------------------------------------
    # Lifecycle — close, context manager protocol, GC fallback
    # ------------------------------------------------------------------

    def _check_open(self) -> None:
        """Raise ``StorageError`` if the adapter has been closed.

        Called at the top of every public method. Replaces the
        opaque ``TypeError: 'NoneType' object does not support the
        context manager protocol`` that a post-close
        ``with self._conn:`` would raise with a clear, actionable
        "adapter is closed" signal for the caller.
        """
        if self._conn is None:
            raise StorageError("SQLiteAdapter is closed")

    def close(self) -> None:
        """Close the underlying connection. Idempotent and safe to call
        multiple times. Does nothing if the connection was never opened
        or has already been closed.

        Prefer calling this explicitly (or using the adapter as a
        context manager) over relying on ``__del__``, which runs on
        garbage collection — timing is non-deterministic and on
        Python 3.13+ pytest's unraisable-exception plugin escalates
        any missed cleanups into test failures.
        """
        conn = getattr(self, "_conn", None)
        if conn is None:
            return
        with contextlib.suppress(sqlite3.Error):
            conn.close()
        self._conn = None  # type: ignore[assignment]

    def __enter__(self) -> SQLiteAdapter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        """Best-effort cleanup on garbage collection.

        Required for Python 3.13 + pytest: the stdlib ``sqlite3``
        module emits ``ResourceWarning: unclosed database`` when a
        connection is GC'd without an explicit close, and pytest's
        unraisable-exception plugin turns that warning into a test
        failure (attributed to whichever unrelated test happened to
        be running when GC fired). Adding ``__del__`` silences the
        warning by closing the connection before it gets a chance
        to complain.

        This is a fallback, not a replacement for explicit cleanup.
        Tests and production code SHOULD use ``close()`` or the
        context-manager protocol. See ``packages/neuromem-core/tests/
        test_sqlite_adapter.py::TestLifecycle`` for regression tests.
        """
        # __del__ must never raise. Narrow the suppress scope to
        # the two error classes a well-formed close() can
        # legitimately produce:
        #   sqlite3.Error — the underlying connection close failing
        #   AttributeError — partial construction (e.g., __init__
        #                    raised before _conn was assigned)
        # Anything else is a real bug and should propagate to
        # Python's unraisable-exception handler (visible on stderr
        # in CI). suppress(Exception) would silently mask real
        # subclass bugs; (sqlite3.Error, AttributeError) catches
        # exactly the cleanup-time failures we intend to tolerate.
        with contextlib.suppress(sqlite3.Error, AttributeError):
            self.close()


def _row_to_memory_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Convert a ``memories`` row into the public memory dict shape."""
    metadata: dict[str, Any] | None
    if row["metadata"] is None:
        metadata = None
    else:
        try:
            metadata = json.loads(row["metadata"])
        except json.JSONDecodeError as exc:
            raise StorageError(f"corrupt metadata for memory {row['id']}: {exc}") from exc

    # named_entities was added post-v0.1.0 as a dedicated column. The
    # migration back-fills existing rows to '[]' and the CREATE TABLE
    # has NOT NULL DEFAULT '[]', so a missing/NULL value in practice
    # would only come from hand-crafted test rows — we tolerate it
    # here to keep those simple.
    # sqlite3.Row's __contains__ checks values (not keys) unlike a dict,
    # so .keys() is required here — ruff SIM118 is a false positive.
    raw_entities = (
        row["named_entities"] if "named_entities" in row.keys() else None  # noqa: SIM118
    )
    named_entities: list[str]
    if raw_entities is None or raw_entities == "":
        named_entities = []
    else:
        try:
            parsed = json.loads(raw_entities)
        except json.JSONDecodeError as exc:
            raise StorageError(f"corrupt named_entities for memory {row['id']}: {exc}") from exc
        if not isinstance(parsed, list):
            raise StorageError(f"named_entities for memory {row['id']} is not a JSON list")
        # Coerce to list[str]; non-string elements are a contract violation.
        named_entities = [str(e) for e in parsed]

    return {
        "id": row["id"],
        "raw_content": row["raw_content"],
        "summary": row["summary"],
        "status": row["status"],
        "access_weight": float(row["access_weight"]),
        "last_accessed": int(row["last_accessed"]) if row["last_accessed"] is not None else None,
        "created_at": int(row["created_at"]),
        "metadata": metadata,
        "named_entities": named_entities,
    }


def _row_to_node_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Convert a ``nodes`` row into the public node dict shape.

    Deserialises the float32 blob via ``np.frombuffer`` and validates
    that the length matches ``embedding_dim`` — mismatch raises
    ``StorageError`` (corruption indicator per data-model.md).
    """
    blob = row["embedding"]
    expected_dim = int(row["embedding_dim"])
    # float32 → 4 bytes per element. Exact comparison (not `// 4`)
    # is essential: integer division rounds down, so a blob of
    # `expected_dim * 4 + r` for r ∈ {1, 2, 3} would pass the check
    # and produce a silently-truncated embedding at read time.
    if len(blob) != expected_dim * 4:
        raise StorageError(
            f"corrupt embedding for node {row['id']}: "
            f"blob size {len(blob)} does not match expected "
            f"{expected_dim} × 4 = {expected_dim * 4} bytes"
        )
    embedding = np.frombuffer(blob, dtype=np.float32).copy()  # .copy() makes it writable
    # paragraph_summary was added in ADR-003. sqlite3.Row.keys() includes
    # it on fresh schemas; on pre-migration rows it may be absent. Treat
    # missing-column as NULL (same pattern as named_entities).
    summary: str | None
    try:
        raw = row["paragraph_summary"]
    except (KeyError, IndexError):
        raw = None
    summary = str(raw) if raw is not None else None
    return {
        "id": row["id"],
        "label": row["label"],
        "embedding": embedding,
        "is_centroid": bool(row["is_centroid"]),
        "paragraph_summary": summary,
    }


def _edge_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Convert an ``edges`` row into the public edge dict shape."""
    return {
        "source_id": row["source_id"],
        "target_id": row["target_id"],
        "weight": float(row["weight"]),
        "relationship": row["relationship"],
    }
