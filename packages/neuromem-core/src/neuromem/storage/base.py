"""The ``StorageAdapter`` abstract base class and ``StorageError``.

This is the contract layer of the storage subsystem per Constitution
v2.0.0 Principle III (Layered, Modular, Pluggable Architecture). The
orchestration layer (``neuromem.system``, ``neuromem.context``,
``neuromem.tools``) depends only on this file; concrete adapters
(``SQLiteAdapter``, future ``PostgresAdapter``, etc.) live below it
and are never imported by orchestration code.

A concrete adapter implements all 13 abstract methods. If any are
missing, Python's ABC machinery raises ``TypeError`` at instantiation
time (see ``test_storage_base.py`` contract tests).

See specs/001-neuromem-core/contracts/storage-adapter.md for the
full per-method behavioural contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

# numpy is already a mandatory runtime dependency per Constitution v2.0.0
# Principle II. Importing NDArray at module top-level (rather than under
# TYPE_CHECKING) lets typing.get_type_hints() work correctly for downstream
# tools introspecting adapter signatures, and prevents future concrete
# adapters (SQLiteAdapter, etc.) from copying the broken TYPE_CHECKING
# pattern.


class StorageError(RuntimeError):
    """Raised when a storage adapter encounters an I/O or DB failure.

    Callers should treat this as "the adapter is broken, stop". Not
    recoverable in v1. For argument validation failures (bad status,
    empty vector, etc.) adapters raise ``ValueError`` instead.
    """


class StorageAdapter(ABC):
    """The persistence contract every backend must satisfy.

    13 abstract methods, grouped by cognitive subsystem:

    - Acquisition (Hippocampus): insert_memory, count_memories_by_status,
      get_memories_by_status, update_memory_status, get_memory_by_id
    - Consolidation (Neocortex): upsert_node, get_all_nodes, insert_edge,
      remove_edges_for_memory
    - Recall (Prefrontal Cortex): get_nearest_nodes, get_subgraph
    - Forgetting (Synaptic Pruning): apply_decay_and_archive,
      spike_access_weight

    No default implementations. Missing any method raises ``TypeError``
    at instantiation time — that is the whole point of the ABC: it
    makes partial adapters fail loudly.
    """

    # ------------------------------------------------------------------
    # Acquisition (Hippocampus)
    # ------------------------------------------------------------------

    @abstractmethod
    def insert_memory(
        self,
        raw_content: str,
        summary: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Persist a new memory with ``status='inbox'``, ``access_weight=1.0``.

        Returns the new memory's UUID string. Atomic: either the row
        exists fully after return or not at all.
        """

    @abstractmethod
    def count_memories_by_status(self, status: str) -> int:
        """Return the count of memories with the given status.

        Must be cheap (``O(log n)`` or better via an index on
        ``status``) because it's called on every ``enqueue()`` to
        check the dream threshold.
        """

    @abstractmethod
    def get_memories_by_status(
        self,
        status: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return memory records matching ``status``, up to ``limit`` rows.

        Each dict has the full Memory shape per data-model.md §Memory.
        Order should be deterministic within a single call (recommended:
        ``ORDER BY created_at ASC``).
        """

    @abstractmethod
    def update_memory_status(
        self,
        memory_ids: list[str],
        new_status: str,
    ) -> None:
        """Atomically flip the status of the given memory IDs.

        Used by the double-buffer pattern in the dreaming pipeline.
        MUST be a single SQL statement (or equivalent atomic op) so
        the flip is indivisible.
        """

    @abstractmethod
    def get_memory_by_id(self, memory_id: str) -> dict[str, Any] | None:
        """Return a single memory record, or ``None`` if not found.

        Missing is not an error — callers (``retrieve_memories``)
        rely on silent skip.
        """

    @abstractmethod
    def set_summaries(self, updates: dict[str, str]) -> None:
        """Batch-write summary text onto existing memory rows (ADR-004).

        ``updates`` maps memory_id → summary string. Called from the
        dream cycle's step 2 to backfill the ``summary`` column for
        memories inserted via the new non-blocking ``enqueue`` path
        (which writes them with an empty summary).

        Contract for overrides:
        - Empty ``updates`` → no-op.
        - Missing memory IDs are silently skipped.
        - Atomic per call (single transaction in supporting backends).
        - Overwrites any existing summary value (the dream cycle is
          the authoritative summariser; previous values were either
          empty placeholders or stale from a rolled-back cycle).

        ABSTRACT because ADR-004's non-blocking enqueue path writes
        memories with empty summaries that MUST be backfilled in the
        dream cycle. An adapter that silently dropped summary writes
        here would leave every post-ADR-004 memory with an empty
        ``summary`` column, which downstream tag extraction + NER
        operate on — a silent data-loss failure mode the reviewer
        caught on PR #59. Making this abstract forces every adapter
        (test stubs included) to acknowledge the contract.
        """

    def set_named_entities(self, updates: dict[str, list[str]]) -> None:
        """Batch-write named-entity lists onto existing memory rows.

        ``updates`` maps memory_id → list of entity strings. Called
        from the dream cycle once per cycle, after NER extraction.

        NON-abstract with a safe default ``raise NotImplementedError``-
        free no-op fallback in the ABC: the dream cycle is defensive
        about providers that don't support entity storage, so an
        adapter that doesn't implement this (e.g., an in-memory dict
        adapter used in unit tests) simply skips entity persistence
        without breaking the cycle. Persistent adapters (SQLite,
        future Postgres) MUST override.

        Contract for overrides:
        - Empty ``updates`` → no-op.
        - Missing memory IDs are silently skipped.
        - Implementation is atomic per call (single transaction).

        Added post-v0.1.0 to support named-entity rendering in the
        context tree without forcing every existing adapter to
        implement it simultaneously.
        """
        # Default: do nothing. Test-only adapters may rely on this.
        _ = updates  # avoid unused-argument lint when overridden later
        return

    # ------------------------------------------------------------------
    # Consolidation (Neocortex / Graph)
    # ------------------------------------------------------------------

    @abstractmethod
    def upsert_node(
        self,
        node_id: str,
        label: str,
        embedding: NDArray[np.floating] | list[float],
        is_centroid: bool,
    ) -> None:
        """Insert or update a concept node.

        ``embedding`` length must match every other node in storage.
        Raises ``ValueError`` on dimension mismatch.
        """

    @abstractmethod
    def update_node_labels(self, updates: dict[str, str]) -> None:
        """Atomically rename a batch of nodes by id.

        Used by lazy centroid naming (ADR-002): centroids are written
        with placeholder labels during the dream cycle and renamed at
        render time to LLM-generated semantic ones. Persisting via
        this method means the second render of the same query is a
        cache hit.

        Contract:
        - Empty ``updates`` → no-op.
        - Missing node IDs are silently skipped (a node was deleted
          between subgraph fetch and naming — uncommon, harmless).
        - Implementation MUST be atomic per call (single transaction
          for storage that supports it).
        - Existing ``embedding`` and ``is_centroid`` of the renamed
          node are unchanged — only ``label`` updates.
        """

    @abstractmethod
    def update_junction_summaries(self, updates: dict[str, str]) -> None:
        """Atomically write a batch of per-centroid paragraph summaries.

        Used by ADR-003 hybrid summary caching: the dream cycle's
        trunk-summarisation step writes eagerly; ``resolve_junction_summaries``
        writes lazily on first render of deeper centroids. Sibling to
        ``update_node_labels``.

        Contract:
        - Empty ``updates`` → no-op.
        - Missing node IDs are silently skipped.
        - Implementation MUST be atomic per call.
        - Overwrites any existing summary (regeneration is expected when
          a subtree's membership changes across dream cycles).
        - Existing ``label``, ``embedding``, ``is_centroid`` unchanged.
        """

    @abstractmethod
    def get_all_nodes(self) -> list[dict[str, Any]]:
        """Return every node in storage with deserialised embeddings.

        Each dict has ``{"id", "label", "embedding", "is_centroid"}``
        where ``embedding`` is a ``np.ndarray`` (not a raw blob).
        """

    @abstractmethod
    def insert_edge(
        self,
        source_id: str,
        target_id: str,
        weight: float,
        relationship: str,
    ) -> None:
        """Insert a directed edge between two entities.

        Idempotent: if an edge with the same
        ``(source_id, target_id, relationship)`` triple exists, this
        is a silent no-op. ``relationship`` must be one of
        ``'has_tag'`` or ``'child_of'`` in v1.
        """

    @abstractmethod
    def remove_edges_for_memory(self, memory_id: str) -> None:
        """Remove all ``has_tag`` edges with the given memory as source.

        Called by the archival step in the dreaming cycle. Does NOT
        remove ``child_of`` edges or delete tag nodes.
        """

    # ------------------------------------------------------------------
    # Recall (Prefrontal Cortex)
    # ------------------------------------------------------------------

    @abstractmethod
    def get_nearest_nodes(
        self,
        query_embedding: NDArray[np.floating] | list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return the ``top_k`` nodes closest to ``query_embedding``.

        Sorted by decreasing cosine similarity. Each returned dict
        has the standard node shape plus an added ``"similarity"``
        key with the computed cosine value.
        """

    @abstractmethod
    def get_subgraph(
        self,
        root_node_ids: list[str],
        depth: int = 2,
    ) -> dict[str, Any]:
        """Traverse edges from ``root_node_ids`` up to ``depth`` hops.

        Returns a dict with keys:
          - ``"nodes"``: list of reached node records
          - ``"edges"``: list of connecting edges
          - ``"memories"``: list of non-archived memories attached
            via ``has_tag`` edges to any reached node
        """

    # ------------------------------------------------------------------
    # Forgetting (Synaptic Pruning)
    # ------------------------------------------------------------------

    @abstractmethod
    def apply_decay_and_archive(
        self,
        decay_lambda: float,
        archive_threshold: float,
        current_timestamp: int,
    ) -> list[str]:
        """Apply exponential decay and archive below-threshold memories.

        For every ``status='consolidated'`` memory:
          - ``W_new = W_old * exp(-lambda * (now - last_accessed))``
          - If ``W_new < archive_threshold``: set ``status='archived'``
            and call ``remove_edges_for_memory(id)``.

        Returns the list of memory IDs that were archived this call.
        """

    @abstractmethod
    def spike_access_weight(
        self,
        memory_ids: list[str],
        timestamp: int,
    ) -> None:
        """Long-Term Potentiation: reset ``access_weight`` to 1.0.

        For every memory in ``memory_ids`` with
        ``status='consolidated'``, set ``access_weight=1.0`` and
        ``last_accessed=timestamp``. Memories with other statuses
        are silently skipped — no LTP for inbox / dreaming / archived.
        """
