"""The ``NeuroMemory`` orchestration engine.

This is the orchestration layer per Constitution v2.0.0 Principle III
(Layered, Modular, Pluggable Architecture). ``NeuroMemory`` depends
only on the contract layer (``providers.py`` and ``storage.base``),
never on concrete adapters. Concrete adapters and providers are
injected via the constructor.

Public API:

    memory = NeuroMemory(
        storage=SQLiteAdapter("memory.db"),
        llm=MyLLMProvider(),
        embedder=MyEmbeddingProvider(),
    )
    memory.enqueue("raw text")          # Hippocampus
    memory.force_dream(block=True)      # Manually trigger dreaming
    memory.is_dreaming                  # Read-only status property

``enqueue`` runs on the caller's thread and triggers a background
dreaming cycle when the inbox count hits ``dream_threshold``. The
dreaming cycle runs on a daemon thread and is guarded by a single
``threading.Lock`` to prevent concurrent cycles.

See specs/001-neuromem-core/contracts/public-api.md for the full
behavioural contract per method.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid

from .providers import EmbeddingProvider, LLMProvider
from .storage.base import StorageAdapter

logger = logging.getLogger("neuromem.system")

# Defaults come from specs/001-neuromem-core/spec.md §Resolved Design
# Decisions and §Assumptions. Every default is overridable via the
# NeuroMemory constructor.
DEFAULT_DREAM_THRESHOLD = 10
DEFAULT_DECAY_LAMBDA = 3e-7  # per second → ~30-day half-life
DEFAULT_ARCHIVE_THRESHOLD = 0.1
DEFAULT_CLUSTER_THRESHOLD = 0.82


class NeuroMemory:
    """Cognitive orchestration engine for ``neuromem-core``.

    Accepts a ``StorageAdapter``, ``LLMProvider``, and
    ``EmbeddingProvider`` via constructor injection. Exposes
    ``enqueue`` (Hippocampus) and ``force_dream`` (manual Neocortex
    trigger) plus the ``is_dreaming`` read-only property.
    """

    def __init__(
        self,
        storage: StorageAdapter,
        llm: LLMProvider,
        embedder: EmbeddingProvider,
        dream_threshold: int = DEFAULT_DREAM_THRESHOLD,
        decay_lambda: float = DEFAULT_DECAY_LAMBDA,
        archive_threshold: float = DEFAULT_ARCHIVE_THRESHOLD,
        cluster_threshold: float = DEFAULT_CLUSTER_THRESHOLD,
    ) -> None:
        if not isinstance(storage, StorageAdapter):
            raise TypeError(f"storage must be a StorageAdapter, got {type(storage).__name__}")
        if not isinstance(llm, LLMProvider):
            raise TypeError(f"llm must be an LLMProvider, got {type(llm).__name__}")
        if not isinstance(embedder, EmbeddingProvider):
            raise TypeError(f"embedder must be an EmbeddingProvider, got {type(embedder).__name__}")
        if dream_threshold < 1:
            raise ValueError(f"dream_threshold must be >= 1, got {dream_threshold}")
        if decay_lambda <= 0:
            raise ValueError(f"decay_lambda must be positive, got {decay_lambda}")
        if not (0 <= archive_threshold < 1.0):
            raise ValueError(f"archive_threshold must be in [0, 1.0), got {archive_threshold}")
        if not (0 < cluster_threshold <= 1.0):
            raise ValueError(f"cluster_threshold must be in (0, 1.0], got {cluster_threshold}")

        self.storage = storage
        self.llm = llm
        self.embedder = embedder
        self.dream_threshold = dream_threshold
        self.decay_lambda = decay_lambda
        self.archive_threshold = archive_threshold
        self.cluster_threshold = cluster_threshold

        # Single lock gating the dreaming cycle. _is_dreaming is the
        # authoritative "is a cycle in flight" flag; _dream_lock is
        # only used to make the check-and-set of that flag atomic.
        # Threading support is wired up in T021.
        self._dream_lock = threading.Lock()
        self._is_dreaming = False

    @property
    def is_dreaming(self) -> bool:
        """True iff a background dreaming cycle is currently running."""
        return self._is_dreaming

    def enqueue(self, raw_text: str, metadata: dict | None = None) -> str:
        """Insert ``raw_text`` into the inbox and return its memory id.

        Calls ``llm.generate_summary`` synchronously on the caller's
        thread (per Resolved Design Decision #3 — KISS). If the
        post-insert inbox count meets ``dream_threshold``, T021 will
        spawn a background thread here to run the dreaming cycle; for
        now T019 lands just the synchronous path.

        Raises ``ValueError`` on empty ``raw_text`` or non-JSON
        metadata. Propagates any provider or storage exceptions.
        """
        if not raw_text:
            raise ValueError("raw_text must be non-empty")

        summary = self.llm.generate_summary(raw_text)
        memory_id = self.storage.insert_memory(
            raw_content=raw_text,
            summary=summary,
            metadata=metadata,
        )

        # T021 will wire up the background thread trigger here. For now
        # we just check the count so the instrumentation is in place.
        inbox_count = self.storage.count_memories_by_status("inbox")
        if inbox_count >= self.dream_threshold:
            logger.debug(
                "enqueue: inbox count %d >= threshold %d (dreaming spawn lands in T021)",
                inbox_count,
                self.dream_threshold,
            )

        return memory_id

    def force_dream(self, block: bool = True) -> None:
        """Manually trigger a dreaming cycle.

        T020 Half A: ``block=True`` runs the cycle synchronously on
        the caller's thread. ``block=False`` is not yet supported —
        the background-thread plumbing lands in T021 alongside the
        ``enqueue`` threshold-triggered spawn.
        """
        if not block:
            raise NotImplementedError(
                "force_dream(block=False) lands in T021 — background "
                "thread support not implemented yet"
            )
        self._run_dream_cycle()

    # ------------------------------------------------------------------
    # Dreaming cycle (Neocortex) — T020
    #
    # T020 lands in two halves:
    #   Half A (this method body): status flip, tag extraction, batch
    #     embed, has_tag wiring, apply_decay_and_archive, consolidate
    #     flip, rollback on exception. Clustering is a no-op stub.
    #   Half B (follow-up commit): the agglomerative clustering loop
    #     populated into _run_clustering.
    # ------------------------------------------------------------------

    def _run_dream_cycle(self) -> None:
        """Process every inbox memory through the consolidation pipeline.

        Acquires ``_dream_lock`` non-blockingly. If the lock is already
        held, returns immediately — a second cycle can't run concurrently
        with the first. Empty inbox is also a no-op.

        Pipeline steps (see spec.md §Phase B):
          1. Flip inbox batch → dreaming (atomic SQL update).
          2. Extract tags per memory via llm.extract_tags(summary).
          3. Dedupe new labels and identify ones not already in storage.
          4. Batch-embed truly-new labels in one embedder.get_embeddings call.
          5. Upsert new leaf tag nodes (is_centroid=False).
          6. Run agglomerative clustering (Half B — currently a no-op).
          7. Wire has_tag edges from each memory to its tag nodes.
          8. Apply decay and archive memories whose access_weight dropped.
          9. Flip dreaming batch → consolidated.

        On any exception in steps 2-9: log, flip the batch back to
        inbox, re-raise. The lock is always released in the finally
        block.
        """
        if not self._dream_lock.acquire(blocking=False):
            logger.debug("_run_dream_cycle: lock already held, skipping")
            return

        dreaming_ids: list[str] = []
        try:
            self._is_dreaming = True

            # 1. Snapshot inbox and flip to dreaming atomically.
            inbox_memories = self.storage.get_memories_by_status("inbox")
            if not inbox_memories:
                logger.debug("_run_dream_cycle: empty inbox, nothing to do")
                return

            dreaming_ids = [m["id"] for m in inbox_memories]
            self.storage.update_memory_status(dreaming_ids, "dreaming")

            # 2. Extract tag labels per memory.
            memories_with_tags = [(m, self.llm.extract_tags(m["summary"])) for m in inbox_memories]

            # 3-5. Resolve labels → node records, creating nodes for any
            #      label that isn't already in storage.
            label_to_node = self._ensure_tag_nodes(memories_with_tags)

            # 6. Agglomerative clustering (Half B — no-op in this commit).
            self._run_clustering(label_to_node)

            # 7. Wire has_tag edges from each memory to its tag nodes.
            for memory, tag_labels in memories_with_tags:
                for label in tag_labels:
                    node = label_to_node.get(label)
                    if node is None:
                        continue
                    self.storage.insert_edge(
                        source_id=memory["id"],
                        target_id=node["id"],
                        weight=1.0,
                        relationship="has_tag",
                    )

            # 8. Apply decay + archive.
            now = int(time.time())
            self.storage.apply_decay_and_archive(
                decay_lambda=self.decay_lambda,
                archive_threshold=self.archive_threshold,
                current_timestamp=now,
            )

            # 9. Flip dreaming → consolidated.
            self.storage.update_memory_status(dreaming_ids, "consolidated")

        except Exception:
            logger.exception(
                "dream cycle failed, rolling back %d memories to inbox",
                len(dreaming_ids),
            )
            if dreaming_ids:
                try:
                    self.storage.update_memory_status(dreaming_ids, "inbox")
                except Exception:
                    logger.exception("rollback itself failed — manual cleanup required")
            raise
        finally:
            self._is_dreaming = False
            self._dream_lock.release()

    def _ensure_tag_nodes(
        self,
        memories_with_tags: list[tuple[dict, list[str]]],
    ) -> dict[str, dict]:
        """Make sure every distinct tag label has a node in storage.

        Returns a dict mapping label → node record (with ``id``,
        ``label``, ``embedding``, ``is_centroid``). Labels that already
        exist in storage are reused; truly-new labels are embedded in
        a single ``embedder.get_embeddings`` call and inserted via
        ``upsert_node``.
        """
        all_labels: set[str] = set()
        for _memory, tags in memories_with_tags:
            all_labels.update(tags)

        existing_nodes = self.storage.get_all_nodes()
        label_to_node: dict[str, dict] = {n["label"]: n for n in existing_nodes}
        truly_new_labels = sorted(all_labels - label_to_node.keys())

        if not truly_new_labels:
            return label_to_node

        new_embeddings = self.embedder.get_embeddings(truly_new_labels)
        for label, embedding in zip(truly_new_labels, new_embeddings, strict=True):
            node_id = f"node_{uuid.uuid4()}"
            self.storage.upsert_node(
                node_id=node_id,
                label=label,
                embedding=embedding,
                is_centroid=False,
            )
            label_to_node[label] = {
                "id": node_id,
                "label": label,
                "embedding": embedding,
                "is_centroid": False,
            }

        return label_to_node

    def _run_clustering(self, label_to_node: dict[str, dict]) -> None:
        """Agglomerative clustering over all current nodes.

        T020 Half A: stub — does nothing. Half B lands the greedy
        pairwise-merge loop that creates centroid parent nodes and
        ``child_of`` edges per spec.md §Phase B step 8.
        """
        _ = label_to_node  # used in Half B
