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

import numpy as np

from .providers import EmbeddingProvider, LLMProvider
from .storage.base import StorageAdapter
from .vectors import compute_centroid

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
        # acquired non-blockingly inside _run_dream_cycle to make
        # concurrent cycles impossible — a second acquire attempt
        # returns immediately and the attempted cycle is a no-op.
        self._dream_lock = threading.Lock()
        self._is_dreaming = False

        # Most recent background dreaming thread, or None if no
        # background cycle has ever been spawned. Tests can join()
        # this to wait for completion; orchestration uses it for the
        # "join existing thread before running my own cycle" path in
        # force_dream(block=True).
        self._dream_thread: threading.Thread | None = None

    @property
    def is_dreaming(self) -> bool:
        """True iff a background dreaming cycle is currently running."""
        return self._is_dreaming

    def enqueue(self, raw_text: str, metadata: dict | None = None) -> str:
        """Insert ``raw_text`` into the inbox and return its memory id.

        Calls ``llm.generate_summary`` synchronously on the caller's
        thread (per Resolved Design Decision #3 — KISS), inserts the
        memory with ``status='inbox'`` via the storage adapter, and
        then — if the post-insert inbox count meets ``dream_threshold``
        AND no dream cycle is currently in flight — spawns a daemon
        background thread to run ``_run_dream_cycle``.

        The ``and not self._is_dreaming`` guard is a cheap optimisation
        to avoid creating a thread that will immediately no-op. It is
        a soft check (there's a small race between the test and the
        spawn), but the authoritative serialisation is
        ``_run_dream_cycle``'s own non-blocking lock acquire — a second
        thread sees the lock held and returns immediately. Correctness
        comes from the lock, not from this check.

        Raises ``ValueError`` on empty ``raw_text`` or non-JSON-
        serialisable metadata. Propagates any provider or storage
        exceptions.
        """
        if not raw_text:
            raise ValueError("raw_text must be non-empty")

        summary = self.llm.generate_summary(raw_text)
        memory_id = self.storage.insert_memory(
            raw_content=raw_text,
            summary=summary,
            metadata=metadata,
        )

        inbox_count = self.storage.count_memories_by_status("inbox")
        if inbox_count >= self.dream_threshold and not self._is_dreaming:
            logger.debug(
                "enqueue: inbox count %d >= threshold %d — spawning dream thread",
                inbox_count,
                self.dream_threshold,
            )
            self._spawn_dream_thread()

        return memory_id

    def _spawn_dream_thread(self) -> threading.Thread:
        """Start a daemon thread running ``_run_dream_cycle``.

        Stores a reference to the thread on ``self._dream_thread`` so
        tests can ``join()`` it and ``force_dream`` can wait for it.
        Returns the thread object for convenience.

        Safe to call even when a cycle is already in flight: the new
        thread's ``_run_dream_cycle`` call sees the lock held and
        returns immediately without touching any memories.
        """
        thread = threading.Thread(
            target=self._run_dream_cycle,
            name="neuromem-dream",
            daemon=True,
        )
        self._dream_thread = thread
        thread.start()
        return thread

    def force_dream(self, block: bool = True) -> None:
        """Manually trigger a dreaming cycle.

        Two modes:

        - ``block=True`` (default): if a background cycle is already
          in flight, join it first, then run ``_run_dream_cycle``
          synchronously on the caller's thread to process any memories
          that arrived during (or were enqueued immediately after) the
          background cycle. Returns when everything has been
          consolidated.
        - ``block=False``: if a background cycle is already in flight,
          do nothing (it will eventually finish and there is no point
          spawning another). Otherwise spawn a fresh daemon thread and
          return immediately.

        Tests that need a block=False call to have actually finished
        can retrieve the spawned thread via ``system._dream_thread``
        and call ``.join()`` on it.
        """
        existing = self._dream_thread
        if existing is not None and existing.is_alive():
            if block:
                existing.join()
            else:
                # Another thread is already running — the caller's
                # request for "start a cycle" is already satisfied.
                return

        if block:
            self._run_dream_cycle()
        else:
            self._spawn_dream_thread()

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

            # 2. Extract tag labels per memory in one batched call.
            # Providers that override LLMProvider.extract_tags_batch
            # with a real batched LLM call (GeminiLLMProvider, etc.)
            # turn this from N serial requests into O(1). Providers
            # that don't override get a default implementation that
            # loops over extract_tags — correctness-identical to
            # the pre-#44 behaviour, just slower. See issue #44.
            summaries = [m["summary"] for m in inbox_memories]
            tag_lists = self.llm.extract_tags_batch(summaries)
            memories_with_tags = list(zip(inbox_memories, tag_lists, strict=True))

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
        """Greedy agglomerative clustering over all current nodes.

        Builds a pairwise cosine-similarity matrix via numpy, finds
        the highest-similarity pair, merges them into a centroid
        node (``is_centroid=True``) with an LLM-generated one-word
        label, writes ``child_of`` edges from the centroid to both
        members, and repeats until no remaining pair exceeds
        ``cluster_threshold``.

        The method MUTATES storage (upserts centroid nodes, inserts
        child_of edges) but does NOT mutate ``label_to_node`` — leaf
        tag nodes stay addressable so ``_run_dream_cycle`` can still
        wire ``has_tag`` edges from memories to their direct tags
        (not to their centroid ancestors).

        Safety bound: the loop runs at most ``2 * len(candidates)``
        iterations so a pathological similarity matrix cannot hang.

        Design rationale — why hand-rolled numpy and not scipy /
        sklearn / HDBSCAN: see ``docs/decisions/ADR-001-clustering-
        library-choice.md``. TL;DR: numpy is already a mandatory
        runtime dep (Principle II); the per-merge
        ``llm.generate_category_name`` callback is the reason the
        loop is custom (libraries expose dendrograms, not merge
        hooks); and at k ≤ 5000 the dense pairwise matrix is
        single-digit milliseconds — the libraries would add install
        surface without changing the hot-path cost. Revisit when any
        of (a) k > 5000 hits SC-003, (b) LongMemEval shows cluster-
        quality regressions vs. a library baseline, or (c) a
        downstream wrapper needs Ward linkage / soft clustering /
        noise labels. See the ADR for the full trade-off analysis
        and the amendment procedure required to swap.
        """
        # Copy the starting nodes into a working list. Appending
        # centroids to this list grows the pool — a centroid can
        # itself merge with another node on a later iteration,
        # producing a multi-level hierarchy.
        candidates: list[dict] = [
            {
                "id": node["id"],
                "label": node["label"],
                "embedding": np.asarray(node["embedding"], dtype=np.float64),
                "is_centroid": bool(node["is_centroid"]),
            }
            for node in label_to_node.values()
        ]
        alive = [True] * len(candidates)

        max_iterations = max(2, 2 * len(candidates))
        for _iteration in range(max_iterations):
            alive_indices = [i for i, a in enumerate(alive) if a]
            if len(alive_indices) < 2:
                break

            # Build a (k, D) matrix of only the alive embeddings.
            alive_embs = np.stack([candidates[i]["embedding"] for i in alive_indices])

            # Pairwise cosine similarity: normalise rows, then dot-product.
            row_norms = np.linalg.norm(alive_embs, axis=1, keepdims=True)
            safe_norms = np.where(row_norms > 0.0, row_norms, 1.0)
            normed = alive_embs / safe_norms
            sim = normed @ normed.T
            np.fill_diagonal(sim, -np.inf)  # never merge with self

            # Find the highest-similarity pair.
            flat_idx = int(np.argmax(sim))
            k = len(alive_indices)
            i_rel, j_rel = divmod(flat_idx, k)
            best_sim = float(sim[i_rel, j_rel])
            if best_sim < self.cluster_threshold:
                break

            # Resolve back to indices in the full candidates list.
            i = alive_indices[i_rel]
            j = alive_indices[j_rel]
            a = candidates[i]
            b = candidates[j]

            # Merge: compute centroid, name it, persist.
            centroid_emb = compute_centroid([a["embedding"], b["embedding"]])
            raw_label = self.llm.generate_category_name([a["label"], b["label"]])
            parent_label = _sanitise_category_name(raw_label)
            centroid_id = f"node_{uuid.uuid4()}"

            self.storage.upsert_node(
                node_id=centroid_id,
                label=parent_label,
                embedding=centroid_emb,
                is_centroid=True,
            )
            self.storage.insert_edge(
                source_id=centroid_id,
                target_id=a["id"],
                weight=best_sim,
                relationship="child_of",
            )
            self.storage.insert_edge(
                source_id=centroid_id,
                target_id=b["id"],
                weight=best_sim,
                relationship="child_of",
            )

            # Retire the two members and add the centroid as a new
            # live candidate so the next iteration can merge IT too.
            alive[i] = False
            alive[j] = False
            candidates.append(
                {
                    "id": centroid_id,
                    "label": parent_label,
                    "embedding": centroid_emb,
                    "is_centroid": True,
                }
            )
            alive.append(True)


def _sanitise_category_name(raw: str) -> str:
    """Enforce the one-word contract on LLMProvider.generate_category_name.

    If the LLM returns multiple words or an empty string, take the
    first word (or a safe default) and log a warning. This is the
    defensive layer that protects the concept graph from free-form
    LLM output sneaking in as a node label.
    """
    stripped = (raw or "").strip()
    if not stripped:
        return "Category"
    first_word = stripped.split()[0]
    if len(stripped.split()) > 1:
        logger.warning(
            "generate_category_name returned multi-word %r; using %r",
            raw,
            first_word,
        )
    return first_word
