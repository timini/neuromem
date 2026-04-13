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

# Centroids written during ``_run_clustering`` get a placeholder label
# of the form ``cluster_<12-hex-chars>``. The lazy naming flow
# (ADR-002) recognises this prefix at render time and replaces the
# placeholder with an LLM-generated semantic name. The 12-hex suffix
# is sourced from the centroid's UUID so labels remain unique even
# before naming.
#
# An LLM SHOULD NOT ever return a label that matches this exact form
# (lowercase "cluster_" + 12 hex chars) — but if it did, the resolver
# would treat it as still-placeholder and rename it again. Harmless.
_PLACEHOLDER_LABEL_PREFIX = "cluster_"


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

    def enqueue_session(
        self,
        turns: list[dict[str, str]],
        metadata: dict | None = None,
    ) -> str:
        """Ingest a whole multi-turn session as a single memory.

        A "session" is a coherent stretch of conversation between one
        user and the agent — the natural episodic unit for hippocampal-
        style encoding. Compare to ``enqueue``, which treats its input
        as a single memory regardless of internal structure.

        This method is preferred over looping ``enqueue`` per turn
        when the caller has explicit session boundaries (chat app,
        benchmark harness, log replay). Benefits vs per-turn ingestion:

        - **N× fewer LLM calls**: one ``generate_summary`` per session
          instead of per turn. A 4-turn session goes from 4 summary
          calls to 1. For a 200-turn / 40-session input that's a 40×
          call-count reduction and a 40× reduction in rate-limit
          pressure.
        - **Denser memories**: a single session summary captures the
          narrative arc of a conversation, preserving relationships
          between adjacent turns that a turn-isolated summary would
          miss. E.g., turn 4 says "I redeemed a coupon on creamer"
          and turn 5's reply mentions Target — a session summary
          naturally links them; per-turn summaries may not.
        - **Raw transcript preserved**: ``raw_content`` stores the
          full "role: text" transcript (turns joined by blank lines),
          so ``retrieve_memories`` can still surface the original
          turn-by-turn conversation when the LLM drills in.

        ``turns`` is a list of dicts, each with ``"role"`` and
        ``"text"`` keys. Other keys are ignored. Empty list raises
        ``ValueError``. A single turn is valid — it ingests as one
        memory with "role: text" as the raw content.

        Metadata is attached to the single resulting memory. Useful
        shape: ``{"session_index": i, "turn_count": len(turns)}``.
        """
        if not turns:
            raise ValueError("turns must be non-empty")
        for idx, turn in enumerate(turns):
            if not isinstance(turn, dict):
                raise ValueError(f"turns[{idx}] must be a dict, got {type(turn).__name__}")
            if "role" not in turn or "text" not in turn:
                raise ValueError(
                    f"turns[{idx}] must have 'role' and 'text' keys, got {list(turn.keys())}"
                )
        transcript = "\n\n".join(f"{t['role']}: {t['text']}" for t in turns)
        return self.enqueue(transcript, metadata=metadata)

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

    def resolve_centroid_names(self, nodes: list[dict]) -> None:
        """Lazily name centroids that still have placeholder labels.

        Per ADR-002 (lazy centroid naming): the dream-cycle clustering
        loop writes centroids with placeholder labels of the form
        ``cluster_<12-hex-chars>``. This method, called from
        ``ContextHelper.build_prompt_context`` at render time, replaces
        those placeholders with LLM-generated semantic ones — but
        ONLY for centroids actually appearing in the rendered subgraph.

        Algorithm:
        1. Filter ``nodes`` to centroids whose label still has the
           placeholder prefix. If none, return — nothing to do.
        2. For each placeholder centroid, fetch its ``child_of``
           children from storage. Build a list of pairs (the two
           child labels) per centroid.
        3. Try ``llm.generate_category_names_batch(pairs)`` — one
           round-trip for all of them.
        4. On any failure (exception, length mismatch, empty name),
           fall back to numpy nearest-neighbour for the failed
           centroids: the centroid is renamed to the label of its
           child whose embedding has the highest cosine similarity
           to the centroid's own embedding.
        5. Persist all updates via ``storage.update_node_labels`` —
           a single transaction. Subsequent renders touching the
           same centroids skip naming entirely.
        6. Mutate the in-memory ``nodes`` list dicts so the immediate
           render uses the new labels without a re-fetch.

        NEVER raises. Worst case: every placeholder stays as
        ``cluster_a3f8b1c01234`` — ugly but the renderer still
        returns a tree.
        """
        try:
            self._do_resolve_centroid_names(nodes)
        except Exception:
            logger.exception(
                "resolve_centroid_names: unexpected error, leaving centroid "
                "labels as placeholders. Render will still succeed with "
                "ugly labels."
            )

    def _do_resolve_centroid_names(self, nodes: list[dict]) -> None:
        """Inner implementation of resolve_centroid_names. May raise;
        the public wrapper catches and logs."""
        placeholders = [
            n for n in nodes if n.get("label", "").startswith(_PLACEHOLDER_LABEL_PREFIX)
        ]
        if not placeholders:
            return

        # Gather each placeholder centroid's children once. Store the
        # full child node records (id, label, embedding) so the numpy
        # fallback path can use embeddings without a second fetch.
        children_by_centroid: dict[str, list[dict]] = {}
        for centroid in placeholders:
            child_records = self._fetch_child_records(centroid["id"])
            if child_records:
                children_by_centroid[centroid["id"]] = child_records
        if not children_by_centroid:
            return

        # Try batched LLM naming, then numpy fallback for anything left
        # unnamed. Each helper returns a {centroid_id: new_label} dict.
        ordered_centroid_ids = list(children_by_centroid.keys())
        updates = self._try_batched_naming(ordered_centroid_ids, children_by_centroid)
        for centroid in placeholders:
            cid = centroid["id"]
            if cid in updates or cid not in children_by_centroid:
                continue
            updates[cid] = self._nearest_child_label(centroid, children_by_centroid[cid])

        if not updates:
            return

        # Persist + mutate the in-memory nodes list in place.
        self.storage.update_node_labels(updates)
        for node in nodes:
            new_label = updates.get(node["id"])
            if new_label is not None:
                node["label"] = new_label

    def _try_batched_naming(
        self,
        ordered_centroid_ids: list[str],
        children_by_centroid: dict[str, list[dict]],
    ) -> dict[str, str]:
        """Try the batched LLM naming. Return a partial
        ``{centroid_id: name}`` dict on success (skipping centroids
        whose returned name was empty). Return an empty dict on any
        failure — the caller falls back to numpy nearest-neighbour for
        every still-unnamed centroid.
        """
        pair_list: list[list[str]] = [
            [c["label"] for c in children_by_centroid[cid][:2]] for cid in ordered_centroid_ids
        ]
        try:
            names = self.llm.generate_category_names_batch(pair_list)
            if len(names) != len(pair_list):
                raise ValueError(
                    f"generate_category_names_batch returned {len(names)} "
                    f"names for {len(pair_list)} pairs"
                )
        except Exception:
            logger.warning(
                "resolve_centroid_names: batched LLM naming failed; "
                "falling back to numpy nearest-neighbour for %d centroid(s)",
                len(ordered_centroid_ids),
                exc_info=True,
            )
            return {}

        updates: dict[str, str] = {}
        for cid, name in zip(ordered_centroid_ids, names, strict=True):
            first_word = (name or "").strip().split()[0:1]
            if first_word:
                updates[cid] = first_word[0]
        return updates

    def _fetch_child_records(self, centroid_id: str) -> list[dict]:
        """Return the centroid's child node records via a depth-1
        subgraph fetch. Each record has ``id``, ``label``, and
        ``embedding`` (the latter is needed for the numpy fallback).
        Returns an empty list if the centroid has no child_of edges.
        """
        sub = self.storage.get_subgraph([centroid_id], depth=1)
        nodes_by_id = {n["id"]: n for n in sub.get("nodes", [])}
        child_ids = [
            e["target_id"]
            for e in sub.get("edges", [])
            if e.get("relationship") == "child_of" and e.get("source_id") == centroid_id
        ]
        return [nodes_by_id[cid] for cid in child_ids if cid in nodes_by_id]

    @staticmethod
    def _nearest_child_label(centroid: dict, children: list[dict]) -> str:
        """Return the label of whichever child has the highest cosine
        similarity to ``centroid``'s embedding. Used as the numpy
        fallback when the LLM batched naming fails.

        If ``centroid`` lacks an embedding (shouldn't happen — clustering
        always writes one), or if all children lack embeddings, returns
        the first child's label as a last-ditch fallback.
        """
        centroid_emb = centroid.get("embedding")
        if centroid_emb is None or not children:
            return children[0]["label"] if children else _PLACEHOLDER_LABEL_PREFIX
        c_vec = np.asarray(centroid_emb, dtype=np.float64)
        c_norm = float(np.linalg.norm(c_vec))
        if c_norm == 0.0:
            return children[0]["label"]
        best_label = children[0]["label"]
        best_sim = -np.inf
        for child in children:
            emb = child.get("embedding")
            if emb is None:
                continue
            v = np.asarray(emb, dtype=np.float64)
            n = float(np.linalg.norm(v))
            if n == 0.0:
                continue
            sim = float(np.dot(c_vec, v) / (c_norm * n))
            if sim > best_sim:
                best_sim = sim
                best_label = child["label"]
        return best_label

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

            # 2a. Extract tag labels per memory in one batched call.
            # Providers that override LLMProvider.extract_tags_batch
            # with a real batched LLM call (GeminiLLMProvider, etc.)
            # turn this from N serial requests into O(1). Providers
            # that don't override get a default implementation that
            # loops over extract_tags — correctness-identical to
            # the pre-#44 behaviour, just slower. See issue #44.
            summaries = [m["summary"] for m in inbox_memories]
            tag_lists = self.llm.extract_tags_batch(summaries)
            memories_with_tags = list(zip(inbox_memories, tag_lists, strict=True))

            # 2b. Extract named entities per memory (also in one batch).
            # Unlike tags (which feed clustering), entities are purely
            # annotative — stored on the memory row for rendering in
            # the context tree later. Providers that don't implement
            # NER get the ABC default which returns empty lists for
            # every summary, so the code path is correctness-safe
            # whether or not the provider has been upgraded.
            #
            # Rollback intent: this write commits BEFORE steps 3–9. If
            # a later step raises, the exception handler flips these
            # memories back to 'inbox' but does NOT clear the
            # named_entities column. This is intentional — the next
            # dream cycle will run NER again on the same summaries and
            # overwrite the column with an equivalent (or identical)
            # result, so end-state correctness is preserved and the
            # partially-written entity data never leaks into the graph.
            # Tags/edges are rolled back implicitly because they were
            # never committed (status flip gates visibility).
            entity_lists = self.llm.extract_named_entities_batch(summaries)
            entity_updates = {
                mem["id"]: entities
                for mem, entities in zip(inbox_memories, entity_lists, strict=True)
            }
            self.storage.set_named_entities(entity_updates)

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

            # Merge: compute centroid, write with placeholder label,
            # persist. ADR-002 (lazy centroid naming) defers naming to
            # render time — see ``ContextHelper.build_prompt_context``
            # which calls ``NeuroMemory.resolve_centroid_names`` to
            # rename placeholders to LLM-generated semantic ones, but
            # only for centroids that actually appear in the rendered
            # subgraph. The placeholder format is recognised by the
            # resolver via the "cluster_" prefix.
            centroid_emb = compute_centroid([a["embedding"], b["embedding"]])
            centroid_id = f"node_{uuid.uuid4()}"
            parent_label = _PLACEHOLDER_LABEL_PREFIX + centroid_id[5:17]

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
