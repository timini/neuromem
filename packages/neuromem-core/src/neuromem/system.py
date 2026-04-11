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
        """Manually trigger a dreaming cycle. Lands in T020/T021."""
        raise NotImplementedError("force_dream lands in T020 (dream cycle) + T021 (threading)")
