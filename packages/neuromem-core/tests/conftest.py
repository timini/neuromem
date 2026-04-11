"""Shared pytest fixtures for neuromem-core.

Populated incrementally by:
  - T011: MockEmbeddingProvider, MockLLMProvider fixtures
  - T012: storage_adapter parametrised fixture (initially empty params)
  - T013: SQLiteAdapter added to storage_adapter params
  - T025: DictStorageAdapter added to storage_adapter params
"""

from __future__ import annotations

import copy
import hashlib
import math
import time
import uuid
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from neuromem.providers import EmbeddingProvider, LLMProvider
from neuromem.storage.base import StorageAdapter, StorageError
from neuromem.vectors import batch_cosine_similarity

# ---------------------------------------------------------------------------
# Mock providers (T011)
# ---------------------------------------------------------------------------


class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic seeded-RNG embedding for tests.

    For any input string, produces a fixed-dimension float32 vector
    derived from the md5 hash of the string. Two identical input
    strings always yield the same vector. Different input strings
    yield statistically uncorrelated vectors. Zero network access.
    """

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim

    def get_embeddings(self, texts: list[str]) -> np.ndarray:
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            digest = hashlib.md5(text.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:8], "big")
            rng = np.random.default_rng(seed)
            out[i] = rng.uniform(-1.0, 1.0, size=self.dim).astype(np.float32)
        return out


class MockLLMProvider(LLMProvider):
    """Deterministic canned-response LLM for tests.

    - generate_summary: first 80 chars of raw_text
    - extract_tags: first 3 alpha tokens from the summary
    - generate_category_name: 'Cat' + first letter of each concept

    Fast, deterministic, zero network access. The real test of the
    cognitive pipeline is that it calls these methods in the right
    order with the right arguments, which MockLLMProvider captures
    structurally (just by being a valid provider).
    """

    def generate_summary(self, raw_text: str) -> str:
        return raw_text[:80]

    def extract_tags(self, summary: str) -> list[str]:
        return [w for w in summary.split() if w.isalpha()][:3]

    def generate_category_name(self, concepts: list[str]) -> str:
        return "Cat" + "".join(c[:1].upper() for c in concepts)


@pytest.fixture
def mock_embedder() -> MockEmbeddingProvider:
    """A freshly-constructed 16-dim MockEmbeddingProvider."""
    return MockEmbeddingProvider(dim=16)


@pytest.fixture
def mock_llm() -> MockLLMProvider:
    """A freshly-constructed MockLLMProvider."""
    return MockLLMProvider()


# ---------------------------------------------------------------------------
# StorageAdapter parametrised fixture (T012)
# ---------------------------------------------------------------------------

# List of ``() -> StorageAdapter`` callables. Each entry produces a
# fresh adapter instance per test. T013 appends SQLiteAdapter;
# T025 appends DictStorageAdapter.
STORAGE_ADAPTER_FACTORIES: list[Callable[[], StorageAdapter]] = []


def sqlite_factory() -> StorageAdapter:
    """Return a fresh in-memory ``SQLiteAdapter`` for contract tests."""
    from neuromem.storage.sqlite import SQLiteAdapter

    return SQLiteAdapter(":memory:")


STORAGE_ADAPTER_FACTORIES.append(sqlite_factory)


# ---------------------------------------------------------------------------
# DictStorageAdapter (T025) — test-only in-memory adapter
# ---------------------------------------------------------------------------

_VALID_STATUSES = frozenset(("inbox", "dreaming", "consolidated", "archived"))
_VALID_RELATIONSHIPS = frozenset(("has_tag", "child_of"))


class DictStorageAdapter(StorageAdapter):
    """Test-only in-memory ``StorageAdapter`` backed by plain Python dicts.

    Deliberately lives in ``conftest.py`` — not in ``src/neuromem/`` —
    per Constitution v2.0.0 Principle III (Layered, Modular, Pluggable
    Architecture) and Principle V (Test-First with Mocks): test
    infrastructure stays out of the production package so the published
    wheel never ships mock adapters.

    The sole purpose of this adapter is to exercise the contract suite
    in ``test_storage_adapter_contract.py`` against a second backend,
    proving that orchestration code (``NeuroMemory``, ``ContextHelper``,
    ``neuromem.tools``) is coupled only to the ``StorageAdapter`` ABC —
    not to any sqlite-specific quirk. If the 13 contract tests pass for
    both ``SQLiteAdapter`` and ``DictStorageAdapter``, the adapter-swap
    property of Principle III is protected at CI time.

    State is held in three containers:

    - ``_memories``: ``dict[str, dict]`` keyed by memory id.
    - ``_nodes``: ``dict[str, dict]`` keyed by node id (each stores its
      own ``embedding_dim`` so the dim-consistency invariant matches
      ``SQLiteAdapter``'s schema-level guarantee).
    - ``_edges``: ``list[dict]`` — small enough that O(E) scans on every
      ``insert_edge`` / ``remove_edges_for_memory`` call are fine for
      the test workloads. Deduplication is done manually because the
      SQLite UNIQUE constraint has no in-memory equivalent.

    Every method that returns memory / node / edge dicts returns
    **defensive copies** so a caller cannot mutate internal state by
    holding a reference to a returned dict. This matches the behaviour
    of ``SQLiteAdapter``, whose rows come out of ``sqlite3`` as fresh
    ``Row`` objects per fetch.
    """

    def __init__(self) -> None:
        self._memories: dict[str, dict[str, Any]] = {}
        self._nodes: dict[str, dict[str, Any]] = {}
        self._edges: list[dict[str, Any]] = []
        self._closed = False

    def _check_open(self) -> None:
        if self._closed:
            raise StorageError("DictStorageAdapter is closed")

    @staticmethod
    def _copy_memory(m: dict[str, Any]) -> dict[str, Any]:
        """Return a defensive copy of a stored memory dict."""
        return {
            "id": m["id"],
            "raw_content": m["raw_content"],
            "summary": m["summary"],
            "status": m["status"],
            "access_weight": m["access_weight"],
            "last_accessed": m["last_accessed"],
            "created_at": m["created_at"],
            "metadata": copy.deepcopy(m["metadata"]),
        }

    @staticmethod
    def _copy_node(n: dict[str, Any]) -> dict[str, Any]:
        """Return a defensive copy of a stored node dict (without ``embedding_dim``)."""
        return {
            "id": n["id"],
            "label": n["label"],
            "embedding": n["embedding"].copy(),
            "is_centroid": n["is_centroid"],
        }

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

        memory_id = f"mem_{uuid.uuid4().hex}"
        self._memories[memory_id] = {
            "id": memory_id,
            "raw_content": raw_content,
            "summary": summary,
            "status": "inbox",
            "access_weight": 1.0,
            "last_accessed": None,
            "created_at": int(time.time()),
            "metadata": copy.deepcopy(metadata),
        }
        return memory_id

    def count_memories_by_status(self, status: str) -> int:
        self._check_open()
        if status not in _VALID_STATUSES:
            raise ValueError(f"invalid status: {status!r}")
        return sum(1 for m in self._memories.values() if m["status"] == status)

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

        rows = sorted(
            (m for m in self._memories.values() if m["status"] == status),
            key=lambda m: (m["created_at"], m["id"]),
        )
        if limit is not None:
            rows = rows[:limit]
        return [self._copy_memory(m) for m in rows]

    def update_memory_status(
        self,
        memory_ids: list[str],
        new_status: str,
    ) -> None:
        self._check_open()
        if new_status not in _VALID_STATUSES:
            raise ValueError(f"invalid status: {new_status!r}")
        if not memory_ids:
            return
        # Atomic from the caller's perspective: nothing else is running
        # concurrently in a single-threaded test. Missing ids are silently
        # skipped — matches SQLiteAdapter's "UPDATE ... WHERE id IN (...)"
        # which also no-ops on unknown ids.
        for mem_id in memory_ids:
            if mem_id in self._memories:
                self._memories[mem_id]["status"] = new_status

    def get_memory_by_id(self, memory_id: str) -> dict[str, Any] | None:
        self._check_open()
        row = self._memories.get(memory_id)
        if row is None:
            return None
        return self._copy_memory(row)

    # ------------------------------------------------------------------
    # Consolidation (Neocortex / Graph)
    # ------------------------------------------------------------------

    def upsert_node(
        self,
        node_id: str,
        label: str,
        embedding: Any,
        is_centroid: bool,
    ) -> None:
        self._check_open()
        arr = np.asarray(embedding, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError(f"embedding must be 1-D, got shape {arr.shape}")
        if arr.size == 0:
            raise ValueError("embedding must be non-empty")

        # Enforce the dimension-match invariant (data-model.md I-N3):
        # every node must have the same embedding dim as the first one.
        if self._nodes:
            existing_dim = next(iter(self._nodes.values()))["embedding_dim"]
            if arr.size != existing_dim:
                raise ValueError(f"embedding dim mismatch: expected {existing_dim}, got {arr.size}")

        self._nodes[node_id] = {
            "id": node_id,
            "label": label,
            # .copy() so a mutation of the caller's array doesn't leak in.
            "embedding": arr.copy(),
            "embedding_dim": int(arr.size),
            "is_centroid": bool(is_centroid),
        }

    def get_all_nodes(self) -> list[dict[str, Any]]:
        self._check_open()
        return [self._copy_node(n) for n in sorted(self._nodes.values(), key=lambda n: n["id"])]

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

        # Manual uniqueness check — matches SQLite's UNIQUE(source, target,
        # relationship) constraint + INSERT OR IGNORE semantics.
        key = (source_id, target_id, relationship)
        for e in self._edges:
            if (e["source_id"], e["target_id"], e["relationship"]) == key:
                return  # silent no-op — idempotent
        self._edges.append(
            {
                "source_id": source_id,
                "target_id": target_id,
                "weight": float(weight),
                "relationship": relationship,
            }
        )

    def remove_edges_for_memory(self, memory_id: str) -> None:
        self._check_open()
        # Only has_tag edges originate from memories. child_of edges are
        # between nodes and MUST be preserved (contract invariant — see
        # test_07b_remove_edges_preserves_child_of).
        self._edges = [
            e
            for e in self._edges
            if not (e["relationship"] == "has_tag" and e["source_id"] == memory_id)
        ]

    # ------------------------------------------------------------------
    # Recall (Prefrontal Cortex)
    # ------------------------------------------------------------------

    def get_nearest_nodes(
        self,
        query_embedding: Any,
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

        matrix = np.stack([n["embedding"] for n in nodes]).astype(np.float64)
        if matrix.shape[1] != query.size:
            raise ValueError(f"query dim {query.size} != stored node dim {matrix.shape[1]}")

        sims = batch_cosine_similarity(query, matrix)
        effective_k = min(top_k, len(nodes))
        top_idx = np.argpartition(-sims, effective_k - 1)[:effective_k]
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

        reached_node_ids: set[str] = set(root_node_ids)
        frontier: set[str] = set(root_node_ids)
        child_of_edges_visited: list[dict[str, Any]] = []

        # BFS over child_of edges in BOTH directions. Mirrors the
        # SQLiteAdapter implementation's behaviour so the same subgraph
        # emerges from both backends for identical input.
        for _ in range(depth):
            if not frontier:
                break
            new_frontier: set[str] = set()
            for e in self._edges:
                if e["relationship"] != "child_of":
                    continue
                if e["source_id"] in frontier or e["target_id"] in frontier:
                    child_of_edges_visited.append(dict(e))
                    for endpoint in (e["source_id"], e["target_id"]):
                        if endpoint not in reached_node_ids:
                            new_frontier.add(endpoint)
                            reached_node_ids.add(endpoint)
            frontier = new_frontier

        nodes_out = [
            self._copy_node(self._nodes[nid]) for nid in reached_node_ids if nid in self._nodes
        ]

        # Collect has_tag edges where the target is a reached node and the
        # source memory is non-archived (archived memories must not bleed
        # back into recall results — v1 forgetting is soft-delete).
        mem_edges_out: list[dict[str, Any]] = []
        for e in self._edges:
            if e["relationship"] != "has_tag":
                continue
            if e["target_id"] not in reached_node_ids:
                continue
            src = self._memories.get(e["source_id"])
            if src is None or src["status"] == "archived":
                continue
            mem_edges_out.append(dict(e))

        memory_ids_seen = {e["source_id"] for e in mem_edges_out}
        memories_out = [
            self._copy_memory(self._memories[mid])
            for mid in memory_ids_seen
            if mid in self._memories
        ]

        return {
            "nodes": nodes_out,
            "edges": child_of_edges_visited + mem_edges_out,
            "memories": memories_out,
        }

    # ------------------------------------------------------------------
    # Forgetting (Synaptic Pruning)
    # ------------------------------------------------------------------

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
        # list(...) snapshot so concurrent dict mutation during the loop is
        # harmless. (No concurrency here, but it matches SQLiteAdapter's
        # fetch-then-update pattern.)
        for mem_id, m in list(self._memories.items()):
            if m["status"] != "consolidated":
                continue
            anchor = m["last_accessed"]
            if anchor is None:
                anchor = m["created_at"]
            t = max(0, current_timestamp - int(anchor))
            w_new = float(m["access_weight"]) * math.exp(-decay_lambda * t)
            if w_new < archive_threshold:
                m["status"] = "archived"
                m["access_weight"] = w_new
                # Strip has_tag edges — matches the shared helper on
                # SQLiteAdapter. Inline here because the list comprehension
                # is already shared with ``remove_edges_for_memory``.
                self._edges = [
                    e
                    for e in self._edges
                    if not (e["relationship"] == "has_tag" and e["source_id"] == mem_id)
                ]
                archived.append(mem_id)
            else:
                m["access_weight"] = w_new
        return archived

    def spike_access_weight(
        self,
        memory_ids: list[str],
        timestamp: int,
    ) -> None:
        self._check_open()
        if not memory_ids:
            return
        ts = int(timestamp)
        for mem_id in memory_ids:
            m = self._memories.get(mem_id)
            if m is None:
                continue
            # LTP is reserved for consolidated memories per data-model.md
            # V-M4 / V-M5 and the contract test 11 — silent skip on any
            # other status.
            if m["status"] != "consolidated":
                continue
            m["access_weight"] = 1.0
            m["last_accessed"] = ts


def dict_factory() -> StorageAdapter:
    """Return a fresh ``DictStorageAdapter`` for contract tests."""
    return DictStorageAdapter()


STORAGE_ADAPTER_FACTORIES.append(dict_factory)


# Skip-sentinel used when STORAGE_ADAPTER_FACTORIES is empty. Without it,
# pytest collects the contract tests with 0 parameterisations and they
# disappear silently from the run output — a broken registration (e.g.,
# T013 or T025's append dropped in a merge conflict) would be
# indistinguishable from "adapters not yet implemented." With the sentinel,
# the tests always appear, marked clearly SKIPPED with an actionable reason.
_SKIP_SENTINEL = object()


def _fixture_id(factory_or_sentinel: object) -> str:
    if factory_or_sentinel is _SKIP_SENTINEL:
        return "no-adapters-registered"
    return getattr(factory_or_sentinel, "__name__", "anon").replace("_factory", "")


@pytest.fixture(
    params=STORAGE_ADAPTER_FACTORIES or [_SKIP_SENTINEL],
    ids=_fixture_id,
)
def storage_adapter(request: pytest.FixtureRequest) -> StorageAdapter:
    """Parametrised fixture over every concrete StorageAdapter.

    Used by test_storage_adapter_contract.py to run the 13-item
    contract test suite against every implementation in a single
    pytest run.

    Registration lifecycle:
      - Starts empty (no adapters registered).
      - T013 appends ``SQLiteAdapter`` factory.
      - T025 appends ``DictStorageAdapter`` factory.

    When the factory list is empty, this fixture yields a single
    skipped test instance per contract test function (via the
    ``_SKIP_SENTINEL``) with a clear reason, rather than silently
    omitting the tests from collection entirely. This makes a broken
    registration visible at CI time: ``pytest -v`` shows the skipped
    tests, and ``pytest --collect-only`` makes it clear that the
    contract suite exists.
    """
    if request.param is _SKIP_SENTINEL:
        pytest.skip(
            "No StorageAdapter factories registered in "
            "STORAGE_ADAPTER_FACTORIES. This is expected during "
            "T008-T012. Once T013 lands, SQLiteAdapter should be "
            "appended and this skip should disappear."
        )
    factory: Callable[[], StorageAdapter] = request.param
    return factory()
