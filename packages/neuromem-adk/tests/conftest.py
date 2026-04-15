"""Shared fixtures for the neuromem-adk test suite.

Two fixture groups:

1. **`gemini_api_key`** — session-scoped credential resolver. Reads
   from ``GOOGLE_API_KEY`` first (the canonical ADK env var), then
   ``GEMINI_API_KEY`` as a fallback (which is what the existing
   ``neuromem-gemini`` tests use), then parses the repo-root
   ``.env`` file as a last resort. Skips the test with an actionable
   message if no key is found anywhere. Used by the integration
   test in ``test_adk_integration.py``.

2. **Mock-backed unit-test fixtures** — ``mock_adk_agent`` builds a
   real ``google.adk.agents.Agent`` instance (not a home-rolled
   mock — we want to test against the real ADK class surface), and
   ``mock_neuromem_system`` builds a live ``NeuroMemory`` backed by
   an in-memory ``DictStorageAdapter`` with the deterministic mock
   providers from ``neuromem-core``'s test harness. Together they
   let us exercise the whole ``enable_memory`` wiring path without
   hitting the network.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

# Suppress the experimental-feature UserWarning google-adk 1.29
# emits at import time. The exact message shape differs across
# Python versions:
#   py3.10: "[EXPERIMENTAL] feature PLUGGABLE_AUTH is enabled."
#   py3.11+: "[EXPERIMENTAL] feature FeatureName.PLUGGABLE_AUTH is enabled."
# Match both shapes via the stable `[EXPERIMENTAL] ... is enabled`
# prefix/suffix pair. This also futureproofs against ADK adding
# other experimental features with the same warning template.
#
# Must happen BEFORE the `from google.adk.agents import Agent` line
# below — otherwise the workspace-root ``filterwarnings = ['error']``
# config escalates the warning into a hard failure during conftest
# loading. Scoped to the google.adk module tree so we never silence
# warnings from our own code or from neuromem-core.
warnings.filterwarnings(
    "ignore",
    message=r".*\[EXPERIMENTAL\].*is enabled",
    category=UserWarning,
    module=r"google\.adk\..*",
)

import copy  # noqa: E402
import hashlib  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402 — conftest filter has to come before ADK import
from google.adk.agents import Agent  # noqa: E402
from neuromem import NeuroMemory  # noqa: E402
from neuromem.providers import (  # noqa: E402
    EmbeddingProvider,
    LLMProvider,
)
from neuromem.storage.base import StorageAdapter, StorageError  # noqa: E402
from neuromem.vectors import batch_cosine_similarity  # noqa: E402

# ---------------------------------------------------------------------------
# Credential resolver (session-scoped)
# ---------------------------------------------------------------------------

# The env-var name the default provider path reads. ADK uses
# ``GOOGLE_API_KEY``; the existing ``neuromem-gemini`` tests use
# ``GEMINI_API_KEY``. We accept both so a developer who has already
# configured one doesn't need to configure a second one for the ADK
# integration tests. Order: GOOGLE_API_KEY first (the canonical ADK
# name), then GEMINI_API_KEY (our sibling's convention), then the
# .env file fallback.
_ENV_VAR_CANDIDATES = ("GOOGLE_API_KEY", "GEMINI_API_KEY")


def _parse_dotenv(path: Path) -> dict[str, str]:
    """Minimal ``.env`` parser — no interpolation, no export prefix.

    Accepts ``KEY=value`` lines, ignores blanks and ``#`` comments,
    strips matching single or double quotes. 10 lines of stdlib so we
    don't drag ``python-dotenv`` into the dev deps.
    """
    result: dict[str, str] = {}
    if not path.is_file():
        return result
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        result[key] = value
    return result


def _find_repo_root(start: Path) -> Path | None:
    """Walk upward from ``start`` looking for a directory containing
    both a ``.git`` folder and a workspace-root ``pyproject.toml``.
    Returns ``None`` if we hit the filesystem root without finding one.
    """
    current = start.resolve()
    for parent in (current, *current.parents):
        if (parent / ".git").exists() and (parent / "pyproject.toml").exists():
            return parent
    return None


@pytest.fixture(scope="session")
def gemini_api_key() -> str:
    """Return the Gemini API key, or skip the test if it's unavailable.

    Session-scoped so the resolution runs once per pytest session, not
    per test. See the module docstring for the resolution order.
    """
    # 1. Try environment variables in order.
    for env_name in _ENV_VAR_CANDIDATES:
        value = os.environ.get(env_name, "").strip()
        if value:
            return value

    # 2. Fall back to the repo-root .env file.
    repo_root = _find_repo_root(Path(__file__).parent)
    if repo_root is not None:
        env_vars = _parse_dotenv(repo_root / ".env")
        for env_name in _ENV_VAR_CANDIDATES:
            value = env_vars.get(env_name, "").strip()
            if value:
                return value

    pytest.skip(
        f"No Gemini API key found. Set one of {_ENV_VAR_CANDIDATES} as an "
        "environment variable or add it to the repo-root .env file to run "
        "the neuromem-adk integration tests. Example: "
        "`GOOGLE_API_KEY=your-key uv run pytest packages/neuromem-adk/ "
        "-m integration --no-cov`. The `--no-cov` flag is required because "
        "the workspace coverage gate is scoped to neuromem-core."
    )


# ---------------------------------------------------------------------------
# Mock ADK agent + NeuroMemory fixtures (unit tests, no network)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_adk_agent() -> Agent:
    """Build a minimal ``google.adk.agents.Agent`` for unit tests.

    Uses the real ADK ``Agent`` class (NOT a home-rolled mock) because
    we want to exercise the exact attribute surface ``enable_memory``
    mutates — tools list, callback slots, instruction — and we want
    the tests to fail loudly if ADK changes that surface in a patch
    release.

    The agent is configured with a valid model name but never actually
    invokes that model in unit tests — unit tests call the callback
    closures directly with synthetic ``Context`` / ``LlmRequest``
    arguments, so no network round-trip happens.
    """
    return Agent(
        model="gemini-2.0-flash-001",
        name="test_agent",
        instruction="You are a test assistant.",
    )


@pytest.fixture
def mock_neuromem_system() -> NeuroMemory:
    """Build a ``NeuroMemory`` backed by an in-memory dict adapter and
    deterministic mock providers. Fully offline, fully deterministic,
    re-instantiated per test (fresh state, no leakage)."""
    return NeuroMemory(
        storage=DictStorageAdapter(),
        llm=MockLLMProvider(),
        embedder=MockEmbeddingProvider(),
    )


# ---------------------------------------------------------------------------
# Mock providers + DictStorageAdapter (duplicated from neuromem-core's
# tests/conftest.py)
#
# These are duplicated from ``packages/neuromem-core/tests/conftest.py``
# rather than imported because pytest puts each package's test directory
# on sys.path independently — there is no reliable cross-package
# ``from tests.conftest import ...`` path. Duplicating ~120 lines of
# test infrastructure is the right call: these classes are stable and
# tiny, and self-contained test fixtures per sibling package is a
# cleaner pattern than trying to share tests directories across the
# workspace. If the core mock classes ever change signature, any
# divergence here surfaces as a test failure in neuromem-adk — which
# is exactly the feedback loop we want.
# ---------------------------------------------------------------------------


class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic seeded-RNG embedding for tests."""

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
    """Deterministic canned-response LLM for tests."""

    def generate_summary(self, raw_text: str) -> str:
        return raw_text[:80]

    def extract_tags(self, summary: str) -> list[str]:
        return [w for w in summary.split() if w.isalpha()][:3]

    def generate_category_name(self, concepts: list[str]) -> str:
        return "Cat" + "".join(c[:1].upper() for c in concepts)


_VALID_STATUSES = frozenset(("inbox", "dreaming", "consolidated", "archived"))
_VALID_RELATIONSHIPS = frozenset(("has_tag", "child_of"))


class DictStorageAdapter(StorageAdapter):
    """Test-only in-memory StorageAdapter backed by plain Python dicts.

    Duplicated from ``packages/neuromem-core/tests/conftest.py`` —
    see the comment above for the reasoning.
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
        return {
            "id": n["id"],
            "label": n["label"],
            "embedding": n["embedding"].copy(),
            "is_centroid": n["is_centroid"],
        }

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

    def update_memory_status(self, memory_ids: list[str], new_status: str) -> None:
        self._check_open()
        if new_status not in _VALID_STATUSES:
            raise ValueError(f"invalid status: {new_status!r}")
        for mem_id in memory_ids:
            if mem_id in self._memories:
                self._memories[mem_id]["status"] = new_status

    def get_memory_by_id(self, memory_id: str) -> dict[str, Any] | None:
        self._check_open()
        row = self._memories.get(memory_id)
        return None if row is None else self._copy_memory(row)

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
        if self._nodes:
            existing_dim = next(iter(self._nodes.values()))["embedding_dim"]
            if arr.size != existing_dim:
                raise ValueError(f"embedding dim mismatch: expected {existing_dim}, got {arr.size}")
        self._nodes[node_id] = {
            "id": node_id,
            "label": label,
            "embedding": arr.copy(),
            "embedding_dim": int(arr.size),
            "is_centroid": bool(is_centroid),
        }

    def update_node_labels(self, updates: dict[str, str]) -> None:
        self._check_open()
        if not updates:
            return
        for node_id, label in updates.items():
            if node_id in self._nodes:
                self._nodes[node_id]["label"] = label

    def update_junction_summaries(self, updates: dict[str, str]) -> None:
        self._check_open()
        if not updates:
            return
        for node_id, summary in updates.items():
            if node_id in self._nodes:
                self._nodes[node_id]["paragraph_summary"] = summary

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
        key = (source_id, target_id, relationship)
        for e in self._edges:
            if (e["source_id"], e["target_id"], e["relationship"]) == key:
                return
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
        self._edges = [
            e
            for e in self._edges
            if not (e["relationship"] == "has_tag" and e["source_id"] == memory_id)
        ]

    def get_nearest_nodes(
        self,
        query_embedding: Any,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        self._check_open()
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        query = np.asarray(query_embedding, dtype=np.float64)
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
        child_of_edges: list[dict[str, Any]] = []
        for _ in range(depth):
            if not frontier:
                break
            new_frontier: set[str] = set()
            for e in self._edges:
                if e["relationship"] != "child_of":
                    continue
                if e["source_id"] in frontier or e["target_id"] in frontier:
                    child_of_edges.append(dict(e))
                    for endpoint in (e["source_id"], e["target_id"]):
                        if endpoint not in reached_node_ids:
                            new_frontier.add(endpoint)
                            reached_node_ids.add(endpoint)
            frontier = new_frontier
        nodes_out = [
            self._copy_node(self._nodes[nid]) for nid in reached_node_ids if nid in self._nodes
        ]
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
            "edges": child_of_edges + mem_edges_out,
            "memories": memories_out,
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
                self._edges = [
                    e
                    for e in self._edges
                    if not (e["relationship"] == "has_tag" and e["source_id"] == mem_id)
                ]
                archived.append(mem_id)
            else:
                m["access_weight"] = w_new
        return archived

    def spike_access_weight(self, memory_ids: list[str], timestamp: int) -> None:
        self._check_open()
        if not memory_ids:
            return
        ts = int(timestamp)
        for mem_id in memory_ids:
            m = self._memories.get(mem_id)
            if m is None or m["status"] != "consolidated":
                continue
            m["access_weight"] = 1.0
            m["last_accessed"] = ts
