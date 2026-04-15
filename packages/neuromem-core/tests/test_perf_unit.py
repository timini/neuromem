"""Unit-level performance tripwires for ADR-003 NFRs.

These are tight, deterministic, CI-safe perf assertions. Each one
runs in < 10 seconds on reasonable hardware and catches order-of-
magnitude regressions before they reach the full-stack performance
harness.

Targets are chosen conservatively against the ADR-003 non-functional
requirements. If a test fails on slower CI hardware, raise the
threshold but leave a comment on what the baseline was — don't
delete the test. These wrappers pay off the first time a lazy
implementation sneaks in an O(N²) scan where the old code was O(N).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest
from neuromem.clustering import HDBSCANClusteringProvider
from neuromem.context import ContextHelper, _enforce_node_cap, _render_ascii_tree
from neuromem.providers import ClusteringProvider, LLMProvider
from neuromem.storage.sqlite import SQLiteAdapter
from neuromem.system import NeuroMemory

from tests.conftest import MockEmbeddingProvider, MockLLMProvider

# ---------------------------------------------------------------------------
# HDBSCAN clustering throughput (supports NF4 dream-cycle bound)
# ---------------------------------------------------------------------------


class TestHDBSCANClusteringPerf:
    def test_clusters_500_nodes_under_3s(self) -> None:
        """HDBSCAN on 500 synthetic 16-D embeddings completes in < 3s.

        Realistic size: a typical dream cycle on a LongMemEval
        instance produces 500-1000 tag nodes post-extract_tags. The
        clusterer is called ONCE per cycle and must not dominate
        wall time.

        Budget is generous (~3s) because HDBSCAN's per-level compile
        of the condensed tree is non-trivial and our recursive-mode
        adds a second level. Typical wall is 0.5-1.5s; we want to
        catch 10× regressions, not 1.5× noise.
        """
        rng = np.random.default_rng(seed=42)
        dim = 16
        centres = rng.normal(scale=5.0, size=(10, dim))
        nodes: list[tuple[str, np.ndarray]] = []
        for b, centre in enumerate(centres):
            for i in range(50):
                emb = centre + rng.normal(scale=0.2, size=dim)
                nodes.append((f"b{b}_n{i}", emb.astype(np.float32)))
        provider = HDBSCANClusteringProvider()

        start = time.perf_counter()
        clusters = provider.cluster(nodes)
        elapsed = time.perf_counter() - start

        assert clusters, "HDBSCAN should produce clusters on 10 well-separated blobs"
        assert elapsed < 3.0, (
            f"HDBSCAN on 500 nodes took {elapsed:.3f}s (budget 3.0s). "
            f"This suggests an O(N²) regression somewhere in the "
            f"recursive-level walk."
        )


# ---------------------------------------------------------------------------
# update_junction_summaries write throughput (supports NF4)
# ---------------------------------------------------------------------------


class TestStorageWritePerf:
    def test_update_1000_junction_summaries_under_500ms(self) -> None:
        """Batched write of 1000 summaries completes in < 500ms.

        update_junction_summaries is called twice per dream cycle
        (eager trunk, each level) and once per render (lazy). If it
        becomes slow, ingestion cost balloons. 1000 rows is a
        realistic upper bound for a ~5K-node corpus.
        """
        adapter = SQLiteAdapter(":memory:")
        # Pre-create 1000 centroid nodes.
        for i in range(1000):
            adapter.upsert_node(
                node_id=f"c_{i}",
                label=f"label_{i}",
                embedding=np.array([float(i), 0.0, 0.0], dtype=np.float32),
                is_centroid=True,
            )

        updates = {f"c_{i}": f"summary for centroid {i}, ~30 chars." for i in range(1000)}

        start = time.perf_counter()
        adapter.update_junction_summaries(updates)
        elapsed = time.perf_counter() - start

        # Verify round-trip.
        first = adapter.get_all_nodes()[0]
        assert first["paragraph_summary"] is not None
        assert elapsed < 0.5, (
            f"update_junction_summaries on 1000 rows took {elapsed:.3f}s "
            f"(budget 500ms). Check we're still using the single-"
            f"transaction pattern."
        )


# ---------------------------------------------------------------------------
# ASCII tree render throughput (supports NF1/NF2 render budgets)
# ---------------------------------------------------------------------------


class TestRenderPerf:
    def _build_subgraph(self, n_nodes: int, n_memories: int) -> dict:
        """80-node subgraph with paragraph summaries populated on half
        the centroids (the realistic cached-post-first-render state)."""
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        # First node is a root centroid; rest are children in a shallow
        # fan-out pattern: each centroid has ~4 children.
        for i in range(n_nodes):
            is_centroid = i < (n_nodes // 3)  # ~1/3 centroids
            nodes.append(
                {
                    "id": f"n{i}",
                    "label": f"label_{i}",
                    "is_centroid": is_centroid,
                    "embedding": [0.0, 0.0],
                    "paragraph_summary": (
                        f"Paragraph summary for node {i}. " * 3
                        if is_centroid and i % 2 == 0
                        else None
                    ),
                }
            )
            if i > 0:
                parent_idx = (i - 1) // 4
                edges.append(
                    {
                        "source_id": f"n{parent_idx}",
                        "target_id": f"n{i}",
                        "weight": 0.8,
                        "relationship": "child_of",
                    }
                )
        memories: list[dict[str, Any]] = []
        for m in range(n_memories):
            # Attach each memory to a leaf-ish node.
            target = f"n{n_nodes - 1 - (m % 10)}"
            memories.append(
                {
                    "id": f"mem_{m}",
                    "summary": f"Memory summary for memory {m}. " * 4,
                    "named_entities": [],
                }
            )
            edges.append(
                {
                    "source_id": f"mem_{m}",
                    "target_id": target,
                    "weight": 1.0,
                    "relationship": "has_tag",
                }
            )
        return {"nodes": nodes, "edges": edges, "memories": memories}

    def test_render_80_nodes_under_50ms(self) -> None:
        """Rendering an 80-node subgraph with mixed populated/unpopulated
        paragraph summaries completes in < 50ms. No LLM calls, no DB —
        pure string/dict work."""
        sub = self._build_subgraph(n_nodes=80, n_memories=20)

        start = time.perf_counter()
        result = _render_ascii_tree(sub)
        elapsed = time.perf_counter() - start

        assert result, "render should produce non-empty output"
        assert elapsed < 0.05, (
            f"_render_ascii_tree on 80 nodes took {elapsed * 1000:.1f}ms (budget 50ms)."
        )

    def test_enforce_node_cap_on_large_subgraph_under_100ms(self) -> None:
        """Capping a 500-node subgraph down to 80 via BFS + priority
        sort completes in < 100ms. BFS is O(E) on the child_of
        adjacency — a 500-node graph with ~500 edges should be trivial."""
        # Build a 500-node chain from the seed.
        nodes = [
            {
                "id": f"n{i}",
                "label": f"l{i}",
                "is_centroid": True,
                "embedding": [0.0, 0.0],
                "paragraph_summary": None,
            }
            for i in range(500)
        ]
        edges = [
            {
                "source_id": f"n{i}",
                "target_id": f"n{i + 1}",
                "weight": 1.0,
                "relationship": "child_of",
            }
            for i in range(499)
        ]
        sub = {"nodes": nodes, "edges": edges, "memories": []}

        start = time.perf_counter()
        _enforce_node_cap(sub, seed_ids=["n0"], cap=80)
        elapsed = time.perf_counter() - start

        assert len(sub["nodes"]) == 80
        assert elapsed < 0.1, (
            f"_enforce_node_cap on 500 nodes took {elapsed * 1000:.1f}ms (budget 100ms)."
        )


# ---------------------------------------------------------------------------
# Lazy-cache behaviour (supports NF2 warm-cache render)
# ---------------------------------------------------------------------------


class _LLMCallCounter(MockLLMProvider):
    """Mock LLM that records call counts for the summary + naming APIs."""

    def __init__(self) -> None:
        super().__init__()
        self.junction_batch_calls = 0
        self.junction_single_calls = 0
        self.name_batch_calls = 0

    def generate_junction_summaries_batch(self, groups: list[list[str]]) -> list[str]:
        self.junction_batch_calls += 1
        return [f"summary-{i}" for i in range(len(groups))]

    def generate_junction_summary(self, children_summaries: list[str]) -> str:
        self.junction_single_calls += 1
        return "single-summary"

    def generate_category_names_batch(self, pairs: list[list[str]]) -> list[str]:
        self.name_batch_calls += 1
        return [f"name{i}" for i in range(len(pairs))]


class TestLazyCachePerf:
    def _populated_system(self) -> tuple[NeuroMemory, _LLMCallCounter]:
        llm = _LLMCallCounter()
        system = NeuroMemory(
            storage=SQLiteAdapter(":memory:"),
            llm=llm,
            embedder=MockEmbeddingProvider(),
        )
        # Ingest enough text to produce a non-trivial graph.
        for phrase in [
            "alpha beta gamma",
            "delta epsilon zeta",
            "eta theta iota",
            "alpha zeta theta",
        ]:
            system.enqueue(phrase)
        system.force_dream(block=True)
        return system, llm

    def test_warm_cache_second_render_zero_llm_calls(self) -> None:
        """build_prompt_context called twice on the same query: second
        call makes ZERO new summary OR name batched LLM calls because
        everything is cached in storage (ADR-003 NF2)."""
        system, llm = self._populated_system()
        helper = ContextHelper(system)

        # First render warms the cache (eager trunk already populated
        # much of it during force_dream, but depth-3 might still hit
        # lazy summaries).
        first = helper.build_prompt_context("some query words")
        assert first, "first render should return content"
        warm_junction = llm.junction_batch_calls
        warm_name = llm.name_batch_calls

        # Second render of the same query → zero new LLM calls.
        second = helper.build_prompt_context("some query words")
        assert second, "second render should also return content"

        assert llm.junction_batch_calls == warm_junction, (
            f"second render made {llm.junction_batch_calls - warm_junction} "
            f"extra junction summary batch calls (expected 0 cache hits)"
        )
        assert llm.name_batch_calls == warm_name, (
            f"second render made {llm.name_batch_calls - warm_name} "
            f"extra centroid name batch calls (expected 0 cache hits)"
        )


# ---------------------------------------------------------------------------
# Floor-up ordering property (ADR-003 D2)
# ---------------------------------------------------------------------------


class _TrackingLLM(MockLLMProvider):
    """LLM that records the groups passed to each junction batch call."""

    def __init__(self) -> None:
        super().__init__()
        self.seen_groups: list[list[list[str]]] = []

    def generate_junction_summaries_batch(self, groups: list[list[str]]) -> list[str]:
        # Deep copy the groups so later mutation doesn't affect
        # recorded state.
        self.seen_groups.append([list(g) for g in groups])
        return [f"resolved-summary-{i}" for i in range(len(groups))]


class _PairClusterer(ClusteringProvider):
    """Deterministic clusterer that pairs input nodes and then pairs
    those centroids, producing a 2-level hierarchy we can use to test
    floor-up ordering."""

    def cluster(self, nodes: list[tuple[str, np.ndarray]]) -> list[Any]:
        from neuromem.providers import Cluster  # noqa: PLC0415

        if len(nodes) < 2:
            return []
        # Level 0: pair adjacent nodes.
        level0: list[Cluster] = []
        for i in range(0, len(nodes) - 1, 2):
            a_id, a_emb = nodes[i]
            b_id, b_emb = nodes[i + 1]
            level0.append(
                Cluster(
                    id=f"c0_{i // 2}",
                    embedding=(np.asarray(a_emb) + np.asarray(b_emb)) / 2.0,
                    child_ids=[a_id, b_id],
                    cohesion=0.9,
                )
            )
        out = list(level0)
        # Level 1: pair level-0 centroids.
        if len(level0) >= 2:
            for i in range(0, len(level0) - 1, 2):
                a, b = level0[i], level0[i + 1]
                out.append(
                    Cluster(
                        id=f"c1_{i // 2}",
                        embedding=(a.embedding + b.embedding) / 2.0,
                        child_ids=[a.id, b.id],
                        cohesion=0.8,
                    )
                )
        return out


@pytest.mark.skip(
    reason="Floor-up property is indirectly exercised by TestEagerTrunkSummarisation; "
    "a direct property test requires full control over eager-trunk "
    "serialisation which is easier to reason about in the benchmark."
)
class TestFloorUpOrdering:
    """Placeholder for a stricter floor-up property test. The
    existing tests in TestResolveJunctionSummaries cover happy path
    (level-0 before level-1) via the integration test harness. A
    tighter property test — "level-1 batch groups contain the
    just-written level-0 summary, not just a label" — is left for a
    follow-up when the level-1 boundary becomes observable without
    monkeypatching."""


# Silence unused-import lint: these types are imported for clarity and
# to anchor future property tests. numpy and pytest are obviously used.
_ = LLMProvider
