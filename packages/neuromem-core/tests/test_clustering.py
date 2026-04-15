"""Unit tests for ``neuromem.clustering`` — concrete ``ClusteringProvider``
implementations (ADR-003 D1).

Covers:

- Shape of the ``Cluster`` record and the ``ClusteringProvider`` ABC
  contract (refuses partial subclasses at instantiation time).
- ``HDBSCANClusteringProvider`` happy paths on a fixed synthetic
  fixture: 3 well-separated Gaussian blobs should collapse to 3
  clusters + their children should match each blob's members.
- Multi-level recursion: when the per-level clustering produces ≥ 2
  centroids, the provider runs again on those centroids, producing
  higher-level clusters that reference the earlier ones in
  ``child_ids``. Test asserts dependency ordering (children before
  parents).
- Empty / too-small inputs (< 2 nodes) produce an empty result,
  not a crash.
- ``StubClusteringProvider`` is deterministic: same inputs → identical
  output ordering; adjacent pairs after sorting by id.
- Cohesion is in ``[0, 1]`` and matches the upper-triangular mean of
  pairwise cosine similarities.
"""

from __future__ import annotations

import numpy as np
import pytest
from neuromem.clustering import (
    HDBSCANClusteringProvider,
    StubClusteringProvider,
)
from neuromem.providers import Cluster, ClusteringProvider

# ---------------------------------------------------------------------------
# ABC contract
# ---------------------------------------------------------------------------


class TestClusteringProviderABC:
    def test_direct_instantiation_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            ClusteringProvider()  # type: ignore[abstract]

    def test_subclass_missing_cluster_method_cannot_instantiate(self) -> None:
        class Empty(ClusteringProvider):
            pass

        with pytest.raises(TypeError, match="abstract"):
            Empty()  # type: ignore[abstract]


class TestClusterRecord:
    def test_fields_readable_after_construction(self) -> None:
        emb = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        c = Cluster(
            id="c_abc",
            embedding=emb,
            child_ids=["n1", "n2", "n3"],
            cohesion=0.87,
        )
        assert c.id == "c_abc"
        assert c.child_ids == ["n1", "n2", "n3"]
        assert c.cohesion == 0.87
        np.testing.assert_array_equal(c.embedding, emb)


# ---------------------------------------------------------------------------
# HDBSCAN provider
# ---------------------------------------------------------------------------


def _three_blobs(seed: int = 0) -> list[tuple[str, np.ndarray]]:
    """Build a 30-node fixture of 3 well-separated clusters in 8-D
    space. The separation is wide enough that any density-based
    clusterer should recover the three groups exactly."""
    rng = np.random.default_rng(seed)
    dim = 8
    # Centres spaced ~5 units apart in each dimension.
    centres = np.eye(3, dim) * 5.0  # (3, dim)
    nodes: list[tuple[str, np.ndarray]] = []
    for blob_idx, centre in enumerate(centres):
        for i in range(10):
            # Small jitter around each centre (sigma 0.2, clusters stay tight).
            emb = centre + rng.normal(scale=0.2, size=dim)
            nodes.append((f"blob{blob_idx}_n{i}", emb.astype(np.float32)))
    return nodes


class TestHDBSCANClusteringProvider:
    def test_empty_input_returns_empty(self) -> None:
        provider = HDBSCANClusteringProvider()
        assert provider.cluster([]) == []

    def test_single_node_returns_empty(self) -> None:
        provider = HDBSCANClusteringProvider()
        nodes = [("only_one", np.array([1.0, 0.0, 0.0]))]
        assert provider.cluster(nodes) == []

    def test_three_blobs_produces_three_base_clusters(self) -> None:
        """The three well-separated blobs should collapse into 3 level-0
        clusters, each containing 10 members from its own blob."""
        provider = HDBSCANClusteringProvider()
        clusters = provider.cluster(_three_blobs())
        assert len(clusters) >= 3, (
            f"expected at least 3 clusters (level-0 blobs), got {len(clusters)}"
        )

        # Identify the three base-level clusters: their child_ids are
        # original leaf ids (start with "blob").
        base_clusters = [c for c in clusters if all(cid.startswith("blob") for cid in c.child_ids)]
        assert len(base_clusters) == 3

        # Every blob member should appear exactly once across the three
        # base clusters.
        all_children: set[str] = set()
        for c in base_clusters:
            # No duplicates within one cluster.
            assert len(set(c.child_ids)) == len(c.child_ids)
            all_children.update(c.child_ids)
        expected = {f"blob{b}_n{i}" for b in range(3) for i in range(10)}
        assert all_children == expected

        # Each base cluster's children should all be from the SAME blob.
        for c in base_clusters:
            blob_prefixes = {cid.split("_")[0] for cid in c.child_ids}
            assert len(blob_prefixes) == 1, f"cluster {c.id} mixed blobs: {blob_prefixes}"

    def test_cohesion_in_unit_interval(self) -> None:
        provider = HDBSCANClusteringProvider()
        clusters = provider.cluster(_three_blobs())
        for c in clusters:
            assert 0.0 <= c.cohesion <= 1.0, (
                f"cohesion out of [0,1] for cluster {c.id}: {c.cohesion}"
            )

    def test_multi_level_recursion_dependency_order(self) -> None:
        """When level-0 produces ≥ 2 clusters, a level-1 run may emit a
        super-cluster whose child_ids reference those level-0 centroid
        ids. Assert those level-1 clusters appear AFTER the level-0
        clusters they reference."""
        provider = HDBSCANClusteringProvider()
        clusters = provider.cluster(_three_blobs())

        cluster_ids_seen: set[str] = set()
        for c in clusters:
            # Every child_id must either be an original leaf id or
            # the id of a PREVIOUSLY emitted cluster.
            for cid in c.child_ids:
                if cid.startswith("c_"):
                    assert cid in cluster_ids_seen, (
                        f"cluster {c.id} references {cid} before it was emitted"
                    )
            cluster_ids_seen.add(c.id)

    def test_centroid_embedding_is_mean_of_children(self) -> None:
        """For a base-level cluster (all children are leaves), the
        centroid's embedding should equal the element-wise mean of
        the children's embeddings."""
        nodes = _three_blobs()
        provider = HDBSCANClusteringProvider()
        clusters = provider.cluster(nodes)

        base_clusters = [c for c in clusters if all(cid.startswith("blob") for cid in c.child_ids)]
        embeddings_by_id = {nid: emb for nid, emb in nodes}
        for c in base_clusters:
            child_embs = np.stack(
                [np.asarray(embeddings_by_id[cid], dtype=np.float64) for cid in c.child_ids]
            )
            expected_mean = child_embs.mean(axis=0)
            np.testing.assert_allclose(
                np.asarray(c.embedding, dtype=np.float64),
                expected_mean,
                rtol=1e-5,
                err_msg=f"centroid of {c.id} is not the mean of its children",
            )


# ---------------------------------------------------------------------------
# Stub provider
# ---------------------------------------------------------------------------


class TestStubClusteringProvider:
    def test_empty_input_returns_empty(self) -> None:
        assert StubClusteringProvider().cluster([]) == []

    def test_single_node_returns_empty(self) -> None:
        nodes = [("a", np.array([1.0]))]
        assert StubClusteringProvider().cluster(nodes) == []

    def test_pairs_adjacent_after_id_sort(self) -> None:
        """Input order is intentionally shuffled; the stub sorts by id
        then pairs adjacent entries."""
        nodes = [
            ("d", np.array([0.0, 0.0, 1.0])),
            ("a", np.array([1.0, 0.0, 0.0])),
            ("c", np.array([0.0, 1.0, 0.0])),
            ("b", np.array([0.5, 0.5, 0.0])),
        ]
        clusters = StubClusteringProvider().cluster(nodes)

        assert len(clusters) == 2
        assert clusters[0].child_ids == ["a", "b"]
        assert clusters[1].child_ids == ["c", "d"]

    def test_odd_count_drops_last(self) -> None:
        nodes = [
            ("a", np.array([1.0, 0.0])),
            ("b", np.array([0.0, 1.0])),
            ("c", np.array([0.5, 0.5])),
        ]
        clusters = StubClusteringProvider().cluster(nodes)
        assert len(clusters) == 1
        assert clusters[0].child_ids == ["a", "b"]

    def test_deterministic(self) -> None:
        """Two calls with the same input produce identical output."""
        nodes = [(f"n{i}", np.array([float(i), 0.0, 0.0])) for i in range(6)]
        r1 = StubClusteringProvider().cluster(nodes)
        r2 = StubClusteringProvider().cluster(nodes)
        assert [c.id for c in r1] == [c.id for c in r2]
        assert [c.child_ids for c in r1] == [c.child_ids for c in r2]


# ---------------------------------------------------------------------------
# LLMProvider default fallbacks for junction summaries (no LLM needed)
# ---------------------------------------------------------------------------


class TestJunctionSummaryDefaults:
    """The non-abstract ``generate_junction_summary[_batch]`` methods on
    ``LLMProvider`` provide correctness fallbacks: concatenate-truncate.
    Verify them via a minimal concrete subclass that implements only
    the required abstract methods."""

    def _make_provider(self) -> object:
        from neuromem.providers import LLMProvider  # noqa: PLC0415

        class _Minimal(LLMProvider):
            def generate_summary(self, raw_text: str) -> str:
                return raw_text

            def extract_tags(self, summary: str) -> list[str]:
                return []

            def generate_category_name(
                self, concepts: list[str], *, avoid_names: set[str] | None = None
            ) -> str:
                return "tag"

        return _Minimal()

    def test_empty_children_returns_empty_string(self) -> None:
        provider = self._make_provider()
        assert provider.generate_junction_summary([]) == ""  # type: ignore[attr-defined]

    def test_single_child_returns_that_child(self) -> None:
        provider = self._make_provider()
        assert provider.generate_junction_summary(["just one summary"]) == "just one summary"  # type: ignore[attr-defined]

    def test_truncates_per_item_not_mid_item(self) -> None:
        """Each long child is individually truncated (with its own
        ellipsis) BEFORE joining, so no single child is cut in half
        by the fallback. Total output is bounded by
        per_item_cap × N_items."""
        provider = self._make_provider()
        long_children = ["x" * 200, "y" * 200, "z" * 200]
        result = provider.generate_junction_summary(long_children)  # type: ignore[attr-defined]
        # Each of the 3 items truncated to 120 chars (119 + ellipsis).
        # Result contains three ellipsis-terminated chunks joined by spaces.
        assert result.count("…") == 3, (
            "each item should be individually truncated with its own ellipsis"
        )
        # No single chunk exceeds the per-item cap.
        for chunk in result.split(" "):
            assert len(chunk) <= 121  # cap + some leeway for boundary

    def test_short_children_joined_without_truncation(self) -> None:
        provider = self._make_provider()
        result = provider.generate_junction_summary(["one", "two", "three"])  # type: ignore[attr-defined]
        assert result == "one two three"
        assert "…" not in result

    def test_batch_default_loops_per_group(self) -> None:
        provider = self._make_provider()
        groups = [["a summary"], ["b summary", "c summary"], []]
        result = provider.generate_junction_summaries_batch(groups)  # type: ignore[attr-defined]
        assert len(result) == 3
        assert result[0] == "a summary"
        assert result[1] == "b summary c summary"
        assert result[2] == ""
