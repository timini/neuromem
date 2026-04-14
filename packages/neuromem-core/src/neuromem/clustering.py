"""Concrete ``ClusteringProvider`` implementations (ADR-003 D1).

Ships two providers:

- :class:`HDBSCANClusteringProvider` (default, recommended) — density-
  based hierarchical clustering via the ``hdbscan`` library. Produces
  natural-fanout centroids (2-N children per cluster) and correctly
  models "noise points" that don't belong in any cluster.

- :class:`StubClusteringProvider` — deterministic pair-wise grouper
  that sorts inputs by id and merges adjacent pairs. Used in unit
  tests so cluster-dependent assertions don't depend on HDBSCAN's
  internal randomness or version. NOT suitable for production.

The provider abstraction lives in :mod:`neuromem.providers`; the
implementations here import it. This keeps `providers.py` focused on
abstract contracts (no ``hdbscan`` import there, so type-only
introspection doesn't pull the heavy dep).
"""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .providers import Cluster, ClusteringProvider

# Minimum cluster size for HDBSCAN. Two is the smallest group that can
# meaningfully be called a cluster; with few tags in a dream cycle we
# want every natural pair to surface rather than be called noise.
_DEFAULT_MIN_CLUSTER_SIZE = 2

# Recursion stop: if a level produces < 2 clusters, stop (nothing to
# merge at the next level up).
_MIN_CLUSTERS_TO_RECURSE = 2

# Safety cap on recursion depth. HDBSCAN levels typically compress 3x
# per step, so depth 8 covers up to ~6500 leaves — well above the
# ~10k-node SC-003 target.
_MAX_RECURSION_DEPTH = 8


def _normalise_rows(embeddings: NDArray[np.floating]) -> NDArray[np.floating]:
    """L2-normalise each row so euclidean distance ≈ angular distance."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    safe = np.where(norms > 0.0, norms, 1.0)
    return embeddings / safe


def _mean_pairwise_cosine(embeddings: NDArray[np.floating]) -> float:
    """Mean upper-triangular cosine similarity among embeddings.

    Used as the ``cohesion`` field on :class:`Cluster`. Returns ``1.0``
    for a single-row matrix (degenerate but harmless).
    """
    if embeddings.shape[0] < 2:
        return 1.0
    normed = _normalise_rows(embeddings.astype(np.float64))
    sim = normed @ normed.T
    # Upper triangle excluding diagonal.
    n = sim.shape[0]
    iu = np.triu_indices(n, k=1)
    return float(np.mean(sim[iu]))


class HDBSCANClusteringProvider(ClusteringProvider):
    """Default clustering provider. Uses the ``hdbscan`` library
    recursively to build a multi-level hierarchy from raw embeddings.

    At each recursion level:

    1. L2-normalise the input embeddings so HDBSCAN's Euclidean metric
       is equivalent to angular distance (preserving the cosine-based
       semantics we use everywhere else).
    2. Run HDBSCAN with ``min_cluster_size=2`` — everything smaller
       than a pair of tags is treated as noise and not clustered.
    3. For each resulting cluster, emit a :class:`Cluster` whose
       ``embedding`` is the mean of its members and whose
       ``child_ids`` are the members' ids.
    4. If at least two clusters were produced, recurse on the
       centroids as the next level's input.

    Noise points (HDBSCAN label -1) are left as-is at the current
    level: they stay addressable via their original ids and will
    render as top-level leaves.

    The ``random_state`` constructor arg makes HDBSCAN deterministic
    across runs (used by tests and for benchmark reproducibility).
    """

    def __init__(
        self,
        *,
        min_cluster_size: int = _DEFAULT_MIN_CLUSTER_SIZE,
        random_state: int | None = 42,
    ) -> None:
        if min_cluster_size < 2:
            raise ValueError(
                f"min_cluster_size must be >= 2 (clusters below that are not "
                f"semantically meaningful), got {min_cluster_size}"
            )
        self._min_cluster_size = int(min_cluster_size)
        self._random_state = random_state

    def cluster(
        self,
        nodes: list[tuple[str, NDArray[np.floating]]],
    ) -> list[Cluster]:
        if len(nodes) < 2:
            return []
        out: list[Cluster] = []
        self._cluster_level(nodes, depth=0, out=out)
        return out

    def _cluster_level(
        self,
        nodes: list[tuple[str, NDArray[np.floating]]],
        *,
        depth: int,
        out: list[Cluster],
    ) -> None:
        """Run one HDBSCAN pass, emit centroids, recurse on centroids.

        ``out`` accumulates the dependency-ordered cluster list.
        Children appear before their parents because each level's
        centroids are emitted BEFORE the recursive call that might
        produce clusters referencing them.
        """
        if depth >= _MAX_RECURSION_DEPTH:
            return
        if len(nodes) < self._min_cluster_size:
            return

        ids = [nid for nid, _ in nodes]
        embeddings = np.stack([np.asarray(emb, dtype=np.float64) for _, emb in nodes])
        normed = _normalise_rows(embeddings)

        labels = self._run_hdbscan(normed)
        # Group by cluster label, skipping -1 (noise).
        clusters_by_label: dict[int, list[int]] = {}
        for idx, lbl in enumerate(labels):
            if lbl < 0:
                continue
            clusters_by_label.setdefault(int(lbl), []).append(idx)

        if not clusters_by_label:
            return

        # Emit one Cluster per group. Record (new_id, embedding) so we
        # can recurse on the centroids.
        next_level: list[tuple[str, NDArray[np.floating]]] = []
        for member_indices in clusters_by_label.values():
            member_ids = [ids[i] for i in member_indices]
            member_embs = embeddings[member_indices]
            centroid_emb = member_embs.mean(axis=0)
            cohesion = _mean_pairwise_cosine(member_embs)
            cluster_id = f"c_{uuid.uuid4().hex[:16]}"
            out.append(
                Cluster(
                    id=cluster_id,
                    embedding=centroid_emb,
                    child_ids=member_ids,
                    cohesion=cohesion,
                )
            )
            next_level.append((cluster_id, centroid_emb))

        # Only recurse if we actually have enough centroids to cluster
        # further. A single cluster at this level is already the root.
        if len(next_level) >= _MIN_CLUSTERS_TO_RECURSE:
            self._cluster_level(next_level, depth=depth + 1, out=out)

    def _run_hdbscan(self, normed_embeddings: NDArray[np.floating]) -> Any:
        """Invoke HDBSCAN and return the fit_predict cluster labels.

        Isolated into a method so unit tests can monkeypatch it when
        they want to test the provider's hierarchy-assembly logic
        without exercising HDBSCAN itself.
        """
        # Import here to keep module-import cost low for callers that
        # only need the abstract types — though `hdbscan` is now a
        # mandatory runtime dep, its import triggers numpy/scipy/
        # scikit-learn pulls that add ~100ms on first use.
        import hdbscan  # noqa: PLC0415

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self._min_cluster_size,
            metric="euclidean",  # on L2-normalised inputs ≈ angular distance
            allow_single_cluster=True,
            # Note: `hdbscan.HDBSCAN` does not accept `random_state`
            # directly — its clustering is deterministic given inputs.
            # random_state only matters if we switch to approximations.
        )
        return clusterer.fit_predict(normed_embeddings)


class StubClusteringProvider(ClusteringProvider):
    """Deterministic pair-wise clustering for unit tests.

    Sorts input nodes by id, pairs adjacent entries, and emits one
    :class:`Cluster` per pair (element-wise mean embedding). Single
    level only — does NOT produce a multi-level hierarchy. Used where
    tests need a predictable cluster output regardless of embedding
    values or HDBSCAN version quirks.

    Not exported for production use; don't inject this into a real
    ``NeuroMemory``. The test suite imports it directly.
    """

    def cluster(
        self,
        nodes: list[tuple[str, NDArray[np.floating]]],
    ) -> list[Cluster]:
        if len(nodes) < 2:
            return []
        ordered = sorted(nodes, key=lambda pair: pair[0])
        out: list[Cluster] = []
        for i in range(0, len(ordered) - 1, 2):
            a_id, a_emb = ordered[i]
            b_id, b_emb = ordered[i + 1]
            a = np.asarray(a_emb, dtype=np.float64)
            b = np.asarray(b_emb, dtype=np.float64)
            centroid = (a + b) / 2.0
            cohesion = _mean_pairwise_cosine(np.stack([a, b]))
            out.append(
                Cluster(
                    id=f"stub_c_{i // 2}",
                    embedding=centroid,
                    child_ids=[a_id, b_id],
                    cohesion=cohesion,
                )
            )
        return out
