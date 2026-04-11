# ADR-001: Hand-rolled numpy clustering vs scipy / sklearn / HDBSCAN

**Status**: Accepted
**Date**: 2026-04-11
**Deciders**: Tim Richardson (author), `feature-dev:code-reviewer` agent
**Context**: T020 — `NeuroMemory._run_clustering` agglomerative clustering loop

## Context

The dreaming pipeline (`NeuroMemory._run_dream_cycle`) needs to agglomerate
newly-extracted tag labels into a hierarchy of centroid "parent concept" nodes.
The algorithm is **greedy bottom-up merging**: find the closest pair of nodes
by cosine similarity, merge them into a centroid (with element-wise mean
embedding), name the centroid via an LLM call, record the `child_of` edges, and
repeat until no remaining pair exceeds `cluster_threshold` (default 0.82).

The question: should this clustering loop be implemented **by hand in pure
numpy**, or should the project import an established clustering library?

## Constraints

Four non-negotiables frame the decision:

1. **Constitution v2.0.0 Principle II — Lean Dependency Set.** Runtime deps are
   the Python 3.10+ standard library plus `numpy` plus `pandas`. Any additional
   runtime dependency requires a MINOR constitutional amendment with a
   documented justification.

2. **Per-merge LLM callback.** The centroid-naming step
   (`llm.generate_category_name(labels)`) is called **at each merge in the
   loop**, not as a separate post-processing pass. The centroid's name feeds
   into later merges — a centroid can itself merge into a grander centroid on
   the next iteration.

3. **Threshold-driven stop condition.** Clustering halts when no candidate pair
   exceeds the cosine threshold. This is **not** "cut the dendrogram at K
   clusters" and **not** "merge until a density criterion is met" — it is a
   direct threshold on the pairwise similarity score.

4. **Scale.** The dreaming cycle typically clusters a few dozen to a few
   hundred tag labels per invocation. The spec caps `neuromem-core` at ~10,000
   total nodes before recommending a vector-native adapter.

## Decision

**Hand-roll the clustering loop in numpy.** The implementation lives in
`packages/neuromem-core/src/neuromem/system.py::_run_clustering`:

1. Build a pairwise cosine-similarity matrix via `(normed @ normed.T)` — a
   single BLAS matmul.
2. `np.argmax` for the highest-similarity pair, clamp by `cluster_threshold`.
3. Merge via `compute_centroid` from `neuromem.vectors`.
4. Call `llm.generate_category_name` + `_sanitise_category_name` inside the
   loop to name the new centroid.
5. `upsert_node(centroid, is_centroid=True)` + two `insert_edge(... 'child_of')`
   calls for the parent-child links.
6. Retire both members from the `alive` list and append the new centroid as a
   fresh candidate — multi-level hierarchies fall out of the loop naturally.
7. Repeat until no pair exceeds the threshold or only one node remains.

Total implementation: ~100 lines of Python, plus a 6-line
`_sanitise_category_name` helper for the "one word per LLM response" contract.

## Alternatives Considered

### Option A — `scipy.cluster.hierarchy.linkage`

Full academic-grade hierarchical clustering with single, complete, average, and
Ward linkage options.

**Pros**: robust, well-tested, O(k² log k) instead of the hand-rolled
O(k²·merges). Ward linkage in particular produces empirically tighter clusters
on well-separated data.

**Cons rejected for v1**:

- Pulls in `scipy` (~35 MB wheel), which brings LAPACK/BLAS bindings and Cython
  extensions. Non-trivial install surface and transitive CVE risk.
- `linkage` produces a full dendrogram in one call and then requires a second
  pass (`fcluster`) to cut it. The **per-merge LLM callback doesn't fit** —
  you'd either (a) run linkage to completion and then walk the tree making LLM
  calls (losing the early-stop "merge while above threshold" property and the
  ability to influence subsequent merges with the LLM-generated name), or
  (b) re-implement the linkage algorithm yourself to insert the callback,
  which defeats the point of importing scipy in the first place.
- Scipy's `linkage` + `fcluster` doesn't natively support "cut at similarity
  threshold" — it supports "cut at distance threshold" (the inverse). Adapter
  arithmetic adds confusion for marginal benefit.

### Option B — `sklearn.cluster.AgglomerativeClustering`

Same capabilities as scipy's linkage with a slightly more modern API.

**Cons rejected for v1**: everything in Option A, plus sklearn pulls `scipy` +
`joblib` + `threadpoolctl` + several smaller transitive deps (~50 MB total
install). Biggest dependency footprint of the three library options for zero
additional capability that our use case needs.

### Option C — `hdbscan`

Density-based hierarchical clustering, widely used for clustering noisy
real-world embedding data.

**Cons rejected for v1**:

- Produces a **flat partition** with an embedded hierarchy that still has to be
  walked to extract parent-child relationships for our concept graph.
- The HDBSCAN value proposition is **discovering the right number of clusters
  without a threshold** — but we explicitly *want* a threshold-driven stop
  condition. Using HDBSCAN here would be using the right tool for the wrong
  problem.
- Heaviest transitive dep of the three options (pulls scipy + Cython-compiled
  C extensions + scikit-learn-style validation).
- Per-merge LLM callback fits even worse than scipy — HDBSCAN's internal data
  structures are opaque and its single-linkage tree isn't directly exposed for
  insertion of named-centroid callbacks.

### Option D — Pure-Python `math` loops (no numpy)

Rejected by the v1.1.0 → v2.0.0 constitution bump. numpy is now a mandatory
runtime dependency, and doing vector math in pure-Python loops was the
"ideology, not engineering" pattern v2.0.0 explicitly corrected.

## Consequences

### Positive

- **Zero new runtime dependencies.** `neuromem-core`'s runtime surface stays at
  exactly `numpy + pandas + stdlib`, satisfying Principle II.
- **Per-merge LLM integration is trivially clean** — it's just a function call
  inside the loop body, no callback plumbing required.
- **Full algorithm in ~100 lines**, readable top to bottom without having to
  cross-reference library documentation. Every branch is visible in one file.
- **Performance is comfortable inside the spec envelope.** At `k=500, D=1536`,
  the pairwise similarity matrix is a single BLAS matmul (~5 ms) and the inner
  loop finds argmax in microseconds. The dominant cost is the LLM call for
  centroid naming — which no library would avoid.

### Negative

- **At `k > ~5000`, the dense O(k²) matrix starts to hurt.** Scipy's
  `linkage` is faster in practice at that scale because it uses the
  nearest-neighbour chain algorithm. Not a concern inside the 10,000-node v1
  cap, but real.
- **No Ward linkage** (minimise variance increase on merge) or other linkage
  variants. The current algorithm is closer to "mean-linkage with centroid
  representatives": each merge produces a centroid used for subsequent
  comparisons. Cluster quality may be lower on tricky data than what Ward
  would produce.
- **No soft clustering** — a node belonging partially to multiple centroids is
  not supported. Once a node is merged, it is retired from the `alive` set.
- **No noise handling.** HDBSCAN explicitly labels points as noise when they
  don't fit any dense region; our greedy loop just leaves them as orphan leaf
  tags without a centroid parent, which is arguably the right behaviour but is
  worth noting.

### Revisit criteria

Swap to a library-backed clustering algorithm when **any one of** these
is true:

1. `k > 5000` in the dreaming cycle causes measurable latency regressions
   against the SC-003 performance target (~500 ms for `search_memory` at
   10,000 nodes). Unlikely inside v1 scope but worth monitoring.
2. A LongMemEval / GraphRAG-Bench / AMB evaluation run (planned in T033 polish)
   shows measurably degraded recall quality compared to a library baseline,
   **and** the quality gap is attributable to cluster quality rather than some
   other factor in the pipeline.
3. A downstream framework-wrapper package has a concrete use case that
   requires Ward linkage, soft clustering, or noise labelling that the current
   threshold-merge approach cannot satisfy.

Swapping would require, in order:

1. A **constitutional amendment** bumping Principle II's runtime dep list
   (MINOR bump, e.g., `scipy>=1.13` added to the whitelist). See
   `.specify/memory/constitution.md` for the amendment procedure.
2. A **research.md supersession entry** (follow-up to Decision 3) noting the
   trade-off and the evidence that drove the swap.
3. A **rewrite of `_run_clustering`** to integrate the new library's API while
   preserving the per-merge LLM callback — or, if the callback is no longer
   needed (e.g., because centroid naming moves to a post-processing pass),
   document that design change explicitly.

## References

- Implementation: `packages/neuromem-core/src/neuromem/system.py::_run_clustering`
- Tests: `packages/neuromem-core/tests/test_system.py::TestAgglomerativeClustering`
- Original decision brief: `specs/001-neuromem-core/research.md` §Decision 3
- Pipeline context: `specs/001-neuromem-core/spec.md` §Phase B (Consolidation)
- Dependency governance: `.specify/memory/constitution.md` §Principle II
