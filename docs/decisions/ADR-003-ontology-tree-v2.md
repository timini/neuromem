# ADR-003: Ontology Tree v2 — HDBSCAN clustering + per-junction summaries + `expand_node`

**Status**: Accepted
**Date**: 2026-04-14
**Deciders**: Tim Richardson + Claude
**Supersedes**: ADR-001 (hand-rolled numpy clustering), in part — the clustering algorithm choice is revisited.
**Extends**: ADR-002 (lazy centroid naming) — the same lazy-cache pattern is applied to a new per-junction paragraph summary field.

## Context

Three empirical observations on the n=20 LongMemEval-s run (mean 0.950, recorded in `docs/benchmarks/longmemeval-s-neuromem-n20-topk20.jsonl`) drove this reconsideration:

1. **The rendered tree has meaningless deep chains.** Binary agglomerative merges produce hierarchies like `guidance → inquiry → presentation → learning → measurement → degree → mem_xxx`. Each "middle" centroid has exactly one sibling at its own level and its LLM-generated label is forced to abstract over only two things, which rarely yields a real concept. The user flagged this directly ("this tree looks v strange to me what the hell is it about?").
2. **Centroids have nothing but a one-word label.** The answering LLM sees a folder name and has to guess what's inside — or click through via `retrieve_memories` one memory at a time. A paragraph summary per junction would let the LLM skim the tree and decide which branches are worth drilling into, without fetching raw content for everything.
3. **Render is a one-shot, one-depth pull.** `build_prompt_context` returns a depth-2 subgraph and the agent can't ask to expand a specific interesting branch without re-running the whole pipeline. For per-turn usage inside a conversation, we want the agent to explore selectively.

ADR-001 chose a hand-rolled numpy clustering loop because it needed a per-merge LLM callback (for naming). ADR-002 eliminated that callback entirely by moving naming to render time. With the callback gone, **the main constraint that justified hand-rolling no longer applies**, and we can adopt a real clustering algorithm that addresses observation #1.

The ontology-v2 rework also introduces two new capabilities addressing observations #2 and #3: per-junction paragraph summaries (cached like ADR-002's labels) and a new `expand_node` agent tool.

## Decisions

### D1 — Non-binary clustering via HDBSCAN

The dream cycle's clustering step is rewritten to produce **natural-group centroids with 2–N children** (N typically 2–8, hard-capped at 20). The default implementation uses **HDBSCAN** (`hdbscan` on PyPI), which is purpose-built for varying-density cluster discovery and handles the "some tags just don't cluster with anything" noise-point case cleanly (they stay as top-level leaves rather than being force-merged).

`hdbscan` becomes a first-class runtime dependency of `neuromem-core` alongside `numpy` and `pandas`. Constitution v2.0.0 Principle II is read as "minimal deps, not zero deps" — the current binary-clustering mess is direct evidence that rolling our own was a mistake once the per-merge-callback constraint was removed.

To keep testing and future experimentation clean, we introduce a `ClusteringProvider` abstraction alongside `LLMProvider` / `EmbeddingProvider` / `StorageAdapter`. The default and recommended impl is `HDBSCANClusteringProvider`. Tests use a deterministic stub (`StubClusteringProvider`) that produces a known cluster layout so cluster-dependent assertions don't depend on HDBSCAN's internal randomness or version.

### D2 — Per-junction paragraph summary, hybrid caching

Each centroid node gains a new `paragraph_summary TEXT` column (additive SQLite migration, idempotent). The summary is 2–4 sentences summarising the union of all descendant memories and is computed by an LLM call over the children's own summaries (leaves' "summary" being each member memory's `summary`; internal nodes' being their own `paragraph_summary`).

**Caching is hybrid:**
- **Eager at dream time** for the *trunk* of the clustering hierarchy — all centroids whose immediate children include at least one memory (level 0) plus those centroids' immediate parents (level 1). One batched LLM call per level, mirroring `extract_tags_batch`.
- **Lazy at render time** for deeper internal centroids. When `build_prompt_context` or `expand_node` renders a centroid whose `paragraph_summary` is NULL, a new `NeuroMemory.resolve_junction_summaries(nodes)` method batches one LLM call for all missing summaries in the rendered subgraph, persists via a new `StorageAdapter.update_junction_summaries(updates)` batch-write method, and mutates the nodes list in place for immediate use. The pattern mirrors `resolve_centroid_names` from ADR-002 exactly.

**Fallback on LLM failure:** truncated concatenation of child summaries (first 300 chars). Persisted identically so the fallback doesn't retry on every render. Render NEVER blocks on LLM availability — same contract as ADR-002.

### D3 — Render at depth 3 with an 80-node cap

`ContextHelper.build_prompt_context` defaults change from `depth=2` to `depth=3`. The depth-3 subgraph BFS is bounded by a hard cap of **80 rendered nodes**: when a walk would exceed the cap, nodes are dropped in reverse-priority order (the originally-retrieved `top_k` seed nodes are never dropped; among non-seeds, nodes farther from the closest seed are dropped first). The cap is chosen to stay well inside Gemini Flash's prompt window with 80 nodes × ~400 chars of tree-row text ≈ 32 KB.

### D4 — `expand_node(node_id, depth=2)` agent tool

A new tool alongside `search_memory` and `retrieve_memories`. Returns an ASCII subtree rooted at `node_id` at the given depth, rendered identically to `build_prompt_context` output (paragraph summaries included, lazy-cached on first touch). The motivating use case: the answering LLM sees a centroid labelled "degree" in the top-level tree, wants to know what's in it, and calls `expand_node("node_id_of_degree")` rather than re-issuing a wider `search_memory` query. Unlike `retrieve_memories`, `expand_node` **does not spike `access_weight`** — browsing the ontology is a different action from recalling a specific memory.

## Consequences

### Positive

- Binary-tower deep chains disappear. Meaningful fanout (2–8 children typically) + meaningful labels over those groups = trees the LLM can actually reason about at a glance.
- Paragraph summaries let the LLM skim before drilling. Lower expected tool-call count per answer (fewer `retrieve_memories` calls to figure out what a branch contains).
- `expand_node` gives the agent a real "zoom in" primitive without re-running retrieval.
- The provider-boundary pattern (`ClusteringProvider`) matches the rest of the codebase; local-LLM or alternative-algorithm swaps later remain cheap.

### Negative / trade-offs

- New runtime dep (`hdbscan`). Explicit Constitution Principle II reinterpretation. Documented in `packages/neuromem-core/README.md`.
- Additional LLM cost at dream time (trunk summarisation) and at first render of deep nodes. Bounded by batching + caching; NF4 caps at ≤ 2× baseline dream-cycle wall.
- Storage size grows by ~400 bytes per centroid. Negligible for `neuromem-core`'s ~10K-node target.
- HDBSCAN's non-determinism under clustering parameters could produce different trees across runs. Mitigated by (a) `StubClusteringProvider` in tests, (b) HDBSCAN's own `random_state` knob, (c) the lazy-name cache meaning different trees across runs still share label work at render time.

## Alternatives considered (rejected)

### Clustering

- **Average-linkage agglomerative with natural-cut (pure numpy).** Keeps Constitution II "zero new deps" literally, fits the existing dream-cycle shape. Rejected because HDBSCAN does the same job better with well-tested code, and the dep cost is low. Kept as an optional fallback pattern but not the default.
- **Per-level k-means with silhouette-chosen k.** Clean fanout, pure numpy. Rejected because k-means forces every point into a cluster; noise points are a real case in the memory graph (a one-off mention isn't usefully clustered) and HDBSCAN's noise-point output models this correctly.
- **Keep the existing binary agglomerative.** Rejected per observation #1 — it's the direct cause of the meaningless deep chains.

### Paragraph-summary timing

- **All eager at dream time.** Cleanest invariant (every centroid always has a summary) but maximises dream-cycle cost even when the query will never touch most of the graph. The hybrid captures the eager-invariant benefit for the most-frequently-read nodes while keeping ingestion bounded.
- **All lazy at render time** (same pattern as ADR-002 labels). Cheapest ingestion but the first render of any previously-unseen query pays for a lot of cold summarisation. The hybrid's trunk-eager phase pre-warms precisely the nodes most queries will pass through.

### Agent tool shape

- **`expand_subgraph(memory_id)`** — expand outward from a memory rather than a centroid. Rejected as redundant: the memory's parent centroid is already reachable via the tree the LLM is reading, and expanding from a centroid is strictly more useful (it shows siblings).
- **`expand_node` triggers LTP on all contained memories.** Rejected because browsing ≠ recall. LTP should fire on explicit `retrieve_memories` (deliberate recall of a specific memory), not on "walk around in the ontology".

## Revisit when

- HDBSCAN turns out to be too slow or its non-determinism bites us on benchmark reproducibility. First mitigation is a fixed `random_state`; second is switching the default provider to the pure-numpy agglomerative fallback.
- The 80-node render cap routinely clips content the LLM needs. First mitigation is a token-budget-based cap; second is a `depth` or `top_k` knob tweak.
- Dream-cycle cost exceeds the 2× NF4 budget — first mitigation is tightening which levels are eagerly summarised (maybe just level 0), second is moving all summarisation to lazy.

## Interplay with prior ADRs

- **ADR-001** (hand-rolled numpy clustering): superseded on the choice of algorithm. Its per-merge-LLM-callback constraint no longer holds (ADR-002 eliminated it). The "no scipy, no sklearn" arguments still hold for those specific deps; `hdbscan` has only numpy/scipy/scikit-learn in its own tree and is judged worth the cost directly, not via a transitive-scipy argument.
- **ADR-002** (lazy centroid naming): the label-naming design is preserved as-is. The paragraph-summary design in D2 is the same pattern applied to a second field (label + paragraph = two lazy-cached fields on every centroid). `resolve_centroid_names` and `resolve_junction_summaries` are sibling methods with the same shape. The Constitution II discussion in ADR-002 stays correct for label naming (no new deps); the new D1 decision adds `hdbscan` specifically with its own justification.
