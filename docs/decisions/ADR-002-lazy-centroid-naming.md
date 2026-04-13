# ADR-002: Lazy LLM naming for cluster centroids

**Status**: Accepted
**Date**: 2026-04-13
**Deciders**: Tim Richardson (author), Claude Opus 4.6
**Context**: `NeuroMemory._run_dream_cycle` agglomerative clustering wall-time bottleneck

## Context

A 1-instance LongMemEval_s benchmark on `gemini-flash-lite-latest` recently took **22 minutes** for a single 550-memory instance. Phase-marker logging showed **95% of the wall time** was inside `_run_clustering`: **1188 sequential `LLMProvider.generate_category_name` calls** at ~1 second each, naming every centroid produced by greedy agglomerative merging.

Closer inspection revealed an asymmetry that makes the cost almost entirely wasted:

- `ContextHelper.build_prompt_context` already does **vector search → subgraph → render**. It calls `storage.get_nearest_nodes(query, top_k=5)` and walks `depth=2` hops outward, then renders only that subgraph. Typical render: 10–30 centroids appear in the tree shown to the answering LLM.
- The other ~1100 centroids in the graph are **never read by anything**. We named them at clustering time, paid 1+ second of API latency each, and the names sat unused in storage.

The fundamental design question was: should the labels exist eagerly, lazily, or not at all?

## Constraints

The decision is bounded by four non-negotiables:

1. **Centroid labels are NOT cosmetic.** They encode the cognitive loop's levels of abstraction. The rendered tree shows the answering LLM a hierarchy like `📁 retail → 📁 groceries → 📁 coupons → 📄 mem_xxx`, and that hierarchy is the *whole point* of having neocortex-style consolidation above raw hippocampal traces. Any design that produces semantically-empty labels at the abstraction layers (e.g. `cluster_42` everywhere) defeats the cognitive-loop architecture the project is built around.

2. **Constitution v2.0.0 Principle II — Lean Dependency Set.** No new runtime deps. Numpy is already mandatory. The fallback path must use what's already in the workspace, not pull in vec2text, ollama, or anything else.

3. **The renderer MUST NOT block on naming.** A flaky LLM provider, a network outage, a rate-limit cliff — none of these can prevent retrieval from working. Rendering must always return *some* tree, even if the labels degrade.

4. **Naming work should scale with what is queried, not with corpus size.** A million-memory neuromem on a quiet day should not pay 1000× a hundred-memory neuromem's naming cost. Cost should be proportional to LLM-side reads of the rendered tree.

## Decision

**Defer centroid naming from clustering time to render time, with automatic numpy fallback.**

### Concrete behaviour

1. **During `_run_dream_cycle`'s clustering loop**, each merge writes the new centroid with a **placeholder label**: `f"cluster_{centroid_id[:12]}"`. **No LLM call inside the clustering loop.** Clustering becomes pure numpy + storage I/O.

2. **At render time inside `ContextHelper.build_prompt_context`**, after `storage.get_subgraph(root_ids, depth)` returns, the helper calls a new `NeuroMemory.resolve_centroid_names(subgraph["nodes"])`:
   - Filter `nodes` to centroids whose label still starts with `"cluster_"` (placeholder pattern).
   - Look up each placeholder centroid's two `child_of` children in storage (cheap targeted query — small N).
   - Build a list of `[child_a_label, child_b_label]` pairs.
   - Call `self.llm.generate_category_names_batch(pairs)` — a NEW provider method that asks for N one-word names in ONE LLM round-trip.
   - Persist the resulting names via a NEW `storage.update_node_labels(updates)` method so subsequent renders touching the same centroids skip the call.
   - Mutate the in-memory `nodes` list in place so the immediate render uses the new labels.

3. **On any failure of the batched LLM call** (timeout, exception, JSON parse error, length mismatch, empty response), the resolver **automatically falls through to numpy nearest-neighbour**: each unnamed centroid is renamed to the label of whichever of its children has the highest cosine similarity to the centroid's own embedding. Persisted the same way. NO further LLM calls.

4. **The renderer never raises out of `resolve_centroid_names`.** Worst case: every centroid stays as `cluster_a3f8b1c0…`. The tree is uglier but still functional — the LLM can still see the structure and the memory snippets.

### What this preserves

- Centroid labels in any rendered tree are LLM-quality (when the provider is healthy) or at minimum a real existing tag label (under fallback). The hierarchy of abstraction the user described — `retail → groceries → coupons` — survives.
- The clustering algorithm is unchanged: same cosine-similarity merges, same `child_of` edge structure, same termination at `cluster_threshold`. Only the side-effect of naming each centroid is moved.
- Cache hits are free: a centroid renamed once stays renamed in storage. The same query against the same dream-cycle output costs nothing on the second render.

### What this changes

- Per-instance dream-cycle wall time on the LongMemEval benchmark: **~22 min → ~2 min** (clustering drops from ~1200s to ~0.5s; new render-time naming cost is ~1-2s per first render).
- LLM calls per dream cycle: **1188 → 0**.
- LLM calls per query: **0 (cached) → ~1 batched call** worst case (first render after a dream cycle, with 10–30 centroids needing names).
- Total naming cost over a session is bounded by the number of unique centroids that appear in any rendered tree across the lifetime of the storage. For most retrieval-driven workloads this is tiny.

## Alternatives Considered

### Option A — Per-merge LLM naming during clustering (the pre-ADR behaviour)

The original implementation. Correct semantics: every centroid has an LLM-quality name as soon as it exists. **Problems**: 1188 serial API calls dominate wall time; flaky API or rate limit kills the dream cycle; pays the cost for centroids that nobody ever reads.

Rejected because the cost is unbounded (scales with corpus, not query) and most of the work is wasted.

### Option B — Pure numpy nearest-neighbour during clustering (no LLM at all)

Each centroid's label = its closest child's label. Free, fast, no API dep. **Problem**: centroid hierarchy collapses semantically. A centroid merging `coupon` and `voucher` would be named `coupon` (one of its children). The next centroid up merging `coupon` (the rebranded centroid) and `discount` would be named `coupon` again. The rendered tree degenerates to `coupon → coupon → coupon → mem_xxx`, losing the levels of abstraction that justify the whole clustering pipeline.

Rejected because it violates Constraint #1 — labels are not cosmetic.

Kept as the **fallback path** within the chosen design (Option D), where it only fires when the LLM is unavailable, and where the fallback labels at least won't be entirely wrong (closest-child is a reasonable degraded approximation).

### Option C — vec2text / true embedding-to-text inversion

Cornell's vec2text (Morris et al., EMNLP 2023) inverts an embedding back to natural language via an iterative T5 decoder. Could in principle take a centroid embedding and produce a fresh string label.

**Problems**:
- The published pre-trained corrector model only works for OpenAI's `text-embedding-ada-002`. Using it on `gemini-embedding-001` would require training a new corrector against Google's training corpus, which we don't have access to.
- Adds a multi-hundred-MB HuggingFace model dep — violates Constraint #2 (lean deps).
- Requires GPU for tolerable latency (~50-100ms); CPU inference is slow.
- Even if we had a Gemini-compatible corrector, vec2text's accuracy is ~92% exact-match on 32-token sequences — fine for full sentences, overkill and unnecessary for one-word category labels.

Rejected: not Gemini-compatible, dep-heavy, and the "embedding inversion" framing isn't actually what we need (we have the children's labels right there; we just want one good name combining them, which is a much smaller LLM ask).

### Option D — R-NN parallel batched naming during clustering (ParChain pattern)

Reciprocal Nearest Neighbour pairs in agglomerative clustering are guaranteed disjoint within one round — A is B's nearest AND B is A's nearest. They can be merged in parallel and named in one batched LLM call per round. ParChain (2021) demonstrates 30× call reduction in production.

**Problems**:
- Still pays the naming cost for every centroid the algorithm produces, including the ~1100 that nobody ever reads.
- Lazy naming (the chosen approach) is *strictly* better: it names a *subset* of centroids that R-NN-batching would name. The one batched call lazy naming makes per query is a strict subset of the work R-NN-batching would do per dream cycle.

Rejected. R-NN batching is a real optimisation but lazy naming subsumes its benefit and adds the additional benefit of skipping unread centroids entirely.

### Option E — Local Gemma chat model for naming (`gemma-2-2b-it` or similar)

Run a small local model to do the per-merge naming. Free per call, no rate limits, ~300-500ms per call on Apple Silicon → ~6-10 min clustering instead of 22 min.

**Trade-offs**: still pays the cost for every centroid (doesn't fix the unread-centroid waste). Adds a HuggingFace / Ollama runtime dep. Requires model download on first run.

Deferred to a future `neuromem-gemma` sibling package. Slots into the same `LLMProvider.generate_category_names_batch` hook this ADR establishes — when local Gemma is available, it just becomes the implementation of that method and lazy naming gets even cheaper.

## Consequences

### Positive

- **Wall-time drops to ~2 min per benchmark instance** (from ~22 min). Clustering becomes pure numpy.
- **LLM cost is now bounded by query volume**, not corpus size. A million-memory store costs the same to operate as a hundred-memory store if you only query small subtrees.
- **The system survives LLM outages.** Numpy fallback ensures the renderer always returns a tree, even degraded.
- **Centroid labels in rendered trees are still LLM-quality** in the common case. The hierarchy of abstraction is preserved.
- **Cache amortises cost.** Once a centroid is named, future queries that touch it pay nothing.
- **The new `LLMProvider.generate_category_names_batch` hook is a clean extension point** for future provider implementations (local Gemma, custom on-prem LLMs, etc.).

### Negative

- **First render after a dream cycle is slower** by the cost of one batched LLM call (~1-2s for 10-30 names). Subsequent renders are unaffected.
- **Centroids that have never been queried have placeholder labels.** If anyone introspects storage directly (tests, debugging, custom tooling), they'll see `cluster_a3f8b1c0` for unread centroids. This is by design; the labels are not user-facing.
- **Numpy fallback labels persist forever.** If the LLM happens to be down on the first render touching a centroid, that centroid keeps its degraded name even after the LLM recovers. There's no automatic re-naming pass. Acceptable: future dream cycles produce new centroids that get the lazy-naming flow afresh, so labels heal as the corpus is touched.
- **One narrow correctness assumption**: the `"cluster_"` prefix on placeholder labels must NEVER collide with a label a real LLM might produce. We cap at 12 hex chars from the centroid id (`cluster_a3f8b1c01234`) — extremely unlikely for an LLM to emit that exact form. If it ever did, the resolver would treat that centroid as needing a name and re-name it, which is harmless.

### Revisit criteria

- The numpy fallback fires on more than ~5% of renders in production. Means the LLM provider is unhealthy — fix the provider, but also reconsider whether the fallback labels are good enough.
- A future render is asked to surface so much of the graph (`top_k` or `depth` cranked way up) that batched naming becomes the bottleneck for that render. Then we'd want to cap how many centroids we name per call, or stream the rendering.
- A Gemini-compatible embedding-to-text decoder ships somewhere we can use. Worth re-evaluating the fallback strategy then.
- If query volume × subtree size × naming cost ever approaches the wasted cost of the original eager design — but at that point you'd have ~1000 unique centroids surfaced per session, which means the workload is genuinely retrieval-heavy and the cost is honestly earned.

## References

- ADR-001 — clustering library choice. Lazy naming means the per-merge LLM callback no longer holds the bottleneck during clustering. The custom-loop choice in ADR-001 still holds because lazy naming still calls into provider-supplied logic.
- `packages/neuromem-core/src/neuromem/system.py::_run_clustering` — pre-ADR call site of `generate_category_name`.
- `packages/neuromem-core/src/neuromem/context.py::ContextHelper.build_prompt_context` — render-time hook for lazy resolution.
- `packages/neuromem-core/src/neuromem/providers.py::LLMProvider` — gains `generate_category_names_batch`.
- `packages/neuromem-core/src/neuromem/storage/base.py::StorageAdapter` — gains `update_node_labels`.
- Morris et al., "Text Embeddings Reveal (Almost) As Much As Text" (EMNLP 2023) — vec2text reference.
- ParChain (2021) — R-NN batched parallel agglomerative clustering reference.
- Wang et al., "EmbeddingGemma: A Distilled Multilingual Embedding Model" (2024) — surveyed during the decision; one-way embedding model only.
