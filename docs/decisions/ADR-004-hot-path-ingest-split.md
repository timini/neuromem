# ADR-004: Hot-Path / Ingest Split — long-term memory as a pre-computed background artifact

**Status**: Accepted
**Date**: 2026-04-14
**Deciders**: Tim Richardson + Claude
**Extends**: ADR-003 (ontology tree v2)

## Context

After ADR-003 landed, the in-context memory tree carried real paragraph summaries and tool-driven drill-down. But a seam remained: **retrieval on the conversation hot path could still make LLM calls.** `ContextHelper.build_prompt_context` → `resolve_centroid_names` / `resolve_junction_summaries` paid Gemini round-trips whenever a centroid in the rendered subgraph had a missing label or summary. For warm cache that was fine; for any fresh query touching an unresolved centroid it was a several-hundred-millisecond to multi-second block in the user-visible turn path.

The user's mental model of long-term memory is sharper than the existing implementation:

- **Current session** is in chat history — the LLM's own context window (≈ 20K tokens). Nothing about it needs to live in memory until the session ends.
- **Sessions N–1 and older** are the domain of the retrieval machinery.
- Because memory only ever contains *past* sessions, ALL expensive work can run as a batch in the background. By the time session N starts, memory for sessions 1..N-1 is fully baked.

That in turn means the hot path is a pure read: embed the query, walk the graph, render the tree. Zero LLM calls beyond the single embedder RTT. Ingestion cost, while non-trivial, is not felt by the user because it never blocks a turn.

## Decision

### D1 — `enqueue` is non-blocking; summary generation moves to the background worker

Today `NeuroMemory.enqueue(raw_text)` calls `llm.generate_summary(raw_text)` synchronously on the caller thread, then inserts the memory with `summary` populated. Under ADR-004:

- `enqueue` writes the row with `raw_content` set and `summary = NULL`. One SQL insert, returns in milliseconds. **No LLM call on the caller thread.**
- `_run_dream_cycle` gains a new step 2: `generate_summary_batch(raw_contents)` over every inbox memory pulled into the current batch. One batched Gemini call per ~N memories, run in parallel chunks (same pattern as `extract_tags_batch`).
- A new non-abstract `LLMProvider.generate_summary_batch(list[str]) -> list[str]` default-loops over `generate_summary`. `GeminiLLMProvider` overrides with a chunked + `ThreadPoolExecutor` batched implementation.

### D2 — Dream cycle becomes full-sweep eager

Two new consolidation steps run after clustering and before the consolidated-flip:

- **Step 8 — `_run_all_centroid_naming`.** Full-sweep variant of ADR-002's `resolve_centroid_names`. Finds *every* centroid with a `cluster_*` placeholder label and runs the batched naming path over all of them (not just the query-rendered subset).
- **Step 9 — `_run_all_junction_summarisation`.** Full-sweep variant of ADR-003's `resolve_junction_summaries`. Floor-up batched pass that names every centroid with `paragraph_summary IS NULL`.

By cycle end, every consolidated memory has a non-NULL summary; every centroid has a real label and a paragraph summary. The render-time lazy resolvers become no-ops in the happy path.

### D3 — Hot-path contract: zero LLM calls in retrieval methods

`ContextHelper.build_prompt_context`, `expand_node`, and `retrieve_memories` make no LLM calls in the happy path. `resolve_centroid_names` / `resolve_junction_summaries` stay on `NeuroMemory` as defensive safety nets; they early-return when every node is already baked (always true under ADR-004). If they ever DO fire they log a WARNING — it means the background worker didn't complete before the read and the invariant is broken.

The single exception to "zero LLM calls" is the query embedder inside `build_prompt_context`. That's one round-trip per user turn, out of scope for this invariant (and cacheable at the Runner level if needed).

### D4 — ADK per-turn capture stays on by default

`neuromem_adk.enable_memory`'s `after_agent_turn_capturer` keeps firing per turn. Each turn → `memory.enqueue(...)` → instant inbox row, no LLM. When the inbox crosses `dream_threshold` (default 10), the background worker kicks off and eventually publishes the full-baked batch to `consolidated`. The current session's turns stay in `status = 'inbox'` until the worker runs, so retrieval doesn't surface them — matching the "current session is in chat history, past sessions are in memory" boundary.

### D5 — In-flight cycles never block hot-path reads

If session N+1 starts retrieval while session N's consolidation is mid-flight, the hot path reads the last committed snapshot. New data becomes visible atomically when the cycle flips `dreaming → consolidated`. No stalls, no explicit flush API.

## Consequences

### Positive

- **Sub-second retrieval** is structurally enforceable. The only variable latency is the embedder RTT.
- The hot-path-zero-LLM invariant becomes a testable property (`test_hot_path_invariant.py` asserts it with a call-counting mock provider).
- `enqueue` goes from "potentially ~1-2s synchronous call" to "SQL insert, returns in ms" — matters for per-turn capture where the caller is an agent on the response-generation hot path.
- Ingestion cost is still bounded by the number of inbox memories × per-memory LLM latency, but it runs off the critical path. When we optimise ingestion later (parallelism, caching, model swaps) the user-visible impact is a background win, not a blocker.

### Negative / trade-offs

- **A memory enqueued during session N is not retrievable in session N.** Its `status` stays `inbox` until the threshold-triggered worker runs. This is deliberate (current session lives in chat history; memory is "stuff from before this conversation"), but app authors who expect "I just said X; can I query X?" semantics need to understand the boundary.
- **Dream-cycle wall time grows.** Steps 8 + 9 add two batched LLM passes (all-unnamed centroids, all-unsummarised junctions) that were previously deferred to render time. Measured, not gated — NF-I2 tracks the regression but doesn't fail CI on it.
- **Render-time resolvers become safety-net code.** They stay in the codebase with a no-op-on-baked-node fast path. Slight complexity overhead; documented.

## Alternatives considered (rejected)

- **Synchronous `enqueue_session` with a full-baked return.** Would block the caller for however long the full pipeline takes. Great for test determinism, terrible for the per-turn hot path. Rejected.
- **Explicit session-end API (`ingest_session(turns) -> IngestHandle`).** Clean boundary, but requires every caller to know when a session ends. ADK has no session-end hook (confirmed via code exploration), so the app would have to thread the information through. The existing threshold-triggered batch model achieves "eventual materialisation" without requiring app support. Rejected in favour of reusing the current machinery.
- **Per-turn eager resolution (resolve names + summaries on every turn if stale).** Would cap hot-path latency to "one batched LLM call at worst" rather than zero. Not good enough — we want sub-500ms p50 and the embedder RTT is already half that budget.
- **Multi-process ingestion worker.** Out of scope for v1. In-process threading (current machinery) is adequate until concurrency pressure demands otherwise.

## Revisit when

- Ingestion wall-time grows large enough that a heavily-ingesting system starves the hot path via contention on the GIL or SQLite lock. Move to a subprocess worker.
- App authors consistently need "just-said X, query X" semantics during a session. Add an explicit in-session short-term store layered above memory.
- Query-embedder latency becomes the hot-path bottleneck (plausible at > 100 QPS). Cache embeddings per conversation turn.
