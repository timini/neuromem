# Performance — rolling baseline

Tracks the ADR-004 hot-path / ingest-split perf numbers across
commits. The **retrieval** harness (`perf_retrieval.py`) is a HARD
gate — regressions block merge. The **ingestion** harness
(`perf_ingestion.py`) is MEASUREMENT ONLY — regressions are
logged but don't fail CI. Ingestion runs in the background between
sessions; optimisation is deferred.

## Retrieval (ADR-004 D3) — sub-second hot path

| SHA | Date | Embedder | enq p50 | build p50 | build p95 | expand p50 | retrieve p50 | LLM calls | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| c9dad63 | 2026-04-14 | mock | 0.0 ms | 2.9 ms | 3.2 ms | 2.3 ms | 0.0 ms | **0** | PASS |

Gate thresholds:
- enqueue p50 ≤ 20 ms (NF-H1)
- build_prompt_context p50 ≤ 50 ms (mock) / ≤ 500 ms (Gemini) (NF-H2/H3)
- build_prompt_context p95 ≤ 1500 ms (mock) / ≤ 1000 ms (Gemini) (NF-H4)
- expand_node p50 ≤ 200 ms (NF-H5)
- retrieve_memories p50 ≤ 50 ms (NF-H6)
- **LLM calls from retrieval methods = 0** (NF-H7 — the invariant)

## Ingestion (ADR-004 D2) — background, measurement only

| SHA | Date | Embedder | n | Mean wall | Stddev |
|---|---|---|---|---|---|
| c9dad63 | 2026-04-14 | mock | 100 | 0.69 s | 0.97 s |

No hard targets. Flip `--fail-on-regression` on the harness once the
pinned baseline is stable.
