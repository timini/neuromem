# neuromem-bench

> Benchmark harness for the [neuromem](../../) long-term memory library. Runs published memory-evaluation benchmarks against agents that use neuromem's cognitive loop, and publishes scored results.

This package is a development tool, not a consumer library. It exists so the neuromem project can measure its own cognitive-loop quality against published benchmarks (SC-008 in the neuromem-core spec).

**Status**: v0.1.0 alpha. First benchmark target: **LongMemEval** (ICLR 2025).

---

## Install

For monorepo development:

```bash
uv sync --dev
```

The package is a workspace member and pulls in `neuromem-core`, `neuromem-gemini`, and `neuromem-adk` via workspace source resolution. No PyPI release planned for v0.1 — this is a tool for the neuromem project itself, not for general consumption.

---

## Running a benchmark

```bash
# Run LongMemEval_s (the smallest variant) against the neuromem
# cognitive loop with real Gemini providers:
export GOOGLE_API_KEY="your-key-here"
uv run python packages/neuromem-bench/scripts/run_longmemeval.py \
    --split s \
    --sample-size 10 \
    --agent neuromem \
    --output docs/benchmarks/longmemeval-s-$(date +%Y%m%d).jsonl
```

This produces one JSONL line per benchmark instance with the predicted answer, ground-truth answer, score, and latency. An aggregate summary markdown file is written alongside.

---

## Architecture

```
neuromem-bench/
├── src/neuromem_bench/
│   ├── __init__.py
│   ├── agent.py               # BaseAgent ABC + concrete implementations
│   │                          #   - NullAgent      (no memory baseline)
│   │                          #   - NaiveRagAgent  (vector-only baseline)
│   │                          #   - NeuromemAgent  (full cognitive loop)
│   ├── runner.py              # Orchestrator: load dataset → drive agent → collect metrics
│   ├── metrics.py             # exact_match, contains_match, llm_judge
│   ├── _client.py             # GeminiAnsweringClient — shared by agent.py + metrics.py
│   └── datasets/
│       ├── base.py            # Dataset ABC with load() → iter[Instance]
│       └── longmemeval.py     # LongMemEval loader + scorer
├── tests/
├── scripts/
│   ├── run_longmemeval.py     # CLI entrypoint for a benchmark run
│   └── rescore_with_llm_judge.py  # re-score an existing JSONL with llm_judge
└── docs/benchmarks/           # results are written here, one file per run
```

## Benchmarks supported

| Benchmark | Status | Notes |
|---|---|---|
| **LongMemEval** | v0.1 target | ICLR 2025. Multi-session chat QA across 40–500 sessions per instance. GPT-4o-as-judge scoring (we start with exact-match, add LLM judge later). Canonical repo: `xiaowu0162/LongMemEval`. |
| **MemoryAgentBench** | v1.5 | ICLR 2026. Tests retrieval / test-time learning / long-range / conflict resolution. Richer than LongMemEval but more complex harness. |
| **GraphRAG-Bench** | Deprioritised | Single-shot graph-RAG QA, orthogonal to neuromem's long-term memory thesis. |

---

## Agents supported

| Agent | Purpose | Memory backend |
|---|---|---|
| `NullAgent` | No-memory baseline | None — feeds each turn straight to the LLM, capped to the last N turns |
| `NaiveRagAgent` | Vector-only baseline | Per-turn embedding + cosine retrieval — no clustering, decay, or LTP |
| `NeuromemAgent` | The thing we're validating | `NeuroMemory` + `SQLiteAdapter` + `GeminiLLMProvider` + `GeminiEmbeddingProvider` — full cognitive loop |

`NeuromemAgent` deliberately exercises `NeuroMemory` directly rather than going through `neuromem-adk.enable_memory`. The `neuromem-adk` wrapper adds per-turn ADK `Runner` latency that's orthogonal to measuring memory-graph quality; the T015 integration test inside `neuromem-adk` already proves the ADK wiring works end-to-end. If you specifically want to measure the ADK integration path (tool-call behaviour, session-end consolidation, etc.) that's a different agent class worth adding.

Each agent satisfies a small `BaseAgent` protocol:

```python
class BaseAgent(Protocol):
    def process_turn(self, text: str, *, role: str) -> None: ...
    def answer(self, question: str) -> str: ...
    def reset(self) -> None: ...
```

`reset()` is called between benchmark instances to clear state.

---

## Running results

Results land in `docs/benchmarks/RESULTS.md` as a markdown leaderboard. Each row is one benchmark run with:

- Date + git commit hash
- Benchmark + split + sample size
- Agent configuration
- Metric value
- Total cost (if run against real APIs)
- Wall-clock runtime

---

## License

MIT. See [LICENSE](../../LICENSE) at the repo root.
