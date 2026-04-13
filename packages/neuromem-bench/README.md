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
| `NeuromemAgent` | Direct-NeuroMemory cognitive-loop baseline | `NeuroMemory` + `SQLiteAdapter` + `GeminiLLMProvider` + `GeminiEmbeddingProvider`. One-shot answer path: the ASCII context tree is injected as system prompt, no tool calls. Fast per-instance. |
| `NeuromemAdkAgent` | The real product | Real `google.adk.agents.Agent` with `neuromem_adk.enable_memory`. Answer LLM has `search_memory` + `retrieve_memories` as function tools, so it can drill into `raw_content` mid-answer when summaries aren't specific enough. 2-5× slower per answer than `NeuromemAgent`; measures the full cognitive loop rather than the handicapped one-shot path. |

`NeuromemAgent` and `NeuromemAdkAgent` are complementary benchmark arms. The direct variant isolates "how good is the memory graph as a prompt-injection source?". The ADK variant adds "and does the LLM know when to reach for a tool when the prompt isn't enough?". Large score gaps between them indicate the tool-call path is pulling real weight.

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
