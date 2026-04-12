# neuromem-bench

> Benchmark harness for the [neuromem](../../) long-term memory library. Runs published memory-evaluation benchmarks against agents built with [`neuromem-adk`](../neuromem-adk/) and publishes scored results.

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
# Run LongMemEval_s (the smallest variant) against neuromem-adk
# with real Gemini providers:
export GOOGLE_API_KEY="your-key-here"
uv run python packages/neuromem-bench/scripts/run_longmemeval.py \
    --split s \
    --sample-size 10 \
    --agent neuromem-adk \
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
│   │                          #   - NullAgent (no memory baseline)
│   │                          #   - NeuromemAdkAgent (uses enable_memory)
│   ├── runner.py              # Orchestrator: load dataset → drive agent → collect metrics
│   ├── metrics.py             # exact_match, accuracy, (future) LLM-as-judge
│   └── datasets/
│       ├── base.py            # Dataset ABC with load() → iter[Instance]
│       └── longmemeval.py     # LongMemEval loader + scorer
├── tests/
├── scripts/
│   └── run_longmemeval.py     # CLI entrypoint for a benchmark run
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
| `NullAgent` | No-memory baseline | None — feeds each turn straight to the LLM |
| `NeuromemAdkAgent` | The thing we're validating | `neuromem-adk.enable_memory` with Gemini providers |
| `NaiveRagAgent` | Vector-only baseline | Single SQLite table, no clustering / decay / LTP |

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
