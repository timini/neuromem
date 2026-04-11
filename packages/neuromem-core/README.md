# neuromem-core

> Neuroscience-inspired long-term memory for AI agents. Adapter-based, framework-agnostic, test-first.

`neuromem-core` is a Python library that gives any LLM agent a persistent, self-organising memory modelled on the hippocampal → neocortex consolidation loop. It captures conversation turns as they happen, consolidates them into a concept graph during background "dream" cycles, decays unused memories, and reinforces recalled ones (Long-Term Potentiation). The whole library is framework-agnostic — it imports **zero** LLM or agent-framework SDKs — so it drops into any stack (ADK, Anthropic, LangChain, LangGraph, custom).

**Status**: v0.1.0 alpha (Phase 8 polish). Published from the [`neuromem`](https://github.com/timini/neuromem) monorepo.

---

## Install

```bash
uv add neuromem-core
```

During early development, clone the monorepo and sync the workspace:

```bash
git clone git@github.com:timini/neuromem.git
cd neuromem
uv sync --dev
```

Runtime dependencies are deliberately small (Constitution Principle II): Python 3.10+, `numpy>=1.26`, `pandas>=2.1`. Nothing else.

---

## 30-second example

```python
import numpy as np
from neuromem import NeuroMemory, ContextHelper, SQLiteAdapter
from neuromem.providers import LLMProvider, EmbeddingProvider

# Bring your own providers — neuromem-core does not import any LLM SDK.
class MyLLM(LLMProvider):
    def generate_summary(self, raw_text): ...
    def extract_tags(self, summary): ...
    def generate_category_name(self, concepts): ...

class MyEmbedder(EmbeddingProvider):
    def get_embeddings(self, texts): ...  # returns np.ndarray shape (N, D)

memory = NeuroMemory(
    storage=SQLiteAdapter("memory.db"),
    llm=MyLLM(),
    embedder=MyEmbedder(),
)

# Capture every turn (hippocampal acquisition — non-blocking background dream).
memory.enqueue("The user is debugging a WAL-mode SQLite locking issue.")
memory.enqueue("Agent suggested PRAGMA synchronous=NORMAL.")

# Inject relevant context before every LLM call (prefrontal recall).
helper = ContextHelper(memory)
context_block = helper.build_prompt_context("sqlite performance tuning")
# context_block is an ASCII tree ready to drop into a system prompt:
#   📁 Databases
#   └── 📁 SQLite
#       ├── 📄 mem_7732: "The user is debugging a WAL-mode..."
#       └── 📄 mem_9104: "Agent suggested PRAGMA synchronous..."
```

For the full agent integration — tool registration, mock providers for testing, tuning knobs, backend swaps — read the [complete quickstart](../../specs/001-neuromem-core/quickstart.md).

---

## Why it exists

Most "memory for agents" libraries bolt a vector index onto the side of a chat loop. `neuromem-core` instead models the **cognitive loop** end-to-end:

- **Hippocampus** — `enqueue()` captures turns into an inbox. No latency impact.
- **Neocortex** — a background *dream cycle* consolidates the inbox into a concept graph: tag extraction, agglomerative clustering, centroid naming. Runs on a daemon thread, doesn't block the caller.
- **Prefrontal cortex** — `ContextHelper.build_prompt_context(query)` and `search_memory(query, system)` traverse the graph to surface the smallest relevant sub-graph.
- **Synaptic pruning** — every dream cycle decays `access_weight` on consolidated memories via an exponential formula and archives anything that drops below a threshold.
- **Long-Term Potentiation** — `retrieve_memories()` spikes `access_weight=1.0` on every memory the agent actually looked at, so frequently-recalled items self-reinforce and survive pruning.

Everything runs against the `StorageAdapter` abstract base class (13 methods). The bundled `SQLiteAdapter` has zero external dependencies; swap in Postgres/pgvector, Qdrant, or Firebase by writing a ~400-line adapter and injecting it.

---

## Key properties

- **Framework-agnostic.** No `openai`, `anthropic`, `langchain`, `google.adk`, etc. imported anywhere in `src/`. Enforced at CI time by a tripwire test.
- **Lean dependency set.** `numpy + pandas` is the entire runtime dep list. Enforced at CI time by a tripwire test.
- **Pluggable.** Providers (LLM, embedder) and storage are all injected. 14-test contract suite exercises every `StorageAdapter` implementation identically.
- **Test-first.** 200+ tests (unit + integration + contract) cover every module. Mock providers in `tests/conftest.py` let you run the cognitive loop end-to-end with zero network calls.
- **Atomic commits.** Every change ≤500 lines, enforced by pre-commit hooks (ruff + pytest). Never `--no-verify`.

---

## Workspace layout

`neuromem-core` lives inside the [`neuromem`](https://github.com/timini/neuromem) monorepo:

```
neuromem/                           # uv workspace root
├── packages/
│   └── neuromem-core/              # ← you are here — this README
│       ├── src/neuromem/
│       │   ├── __init__.py         # public API re-exports
│       │   ├── system.py           # NeuroMemory orchestration
│       │   ├── context.py          # ContextHelper (ASCII tree renderer)
│       │   ├── tools.py            # search_memory + retrieve_memories
│       │   ├── providers.py        # LLMProvider + EmbeddingProvider ABCs
│       │   ├── vectors.py          # numpy cosine / centroid helpers
│       │   └── storage/
│       │       ├── base.py         # StorageAdapter ABC (13 methods)
│       │       └── sqlite.py       # SQLiteAdapter (stdlib sqlite3)
│       └── tests/                  # 193 tests — parametrised contract suite
├── specs/001-neuromem-core/        # full feature spec, plan, tasks, ADRs
└── .specify/memory/constitution.md # project-wide engineering constitution
```

Future sibling packages (framework wrappers, alternative storage backends) will live alongside `neuromem-core/` under `packages/` and declare it as a workspace dependency.

---

## Documentation

| Doc | What it covers |
|---|---|
| [Quickstart](../../specs/001-neuromem-core/quickstart.md) | End-to-end agent integration, mock providers, tuning knobs, backend swaps. |
| [Feature specification](../../specs/001-neuromem-core/spec.md) | User stories, acceptance criteria, cognitive model, scope. |
| [Implementation plan](../../specs/001-neuromem-core/plan.md) | Architecture, tech decisions, file layout. |
| [Data model](../../specs/001-neuromem-core/data-model.md) | Memory / Node / Edge schemas, invariants, state machines. |
| [Storage adapter contract](../../specs/001-neuromem-core/contracts/storage-adapter.md) | Per-method behavioural contract for every `StorageAdapter` implementation. |
| [Public API contract](../../specs/001-neuromem-core/contracts/public-api.md) | `NeuroMemory`, `ContextHelper`, `search_memory`, `retrieve_memories` signatures + semantics. |
| [ADR-001: Clustering library choice](../../docs/decisions/ADR-001-clustering-library-choice.md) | Why the dream cycle hand-rolls agglomerative clustering on numpy instead of pulling in scipy / sklearn / HDBSCAN. |
| [Constitution v2.0.0](../../.specify/memory/constitution.md) | Project-wide engineering rules: framework-agnostic core, lean deps, layered architecture, test-first, atomic commits. |

---

## License

MIT. See [LICENSE](../../LICENSE) at the repo root.
