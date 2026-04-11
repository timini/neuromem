# Quickstart: neuromem Core Library

**Feature**: 001-neuromem-core
**Audience**: Agent developers who want to wire `neuromem` into an existing Python agent in under 15 minutes (SC-001).

---

## Install

```bash
uv add neuromem-core
```

Once v1 ships. During early development, clone the monorepo and sync the workspace:

```bash
git clone git@github.com:timini/neuromem.git
cd neuromem            # repo root — this is the uv workspace
uv sync --dev          # installs all workspace packages + dev tools
```

The repo is a **`uv` workspace monorepo** (Constitution v2.0.0 Additional Constraints). The core package lives at `packages/neuromem-core/`, and future sibling packages will live alongside it. `uv sync` at the repo root installs every workspace member in editable mode against a shared `uv.lock`.

`neuromem-core`'s runtime dependencies are deliberately small:

- Python 3.10+ standard library
- `numpy >= 1.26`
- `pandas >= 2.1`

Nothing else. No LLM SDKs, no framework SDKs, no vector-DB clients. Verify with `uv tree packages/neuromem-core` — the non-stdlib children should be exactly `numpy` and `pandas` (plus their own small transitive set).

---

## Minimum viable wiring

You need three things:

1. An `EmbeddingProvider` (wraps your embedding API).
2. An `LLMProvider` (wraps your LLM for summarisation, tagging, cluster naming).
3. A `StorageAdapter` (use the bundled `SQLiteAdapter` unless you have a reason to swap).

Then instantiate `NeuroMemory` and call `enqueue()` on every conversation turn.

```python
import numpy as np
from neuromem import NeuroMemory, ContextHelper, SQLiteAdapter
from neuromem.providers import LLMProvider, EmbeddingProvider

# --- Your provider implementations (example: OpenAI) ---
# These live in YOUR code or in a future neuromem-openai sibling package.
# The core neuromem-core package does NOT import openai.

class MyEmbeddingProvider(EmbeddingProvider):
    def __init__(self, client, model="text-embedding-3-small"):
        self.client = client
        self.model = model

    def get_embeddings(self, texts: list[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        # Return a 2-D numpy array shape (len(texts), D). Stack the rows.
        return np.array([d.embedding for d in resp.data], dtype=np.float32)

class MyLLMProvider(LLMProvider):
    def __init__(self, client, model="gpt-4o-mini"):
        self.client = client
        self.model = model

    def generate_summary(self, raw_text: str) -> str:
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Summarise in 1-2 sentences."},
                {"role": "user", "content": raw_text},
            ],
        )
        return r.choices[0].message.content

    def extract_tags(self, summary: str) -> list[str]:
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system",
                 "content": "List 3-7 key concepts as a comma-separated list, no explanation."},
                {"role": "user", "content": summary},
            ],
        )
        return [t.strip() for t in r.choices[0].message.content.split(",") if t.strip()]

    def generate_category_name(self, concepts: list[str]) -> str:
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system",
                 "content": "Reply with ONE word: the category that encompasses these concepts."},
                {"role": "user", "content": ", ".join(concepts)},
            ],
        )
        return r.choices[0].message.content.strip().split()[0]

# --- Wire up the memory system ---

import openai
client = openai.OpenAI()  # reads OPENAI_API_KEY

memory = NeuroMemory(
    storage=SQLiteAdapter("memory.db"),
    llm=MyLLMProvider(client),
    embedder=MyEmbeddingProvider(client),
)
```

That's the one-time setup. From this point, `memory` is the object your agent talks to.

---

## The two-call loop

### Call 1: capture every turn (hippocampal acquisition)

On every conversation turn — after the user speaks, after the agent replies, after a tool call — hand the text to `enqueue()`:

```python
user_message = "I want to optimize the WAL mode config on my SQLite database"
memory.enqueue(user_message)

agent_reply = "You should check the pragma journal_mode and synchronous settings..."
memory.enqueue(agent_reply, metadata={"role": "assistant"})
```

`enqueue()` returns a UUID and, under the hood, spawns a background dreaming thread every 10 inserts to consolidate the batch into the knowledge graph. **You never wait for this** — the call returns in ~50 ms (excluding the synchronous `generate_summary` call, which is the only unavoidable LLM latency per Resolved Design Decision #3).

### Call 2: inject relevant context (prefrontal recall)

Before every LLM call from your agent, ask `neuromem` for the relevant memory sub-graph:

```python
helper = ContextHelper(memory)

def run_agent_turn(task: str) -> str:
    context_block = helper.build_prompt_context(task)

    system_prompt = f"""You are a helpful agent.

Relevant memories:
{context_block}
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ],
    )
    return response.choices[0].message.content
```

`context_block` will look like:

```
📁 Databases
├── 📁 SQLite
│   ├── 📄 mem_7732: "WAL mode discussion — pragma journal_mode=WAL..."
│   └── 📄 mem_9104: "Pragma synchronous=NORMAL vs FULL tradeoff..."
└── 📁 Indexing
    └── 📄 mem_4421: "Composite indexes for multi-column WHERE..."
```

If the query has no match in the graph (cold start, or totally unrelated topic), `build_prompt_context` returns an **empty string** — your agent just runs without memory context (per Resolved Design Decision #2: return empty, no on-demand dreaming, KISS).

---

## Giving the agent its own recall tool

`ContextHelper` is the *passive* channel — it injects context before every LLM call. You also want the *active* channel: let the agent itself decide to go search memory for a specific topic it can't see in the injected context.

```python
from neuromem import search_memory, retrieve_memories

# Expose as agent tools (format varies by framework)
tools = [
    {
        "name": "search_memory",
        "description": "Search long-term memory for concepts related to a query. "
                       "Returns an ASCII tree with memory IDs.",
        "parameters": {"query": "string"},
        "function": lambda query: search_memory(query, system=memory),
    },
    {
        "name": "retrieve_memories",
        "description": "Fetch the full content of specific memories by ID. "
                       "Use IDs returned from search_memory.",
        "parameters": {"memory_ids": "list[string]"},
        "function": lambda memory_ids: retrieve_memories(memory_ids, system=memory),
    },
]
```

The exact tool-registration syntax depends on your agent framework (ADK, Anthropic tool use, LangChain, etc.) — the important part is that `search_memory` and `retrieve_memories` are plain functions with the correct signatures, ready to bind.

**What happens when the agent calls `retrieve_memories`**: the library fetches the full memory records AND spikes each memory's `access_weight` back to 1.0 (Long-Term Potentiation). Frequently-used memories self-reinforce and survive the synaptic pruning pass.

---

## Testing without network calls

Principle V mandates mocks. Write your tests using the mock providers:

```python
# tests/test_my_agent.py
import pytest
from neuromem import NeuroMemory, ContextHelper, SQLiteAdapter
from neuromem.providers import LLMProvider, EmbeddingProvider

class MockEmbeddingProvider(EmbeddingProvider):
    def get_embeddings(self, texts):
        # Deterministic hash-seeded embedding for testing
        import hashlib
        out = np.empty((len(texts), 16), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = int.from_bytes(hashlib.md5(t.encode()).digest()[:8], "big")
            rng = np.random.default_rng(seed)
            out[i] = rng.uniform(-1.0, 1.0, size=16).astype(np.float32)
        return out

class MockLLMProvider(LLMProvider):
    def generate_summary(self, raw_text): return raw_text[:80]
    def extract_tags(self, summary): return summary.split()[:3]
    def generate_category_name(self, concepts): return "TestCat"

def test_my_agent_remembers():
    memory = NeuroMemory(
        storage=SQLiteAdapter(":memory:"),
        llm=MockLLMProvider(),
        embedder=MockEmbeddingProvider(),
        dream_threshold=3,  # lower for tests
    )

    # Enqueue 3 turns to trigger dreaming
    memory.enqueue("SQLite is a lightweight SQL database")
    memory.enqueue("Python is a high-level language")
    memory.enqueue("uv is a fast Python package manager")

    memory.force_dream(block=True)  # ensure dreaming completes before assertions

    helper = ContextHelper(memory)
    ctx = helper.build_prompt_context("database question")
    assert "📄 mem_" in ctx
```

`SQLiteAdapter(":memory:")` gives you a fresh in-memory database per test, zero cleanup needed.

---

## Tuning

All thresholds are constructor arguments with sensible defaults:

```python
NeuroMemory(
    storage=...,
    llm=...,
    embedder=...,
    dream_threshold=10,      # inbox count that triggers dreaming
    decay_lambda=3e-7,       # per-second decay rate (~30-day half-life)
    archive_threshold=0.1,   # access_weight below this → archived
    cluster_threshold=0.82,  # cosine similarity for agglomerative merging
)
```

Common tuning patterns:

- **Short-lived agents / session testing**: `dream_threshold=3` so you don't need many turns to see the graph form.
- **Long-lived personal agents**: defaults are fine.
- **High-volume logging agents**: `dream_threshold=50` to reduce LLM API costs; `archive_threshold=0.3` to prune faster.
- **Never forget mode**: `decay_lambda=1e-9` (effectively infinite half-life, decay never archives anything).

---

## Swapping the storage backend

This is the whole point of the adapter pattern. To run against Postgres/pgvector, Firebase, Qdrant, etc., implement `StorageAdapter` and inject it:

```python
from neuromem.storage.base import StorageAdapter

class PostgresAdapter(StorageAdapter):
    def __init__(self, dsn: str):
        ...
    # Implement all 13 abstract methods — see contracts/storage-adapter.md
    ...

memory = NeuroMemory(
    storage=PostgresAdapter("postgresql://..."),
    llm=MyLLMProvider(...),
    embedder=MyEmbeddingProvider(...),
)
```

Everything else works identically. The orchestration layer does not know or care what's behind the ABC.

---

## What comes next

All of these are future **sibling packages** under `packages/` in the same monorepo, not separate repos:

- **Framework wrappers**: `packages/neuromem-adk/`, `packages/neuromem-anthropic/`, `packages/neuromem-langchain/`. Each bundles a provider pair + the `enqueue()` hook into the respective framework's message loop, declares `neuromem-core` as a workspace dependency in its `pyproject.toml`, and publishes to PyPI under its own name (`neuromem-adk`, etc.). None of those are in this feature's scope.
- **Built-in embedding provider**: `packages/neuromem-fastembed/` or similar — a local-embedding provider that doesn't require an API call, so small projects can run fully offline.
- **Alternative storage adapters**: Postgres/pgvector (`packages/neuromem-pgvector/`?), Firebase, Qdrant. Each can ship as its own sibling package OR as an optional-dependency extra inside `neuromem-core`. The adapter pattern guarantees they slot in without touching orchestration code.
- **Benchmarks**: GraphRAG-Bench, AMB, LongMemEval (per SC-008). Validate the cognitive loop against standard retrieval benchmarks; baseline measurements are the first goal.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `enqueue()` is slow | Your `generate_summary` is hitting a real LLM on the caller thread. | Inject a faster model, a local model, or a passthrough provider (`return raw_text[:200]`). |
| `build_prompt_context` returns empty string | No consolidated memories match the query yet. | Enqueue at least `dream_threshold` items and call `force_dream(block=True)` in tests. |
| Pre-commit hook fails on `--no-verify` attempt | It's forbidden by Constitution Principle VI. | Fix the underlying lint/format/test failure. Never bypass. |
| `TypeError: Can't instantiate abstract class MyAdapter with abstract method X` | Your `StorageAdapter` subclass is missing a method. | Check `contracts/storage-adapter.md` — all 13 methods must be implemented. |
| Memories from a year ago keep showing up | Decay rate is too slow. | Increase `decay_lambda` or lower `archive_threshold`. |
