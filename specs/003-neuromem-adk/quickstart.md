# Quickstart: neuromem-adk

**Feature**: 003-neuromem-adk
**Audience**: Developers building agents with Google's Agent Development Kit who want persistent long-term memory in one line of code.

---

## Install

```bash
uv add neuromem-adk
```

This pulls in `neuromem-core`, `neuromem-gemini`, `google-adk`, and `numpy` transitively. Set your Gemini API key in the environment:

```bash
export GOOGLE_API_KEY="your-key-here"
```

(During monorepo development, clone the repo and run `uv sync --dev` at the root — the workspace pulls everything in automatically.)

---

## 30-second example

```python
from google.adk.agents import Agent
from neuromem_adk import enable_memory

# 1. Build a regular ADK agent.
agent = Agent(
    model="gemini-2.0-flash-001",
    name="assistant",
    instruction="You are a helpful assistant.",
)

# 2. Attach persistent memory with one function call.
memory = enable_memory(agent, db_path="memory.db")

# 3. Use the agent as normal — memory is captured and injected automatically.
#    (ADK runner setup omitted for brevity — use your existing runner.)
```

That's everything. From this point:

- Every completed turn flows into memory via `after_agent_callback`.
- Before every model call, relevant past memories are prepended to the system prompt via `before_model_callback`.
- The LLM can also explicitly call `search_memory` and `retrieve_memories` tools if it decides to look something up.
- ADK's native memory-service slot is wired through `NeuromemMemoryService`, so session-end hooks and future ADK features that depend on the memory slot work automatically.

---

## What just happened under the hood

`enable_memory(agent, db_path="memory.db")` performed these steps:

1. **Instantiated providers**: created `GeminiLLMProvider` and `GeminiEmbeddingProvider` reading `GOOGLE_API_KEY` from the environment.
2. **Built the memory system**: `NeuroMemory(storage=SQLiteAdapter("memory.db"), llm=..., embedder=...)`.
3. **Appended two function tools** to `agent.tools`:
   - `search_memory(query, top_k=5, depth=2) -> str` — pre-bound to the memory instance.
   - `retrieve_memories(memory_ids) -> list[dict]` — same.
4. **Chained two callbacks** onto the agent:
   - `before_model_callback` calls `ContextHelper.build_prompt_context(user_message)` and prepends the resulting ASCII tree to the agent's instruction before the model call.
   - `after_agent_callback` extracts the (user, assistant) turn pair and calls `memory.enqueue()`.
5. **Wired `NeuromemMemoryService`** into the ADK runner so session-end consolidation and the framework's native memory-search flow both work.
6. **Returned** the `NeuroMemory` handle for inspection.

---

## Testing without network

For tests, you can pass explicit mock providers and skip the `GOOGLE_API_KEY` requirement:

```python
from google.adk.agents import Agent
from neuromem_adk import enable_memory
from neuromem.providers import LLMProvider, EmbeddingProvider
import numpy as np

class MockLLM(LLMProvider):
    def generate_summary(self, raw_text): return raw_text[:80]
    def extract_tags(self, summary): return summary.split()[:3]
    def generate_category_name(self, concepts): return "TestCat"

class MockEmbedder(EmbeddingProvider):
    def get_embeddings(self, texts):
        return np.zeros((len(texts), 16), dtype=np.float32)

agent = Agent(model="gemini-2.0-flash-001", name="test", instruction="test")
memory = enable_memory(
    agent,
    db_path=":memory:",   # SQLite in-memory database
    llm=MockLLM(),
    embedder=MockEmbedder(),
)
```

`SQLiteAdapter(":memory:")` gives you a fresh in-process database per test, zero cleanup needed.

---

## Advanced: custom NeuromemMemoryService wiring

If you're building a custom ADK runner configuration and need to pass a memory service explicitly (instead of going through `enable_memory`'s callback-based wiring), instantiate `NeuromemMemoryService` directly:

```python
from neuromem import NeuroMemory, SQLiteAdapter
from neuromem_gemini import GeminiLLMProvider, GeminiEmbeddingProvider
from neuromem_adk import NeuromemMemoryService
import os

api_key = os.environ["GOOGLE_API_KEY"]

# Build your own NeuroMemory with custom configuration.
memory = NeuroMemory(
    storage=SQLiteAdapter("memory.db"),
    llm=GeminiLLMProvider(api_key=api_key),
    embedder=GeminiEmbeddingProvider(api_key=api_key),
    dream_threshold=5,              # more eager consolidation
    decay_lambda=1e-6,              # faster forgetting
    archive_threshold=0.3,          # more aggressive archival
)

# Wrap it in a MemoryService you can pass to ADK.
memory_service = NeuromemMemoryService(memory)

# Pass memory_service into your ADK Runner setup.
```

This path is for the 10% case. The `enable_memory` one-liner is what you want 90% of the time.

---

## Overriding the default providers

For development against a local embedder (no network calls) or for cost-conscious model selection:

```python
from neuromem_gemini import GeminiLLMProvider, GeminiEmbeddingProvider

agent = Agent(...)

memory = enable_memory(
    agent,
    db_path="memory.db",
    llm=GeminiLLMProvider(api_key=api_key, model="gemini-2.0-flash-lite"),
    embedder=GeminiEmbeddingProvider(api_key=api_key, model="gemini-embedding-001"),
)
```

Or, to avoid network calls entirely during tests, pass mock providers (see "Testing without network" above).

---

## Integration test invocation

The package ships one end-to-end integration test that hits the real Gemini API. It's gated behind the `integration` pytest marker so it never runs during normal development flow:

```bash
# Default runs skip the integration suite (fast, no network):
uv run pytest                            # 208+ core tests + adk unit tests, no network

# Opt in to run the real ADK integration test:
export GOOGLE_API_KEY="your-key-here"
uv run pytest packages/neuromem-adk/tests/ -m integration --no-cov -v
```

The `--no-cov` flag is required because the repo's coverage gate is scoped to `neuromem-core`'s four core modules — a sibling-package-only run would trivially miss the 90% threshold.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `KeyError: 'GOOGLE_API_KEY'` when calling `enable_memory` | Environment variable not set | `export GOOGLE_API_KEY=...` or pass explicit providers |
| `ValueError: agent already has neuromem enabled` | `enable_memory` called twice on the same agent | Create a fresh `Agent` instance per call |
| Tool schema error at ADK's agent construction time | `functools.partial` incompatibility with ADK's schema generator | Report an issue; fallback path uses wrapper functions instead of partials |
| Agent responses don't seem to use memory | Dream cycle not yet fired | Either enqueue enough turns to hit `dream_threshold` (default 10) or call `memory.force_dream(block=True)` manually to trigger consolidation |
| `StorageError: database is locked` | Another process has the SQLite file open | Use a per-process db_path, or upgrade to a Postgres-backed `StorageAdapter` (future work) |
| Pre-commit hook fails trying to `--no-verify` | Forbidden by Constitution Principle VI | Fix the underlying lint/format/test failure. Never bypass. |
