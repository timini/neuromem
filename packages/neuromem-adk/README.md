# neuromem-adk

> Google Agent Development Kit integration for [`neuromem-core`](../neuromem-core/). One-line memory attachment for ADK agents.

`neuromem-adk` adds persistent long-term memory to any Google ADK agent with a single function call. The agent automatically captures every conversation turn, injects relevant past context into every model call, exposes the memory as LLM-callable tools, and plugs into ADK's native `BaseMemoryService` slot — all from one `enable_memory()` call.

**Status**: v0.1.0 alpha.

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

That's it. From this point:

- Every completed turn flows into memory via `after_agent_callback` (the `enqueue` hook).
- Before every model call, relevant past memories are prepended to the system prompt via `before_model_callback` (the `ContextHelper` hook).
- The LLM can also explicitly call `search_memory` and `retrieve_memories` tools if it decides to look something up.
- ADK's native `BaseMemoryService` slot is wired through `NeuromemMemoryService`, so framework-level features that depend on the memory slot work automatically.

---

## What `enable_memory` actually does

Under the hood, the one-line call performs these steps:

1. **Instantiates providers**: creates `GeminiLLMProvider` and `GeminiEmbeddingProvider` from [`neuromem-gemini`](../neuromem-gemini/) reading `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) from the environment.
2. **Builds the memory system**: `NeuroMemory(storage=SQLiteAdapter(db_path), llm=..., embedder=...)`.
3. **Appends two function tools** to `agent.tools`:
   - `search_memory(query, top_k=5, depth=2) -> str` — bound wrapper so the LLM never sees the internal `system` handle.
   - `retrieve_memories(memory_ids) -> list[dict]` — same.
4. **Chains two callbacks** onto the agent's slots:
   - `before_model_callback` → prepends the rendered memory tree to `llm_request.config.system_instruction`.
   - `after_agent_callback` → extracts the latest user/assistant pair from the session and enqueues both into memory with role metadata.
5. **Returns** the `NeuroMemory` handle so you can inspect memory state, force consolidation (`memory.force_dream(block=True)`), or override behaviour in tests.

All four mechanisms complement each other: passive injection handles "context I probably need right now", tool calls handle "let me go look up something specific", and the `BaseMemoryService` path handles "session-end consolidation and native ADK memory features".

---

## Overriding the default providers

For development against a local embedder, different Gemini models, or mock providers in tests:

```python
from neuromem_gemini import GeminiLLMProvider, GeminiEmbeddingProvider

agent = Agent(...)

memory = enable_memory(
    agent,
    db_path="memory.db",
    llm=GeminiLLMProvider(api_key=api_key, model="gemini-2.5-flash"),
    embedder=GeminiEmbeddingProvider(api_key=api_key, model="gemini-embedding-001"),
)
```

Omitting one of `llm=` or `embedder=` falls back to the Gemini default for just that slot.

For mock providers in tests, see the [quickstart](../../specs/003-neuromem-adk/quickstart.md) §Testing without network.

---

## Advanced: custom `NeuromemMemoryService` wiring

If you're building a custom ADK `Runner` configuration and need to pass a memory service explicitly, instantiate `NeuromemMemoryService` directly:

```python
from neuromem import NeuroMemory, SQLiteAdapter
from neuromem_gemini import GeminiLLMProvider, GeminiEmbeddingProvider
from neuromem_adk import NeuromemMemoryService

memory = NeuroMemory(
    storage=SQLiteAdapter("memory.db"),
    llm=GeminiLLMProvider(api_key=api_key),
    embedder=GeminiEmbeddingProvider(api_key=api_key),
    dream_threshold=5,   # more eager consolidation
    decay_lambda=1e-6,   # faster forgetting
)

memory_service = NeuromemMemoryService(memory)
# Pass memory_service into your ADK Runner setup.
```

This path is for the 10% case. `enable_memory` is what you want 90% of the time.

---

## Integration tests

This package ships one end-to-end test that hits the real Gemini API. It's gated behind the `integration` pytest marker so it never runs during normal development flow:

```bash
# Default runs skip the integration suite (fast, no network):
uv run pytest                            # 248+ tests, no network

# Opt in to run the real ADK integration test:
export GOOGLE_API_KEY="your-key-here"
uv run pytest packages/neuromem-adk/tests/ -m integration --no-cov -v
```

The `--no-cov` flag is required because the repo's coverage gate is scoped to `neuromem-core`'s four core modules — a sibling-package-only run would trivially miss the 90% threshold.

The integration test:

1. Builds a real `google.adk.agents.Agent`.
2. Attaches memory via `enable_memory(agent, ":memory:")`.
3. Runs a 3-turn scripted conversation through ADK's real `Runner` + real `google-genai` backend.
4. Asserts that (a) every turn was captured, (b) turn 3 correctly references a fact mentioned in turn 1 (proving passive injection worked), and (c) `retrieve_memories` spikes `access_weight` back to 1.0.

Cost per run: ~5000 tokens, <$0.01 at current Gemini 2.0 Flash rates. Runtime: ~10 seconds. CI runs this nightly plus on every merge to `main` via `.github/workflows/integration.yml`.

---

## How it compares to mem0's ADK integration

[mem0](https://docs.mem0.ai/integrations/google-ai-adk) also integrates memory into ADK but takes a **tool-registration-only** approach — it registers `search_memory` and `save_memory` as plain Python functions the LLM can call, and the developer is responsible for calling `mem0.add()` explicitly after every turn.

`neuromem-adk` does the same tool registration **plus three more things** mem0 doesn't:

1. **Automatic turn capture** via `after_agent_callback` — you never have to remember to save a turn.
2. **Passive context injection** via `before_model_callback` — the LLM sees relevant memory before it even starts reasoning, without having to call a tool.
3. **Native `BaseMemoryService` integration** — plugs into ADK's framework-level memory slot so future ADK features that depend on that slot work automatically.

All of these wire up from the same one-line `enable_memory()` call.

---

## License

MIT. See [LICENSE](../../LICENSE) at the repo root.
