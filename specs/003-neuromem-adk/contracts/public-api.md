# Public API contract — `neuromem_adk`

Everything in this document defines what the `neuromem-adk` package commits to keeping stable across v0.1.x patch releases. Anything NOT listed here is private and may change without a version bump.

## Public namespace

The `neuromem_adk` top-level package exports exactly three names:

```python
from neuromem_adk import enable_memory, NeuromemMemoryService, __version__
```

Nothing else. Internal helpers live in `neuromem_adk._enable`, `neuromem_adk._callbacks`, `neuromem_adk._memory_service`, etc., and may be renamed / removed between patch releases.

---

## `enable_memory(agent, db_path, *, llm=None, embedder=None)`

The one-function wire-up helper. Attaches persistent memory to an existing ADK `Agent` with a single call.

### Signature

```python
def enable_memory(
    agent: "google.adk.agents.Agent",
    db_path: str | os.PathLike[str],
    *,
    llm: LLMProvider | None = None,
    embedder: EmbeddingProvider | None = None,
) -> NeuroMemory:
    ...
```

### Arguments

- **`agent`** *(required)*: An existing `google.adk.agents.Agent` instance. The function mutates the agent in place — appends to `agent.tools`, chains `agent.before_model_callback` and `agent.after_agent_callback` — so callers must not pass an agent that has already been passed through `enable_memory` (see Errors below).

- **`db_path`** *(required)*: Filesystem path to the SQLite database the attached `NeuroMemory` will read and write. If the file doesn't exist, it will be created on first write. If it exists, the existing memory graph is loaded and continues from its current state. Accepts `str` or any `os.PathLike`.

- **`llm`** *(optional, keyword-only)*: An `LLMProvider` instance to use for summarisation, tag extraction, and cluster naming. If `None`, `enable_memory` instantiates `GeminiLLMProvider(api_key=os.environ["GOOGLE_API_KEY"])` from the `neuromem-gemini` sibling package.

- **`embedder`** *(optional, keyword-only)*: An `EmbeddingProvider` instance to use for tag-node embedding. If `None`, `enable_memory` instantiates `GeminiEmbeddingProvider(api_key=os.environ["GOOGLE_API_KEY"])`.

Both `llm` and `embedder` can be passed independently; passing one and omitting the other is allowed — the omitted slot falls back to the default.

### Returns

A `NeuroMemory` instance (from `neuromem-core`) that is fully wired and ready to use. Callers can use the returned handle to:

- Inspect memory state: `memory.storage.count_memories_by_status("consolidated")`
- Force a dream cycle manually: `memory.force_dream(block=True)` — useful in tests and benchmarks that need deterministic state
- Introspect the attached database: `memory.storage.get_all_nodes()`, `memory.storage.get_memories_by_status(...)`

### Side effects

Calling `enable_memory(agent, db_path)` performs the following mutations on the agent, in order:

1. **Instantiates the memory system** — `NeuroMemory(storage=SQLiteAdapter(db_path), llm=llm_or_default, embedder=embedder_or_default)`.
2. **Appends two function tools** to `agent.tools`:
   - A partial-bound `search_memory` with signature `(query: str, top_k: int = 5, depth: int = 2) -> str`.
   - A partial-bound `retrieve_memories` with signature `(memory_ids: list[str]) -> list[dict]`.
   Existing tools are preserved; the new tools are appended to the end of the list.
3. **Chains two callbacks** onto the agent:
   - `before_model_callback` is replaced by a composed callback that first calls the previous `before_model_callback` (if any), then calls the neuromem context injector, returning the combined result.
   - `after_agent_callback` is chained similarly, invoking any previous callback before the neuromem turn capturer.
4. **Returns** the `NeuroMemory` handle.

The function does NOT:

- Call the LLM or embedder (except transitively, if `enable_memory` is wrapped in a test that exercises a subsequent model call).
- Start any background thread. Background dream cycles are spawned lazily by `NeuroMemory` itself when the inbox threshold is hit.
- Modify any global state. The only side effects are on the `agent` object and the SQLite file at `db_path`.

### Errors

- **`ValueError`**: Raised if `agent` is already wired by a previous `enable_memory` call. Detection is via an attribute marker set on the agent (`agent._neuromem_enabled = True`); attaching twice would double-wire callbacks and tools, which is never the intended behaviour.
- **`KeyError`**: Raised if `llm` or `embedder` is `None` and the environment variable `GOOGLE_API_KEY` is not set. The error message names the missing variable and points to the package README.
- **`StorageError`**: Raised by the underlying `SQLiteAdapter` if the database file is corrupt or unreadable. The error message includes the full `db_path` for debugging.
- **`ImportError`**: If `google-adk` is not installed (should never happen for users who installed `neuromem-adk`, because it's a runtime dep), re-raised from the `from google.adk.agents import Agent` statement at the top of the module.

### Thread safety

`enable_memory` is not thread-safe. It MUST be called once per agent, from a single thread, during agent setup. After the call returns, the returned `NeuroMemory` handle inherits the thread-safety guarantees of `neuromem-core` (background dreaming thread + non-blocking enqueue from the caller thread).

---

## `NeuromemMemoryService`

An implementation of ADK's `google.adk.memory.BaseMemoryService` abstract base class. Advanced users who need fine-grained control of the memory service wiring (e.g., when building a custom `Runner` or integrating neuromem with ADK features beyond `enable_memory`'s simple callback path) instantiate this class directly.

### Signature

```python
class NeuromemMemoryService(BaseMemoryService):
    def __init__(self, memory: NeuroMemory) -> None: ...
```

### Arguments

- **`memory`** *(required)*: A `NeuroMemory` instance that the service will delegate to. Typically, users who instantiate `NeuromemMemoryService` directly have already constructed a `NeuroMemory` with their own choice of storage / providers and want to plug it into ADK's native memory slot.

### Methods

The class implements the `BaseMemoryService` ABC:

- **`add_session_to_memory(session)`**: Called by ADK when a session completes. Iterates over the session's recorded events, extracts (user, assistant) pairs, calls `memory.enqueue()` for each, then calls `memory.force_dream(block=True)` to consolidate before returning.

- **`search_memory(query: str, *, user_id: str | None = None)`**: Called by ADK when an agent uses the built-in memory-search flow (for example, via the `LoadMemory` or `PreloadMemory` tools). Delegates to `neuromem.tools.search_memory(query, system=self._memory)` and returns the result in whatever shape `BaseMemoryService` expects (structured result object, not just the ASCII tree — the exact ADK return type is documented in ADK's own contract).

### Errors

Same underlying error set as `NeuroMemory` itself — `StorageError` for database issues, `ValueError` for invalid inputs. The service does not wrap or re-raise; errors propagate directly.

### When to use

- You're building a custom `Runner` configuration and need to pass a memory service explicitly.
- You want multiple agents to share a single memory store via ADK's native memory-sharing mechanism.
- You're writing ADK-framework-level integration tests that invoke memory operations through the native code path instead of through the function tools.

For the 90% case of "I want my one ADK agent to have memory," use `enable_memory` instead. It creates a `NeuromemMemoryService` internally and wires it into the agent's runner without the caller having to do it.

---

## Stability guarantee

Within the v0.1.x patch range (v0.1.0 → v0.1.9, plus any v0.1.xN releases):

- The `enable_memory` signature is stable. New keyword arguments may be added with backward-compatible defaults. Positional arguments are fixed at `(agent, db_path)`.
- The `NeuromemMemoryService` class name and its two public methods (`add_session_to_memory`, `search_memory`) are stable.
- The `__version__` string follows `<major>.<minor>.<patch>` semantic versioning.
- Private helpers (anything under `neuromem_adk._*`) may be renamed or removed without a version bump.

Breaking changes (removing or renaming `enable_memory`, removing `NeuromemMemoryService`, changing positional argument order) require a minor version bump (v0.2.0). Major version bumps follow when the underlying `neuromem-core` package does.

---

## Tool output contracts (for LLM interaction)

The two tools registered by `enable_memory` have the following return-value shapes. These are the shapes the language model will see in its tool-result context.

### `search_memory` tool return

A string containing an ASCII tree (same format as `neuromem.tools.search_memory`). Example:

```
Relevant Memory Context:
📁 SQLite
├── 📄 mem_abc123: "WAL mode discussion — pragma journal_mode=WAL..."
└── 📄 mem_def456: "Pragma synchronous=NORMAL vs FULL tradeoff..."
📁 Python
└── 📄 mem_ghi789: "asyncio event loop unresponsiveness under CPU-bound work..."
```

If no memories match, returns an empty string. The language model receives the empty string as-is and can decide not to use memory for this turn.

### `retrieve_memories` tool return

A list of dicts, one per successfully-retrieved memory. Each dict has the full memory record shape documented in `neuromem-core`'s `public-api.md`:

```python
[
    {
        "id": "mem_abc123",
        "raw_content": "...",
        "summary": "...",
        "status": "consolidated",
        "access_weight": 1.0,        # spiked by LTP
        "created_at": 1712345678,
        "last_accessed": 1712999999, # spiked by LTP
        "metadata": {"role": "user", "turn": 5},
    },
    ...
]
```

Memory IDs that don't exist are silently skipped (not errors). The tool call never raises due to unknown IDs.
