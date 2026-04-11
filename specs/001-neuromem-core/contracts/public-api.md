# Contract: Public API Surface

**Feature**: 001-neuromem-core
**Scope**: Every symbol exported from `neuromem` that downstream users (agent developers, framework wrapper authors) depend on.
**Binding**: These contracts are part of the library's public interface. Breaking them requires a MAJOR version bump per standard semver.

---

## `neuromem.system.NeuroMemory`

The orchestration engine.

### Constructor

```python
class NeuroMemory:
    def __init__(
        self,
        storage: StorageAdapter,
        llm: LLMProvider,
        embedder: EmbeddingProvider,
        dream_threshold: int = 10,
        decay_lambda: float = 3e-7,
        archive_threshold: float = 0.1,
        cluster_threshold: float = 0.82,
    ) -> None:
        """Constructor. Uses numpy and pandas internally per Constitution v2.0.0
        Principle II. Import path: `from neuromem import NeuroMemory`."""
```

**Preconditions**:
- `storage` is an instance of a concrete `StorageAdapter` subclass. Passing the ABC directly or `None` raises `TypeError` (ABC's normal behaviour — no explicit check needed).
- `llm` is an instance of a concrete `LLMProvider` subclass.
- `embedder` is an instance of a concrete `EmbeddingProvider` subclass.
- `dream_threshold >= 1`. Values `< 1` raise `ValueError`.
- `0 < decay_lambda`. Non-positive values raise `ValueError`.
- `0 <= archive_threshold < 1.0`. Out-of-range raises `ValueError`.
- `0 < cluster_threshold <= 1.0`. Out-of-range raises `ValueError`.

**Postconditions**:
- Instance is ready to call `enqueue`, `force_dream`, `is_dreaming`.
- `storage` has its schema initialised (the adapter's `__init__` or a lazy init on first call must handle this).
- No background thread is running yet.

**Errors**: `TypeError` on bad `storage`/`llm`/`embedder` types, `ValueError` on out-of-range thresholds.

---

### `enqueue(raw_text, metadata=None) -> str`

```python
def enqueue(self, raw_text: str, metadata: dict | None = None) -> str: ...
```

**Purpose**: Insert a new memory into the inbox. Triggers a background dreaming cycle if the inbox threshold is met after insertion.

**Preconditions**:
- `raw_text` is a non-empty UTF-8 string.
- `metadata`, if provided, is a JSON-serialisable dict (`json.dumps(metadata)` succeeds).

**Postconditions**:
- A new row exists in `memories` with `status='inbox'`, `access_weight=1.0`, `created_at=<now>`, `summary=<llm.generate_summary(raw_text)>`.
- Returns the new memory's UUID string.
- If inbox count ≥ `dream_threshold` and no dreaming cycle is currently running, a background thread has been spawned (non-blocking) and is processing the flipped batch.
- The caller thread returns within 50 ms *excluding* the `llm.generate_summary()` call (SC-002).

**Errors**:
- `ValueError` if `raw_text` is empty or `None`.
- `ValueError` if `metadata` is not JSON-serialisable.
- `StorageError` if the underlying adapter cannot write (e.g., read-only filesystem, disk full).
- Any exception from `llm.generate_summary(raw_text)` propagates. The memory is NOT inserted in that case.

**Thread safety**: Safe to call from any thread. SQLite serialises the write at the connection level.

---

### `force_dream(block=True) -> None`

```python
def force_dream(self, block: bool = True) -> None: ...
```

**Purpose**: Manually trigger a dreaming cycle, bypassing the `dream_threshold` check. Useful for CLI tooling and tests.

**Preconditions**: None.

**Postconditions**:
- If `block=True`: all memories currently in `status='inbox'` have been processed through the pipeline (now `status='consolidated'` or `'archived'`) and the method has returned.
- If `block=False`: a background thread has been spawned to process them; method returns immediately.
- If the inbox is empty, the method returns immediately with no work done.
- If a dreaming cycle is already running, the method returns immediately (does NOT queue a second cycle).

**Errors**:
- Provider exceptions during the cycle bubble up *only* when `block=True`. Otherwise they are logged and the batch is rolled back (FR-029).

---

### `is_dreaming` (read-only property)

```python
@property
def is_dreaming(self) -> bool: ...
```

**Returns**: `True` iff a background dreaming thread currently holds the `_dream_lock`. `False` otherwise.

**Use case**: Test assertions and observability. Not a synchronisation primitive — do not spin-wait on it.

---

## `neuromem.context.ContextHelper`

Prompt-injection helper. Takes a task description and renders a relevant sub-graph.

### Constructor

```python
class ContextHelper:
    def __init__(self, memory_system: NeuroMemory) -> None: ...
```

**Preconditions**: `memory_system` is a `NeuroMemory` instance.

**Postconditions**: Instance holds a reference to the memory system. No side effects.

---

### `build_prompt_context(current_task_description, top_k=5, depth=2) -> str`

```python
def build_prompt_context(
    self,
    current_task_description: str,
    top_k: int = 5,
    depth: int = 2,
) -> str: ...
```

**Purpose**: Embed the task description, find the nearest nodes, traverse the sub-graph up to `depth` hops, and render an ASCII tree suitable for inclusion in the agent's system prompt.

**Preconditions**:
- `current_task_description` is a non-empty string.
- `top_k >= 1`, `depth >= 0`.

**Postconditions**:
- Returns a string formatted as an ASCII tree with box-drawing characters, or the empty string `""` if no relevant nodes are found (per Resolved Design Decision #2: simple KISS behaviour, no on-demand dream cycle).
- The string, if non-empty, contains at least one line starting with `📁 ` (node) and at least one line starting with `📄 mem_` (memory reference).
- **Side effect**: one call to `embedder.get_embeddings([current_task_description])`. **No** call to the LLM provider. **No** call to `retrieve_memories` (so no LTP spike — retrieval of the specific memory is the agent's follow-up action).

**Output format** (binding):

```
📁 <parent-label>
├── 📁 <child-label>
│   ├── 📄 mem_<uuid>: <summary snippet ≤80 chars>
│   └── 📄 mem_<uuid>: <summary snippet>
└── 📁 <child-label>
    └── 📄 mem_<uuid>: <summary snippet>
```

The renderer handles multi-parent nodes by appending `(also under: <other-parent>)` on second and later mentions to avoid duplicating entire subtrees.

**Errors**: Propagates embedding provider errors. No recovery logic — caller decides how to handle.

---

## `neuromem.tools.search_memory`

```python
def search_memory(
    query: str,
    system: NeuroMemory,
    top_k: int = 5,
    depth: int = 2,
) -> str: ...
```

**Purpose**: Agent-facing tool function. Same semantics as `ContextHelper.build_prompt_context()` — in fact, v1's implementation can be a thin wrapper around `ContextHelper(system).build_prompt_context(query, top_k, depth)`.

**Why the redundancy**: Framework wrappers (ADK, Anthropic SDK, LangChain) expect tool functions with a flat signature — `fn(arg1, arg2, ...) -> str` — not class methods. `search_memory` is the signature those wrappers will bind as a tool; `ContextHelper` is the object-oriented API for callers who want to hold a helper instance.

**Contract**: Identical to `ContextHelper.build_prompt_context`.

---

## `neuromem.tools.retrieve_memories`

```python
def retrieve_memories(
    memory_ids: list[str],
    system: NeuroMemory,
) -> list[dict]: ...
```

**Purpose**: Agent-facing tool function. Given a list of memory IDs (obtained from an earlier `search_memory()` result), fetch their full records.

**Preconditions**:
- `memory_ids` is a list of strings. Empty list is valid (returns `[]`).
- `system` is a `NeuroMemory` instance.

**Postconditions**:
- Returns a list of dicts, one per *found* memory (missing IDs are silently skipped, User Story 2 acceptance scenario #3).
- Each returned dict has the shape:

```python
{
    "id": str,            # UUID
    "raw_content": str,
    "summary": str | None,
    "status": str,        # 'inbox' | 'dreaming' | 'consolidated' | 'archived'
    "access_weight": float,
    "created_at": int,    # Unix epoch seconds
    "last_accessed": int | None,
    "metadata": dict | None,
}
```

- **Side effect (LTP)**: For every returned memory whose `status == 'consolidated'`, `access_weight` is set to `1.0` and `last_accessed` is set to `time.time()` (FR-021).
- Memories with `status == 'inbox'` or `'dreaming'` are returned as-is but **do not** receive LTP (they have not been consolidated yet, so "reinforcement" has no meaning).
- Memories with `status == 'archived'` are returned as-is but **do not** receive LTP and are **not** automatically resurrected to `'consolidated'` (User Story 3 acceptance scenario #2).

**Errors**: Only I/O-level `StorageError` propagates. Missing IDs, empty input, and bad states all return cleanly.

---

## `neuromem.vectors`

numpy-backed vector helpers. Used internally by `NeuroMemory._run_dream_cycle` and by the `SQLiteAdapter.get_nearest_nodes` implementation. Exposed publicly so alternative storage adapters can reuse them. Module is named `vectors.py` (not `math.py`) to avoid shadowing the stdlib `math` module.

### `cosine_similarity(vec_a, vec_b) -> float`

```python
import numpy as np
from numpy.typing import NDArray

def cosine_similarity(
    vec_a: NDArray[np.floating] | list[float],
    vec_b: NDArray[np.floating] | list[float],
) -> float: ...
```

**Preconditions**:
- `vec_a` and `vec_b` are numpy arrays or Python lists of floats of equal length.
- Length > 0.

**Postconditions**:
- Returns a Python `float` in `[-1.0, 1.0]` (for non-negative embeddings, practically `[0.0, 1.0]`).
- If either vector has zero magnitude, returns `0.0`.
- Python list inputs are converted to `np.float64` internally.

**Errors**:
- `ValueError` on shape mismatch.
- `ValueError` on empty vectors.

**Implementation**: `float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))`, with zero-magnitude guard.

---

### `batch_cosine_similarity(query, matrix) -> np.ndarray`

```python
def batch_cosine_similarity(
    query: NDArray[np.floating],
    matrix: NDArray[np.floating],
) -> NDArray[np.float64]: ...
```

**Preconditions**:
- `query` is a 1-D array of shape `(D,)`.
- `matrix` is a 2-D array of shape `(N, D)` with the same `D`.
- `N >= 1`, `D >= 1`.

**Postconditions**:
- Returns a 1-D `np.float64` array of shape `(N,)` where element `i` is `cosine_similarity(query, matrix[i])`.
- Rows of `matrix` with zero magnitude yield `0.0` (not NaN).
- Does not mutate inputs.

**Errors**:
- `ValueError` on dimension mismatch.

**Use case**: `SQLiteAdapter.get_nearest_nodes()` fetches the full `(N, D)` node embedding matrix once, then calls this function to produce all N similarity scores in a single BLAS call (`matrix @ query` under the hood). This is the SC-003 performance path.

---

### `compute_centroid(vectors) -> np.ndarray`

```python
def compute_centroid(
    vectors: NDArray[np.floating]
             | list[NDArray[np.floating]]
             | list[list[float]],
) -> NDArray[np.float64]: ...
```

**Preconditions**:
- `vectors` is non-empty.
- If 2-D: shape `(N, D)` with `N >= 1`.
- If list-of-vectors: all inner vectors have the same length.

**Postconditions**:
- Returns a new 1-D `np.float64` array of length `D`, the element-wise mean of all inputs.
- Does not mutate any input.

**Errors**:
- `ValueError` if `vectors` is empty.
- `ValueError` if inner lengths differ.

**Implementation**: `np.asarray(vectors, dtype=np.float64).mean(axis=0)`.

---

## `neuromem.__init__` re-exports

The package (`packages/neuromem-core/src/neuromem/__init__.py`) exposes, at the top level, exactly these names:

```python
# from neuromem import ...
NeuroMemory             # from .system
ContextHelper           # from .context
search_memory           # from .tools
retrieve_memories       # from .tools
StorageAdapter          # from .storage.base
SQLiteAdapter           # from .storage.sqlite
LLMProvider             # from .providers
EmbeddingProvider       # from .providers
cosine_similarity       # from .vectors
batch_cosine_similarity # from .vectors
compute_centroid        # from .vectors
__version__             # str
```

`__version__` is a standard `str` attribute (FR-030). Format: semver `"<major>.<minor>.<patch>"`.

Nothing else. Internal helpers (`_run_dream_cycle`, `_render_ascii_tree`) are module-private and never re-exported.

**Package distribution name vs import name**: The PyPI package is `neuromem-core` (hyphenated), the import name is `neuromem` (unhyphenated). These differ by design so that future sibling packages (`neuromem-adk` → `import neuromem_adk`, etc.) have their own unambiguous top-level namespaces. Users install `uv add neuromem-core` and then `from neuromem import NeuroMemory`.
