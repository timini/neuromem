# Feature Specification: neuromem Core Library

**Feature Branch**: `001-neuromem-core`
**Created**: 2026-04-11
**Status**: Draft
**Input**: Gemini design conversation (see /tmp/gemini_convo.txt) + user brief

---

## User Scenarios & Testing

### User Story 1 — Agent Developer Wires Up Long-Term Memory in Under 10 Lines (Priority: P1)

An agent developer has a working conversational agent and wants to give it persistent, searchable memory across sessions. They install `neuromem`, instantiate `NeuroMemory` with their preferred embedding and LLM providers, and call `enqueue()` on every conversation turn. The system silently accumulates memories and consolidates them in the background. On the next session, the developer calls `ContextHelper.build_prompt_context()` before each LLM call and the agent's system prompt now includes a relevant ASCII sub-graph of past knowledge.

**Why this priority**: This is the entire value proposition of the library. Every other story depends on this one working first. Without acquisition + contextual recall, nothing else matters.

**Independent Test**: Can be fully tested by creating a `NeuroMemory` instance with a `MockLLMProvider` and `MockEmbeddingProvider`, calling `enqueue()` ten or more times to trigger dreaming, then calling `ContextHelper.build_prompt_context("test query")` and asserting that the returned string is non-empty and contains at least one memory ID.

**Acceptance Scenarios**:

1. **Given** a freshly initialised `NeuroMemory` with a `SQLiteAdapter` and mock providers, **When** the developer calls `enqueue(raw_text)` eleven times, **Then** a background dreaming thread starts automatically, the inbox rows transition from `status='inbox'` to `status='dreaming'` and finally to `status='consolidated'`, and no exception is raised in the calling thread.

2. **Given** a `NeuroMemory` instance with at least one consolidated memory, **When** `ContextHelper.build_prompt_context("any topic")` is called, **Then** the return value is a non-empty string containing an ASCII tree block with at least one `📄 mem_` line.

3. **Given** a `NeuroMemory` instance with providers injected, **When** the developer passes `NeuroMemory(storage=SQLiteAdapter(":memory:"), llm=my_llm, embedder=my_embedder)`, **Then** the object initialises without error and `enqueue()` is callable immediately.

---

### User Story 2 — Agent Uses `search_memory` Tool to Retrieve Relevant Knowledge During a Task (Priority: P1)

During execution, the agent determines it needs to recall information about a topic not covered in its current context window. It calls the `search_memory(query)` tool function, receives an ASCII sub-graph with memory IDs, selects the relevant IDs, and calls `retrieve_memories(memory_ids)` to get full content. Retrieving the memories spikes their `access_weight` (Long-Term Potentiation), reinforcing them against decay.

**Why this priority**: This is the active recall loop — without it, contextual injection (Story 1) is the only channel to memory, which is passive and may not surface the right nodes for every task.

**Independent Test**: Can be tested with a pre-populated in-memory SQLite database (bypassing acquisition) by calling `search_memory("database optimisation", system=nm)` and asserting the return value is a string containing node labels and memory IDs, then calling `retrieve_memories(["mem_001"])` and asserting the returned dict contains `raw_content` and that `access_weight` for that memory increased.

**Acceptance Scenarios**:

1. **Given** a `NeuroMemory` instance with consolidated memories tagged `SQLite` and `Indexing`, **When** `search_memory("database performance")` is called, **Then** the return value is a rendered ASCII tree that includes nodes semantically related to databases and at least one `📄 mem_` line with a memory ID.

2. **Given** a memory with `access_weight = 0.4` (partially decayed), **When** `retrieve_memories(["<that_memory_id>"])` is called, **Then** the returned dict contains the full `raw_content` and `summary`, and the `access_weight` in storage is reset to `1.0` (or the configured spike value).

3. **Given** `retrieve_memories` is called with a list of IDs some of which do not exist, **When** the call completes, **Then** it returns results only for valid IDs and does not raise an exception for missing IDs.

---

### User Story 3 — Memories Decay and Are Archived When Not Accessed (Priority: P2)

An agent has been running for weeks. Memories from early sessions that have never been retrieved gradually lose their `access_weight`. At the end of each dreaming cycle, the pruning step archives any memory whose weight has dropped below the configured threshold. Archived memories are preserved in storage (raw content intact) but removed from active graph traversal, so they do not appear in search results or context trees.

**Why this priority**: Essential for production long-lived agents — without pruning, the graph grows unbounded and search latency degrades. However, a v1 agent can function temporarily without active pruning, making it P2.

**Independent Test**: Can be tested by inserting a memory directly into the SQLite database with `last_accessed` set to a timestamp far in the past, running `force_dream()`, and asserting that the memory's `status` is `'archived'` and that `search_memory` on a related query no longer returns it.

**Acceptance Scenarios**:

1. **Given** a consolidated memory with `last_accessed` set 30 days in the past and `access_weight` already below the configured threshold, **When** a dreaming cycle completes, **Then** the memory's `status` becomes `'archived'` and its edges are removed from the active graph.

2. **Given** an archived memory, **When** `retrieve_memories(["<archived_id>"])` is called, **Then** the method returns the memory's `raw_content` (deep retrieval still works) and the memory's `status` is NOT automatically restored to `consolidated`.

3. **Given** a memory that was partially decayed but still above threshold, **When** a dreaming cycle applies the decay formula, **Then** `access_weight` decreases but remains above the archive threshold and `status` stays `'consolidated'`.

---

### User Story 4 — Developer Swaps SQLite for a Custom Storage Backend (Priority: P2)

A developer deploying an agent to a cloud environment (e.g., Firebase, Postgres/pgvector) wants to reuse all neuromem logic but persist data to their cloud store. They implement the `StorageAdapter` ABC, inject it into `NeuroMemory`, and the entire acquisition/dreaming/recall pipeline works without modification.

**Why this priority**: The adapter pattern is a core architectural promise of the library. Validating that the `StorageAdapter` contract is sufficient to swap implementations is essential before v1 ships, but the feature set (Stories 1–3) works with just `SQLiteAdapter`.

**Independent Test**: Can be tested by implementing a `DictStorageAdapter` (in-memory dict, no SQLite) that satisfies every method in the `StorageAdapter` ABC, injecting it into `NeuroMemory`, and running the full Story 1 acceptance flow against it.

**Acceptance Scenarios**:

1. **Given** a class that inherits from `StorageAdapter` and implements all abstract methods, **When** it is passed as the `storage` argument to `NeuroMemory`, **Then** all public methods (`enqueue`, `force_dream`, `search_memory`, `retrieve_memories`) work correctly with no changes to core code.

2. **Given** a `StorageAdapter` subclass that does NOT implement all abstract methods, **When** it is instantiated, **Then** Python raises `TypeError` at instantiation time (standard ABC behaviour).

3. **Given** a `StorageAdapter` where `get_nearest_nodes` delegates to a native vector DB, **When** `search_memory` is called, **Then** no local cosine similarity computation is performed by the core (the adapter is solely responsible for similarity logic).

---

### User Story 5 — Developer Manually Forces a Dreaming Cycle (Priority: P3)

In a testing or CLI context, a developer wants to consolidate inbox memories immediately without waiting for the threshold. They call `NeuroMemory.force_dream()` and the full consolidation pipeline runs synchronously (or with an optional `block=True` argument).

**Why this priority**: Useful for testing and for CLI tooling, but not on the critical path for production agent usage.

**Independent Test**: Can be tested by enqueuing two memories (below the auto-trigger threshold), calling `force_dream()`, and asserting all previously inbox memories now have `status='consolidated'`.

**Acceptance Scenarios**:

1. **Given** three inbox memories and a threshold of 10, **When** `force_dream()` is called, **Then** all three are processed through the dreaming pipeline and their `status` becomes `'consolidated'`.

2. **Given** an empty inbox, **When** `force_dream()` is called, **Then** the method returns immediately without error.

---

### Edge Cases

- What happens when `enqueue()` is called while a dreaming thread is already running? New arrivals MUST be written as `status='inbox'` without interference (double-buffer pattern prevents collision). The running thread only processes rows it already flipped to `status='dreaming'`.
- What happens when the embedding provider returns an error (network timeout, rate limit)? The dreaming cycle MUST catch the exception, log it, and roll back the status of the batch from `'dreaming'` back to `'inbox'` so those memories are retried on the next cycle.
- What happens when the LLM provider returns a category name longer than a configured maximum (e.g., a sentence instead of one word)? The system MUST truncate or fall back to a safe default label such as `"Category_<short_hash>"`.
- What happens when `search_memory` is called with a query that has zero cosine similarity to every node in the graph? See Resolved Design Decisions #2 below: return an empty string immediately (KISS).
- What happens when `retrieve_memories` is called before any dreaming cycle has run (all memories still in inbox)? The method MUST still return raw content for valid IDs (reading from the `memories` table directly) but MUST NOT trigger LTP weight changes since the memory has not yet been consolidated.
- What happens when two dreaming threads are triggered in rapid succession before the first has started? The system MUST ensure only one dreaming thread runs at a time (use a threading lock or check for in-progress status before spawning).
- What happens when the SQLite database file is on a read-only filesystem? `NeuroMemory.__init__` MUST raise a clear `StorageError` with an actionable message.
- What happens when `metadata` passed to `enqueue()` is not JSON-serialisable? The adapter MUST raise a `ValueError` before attempting to write.

---

## Requirements

### Functional Requirements

- **FR-001**: The library MUST provide a `NeuroMemory` class in `neuromem.system` that accepts a `StorageAdapter`, an `LLMProvider`, and an `EmbeddingProvider` via constructor injection.
- **FR-002**: `NeuroMemory.enqueue(raw_text, metadata=None)` MUST insert a memory into storage with `status='inbox'` and `access_weight=1.0` and then check whether the inbox count meets or exceeds `dream_threshold`.
- **FR-003**: When the inbox count meets or exceeds `dream_threshold`, `enqueue()` MUST spawn a background thread to execute the dreaming/consolidation pipeline without blocking the calling thread.
- **FR-004**: The dreaming pipeline MUST atomically flip the target batch from `status='inbox'` to `status='dreaming'` before any processing begins (double-buffer pattern).
- **FR-005**: The dreaming pipeline MUST pass each memory's summary to `LLMProvider.extract_tags()` to obtain a list of concept label strings.
- **FR-006**: The dreaming pipeline MUST call `EmbeddingProvider.get_embeddings()` with all newly extracted tag labels to obtain their vector representations.
- **FR-007**: The dreaming pipeline MUST execute agglomerative (bottom-up) clustering on the full set of tag embeddings (new and existing nodes) using cosine similarity as the distance metric.
- **FR-008**: When a new cluster is formed during consolidation, the system MUST compute the centroid vector as the element-wise mean of all member tag embeddings.
- **FR-009**: When a new cluster is formed, the system MUST call `LLMProvider.generate_category_name(concepts: list[str]) -> str` with the cluster's member labels to obtain a human-readable one-word parent category name.
- **FR-010**: New cluster centroid nodes MUST be stored with `is_centroid=True` and connected to their member tag nodes via edges with `relationship='child_of'`.
- **FR-011**: Each memory MUST be connected to its extracted tag nodes via edges with `relationship='has_tag'`.
- **FR-012**: After successful processing, the dreaming pipeline MUST update all processed memories from `status='dreaming'` to `status='consolidated'`.
- **FR-013**: After each dreaming cycle, the system MUST apply exponential decay `W_new = W_old * exp(-λ * t)` to the `access_weight` of all `consolidated` memories, where `t` is elapsed seconds since `last_accessed` and `λ` is a configurable decay rate.
- **FR-014**: Any memory whose `access_weight` falls below the configurable `archive_threshold` after decay MUST have its `status` set to `'archived'` and its active graph edges removed from traversal queries.
- **FR-015**: Archival MUST preserve the `raw_content` and `summary` columns; content MUST NOT be deleted.
- **FR-016**: The library MUST provide a `ContextHelper` class in `neuromem.context` with a `build_prompt_context(current_task_description: str) -> str` method.
- **FR-017**: `build_prompt_context()` MUST embed the task description, find the top-k nearest nodes via the storage adapter, traverse their connected sub-graph, and return a formatted ASCII tree string.
- **FR-018**: The ASCII tree string MUST include node labels (folders) and memory summaries with IDs (files) in a hierarchical indented format using box-drawing characters.
- **FR-019**: The library MUST provide `search_memory(query: str, system: NeuroMemory) -> str` in `neuromem.tools` that returns an ASCII sub-graph string identical in format to `build_prompt_context()`.
- **FR-020**: The library MUST provide `retrieve_memories(memory_ids: list[str], system: NeuroMemory) -> list[dict]` in `neuromem.tools` that returns full memory content for the given IDs.
- **FR-021**: `retrieve_memories()` MUST update `last_accessed` to the current timestamp and reset `access_weight` to `1.0` for each successfully retrieved consolidated memory (Long-Term Potentiation).
- **FR-022**: The library MUST provide a `StorageAdapter` ABC in `neuromem.storage.base` defining the complete storage contract.
- **FR-023**: The library MUST ship a `SQLiteAdapter` in `neuromem.storage.sqlite` that satisfies the `StorageAdapter` contract using only Python's stdlib `sqlite3` module.
- **FR-024**: The `SQLiteAdapter.get_nearest_nodes()` MUST implement cosine similarity locally using **numpy** (vectorised matrix-vector operations over the full node-embedding matrix). This is the performance path SC-003 is calibrated against.
- **FR-025**: The library MUST provide `EmbeddingProvider` and `LLMProvider` ABCs in `neuromem.providers`. The core library MUST NOT import `openai`, `anthropic`, `google-genai`, or any specific LLM/embedding SDK. (numpy and pandas are NOT LLM/embedding SDKs — they are permitted per Constitution Principle II v2.0.0.)
- **FR-026**: The repository MUST be a `uv` workspace monorepo. The core library MUST live at `packages/neuromem-core/` with its own `pyproject.toml`, and MUST be published to PyPI as `neuromem-core` with import name `neuromem`. Runtime dependencies of `neuromem-core` are: Python 3.10+ standard library, `numpy` (>= 1.26), and `pandas` (>= 2.1). No other runtime dependencies without a constitutional amendment.
- **FR-027**: `NeuroMemory.force_dream()` MUST trigger the dreaming pipeline immediately, processing all current inbox memories, and MUST block until completion unless `block=False` is passed.
- **FR-028**: The system MUST prevent concurrent dreaming threads: if a dreaming cycle is already running, a second trigger MUST NOT spawn a second thread.
- **FR-029**: If the dreaming pipeline fails mid-cycle (provider exception), it MUST roll back the batch status from `'dreaming'` to `'inbox'` so the batch is retried in the next cycle.
- **FR-030**: The library MUST expose a `__version__` string in `neuromem.__init__`.

### Key Entities

- **Memory**: A single atomic unit of agent experience. Has a unique UUID, raw content, LLM-generated summary, lifecycle status (`inbox`, `dreaming`, `consolidated`, `archived`), a real-valued `access_weight` initialised at 1.0, and Unix timestamps for creation and last access. Carries optional freeform `metadata` as a JSON-serialisable dict.

- **Node**: A concept node in the knowledge graph. Represents either a leaf tag (e.g., "SQLite") extracted from a memory summary, or a centroid parent category (e.g., "Databases") generated during clustering. Carries a UUID, a human-readable `label`, a serialised embedding vector, and an `is_centroid` boolean flag.

- **Edge**: A directional weighted relationship between two graph entities. Connects node-to-node (taxonomy hierarchy) or node-to-memory (tagging). Carries `source_id`, `target_id`, a cosine `weight` between 0.0 and 1.0, and a `relationship` discriminator string (`has_tag`, `child_of`).

- **StorageAdapter**: The dependency-inversion interface through which all persistence operations are mediated. A pluggable contract that decouples core cognitive logic from any specific database technology.

- **EmbeddingProvider**: An injectable abstraction that converts lists of text strings into lists of float vectors. The library defines the interface; implementations live in separate packages.

- **LLMProvider**: An injectable abstraction providing text summarisation, tag extraction, and cluster category naming. The library defines the interface; implementations live in separate packages.

---

## Success Criteria

### Measurable Outcomes

- **SC-001**: An agent developer can go from `pip install neuromem` to a working `NeuroMemory.enqueue()` call in under 15 minutes, using only the library's own README and docstrings.
- **SC-002**: `enqueue()` must complete and return in under 50 ms on the calling thread **excluding the `LLMProvider.generate_summary()` call**, which is the caller's responsibility and whose latency depends on the injected provider. The dreaming thread is asynchronous and does not count against this budget. Callers that need true hot-path latency MUST supply a fast local or no-op summary provider.
- **SC-003**: `search_memory()` must return a rendered ASCII tree in under 500 ms for a graph containing up to 10,000 nodes, using the `SQLiteAdapter` with local cosine similarity computation.
- **SC-004**: The `StorageAdapter` ABC must be sufficient to implement a fully functional alternative backend (demonstrated by a `DictStorageAdapter` used in the test suite) without modifying any file in `neuromem/` outside `storage/`.
- **SC-005**: Zero runtime imports of any named LLM or embedding vendor SDK (`openai`, `anthropic`, `google-genai`, `cohere`, etc.) and zero imports of any agent framework SDK (Google ADK, LangChain, LlamaIndex, LangGraph, Anthropic SDK agent scaffolding) anywhere in `packages/neuromem-core/src/neuromem/`. Verified by a `tests/test_no_forbidden_imports.py` file-walk test.
- **SC-006**: `packages/neuromem-core/pyproject.toml` runtime `dependencies` list contains exactly three entries: `numpy`, `pandas`, and (optionally, only if needed) a lightweight HTTP client. Verified by parsing the TOML in a test.
- **SC-007**: The library's test suite (using a `MockStorageAdapter` and mock providers) must achieve >= 90% line coverage on `system.py`, `context.py`, `tools.py`, and `math.py`.
- **SC-008**: The system must be evaluated against GraphRAG-Bench, AMB (Agent Memory Benchmark), and LongMemEval as the intended benchmark harnesses. No specific score targets are set at v1; the goal is baseline measurement for future improvement.
- **SC-009**: Archived memories must remain retrievable via `retrieve_memories()` — zero data loss on archival.
- **SC-010**: A dreaming thread failure (provider exception) must leave the database in a consistent state with no orphaned `status='dreaming'` rows after rollback.

---

## Assumptions

- Agent developers are comfortable with Python 3.10+ and with dependency injection patterns.
- The caller is responsible for providing working `LLMProvider` and `EmbeddingProvider` implementations; the library does not validate that the providers produce correct output (only that they return the correct types).
- `enqueue()` is called on conversational text. Binary media (images, audio) is out of scope for v1; `raw_content` is always a UTF-8 string.
- The SQLite database file is local to the process; multi-process concurrent writes to the same SQLite file are out of scope for v1.
- Framework integrations (Google ADK, Anthropic SDK, LangChain hooks) will be separate packages published under the same `uv` workspace monorepo at `packages/neuromem-adk/`, `packages/neuromem-anthropic/`, `packages/neuromem-langchain/`, each depending on `neuromem-core` as a workspace dependency. This spec covers only `packages/neuromem-core/`.
- Embedding dimensionality is determined entirely by the injected `EmbeddingProvider`; the library does not enforce a fixed dimension. The `SQLiteAdapter` MUST log a performance warning if any embedding exceeds 4096 dimensions (JSON-serialised BLOB storage starts to dominate node-table size beyond that).
- The `dream_threshold` default is **10** inbox memories. Configurable via `NeuroMemory(dream_threshold=…)`.
- The decay rate `λ` default is **3e-7 per second**, giving a half-life of roughly 30 days of elapsed time since `last_accessed`. Configurable via `NeuroMemory(decay_lambda=…)`.
- The archive threshold default is `access_weight < 0.1`.
- The agglomerative clustering merge threshold is **0.82** cosine similarity (the same threshold the Gemini design conversation used as an example for edge creation). Configurable as a constructor argument.
- The core library uses a **single** `LLMProvider` instance for all LLM calls (summary, tag extraction, category naming). A second "cheap" provider slot may be added in a future minor version as a non-breaking addition; it is NOT in v1 scope.
- FUSE (filesystem in userspace) mounting is explicitly out of scope for v1. The conversation evolved away from FUSE toward `ContextHelper`/`search_memory` rendering. FUSE remains a future possibility.

---

## Architecture Overview

### Neuroscience Mapping

| Cognitive System | neuromem Subsystem | Key Mechanism |
|---|---|---|
| Hippocampus | Acquisition / Inbox | Fast write, deferred processing, `status='inbox'` |
| Neocortex | Consolidation / Dreaming | Async background thread, agglomerative clustering, centroid naming |
| Synaptic Pruning | Forgetting / Decay | Exponential decay of `access_weight`, archival below threshold |
| Long-Term Potentiation | Retrieval Reinforcement | `access_weight` reset to 1.0 on `retrieve_memories()` |
| Prefrontal Cortex | Contextual Recall | Embedding-based sub-graph traversal, ASCII tree rendering |

### Architectural Principles

**Dependency Inversion.** The core never depends on storage or provider implementations. `NeuroMemory` holds references to `StorageAdapter`, `LLMProvider`, and `EmbeddingProvider` interfaces only.

**Double-Buffer Concurrency.** The `inbox` → `dreaming` → `consolidated` status flip is the sole concurrency primitive. The calling thread always writes to `inbox`; the background thread exclusively touches `dreaming` rows. No explicit database-level locking beyond SQLite's built-in serialisation is needed for v1.

**Knowledge Graph, Not Pure Tree.** The underlying data structure is a directed weighted graph (nodes + edges). It is projected as a tree for rendering using a depth-limited traversal from the nearest matching nodes, with multi-parent relationships expressed via repeated subtree inclusion (mimicking symlinks semantically).

**Local Math, Remote Embeddings.** Embedding generation is remote (provider API). All similarity computation, centroid arithmetic, and clustering are local using numpy (vectorised dot products + broadcasting). pandas is used for tabular transforms during the dreaming cycle (memory/node batch frames) where clarity benefits.

---

## Package Layout

The repository is a `uv` workspace monorepo (Constitution v2.0.0 Additional Constraints — Repository Structure). The v1 target is a single package, `neuromem-core`, living at `packages/neuromem-core/`. Future framework-wrapper packages land as siblings under `packages/` in later features.

```
neuromem/                                        # repo root (git)
├── pyproject.toml                               # uv workspace root:
│                                                # [tool.uv.workspace]
│                                                # members = ["packages/*"]
├── uv.lock                                      # shared lockfile
├── .pre-commit-config.yaml                      # already landed
├── .gitignore                                   # already landed
├── README.md                                    # repo-level (TBD)
├── .specify/                                    # speckit scaffolding
├── specs/001-neuromem-core/                     # this feature
├── CLAUDE.md                                    # agent context file
└── packages/
    └── neuromem-core/                           # v1 package
        ├── pyproject.toml                       # package config:
        │                                        # name = "neuromem-core"
        │                                        # dependencies = [
        │                                        #   "numpy>=1.26",
        │                                        #   "pandas>=2.1",
        │                                        # ]
        ├── README.md                            # package-level
        ├── src/
        │   └── neuromem/                        # import name
        │       ├── __init__.py                  # __version__, re-exports
        │       ├── system.py                    # NeuroMemory orchestration engine
        │       ├── context.py                   # ContextHelper prompt injector
        │       ├── tools.py                     # search_memory, retrieve_memories
        │       ├── vectors.py                   # cosine_similarity, compute_centroid
        │       │                                # (numpy-based; renamed from math.py
        │       │                                # to avoid stdlib name shadowing)
        │       ├── providers.py                 # EmbeddingProvider, LLMProvider ABCs
        │       └── storage/
        │           ├── __init__.py              # re-exports
        │           ├── base.py                  # StorageAdapter ABC
        │           └── sqlite.py                # SQLiteAdapter (stdlib sqlite3 +
        │                                        # numpy for embedding (de)serialisation
        │                                        # and vectorised get_nearest_nodes)
        └── tests/
            ├── __init__.py
            ├── conftest.py                      # MockStorageAdapter, MockLLMProvider,
            │                                    # MockEmbeddingProvider fixtures
            ├── test_vectors.py                  # cosine_similarity, compute_centroid
            ├── test_storage_sqlite.py           # StorageAdapter contract tests
            ├── test_system.py                   # NeuroMemory + dreaming cycle
            ├── test_context.py                  # ContextHelper + ASCII tree
            ├── test_tools.py                    # search_memory, retrieve_memories, LTP
            └── test_no_forbidden_imports.py     # SC-005 enforcement: grep src/
                                                  # for openai/anthropic/etc
```

**Naming note**: The module originally called `math.py` in the first draft is renamed to `vectors.py` here because (a) `math` shadows the stdlib `math` module and confuses tooling, and (b) with numpy in the mix, the module is a vector operations module, not a scalar math module. Public import path is `from neuromem.vectors import cosine_similarity, compute_centroid`.

---

## Public API Surface

### `neuromem.system`

```python
class NeuroMemory:
    def __init__(
        self,
        storage: StorageAdapter,
        llm: LLMProvider,
        embedder: EmbeddingProvider,
        dream_threshold: int = 10,
        decay_lambda: float = 0.01,
        archive_threshold: float = 0.1,
    ) -> None: ...

    def enqueue(self, raw_text: str, metadata: dict | None = None) -> str:
        """Insert raw_text into inbox. Returns the new memory UUID.
        Triggers background dreaming when inbox count >= dream_threshold."""

    def force_dream(self, block: bool = True) -> None:
        """Manually trigger the dreaming/consolidation pipeline.
        If block=True, waits for completion before returning."""

    @property
    def is_dreaming(self) -> bool:
        """True if a dreaming thread is currently running."""
```

### `neuromem.context`

```python
class ContextHelper:
    def __init__(self, memory_system: NeuroMemory) -> None: ...

    def build_prompt_context(
        self,
        current_task_description: str,
        top_k: int = 5,
        depth: int = 2,
    ) -> str:
        """Embed task description, find nearest nodes, traverse sub-graph,
        return formatted ASCII tree string for system prompt injection.
        Returns empty string if no relevant nodes found."""
```

### `neuromem.tools`

```python
def search_memory(
    query: str,
    system: NeuroMemory,
    top_k: int = 5,
    depth: int = 2,
) -> str:
    """Embed query, find nearest graph nodes, render ASCII sub-graph.
    Returns formatted string with node labels and memory IDs.
    Returns empty string if graph contains no nodes."""

def retrieve_memories(
    memory_ids: list[str],
    system: NeuroMemory,
) -> list[dict]:
    """Fetch full memory records by ID. Triggers LTP (access_weight=1.0)
    for each consolidated memory found. Silently skips missing IDs.

    Each returned dict contains:
      id: str
      raw_content: str
      summary: str
      status: str
      access_weight: float
      created_at: int      # Unix timestamp
      last_accessed: int   # Unix timestamp
      metadata: dict | None
    """
```

### `neuromem.vectors`

```python
import numpy as np
from numpy.typing import NDArray

def cosine_similarity(
    vec_a: NDArray[np.floating] | list[float],
    vec_b: NDArray[np.floating] | list[float],
) -> float:
    """Cosine similarity in [-1.0, 1.0] (practically [0.0, 1.0] for embeddings).
    Accepts numpy arrays or Python lists (the latter are converted).
    Returns 0.0 if either vector has zero magnitude.
    Raises ValueError on length mismatch."""

def batch_cosine_similarity(
    query: NDArray[np.floating],
    matrix: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Vectorised cosine similarity between one query vector and an (N, D)
    matrix of candidate vectors. Returns a 1-D array of N similarity scores.
    Used by SQLiteAdapter.get_nearest_nodes() for the full-table scan path."""

def compute_centroid(
    vectors: NDArray[np.floating] | list[NDArray[np.floating]] | list[list[float]],
) -> NDArray[np.float64]:
    """Element-wise mean. Accepts a 2-D array OR a list of 1-D arrays/lists.
    Returns a new 1-D numpy array. Raises ValueError if empty or if lengths
    differ."""
```

### `neuromem.providers`

```python
from abc import ABC, abstractmethod

class EmbeddingProvider(ABC):
    @abstractmethod
    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text, in the same order."""

class LLMProvider(ABC):
    @abstractmethod
    def generate_summary(self, raw_text: str) -> str:
        """Return a 1–2 sentence episodic summary of raw_text."""

    @abstractmethod
    def extract_tags(self, summary: str) -> list[str]:
        """Return a list of discrete concept label strings from summary."""

    @abstractmethod
    def generate_category_name(self, concepts: list[str]) -> str:
        """Given a list of concept labels, return a single one-word
        category name that encompasses them (e.g., ['SQLite','Neo4j'] -> 'Databases')."""
```

---

## Storage Adapter Interface

All storage backends MUST implement the following ABC. Method contracts are binding; implementations MAY add vendor-specific configuration in their `__init__` but MUST NOT require callers to use it.

```python
# neuromem/storage/base.py
from abc import ABC, abstractmethod
from typing import Any

class StorageAdapter(ABC):

    # --- Acquisition (Hippocampus) ---

    @abstractmethod
    def insert_memory(
        self,
        raw_content: str,
        summary: str,
        metadata: dict | None = None,
    ) -> str:
        """Persist a new memory with status='inbox', access_weight=1.0.
        Returns the new memory UUID."""

    @abstractmethod
    def count_memories_by_status(self, status: str) -> int:
        """Return the count of memories with the given status."""

    @abstractmethod
    def get_memories_by_status(
        self,
        status: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return memory records matching status, up to limit rows."""

    @abstractmethod
    def update_memory_status(
        self,
        memory_ids: list[str],
        new_status: str,
    ) -> None:
        """Atomically flip the status of the given memory IDs."""

    @abstractmethod
    def get_memory_by_id(self, memory_id: str) -> dict[str, Any] | None:
        """Return a single memory record, or None if not found."""

    # --- Consolidation (Neocortex / Graph) ---

    @abstractmethod
    def upsert_node(
        self,
        node_id: str,
        label: str,
        embedding: list[float],
        is_centroid: bool,
    ) -> None:
        """Insert or update a concept node."""

    @abstractmethod
    def get_all_nodes(self) -> list[dict[str, Any]]:
        """Return all active (non-archived) nodes with their embeddings."""

    @abstractmethod
    def insert_edge(
        self,
        source_id: str,
        target_id: str,
        weight: float,
        relationship: str,
    ) -> None:
        """Insert a directed edge between two entities."""

    @abstractmethod
    def remove_edges_for_memory(self, memory_id: str) -> None:
        """Remove all edges connected to memory_id (used on archival)."""

    # --- Recall (Prefrontal Cortex) ---

    @abstractmethod
    def get_nearest_nodes(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return the top_k nodes closest to query_embedding.
        SQLite adapter: fetches all nodes, computes cosine similarity locally.
        Native-vector adapters: delegate to DB-level ANN search."""

    @abstractmethod
    def get_subgraph(
        self,
        root_node_ids: list[str],
        depth: int = 2,
    ) -> dict[str, Any]:
        """Traverse edges from root_node_ids up to depth hops.
        Returns a dict with 'nodes' (list) and 'edges' (list) keys,
        plus 'memories' (list) for memories attached via has_tag edges."""

    # --- Forgetting (Synaptic Pruning) ---

    @abstractmethod
    def apply_decay_and_archive(
        self,
        decay_lambda: float,
        archive_threshold: float,
        current_timestamp: int,
    ) -> list[str]:
        """For every consolidated memory, compute W_new = W_old * exp(-λ * t)
        where t = current_timestamp - last_accessed.
        Update access_weight. Archive (set status='archived', remove edges)
        any memory where W_new < archive_threshold.
        Returns list of archived memory IDs."""

    @abstractmethod
    def spike_access_weight(
        self,
        memory_ids: list[str],
        timestamp: int,
    ) -> None:
        """Set access_weight=1.0 and last_accessed=timestamp for each ID
        (Long-Term Potentiation)."""
```

---

## Provider Interfaces

Detailed in the Public API Surface section above. Additional design notes:

- `LLMProvider.extract_tags()` is a separate method from `generate_summary()` because different use cases may want to combine these into one API call or keep them separate, and because framework wrappers (e.g., structured output providers) may implement them differently.
- `generate_category_name()` MUST return a single string of at most one word (no spaces). If the LLM returns multiple words, `NeuroMemory` will take the first word and log a warning.
- `EmbeddingProvider.get_embeddings()` accepts a batch of texts and returns a batch of vectors. Callers always pass lists, even for single texts, to reduce API round-trips during dreaming.
- Providers are not required to be thread-safe by the contract; `NeuroMemory` calls providers only from the dreaming thread, not from the `enqueue()` call path.

---

## Data Model

### SQLite Schema (canonical — SQLiteAdapter MUST implement exactly this)

```sql
CREATE TABLE IF NOT EXISTS memories (
    id           TEXT PRIMARY KEY,          -- UUID v4
    raw_content  TEXT NOT NULL,
    summary      TEXT,                      -- populated after LLM summary call
    status       TEXT NOT NULL DEFAULT 'inbox',
                                            -- 'inbox' | 'dreaming' |
                                            -- 'consolidated' | 'archived'
    access_weight REAL NOT NULL DEFAULT 1.0,
    last_accessed INTEGER,                  -- Unix epoch seconds
    created_at   INTEGER NOT NULL,          -- Unix epoch seconds
    metadata     TEXT                       -- JSON string or NULL
);

CREATE TABLE IF NOT EXISTS nodes (
    id           TEXT PRIMARY KEY,          -- UUID v4
    label        TEXT NOT NULL,             -- human-readable concept name
    embedding    BLOB NOT NULL,             -- numpy float32 bytes
                                            -- (np.ndarray.astype(np.float32).tobytes())
    embedding_dim INTEGER NOT NULL,         -- length of the array, for
                                            -- np.frombuffer reshape on read
    is_centroid  INTEGER NOT NULL DEFAULT 0 -- 0=leaf tag, 1=centroid parent
);

CREATE TABLE IF NOT EXISTS edges (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id    TEXT NOT NULL,
    target_id    TEXT NOT NULL,
    weight       REAL NOT NULL DEFAULT 1.0, -- cosine similarity
    relationship TEXT NOT NULL,             -- 'has_tag' | 'child_of'
    UNIQUE(source_id, target_id, relationship)
);
```

**Embedding storage in SQLiteAdapter.** Embeddings are stored as raw numpy `float32` bytes in a `BLOB` column via `arr.astype(np.float32).tobytes()`. On read, `np.frombuffer(blob, dtype=np.float32)` reconstructs the 1-D array (the `embedding_dim` column is redundant check data but catches corruption). This is ~4× more compact than JSON text and ~10× faster to deserialise; the debuggability loss is acceptable because an operator can always `SELECT id, label FROM nodes` and pull embeddings programmatically when needed. No SQLite vector extension is required.

**Status state machine.**

```
inbox -> dreaming -> consolidated -> archived
  ^                        |
  |_____ (rollback on error)
```

New arrivals during an active dream cycle are always inserted as `inbox` and processed in a subsequent cycle.

**Index recommendations for SQLiteAdapter.**

```sql
CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status);
CREATE INDEX IF NOT EXISTS idx_edges_source    ON edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target    ON edges(target_id);
```

---

## Subsystem Workflows

### Phase A — Acquisition

```
Caller thread:
  enqueue(raw_text, metadata)
    -> llm.generate_summary(raw_text)           # LLM call (blocking, in caller thread)
    -> storage.insert_memory(raw_content, summary, metadata)
       status='inbox', access_weight=1.0, created_at=now()
    -> storage.count_memories_by_status('inbox')
    -> if count >= dream_threshold and not is_dreaming:
         spawn background thread -> _run_dream_cycle()
    -> return memory_id
```

Note: The LLM summary call happens in the caller thread per Resolved Design Decision #3 (KISS). Callers requiring low latency must inject a fast or no-op `generate_summary` provider.

### Phase B — Consolidation / Dreaming

```
Background thread: _run_dream_cycle()
  1. Acquire dreaming lock (return immediately if already locked)
  2. storage.update_memory_status(inbox_ids -> 'dreaming')   # atomic flip
  3. For each dreaming memory:
       tags = llm.extract_tags(memory.summary)
  4. all_new_tags = deduplicated union of all extracted tags
  5. new_embeddings = embedder.get_embeddings(all_new_tags)  # batched API call
  6. existing_nodes = storage.get_all_nodes()
  7. merged_tags = existing_nodes + new_tags_with_embeddings
  8. Run agglomerative clustering on merged_tags:
       a. Build similarity matrix (pairwise cosine_similarity)
       b. Find pair with highest similarity above cluster_threshold
       c. Merge pair -> compute centroid via compute_centroid()
       d. Call llm.generate_category_name(member_labels) -> parent_label
       e. storage.upsert_node(centroid) with is_centroid=True
       f. storage.insert_edge(centroid -> each_member, 'child_of')
       g. Repeat until no pair exceeds threshold
  9. For each (memory, tags) pair:
       storage.insert_edge(memory_id -> tag_node_id, 'has_tag')
  10. storage.apply_decay_and_archive(decay_lambda, archive_threshold, now())
  11. storage.update_memory_status(dreaming_ids -> 'consolidated')
  12. Release dreaming lock

  On any exception in steps 3-11:
    storage.update_memory_status(dreaming_ids -> 'inbox')  # rollback
    log error
    release dreaming lock
```

### Phase C — Synaptic Pruning (integrated into Phase B, step 10)

The decay-and-archive operation is called at the end of every dreaming cycle, not on a separate timer. This ensures pruning only consumes compute when the system is already doing background work.

Formula: `W_new = W_old * exp(-λ * t)` where `t` is seconds since `last_accessed`.

If `W_new < archive_threshold`:
- Set `status = 'archived'`
- Call `storage.remove_edges_for_memory(memory_id)`
- The memory row itself is preserved (raw_content, summary remain intact)

### Phase D — Contextual Recall

```
ContextHelper.build_prompt_context(task_description):
  1. query_vec = embedder.get_embeddings([task_description])[0]
  2. nearest = storage.get_nearest_nodes(query_vec, top_k)
  3. if nearest is empty: return ""
  4. subgraph = storage.get_subgraph([n['id'] for n in nearest], depth)
  5. return _render_ascii_tree(subgraph)

_render_ascii_tree(subgraph) -> str:
  Produces output of the form:
    Relevant Memory Context:
    📁 Databases
    ├── 📁 SQLite
    │   ├── 📄 mem_7732: "WAL mode discussion..."
    │   └── 📄 mem_9104: "Massive inserts perf..."
    └── 📁 Indexing
        └── 📄 mem_4421: "Composite indexes..."
```

The graph is NOT a pure tree: a node may appear under multiple parents. The renderer handles this by including the node under each parent that appears in the traversal result, with a note `(also under: ...)` on subsequent appearances to avoid full subtree duplication.

---

## Concurrency Model

- `enqueue()` is called from the **agent's main thread** (or event loop). It must never block on I/O beyond the LLM summary call and a fast SQLite write.
- The dreaming pipeline runs in a single **background daemon thread** spawned via `threading.Thread(daemon=True)`.
- A `threading.Lock` (`_dream_lock`) on the `NeuroMemory` instance prevents concurrent dreaming cycles.
- The double-buffer pattern (status flip from `inbox` to `dreaming`) is the only inter-thread synchronisation mechanism needed for data safety; SQLite handles write serialisation at the DB level.
- `force_dream(block=True)` joins the spawned thread before returning.
- `force_dream(block=False)` spawns the thread and returns immediately.
- The library does not use `asyncio`. Async compatibility is left to framework wrapper packages (which can run `enqueue()` via `asyncio.to_thread()` or a thread pool executor).
- Multi-process safety (multiple processes writing to the same SQLite file) is out of scope for v1.

---

## Resolved Design Decisions

All design questions raised during source discussions are now resolved. The following decisions are binding for v1 and MUST NOT be revisited without a spec amendment.

1. **Library name: `neuromem`.** Confirmed as the canonical PyPI package name. Alternatives (`engram`, `synaptree`, `cortex-db`, `hippocamp`, `onto-mem`, `mem-fs`) proposed in the Gemini design conversation are rejected for v1.

2. **Zero-similarity recall: return empty string.** When `search_memory()` or `ContextHelper.build_prompt_context()` is called with a query whose embedding has no meaningful cosine similarity to any node in the active graph, the system MUST return an empty string immediately. No on-demand dreaming cycle is triggered; no warning is logged. The agent is responsible for interpreting an empty result as "no relevant memory".

   **Rationale**: Simplicity. Search calls stay deterministic and fast. If the inbox is genuinely important, the next `enqueue()` will trigger dreaming via the threshold mechanism anyway.

3. **Summary generation: caller thread (KISS).** `LLMProvider.generate_summary()` is called synchronously inside `enqueue()` on the caller thread. If the agent's latency budget cannot afford a real LLM call at every conversation turn, the caller MUST supply a fast summary provider (mock, local model, or no-op that passes `raw_text` through as the summary). No configurable sync/async mode is exposed in v1 — keep it simple.

   **Rationale**: A configurable `summary_mode` argument doubles the test matrix and introduces a state (`summary=NULL` until dreaming) that the rest of the pipeline would have to handle. Pushing the choice back to the caller's provider implementation is the simplest contract.

4. **Dream threshold N** → default 10 (configurable via `NeuroMemory(dream_threshold=…)`).

5. **Decay rate λ** → default 3e-7 per second (~30-day half-life, configurable via `NeuroMemory(decay_lambda=…)`).

6. **Embedding dimensionality** → no enforced limit; `SQLiteAdapter` MUST log a performance warning above 4096 dimensions.

7. **LLM provider multiplicity** → single `LLMProvider` instance handles all LLM calls (summary, tag extraction, category naming) in v1. A second "cheap-model" provider slot MAY be added in a future minor version as a non-breaking addition.

8. **Clustering merge threshold** → default 0.82 cosine similarity (from the Gemini conversation's edge-creation example, configurable).
