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
- What happens when `search_memory` is called with a query that has zero cosine similarity to every node in the graph? See the Resolved Design Decisions appendix (commit F02): return empty string immediately (KISS).
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
- **FR-024**: The `SQLiteAdapter.get_nearest_nodes()` MUST implement cosine similarity locally using the pure-Python functions in `neuromem.math` (no numpy required).
- **FR-025**: The library MUST provide `EmbeddingProvider` and `LLMProvider` ABCs in `neuromem.providers`. The core library MUST NOT import `openai`, `anthropic`, `google-genai`, or any specific LLM/embedding SDK.
- **FR-026**: The library MUST be packaged with `pyproject.toml` and managed with `uv`. Runtime dependencies MUST be limited to the Python 3.x standard library; `requests` (or `httpx`) is the only permitted non-stdlib runtime dependency for optional HTTP utilities.
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
- **SC-005**: Zero runtime imports of any named LLM or embedding vendor SDK (`openai`, `anthropic`, `google-genai`, `cohere`, etc.) anywhere in `src/neuromem/`.
- **SC-006**: The `SQLiteAdapter` must have zero non-stdlib runtime dependencies (verified by checking `pyproject.toml` optional-dependency groups).
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
- Framework integrations (Google ADK, Anthropic SDK, LangChain hooks) will be separate PyPI packages; this spec covers only the core library.
- Embedding dimensionality is determined entirely by the injected `EmbeddingProvider`; the library does not enforce a fixed dimension. The `SQLiteAdapter` MUST log a performance warning if any embedding exceeds 4096 dimensions (JSON-serialised BLOB storage starts to dominate node-table size beyond that).
- The `dream_threshold` default is **10** inbox memories. Configurable via `NeuroMemory(dream_threshold=…)`.
- The decay rate `λ` default is **3e-7 per second**, giving a half-life of roughly 30 days of elapsed time since `last_accessed`. Configurable via `NeuroMemory(decay_lambda=…)`.
- The archive threshold default is `access_weight < 0.1`.
- The agglomerative clustering merge threshold is **0.82** cosine similarity (the same threshold the Gemini design conversation used as an example for edge creation). Configurable as a constructor argument.
- The core library uses a **single** `LLMProvider` instance for all LLM calls (summary, tag extraction, category naming). A second "cheap" provider slot may be added in a future minor version as a non-breaking addition; it is NOT in v1 scope.
- FUSE (filesystem in userspace) mounting is explicitly out of scope for v1. The conversation evolved away from FUSE toward `ContextHelper`/`search_memory` rendering. FUSE remains a future possibility.

<!--
  The engineering appendix (Architecture Overview, Package Layout, Public API
  Surface, Storage Adapter Interface, Provider Interfaces, Data Model,
  Subsystem Workflows, Concurrency Model, Resolved Design Decisions) is
  landed in a follow-up commit to keep each commit under the 500-line cap
  mandated by Constitution Principle VI.
-->
