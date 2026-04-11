---

description: "Task list for the neuromem-core implementation (001-neuromem-core)"
---

# Tasks: neuromem Core Library

**Input**: Design documents from `/specs/001-neuromem-core/`
**Prerequisites**: spec.md, plan.md, research.md, data-model.md, contracts/ (all landed)
**Constitution**: v2.0.0

**Tests**: Test tasks are included throughout. Per Constitution v2.0.0 Principle V (NON-NEGOTIABLE), every implementation MUST follow the red-green-refactor cycle. Because Principle VI mandates pre-commit hooks that run `pytest` on every commit, **failing tests cannot land in their own commit** — a failing-test commit would be blocked by the hook. Therefore test tasks and their corresponding implementation tasks are **co-committed as a single atomic unit** (Principle V explicitly permits "co-committed" as an alternative to "preceding"). Each task row below represents one commit unless explicitly noted otherwise.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing. Within each user story phase, tests come first in the row order (write them, watch them fail locally), then implementations make them pass, then the whole unit is staged and committed together.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel with other [P] tasks (different files, no shared state, no dependency on incomplete tasks in the same phase)
- **[Story]**: `[US1]`..`[US5]` — only on user-story phase tasks. Setup, Foundational, and Polish phases have no story label.
- File paths are absolute from the repo root unless obvious from context.

## Path Conventions

This is a `uv` workspace monorepo (Constitution v2.0.0). All package code and tests live under `packages/neuromem-core/`. The core package imports as `neuromem`.

- **Package source**: `packages/neuromem-core/src/neuromem/`
- **Package tests**: `packages/neuromem-core/tests/`
- **Package config**: `packages/neuromem-core/pyproject.toml`
- **Workspace config**: `pyproject.toml` (repo root)
- **Spec artefacts**: `specs/001-neuromem-core/` (already landed)

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Bootstrap the `uv` workspace monorepo and the `neuromem-core` package skeleton.

- [X] T001 Create repo-root workspace `pyproject.toml` with `[tool.uv.workspace]` declaring `members = ["packages/*"]`. Add shared `[tool.ruff]` config (line-length, target-version = "py310") and `[tool.pytest.ini_options]` (minimum-version, testpaths pointing at `packages/*/tests`). NO `[project]` table — the root is a workspace, not a package.
- [ ] T002 Create `packages/neuromem-core/pyproject.toml` with `[project]` section (`name = "neuromem-core"`, `version = "0.1.0"`, `requires-python = ">=3.10"`, `dependencies = ["numpy>=1.26", "pandas>=2.1"]`), `[dependency-groups] dev = ["pytest>=8", "pytest-cov>=5", "ruff>=0.6"]`, `[build-system]` (`requires = ["hatchling"]`, `build-backend = "hatchling.build"`), and `[tool.hatch.build.targets.wheel] packages = ["src/neuromem"]`.
- [ ] T003 [P] Create `packages/neuromem-core/README.md` placeholder with package name, one-sentence description, link to `specs/001-neuromem-core/spec.md`, and "under construction" notice. Final README lands in T031.
- [ ] T004 [P] Create `packages/neuromem-core/src/neuromem/__init__.py` with `__version__ = "0.1.0"`. Leave re-exports commented out; they'll be uncommented as downstream modules land.
- [ ] T005 [P] Create `packages/neuromem-core/src/neuromem/storage/__init__.py` as an empty package marker. Create `packages/neuromem-core/tests/__init__.py` (empty) and `packages/neuromem-core/tests/conftest.py` with a single `pytest_plugins = []` line as a placeholder (fixtures added in T011).
- [ ] T006 Run `uv sync --dev` at the repo root. Verify both TOML files parse, the workspace is detected, `neuromem-core` installs in editable mode, and `uv tree packages/neuromem-core` shows `numpy` and `pandas` as the only non-stdlib runtime deps.
- [ ] T007 Update `.pre-commit-config.yaml` pytest hook entry to run `uv run --package neuromem-core pytest -q -x packages/neuromem-core/tests/` (so the hook finds the monorepo test location and runs inside the workspace). Verify the hook still passes on a no-op commit (`git commit --allow-empty`).

**Checkpoint**: Workspace is installable, pre-commit hooks active, empty package skeleton in place. Ready for Foundational phase.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Build the contract layer (ABCs), the numerical helpers, and the mock fixtures that every user story depends on.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete. Principle III's layered architecture is established here — the ABCs below define the contract layer that every subsequent task imports from.

Each task below is one atomic commit containing both the failing tests and the implementation that makes them pass (Principle V co-commit pattern).

- [ ] T008 [P] Test-first `neuromem.vectors` module. Write failing tests in `packages/neuromem-core/tests/test_vectors.py` covering: `cosine_similarity` on equal vectors (→ 1.0), orthogonal vectors (→ 0.0), zero-magnitude input (→ 0.0 not NaN), length mismatch (→ ValueError), Python list input accepted; `batch_cosine_similarity` on a `(5, 16)` matrix with known-similarity rows; `compute_centroid` on 2-D array input, list-of-1D-arrays input, empty input (→ ValueError), length-mismatched inputs (→ ValueError). Then implement `packages/neuromem-core/src/neuromem/vectors.py` with numpy-backed `cosine_similarity`, `batch_cosine_similarity`, and `compute_centroid` per contracts/public-api.md §`neuromem.vectors`.
- [ ] T009 [P] Test-first `neuromem.providers` ABCs. Write failing tests in `packages/neuromem-core/tests/test_providers.py` asserting: (a) `EmbeddingProvider()` raises `TypeError`; (b) `LLMProvider()` raises `TypeError`; (c) a subclass that implements only 2 of the 3 `LLMProvider` methods still raises `TypeError` at instantiation; (d) a fully implemented stub subclass instantiates successfully. Then implement `packages/neuromem-core/src/neuromem/providers.py` with `EmbeddingProvider` (one abstract method `get_embeddings`) and `LLMProvider` (three abstract methods: `generate_summary`, `extract_tags`, `generate_category_name`) per contracts/providers.md.
- [ ] T010 [P] Test-first `neuromem.storage.base.StorageAdapter` ABC + `StorageError` exception class. Write failing tests in `packages/neuromem-core/tests/test_storage_base.py` asserting: (a) `StorageAdapter()` raises `TypeError`; (b) a subclass missing any one of the 13 abstract methods raises `TypeError`; (c) a fully-stubbed subclass instantiates; (d) `StorageError` is a `RuntimeError` subclass. Then implement `packages/neuromem-core/src/neuromem/storage/base.py` with the `StorageAdapter` ABC (13 `@abstractmethod` methods per contracts/storage-adapter.md) and a `StorageError` class.
- [ ] T011 Add `MockEmbeddingProvider` and `MockLLMProvider` fixtures to `packages/neuromem-core/tests/conftest.py`. `MockEmbeddingProvider` returns deterministic seeded-RNG float32 numpy arrays per contracts/providers.md §`MockEmbeddingProvider`; `MockLLMProvider` returns canned responses (`generate_summary` = first 80 chars, `extract_tags` = first 3 alpha tokens, `generate_category_name` = `"Cat" + first-letters`). Export both as pytest fixtures. Depends on T009 (provider ABCs must exist).
- [ ] T012 Write the 13 contract test function skeletons for `StorageAdapter` in `packages/neuromem-core/tests/test_storage_adapter_contract.py`, parameterised over a `storage_adapter` pytest fixture. The fixture lives in `conftest.py` and initially has `params=[]` (empty), which makes pytest skip every parameterised test with no failures. Later tasks (T018 for SQLiteAdapter, T025 for DictStorageAdapter) add their adapters to the params list. Test bodies come from contracts/storage-adapter.md §Contract tests (13 items). Depends on T010.

**Checkpoint**: Contract layer complete. Mock providers available. Empty StorageAdapter contract test skeleton in place. User story phases can now begin.

---

## Phase 3: User Story 1 — Agent Developer Wires Up Long-Term Memory (Priority: P1) 🎯 MVP

**Goal**: An agent developer can instantiate `NeuroMemory` with the shipped `SQLiteAdapter` and injected providers, call `enqueue()` on every turn, and inject the resulting context into the next LLM call via `ContextHelper.build_prompt_context()`. This is the complete passive cognitive loop: acquisition → dreaming → contextual recall.

**Independent Test**: Instantiate `NeuroMemory(storage=SQLiteAdapter(":memory:"), llm=MockLLMProvider(), embedder=MockEmbeddingProvider())` with `dream_threshold=3`. Call `enqueue("text 1")`, `enqueue("text 2")`, `enqueue("text 3")`, then `force_dream(block=True)`. Call `ContextHelper(memory).build_prompt_context("any query")` and assert the result is a non-empty string containing at least one `📄 mem_` line.

### SQLiteAdapter (implementation of the contract from Phase 2)

Each SQLiteAdapter task incrementally fills `packages/neuromem-core/src/neuromem/storage/sqlite.py` and adds a corresponding block of assertions to `packages/neuromem-core/tests/test_storage_adapter_contract.py` (via the params list). Because all six tasks write to the same `sqlite.py` file, they are **sequential** (no [P] marker).

- [ ] T013 [US1] Implement `SQLiteAdapter.__init__`, schema init (DDL for `memories`/`nodes`/`edges` + indexes per data-model.md), `insert_memory`, `get_memory_by_id`, and `count_memories_by_status` in `packages/neuromem-core/src/neuromem/storage/sqlite.py`. Add `SQLiteAdapter` to the `storage_adapter` fixture params in `conftest.py` so the T012 contract tests start running against it. Tests pass: contract tests 1 (construction), 2 (insert_memory round-trip), and 4 (count_memories_by_status).
- [ ] T014 [US1] Implement `SQLiteAdapter.get_memories_by_status` and `SQLiteAdapter.update_memory_status` (atomic multi-row flip via single SQL statement — this is the double-buffer primitive used by the dreaming cycle). Extends `sqlite.py`. Contract test 3 (status state machine round-trip) now passes.
- [ ] T015 [US1] Implement `SQLiteAdapter.upsert_node` (INSERT OR REPLACE), `SQLiteAdapter.get_all_nodes` (deserialise embeddings via `np.frombuffer(blob, dtype=np.float32)`), and the `embedding_dim` corruption check (`len(blob) // 4 != embedding_dim → StorageError`). Extends `sqlite.py`. Contract tests 5 (upsert round-trip), 6 (dimension mismatch → ValueError) now pass.
- [ ] T016 [US1] Implement `SQLiteAdapter.insert_edge` (with `INSERT OR IGNORE` for idempotency), `SQLiteAdapter.remove_edges_for_memory`, and `SQLiteAdapter.get_subgraph` (the graph-traversal method: BFS up to `depth` hops following `child_of` edges in both directions, collects `has_tag` edges to reach memories, filters out archived memories). Extends `sqlite.py`. Contract tests 7 (edge idempotency) and 9 (get_subgraph) now pass.
- [ ] T017 [US1] Implement `SQLiteAdapter.get_nearest_nodes` — fetch all rows from `nodes`, stack embeddings into a single `(N, D)` `np.float32` matrix, call `neuromem.vectors.batch_cosine_similarity(query, matrix)` once, `np.argpartition` for top-k, sort and return. Adds a `similarity` key to each returned dict. Extends `sqlite.py`. Contract test 8 (nearest-nodes ranking) now passes.
- [ ] T018 [US1] Implement `SQLiteAdapter.apply_decay_and_archive` (exponential decay `W_new = W_old * exp(-λ * (now - last_accessed))`; set `status='archived'` and call `remove_edges_for_memory` when `W_new < archive_threshold`; return archived IDs) and `SQLiteAdapter.spike_access_weight` (reset `access_weight=1.0` and `last_accessed=timestamp` for consolidated memories only). Extends `sqlite.py`. Contract tests 10 (decay + archive), 11 (LTP ignores non-consolidated), 12 (archival preserves content), 13 (missing ID silent skip) now pass. **All 13 contract tests must now pass against SQLiteAdapter.**

### NeuroMemory orchestration engine

- [ ] T019 [US1] Implement `NeuroMemory.__init__` (constructor validation, stores injected deps, creates `_dream_lock`) and `NeuroMemory.enqueue` synchronous path (calls `llm.generate_summary`, calls `storage.insert_memory`, calls `storage.count_memories_by_status('inbox')`, checks threshold — but does NOT yet spawn a background thread; that's T021). File: `packages/neuromem-core/src/neuromem/system.py`. Tests in `packages/neuromem-core/tests/test_system.py`: constructor error cases (invalid `dream_threshold`, wrong provider type), enqueue round-trip, enqueue latency budget (SC-002: under 50 ms excluding the provider call). Depends on T013–T018 (needs working SQLiteAdapter).
- [ ] T020 [US1] Implement `NeuroMemory._run_dream_cycle` private method — the complete dreaming pipeline per spec.md §Phase B: acquire `_dream_lock`, flip batch to `status='dreaming'`, call `llm.extract_tags` per memory, dedupe and embed new tags via `embedder.get_embeddings` (one batched call), fetch existing nodes, run the agglomerative clustering loop (numpy pairwise similarity matrix + `np.argmax` + pandas DataFrame for node bookkeeping + `llm.generate_category_name` at each merge), write centroid nodes and `child_of` edges, write `has_tag` edges from memories to their tag nodes, call `apply_decay_and_archive`, flip batch to `status='consolidated'`, release lock. On any exception in the critical section: rollback the batch status from `dreaming` back to `inbox`, log the error, release lock. Extends `system.py` and `tests/test_system.py`. Tests cover: happy path with 3-memory batch → 3 consolidated rows + new tag nodes + expected edges; provider exception → rollback; cluster merge → centroid node created with LLM-named label.
- [ ] T021 [US1] Implement `NeuroMemory.enqueue` background thread spawning (when `count >= dream_threshold and not self.is_dreaming`, spawn `threading.Thread(target=self._run_dream_cycle, daemon=True).start()`); `NeuroMemory.is_dreaming` property (returns whether `_dream_lock` is currently held); `NeuroMemory.force_dream(block=True)` method (spawns the cycle immediately; if `block=True`, `thread.join()`s before returning). Extends `system.py` and `tests/test_system.py`. Concurrency tests: enqueuing past the threshold spawns exactly one thread; concurrent enqueue during dreaming does NOT spawn a second thread; `force_dream(block=True)` waits; `force_dream(block=False)` returns immediately; double-buffer correctness — memories inserted during an active dreaming cycle are correctly flipped in the next cycle, not the current one.
- [ ] T022 [US1] Implement `ContextHelper` class and the private `_render_ascii_tree` helper. `build_prompt_context(task, top_k=5, depth=2)` → embed the task via `embedder.get_embeddings([task])[0]`, call `storage.get_nearest_nodes(query, top_k)`, call `storage.get_subgraph(root_ids, depth)`, render via `_render_ascii_tree`. Empty graph returns `""` (Resolved Design Decision #2). Multi-parent nodes render under each parent with `(also under: …)` suffix on second and later mentions. File: `packages/neuromem-core/src/neuromem/context.py` + `tests/test_context.py`. Tests: ASCII output contains `📁 ` node markers and `📄 mem_` memory markers; empty graph → empty string; multi-parent handling correct.

**Checkpoint** (US1 complete): The full cognitive loop works with the SQLiteAdapter + MockLLMProvider + MockEmbeddingProvider. The Independent Test above passes. This is the **MVP** — a deployable state where an agent can be wired up with passive context injection.

---

## Phase 4: User Story 2 — Agent Uses `search_memory` Tool (Priority: P1)

**Goal**: Expose `search_memory(query)` and `retrieve_memories(ids)` as framework-agnostic tool functions that agent wrapper packages can bind directly. `retrieve_memories` triggers Long-Term Potentiation (access_weight spike) for consolidated memories.

**Independent Test**: With the same setup as US1's Independent Test, call `search_memory("database performance", system=memory)` and assert it returns a non-empty ASCII tree string. Extract a memory ID from that output, call `retrieve_memories([that_id], system=memory)`, assert the returned dict has `raw_content` and `summary` populated and that the memory's `access_weight` was spiked to 1.0.

- [ ] T023 [US2] Implement `neuromem.tools.search_memory(query, system, top_k=5, depth=2) → str` (thin wrapper: `ContextHelper(system).build_prompt_context(query, top_k, depth)`) and `neuromem.tools.retrieve_memories(memory_ids, system) → list[dict]` (loops `storage.get_memory_by_id`, filters None, calls `storage.spike_access_weight(consolidated_ids, now)` for LTP, returns full dicts). File: `packages/neuromem-core/src/neuromem/tools.py` + `packages/neuromem-core/tests/test_tools.py`. Tests cover: `search_memory` returns same content as `ContextHelper.build_prompt_context`; `retrieve_memories` with mix of valid/missing IDs returns only valid rows; LTP spike happens only for consolidated memories (inbox/dreaming/archived memories are returned but their `access_weight` is unchanged); `retrieve_memories([])` → `[]`.

**Checkpoint** (US2 complete): Agent-facing tool functions work. Framework wrapper packages can now bind them directly. Combined with US1, this is the full passive + active recall loop.

---

## Phase 5: User Story 3 — Memories Decay and Are Archived When Not Accessed (Priority: P2)

**Goal**: Validate that the decay + archival pipeline (implemented as part of US1 T018/T020) actually archives memories when their `access_weight` drops below the threshold, and that archived memories are preserved but invisible to active recall.

**Independent Test**: Insert a `consolidated` memory directly into the SQLite database with `last_accessed` set to 60 days in the past and `access_weight=0.3`. Call `force_dream(block=True)`. Assert the memory's `status` is now `'archived'`, its edges are gone from active traversal, and `search_memory("matching query")` does NOT return it. Call `retrieve_memories([its_id])` and assert the `raw_content` is still intact.

- [ ] T024 [US3] Add end-to-end integration tests for the decay and archival lifecycle to `packages/neuromem-core/tests/test_system.py`. Scenarios: (a) directly-injected consolidated memory with `last_accessed` 60 days past gets archived after `force_dream`; (b) consolidated memory with `last_accessed` 7 days past + `decay_lambda=3e-7` stays consolidated with slightly reduced `access_weight`; (c) `retrieve_memories` on an archived memory returns `raw_content`/`summary` but does NOT flip status back to `consolidated` and does NOT trigger LTP; (d) archived memory no longer appears in `search_memory` results for a query that previously matched it. **No new production code** — this task is integration tests only; the decay and archival behaviour was implemented in T018 (`apply_decay_and_archive`) and wired into the dreaming cycle in T020.

**Checkpoint** (US3 complete): Synaptic pruning works end-to-end. The forgetting subsystem (Constitution Principle IV) is validated.

---

## Phase 6: User Story 4 — Developer Swaps SQLite for a Custom Storage Backend (Priority: P2)

**Goal**: Prove the `StorageAdapter` ABC is sufficient to implement an alternative backend with zero changes to orchestration-layer code. This is the test for Constitution Principle III (Layered, Modular, Pluggable Architecture).

**Independent Test**: With a fully-implemented `DictStorageAdapter` injected as `storage=DictStorageAdapter()`, run the full US1 + US2 flow and assert identical behaviour to the SQLite path. Then run the 13 parameterised contract tests against BOTH `SQLiteAdapter` and `DictStorageAdapter` and assert all 26 pass.

- [ ] T025 [US4] Implement `DictStorageAdapter` (in-memory dict-based `StorageAdapter` subclass) in `packages/neuromem-core/tests/conftest.py` as test infrastructure (NOT in `src/neuromem/`, per Principle V's test-fixture separation). All 13 abstract methods implemented over `dict[str, Memory]`, `dict[str, Node]`, `list[Edge]` in-process state. Add `DictStorageAdapter` to the `storage_adapter` fixture's `params` list alongside `SQLiteAdapter`, so every contract test in `test_storage_adapter_contract.py` runs against both adapters automatically. Add one integration test at `packages/neuromem-core/tests/test_system.py::test_dict_adapter_full_loop` that runs the US1 `enqueue → force_dream → build_prompt_context` flow with `DictStorageAdapter` and asserts behaviour parity with the SQLite path. Depends on T018 (SQLiteAdapter fully implemented) so the two-adapter parameterisation lines up, and on T022 (ContextHelper) so the integration test has something to render.

**Checkpoint** (US4 complete): The adapter contract is validated as a real abstraction, not an aspirational one. Future backends (Postgres, Firebase, Qdrant) can slot in without touching `system.py`, `context.py`, `tools.py`, or `vectors.py`.

---

## Phase 7: User Story 5 — Developer Manually Forces a Dreaming Cycle (Priority: P3)

**Goal**: `force_dream()` works for test/CLI use cases — dreaming runs synchronously when asked, empty inbox is a no-op, and block=False actually returns non-blocking.

**Independent Test**: Enqueue 2 memories (below the default threshold of 10), assert they are in `status='inbox'`, call `force_dream(block=True)`, assert both are now `status='consolidated'`. Separately, with an empty inbox, call `force_dream(block=True)` and assert it returns cleanly without error.

**Note**: `force_dream` was implemented as part of T021 (bundled with the background-thread infrastructure). This phase is tests-only.

- [ ] T026 [US5] Add end-to-end acceptance tests for `force_dream` to `packages/neuromem-core/tests/test_system.py`. Scenarios: (a) 3 inbox memories + `dream_threshold=10` + `force_dream(block=True)` → all 3 become `consolidated`; (b) empty inbox + `force_dream(block=True)` → returns immediately, no error, no state change; (c) `force_dream(block=False)` → returns before dreaming finishes (measurable by `is_dreaming` still being `True` briefly after the call); (d) calling `force_dream` while a previous cycle is in progress does not spawn a second thread. **No new production code.**

**Checkpoint** (US5 complete): All 5 user stories validated. Full feature set from spec.md §User Scenarios landed.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Enforcement tests for success criteria (SC-005, SC-006, SC-007), documentation, and publish-readiness.

- [ ] T027 [P] Write `packages/neuromem-core/tests/test_no_forbidden_imports.py` — a file-walk test that recursively reads every `.py` file under `packages/neuromem-core/src/neuromem/` and asserts that none of the following appear as `import` statements: `openai`, `anthropic`, `google.genai`, `google_genai`, `cohere`, `voyageai`, `langchain`, `langgraph`, `llama_index`, `google.adk`, `anthropic_ai_agent`. Enforces SC-005 and Constitution Principle I at CI time.
- [ ] T028 [P] Write `packages/neuromem-core/tests/test_pyproject_dependencies.py` — parse `packages/neuromem-core/pyproject.toml` using `tomllib` (stdlib) and assert `[project].dependencies` is exactly `["numpy>=1.26", "pandas>=2.1"]` (set-equal comparison, ignore ordering). Enforces SC-006 and Constitution Principle II at CI time.
- [ ] T029 [P] Configure the `neuromem` logger namespace in `packages/neuromem-core/src/neuromem/__init__.py`: `logger = logging.getLogger("neuromem")`; add a `NullHandler` so importers don't see "no handlers could be found" warnings. Add a test in `packages/neuromem-core/tests/test_logging.py` verifying the namespace exists, defaults to `WARNING`, and can be reconfigured by callers.
- [ ] T030 Coverage gate — add `--cov=neuromem --cov-fail-under=90 --cov-report=term-missing` to the `[tool.pytest.ini_options] addopts` list in the root `pyproject.toml`. Run the full suite; add focused tests for any uncovered branches in `system.py`, `context.py`, `tools.py`, `vectors.py` to get ≥90% coverage on those four files (SC-007). This task may require adding test cases for: `ValueError` branches in `NeuroMemory.__init__`, the rollback path in `_run_dream_cycle`, the empty-graph path in `ContextHelper.build_prompt_context`, the missing-node branch in `_render_ascii_tree`.
- [ ] T031 [P] Rewrite `packages/neuromem-core/README.md` with the real content: install command (`uv add neuromem-core`), 30-line minimal usage example (lifted from `specs/001-neuromem-core/quickstart.md`), link to the full quickstart, link to the spec, note about workspace layout. Replaces the stub created in T003.
- [ ] T032 [P] Update repo-root `README.md` with monorepo navigation: what's in `packages/`, how to add a new sibling package (link to constitution §Additional Constraints — Repository Structure), how to run tests across the whole workspace, link to the spec-kit workflow and `.specify/memory/constitution.md`. (The root README is currently absent — this creates it.)
- [ ] T033 Manual quickstart validation — run `specs/001-neuromem-core/quickstart.md` top to bottom: copy the install commands, write the `MockLLMProvider`/`MockEmbeddingProvider` sketches, execute each code snippet in a Python REPL, confirm the ASCII tree renders correctly, confirm `retrieve_memories` returns LTP-spiked rows. Document any friction in a follow-up issue or spec amendment. This is a manual / interactive task — no code lands.
- [ ] T034 [P] Publish dry-run — run `uv build` inside `packages/neuromem-core/`, verify both wheel (`.whl`) and sdist (`.tar.gz`) are produced in `dist/`, inspect the wheel contents with `python -m zipfile -l` to confirm: (a) `neuromem/` package is included, (b) no `tests/` directory is shipped, (c) no `__pycache__/`, (d) `METADATA` file correctly lists `numpy>=1.26` and `pandas>=2.1` as runtime deps. Do NOT upload — this is a dry-run only.

**Final Checkpoint**: All 5 user stories functional. SC-005, SC-006, SC-007 enforced by tests. Package builds cleanly. Quickstart validated end-to-end. Ready for `/speckit.implement` → first `0.1.0` release.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies. Can start immediately.
- **Phase 2 (Foundational)**: Depends on Phase 1 completing. T008, T009, T010 can run in parallel. T011 depends on T009. T012 depends on T010.
- **Phase 3 (US1, MVP)**: Depends on Phase 2. T013–T018 are sequential on the same file (`sqlite.py`). T019 depends on T018 (needs a fully working adapter). T020 depends on T019. T021 depends on T020. T022 can run in parallel with T019–T021 once the contract layer is in place but before wrapping up T020 (it only needs the `StorageAdapter` methods `get_nearest_nodes` and `get_subgraph` — both land in T016–T017).
- **Phase 4 (US2)**: Depends on Phase 3 complete. T023 needs `ContextHelper` (T022) for `search_memory` and `storage.get_memory_by_id` + `storage.spike_access_weight` (T013, T018) for `retrieve_memories`.
- **Phase 5 (US3)**: Depends on Phase 3 complete. T024 is test-only and exercises the already-implemented decay path from T018/T020.
- **Phase 6 (US4)**: Depends on Phase 3 complete (needs SQLiteAdapter + ContextHelper for the integration test). T025 parameterises the contract tests created in T012.
- **Phase 7 (US5)**: Depends on Phase 3 complete. T026 is test-only.
- **Phase 8 (Polish)**: Depends on all user stories complete. T027, T028, T029, T031, T032, T034 are [P] and can run in parallel. T030 and T033 are sequential after the parallel group.

### User Story Dependencies

- **US1 (P1)**: Foundational phase only. This is the MVP and is the *sole prerequisite* for every other user story — all the cognitive-loop machinery lives here.
- **US2 (P1)**: US1 complete. Touches `tools.py` which imports from `context.py` (US1) and the storage adapter methods (US1).
- **US3 (P2)**: US1 complete. Test-only — no production code.
- **US4 (P2)**: US1 complete. DictStorageAdapter is test infrastructure added to `conftest.py`, parameterised against the existing contract tests from T012.
- **US5 (P3)**: US1 complete. Test-only — `force_dream` was bundled into T021.

### Within Each User Story (Ordering Rules)

1. Storage/persistence layer first (`sqlite.py` methods land before orchestration uses them).
2. Contract layer second (ABCs already in Phase 2).
3. Orchestration layer third (`system.py`, `context.py`, `tools.py`).
4. Integration/acceptance tests last.
5. Principle V red-green-refactor: every task in a user story phase is one atomic commit containing both failing tests and the implementation that makes them pass (co-commit pattern).

### Parallel Opportunities

- **Foundational phase**: T008, T009, T010 run in parallel (three independent modules, three independent test files).
- **US2**: T023 is a single task (`tools.py` + `tests/test_tools.py`), not parallelisable within itself.
- **Polish phase**: T027, T028, T029, T031, T032, T034 are all independent files and can land in any order. T030 and T033 run after the parallel group because they depend on the full suite being complete.
- **Cross-story parallelism**: Once US1 is complete, US2/US3/US4/US5 can be worked on in parallel by different developers (all four depend only on US1).

---

## Parallel Example: Foundational Phase

```bash
# All three can run simultaneously — different files, no shared state.
# Each task is one commit containing both tests and implementation.
Task T008: Test-first neuromem.vectors (tests/test_vectors.py + src/neuromem/vectors.py)
Task T009: Test-first neuromem.providers (tests/test_providers.py + src/neuromem/providers.py)
Task T010: Test-first neuromem.storage.base (tests/test_storage_base.py + src/neuromem/storage/base.py)

# Then sequentially (each depends on the one before):
Task T011: MockEmbeddingProvider + MockLLMProvider fixtures (depends on T009)
Task T012: StorageAdapter contract test skeleton (depends on T010)
```

## Parallel Example: Polish Phase

```bash
# Once all user stories are complete, these six run in parallel:
Task T027: tests/test_no_forbidden_imports.py
Task T028: tests/test_pyproject_dependencies.py
Task T029: logger configuration + tests/test_logging.py
Task T031: packages/neuromem-core/README.md (real content)
Task T032: repo-root README.md
Task T034: uv build dry-run

# Then sequentially:
Task T030: coverage gate + fill-in tests (needs T027-T029's test files in place)
Task T033: manual quickstart walk-through (last before release)
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. **Phase 1: Setup** (T001–T007, ~1 day). Workspace, package skeleton, pre-commit refresh, `uv sync` verified.
2. **Phase 2: Foundational** (T008–T012, ~1 day). Contract layer ABCs, vectors module, mock fixtures.
3. **Phase 3: User Story 1** (T013–T022, ~3–4 days). SQLiteAdapter, NeuroMemory, ContextHelper. 10 atomic commits.
4. **STOP and VALIDATE**: Run the US1 Independent Test end-to-end. Manually walk through `specs/001-neuromem-core/quickstart.md` §"The two-call loop" to confirm the passive context injection path works with SQLiteAdapter.
5. **Deploy/demo** the MVP if desired. At this point `neuromem-core 0.1.0-rc1` is an internally usable library.

### Incremental Delivery

1. MVP (US1) → tag `0.1.0-rc1`, optionally publish to TestPyPI.
2. Add US2 (T023, ~0.5 days) → agent-facing tool functions → tag `0.1.0-rc2`.
3. Add US3 (T024, ~0.5 days) → decay + archival validated end-to-end → tag `0.1.0-rc3`.
4. Add US4 (T025, ~1 day) → DictStorageAdapter proves the contract → tag `0.1.0-rc4`.
5. Add US5 (T026, ~0.25 days) → force_dream test coverage → tag `0.1.0-rc5`.
6. Polish (T027–T034, ~1–2 days) → SC enforcement, README, coverage, publish dry-run → tag `0.1.0` and release.

### Parallel Team Strategy (if multiple developers)

- **Days 1–2**: One developer completes Setup + Foundational. Others wait (tight serial dependency).
- **Days 3–6** (after Foundational): One developer owns US1 (T013–T022), another picks up US4 prep (T025 can be started in parallel once T010 lands — DictStorageAdapter depends only on the `StorageAdapter` ABC, not on `SQLiteAdapter`).
- **Days 6–7**: US2, US3, US5 are light touches and can be split across developers.
- **Day 8**: Polish phase — the six [P] tasks split freely; T030/T033 sequentially last.

---

## Notes

- Every task row = one atomic commit. Principle VI caps commits at 500 lines of diff (including tests). Every task in this list is sized to fit that cap; if a task grows beyond 500 lines during implementation, split it (rename to T013a / T013b, document the split in the commit message).
- Tests and implementation are **co-committed** per Principle V. The alternative (test-only commit → impl commit) would fail the pre-commit pytest hook in the test-only commit, and Principle VI forbids `--no-verify`.
- `[P]` = different files, no dependencies on incomplete work in the same phase. `[Story]` = traceability label tying the task to a user story in spec.md.
- Every user story is independently completable and testable against its own Independent Test criterion.
- Contract tests for `StorageAdapter` (T012 + T018 + T025) exercise the same 13 test functions against two adapter implementations (26 test runs total). This is the test-side enforcement of Principle III.
- Test files named `test_*.py` are picked up automatically by pytest from the `[tool.pytest.ini_options] testpaths = ["packages/neuromem-core/tests"]` config.
- Do NOT add real network-calling provider implementations (`OpenAIProvider`, `AnthropicProvider`, etc.) during this feature — those land in future features as **sibling packages** under `packages/` per Constitution §Additional Constraints — Repository Structure.
- Do NOT add additional runtime dependencies beyond numpy + pandas without a constitutional amendment (Principle II).
