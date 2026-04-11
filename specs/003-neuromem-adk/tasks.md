---

description: "Task list for the neuromem-adk implementation (003-neuromem-adk)"
---

# Tasks: neuromem-adk — Google ADK Integration

**Input**: Design documents from `/specs/003-neuromem-adk/`
**Prerequisites**: spec.md, plan.md, research.md, contracts/public-api.md, quickstart.md (all landed)
**Constitution**: v2.0.0

**Tests**: Test tasks are included throughout. Per Constitution v2.0.0 Principle V (NON-NEGOTIABLE), every implementation MUST follow the red-green-refactor cycle. Because Principle VI mandates pre-commit hooks that run `pytest` on every commit, **failing tests cannot land in their own commit** — a failing-test commit would be blocked by the hook. Therefore test tasks and their corresponding implementation tasks are **co-committed as a single atomic unit** (Principle V explicitly permits "co-committed" as an alternative to "preceding"). Each task row below represents one commit unless explicitly noted otherwise.

**Organization**: Tasks are grouped by user story so each story can be implemented, tested, and (hypothetically) delivered independently. The dependency order within each phase is: tests first in the row → implementation that makes them pass → stage + commit.

## Format: `[ID] [P?] [Story?] Description`

- **[P]**: Can run in parallel with other [P] tasks in the same phase (different files, no shared state, no dependency on incomplete tasks).
- **[Story]**: `[US1]`..`[US6]` — only on user-story phase tasks. Setup, Foundational, Integration-test/CI, and Polish phases have no story label.
- File paths are absolute from the repo root unless obvious from context.

## Path Conventions

New sibling package under `packages/neuromem-adk/`. The package imports as `neuromem_adk` (underscores, not hyphens — Python module name convention).

- **Package source**: `packages/neuromem-adk/src/neuromem_adk/`
- **Package tests**: `packages/neuromem-adk/tests/`
- **Package config**: `packages/neuromem-adk/pyproject.toml`
- **Workspace config**: `pyproject.toml` (repo root — modified by T005)
- **Spec artefacts**: `specs/003-neuromem-adk/` (already landed)

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Bootstrap the `neuromem-adk` sibling package scaffold. Nothing inside `packages/neuromem-core/` or `packages/neuromem-gemini/` may change.

- [ ] T001 Create `packages/neuromem-adk/pyproject.toml` with `[project]` section (`name = "neuromem-adk"`, `version = "0.1.0"`, `requires-python = ">=3.10"`, `dependencies = ["neuromem-core", "neuromem-gemini", "google-adk>=1.29", "numpy>=1.26"]`), `[dependency-groups] dev = ["pytest>=8.0"]`, `[build-system]` hatchling config, `[tool.hatch.build.targets.wheel] packages = ["src/neuromem_adk"]`, `[tool.hatch.build.targets.sdist]` include list, and `[tool.uv.sources]` declaring both `neuromem-core` and `neuromem-gemini` as `{ workspace = true }`. Copy the exact style of `packages/neuromem-gemini/pyproject.toml` so the template is consistent.
- [ ] T002 [P] Create `packages/neuromem-adk/README.md` placeholder with package name, one-sentence description, link to `specs/003-neuromem-adk/spec.md`, and "under construction" notice. The real content lands in T017.
- [ ] T003 [P] Create `packages/neuromem-adk/src/neuromem_adk/__init__.py` with `__version__ = "0.1.0"`. Leave the `enable_memory` and `NeuromemMemoryService` re-exports commented out; they get uncommented as T008 and T013 land.
- [ ] T004 Modify the workspace root `pyproject.toml` `[tool.pytest.ini_options].testpaths` list to add `packages/neuromem-adk/tests`. Verify `uv run pytest --collect-only` still collects 208+ existing tests plus 0 new ones (no tests exist yet but the path is discoverable).
- [ ] T005 Run `uv sync --dev` at the repo root. Verify the workspace picks up `neuromem-adk` as a new member, installs it in editable mode, pulls `google-adk>=1.29` and its transitive deps, and `python -c "import neuromem_adk; print(neuromem_adk.__version__)"` prints `0.1.0`.

**Checkpoint**: Empty package scaffold in place, workspace recognises it, installable in editable mode. Ready for Foundational phase.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Test fixtures every user story depends on — the `gemini_api_key` session-scoped fixture and the `mock_adk_agent` fixture for unit tests that don't touch the network.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [ ] T006 Create `packages/neuromem-adk/tests/conftest.py` with a session-scoped `gemini_api_key` fixture that reads `GOOGLE_API_KEY` from `os.environ` and falls back to stdlib-parsing the repo-root `.env` file. Raises `pytest.skip(...)` with an actionable message if no key is found. Reuse the 10-line `_parse_dotenv` helper pattern from `packages/neuromem-gemini/tests/conftest.py`. No `python-dotenv` dependency.
- [ ] T007 Extend `packages/neuromem-adk/tests/conftest.py` with a `mock_adk_agent` fixture that builds a minimal ADK `Agent` instance backed by a no-op model stub so unit tests can inspect `agent.tools`, `agent.before_model_callback`, and `agent.after_agent_callback` without triggering any network call. Uses `google.adk.agents.Agent` directly — not a home-rolled mock — because we want to test against the real ADK class surface.

**Checkpoint**: Test infrastructure ready. User story phases can now begin.

---

## Phase 3: User Story 1 — One-Line Memory Attachment (Priority: P1) 🎯 MVP

**Goal**: `enable_memory(agent, db_path="memory.db")` wires a minimal attached-memory happy path: instantiates default providers, builds a `NeuroMemory`, returns the handle. This single task delivers the MVP promise — one line, working memory.

**Independent Test**: In a unit test, build a `mock_adk_agent`, call `enable_memory(agent, ":memory:")` with mocked LLM and embedder providers, assert the returned object is a `NeuroMemory` instance and that `agent` has been marked as neuromem-enabled.

- [ ] T008 [US1] Write `packages/neuromem-adk/tests/test_enable.py` with the US1 happy-path unit test: given a `mock_adk_agent`, a `MockLLMProvider`, and a `MockEmbeddingProvider` (reuse the mocks from `packages/neuromem-core/tests/conftest.py` by importing them), when `enable_memory(agent, ":memory:", llm=mock_llm, embedder=mock_embed)` is called, then (a) the return value is an instance of `NeuroMemory`, (b) the returned object's `storage` is a `SQLiteAdapter`, (c) the agent has an internal marker attribute set. Then implement `packages/neuromem-adk/src/neuromem_adk/enable.py` with the minimal `enable_memory(agent, db_path, *, llm=None, embedder=None) -> NeuroMemory` function that creates the `NeuroMemory`, sets the marker, and returns the handle. Uncomment the `enable_memory` re-export in `__init__.py`.
- [ ] T009 [US1] Add the double-attachment error test to `test_enable.py`: calling `enable_memory` twice on the same agent must raise `ValueError` with a clear message. Add the marker check at the top of `enable_memory` in `enable.py`. Also add the missing-credentials test: if `llm=None` and `GOOGLE_API_KEY` is not in the environment, `enable_memory` raises `KeyError` with an actionable message naming the env var. (Mock the env by `monkeypatch.delenv("GOOGLE_API_KEY")` — do NOT touch `os.environ` directly.)

**Checkpoint US1 complete**: `enable_memory` returns a working `NeuroMemory` handle, errors on double-attachment, errors on missing credentials. MVP achieved.

---

## Phase 4: User Story 2 — Automatic Turn Capture (Priority: P1) 🎯 MVP

**Goal**: After every completed agent turn, the user message + assistant response pair flows into `memory.enqueue` via an `after_agent_callback`. Developers never have to call a save method.

**Independent Test**: In a unit test, call the `after_agent_turn_capturer` callback directly with a synthetic ADK `CallbackContext` containing a user message and an assistant response. Assert the memory's inbox status count increased by one (or two, depending on how we serialise the turn) after the call.

- [ ] T010 [US2] Write `packages/neuromem-adk/tests/test_callbacks.py` with the US2 turn-capture test: construct a real `NeuroMemory` backed by a `DictStorageAdapter` (reuse from neuromem-core conftest) + `MockLLMProvider` + `MockEmbeddingProvider`, build a synthetic `CallbackContext` with a user message and assistant response, call the `after_agent_turn_capturer` callback closure with the context, assert `memory.storage.count_memories_by_status("inbox") == expected`. Cover: happy path, empty agent response (skip capture), non-UTF-8 content (handled gracefully), exception during `enqueue` (propagates cleanly). Then implement `packages/neuromem-adk/src/neuromem_adk/callbacks.py::build_after_agent_turn_capturer(memory)` returning a closure that extracts user/assistant text from the context and calls `memory.enqueue`. Wire it from `enable_memory` via callback chaining (preserve any existing `after_agent_callback` on the agent).

**Checkpoint US2 complete**: Every completed agent turn flows into memory automatically.

---

## Phase 5: User Story 3 — Passive Context Injection (Priority: P1) 🎯 MVP

**Goal**: Before every model call, `ContextHelper.build_prompt_context(user_message)` is invoked and the resulting ASCII tree is prepended to the agent's instruction. The LLM sees relevant memory "for free" without asking.

**Independent Test**: In a unit test, construct a `NeuroMemory` with one pre-consolidated memory, build a synthetic `LlmRequest` containing a user message, call the `before_model_context_injector` callback, assert the resulting request's instruction has been mutated to include the memory's summary.

- [ ] T011 [US3] Write `packages/neuromem-adk/tests/test_callbacks.py` (extend the existing file) with the US3 context-injection test: pre-consolidate one memory in a `DictStorageAdapter`-backed `NeuroMemory`, build a synthetic `LlmRequest` (or the ADK equivalent), call `build_before_model_context_injector(memory)(context, request)`, assert the returned request has the memory's summary in its system instruction. Cover: happy path, empty-graph path (no injection, request unchanged), bounded injection size (injected tree truncated or limited at a reasonable upper bound so the prompt can't balloon). Then implement `packages/neuromem-adk/src/neuromem_adk/callbacks.py::build_before_model_context_injector(memory)` returning a closure that calls `ContextHelper(memory).build_prompt_context(user_message)` and prepends the result to the request's instruction. Wire it from `enable_memory` via callback chaining.

**Checkpoint US3 complete**: All three P1 user stories done. `enable_memory` now wires the full passive memory loop.

---

## Phase 6: User Story 4 — LLM-Driven Active Memory Search (Priority: P2)

**Goal**: The agent can explicitly invoke `search_memory` and `retrieve_memories` as ADK function tools. The LLM decides when to call them.

**Independent Test**: After `enable_memory(agent, ...)`, inspect `agent.tools`. Assert the list contains two callables bound to the memory handle, with correct names and correct signatures (the internal `system` arg is pre-bound via `functools.partial` and invisible to ADK's schema generator).

- [ ] T012 [US4] Write `packages/neuromem-adk/tests/test_enable.py` (extend) with the US4 tool-registration test: after `enable_memory(agent, ...)`, inspect `agent.tools`. Assert the list contains a pre-bound `search_memory` and a pre-bound `retrieve_memories`, that `inspect.signature` on each shows only user-facing args (`{query, top_k, depth}` for search, `{memory_ids}` for retrieve — NOT `system`), and that calling each tool directly (in-test, without going through the LLM) produces the expected output against the pre-seeded `NeuroMemory`. Then implement the tool-registration logic inside `enable.py::enable_memory`: use `functools.partial(search_memory, system=memory)` and `functools.partial(retrieve_memories, system=memory)`, append both to `agent.tools` (preserving any existing tools). Fallback: if ADK's schema generator rejects partials, swap to local wrapper functions defined inside `enable_memory` — this should be caught by the test assertion on `inspect.signature` showing the right args.

**Checkpoint US4 complete**: LLM can explicitly invoke memory search. US5 (native BaseMemoryService) can now be added in parallel with any polish work.

---

## Phase 7: User Story 5 — Native BaseMemoryService Integration (Priority: P2)

**Goal**: `NeuromemMemoryService(BaseMemoryService)` implements ADK's native memory-service interface so framework-level features that depend on it work automatically. `enable_memory` instantiates one internally and attaches it to the runner.

**Independent Test**: Instantiate `NeuromemMemoryService(memory)` directly (without `enable_memory`), call its `add_session_to_memory(synthetic_session)` with a synthetic session containing a few events, assert the events landed in `memory.storage` as consolidated memories. Call its `search_memory(query)` and assert it returns a result in the shape ADK's `BaseMemoryService` requires.

- [ ] T013 [US5] Write `packages/neuromem-adk/tests/test_memory_service.py` covering the `NeuromemMemoryService` contract: construct a synthetic ADK `Session` with 3 events (user turn + assistant response + user turn), build a `NeuroMemory` over a `DictStorageAdapter`, wrap it in a `NeuromemMemoryService(memory)`, call `add_session_to_memory(session)`, assert all 3 turns landed in the memory store and a dream cycle was forced (all memories consolidated). Call `search_memory("query")` and assert the return shape matches ADK's expected result type. Then implement `packages/neuromem-adk/src/neuromem_adk/memory_service.py::NeuromemMemoryService(BaseMemoryService)` with the two abstract method implementations. Uncomment the `NeuromemMemoryService` re-export in `__init__.py`. Also extend `enable.py::enable_memory` to instantiate a `NeuromemMemoryService` internally and wire it via ADK's runner configuration hook (if the runner is accessible via the agent handle; otherwise this task just provides the class for advanced users to wire manually and the enable_memory path leaves it off).

**Checkpoint US5 complete**: Native memory slot filled. Advanced users have a clean extension point.

---

## Phase 8: User Story 6 — Manual Provider Override (Priority: P3)

**Goal**: Passing explicit `llm=` or `embedder=` to `enable_memory` skips the default Gemini auto-instantiation path and uses the provided instances instead.

**Independent Test**: Call `enable_memory(agent, ":memory:", llm=MockLLM(), embedder=MockEmbed())` with the Gemini environment variable explicitly deleted. Assert no Gemini provider is constructed and the attached memory uses the mock providers for all operations.

- [ ] T014 [US6] Write `packages/neuromem-adk/tests/test_enable.py` (extend) with the US6 override tests: (a) passing both `llm` and `embedder` skips default instantiation entirely; (b) passing only `llm` falls back to the Gemini default for the embedder; (c) passing only `embedder` falls back to the Gemini default for the LLM. Use `monkeypatch` to simulate missing `GOOGLE_API_KEY` and verify the default path would error if it were hit. Then implement the keyword-argument handling inside `enable.py::enable_memory` (the initial T008 implementation already accepts `llm=None` and `embedder=None`; this task adds the per-slot fallback logic and the credentials-check short-circuit).

**Checkpoint US6 complete**: All six user stories done. Package has complete unit test coverage. Ready for integration test.

---

## Phase 9: Integration Test + CI

**Purpose**: The one end-to-end smoke test that proves the whole wiring works against real ADK + real Gemini, plus the CI workflow to run it on a schedule.

- [ ] T015 Write `packages/neuromem-adk/tests/test_adk_integration.py` with the `@pytest.mark.integration`-gated end-to-end test. Module-level `pytestmark` suppresses `ResourceWarning` and `PytestUnraisableExceptionWarning` (same pattern as `neuromem-gemini/tests/test_cognitive_loop_real_llm.py`). The test constructs a real `google.adk.agents.Agent`, calls `enable_memory(agent, ":memory:")`, runs a 3-turn scripted conversation (e.g., "my dog's name is Rex" / "I live in Brighton" / "what did I tell you about my pet?"), asserts (a) the memory store contains the enqueued turns consolidated, (b) the final response references the dog's name (case-insensitive substring check), and (c) the memory's access weight for the relevant memory was spiked to 1.0. Uses the `gemini_api_key` fixture; skips gracefully if no key. Cost budget: <$0.01 per run, <30 seconds wall clock (SC-004).
- [ ] T016 Modify `.github/workflows/integration.yml` to add a new `adk-integration` job alongside the existing `gemini` job. Copy the gemini job's step set, change the pytest invocation to target `packages/neuromem-adk/tests/`, and add `packages/neuromem-adk/**` to the workflow's `paths` trigger filter. The existing `GEMINI_API_KEY` secret serves both jobs (for ADK it's read as `GOOGLE_API_KEY` at runtime — add a small export step in the ADK job that renames the env var).

**Checkpoint**: Integration pipeline green. PR-ready.

---

## Phase 10: Polish & Cross-Cutting Concerns

**Purpose**: README content, manual walkthrough, build dry-run, final cleanup.

- [ ] T017 [P] Rewrite `packages/neuromem-adk/README.md` with real content: install command, 30-line usage example (lift from `quickstart.md`), default model names, integration-test invocation guide with `--no-cov` flag, pointer to `neuromem-core` and `neuromem-gemini` for upstream reference. Replace the T002 placeholder.
- [ ] T018 [P] Add `packages/neuromem-adk/tests/test_no_forbidden_imports.py` — a small file-walk test that asserts no `openai`, `anthropic`, `langchain`, `langgraph`, `llama_index`, etc. appear in any `src/neuromem_adk/` file. Mirrors the `test_no_forbidden_imports.py` pattern from `neuromem-core` but scoped to this package's source tree. Allows `google.adk` and `google.genai` (via neuromem-gemini) since they're the whole point.
- [ ] T019 [P] Manual quickstart validation — run `packages/neuromem-adk/README.md`'s install + usage example top-to-bottom against a fresh venv + real Gemini. Confirm every code block executes as shown. Document any friction in a spec amendment. Manual/interactive task; no code lands unless friction was found.
- [ ] T020 [P] Publish dry-run — run `uv build` inside `packages/neuromem-adk/`. Verify both wheel (`.whl`) and sdist (`.tar.gz`) are produced in `dist/`, inspect the wheel contents with `python -m zipfile -l` to confirm: (a) `neuromem_adk/` package is included, (b) no `tests/` directory is shipped, (c) no `__pycache__/`, (d) `METADATA` correctly lists `neuromem-core`, `neuromem-gemini`, `google-adk>=1.29`, and `numpy>=1.26` as runtime deps. Do NOT upload.
- [ ] T021 Final verification: run the full workspace test suite (`uv run pytest` from repo root). Expect 208 core tests + 3 gemini unit tests + ~15 new `neuromem-adk` unit tests + integration tests deselected. Coverage gate still green at 96.23% on the four core files (sibling-package code is outside the scoped include filter).

**Checkpoint**: v0.1.0 ready for PR and review.

---

## Dependencies

- **Phase 1 (Setup)**: T001 blocks T002-T005 (can't create src without the pyproject declaring the wheel package). T002 and T003 are `[P]` — different files. T004 and T005 are sequential after T001-T003.
- **Phase 2 (Foundational)**: T006 blocks T007 (same file, incremental additions). Both block all US phases.
- **Phase 3 (US1)**: T008 blocks T009 (both write to enable.py; T009 adds to what T008 landed).
- **Phase 4 (US2)** depends on Phase 3 because `enable_memory` must exist to wire the callback. T010 extends enable.py.
- **Phase 5 (US3)** depends on Phase 3. T011 extends callbacks.py AND enable.py.
- **Phase 6 (US4)** depends on Phase 3. T012 extends enable.py.
- **Phase 7 (US5)** depends on Phase 3. T013 creates memory_service.py (independent file) and extends enable.py.
- **Phase 8 (US6)** depends on Phase 3. T014 extends enable.py.
- **Phase 9 (Integration + CI)**: T015 depends on Phases 3-8 (needs the full `enable_memory` surface). T016 is independent of T015 (config-only file change).
- **Phase 10 (Polish)**: T017-T020 are `[P]` — independent files. T021 runs last (full-suite verification).

## Parallelisation opportunities

Within each phase, `[P]`-marked tasks can run in parallel because they write to different files:

- **Phase 1**: T002 and T003 can run in parallel.
- **Phase 10**: T017, T018, T019, T020 can all run in parallel (different files, manual T019 can happen alongside the automated ones).

Within user story phases, tasks are sequential because they all incrementally extend `enable.py` and the test files.

## Total task count

21 tasks across 10 phases:
- Phase 1 (Setup): 5 tasks
- Phase 2 (Foundational): 2 tasks
- Phase 3 (US1): 2 tasks
- Phase 4 (US2): 1 task
- Phase 5 (US3): 1 task
- Phase 6 (US4): 1 task
- Phase 7 (US5): 1 task
- Phase 8 (US6): 1 task
- Phase 9 (Integration + CI): 2 tasks
- Phase 10 (Polish): 5 tasks

## Release ordering

MVP at end of Phase 5 (all three P1 user stories complete). Could ship `v0.1.0-rc1` here if desired.
Incremental adds in Phases 6-8 (P2 and P3 stories) can land as `v0.1.0-rc2` .. `v0.1.0-rc4`.
Integration test + CI + polish land as `v0.1.0` final.

In practice we land everything on a single PR (003-neuromem-adk → main) and cut v0.1.0 at merge.
