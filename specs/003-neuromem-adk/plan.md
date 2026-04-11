# Implementation Plan: neuromem-adk — Google ADK Integration

**Branch**: `003-neuromem-adk` | **Date**: 2026-04-12 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/003-neuromem-adk/spec.md`
**Constitution**: [v2.0.0](../../.specify/memory/constitution.md)

## Summary

Build `neuromem-adk`, a new sibling package under `packages/` that integrates the `neuromem-core` cognitive loop into Google's Agent Development Kit (ADK) so agent developers can add persistent long-term memory to an existing `google.adk.agents.Agent` instance with a single function call.

The integration uses a **hybrid extension strategy**: four distinct ADK extension points, each covering a different memory behaviour that no single existing approach covers:

1. `before_model_callback` for **passive context injection** (relevant memory tree prepended to the system prompt before each model call).
2. `after_agent_callback` for **automatic turn capture** (completed user/assistant pairs flow into `NeuroMemory.enqueue`).
3. Function-tool registration (via `functools.partial`) for **LLM-driven active search** (the agent can call `search_memory` / `retrieve_memories` when it decides it needs to).
4. `NeuromemMemoryService(BaseMemoryService)` subclassing for **native framework memory-slot integration** (session-end consolidation via `add_session_to_memory`, and delegating `search_memory` to neuromem so ADK's native code paths never diverge from the callback path's state).

All four extension points wire up from a single public function:

```python
from google.adk.agents import Agent
from neuromem_adk import enable_memory

agent = Agent(model="gemini-2.0-flash-001", name="assistant", instruction="...")
memory = enable_memory(agent, db_path="memory.db")
```

`enable_memory` auto-instantiates `GeminiLLMProvider` and `GeminiEmbeddingProvider` from the `neuromem-gemini` sibling package (reading `GOOGLE_API_KEY` from the environment), creates a `NeuroMemory`, attaches the two callbacks, registers the two function tools, and wires the `NeuromemMemoryService`. Returns the `NeuroMemory` instance for inspection and manual control.

**Prior art**: mem0's ADK integration (<https://docs.mem0.ai/integrations/google-ai-adk>) uses tool-registration only. This package adds passive injection (US3), automatic capture (US2), and native memory-slot integration (US5) on top of mem0's tool-registration pattern (US4). Design research is documented in [research.md](./research.md).

## Technical Context

**Language/Version**: Python 3.10+ (same as rest of the monorepo; uses PEP 604 `X | None` unions).

**Primary Dependencies**:
- `neuromem-core` — contract layer and orchestrator (workspace source)
- `neuromem-gemini` — default provider pair for LLM and embedder (workspace source)
- `google-adk >= 1.29` — the framework this package wraps
- `numpy >= 1.26` — transitive via `neuromem-core`, also imported directly for any shape adjustments
- Dev: `pytest >= 8.0`

No other runtime deps. Explicitly NOT added: `python-dotenv` (stdlib parse works fine, matches the `neuromem-gemini` pattern), `anthropic`, `openai`, `langchain`, any vendor SDK other than `google-adk` / transitively `google-genai`.

**Storage**: No new storage. Uses whatever `StorageAdapter` the caller's `NeuroMemory` is configured with — defaults to `SQLiteAdapter` at the path the caller provides.

**Testing**: `pytest` following the `neuromem-gemini` template:
- **Unit tests** (run on every PR, no network, mocked ADK runner): cover `enable_memory`, each callback in isolation, `NeuromemMemoryService` contract, double-attachment error, missing-credentials error, tool binding.
- **Integration test** (`@pytest.mark.integration`, opt-in, gated on `GEMINI_API_KEY`): one end-to-end smoke test that builds a real ADK `Agent`, attaches memory via `enable_memory`, runs a 3–4 turn conversation against real Gemini, asserts memory was captured and context injection retrieved a relevant memory on a later turn.

**Target Platform**: Same as `neuromem-core` — any Python 3.10+ environment (macOS, Linux, Windows CI matrix).

**Project Type**: New sibling package under `packages/` inside the `uv` workspace monorepo. First framework-wrapper package; sets the template for future `neuromem-anthropic`, `neuromem-langchain`, etc.

**Performance Goals**:
- `enable_memory` call itself completes in under 500 ms (excluding any first-call provider auth round-trip).
- Callback overhead per turn: under 10 ms excluding the embedded `ContextHelper.build_prompt_context` call (which is already bounded by SC-003 in the core spec at under 500 ms for ≤10K nodes).
- Integration test full run: under 30 seconds wall clock, under $0.01 in API cost (SC-004).
- Unit test suite: under 10 seconds (SC-005).

**Constraints**:
- `packages/neuromem-core/` MUST NOT be modified in any way. T027 (forbidden imports) and T028 (locked deps) tripwires stay green by construction.
- Coverage gate is scoped to the four core-library files (system.py, context.py, tools.py, vectors.py); sibling-package code is not counted. Unchanged by this work.
- Every commit in this branch is ≤500 lines of diff per Constitution Principle VI.
- No `git commit --no-verify` ever. Pre-commit hooks (ruff format, ruff check, pytest) must pass every commit.

**Scale/Scope**: Intended for single-user agents running on a single process. Multi-user / multi-tenant deployments are out of scope for v0.1 (documented in spec assumptions).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Constitution version at time of check: **v2.0.0** (`.specify/memory/constitution.md`).

| # | Principle | Status | Evidence |
|---|---|---|---|
| I | Library-First, Framework-Agnostic Core | ✅ PASS | `neuromem-core` is untouched. This package lives in its own sibling directory and depends on core + gemini + google-adk. The core's T027 forbidden-imports tripwire only scans `packages/neuromem-core/src/neuromem/`, so `google.adk` imports in the new sibling are fine. The spec explicitly forbids touching core files (FR-009, FR-010). |
| II | Lean Dependency Set | ✅ PASS | `neuromem-core`'s locked `[project].dependencies` is untouched. T028 tripwire remains green. The new sibling package has its own dependency list (core + gemini + google-adk), which is allowed — the Principle II lock is package-scoped to neuromem-core only. |
| III | Layered, Modular, Pluggable Architecture | ✅ PASS | This package IS the pluggable layer. It implements against neuromem-core's public ABCs (`LLMProvider`, `EmbeddingProvider`, `StorageAdapter`) and ADK's `BaseMemoryService` ABC. No framework internals are touched; both sides are extension points. |
| IV | Neuroscience-Grounded Subsystem Design | ✅ PASS | This package doesn't add new cognitive-loop subsystems — it wires the existing ones into ADK. The mapping is: ADK turn → `enqueue()` (Hippocampus); ADK session-end → `force_dream()` (Neocortex); ADK model call → `build_prompt_context()` (Prefrontal Cortex); ADK tool call → `retrieve_memories()` with LTP spike (Synaptic Plasticity). |
| V | Test-First with Injected Mocks (NON-NEGOTIABLE) | ✅ PASS | Unit tests against mocked ADK runners cover every wiring component. The integration test against real ADK + real Gemini is opt-in only. Follows the `neuromem-gemini` pattern exactly. |
| VI | Atomic Commits with Enforced Pre-Commit Gates | ✅ PASS | Existing `.pre-commit-config.yaml` already enforces the gates. Every task commit in this branch will be ≤500 lines and must pass ruff format + ruff check + pytest. No `--no-verify` shortcuts. |

**Result**: All 6 gates pass. No violations. Complexity Tracking section below is empty.

### Post-Phase 1 re-check

After the Phase 1 artefacts land (`research.md`, `contracts/public-api.md`, `quickstart.md`), every principle will be re-checked. Expected result: still passing, because this package is purely additive and cannot by construction break any of the core tripwires.

## Project Structure

### Documentation (this feature)

```text
specs/003-neuromem-adk/
├── spec.md                  # Already written
├── plan.md                  # This file
├── research.md              # Resolved ADK + mem0 design decisions
├── contracts/
│   └── public-api.md        # neuromem-adk public API contract
├── quickstart.md            # User-facing usage guide (30-line example)
├── tasks.md                 # Generated by /speckit.tasks — NOT created by this command
└── checklists/
    └── requirements.md      # Already written by /speckit.specify
```

### Source code (repository root)

```text
packages/
├── neuromem-core/           # Unchanged
├── neuromem-gemini/         # Unchanged
└── neuromem-adk/            # NEW
    ├── pyproject.toml       # google-adk>=1.29, neuromem-core, neuromem-gemini runtime deps
    ├── README.md            # Install, usage, design notes, integration-test invocation
    ├── src/
    │   └── neuromem_adk/
    │       ├── __init__.py  # Public API: enable_memory, NeuromemMemoryService, __version__
    │       ├── enable.py    # enable_memory() function — the one-line wire-up
    │       ├── memory_service.py   # NeuromemMemoryService(BaseMemoryService)
    │       └── callbacks.py        # before_model_context_injector, after_agent_turn_capturer
    └── tests/
        ├── conftest.py              # gemini_api_key + mock ADK runner fixtures
        ├── test_enable.py           # unit tests for enable_memory wiring
        ├── test_callbacks.py        # callbacks in isolation
        ├── test_memory_service.py   # NeuromemMemoryService contract
        └── test_adk_integration.py  # @pytest.mark.integration — real ADK + real Gemini
```

**Modified files outside the new package**:
- `pyproject.toml` (workspace root) — add `packages/neuromem-adk/tests` to `testpaths`
- `.github/workflows/integration.yml` — add an `adk-integration` job alongside the existing `gemini` job, same secret, same cadence

**Not modified** (critical — tripwires stay green):
- Anything inside `packages/neuromem-core/`
- Anything inside `packages/neuromem-gemini/`

**Structure Decision**: New sibling package under `packages/neuromem-adk/` following the exact template of `packages/neuromem-gemini/` (pyproject.toml → src/neuromem_adk/ → tests/). This is the first framework-wrapper package and sets the scaffolding pattern for future `neuromem-anthropic`, `neuromem-langchain`, etc.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

No violations. This section is empty by design.
