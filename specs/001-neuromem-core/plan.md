# Implementation Plan: neuromem Core Library

**Branch**: `001-neuromem-core` | **Date**: 2026-04-11 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-neuromem-core/spec.md`
**Constitution**: [v2.0.0](../../.specify/memory/constitution.md)

## Summary

Build `neuromem-core`, a neuroscience-inspired long-term memory library for AI agents, as the first package in a `uv` workspace monorepo. The library exposes:

1. **`NeuroMemory`** — the orchestration engine that accepts a `StorageAdapter`, `LLMProvider`, and `EmbeddingProvider` via constructor injection and provides `enqueue()` and `force_dream()`.
2. **`ContextHelper`** — a prompt-injection helper whose `build_prompt_context(task)` returns a rendered ASCII sub-graph of the most relevant memories for the caller's current task.
3. **Agent tools** — `search_memory(query)` and `retrieve_memories(ids)` exposed as standalone functions for framework wrapper packages (`neuromem-adk`, `neuromem-anthropic`, `neuromem-langchain`) to bind as tool functions in later features.

The cognitive loop is: ingest text via `enqueue()` (Hippocampus), consolidate asynchronously via a background "dreaming" thread when the inbox threshold is reached (Neocortex — agglomerative clustering with LLM-named centroid nodes), apply exponential decay at the end of every dreaming cycle (Synaptic Pruning), and rebuild the relevant contextual sub-graph on demand at query time (Prefrontal Cortex). Retrieval spikes a memory's `access_weight` back to 1.0 (Long-Term Potentiation), keeping frequently-used memories alive.

Technical approach is **numpy + pandas + stdlib**, per Constitution v2.0.0 Principle II. Embeddings come from an injected provider (external API call) but all similarity math is vectorised with numpy. Graph storage defaults to stdlib `sqlite3` via `SQLiteAdapter`, with embeddings stored as `float32` bytes. The adapter pattern (ABC + injected instance) means `SQLiteAdapter` ships as the v1 default while future adapters (Postgres/pgvector, Qdrant, Firebase) drop in without touching any orchestration-layer file.

The **repository is a `uv` workspace monorepo** (Constitution v2.0.0 Additional Constraints — Repository Structure). `neuromem-core` lives at `packages/neuromem-core/` with its own `pyproject.toml`. The root `pyproject.toml` declares `[tool.uv.workspace] members = ["packages/*"]`. Future sibling packages (`packages/neuromem-adk/`, etc.) are explicitly out of scope for this feature — they land in later features and will depend on `neuromem-core` via workspace dependency syntax.

## Technical Context

**Language/Version**: Python 3.10+ (uses PEP 604 `X | None` unions, `typing.Literal` for status discriminators, `zip(..., strict=True)`).
**Primary Dependencies**: Python 3.10+ standard library; **numpy** (>= 1.26) and **pandas** (>= 2.1) as first-class runtime dependencies per Constitution v2.0.0 Principle II. No other runtime dependencies in `neuromem-core`. Dev dependencies (via `[dependency-groups] dev`): `pytest`, `pytest-cov`, `ruff`, `pre-commit`.
**Storage**: SQLite via stdlib `sqlite3` as the default `StorageAdapter`. Three tables (`memories`, `nodes`, `edges`) per the canonical schema in spec.md §Data Model. Embeddings stored as `float32` bytes in a BLOB column via `np.ndarray.astype(np.float32).tobytes()` / `np.frombuffer(blob, dtype=np.float32)`. No SQLite vector extension required.
**Testing**: `pytest` + `pytest-cov`. Every core module tested with injected `MockStorageAdapter`, `MockLLMProvider`, `MockEmbeddingProvider` fixtures living in `packages/neuromem-core/tests/conftest.py`. Zero network access in the test suite. ≥90% line coverage required on `system.py`, `context.py`, `tools.py`, `vectors.py` per SC-007.
**Target Platform**: Library for any Python 3.10+ environment (macOS, Linux, Windows). numpy and pandas both have wheels for every supported platform, so no install friction. No OS-specific code. No FUSE, no filesystem mounting.
**Project Type**: **`uv` workspace monorepo** with a single v1 package (`packages/neuromem-core/`). Future sibling packages (`packages/neuromem-adk/`, `packages/neuromem-anthropic/`, `packages/neuromem-langchain/`, etc.) are planned but explicitly out of scope for this feature.
**Performance Goals**:
- `enqueue()` returns in <50 ms on the caller thread *excluding* the injected `LLMProvider.generate_summary()` call (SC-002).
- `search_memory()` / `ContextHelper.build_prompt_context()` return in <500 ms for a graph of ≤10,000 nodes using the default `SQLiteAdapter` with numpy-vectorised cosine similarity (SC-003). With numpy the full-table scan for 10K × 1536-dim embeddings is ~20 ms on a modest laptop, so the budget is comfortable.
- Dreaming cycle throughput is secondary (runs off-thread) but must not starve `enqueue()` writes (the double-buffer status flip guarantees this).

**Constraints**:
- Core MUST NOT import vendor LLM/embedding SDKs or agent-framework SDKs (Principle I, SC-005). numpy and pandas are permitted; they are NOT LLM/framework SDKs.
- No `asyncio` in the core. Framework wrapper packages call `enqueue()` via `asyncio.to_thread()` if they need async.
- Single-process SQLite only. Multi-process concurrent writes to the same `.db` file are out of scope for v1.
- `raw_content` is always a UTF-8 string. Binary media (images, audio, etc.) is out of scope for v1.
- `git commit --no-verify` is forbidden (Principle VI).
- Every commit ≤500 lines of diff including tests (Principle VI).

**Scale/Scope**: Up to ~10,000 nodes / ~100,000 memories in a single SQLite file. Beyond that the user swaps to a vector-native adapter via the `StorageAdapter` ABC — this is exactly why the adapter pattern exists. Expected v1 usage: personal agents with hundreds-to-thousands of conversations, comfortably inside the 10K-node target.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Constitution version at time of check: **v2.0.0** (`.specify/memory/constitution.md`).

| # | Principle | Status | Evidence |
|---|---|---|---|
| I | Library-First, Framework-Agnostic Core | ✅ PASS | FR-025 forbids vendor LLM/framework SDK imports; SC-005 measures it with a file-walk test. Zero such imports anywhere in `packages/neuromem-core/src/`. numpy and pandas are NOT framework SDKs. |
| II | Lean Dependency Set | ✅ PASS | `packages/neuromem-core/pyproject.toml` will declare exactly `numpy>=1.26` and `pandas>=2.1` in its runtime dependency list, matching the default toolkit permitted by Principle II. SC-006 verifies via a TOML-parse test. No additional runtime deps required in v1. |
| III | Layered, Modular, Pluggable Architecture | ✅ PASS | `StorageAdapter` / `LLMProvider` / `EmbeddingProvider` ABCs are the contract layer (spec.md §Storage Adapter Interface, §Provider Interfaces). Orchestration layer (`system.py`, `context.py`, `tools.py`, `vectors.py`) imports only from the contract layer and from numpy/pandas. Concrete `SQLiteAdapter` lives in `storage/sqlite.py` and is never imported by orchestration. |
| IV | Neuroscience-Grounded Subsystem Design | ✅ PASS | All five canonical analogues present (spec.md §Architecture Overview → Neuroscience Mapping table). Every subsystem named. No orphan modules. |
| V | Test-First with Injected Mocks (NON-NEGOTIABLE) | ✅ PASS | `MockStorageAdapter` / `MockLLMProvider` / `MockEmbeddingProvider` mandated (contracts/providers.md, contracts/storage-adapter.md §Contract tests). Red-green-refactor cycle governs every `/speckit.tasks` task. |
| VI | Atomic Commits with Enforced Pre-Commit Gates | ✅ PASS | `.pre-commit-config.yaml` committed in commit `239f3c9` and `pre-commit install` already run. Hooks (ruff format, ruff check, pytest) active for every subsequent commit. Hook config will be refreshed when Python source lands (ruff/pytest hooks will actually fire on `.py` file commits). 500-line cap applies to every `/speckit.implement` task commit. |

**Monorepo compliance**: The Additional Constraints — Repository Structure section mandates `packages/<name>/` layout. The plan's Project Structure tree below matches this requirement.

**Result**: All 6 gates pass under v2.0.0. No violations. Complexity Tracking section below is empty.

### Post-Phase 1 re-check

After the Phase 1 artefacts (`research.md`, `data-model.md`, `contracts/public-api.md`, `contracts/storage-adapter.md`, `contracts/providers.md`, `quickstart.md`) were generated (and then revised for v2.0.0), every principle was re-checked:

| # | Principle | Post-Phase 1 Status | Phase 1 evidence |
|---|---|---|---|
| I | Library-First, Framework-Agnostic Core | ✅ PASS | `contracts/providers.md` forbids `from neuromem.system import ...` in provider implementations. `quickstart.md` keeps provider wrappers in *user* code, not `packages/neuromem-core/src/`. Framework wrappers are explicitly future sibling packages. |
| II | Lean Dependency Set | ✅ PASS | `research.md` Decision 1 declares runtime deps as `["numpy>=1.26", "pandas>=2.1"]`. Decision 2 uses numpy for vector math. Decision 3 (clustering) uses numpy broadcasting. Decision 5 (embedding storage) uses `np.frombuffer` / `.tobytes()` binary. No other deps introduced. |
| III | Layered, Modular, Pluggable Architecture | ✅ PASS | `contracts/storage-adapter.md` defines 13 abstract methods with no default implementations. `data-model.md` schema is fully owned by the adapter. Orchestration-layer modules listed explicitly and only import from contract layer + numpy + pandas. |
| IV | Neuroscience-Grounded Subsystem Design | ✅ PASS | `data-model.md` §State Machine shows the 4-status lifecycle mapping to hippocampus → neocortex → pruning. `contracts/storage-adapter.md` explicitly groups methods into "Acquisition (Hippocampus)", "Consolidation (Neocortex / Graph)", "Recall (Prefrontal Cortex)", "Forgetting (Synaptic Pruning)". |
| V | Test-First with Injected Mocks (NON-NEGOTIABLE) | ✅ PASS | `contracts/providers.md` specifies `MockEmbeddingProvider` with deterministic seeded RNG and `MockLLMProvider` with canned responses. `contracts/storage-adapter.md` lists 13 contract tests to be written first per task. |
| VI | Atomic Commits with Enforced Pre-Commit Gates | ✅ PASS | No change from pre-check. Pre-commit hooks installed and active. Task decomposition in `/speckit.tasks` will respect the 500-line cap. |

All 6 gates still pass under v2.0.0. Plan is ready for `/speckit.tasks`.

## Project Structure

### Documentation (this feature)

```text
specs/001-neuromem-core/
├── spec.md                # Feature spec (landed, updated for v2.0.0)
├── plan.md                # This file (/speckit.plan command output)
├── research.md            # Phase 0 output — tech choices & rationale
├── data-model.md          # Phase 1 output — entities, schema, state machine
├── quickstart.md          # Phase 1 output — install + minimal usage example
├── contracts/             # Phase 1 output — public interface contracts
│   ├── public-api.md      #   NeuroMemory, ContextHelper, tool functions
│   ├── storage-adapter.md #   StorageAdapter ABC
│   └── providers.md       #   EmbeddingProvider, LLMProvider ABCs
├── checklists/
│   └── requirements.md    # Specification quality checklist (landed)
└── tasks.md               # Phase 2 output (/speckit.tasks — not created here)
```

### Source Code (repository root — `uv` workspace monorepo)

```text
neuromem/                                    # repo root (git + uv workspace)
├── pyproject.toml                           # WORKSPACE ROOT:
│                                            #   [tool.uv.workspace]
│                                            #   members = ["packages/*"]
│                                            #
│                                            # No `[project]` table — the root
│                                            # is a workspace, not a package.
├── uv.lock                                  # shared lockfile for the whole repo
├── .pre-commit-config.yaml                  # landed in commit 239f3c9
├── .gitignore                               # landed in commit 239f3c9
├── README.md                                # repo-level (TBD in /speckit.implement)
├── CLAUDE.md                                # agent context file (regenerated)
├── .specify/                                # speckit scaffolding
├── specs/                                   # speckit feature specs
│   └── 001-neuromem-core/                   # this feature
└── packages/
    └── neuromem-core/                       # v1 package
        ├── pyproject.toml                   # PACKAGE CONFIG:
        │                                    #   [project]
        │                                    #   name = "neuromem-core"
        │                                    #   dependencies = [
        │                                    #     "numpy>=1.26",
        │                                    #     "pandas>=2.1",
        │                                    #   ]
        │                                    #   [build-system]
        │                                    #   requires = ["hatchling"]
        │                                    #   build-backend = "hatchling.build"
        │                                    #
        │                                    #   [tool.hatch.build.targets.wheel]
        │                                    #   packages = ["src/neuromem"]
        ├── README.md                        # package-level README
        ├── src/
        │   └── neuromem/                    # import name (top-level)
        │       ├── __init__.py              # __version__, re-exports
        │       ├── system.py                # NeuroMemory — orchestration engine
        │       ├── context.py               # ContextHelper — prompt injector
        │       ├── tools.py                 # search_memory, retrieve_memories
        │       ├── vectors.py               # cosine_similarity, batch_cosine_similarity,
        │       │                            #   compute_centroid (numpy-backed;
        │       │                            #   renamed from math.py to avoid
        │       │                            #   shadowing stdlib)
        │       ├── providers.py             # EmbeddingProvider, LLMProvider ABCs
        │       └── storage/
        │           ├── __init__.py          # re-exports StorageAdapter, SQLiteAdapter
        │           ├── base.py              # StorageAdapter ABC (contract layer)
        │           └── sqlite.py            # SQLiteAdapter (impl layer)
        └── tests/
            ├── __init__.py
            ├── conftest.py                  # MockStorageAdapter, MockLLMProvider,
            │                                #   MockEmbeddingProvider fixtures
            ├── test_vectors.py              # cosine_similarity, compute_centroid
            ├── test_storage_sqlite.py       # StorageAdapter contract tests
            ├── test_system.py               # NeuroMemory orchestration + dreaming
            ├── test_context.py              # ContextHelper + ASCII tree renderer
            ├── test_tools.py                # search_memory, retrieve_memories + LTP
            └── test_no_forbidden_imports.py # SC-005: file-walk src/ and assert
                                              #   no vendor/framework SDK imports
```

**Structure Decision**: `uv` workspace monorepo with `packages/neuromem-core/` as the sole v1 package. This is mandated by Constitution v2.0.0 Additional Constraints — Repository Structure, not a free choice.

- **Orchestration layer** (Principle III): `system.py`, `context.py`, `tools.py`, `vectors.py` — pure Python + numpy + pandas. Imports only from the contract layer, numpy, pandas, and stdlib.
- **Contract layer**: `providers.py`, `storage/base.py` — ABCs only, zero concrete implementations.
- **Implementation layer**: `storage/sqlite.py` — concrete `SQLiteAdapter` implementing `storage/base.py:StorageAdapter`.

Tests live at `packages/neuromem-core/tests/`, parallel to the package's own `src/`. When future sibling packages (`packages/neuromem-adk/`, etc.) land, each gets its own `src/` and `tests/` in the same pattern. The workspace `uv.lock` is shared across all packages so transitive dependency versions stay consistent.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

*Empty — no Constitution v2.0.0 violations. All 6 principles pass cleanly against the current technical context and project structure.*
