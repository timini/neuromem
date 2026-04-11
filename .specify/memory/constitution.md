<!--
SYNC IMPACT REPORT
==================
Version change: 1.0.0 → 1.1.0
Rationale: MINOR bump. Added Principle VI (Atomic Commits with Enforced
Pre-Commit Gates) and materially expanded Principles III and V.

Modified principles:
  - III. Dependency Inversion via Adapters
    → III. Layered, Modular, Pluggable Architecture
    (expanded to name the layering explicitly and make one-way dependency
     direction a binding rule; original adapter requirements preserved)
  - V. Test-First with Injected Mocks
    → V. Test-First with Injected Mocks (red-green-refactor cycle now
    enumerated step-by-step as binding process; non-negotiable status
    unchanged)

Added sections:
  - Principle VI: Atomic Commits with Enforced Pre-Commit Gates
    (max 500-line commits including tests; mandatory lint/format/test
    pre-commit hooks; --no-verify forbidden)

Removed sections: none

Templates requiring updates:
  - ✅ .specify/templates/plan-template.md — Constitution Check gate
    dynamically references principles at runtime; new Principle VI
    picked up without a template edit.
  - ✅ .specify/templates/spec-template.md — compatible as-is.
  - ✅ .specify/templates/tasks-template.md — compatible as-is; task
    decomposition should now respect the 500-line commit cap.
  - ⚠  Repository has no pre-commit configuration yet. Principle VI is
     prescriptive: a `.pre-commit-config.yaml` MUST be added before the
     first /speckit.implement run.

Follow-up TODOs:
  - Add `.pre-commit-config.yaml` with ruff format, ruff check, and
    pytest (fast unit tests) as the three mandatory hooks.
  - Add `pre-commit` to dev dependencies in `pyproject.toml` once it is
    created.

Previous amendments:
  - 1.0.0 (2026-04-11): Initial ratification. MAJOR bump from placeholder
    template. Established Principles I–V, Additional Constraints
    (Concurrency & Data Safety), Development Workflow (Spec-Kit
    Discipline), and Governance sections.
-->

# neuromem Constitution

## Core Principles

### I. Library-First, Framework-Agnostic Core

`neuromem` is delivered as a pure Python library. The core package (`src/neuromem/`)
MUST NOT import any agent-framework SDK (Google ADK, Anthropic SDK, LangChain, LlamaIndex,
etc.) and MUST NOT import any vendor LLM or embedding SDK (`openai`, `anthropic`,
`google-genai`, `cohere`, etc.). All framework wiring lives in separate, optional
downstream packages that depend on `neuromem`, never the reverse.

**Rationale**: The library must be reusable across every agent runtime without forcing
downstream users to install or pin SDKs they do not use. Coupling the core to any
framework collapses the value proposition to a single ecosystem.

### II. Minimal Dependency Footprint

The core library MUST run on the Python 3 standard library alone whenever possible.
Numerical work (cosine similarity, centroid computation) MUST use `math` and built-in
sequence operations. Graph storage MUST default to stdlib `sqlite3`. The only
non-stdlib runtime dependency permitted is a lightweight HTTP client (`requests` or
`httpx`), and only when an optional transport is required. `numpy`, `pandas`,
`networkx`, `neo4j`, and any vector-DB client library MUST be opt-in extras installed
via `pyproject.toml` optional-dependency groups, never required for a baseline
install.

**Rationale**: Install time, container size, and transitive CVE surface are proxies for
how painful the library is to adopt. A library that pulls `numpy` and `torch` by
default is not lightweight, regardless of how its code reads.

### III. Layered, Modular, Pluggable Architecture

The library is organised into strict layers with one-way dependencies:

1. **Orchestration layer** — `system.py`, `context.py`, `tools.py`. Depends only on
   the contract layer. Contains the cognitive loop and rendering logic.
2. **Contract layer** — `providers.py`, `storage/base.py`. Defines `StorageAdapter`,
   `LLMProvider`, and `EmbeddingProvider` ABCs. Depends on nothing but stdlib.
3. **Implementation layer** — `storage/sqlite.py`, plugin packages. Implements the
   contract layer. May depend on external libraries.

Dependencies MUST flow in one direction only: orchestration → contract ← implementation.
The orchestration layer MUST NOT import from `storage/sqlite.py` or from any concrete
provider. Every cross-layer reference MUST be a dependency inversion: higher layers
declare interfaces, lower layers implement them. No exceptions.

Alternative backends (Postgres, Firebase, Qdrant, Neo4j) and alternative providers
MUST be implementable by adding a new module under `storage/` or a separate plugin
package, without touching any file in the orchestration or contract layers.

**Rationale**: A layered architecture with enforced dependency inversion is the only
way the library can grow new backends, new providers, and new framework hooks
without triggering a rewrite. Shortcuts that bypass the ABCs weld the core to one
implementation forever — and once welded, they are never pried apart.

### IV. Neuroscience-Grounded Subsystem Design

Every subsystem in the library MUST map to a named cognitive analogue and MUST
preserve that analogue in its documented mechanism. The mapping is:

- Acquisition ⇔ Hippocampus (fast write, deferred processing)
- Consolidation / Dreaming ⇔ Neocortex (asynchronous batch clustering)
- Forgetting / Pruning ⇔ Synaptic Pruning (exponential decay, archival not deletion)
- Reinforcement ⇔ Long-Term Potentiation (retrieval spikes `access_weight`)
- Contextual Recall ⇔ Prefrontal Cortex (query-local sub-graph projection)

A new feature that cannot be placed under one of these five subsystems MUST be
justified explicitly in its spec, because it implies the biological model is
incomplete and the architecture needs revision, not a sixth orphan module.

**Rationale**: The neuroscience mapping is not decoration — it is the organising
principle that keeps the codebase cohesive as features land. Dropping it would leave
the project as another bag of loosely related RAG utilities.

### V. Test-First with Injected Mocks (NON-NEGOTIABLE)

Every public class and function MUST be testable without network access and without
real LLM or embedding calls. The test suite MUST include a `MockStorageAdapter`, a
`MockLLMProvider`, and a `MockEmbeddingProvider`, and these mocks MUST satisfy the
same ABC contracts as the production implementations.

**All new code MUST follow the red-green-refactor cycle:**

1. **Red** — Write a failing test that specifies the desired behaviour. Run the
   test suite and confirm the new test fails for the right reason (not a
   syntax error, not a missing import).
2. **Green** — Write the minimum production code required to make the failing test
   pass. No gold-plating. No code that isn't justified by a failing test.
3. **Refactor** — Improve the structure of the code without changing its observable
   behaviour. All tests MUST remain green throughout.

A commit that adds production code without a preceding (or co-committed) failing
test for the same behaviour MUST be rejected in review. Test coverage of
`system.py`, `context.py`, `tools.py`, and `math.py` MUST remain at or above 90%
line coverage at all times.

**Rationale**: Red-green-refactor is the only process that guarantees every line of
production code was driven by a test that would have failed without it. Skipping
**red** means the test may be tautologically passing; skipping **green** means
gold-plating enters the codebase unmeasured; skipping **refactor** means tech debt
compounds silently. Mocks are the only way to keep the cognitive loop deterministic
under test — a memory library whose behaviour depends on opaque API responses is
untestable in CI, unreproducible in bug reports, and unreviewable by anyone not
holding the same API keys.

### VI. Atomic Commits with Enforced Pre-Commit Gates

Commits MUST be atomic and small. A single commit MUST NOT exceed **500 lines of
diff total, including tests**. A feature that cannot be delivered in one ≤500-line
commit MUST be broken into a sequence of commits, each of which individually
compiles, passes the full test suite, and represents a single coherent step.

The repository MUST configure pre-commit hooks (via the `pre-commit` framework or
an equivalent mechanism) that run on every commit and enforce, at minimum:

1. **Formatter** — e.g., `ruff format` or `black`. Auto-fixes are accepted and
   re-staged.
2. **Linter** — e.g., `ruff check`. Zero new warnings permitted.
3. **Test suite** — at minimum the fast unit tests touching changed modules. MUST
   pass before the commit is allowed to land.

**Pre-commit hooks MUST NEVER be bypassed.** The `git commit --no-verify` flag is
forbidden. If a hook is failing, fix the underlying problem — do not skip the hook.
If a hook is genuinely wrong (false positive, misconfiguration), the hook
configuration itself MUST be amended in a separate, justified commit; the failing
commit MUST wait until the hook is corrected.

**Rationale**: Small atomic commits are the only way to keep `git bisect` useful,
code review tractable, and rollback safe. A 2000-line commit is not reviewable —
it is rubber-stamped. Pre-commit hooks are the only enforcement mechanism that
catches style, lint, and test regressions *before* they pollute history. Bypassing
hooks renders all three disciplines ornamental: either they are enforced on every
commit or they may as well not exist. A single `--no-verify` normalises skipping,
and skipping becomes the default.

## Additional Constraints — Concurrency & Data Safety

The dreaming pipeline MUST run in a single background thread. Concurrent dreaming
cycles MUST be prevented by a `threading.Lock` held on the `NeuroMemory` instance.
The double-buffer status flip (`inbox` → `dreaming` → `consolidated`) is the sole
inter-thread synchronisation primitive and MUST be used for every batch.

Archival MUST NOT delete memory content. A memory moved to `status='archived'` MUST
retain its `raw_content` and `summary` columns intact; only its active graph edges
may be removed from traversal queries. A failed dreaming cycle (any exception in any
phase) MUST roll back the batch status from `dreaming` back to `inbox` so the batch
is retried on the next cycle; the database MUST NOT be left with orphaned `dreaming`
rows.

`enqueue()` is latency-critical and MUST return in under 50 ms on the caller thread
(excluding the optional summary LLM call whose placement is a configurable design
choice). `search_memory()` MUST return a rendered ASCII tree in under 500 ms for a
graph of up to 10,000 nodes using the default SQLite adapter.

## Development Workflow — Spec-Kit Discipline

Every non-trivial feature MUST flow through the speckit pipeline in order:

1. `/speckit.specify` → `specs/NNN-feature/spec.md` (user stories, FRs, SCs, open
   questions)
2. `/speckit.clarify` → resolve `[NEEDS CLARIFICATION]` blocks before planning
3. `/speckit.plan` → `specs/NNN-feature/plan.md` (technical context, Constitution
   Check gate against all five principles, project structure decision)
4. `/speckit.tasks` → `specs/NNN-feature/tasks.md` (ordered, test-first task list)
5. `/speckit.implement` → code, with each task's tests landing before its
   implementation

The Constitution Check gate in `/speckit.plan` MUST verify each of the five core
principles explicitly. A plan that violates any principle MUST either justify the
violation in a Complexity Tracking entry (with a concrete reason why the simpler
alternative was rejected) or be sent back for redesign. An unjustified violation is
a blocker, not a warning.

Pull requests MUST reference the spec file they implement and MUST include the tests
required by Principle V. Commits that skip the speckit pipeline are permitted only
for typo fixes, CI configuration, and documentation corrections.

## Governance

This constitution supersedes all other development practices in the repository. Any
conflict between this document and an ad-hoc convention, agent instruction file,
README, or inline comment is resolved in favour of the constitution.

**Amendments** MUST be proposed via a PR that edits `.specify/memory/constitution.md`
directly. The PR description MUST state the version bump type and rationale,
following semantic versioning:

- **MAJOR**: Backward-incompatible change — removing or redefining a principle,
  deleting a section, changing the meaning of an existing rule in a way that breaks
  prior plans.
- **MINOR**: Adding a new principle or section, or materially expanding guidance
  under an existing principle.
- **PATCH**: Wording clarifications, typo fixes, non-semantic refinements.

Amendments MUST be accompanied by a Sync Impact Report (prepended as an HTML comment
at the top of this file) listing modified principles, added/removed sections, and
templates requiring downstream updates.

**Compliance review**: Every `/speckit.plan` run performs an automatic Constitution
Check against all five principles. Every code review MUST verify that tests were
written first and that adapter boundaries were not breached. Complexity MUST be
justified in the Complexity Tracking table of the relevant plan; unjustified
complexity is grounds for rejection.

Runtime development guidance for contributors lives in `AGENTS.md`, `CLAUDE.md`,
and `GEMINI.md`. Those files are subordinate to this constitution; where they
diverge, this constitution wins and those files MUST be updated.

**Version**: 1.1.0 | **Ratified**: 2026-04-11 | **Last Amended**: 2026-04-11
