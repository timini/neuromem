<!--
SYNC IMPACT REPORT
==================
Version change: 1.1.0 → 2.0.0
Rationale: MAJOR bump. Principle II's guidance is REVERSED — numpy and
pandas are now IN the default dependency set, not opt-in extras. This
breaks the "stdlib-only math" stance that the v1.1.0 plan was written
against, which is the definition of a MAJOR change per this document's
own versioning rules. Also adds a new constraints section mandating a
`uv` workspace monorepo layout for the repository, and a handful of
stale references ("five principles" → "six", `src/neuromem/` →
`packages/neuromem-core/src/neuromem/`) are corrected in passing.

Modified principles:
  - I. Library-First, Framework-Agnostic Core
    (no semantic change; path reference updated to reflect the new
    monorepo layout: `src/neuromem/` → `packages/neuromem-core/src/neuromem/`)
  - II. Minimal Dependency Footprint → II. Lean Dependency Set
    (title updated; body reversed to permit numpy + pandas as default
    tools; rationale rewritten to explicitly call out the v1.1.0
    interpretation as "ideology, not engineering" so future amendments
    don't repeat it)

Added sections:
  - Additional Constraints — Repository Structure
    (mandates uv workspace monorepo layout with packages/ subdirectory;
    neuromem-core is the first package; future framework wrappers land
    as siblings under packages/)

Removed sections: none

Also corrected (stale from v1.1.0):
  - "five core principles" / "five principles" references in Development
    Workflow and Compliance Review now say "six" to reflect Principle VI.

Templates / artefacts requiring updates:
  - ⚠ specs/001-neuromem-core/spec.md — FR-024, FR-026, SC-005, SC-006,
    Assumptions, Package Layout, Public API Surface, Data Model all
    reference the numpy ban or the flat src/ layout. Must be revised.
  - ⚠ specs/001-neuromem-core/plan.md — Technical Context, Project
    Structure, Constitution Check table all need refresh.
  - ⚠ specs/001-neuromem-core/research.md — Decisions 1, 2, 3, 5 need
    rewriting. Add a new decision on monorepo layout.
  - ⚠ specs/001-neuromem-core/data-model.md — embedding serialisation
    section changes from JSON-text to numpy float32 binary.
  - ⚠ specs/001-neuromem-core/contracts/public-api.md,
    storage-adapter.md, providers.md — type signatures change to use
    np.ndarray.
  - ⚠ specs/001-neuromem-core/quickstart.md — install command (package
    name) and provider examples change.
  - ⚠ CLAUDE.md — must be re-generated via update-agent-context.sh
    after plan.md is revised.
  - ✅ .specify/templates/* — no change; templates reference principles
    dynamically at runtime, so the revised Principle II and new section
    are picked up without template edits.

Follow-up TODOs:
  - Workspace-root pyproject.toml with `[tool.uv.workspace] members = ["packages/*"]`
  - packages/neuromem-core/pyproject.toml declaring `numpy` and `pandas`
    as runtime dependencies
  - Physical directory move of any already-committed source into
    packages/neuromem-core/ (n/a in practice — no source code exists yet)

Previous amendments:
  - 1.1.0 (2026-04-11): MINOR bump. Added Principle VI (Atomic Commits
    with Enforced Pre-Commit Gates) and materially expanded Principles
    III and V.
  - 1.0.0 (2026-04-11): Initial ratification. MAJOR bump from placeholder
    template. Established Principles I–V, Additional Constraints
    (Concurrency & Data Safety), Development Workflow (Spec-Kit
    Discipline), and Governance sections.
-->

# neuromem Constitution

## Core Principles

### I. Library-First, Framework-Agnostic Core

`neuromem` is delivered as a Python library. The core package
(`packages/neuromem-core/src/neuromem/`) MUST NOT import any agent-framework SDK
(Google ADK, Anthropic SDK, LangChain, LlamaIndex, etc.) and MUST NOT import any
vendor LLM or embedding SDK (`openai`, `anthropic`, `google-genai`, `cohere`, etc.).
All framework wiring lives in separate, optional sibling packages under
`packages/` that depend on `neuromem-core`, never the reverse.

**Rationale**: The library must be reusable across every agent runtime without forcing
downstream users to install or pin SDKs they do not use. Coupling the core to any
framework collapses the value proposition to a single ecosystem.

### II. Lean Dependency Set

The core library ships with a small, intentional dependency set: the Python 3.10+
standard library, **numpy**, and **pandas**. These three are the default toolkit —
reach for them freely where they make code correct, clear, or measurably faster.
Numerical work (cosine similarity, centroid computation, bulk vector operations) uses
`numpy`. Tabular transformations (dreaming-cycle memory/node batches, graph traversal
result sets) use `pandas` where the clarity gain is real. Graph storage defaults to
stdlib `sqlite3`.

Beyond numpy, pandas, and the standard library, every additional runtime dependency
in the `neuromem-core` package requires a documented justification in the feature
spec **and** a constitutional amendment (MINOR bump at minimum). "We might want it
later" is never a sufficient justification. "Writing this in pure Python is 3× the
code and 20× slower, and numpy/pandas cannot express it" is a sufficient
justification. A narrow-use dependency (e.g., a vector-DB client, a specific graph
backend) MUST live in an optional-dependency group, or preferably in a downstream
sibling package under `packages/`, not in core.

Vendor LLM/embedding SDKs (`openai`, `anthropic`, `google-genai`, `cohere`, etc.)
and agent-framework SDKs (Google ADK, Anthropic SDK agent scaffolding, LangChain,
LlamaIndex, LangGraph) remain **forbidden** in `neuromem-core`. That ban is
**Principle I**, and it is unchanged by this principle. The Lean Dependency Set is
about what `neuromem-core` depends on; the Framework-Agnostic rule is about what
`neuromem-core` couples to. Keep them separate in your head.

**Rationale**: The goal is not zero dependencies — it is a dependency set that
every user's laptop already has, every CI system resolves in seconds, and every
reviewer understands without looking up an obscure package. numpy and pandas are
de facto standard library for numerical Python. Forbidding them produced dense,
unreadable pure-Python loops that nobody would write voluntarily and that everyone
would reimplement with numpy the moment they cared about performance. The v1.1.0
ban was ideology, not engineering; this revision is the correction.

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

## Additional Constraints — Repository Structure

The repository is organised as a **`uv` workspace monorepo**. All publishable
Python packages live under `packages/`, one directory per package, each with its
own `pyproject.toml`. A single workspace `pyproject.toml` at the repo root
declares `[tool.uv.workspace] members = ["packages/*"]` and owns the shared
`uv.lock`.

For v1, the monorepo contains exactly one package:

- **`packages/neuromem-core/`** — the core library. Published to PyPI as
  `neuromem-core`. Import name: `neuromem`. This is the package the whole
  001-neuromem-core feature targets.

Future framework wrapper packages MUST land as siblings under `packages/`, with
their own dedicated directories and `pyproject.toml` files. Planned (but not in
v1 scope) examples:

- `packages/neuromem-adk/` — Google ADK hook. Import: `neuromem_adk`. Depends
  on `neuromem-core` as a workspace dependency.
- `packages/neuromem-anthropic/` — Anthropic SDK hook. Import: `neuromem_anthropic`.
- `packages/neuromem-langchain/` — LangChain hook. Import: `neuromem_langchain`.
- `packages/neuromem-fastembed/` — a built-in local-embedding provider. Import:
  `neuromem_fastembed`.

Each sibling package MUST declare its dependency on `neuromem-core` using the
`uv` workspace dependency syntax so the lockfile is shared across the whole
repo, and MUST use a distinct top-level import namespace (`neuromem_adk`,
`neuromem_anthropic`, etc. — underscores, not hyphens) to avoid implicit
namespace package complexity. The `neuromem` top-level import is reserved
exclusively for `neuromem-core`.

**Rationale**: The project was always going to have multiple sibling packages.
Retrofitting a monorepo structure after the first package ships is painful —
import paths change, CI reconfigures, publish scripts break. Starting with the
workspace layout costs ~20 lines of TOML on day one and zero lines later.

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

The Constitution Check gate in `/speckit.plan` MUST verify each of the six core
principles explicitly (I–VI). A plan that violates any principle MUST either
justify the violation in a Complexity Tracking entry (with a concrete reason why
the simpler alternative was rejected) or be sent back for redesign. An unjustified
violation is a blocker, not a warning.

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
Check against all six principles (I–VI). Every code review MUST verify that tests
were written first and that adapter boundaries were not breached. Complexity MUST
be justified in the Complexity Tracking table of the relevant plan; unjustified
complexity is grounds for rejection.

Runtime development guidance for contributors lives in `AGENTS.md`, `CLAUDE.md`,
and `GEMINI.md`. Those files are subordinate to this constitution; where they
diverge, this constitution wins and those files MUST be updated.

**Version**: 2.0.0 | **Ratified**: 2026-04-11 | **Last Amended**: 2026-04-11
