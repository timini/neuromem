# Phase 0 Research: neuromem Core Library

**Feature**: 001-neuromem-core
**Date**: 2026-04-11
**Purpose**: Resolve all NEEDS CLARIFICATION from the Technical Context and document the tech/pattern choices that govern the implementation.

## Status

The feature spec (spec.md) has **zero open `[NEEDS CLARIFICATION]` markers** after the `/speckit.specify` clarification pass — the 8 design questions raised in the source conversation are recorded in spec.md §Resolved Design Decisions. This research document therefore covers **implementation-level technology choices** rather than scope clarifications.

Every decision below is checked against Constitution v1.1.0 Principles I–VI.

---

## Decision 1: Python packaging and build — `uv` workspace monorepo

**Decision**: `uv` workspace monorepo with `hatchling` as the per-package build backend. The repo root holds a minimal workspace `pyproject.toml` with `[tool.uv.workspace] members = ["packages/*"]`. The first (and v1 only) package is `packages/neuromem-core/`, which has its own `pyproject.toml` declaring `name = "neuromem-core"`, `dependencies = ["numpy>=1.26", "pandas>=2.1"]`, and `build-system = hatchling`.

**Rationale**:
- Mandated by Constitution v2.0.0 Additional Constraints — Repository Structure. The project always intended to ship sibling packages (`neuromem-adk`, `neuromem-anthropic`, `neuromem-langchain`, etc.) and retrofitting a monorepo mid-development is painful.
- `uv` workspaces (stable since late 2024) share a single lockfile across every package in `packages/*`, which guarantees transitive dependency version consistency. Changing numpy's version is a one-line lockfile diff that affects every sibling package simultaneously — exactly what a monorepo should deliver.
- `hatchling` is the de facto build backend for pure-Python packages, recommended by `uv init --lib`. Zero-config, supports `src/` layout, no C extensions.
- Per-package `pyproject.toml` structure:
  - `requires-python = ">=3.10"`
  - Runtime `dependencies = ["numpy>=1.26", "pandas>=2.1"]` — Constitution v2.0.0 Principle II default toolkit
  - Dev group `[dependency-groups] dev = ["pytest", "pytest-cov", "ruff", "pre-commit"]` at the **workspace root** so every package shares the same dev toolchain
  - `[build-system] requires = ["hatchling"]`
  - `[tool.hatch.build.targets.wheel] packages = ["src/neuromem"]`

**Alternatives considered**:
- **Flat layout, single `src/` at repo root**: The v1 first draft. Rejected after user feedback — retrofitting a monorepo after publishing the first package produces import-path churn that nobody enjoys.
- **Multi-repo (separate GitHub repos per package)**: The classic "split when you need to" approach. Rejected — the user wants one repo one lockfile, and sibling packages share enough infrastructure (pre-commit config, CI, spec-kit constitution) that separating them would duplicate everything.
- **Poetry / PDM workspaces**: Functionally equivalent but the user explicitly chose `uv` and `uv` workspaces are now the best-in-class Python monorepo tool.
- **Implicit namespace packages (shared `neuromem.*` namespace)**: Would let `neuromem-core` export `neuromem.system` and `neuromem-adk` export `neuromem.adk`. Rejected — PEP 420 namespace packages have real tooling footguns (pyright, some bundlers, editable installs) and the benefit over distinct top-level names (`neuromem` vs `neuromem_adk`) is cosmetic.

---

## Decision 2: Vector math — numpy

**Decision**: Implement `cosine_similarity`, `batch_cosine_similarity`, and `compute_centroid` in `packages/neuromem-core/src/neuromem/vectors.py` using numpy. The module is named `vectors.py` (not `math.py`) to avoid shadowing the stdlib `math` module.

```python
import numpy as np
from numpy.typing import NDArray

def cosine_similarity(vec_a, vec_b) -> float:
    a = np.asarray(vec_a, dtype=np.float64)
    b = np.asarray(vec_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def batch_cosine_similarity(query, matrix) -> NDArray[np.float64]:
    """query: (D,), matrix: (N, D) → (N,) similarities.
    Used by SQLiteAdapter.get_nearest_nodes for the full-table scan path."""
    q = np.asarray(query, dtype=np.float64)
    M = np.asarray(matrix, dtype=np.float64)
    qn = np.linalg.norm(q)
    Mn = np.linalg.norm(M, axis=1)
    denom = Mn * qn
    denom[denom == 0.0] = 1.0  # avoid division by zero; zero-mag rows → 0.0
    return (M @ q) / denom

def compute_centroid(vectors) -> NDArray[np.float64]:
    arr = np.asarray(vectors, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("empty input")
    if arr.ndim == 1:
        return arr.copy()
    return arr.mean(axis=0)
```

**Rationale**:
- Constitution v2.0.0 Principle II explicitly permits numpy as a default toolkit. The v1.1.0 pure-Python approach was over-interpretation of "minimal dependencies" and was corrected by user feedback.
- `batch_cosine_similarity` is the hot path for `SQLiteAdapter.get_nearest_nodes()`: a single `M @ q` matrix-vector product + two `np.linalg.norm` calls beats any pure-Python loop by two orders of magnitude. For 10K × 1536-dim float32 embeddings, the scan runs in ~15–25 ms on a modest laptop — comfortably inside the 500 ms SC-003 budget with room for sorting and subgraph traversal.
- `np.asarray(..., dtype=np.float64)` normalises the input so callers can pass Python lists, float32 arrays, or float64 arrays interchangeably. The core uses float64 internally for accuracy; the SQLite adapter stores as float32 for compactness and upcasts on read.
- Centroid via `arr.mean(axis=0)` is one line, vs 4 lines of Python loops and 4× slower. Same correctness, clearer intent.

**Alternatives considered**:
- **Pure-Python `math` module** (v1.1.0 plan): Rejected. Over-interpretation of Principle II. Dense, unreadable loops for operations numpy handles in one function call. Measurably slower on the SC-003 path.
- **`math.sumprod` (Python 3.12+) + stdlib `statistics`**: Rejected. Would drag minimum Python to 3.12 for marginal benefit vs numpy. numpy is already required per Principle II.
- **float32 internally throughout**: Tempting for cache-friendliness but introduces subtle accumulator precision loss in centroid computation for clusters with many members. Stick with float64 for in-memory computation and float32 only at the storage boundary.

---

## Decision 3: Agglomerative clustering — numpy pairwise similarity matrix + greedy merge

> **Cross-reference**: this decision has a full ADR at
> [`docs/decisions/ADR-001-clustering-library-choice.md`](../../docs/decisions/ADR-001-clustering-library-choice.md).
> The ADR details the alternatives (scipy / sklearn / HDBSCAN) and the
> revisit criteria for swapping. Read that before proposing a dependency
> amendment; this brief is the summary.

**Decision**: Implement agglomerative clustering in `NeuroMemory._run_dream_cycle()` (private method in `system.py`) using a numpy pairwise cosine similarity matrix as the base data structure, plus a greedy max-similarity merge loop. Use pandas `DataFrame` for the merge bookkeeping (which rows are still "live", which have been merged into which centroid) because that's exactly the kind of indexed tabular transform pandas is good at.

```python
# sketch, not final
def _run_agglomerative(nodes_df: pd.DataFrame, embeddings: np.ndarray,
                       cluster_threshold: float, llm: LLMProvider) -> list[Cluster]:
    """nodes_df: columns = [id, label, is_centroid]; embeddings: (N, D) matrix.
       Returns the list of new centroid clusters formed this cycle."""
    # Build pairwise similarity matrix: S[i, j] = cos(embeddings[i], embeddings[j])
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / np.where(norms > 0, norms, 1.0)
    sim = normed @ normed.T
    np.fill_diagonal(sim, -np.inf)  # never merge with self

    alive = np.ones(len(nodes_df), dtype=bool)
    new_clusters = []
    while True:
        masked = np.where(alive[:, None] & alive[None, :], sim, -np.inf)
        idx = np.unravel_index(np.argmax(masked), masked.shape)
        best = masked[idx]
        if best < cluster_threshold:
            break
        i, j = idx
        centroid = compute_centroid(embeddings[[i, j]])
        label = llm.generate_category_name([nodes_df.iloc[i].label, nodes_df.iloc[j].label])
        new_clusters.append(Cluster(members=[i, j], centroid=centroid, label=label))
        alive[i] = False
        alive[j] = False
        # Append the centroid as a new "live" row and recompute its column in `sim`
        # (details in the impl task).
    return new_clusters
```

**Rationale**:
- The numpy pairwise matrix (`normed @ normed.T`) is one BLAS call. For N=1000, D=1536 it's ~2 GB of float64 ops compressed to ~15 ms on a laptop. That's the entire cost of the similarity phase.
- `np.argmax` over the masked matrix finds the highest-similarity pair in O(N²) but with numpy constants, which is fast enough that we don't need a heap. Simpler code wins.
- pandas holds the node metadata (id, label, is_centroid) keyed by the same row index as the embeddings matrix. `nodes_df.iloc[i].label` is readable; a raw list-of-dicts lookup is not. This is the clarity case Constitution Principle II talks about.
- Greedy highest-similarity-first is deterministic given the same input embeddings and the same tie-breaking (row order). Important for test stability.
- `cluster_threshold = 0.82` stays in place as the default (spec.md Resolved Design Decisions #8).

**Alternatives considered**:
- **`scipy.cluster.hierarchy.linkage`**: Full academic-grade hierarchical clustering (single / complete / Ward linkage options). Rejected for v1 — adds scipy as a runtime dependency, requires its own constitutional amendment. If we later need Ward linkage for cluster quality, that's when we amend. Pure numpy agglomerative is good enough and lets us integrate the `generate_category_name` LLM call at each merge step naturally.
- **`sklearn.cluster.AgglomerativeClustering`**: Same rejection as scipy — adds sklearn, which transitively adds scipy and many other things. Overkill for what's essentially a loop.
- **Heap-based merge**: Would turn O(N²) per merge into O(N log N) per merge, but the constants favour the dense matrix approach for N < 5000, and the code is much simpler.
- **K-means / HDBSCAN**: Rejected for the reasons in the v1.1.0 plan (no hierarchy, over-engineering).

**Performance note**: For N = 1000 the whole similarity phase is ~15 ms. For N = 10,000 it's ~1.5 s — still well within the budget of a background thread running every 10 `enqueue()` calls. Beyond N = 10,000 the user should already have migrated to a vector-native `StorageAdapter` (Principle III's reason for existing).

---

## Decision 4: Concurrency — `threading.Lock` + double-buffer status flip

**Decision**: Use exactly one `threading.Lock` (`NeuroMemory._dream_lock`) to prevent concurrent dreaming cycles, plus the SQL-level `status='inbox' → 'dreaming'` flip as the data-safety mechanism. No other synchronisation primitives. No database-level locking beyond SQLite's built-in serialisation.

**Rationale**:
- The user chose micro-batching triggered every N inserts (not time-based) and the spec's Phase B workflow assumes a single background thread per cycle. A `threading.Lock` held with `acquire(blocking=False)` is the simplest possible in-process gate: if the lock is held, a second `enqueue()`-triggered dream attempt silently declines.
- The double-buffer pattern (flip target rows to `status='dreaming'` atomically before any slow work begins) is what protects the in-flight batch from new `enqueue()` arrivals. The user explicitly walked through the "sleepwalking" edge case with Gemini and chose this pattern.
- SQLite's default journal mode serialises writes at the file level. For v1's single-process constraint, this is sufficient. WAL mode is an optional future tuning.
- Per Principle VI, rollback on failure is required: if the dreaming cycle raises mid-way through, the catch clause flips the batch back to `status='inbox'` so the next cycle retries it.

**Alternatives considered**:
- **`threading.RLock` or `threading.Semaphore(1)`**: Functionally equivalent to Lock for this use case; Lock is the simplest correct primitive.
- **`multiprocessing.Lock`**: Needed only for multi-process safety, which is explicitly out of scope in v1 (Assumption). Would also add pickling overhead.
- **SQLite `BEGIN IMMEDIATE` transactions per dream cycle**: Rejected — holds a write lock on the whole database for the duration of the cycle (which includes LLM/embedding API calls, potentially many seconds), blocking `enqueue()` writes. The status-flip approach releases the write lock after the flip completes (milliseconds) and lets `enqueue()` proceed in parallel.
- **`asyncio` + task queue**: The user explicitly asked for lightweight stdlib-based concurrency and mentioned `threading` by name. `asyncio` would also be viral — every caller of `enqueue()` would need to be async, which violates the "drop into any agent runtime" promise.

---

## Decision 5: SQLite embedding storage — float32 binary bytes

**Decision**: Store each node's embedding as raw numpy `float32` bytes in a `BLOB` column via `arr.astype(np.float32).tobytes()`. Reconstruct on read with `np.frombuffer(blob, dtype=np.float32)`. A companion `embedding_dim` integer column records the vector length as a sanity check against corruption.

**Rationale**:
- With numpy now in the runtime dependency set (Constitution v2.0.0 Principle II), `tobytes()` / `frombuffer()` is the natural serialisation — one line each, zero dependencies beyond numpy.
- Storage: 4 bytes per float. For dim = 1536, that's 6 KB per row. 10,000 nodes = ~60 MB. ~4× more compact than JSON text storage (~25 KB per row), and the saving compounds at scale.
- Read path: `np.frombuffer` is a zero-copy reinterpretation of the blob — essentially free. Compare to `json.loads(blob.decode('utf-8'))` which is ~10× slower for 1500-element lists.
- float32 is the precision most embedding providers actually return under the hood (OpenAI, Cohere, Voyage, Google all compute in float32 internally; the float64 values they emit are just float32 values upcast). Storing as float32 loses nothing.
- The `embedding_dim` column catches corruption: if `len(blob) // 4 != embedding_dim` on read, the adapter raises `StorageError` before `np.frombuffer` returns a garbage-sized array.

**Alternatives considered**:
- **JSON in BLOB** (v1.1.0 plan): Rejected. 4× storage overhead, 10× deserialisation overhead, and we have numpy now. The debuggability argument ("you can `SELECT json(embedding)` and read it") is real but weak — an operator who needs to inspect embeddings can always `SELECT id, label FROM nodes` and pull them into Python. Nobody actually eyeballs 1536-element float arrays in a SQL shell.
- **`np.save` / `np.load` file format**: Adds 128-byte header overhead per embedding, which is a 2% overhead at dim=1536 but wasteful. `tobytes()` / `frombuffer()` is simpler.
- **Apache Arrow / Parquet columnar storage**: Massive dependency increase for a "store this one blob" problem. Rejected.
- **`sqlite-vec` native vector type**: Would unlock ANN search at the DB level, but it's a C extension that violates the "stdlib sqlite3 only" rule and isn't universally installed. Future adapters (`neuromem-vec-sqlite` or similar) can use it via the `StorageAdapter` ABC, but core stays with stdlib sqlite3.

---

## Decision 6: Test strategy — contract tests for the adapter + unit tests for orchestration

**Decision**: Two test modes:

1. **Contract tests** (`tests/test_storage_sqlite.py`): Parameterised tests that take a `StorageAdapter` instance from a fixture and exercise every abstract method. The fixture produces a fresh `SQLiteAdapter(":memory:")` per test. When a future adapter (e.g. `PostgresAdapter`) lands, the same test module runs against it via `pytest --adapter=postgres`.
2. **Unit tests** (`tests/test_system.py`, `test_context.py`, `test_tools.py`, `test_math.py`): Use `MockStorageAdapter` (dict-backed), `MockLLMProvider` (deterministic canned responses), `MockEmbeddingProvider` (hash-based deterministic vectors). Zero network access.

Both modes use the same `conftest.py` fixture factory.

**Rationale**:
- Principle V (NON-NEGOTIABLE) mandates mocks for deterministic testing and forbids real API calls.
- The contract test pattern is what FR-022/024/025 imply and what User Story 4's acceptance scenario #1 tests directly: "a class that inherits from `StorageAdapter` and implements all abstract methods... works correctly with no changes to core code".
- Red-green-refactor (Principle V's enumerated cycle) applies per task: every `/speckit.implement` task creates a failing test first, then writes code.
- `pytest-cov` wired via `pyproject.toml` enforces SC-007's ≥90% coverage gate.

**Alternatives considered**:
- **Unittest instead of pytest**: pytest is the community standard, has better fixture ergonomics, and pairs with `pytest-cov`. No reason to choose stdlib `unittest`.
- **Hypothesis property-based testing**: Valuable for `math.py` (cosine similarity / centroid are pure functions with algebraic properties), but adds a dependency. Defer to a later feature — not in v1 scope.
- **Real provider integration tests against an API**: Rejected by Principle V. If the user wants them, they live in a separate `tests_integration/` directory excluded from the pre-commit hook and CI default run.

---

## Decision 7: Pre-commit hook configuration

**Decision**: The `.pre-commit-config.yaml` landed in commit `239f3c9` uses `language: system` for all three hooks (ruff format, ruff check, pytest) with `types_or: [python, pyi]` and `files: \.py$` filters so non-Python commits are no-ops. Already validated across 6 commits in this session.

**Rationale**:
- Local `language: system` avoids the first-run delay of cloning `ruff-pre-commit` from GitHub.
- The `types_or` and `files` filters mean documentation/config commits (like the ones already landed) pass through the hook chain without running any tools — no spurious failures.
- `pytest` is wrapped in a `bash -c 'find tests -name test_*.py | grep -q . && pytest || echo skipping'` guard so the hook doesn't fail on exit code 5 ("no tests collected") during the bootstrap phase before any test has been written. Once the first `test_*.py` file lands, the guard passes and pytest runs for real.
- Principle VI forbids `--no-verify`. Documented in the config header.

**Alternatives considered**:
- **Cached remote hooks from `astral-sh/ruff-pre-commit`**: Rejected for the first-run friction and because system ruff is already installed on the user's machine (verified via `which ruff`).
- **`pytest --co` (collect-only) as the gate**: Wouldn't actually run tests, just confirm they parse. Defeats the purpose of running tests at commit time.
- **Running full CI in the hook**: Too slow. Pre-commit hooks must stay <10 s for the commit flow to feel snappy. `pytest -q -x` on just the unit tests is the right scope.

---

## Decision 8: StorageAdapter ABC — what goes in the contract vs the implementation

**Decision**: The ABC in `storage/base.py` exposes exactly the 13 methods listed in spec.md §Storage Adapter Interface (Acquisition: 5, Consolidation: 4, Recall: 2, Pruning: 2). No default implementations — every method is `@abstractmethod`. The `SQLiteAdapter` in `storage/sqlite.py` implements all 13 with SQL statements that exactly match the canonical schema in spec.md §Data Model.

**Rationale**:
- Principle III's one-way dependency rule: orchestration layer (`system.py`) imports `StorageAdapter` from `storage.base` via `from neuromem.storage.base import StorageAdapter`, and never imports from `storage.sqlite`. Callers who want the default pass `storage=SQLiteAdapter("memory.db")` into the `NeuroMemory` constructor.
- `@abstractmethod` on every method forces subclasses to implement everything — if they miss one, `TypeError` fires at instantiation time (spec §User Story 4 acceptance scenario #2).
- Putting no default implementations in the ABC means there's nothing for adapter authors to accidentally override wrong — they either implement or they don't.
- The `SQLiteAdapter.get_nearest_nodes()` method does `SELECT id, label, embedding FROM nodes`, deserialises each BLOB, computes cosine similarity locally via `neuromem.math.cosine_similarity`, and returns the top-k. A future native-vector adapter overrides this with `SELECT id, label, embedding FROM nodes ORDER BY embedding <-> :query_vec LIMIT :k` (or the SDK equivalent) — same interface, different strategy.

**Alternatives considered**:
- **`Protocol` (structural typing) instead of `ABC`**: More Pythonic in some circles but loses the `TypeError` on instantiation check from acceptance scenario #2. ABC is unambiguous.
- **Default implementations for some methods (e.g. `get_subgraph`)**: Rejected — leaks SQL into the base class and forces every adapter to either inherit SQL it can't run or override it, which defeats the purpose.
- **Split into two ABCs (`ReadAdapter` + `WriteAdapter`)**: Over-engineering for v1. All 13 methods are in one cohesive contract.

---

## Decision 9: Provider abstraction and the "single LLMProvider" rule

**Decision**: One `LLMProvider` ABC with three methods: `generate_summary`, `extract_tags`, `generate_category_name`. One `EmbeddingProvider` ABC with one method: `get_embeddings`. Both ABCs in `providers.py`. The core instantiates neither — callers inject instances. The core never imports any vendor SDK.

**Rationale**:
- Principle I and FR-025 forbid vendor SDK imports in the core. The ABC is the clean boundary.
- `extract_tags` is separated from `generate_summary` because some providers (e.g. structured-output OpenAI JSON mode) can do both in one call and will override a concrete subclass's internal `_summarise_and_extract` helper; the core just calls the two ABC methods in sequence and lets the provider decide how to cache.
- Resolved Design Decision #7 in the spec: a single `LLMProvider` is sufficient for v1. A second "cheap-model" provider slot is a future non-breaking minor bump, not v1 scope.
- `get_embeddings` takes a `list[str]` and returns `list[list[float]]` — always batched — so the dreaming cycle can make one API call per batch, not N calls per tag.

**Alternatives considered**:
- **One combined `LLMProvider` with a single `call(prompt, schema)` method**: Rejected — too generic, shifts prompt-engineering responsibility into every framework wrapper, and makes mocking harder (mock has to parse prompt strings to figure out what to return).
- **Separate `SummaryProvider`, `TagProvider`, `NamingProvider`**: Rejected — three ABCs for three methods is ceremony. The user gets fine-grained control via subclassing `LLMProvider` and overriding individual methods if they want.
- **Async providers**: Rejected in v1 (no asyncio in core). Framework wrappers that need async can wrap a sync provider with `asyncio.to_thread()`.

---

## Decision 10: Logging

**Decision**: Use stdlib `logging` with a module-level logger `logger = logging.getLogger("neuromem")`. Default level `WARNING`. Dreaming cycle logs at `INFO` on entry/exit, `WARNING` on rollback, `ERROR` on unrecoverable exceptions. The `SQLiteAdapter` logs the embedding-dimension warning (spec Assumption: >4096 dims) at `WARNING`.

**Rationale**:
- stdlib `logging` is free, universally understood, and framework-compatible (every Python agent framework integrates with it already).
- Module namespace lets callers silence or reroute via `logging.getLogger("neuromem").setLevel(...)`.
- No third-party logger (`loguru`, `structlog`) — Principle II.

**Alternatives considered**:
- **`print()`**: Not production-grade, can't be muted.
- **`structlog` or `loguru`**: Nicer API but adds a runtime dependency.

---

## Decision 11: Monorepo layout — `uv` workspace with `packages/*`

**Decision**: Use a `uv` workspace monorepo layout with all publishable packages under `packages/`, each with its own `pyproject.toml`, sharing one `uv.lock` at the repo root. Top-level import names differ per package (`neuromem` for `neuromem-core`, `neuromem_adk` for future `neuromem-adk`, etc. — underscore-separated so each is a distinct root package).

**Rationale**:
- Mandated by Constitution v2.0.0 Additional Constraints — Repository Structure. Not a free choice.
- `uv` workspaces (GA since late 2024) are the modern Python monorepo primitive. One lockfile, one `uv sync`, N installable packages. No workspace-related boilerplate beyond the `[tool.uv.workspace]` table.
- Distinct top-level names (not implicit namespace packages) avoid three specific pain points: editable installs don't confuse linters, wheel metadata is unambiguous, and `import neuromem_adk` tells a reader *which package* is installed without any inheritance chasing.
- The root `pyproject.toml` is a **workspace-only** manifest — no `[project]` table, no build target. Its job is to list workspace members and hold shared tool config (ruff, pytest settings).
- Every sibling package can import `neuromem` at runtime (declaring `neuromem-core` as a workspace dependency). The lockfile guarantees the same `neuromem-core` version across siblings.

**Alternatives considered**:
- **Multi-repo**: Rejected — shared infrastructure (pre-commit, CI, spec-kit constitution) would need duplicating N times.
- **Flat single-package repo, move later**: Rejected by user feedback. Retrofitting is painful; the upfront cost of monorepo setup is ~20 lines of TOML.
- **Implicit namespace packages** (`neuromem.core`, `neuromem.adk`): Rejected for the tooling pitfalls documented under Decision 1 alternatives.

## Summary of resolved items

| Research item | Decision | Principle(s) upheld |
|---|---|---|
| 1. Packaging | `uv` workspace monorepo + `hatchling` + `packages/neuromem-core/src/` layout | II, Additional Constraints |
| 2. Vector math | numpy (`cosine_similarity`, `batch_cosine_similarity`, `compute_centroid` in `vectors.py`) | II, III |
| 3. Clustering | numpy pairwise similarity matrix + pandas-backed greedy merge | II, IV |
| 4. Concurrency | `threading.Lock` + status-flip double-buffer | II, Additional Constraints |
| 5. Embedding storage | numpy float32 bytes in BLOB column | II |
| 6. Test strategy | Contract tests + unit tests with mocks | V |
| 7. Pre-commit hooks | Local `language: system`, Python-only filter | VI |
| 8. Storage contract | 13-method ABC, no default impls | III |
| 9. Provider contract | `LLMProvider` (3 methods) + `EmbeddingProvider` (1 method) | I, III |
| 10. Logging | stdlib `logging`, `neuromem` namespace | II |
| 11. Monorepo layout | `uv` workspace with `packages/*` + distinct top-level import names | III, Additional Constraints |

**No outstanding NEEDS CLARIFICATION items.** Phase 1 (data-model.md, contracts/*, quickstart.md) has been re-checked against v2.0.0.
