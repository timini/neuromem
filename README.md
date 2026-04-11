# neuromem

> Neuroscience-inspired long-term memory for AI agents — monorepo root.

`neuromem` is a `uv` workspace monorepo. The first and currently-only package is [`neuromem-core`](packages/neuromem-core/), a Python library that gives any LLM agent a persistent, self-organising memory modelled on the hippocampal → neocortex consolidation loop. Future sibling packages (framework wrappers, alternative storage adapters) will live alongside it under `packages/`.

---

## What's in `packages/`

| Package | Status | What it does |
|---|---|---|
| [`neuromem-core`](packages/neuromem-core/) | v0.1.0 alpha | The core library. Cognitive loop (hippocampus → neocortex → pruning → LTP), `StorageAdapter` ABC, bundled `SQLiteAdapter`, agent-facing `search_memory` / `retrieve_memories` tools. **Framework-agnostic** — imports no LLM or agent SDKs. |

Planned sibling packages (not yet implemented — each will land as its own feature branch):

- `neuromem-adk` — provider pair + `enqueue()` hook for Google ADK agents.
- `neuromem-anthropic` — provider pair for the Anthropic Python SDK.
- `neuromem-langchain` / `neuromem-langgraph` — framework wrappers.
- `neuromem-fastembed` — local-embedding provider (no API calls).
- `neuromem-pgvector` — Postgres/pgvector `StorageAdapter` implementation.

Each sibling package declares `neuromem-core` as a workspace dependency in its own `pyproject.toml` and publishes to PyPI under its own name.

---

## Adding a new sibling package

The workspace is set up to make this a three-step process. Detailed rules live in [Constitution §Additional Constraints — Repository Structure](.specify/memory/constitution.md).

1. **Scaffold**. Create a new directory under `packages/` (e.g. `packages/neuromem-anthropic/`) with the standard layout:

   ```
   packages/neuromem-anthropic/
   ├── src/neuromem_anthropic/
   │   └── __init__.py
   ├── tests/
   │   └── conftest.py
   ├── README.md
   └── pyproject.toml          # declares neuromem-core as a workspace dep
   ```

2. **Register** the package in the workspace-root `pyproject.toml` under `[tool.uv.workspace].members`. Run `uv sync --dev` to install it in editable mode alongside `neuromem-core`.

3. **Ship** each user story as a spec-kit feature branch (`/speckit.specify`, `/speckit.plan`, `/speckit.tasks`, `/speckit.implement`) and follow the Constitution's atomic-commit rule (≤500 lines per commit). The pre-commit hooks (ruff format, ruff check, pytest) run on every commit — never bypass with `--no-verify`.

---

## Running tests across the whole workspace

From the repo root:

```bash
# Install everything (core + dev tools)
uv sync --dev

# Run the full suite across every workspace package
uv run pytest

# Run just neuromem-core
uv run --package neuromem-core pytest packages/neuromem-core/

# Run a single test file
uv run --package neuromem-core pytest packages/neuromem-core/tests/test_system.py -v
```

Every package's `tests/` directory is collected automatically. The workspace-root `pyproject.toml` sets `filterwarnings = ["error"]`, so any `ResourceWarning` / `DeprecationWarning` escalates into a test failure — this is how the Python 3.13 sqlite3-unclosed-connection regression would have been caught at commit time.

---

## Spec-kit workflow

This repo is driven by the [spec-kit](https://github.com/github/spec-kit) workflow: every feature moves through `specify → plan → tasks → implement`, with each phase producing a durable artifact under `specs/NNN-feature-name/`. The currently-shipping feature is:

- [`specs/001-neuromem-core/`](specs/001-neuromem-core/) — initial `neuromem-core` v0.1.0 ([spec](specs/001-neuromem-core/spec.md), [plan](specs/001-neuromem-core/plan.md), [tasks](specs/001-neuromem-core/tasks.md), [quickstart](specs/001-neuromem-core/quickstart.md)).

The local slash-command skills under `.claude/skills/` (invoked as `/speckit.specify`, `/speckit.plan`, etc.) automate the workflow for Claude Code sessions.

---

## Engineering rules

Everything a contributor (or an AI assistant) needs to know about how this project is run lives in [`.specify/memory/constitution.md`](.specify/memory/constitution.md). Highlights:

- **Principle I — Framework-Agnostic Core.** `neuromem-core` never imports an LLM SDK or agent framework. Enforced by a CI tripwire test (`test_no_forbidden_imports.py`).
- **Principle II — Lean Dependency Set.** Runtime deps are exactly `numpy + pandas` plus the stdlib. Adding a new dep requires a constitutional amendment. Enforced by `test_pyproject_dependencies.py`.
- **Principle III — Layered, Modular, Pluggable.** Orchestration depends on abstract base classes; concrete implementations are injected. The `StorageAdapter` contract is exercised by a parametrised 14-test suite against every concrete backend.
- **Principle IV — Neuroscience-Grounded.** Subsystem names and behaviours map to the cognitive loop (hippocampus, neocortex, prefrontal cortex, synaptic pruning, LTP). Not marketing fluff — the tests assert the behavioural mapping.
- **Principle V — Test-First with Mocks.** NON-NEGOTIABLE. Every feature ships with tests that run end-to-end via deterministic mock providers (`MockLLMProvider`, `MockEmbeddingProvider` in `tests/conftest.py`).
- **Principle VI — Atomic Commits.** ≤500 lines per commit. Pre-commit hooks (`ruff format`, `ruff check`, `pytest`) enforce every commit. **Never `--no-verify`.**

---

## License

MIT. See [`LICENSE`](LICENSE).
