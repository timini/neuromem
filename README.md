# neuromem

> Neuroscience-inspired long-term memory for AI agents — `uv` workspace monorepo.

`neuromem` gives any LLM agent a persistent, self-organising memory modelled on the hippocampus → neocortex consolidation loop. The framework-agnostic core library lives in [`neuromem-core`](packages/neuromem-core/); vendor- and framework-specific siblings sit alongside it under `packages/`.

**Latest benchmark:** 0.870 on LongMemEval-s (n=100) with `gemini-3-flash-preview` — above the published EmergenceMem baseline (0.86), approaching OMEGA's 0.954. See [`docs/benchmarks/`](docs/benchmarks/).

---

## What's in `packages/`

### Core

| Package | Role |
|---|---|
| [`neuromem-core`](packages/neuromem-core/) | Cognitive loop (hippocampus → neocortex → pruning → LTP), `StorageAdapter` ABC, bundled `SQLiteAdapter`, agent-facing `search_memory` / `retrieve_memories` / `expand_node` tools. Runtime deps: `numpy + pandas + hdbscan` + stdlib. **No LLM or agent SDK imports.** |

### LLM / embedding providers

Each provider package implements the `LLMProvider` + (usually) `EmbeddingProvider` ABCs from `neuromem.providers`. Callers pick one — or mix-and-match, e.g. Claude for answering, OpenAI for embeddings.

| Package | LLM | Embedder | Default model |
|---|---|---|---|
| [`neuromem-gemini`](packages/neuromem-gemini/) | Google Gemini | ✓ | `gemini-flash-latest` |
| [`neuromem-openai`](packages/neuromem-openai/) | OpenAI | ✓ | `gpt-4.1-mini` + `text-embedding-3-small` |
| [`neuromem-anthropic`](packages/neuromem-anthropic/) | Claude | ✗ (pair with openai / gemma) | `claude-sonnet-4-6` |
| [`neuromem-gemma`](packages/neuromem-gemma/) | Local Gemma via Ollama | ✓ | `gemma3` + `embeddinggemma` |

### Framework wrappers

| Package | Role |
|---|---|
| [`neuromem-adk`](packages/neuromem-adk/) | One-line `enable_memory(agent)` for Google ADK agents — installs the `search_memory` / `retrieve_memories` / `expand_node` function tools and wires per-turn capture via `after_agent_callback`. |

### Tooling

| Package | Role |
|---|---|
| [`neuromem-bench`](packages/neuromem-bench/) | Benchmark harness. LongMemEval runner with trace-by-default, retry, instance-level resilience, and `--workers N` parallel execution. Internal tool — no PyPI release. |

### Planned siblings

- `neuromem-langchain` / `neuromem-langgraph` — framework wrappers.
- `neuromem-pgvector` — Postgres/pgvector `StorageAdapter` implementation.
- `neuromem-fastembed` — local-embedding provider with zero-dep inference.

Each sibling declares `neuromem-core` as a workspace dependency in its own `pyproject.toml` and ships to PyPI under its own name.

---

## Quick start

**Install:**

```bash
uv sync --dev
```

**Run the full workspace test suite:**

```bash
uv run pytest --import-mode=importlib
```

`--import-mode=importlib` is required because several provider packages share common test filenames (`test_llm.py`) — importlib mode distinguishes them by fully-qualified module path. The default prepend mode collapses the two into a single module and crashes at collection time.

**Use the library (Gemini example):**

```python
from neuromem import NeuroMemory, SQLiteAdapter
from neuromem_gemini import GeminiLLMProvider, GeminiEmbeddingProvider

memory = NeuroMemory(
    storage=SQLiteAdapter("memory.db"),
    llm=GeminiLLMProvider(api_key="..."),
    embedder=GeminiEmbeddingProvider(api_key="..."),
)
memory.enqueue("I graduated with a Business Administration degree last year.")
# ... many more enqueues ...
memory.force_dream()  # trigger consolidation

from neuromem.context import ContextHelper
helper = ContextHelper(memory)
print(helper.build_prompt_context("what degree do I have?"))
```

**Run LongMemEval** against an agent of your choice:

```bash
uv run python packages/neuromem-bench/scripts/run_longmemeval.py \
    --split s --sample-size 20 --workers 4 \
    --agent neuromem --metric llm-judge \
    --model gemini-3-flash-preview
```

---

## Architecture decisions

The four merged ADRs describe the load-bearing design calls:

- [ADR-001 — clustering-library choice](docs/decisions/ADR-001-clustering-library-choice.md). Picked HDBSCAN over hand-rolled agglomerative.
- [ADR-002 — lazy centroid naming](docs/decisions/ADR-002-lazy-centroid-naming.md). Centroids get LLM-generated labels at render time, not write time.
- [ADR-003 — ontology tree v2](docs/decisions/ADR-003-ontology-tree-v2.md). HDBSCAN clustering + per-junction paragraph summaries + the `expand_node` agent tool.
- [ADR-004 — hot-path / ingest split](docs/decisions/ADR-004-hot-path-ingest-split.md). Retrieval makes zero LLM calls; every summary, tag, and label is pre-computed in the background dream cycle.

---

## Adding a new sibling package

The workspace makes this a three-step process. Detailed rules live in [Constitution §Additional Constraints — Repository Structure](.specify/memory/constitution.md).

1. **Scaffold** a new directory under `packages/` with the standard layout:

   ```
   packages/neuromem-<name>/
   ├── src/neuromem_<name>/
   │   └── __init__.py
   ├── tests/
   │   └── conftest.py
   ├── README.md
   └── pyproject.toml          # declares neuromem-core as a workspace dep
   ```

2. **Register** the package in the workspace-root `pyproject.toml` under `[tool.uv.workspace].members`. Run `uv sync --dev` to install it in editable mode.

3. **Ship** each user story as a spec-kit feature branch (`/speckit.specify` → `plan` → `tasks` → `implement`) respecting the Constitution's atomic-commit rule (≤500 lines per commit). Pre-commit hooks (`ruff format`, `ruff check`, `pytest`) run on every commit.

---

## Spec-kit workflow

Every feature moves through `specify → plan → tasks → implement`, with each phase producing a durable artifact under `specs/NNN-feature-name/`. Completed specs currently on `main`:

- [`specs/001-neuromem-core/`](specs/001-neuromem-core/) — initial `neuromem-core` v0.1.0.
- [`specs/003-neuromem-adk/`](specs/003-neuromem-adk/) — Google ADK integration.

Subsequent feature work (ADR-003, ADR-004, multi-provider, bench resilience, clustering quality) was shipped under the ADR / directly-merged-PR process rather than spec-kit — see [`docs/decisions/`](docs/decisions/) for the durable write-ups.

The local slash-command skills under `.claude/skills/` (invoked as `/speckit.specify`, `/speckit.plan`, etc.) automate the workflow for Claude Code sessions.

---

## Engineering rules

The binding rules for this project live in [`.specify/memory/constitution.md`](.specify/memory/constitution.md). Highlights:

- **Principle I — Framework-Agnostic Core.** `neuromem-core` never imports an LLM SDK or agent framework. Enforced by a CI tripwire test (`test_no_forbidden_imports.py`).
- **Principle II — Lean Dependency Set.** Runtime deps are `numpy + pandas + hdbscan` plus stdlib. Adding a new dep requires a constitutional amendment. Enforced by `test_pyproject_dependencies.py`.
- **Principle III — Layered, Modular, Pluggable.** Orchestration depends on abstract base classes; concrete implementations are injected. The `StorageAdapter` contract is exercised by a parametrised contract-test suite against every concrete backend.
- **Principle IV — Neuroscience-Grounded.** Subsystem names and behaviours map to the cognitive loop (hippocampus, neocortex, prefrontal cortex, synaptic pruning, LTP). Not marketing fluff — the tests assert the behavioural mapping.
- **Principle V — Test-First with Mocks.** NON-NEGOTIABLE. Every feature ships with tests that run end-to-end via deterministic mock providers.
- **Principle VI — Atomic Commits.** ≤500 lines per commit. Pre-commit hooks enforce every commit. **Never `--no-verify`.**

---

## CI

Two workflows:

- **CI** (`.github/workflows/ci.yml`) — deterministic unit suite across py 3.10/3.11/3.12/3.13, pre-commit hooks, coverage. Runs on every PR and push. **This is the merge gate.**
- **Integration Tests** (`.github/workflows/integration.yml`) — real Gemini API smoke test against `neuromem-gemini` and `neuromem-adk`. Runs on push-to-main, nightly, and manual dispatch. Requires the `GEMINI_API_KEY` repository secret. Not required for merge — an API hiccup doesn't block unrelated PRs.

---

## License

MIT. See [`LICENSE`](LICENSE).
