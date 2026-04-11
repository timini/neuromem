# neuromem-gemini

> Google Gemini provider pair for [`neuromem-core`](../neuromem-core/). Wraps `gemini-2.0-flash-001` for summary/tag/category-name calls and `gemini-embedding-001` for embeddings.

This is the first sibling package in the `neuromem` monorepo. It implements the `LLMProvider` and `EmbeddingProvider` abstract base classes from `neuromem-core` and hands `NeuroMemory` a drop-in way to run its cognitive loop against real Google Gemini models.

**Status**: v0.1.0 alpha.

---

## Install

```bash
uv add neuromem-gemini
```

This pulls in `neuromem-core`, `google-genai`, and `numpy` as transitive runtime deps.

---

## Usage

```python
import os
from neuromem import NeuroMemory, ContextHelper, SQLiteAdapter
from neuromem_gemini import GeminiLLMProvider, GeminiEmbeddingProvider

api_key = os.environ["GEMINI_API_KEY"]

memory = NeuroMemory(
    storage=SQLiteAdapter("memory.db"),
    llm=GeminiLLMProvider(api_key=api_key),
    embedder=GeminiEmbeddingProvider(api_key=api_key),
)

memory.enqueue("The user is debugging a WAL-mode SQLite locking issue.")
memory.enqueue("Agent suggested PRAGMA synchronous=NORMAL as a mitigation.")
memory.force_dream(block=True)

helper = ContextHelper(memory)
context_block = helper.build_prompt_context("sqlite performance tuning")
```

Default models:

- **LLM**: `gemini-2.0-flash-001` — fast, cheap, current. Override via `GeminiLLMProvider(api_key, model="gemini-2.5-flash")`.
- **Embedder**: `gemini-embedding-001` — 3072-dimensional. Override via `GeminiEmbeddingProvider(api_key, model="gemini-embedding-2-preview")` for the preview line.

The embedder dimension is fixed at the first call and cannot change over the lifetime of a `NeuroMemory` instance — this is a `neuromem-core` invariant, not a package-level constraint.

---

## Integration tests

This package ships with a small end-to-end suite that hits the real Gemini API. It's gated behind a pytest marker so it never runs during the default pre-commit / CI flow:

```bash
# Default runs skip the integration suite:
uv run pytest                           # 208 core tests pass, gemini suite skipped

# Opt-in to run the real API tests (pass --no-cov because the
# workspace-root coverage gate is scoped to neuromem-core — the
# sibling-package integration tests don't exercise those modules
# and would otherwise fail the 90% threshold spuriously):
export GEMINI_API_KEY=your-key-here
uv run pytest packages/neuromem-gemini/tests/ -m integration -v --no-cov
```

The integration suite is intentionally small (3 tests) and focused on proving:

1. **Tag quality** — real Gemini tag extraction produces meaningful concepts, not stopwords.
2. **End-to-end recall** — `ContextHelper.build_prompt_context` and `search_memory` return non-empty trees after a real dream cycle.
3. **LTP spike** — `retrieve_memories` correctly spikes `access_weight=1.0` on consolidated rows.

Cost per run: roughly 5000 tokens, <$0.01 at current Gemini 2.0 Flash rates.

---

## Why this is a separate package

Constitution v2.0.0 Principle I (Framework-Agnostic Core) forbids `neuromem-core` from importing any LLM SDK. Principle II locks its runtime dependencies to `numpy + pandas` only. Both principles are enforced by CI tripwire tests inside `neuromem-core` — a `google.genai` import in `packages/neuromem-core/src/neuromem/` would fail the build.

Vendor-specific providers therefore live in sibling packages like this one, declared as workspace dependencies of `neuromem-core`. The monorepo layout (`packages/neuromem-core/`, `packages/neuromem-gemini/`, future `packages/neuromem-anthropic/`, etc.) makes every provider independently versionable, publishable, and swappable without touching the core.

---

## License

MIT. See [LICENSE](../../LICENSE) at the repo root.
