# Contract: `LLMProvider` and `EmbeddingProvider` ABCs

**Feature**: 001-neuromem-core
**Module**: `neuromem.providers`
**Scope**: The external-service contract. The core calls these ABCs; framework wrapper packages (published separately) implement them for OpenAI, Anthropic, Google GenAI, local Ollama, etc.
**Binding**: Breaking these signatures is a MAJOR version bump per semver.

---

## `EmbeddingProvider`

Converts batches of strings into float vectors.

### ABC declaration

```python
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray

class EmbeddingProvider(ABC):
    """
    Contract for converting text into dense vector embeddings.
    Implementations wrap a vendor API (OpenAI, Cohere, Google, Voyage,
    local sentence-transformers, etc.) and are injected into NeuroMemory.
    """

    @abstractmethod
    def get_embeddings(
        self,
        texts: list[str],
    ) -> NDArray[np.floating]:
        ...
```

### `get_embeddings(texts) -> np.ndarray`

**Preconditions**:
- `texts` is a non-empty list of non-empty strings.

**Postconditions**:
- Returns a 2-D numpy array of shape `(len(texts), D)` where `D` is the provider's fixed embedding dimensionality.
- `result[i, :]` is the embedding of `texts[i]` — **row order is preserved**. This is load-bearing: the dreaming pipeline zips the result back against the input tags.
- `D` is the same across every call to the same provider instance for the lifetime of the storage backend. Changing dimensions mid-run is undefined behaviour (the `SQLiteAdapter` raises `ValueError` on the first mismatched insert per data-model.md V-N1).
- Dtype is `float32` or `float64`. The adapter normalises to `float32` for storage. Pure Python `list[list[float]]` returns are **also acceptable** — the core wraps them with `np.asarray(...)` — but numpy-native is the preferred return type because it avoids an unnecessary conversion.

**Errors**:
- Implementations MAY raise on network failure, rate limiting, auth errors, empty input, etc. The core catches these in `_run_dream_cycle` and rolls back the batch (FR-029). Callers of `enqueue()` see propagated exceptions from `generate_summary` only.
- Implementations SHOULD raise a descriptive exception (`RuntimeError`, `ValueError`, or a vendor-specific error class). The core does not inspect exception type — only presence.

**Thread safety**: The core calls `get_embeddings` **only from the dreaming thread** (`_run_dream_cycle`). It also calls it from `ContextHelper.build_prompt_context` and `search_memory`, which run on the caller's thread. Therefore implementations MUST be safe for concurrent calls from these two contexts. A simple rule: don't mutate shared state from inside `get_embeddings`; use connection pools or thread-local clients.

**Performance expectation**: The core batches aggressively — typically one call per dreaming cycle with 10–50 tags in the batch. Implementations should respect the vendor's batch size limit (e.g., OpenAI's 2048 per call) and internally split if needed.

---

## `LLMProvider`

Provides three LLM-backed transformations used by the cognitive loop.

### ABC declaration

```python
from abc import ABC, abstractmethod

class LLMProvider(ABC):
    """
    Contract for LLM-backed text transformations:
      - episodic summarisation (used in enqueue on the caller thread)
      - tag extraction (used in dreaming)
      - cluster category naming (used in dreaming)
    """

    @abstractmethod
    def generate_summary(self, raw_text: str) -> str:
        ...

    @abstractmethod
    def extract_tags(self, summary: str) -> list[str]:
        ...

    @abstractmethod
    def generate_category_name(self, concepts: list[str]) -> str:
        ...
```

---

### `generate_summary(raw_text) -> str`

**Preconditions**: `raw_text` is a non-empty string.

**Postconditions**:
- Returns a concise 1–2 sentence episodic summary of `raw_text`.
- Empty string is permitted but discouraged (it makes tag extraction useless).
- The returned string is stored verbatim in `memories.summary` and is what the dreaming cycle feeds into `extract_tags`.

**Call site**: `NeuroMemory.enqueue()` on the caller thread (Resolved Design Decision #3 — KISS). If the caller needs low latency, they inject a fast provider (e.g., a passthrough that returns the first 200 chars, or a local model).

**Errors**: Propagate any LLM/network error. `enqueue()` does not catch — the memory is NOT inserted and the caller sees the exception. This is intentional: if the summary is un-generatable, the memory has no searchable proxy and should not be stored half-formed.

**Thread safety**: Called from the caller's thread (usually the agent's main loop). No special requirements — called one at a time per `enqueue()` invocation.

---

### `extract_tags(summary) -> list[str]`

**Preconditions**: `summary` is a string (may be empty — implementations should return `[]` gracefully).

**Postconditions**:
- Returns a list of discrete concept labels extracted from the summary.
- Each label is a non-empty string. Labels MAY be multi-word (e.g., `"machine learning"`) — no single-word constraint here (that's only for `generate_category_name`).
- Recommended: 3–7 tags per call. More than 15 is unusual and may overwhelm clustering.
- Order is unspecified — the core deduplicates before embedding.

**Call site**: `NeuroMemory._run_dream_cycle()` in the dreaming thread. Called once per memory in the dreaming batch.

**Errors**: Propagate. The dreaming cycle catches at the outer level and rolls back the entire batch.

**Thread safety**: Called only from the dreaming thread. No concurrency within a single `NeuroMemory` instance.

**Implementation hints for framework wrappers**:
- OpenAI/Anthropic structured output: use a JSON schema `{"type": "array", "items": {"type": "string"}}`.
- Local models: simple prompt template `"List the key concepts in this text as a comma-separated list: {text}"` + split on commas.
- Guard against duplicates and extremely long outputs.

---

### `generate_category_name(concepts) -> str`

**Preconditions**: `concepts` is a non-empty list of non-empty strings (the member labels of a cluster).

**Postconditions**:
- Returns a single **one-word** category name that generalises the input concepts.
- Example: `["SQLite", "Neo4j", "Postgres"]` → `"Databases"`.
- If the LLM returns multiple words or a sentence, the core takes the first word and logs a `WARNING`. This is defensive — the contract still says "one word", and implementations SHOULD comply.
- The returned string is stored as the `label` of a centroid node (`is_centroid=True`).

**Call site**: `NeuroMemory._run_dream_cycle()` in the dreaming thread. Called once per cluster formed during agglomerative merging — typically 0–5 times per dreaming cycle.

**Errors**: Propagate. Dreaming cycle catches and rolls back.

**Implementation hints for framework wrappers**:
- Prompt template: `"Give a single one-word category name that encompasses these concepts: {concepts}. Reply with just the category word, no explanation."`
- Structured output: `{"type": "object", "properties": {"category": {"type": "string"}}}` and extract `.category`.
- Cheap model is fine — the user's source conversation discussed this, and Resolved Design Decision #7 explicitly leaves the "second cheap provider" slot for a future minor release.

---

## What implementations must NOT do

- **Don't import from `neuromem.system`** — that's the orchestration layer. Providers live in (or below) the contract layer.
- **Don't assume the core will retry** — the core catches and rolls back the batch, then the next `enqueue()`-triggered cycle retries. Implementations SHOULD NOT add their own exponential-backoff retry loop; that's a layering violation and makes errors opaque.
- **Don't cache embeddings internally** — that's the storage adapter's job (via `upsert_node`). An `EmbeddingProvider` that caches its own results will drift from the stored embeddings over time and corrupt similarity math.
- **Don't mutate input arguments** — `texts`, `concepts`, etc. are read-only to the provider.
- **Don't return `None`** where the contract specifies a string or list. Either return a valid value or raise.

---

## Mock providers (tests)

Principle V requires `MockLLMProvider` and `MockEmbeddingProvider` to be usable in the test suite without network access. Both live in `tests/conftest.py` as pytest fixtures. They are NOT shipped in `src/neuromem/` — they are test infrastructure only.

### `MockEmbeddingProvider`

Deterministic hash-seeded embedding: for any input string, produce a fixed-dimension vector derived from a stable hash.

```python
import hashlib
import numpy as np
from neuromem.providers import EmbeddingProvider

class MockEmbeddingProvider(EmbeddingProvider):
    def __init__(self, dim: int = 16) -> None:
        self.dim = dim

    def get_embeddings(self, texts: list[str]) -> np.ndarray:
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            seed = int.from_bytes(hashlib.md5(text.encode()).digest()[:8], "big")
            rng = np.random.default_rng(seed)
            out[i] = rng.uniform(-1.0, 1.0, size=self.dim).astype(np.float32)
        return out
```

### `MockLLMProvider`

Canned responses derived deterministically from inputs:

```python
class MockLLMProvider(LLMProvider):
    def generate_summary(self, raw_text: str) -> str:
        return raw_text[:80]  # first 80 chars as "summary"

    def extract_tags(self, summary: str) -> list[str]:
        # Split on whitespace, keep first 3 non-empty tokens
        return [w for w in summary.split() if w.isalpha()][:3]

    def generate_category_name(self, concepts: list[str]) -> str:
        # Deterministic: first letter of each concept joined
        return "Cat" + "".join(c[0].upper() for c in concepts)
```

These mocks are intentionally trivial. The real test of the clustering pipeline is that it calls the ABC methods in the right order with the right arguments, which the mocks can verify via attribute capture.
