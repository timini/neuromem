# neuromem-openai

OpenAI provider pair (`OpenAILLMProvider` + `OpenAIEmbeddingProvider`) for
the [neuromem](https://github.com/timini/neuromem) cognitive memory
library. Implements the `LLMProvider` and `EmbeddingProvider` ABCs from
`neuromem.providers` against the official `openai` Python SDK.

## Install

```bash
pip install neuromem-openai
```

(workspace dev) — `uv sync` at the repo root picks it up automatically.

## Usage

```python
import os
from neuromem import NeuroMemory, SQLiteAdapter
from neuromem_openai import OpenAILLMProvider, OpenAIEmbeddingProvider

api_key = os.environ["OPENAI_API_KEY"]
memory = NeuroMemory(
    storage=SQLiteAdapter("memory.db"),
    llm=OpenAILLMProvider(api_key=api_key, model="gpt-4.1-mini"),
    embedder=OpenAIEmbeddingProvider(api_key=api_key, model="text-embedding-3-small"),
)
```

## Defaults

- LLM: `gpt-4.1-mini` — fast, inexpensive, JSON-reliable, fits the
  neuromem dream-cycle's batched-prompt pattern well.
- Embedder: `text-embedding-3-small` — 1536-dim. Swap to
  `text-embedding-3-large` (3072-dim) if you want higher precision; the
  core library locks embedding dim at first upsert so pick one per
  corpus.

## Prompts

Live under `src/neuromem_openai/prompts/`. Copied from the gemini
provider as a starting point; tune per-model if you find GPT responds
differently to some wording.
