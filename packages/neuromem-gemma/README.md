# neuromem-gemma

Local Gemma provider pair for [neuromem](https://github.com/timini/neuromem),
served through Ollama's OpenAI-compatible endpoint. Zero cost per token,
fully offline, no API keys needed.

## Setup

Install Ollama (`brew install ollama` on macOS / see
<https://ollama.com> for other platforms), then pull the models:

```bash
ollama pull gemma3                  # LLM (~3B-27B depending on tag)
ollama pull embeddinggemma          # embedder (308M, 768-dim)
```

Leave `ollama serve` running in the background.

## Install

```bash
pip install neuromem-gemma
```

## Usage

```python
from neuromem import NeuroMemory, SQLiteAdapter
from neuromem_gemma import GemmaLLMProvider, GemmaEmbeddingProvider

memory = NeuroMemory(
    storage=SQLiteAdapter("memory.db"),
    llm=GemmaLLMProvider(),                    # defaults to gemma3
    embedder=GemmaEmbeddingProvider(),         # defaults to embeddinggemma
)
```

Point at a non-localhost Ollama via the ``base_url`` kwarg:

```python
llm = GemmaLLMProvider(base_url="http://gpu-box.local:11434/v1")
```

## Why this package

`neuromem-openai` already drives any OpenAI-compatible endpoint via
``base_url``. This package is a thin convenience layer that:

- Points ``base_url`` at ``http://localhost:11434/v1`` by default.
- Uses a dummy ``OPENAI_API_KEY=ollama`` since Ollama ignores it but
  the openai SDK requires something non-empty.
- Ships Gemma-sensible default model names.

No custom logic — all LLM methods come from `OpenAILLMProvider` via
inheritance. Per-method prompt overrides live in
`src/neuromem_gemma/prompts/` if you want to tune for Gemma's
response style (starting point: copied from the gemini provider).
