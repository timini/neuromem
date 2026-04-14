# neuromem-anthropic

Anthropic (Claude) LLM provider for [neuromem](https://github.com/timini/neuromem).
Implements the `LLMProvider` ABC from `neuromem.providers` against the
official `anthropic` Python SDK.

**LLM only.** Anthropic has no native embedding API. Pair with the
embedding half of your choice — the simplest is `neuromem-openai`'s
`OpenAIEmbeddingProvider` (text-embedding-3-*) or the upcoming
`neuromem-voyage` wrapper.

## Install

```bash
pip install neuromem-anthropic neuromem-openai
```

## Usage

```python
import os
from neuromem import NeuroMemory, SQLiteAdapter
from neuromem_anthropic import AnthropicLLMProvider
from neuromem_openai import OpenAIEmbeddingProvider

memory = NeuroMemory(
    storage=SQLiteAdapter("memory.db"),
    llm=AnthropicLLMProvider(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model="claude-sonnet-4-6",
    ),
    embedder=OpenAIEmbeddingProvider(
        api_key=os.environ["OPENAI_API_KEY"],
        model="text-embedding-3-small",
    ),
)
```

## Defaults

LLM: `claude-sonnet-4-6` (good quality at modest cost). Other solid
choices: `claude-opus-4-6` (top quality), `claude-haiku-4-5` (fast +
cheap).

## Prompts

`src/neuromem_anthropic/prompts/` — copied from the gemini provider
as a starting point. Claude responds very reliably to explicit "return
only JSON" instructions so we don't need a `response_format` mode like
OpenAI's.
