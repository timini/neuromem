"""neuromem-anthropic — Anthropic (Claude) LLM provider for neuromem.

Implements the ``LLMProvider`` ABC against the ``anthropic`` SDK.
LLM only; pair with an embedding provider from a sibling package
(``neuromem-openai`` is the simplest).
"""

from __future__ import annotations

from neuromem_anthropic.llm import AnthropicLLMProvider

__version__ = "0.1.0"

__all__ = [
    "AnthropicLLMProvider",
    "__version__",
]
