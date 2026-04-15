"""Prompt templates for neuromem-gemma.

Starting point copied from neuromem-gemini; tune per-model here if
Gemma responds differently to certain wording.

Prefer :func:`render_prompt` over ``load_prompt`` + ``.format()`` —
it auto-sanitises ``\\n#`` prompt-injection markers. In practice
``GemmaLLMProvider`` inherits from ``OpenAILLMProvider`` and routes
through ``neuromem_openai.prompts.render_prompt``; this module only
matters if a caller imports the templates directly for custom use.
"""

from __future__ import annotations

from functools import cache
from importlib import resources
from typing import Any

_HEADER_SHADOW_MARKER = "\n#"
_HEADER_SHADOW_REPLACEMENT = "\n "


def _sanitise_value(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace(_HEADER_SHADOW_MARKER, _HEADER_SHADOW_REPLACEMENT)
    if isinstance(value, list | tuple):
        return type(value)(
            (
                v.replace(_HEADER_SHADOW_MARKER, _HEADER_SHADOW_REPLACEMENT)
                if isinstance(v, str)
                else v
            )
            for v in value
        )
    return value


@cache
def load_prompt(name: str) -> str:
    """Raw ``<name>.prompt`` contents. Prefer :func:`render_prompt`."""
    return (resources.files(__name__) / f"{name}.prompt").read_text(encoding="utf-8")


def render_prompt(name: str, **kwargs: Any) -> str:
    """Load ``<name>.prompt`` and render with sanitised kwargs."""
    return load_prompt(name).format(**{k: _sanitise_value(v) for k, v in kwargs.items()})
