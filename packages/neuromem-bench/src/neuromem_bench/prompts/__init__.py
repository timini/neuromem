"""Prompt templates for neuromem-bench.

Plain-text templates with ``{placeholder}`` markers resolved via
:meth:`str.format` at call time. Same pattern as
``neuromem_gemini.prompts``.
"""

from __future__ import annotations

from functools import cache
from importlib import resources


@cache
def load_prompt(name: str) -> str:
    return (resources.files(__name__) / f"{name}.prompt").read_text(encoding="utf-8")
