"""Prompt templates for neuromem-gemma.

Starting point copied from neuromem-gemini; tune per-model here if
Gemma responds differently to certain wording.
"""

from __future__ import annotations

from functools import cache
from importlib import resources


@cache
def load_prompt(name: str) -> str:
    return (resources.files(__name__) / f"{name}.prompt").read_text(encoding="utf-8")
