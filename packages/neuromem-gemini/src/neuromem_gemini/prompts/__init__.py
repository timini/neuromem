"""Prompt templates for neuromem-gemini.

Each ``.prompt`` file in this directory is a plain-text template with
``{placeholder}`` markers resolved via :meth:`str.format` at call time.
Keeping prompts out of Python source makes them editable without
hunting through `llm.py` and keeps the call-site focused on logic.

Usage::

    from neuromem_gemini.prompts import load_prompt
    prompt = load_prompt("generate_summary").format(raw_text=raw_text)
"""

from __future__ import annotations

from functools import cache
from importlib import resources


@cache
def load_prompt(name: str) -> str:
    """Return the contents of ``<name>.prompt`` from this package.

    The LRU cache means repeated calls (one per LLM API call on the
    hot path) don't re-open the file. File I/O is still triggered
    exactly once per process per template.
    """
    return (resources.files(__name__) / f"{name}.prompt").read_text(encoding="utf-8")
