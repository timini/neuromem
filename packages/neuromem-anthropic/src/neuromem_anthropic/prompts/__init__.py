"""Prompt templates for neuromem-anthropic.

Prefer :func:`render_prompt` over ``load_prompt`` + ``.format()`` —
it auto-sanitises ``\\n#`` prompt-injection markers in every string
argument before format-substitution.
"""

from __future__ import annotations

from functools import cache
from importlib import resources
from typing import Any

_HEADER_SHADOW_MARKER = "\n#"
_HEADER_SHADOW_REPLACEMENT = "\n "


def _sanitise_value(value: Any) -> Any:
    """Neutralise ``\\n#`` markers in string / list-of-string values."""
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
    """Load ``<name>.prompt`` and render with sanitised kwargs.

    String values have ``\\n#`` sequences neutralised so a malicious
    memory can't shadow the template's structural ``## Section``
    headers. Non-string kwargs pass through unchanged.
    """
    return load_prompt(name).format(**{k: _sanitise_value(v) for k, v in kwargs.items()})
