"""Prompt templates for neuromem-openai.

Plain-text templates with ``{placeholder}`` markers resolved via
:meth:`str.format` at call time. Same pattern as
``neuromem_gemini.prompts``.

Prefer :func:`render_prompt` over ``load_prompt`` + ``.format()`` — it
auto-sanitises ``\\n#`` prompt-injection markers in every string
argument.
"""

from __future__ import annotations

from functools import cache
from importlib import resources
from typing import Any

# Substitute ``\n#`` → ``\n `` so a user memory can't shadow the
# structural ``## Section`` headers every prompt template uses.
_HEADER_SHADOW_MARKER = "\n#"
_HEADER_SHADOW_REPLACEMENT = "\n "


def _sanitise_value(value: Any) -> Any:
    """Neutralise ``\\n#`` markers in string / list-of-string values.

    Non-string values pass through untouched (the ``n=`` kwarg the
    batched templates take, for instance, is an int and needs no
    sanitisation).
    """
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
    """Return the raw ``<name>.prompt`` contents. Prefer
    :func:`render_prompt` for anything that splices user text."""
    return (resources.files(__name__) / f"{name}.prompt").read_text(encoding="utf-8")


def render_prompt(name: str, **kwargs: Any) -> str:
    """Load ``<name>.prompt`` and render with sanitised kwargs.

    String values have ``\\n#`` neutralised before ``.format()``
    substitution so a malicious or accidentally-adversarial user
    memory can't shadow our structural Markdown headers.
    """
    return load_prompt(name).format(**{k: _sanitise_value(v) for k, v in kwargs.items()})
