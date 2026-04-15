"""Prompt templates for neuromem-gemini.

Each ``.prompt`` file in this directory is a plain-text template with
``{placeholder}`` markers resolved via :meth:`str.format` at call time.
Keeping prompts out of Python source makes them editable without
hunting through `llm.py` and keeps the call-site focused on logic.

Usage — prefer :func:`render_prompt` over the raw ``load_prompt`` +
``.format()`` pattern because it applies prompt-injection sanitisation
(neutralises ``\\n#`` header-shadow attacks) to every string argument
automatically::

    from neuromem_gemini.prompts import render_prompt
    prompt = render_prompt("generate_summary", raw_text=raw_text)
"""

from __future__ import annotations

from functools import cache
from importlib import resources
from typing import Any

# Every prompt template in this directory uses ``\n##`` style Markdown
# headers as structural markers ("## Rules", "## Output", etc.). A user
# memory whose text contains a ``\n## Output format`` sequence would
# shadow our legitimate section and, since ``generate_content`` has no
# system-vs-user separation, the injected section can override our
# instructions. Substituting ``\n#`` → ``\n `` preserves readability
# and removes the attack surface with zero false positives on natural
# text.
_HEADER_SHADOW_MARKER = "\n#"
_HEADER_SHADOW_REPLACEMENT = "\n "


def _sanitise_value(value: Any) -> Any:
    """Neutralise ``\\n#`` prompt-injection markers in string values.

    - ``str`` → sanitised copy.
    - ``list`` / ``tuple`` of strings → element-wise sanitised copy
      (other element types pass through untouched — lists of ints,
      dicts, etc. aren't expected on any current prompt path).
    - Anything else → untouched.

    The sanitiser is intentionally shallow: deep nested structures
    aren't a prompt path today; adding recursion only if/when that
    changes keeps the behaviour obvious.
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
    """Return the contents of ``<name>.prompt`` from this package.

    The LRU cache means repeated calls (one per LLM API call on the
    hot path) don't re-open the file. File I/O is still triggered
    exactly once per process per template.

    Prefer :func:`render_prompt` — it combines ``load_prompt`` with
    automatic sanitisation of string kwargs, which is the correct
    default for any path that splices user-supplied text into the
    template.
    """
    return (resources.files(__name__) / f"{name}.prompt").read_text(encoding="utf-8")


def render_prompt(name: str, **kwargs: Any) -> str:
    """Load ``<name>.prompt`` and render it with sanitised ``kwargs``.

    All string (and list-of-string) values have ``\\n#`` sequences
    neutralised before ``.format()`` substitution. This is the single
    choke point the LLM-call methods in ``llm.py`` go through — if
    a new prompt-facing call-site uses ``render_prompt`` instead of
    ``load_prompt(...).format(...)``, prompt-injection hardening is
    automatic.

    Non-string kwargs (``n=10``, etc.) pass through untouched.
    """
    return load_prompt(name).format(**{k: _sanitise_value(v) for k, v in kwargs.items()})
