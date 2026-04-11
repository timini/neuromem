"""T027 — SC-005 enforcement: no LLM / agent framework imports in the core.

Constitution v2.0.0 Principle I (Framework-Agnostic Core) requires
``neuromem-core`` to stay importable in any agent framework without
forcing a transitive dependency on OpenAI, Anthropic, Google ADK,
LangChain, etc. The published wheel MUST NOT import any of these
packages — not even lazily — because the moment it does, ``uv add
neuromem-core`` pulls their entire dependency tree into every
downstream agent.

This test file-walks ``packages/neuromem-core/src/neuromem/`` and
fails CI if any forbidden top-level package appears in an ``import``
statement. Running at CI time (not just at review time) catches
accidental regressions the second they land, which is the whole
point of SC-005.

Detection strategy:
    Parse every .py file with ``ast`` and walk Import / ImportFrom
    nodes. This is dramatically more robust than a regex grep —
    it's immune to commented-out imports, string literals that
    happen to contain the words ``openai``, multi-line import
    statements, and conditional imports under
    ``if TYPE_CHECKING:``. If the import survives byte-compilation,
    ast.parse sees it; if it doesn't, it's not a real import and
    doesn't count.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

# Top-level package names of every LLM / embedder / agent framework we
# explicitly refuse to depend on from the core. Matches the SC-005
# checklist in spec.md.
#
# Package names are normalised to their Python import form (dots, not
# hyphens): ``google.genai`` not ``google-genai``; ``llama_index`` not
# ``llama-index``. The comparison is done on the leading component only,
# so ``google.genai.foo`` is caught by the ``google.genai`` entry.
FORBIDDEN_PACKAGES: frozenset[str] = frozenset(
    {
        "openai",
        "anthropic",
        "google.genai",
        "google_genai",
        "cohere",
        "voyageai",
        "langchain",
        "langgraph",
        "llama_index",
        "google.adk",
        "anthropic_ai_agent",
    }
)

# Absolute path of the core source tree. Tests run from the workspace
# root, so we anchor on this file's location rather than on cwd.
_CORE_SRC = Path(__file__).resolve().parents[1] / "src" / "neuromem"


def _iter_python_files(root: Path) -> list[Path]:
    """Return every ``.py`` file under ``root``, sorted for stable output."""
    return sorted(root.rglob("*.py"))


def _imports_in(path: Path) -> list[tuple[str, int]]:
    """Return ``(imported_name, lineno)`` for every import in ``path``.

    - ``import foo`` → yields ``("foo", lineno)`` (one per alias).
    - ``import foo.bar`` → yields ``("foo.bar", lineno)``.
    - ``from foo.bar import baz`` → yields ``("foo.bar", lineno)`` AND
      ``("foo.bar.baz", lineno)``. Emitting the fully-qualified
      ``module.name`` form is essential — otherwise a forbidden
      sub-package like ``google.adk`` slips past the check when the
      import is written as ``from google import adk`` (the ``module``
      field in that AST node is just ``"google"``, which is not on the
      forbidden list).
    - Relative imports (``from . import X``) are ignored — they can
      never target a forbidden package by definition.
    """
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:  # pragma: no cover — parse failure is a real bug
        raise RuntimeError(f"failed to parse {path}: {exc}") from exc

    results: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                results.append((alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            # Relative imports have level > 0 and module may be None.
            if node.level > 0 or node.module is None:
                continue
            # Emit the base module (catches `from google.adk import X`)
            # AND every `module.alias` combination (catches the
            # `from google import adk` form). Both shapes must be on
            # the list so `_is_forbidden` can match against either.
            results.append((node.module, node.lineno))
            for alias in node.names:
                results.append((f"{node.module}.{alias.name}", node.lineno))
    return results


def _is_forbidden(module_name: str) -> str | None:
    """Return the matching forbidden package name, or None if allowed."""
    for forbidden in FORBIDDEN_PACKAGES:
        if module_name == forbidden or module_name.startswith(forbidden + "."):
            return forbidden
    return None


def test_core_src_tree_exists() -> None:
    """Sanity: the source tree we're scanning must actually exist.

    This guards against a silent pass caused by the path being wrong —
    if ``_CORE_SRC`` points at a non-existent directory, ``rglob`` would
    return an empty list and the forbidden-imports test would trivially
    pass on zero files. Asserting the tree exists and has .py files
    makes that failure mode visible.
    """
    assert _CORE_SRC.is_dir(), f"core source tree not found at {_CORE_SRC}"
    py_files = _iter_python_files(_CORE_SRC)
    assert len(py_files) > 0, f"no .py files found under {_CORE_SRC}"


def test_no_forbidden_imports_in_core() -> None:
    """SC-005 + Constitution Principle I: no LLM / framework imports.

    Walks every .py file under ``packages/neuromem-core/src/neuromem/``
    and fails if any ``import`` or ``from ... import`` statement
    references a forbidden package.
    """
    violations: list[str] = []
    for py_file in _iter_python_files(_CORE_SRC):
        for module_name, lineno in _imports_in(py_file):
            hit = _is_forbidden(module_name)
            if hit is not None:
                rel = py_file.relative_to(_CORE_SRC.parents[1])
                violations.append(f"{rel}:{lineno} imports forbidden package {hit!r}")

    if violations:
        pytest.fail(
            "Constitution Principle I / SC-005 violated — core imports "
            "forbidden packages:\n  "
            + "\n  ".join(violations)
            + "\n\nThe core library must NEVER depend on LLM or agent-framework "
            "packages. See .specify/memory/constitution.md §Principle I."
        )


# ---------------------------------------------------------------------------
# Detector self-tests — feed the walker synthetic code samples and verify
# every forbidden-import shape is caught. These are the regression guards
# for the AST walker itself.
# ---------------------------------------------------------------------------


class TestDetectorCatchesEveryShape:
    """Lock in that ``_imports_in`` + ``_is_forbidden`` catch every
    syntactic form of a forbidden import, including edge cases like
    ``from google import adk`` where the AST's ``module`` field only
    holds the parent package name.
    """

    @staticmethod
    def _scan_source(source: str, tmp_path: Path) -> list[str]:
        """Write ``source`` to a temp .py file and return the list of
        forbidden package names the walker detects in it."""
        tmp_file = tmp_path / "sample.py"
        tmp_file.write_text(source)
        return [hit for name, _ in _imports_in(tmp_file) if (hit := _is_forbidden(name))]

    def test_catches_bare_import(self, tmp_path: Path) -> None:
        assert self._scan_source("import openai", tmp_path) == ["openai"]

    def test_catches_dotted_import(self, tmp_path: Path) -> None:
        hits = self._scan_source("import google.adk", tmp_path)
        assert "google.adk" in hits

    def test_catches_dotted_deeper_import(self, tmp_path: Path) -> None:
        hits = self._scan_source("import google.adk.agents", tmp_path)
        assert "google.adk" in hits

    def test_catches_from_dotted_import(self, tmp_path: Path) -> None:
        hits = self._scan_source("from google.adk import Agent", tmp_path)
        assert "google.adk" in hits

    def test_catches_from_google_import_adk(self, tmp_path: Path) -> None:
        """Regression: the previous version of `_imports_in` missed
        ``from google import adk`` because the AST ``module`` field
        was just ``"google"``, which is not on the forbidden list.
        The fix emits ``module.alias`` combinations to cover this shape.
        """
        hits = self._scan_source("from google import adk", tmp_path)
        assert "google.adk" in hits

    def test_catches_multi_alias_import(self, tmp_path: Path) -> None:
        hits = self._scan_source("import os, openai, sys", tmp_path)
        assert "openai" in hits

    def test_catches_import_inside_function(self, tmp_path: Path) -> None:
        src = "def f():\n    import anthropic\n    return anthropic.Client()"
        hits = self._scan_source(src, tmp_path)
        assert "anthropic" in hits

    def test_catches_import_inside_if_type_checking(self, tmp_path: Path) -> None:
        src = "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import langchain\n"
        hits = self._scan_source(src, tmp_path)
        assert "langchain" in hits

    def test_relative_imports_ignored(self, tmp_path: Path) -> None:
        """``from . import foo`` can never target a forbidden package."""
        src = "from . import openai_helper"
        hits = self._scan_source(src, tmp_path)
        assert hits == []

    def test_non_forbidden_imports_pass(self, tmp_path: Path) -> None:
        """Sanity: the detector doesn't false-positive on allowed deps."""
        src = "import numpy as np\nimport pandas as pd\nfrom pathlib import Path"
        hits = self._scan_source(src, tmp_path)
        assert hits == []
