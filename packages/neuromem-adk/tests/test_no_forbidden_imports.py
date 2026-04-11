"""Forbidden-imports tripwire for neuromem-adk's source tree.

Similar to neuromem-core's `test_no_forbidden_imports.py` but
scoped to THIS package's `src/neuromem_adk/` tree and with a
different forbidden list:

- Forbidden: every non-Google vendor LLM / agent framework SDK.
  The whole point of neuromem-adk is that it's THE Google ADK
  integration package — so `google.adk` and `google.genai` are
  explicitly allowed. Other vendor SDKs (openai, anthropic,
  langchain, llama_index, cohere, voyageai, langgraph,
  anthropic_ai_agent) are forbidden because they'd cross-
  contaminate the package and make it do two things badly.

This enforces the "one package per vendor" principle inside the
sibling packages — if a user wants Anthropic support, they'll
install `neuromem-anthropic` (future), not have it accidentally
dragged in by `neuromem-adk`.

Uses the same AST-based detector pattern as
`packages/neuromem-core/tests/test_no_forbidden_imports.py`.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

FORBIDDEN_PACKAGES: frozenset[str] = frozenset(
    {
        "openai",
        "anthropic",
        "cohere",
        "voyageai",
        "langchain",
        "langgraph",
        "llama_index",
        "anthropic_ai_agent",
        # NOTE: google.adk and google.genai are DELIBERATELY absent.
        # This package IS the ADK wrapper and directly depends on
        # google-adk + google-genai via neuromem-gemini.
    }
)

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "neuromem_adk"


def _iter_py_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.py"))


def _imports_in(path: Path) -> list[tuple[str, int]]:
    """Return ``(imported_name, lineno)`` for every import in ``path``.

    Covers `import foo`, `import foo.bar`, `from foo.bar import baz`,
    and emits both the base module AND every `module.alias`
    combination from `ImportFrom` nodes so the shape
    `from google import adk` is caught the same way it is in
    neuromem-core's tripwire. Relative imports are skipped.
    """
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:  # pragma: no cover
        raise RuntimeError(f"failed to parse {path}: {exc}") from exc

    results: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                results.append((alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0 or node.module is None:
                continue
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


def test_src_tree_exists() -> None:
    """Sanity: the source tree we're scanning must actually exist.

    Guards against a silent pass caused by the path being wrong —
    an empty `rglob` would trivially satisfy the forbidden-imports
    test on zero files.
    """
    assert _SRC_ROOT.is_dir(), f"neuromem-adk src tree not found at {_SRC_ROOT}"
    py_files = _iter_py_files(_SRC_ROOT)
    assert len(py_files) > 0, f"no .py files found under {_SRC_ROOT}"


def test_no_forbidden_imports_in_neuromem_adk_src() -> None:
    """The sibling package must not import any non-Google vendor
    SDK. google.adk and google.genai are explicitly allowed
    (they're what this package wraps). Every other LLM / framework
    vendor is forbidden — if users want those integrations they
    install a different sibling package.
    """
    violations: list[str] = []
    for py_file in _iter_py_files(_SRC_ROOT):
        for module_name, lineno in _imports_in(py_file):
            hit = _is_forbidden(module_name)
            if hit is not None:
                rel = py_file.relative_to(_SRC_ROOT.parents[1])
                violations.append(f"{rel}:{lineno} imports forbidden package {hit!r}")

    if violations:
        pytest.fail(
            "neuromem-adk imports forbidden vendor packages (non-Google):\n  "
            + "\n  ".join(violations)
            + "\n\nThe 'one package per vendor' principle means this "
            "sibling should only depend on google-adk, google-genai "
            "(via neuromem-gemini), and neuromem-core. Other vendors "
            "belong in their own sibling packages."
        )
