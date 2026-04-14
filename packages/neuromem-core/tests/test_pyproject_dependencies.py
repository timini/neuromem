"""T028 — SC-006 enforcement: ``neuromem-core`` runtime dependencies are frozen.

Constitution v2.0.0 Principle II (Lean Dependency Set) pins the core
library's runtime dependency list. Reading ADR-003, "lean" is treated
as "minimal, not zero" — the current locked set is numpy + pandas +
hdbscan, where hdbscan is justified as the clustering algorithm per
ADR-003 D1. Any *further* addition — even a "small" one like tenacity,
orjson, or structlog — needs a documented ADR first.

This test parses ``packages/neuromem-core/pyproject.toml`` with
stdlib ``tomllib`` and asserts ``[project].dependencies`` is
*set-equal* to the locked-in list. Set-equal (not list-equal) so a
cosmetic reordering of the list in ``pyproject.toml`` doesn't break
CI; only a real change to the dependency set does.

The stdlib ``tomllib`` module is Python 3.11+; the package's
``requires-python`` is ``>=3.10``, so older interpreters fall back to
the ``tomli`` package. Because ``tomli`` is NOT a neuromem-core
runtime dependency, we keep this test 3.11+-only by deferring the
import; on 3.10 it's skipped with a clear reason rather than
pretending to pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Absolute path to the package manifest under test. Anchored on this
# file's location so the test is insensitive to cwd.
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_PYPROJECT = _PACKAGE_ROOT / "pyproject.toml"

# The exact runtime dependency set Constitution v2.0.0 Principle II
# allows the core library to ship. Any deviation must go through a
# constitutional amendment — this test is the CI-time tripwire.
#
# Comparison is done via equality on a frozenset of the raw strings,
# so version pins are part of the lock: changing ``numpy>=1.26`` to
# ``numpy>=2.0`` is a deliberate decision that must update this list.
LOCKED_RUNTIME_DEPS: frozenset[str] = frozenset(
    {
        "numpy>=1.26",
        "pandas>=2.1",
        # ADR-003 D1: HDBSCAN for non-binary clustering in the dream cycle.
        "hdbscan>=0.8.33",
    }
)


def test_pyproject_exists() -> None:
    """Sanity: the file we're parsing must actually exist."""
    assert _PYPROJECT.is_file(), f"pyproject.toml not found at {_PYPROJECT}"


def test_runtime_dependencies_are_locked() -> None:
    """SC-006 + Constitution Principle II: runtime deps are numpy + pandas only.

    Parses ``[project].dependencies`` from
    ``packages/neuromem-core/pyproject.toml`` and asserts it is
    set-equal to ``LOCKED_RUNTIME_DEPS``. Any addition, removal, or
    version-pin change fails the test and forces the author to either
    revert or amend the constitution.
    """
    if sys.version_info < (3, 11):
        pytest.skip("tomllib requires Python 3.11+")

    import tomllib  # noqa: PLC0415 — gated on Python version

    with _PYPROJECT.open("rb") as fh:
        data = tomllib.load(fh)

    deps = data.get("project", {}).get("dependencies")
    assert deps is not None, "[project].dependencies is missing from pyproject.toml"
    assert isinstance(deps, list), f"[project].dependencies must be a list, got {type(deps)}"

    actual = frozenset(deps)

    unexpected = actual - LOCKED_RUNTIME_DEPS
    missing = LOCKED_RUNTIME_DEPS - actual

    messages: list[str] = []
    if unexpected:
        messages.append(f"unexpected deps (forbidden by Principle II): {sorted(unexpected)}")
    if missing:
        messages.append(f"missing required deps: {sorted(missing)}")

    if messages:
        pytest.fail(
            "Constitution Principle II / SC-006 violated — runtime "
            "dependency list has drifted from the locked set:\n  "
            + "\n  ".join(messages)
            + f"\n\nExpected: {sorted(LOCKED_RUNTIME_DEPS)}"
            + f"\nActual:   {sorted(actual)}"
            + "\n\nAdding a runtime dependency requires a constitutional "
            "amendment. See .specify/memory/constitution.md §Principle II."
        )
