"""Shared fixtures for the neuromem-gemini integration test suite.

This is an integration-only suite. Every test file in this directory
requires a live Gemini API key and network access. If the key cannot
be found, ``gemini_api_key`` skips the test — we never want the
absence of a key to be a hard failure, because this package's tests
are opt-in via the ``-m integration`` pytest marker and should also
behave gracefully when someone runs the sibling package in a
key-less sandbox.

Key resolution order:
  1. ``GEMINI_API_KEY`` in ``os.environ`` (the standard pattern —
     set it directly for one-off runs or export it from a shell
     profile for a long-lived dev session).
  2. A ``.env`` file at the workspace root, parsed manually with
     the stdlib (no ``python-dotenv`` dependency — it's a 10-line
     parser and this is the only place we need it).

Nothing writes the key to the environment — it's returned from the
fixture and passed into ``GeminiLLMProvider(api_key=...)`` explicitly
so there's no spooky action-at-a-distance when a test interacts with
other process-level env vars.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def _parse_dotenv(path: Path) -> dict[str, str]:
    """Minimal ``.env`` parser — no interpolation, no export prefix.

    Accepts ``KEY=value`` lines. Ignores blank lines, lines starting
    with ``#``, and lines with no ``=``. Strips matching single or
    double quotes from the value. Later keys win over earlier keys.

    We roll our own instead of depending on ``python-dotenv`` because
    it's 10 lines of code and adding a dependency for this would
    violate the Lean Dependency Set principle (even in test scope,
    the principle is "the fewer moving parts, the better").
    """
    result: dict[str, str] = {}
    if not path.is_file():
        return result
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # Strip surrounding quotes if present.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        result[key] = value
    return result


def _find_repo_root(start: Path) -> Path | None:
    """Walk upward from ``start`` looking for a directory that contains
    both a ``.git`` folder and a workspace-root ``pyproject.toml``.

    Returns the directory, or ``None`` if we hit the filesystem root
    without finding one. Used to locate the repo-root ``.env`` file
    regardless of where pytest was invoked from.
    """
    current = start.resolve()
    for parent in (current, *current.parents):
        if (parent / ".git").exists() and (parent / "pyproject.toml").exists():
            return parent
    return None


@pytest.fixture(scope="session")
def gemini_api_key() -> str:
    """Return the Gemini API key, or skip the test if it's unavailable.

    Session-scoped so a single resolution happens per pytest session
    — the file walk + dotenv parse only runs once even if every test
    in the module asks for the key.
    """
    # 1. Environment variable wins.
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key

    # 2. Fall back to the repo-root .env file.
    repo_root = _find_repo_root(Path(__file__).parent)
    if repo_root is not None:
        env_path = repo_root / ".env"
        env_vars = _parse_dotenv(env_path)
        key = env_vars.get("GEMINI_API_KEY", "").strip()
        if key:
            return key

    pytest.skip(
        "GEMINI_API_KEY not found. Set it as an environment variable or "
        "add it to the repo-root .env file to run the neuromem-gemini "
        "integration tests. Example: "
        "`GEMINI_API_KEY=your-key uv run pytest packages/neuromem-gemini/ "
        "-m integration --no-cov`. The `--no-cov` flag is required "
        "because the workspace-root coverage gate is scoped to the "
        "neuromem-core core modules, which the sibling-package "
        "integration tests don't exercise."
    )
