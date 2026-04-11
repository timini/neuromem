"""Unit tests for neuromem_adk.enable_memory.

Covers T008 (US1 happy path), T009 (US1 error cases), T012 (US4 tool
registration — future), and T014 (US6 manual provider override —
future). Each task's tests land as they're built; this file grows
incrementally.

No network calls. Every test uses mock providers or explicitly
monkeypatches environment variables. Integration tests against real
ADK + real Gemini live in ``test_adk_integration.py`` and are
gated by ``@pytest.mark.integration``.
"""

from __future__ import annotations

import pytest
from conftest import (  # type: ignore[import-not-found]
    MockEmbeddingProvider,
    MockLLMProvider,
)
from google.adk.agents import Agent
from neuromem import NeuroMemory, SQLiteAdapter
from neuromem_adk import enable_memory

# ---------------------------------------------------------------------------
# T008 — US1 happy path
# ---------------------------------------------------------------------------


class TestEnableMemoryHappyPath:
    """US1 acceptance scenarios 1-3: enable_memory returns a working
    NeuroMemory handle, creates a DB file if missing, preserves an
    existing DB."""

    def test_returns_neuromem_instance(
        self,
        mock_adk_agent: Agent,
    ) -> None:
        """enable_memory returns a NeuroMemory backed by a SQLiteAdapter."""
        memory = enable_memory(
            mock_adk_agent,
            ":memory:",
            llm=MockLLMProvider(),
            embedder=MockEmbeddingProvider(),
        )
        assert isinstance(memory, NeuroMemory)
        assert isinstance(memory.storage, SQLiteAdapter)

    def test_sets_enabled_marker_on_agent(
        self,
        mock_adk_agent: Agent,
    ) -> None:
        """The agent gets an internal marker so double-attachment can be
        detected. Marker name is private — we test by inspecting the
        attribute directly (acceptable for an integration like this
        where the marker IS the cross-cut contract)."""
        assert not getattr(mock_adk_agent, "_neuromem_adk_enabled", False)
        enable_memory(
            mock_adk_agent,
            ":memory:",
            llm=MockLLMProvider(),
            embedder=MockEmbeddingProvider(),
        )
        assert getattr(mock_adk_agent, "_neuromem_adk_enabled", False) is True

    def test_memory_handle_is_usable_for_enqueue(
        self,
        mock_adk_agent: Agent,
    ) -> None:
        """The returned handle works for enqueue — proves the full
        wiring path, not just the constructor."""
        memory = enable_memory(
            mock_adk_agent,
            ":memory:",
            llm=MockLLMProvider(),
            embedder=MockEmbeddingProvider(),
        )
        mem_id = memory.enqueue("hello world from the test")
        assert mem_id.startswith("mem_")
        assert memory.storage.count_memories_by_status("inbox") == 1


# ---------------------------------------------------------------------------
# T009 — US1 error cases
# ---------------------------------------------------------------------------


class TestEnableMemoryErrors:
    """US1 acceptance scenarios: double-attachment raises, missing
    credentials raise with a clear message."""

    def test_double_attachment_raises_value_error(
        self,
        mock_adk_agent: Agent,
    ) -> None:
        """Calling enable_memory twice on the same agent must raise
        ValueError with a message that mentions the agent name, so
        the developer sees which agent they already wired."""
        enable_memory(
            mock_adk_agent,
            ":memory:",
            llm=MockLLMProvider(),
            embedder=MockEmbeddingProvider(),
        )
        with pytest.raises(ValueError, match="already has memory attached"):
            enable_memory(
                mock_adk_agent,
                ":memory:",
                llm=MockLLMProvider(),
                embedder=MockEmbeddingProvider(),
            )

    def test_missing_api_key_raises_key_error(
        self,
        mock_adk_agent: Agent,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If llm and embedder are both omitted AND neither env var is
        set, enable_memory must raise KeyError with a message naming
        both env var candidates so the developer knows which one to
        set.

        Uses monkeypatch.delenv to avoid touching os.environ directly
        — restores original state at test teardown."""
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        with pytest.raises(KeyError) as exc_info:
            enable_memory(mock_adk_agent, ":memory:")

        msg = str(exc_info.value)
        assert "GOOGLE_API_KEY" in msg
        assert "GEMINI_API_KEY" in msg

    def test_missing_api_key_does_not_partially_attach(
        self,
        mock_adk_agent: Agent,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The credentials check fires BEFORE any attachment happens,
        so a failed credential lookup leaves the agent untouched and
        a subsequent call with valid providers succeeds."""
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        with pytest.raises(KeyError):
            enable_memory(mock_adk_agent, ":memory:")

        # Marker MUST NOT be set on the agent — the failed attach left
        # the agent pristine.
        assert not getattr(mock_adk_agent, "_neuromem_adk_enabled", False)

        # A subsequent call with explicit providers should succeed.
        memory = enable_memory(
            mock_adk_agent,
            ":memory:",
            llm=MockLLMProvider(),
            embedder=MockEmbeddingProvider(),
        )
        assert isinstance(memory, NeuroMemory)

    def test_explicit_providers_bypass_credential_check(
        self,
        mock_adk_agent: Agent,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Passing both llm and embedder means the default-provider
        path is never hit, so the credential check never runs. This
        is how tests and advanced users skip the env-var requirement.
        """
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        memory = enable_memory(
            mock_adk_agent,
            ":memory:",
            llm=MockLLMProvider(),
            embedder=MockEmbeddingProvider(),
        )
        assert isinstance(memory, NeuroMemory)
