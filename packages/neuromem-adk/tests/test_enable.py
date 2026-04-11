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


# ---------------------------------------------------------------------------
# T012 — US4 LLM-driven active memory search via function tools
# ---------------------------------------------------------------------------


class TestEnableMemoryRegistersFunctionTools:
    """US4 acceptance: after enable_memory, agent.tools contains
    partial-bound search_memory and retrieve_memories tools. The
    internal `system` argument is pre-bound and invisible to ADK's
    schema generator, so the LLM only sees user-facing args."""

    def test_tools_appended_after_enable(
        self,
        mock_adk_agent: Agent,
    ) -> None:
        """agent.tools goes from empty to a 2-element list after
        enable_memory."""
        assert mock_adk_agent.tools == []
        enable_memory(
            mock_adk_agent,
            ":memory:",
            llm=MockLLMProvider(),
            embedder=MockEmbeddingProvider(),
        )
        assert len(mock_adk_agent.tools) == 2

    def test_tool_signatures_hide_internal_system_handle(
        self,
        mock_adk_agent: Agent,
    ) -> None:
        """Both registered tools are wrapper functions (local closures
        over ``memory``) that expose ONLY the user-facing arguments.
        The LLM's schema will see `{query, top_k, depth}` for
        search_memory and `{memory_ids}` for retrieve_memories.

        Wrapper functions are used instead of ``functools.partial``
        because `inspect.signature` on a partial keeps pre-bound
        keyword args in the parameter list (marked with the bound
        value as the default), which would leak the internal
        ``system`` handle into the LLM-visible schema. Wrappers with
        the exact right signature solve this cleanly.
        """
        import inspect  # noqa: PLC0415

        enable_memory(
            mock_adk_agent,
            ":memory:",
            llm=MockLLMProvider(),
            embedder=MockEmbeddingProvider(),
        )

        search_tool, retrieve_tool = mock_adk_agent.tools

        # Both are plain callables (closures), not partials.
        assert callable(search_tool)
        assert callable(retrieve_tool)

        # Signature inspection — what ADK's schema generator sees.
        search_params = set(inspect.signature(search_tool).parameters.keys())
        retrieve_params = set(inspect.signature(retrieve_tool).parameters.keys())

        # Internal plumbing is hidden.
        assert "system" not in search_params
        assert "system" not in retrieve_params
        # Exactly the user-facing args.
        assert search_params == {"query", "top_k", "depth"}
        assert retrieve_params == {"memory_ids"}

        # Docstrings are preserved — ADK uses the docstring as the
        # tool description sent to the LLM.
        assert search_tool.__doc__ is not None
        assert "Search long-term memory" in search_tool.__doc__
        assert retrieve_tool.__doc__ is not None
        assert "Fetch full memory records" in retrieve_tool.__doc__

    def test_search_tool_works_when_invoked_directly(
        self,
        mock_adk_agent: Agent,
    ) -> None:
        """Invoking the partial directly (the way ADK would invoke
        it in response to an LLM tool call) returns the expected
        neuromem.tools.search_memory output."""
        memory = enable_memory(
            mock_adk_agent,
            ":memory:",
            llm=MockLLMProvider(),
            embedder=MockEmbeddingProvider(),
        )
        memory.enqueue("python sqlite async")
        memory.force_dream(block=True)

        search_tool = mock_adk_agent.tools[0]
        result = search_tool(query="python")
        assert isinstance(result, str)
        # Non-empty tree (we just consolidated a memory matching).
        assert "📁" in result or "📄" in result

    def test_retrieve_tool_works_when_invoked_directly(
        self,
        mock_adk_agent: Agent,
    ) -> None:
        """Invoking retrieve_memories directly returns a list of
        dict memory records and spikes access_weight as the LTP
        contract requires."""
        memory = enable_memory(
            mock_adk_agent,
            ":memory:",
            llm=MockLLMProvider(),
            embedder=MockEmbeddingProvider(),
        )
        mem_id = memory.enqueue("test memory")
        memory.force_dream(block=True)

        retrieve_tool = mock_adk_agent.tools[1]
        results = retrieve_tool(memory_ids=[mem_id])
        assert len(results) == 1
        assert results[0]["id"] == mem_id
        assert results[0]["raw_content"] == "test memory"
        assert results[0]["access_weight"] == pytest.approx(1.0)

    def test_preserves_pre_existing_tools(
        self,
        mock_adk_agent: Agent,
    ) -> None:
        """If the agent already has tools in its list, the neuromem
        tools are appended to the end — pre-existing tools stay
        in place."""

        def my_existing_tool(foo: str) -> dict:
            """An unrelated tool the user registered at Agent
            construction time."""
            return {"result": foo}

        mock_adk_agent.tools.append(my_existing_tool)

        enable_memory(
            mock_adk_agent,
            ":memory:",
            llm=MockLLMProvider(),
            embedder=MockEmbeddingProvider(),
        )

        # Pre-existing tool still present at index 0; neuromem's
        # two tools appended at 1 and 2.
        assert len(mock_adk_agent.tools) == 3
        assert mock_adk_agent.tools[0] is my_existing_tool
