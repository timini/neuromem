"""Unit tests for neuromem_adk.callbacks.

Covers T010 (US2 automatic turn capture via after_agent_callback)
and T011 (US3 passive context injection via before_model_callback).

Tests use stdlib ``types.SimpleNamespace`` to build synthetic Context
/ LlmRequest objects. Constructing real ADK objects (`Context`,
`InvocationContext`, `Session`, `Event`, `LlmRequest`) would require a
full runtime tree with memory/artifact/session services, which is
both heavy and orthogonal to what we're testing — the callback
closures only read a handful of attributes. SimpleNamespace is the
right abstraction for a unit test.

Integration tests against real ADK objects live in
``test_adk_integration.py`` and run against real Gemini.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from neuromem import NeuroMemory
from neuromem_adk.callbacks import (
    build_after_agent_turn_capturer,
    build_before_model_context_injector,
)

# Test fixtures come from conftest.py (same directory).


def _make_content(text: str) -> SimpleNamespace:
    """Synthesise a minimal ADK-Content-shaped object with a single
    text part. Matches the ``.parts[i].text`` access pattern the
    callback's ``_extract_text_from_content`` helper uses."""
    return SimpleNamespace(parts=[SimpleNamespace(text=text)])


def _make_event(author: str, text: str) -> SimpleNamespace:
    """Synthesise a minimal Event-shaped object (author + content)."""
    return SimpleNamespace(author=author, content=_make_content(text))


def _make_context(
    *,
    agent_name: str = "test_agent",
    user_text: str = "",
    events: list[SimpleNamespace] | None = None,
) -> SimpleNamespace:
    """Synthesise a Context-shaped object for callback input."""
    return SimpleNamespace(
        agent_name=agent_name,
        user_content=_make_content(user_text) if user_text else None,
        session=SimpleNamespace(events=events or []),
    )


# ---------------------------------------------------------------------------
# T010 — US2 automatic turn capture
# ---------------------------------------------------------------------------


class TestAfterAgentTurnCapturer:
    """Every completed agent turn flows into memory.enqueue via
    after_agent_callback."""

    def test_captures_user_and_assistant_as_two_memories(
        self,
        mock_neuromem_system: NeuroMemory,
    ) -> None:
        """Happy path: user said X, agent responded Y. Two memories
        land in the inbox, tagged with role metadata."""
        capturer = build_after_agent_turn_capturer(mock_neuromem_system)
        ctx = _make_context(
            agent_name="test_agent",
            user_text="What is the capital of France?",
            events=[
                _make_event("user", "What is the capital of France?"),
                _make_event("test_agent", "The capital of France is Paris."),
            ],
        )
        result = capturer(ctx)

        # Callback returns None (don't override the agent's response).
        assert result is None

        # Both turns landed in memory.
        assert mock_neuromem_system.storage.count_memories_by_status("inbox") == 2

        rows = mock_neuromem_system.storage.get_memories_by_status("inbox")
        contents = {r["raw_content"]: r["metadata"] for r in rows}
        assert "What is the capital of France?" in contents
        assert "The capital of France is Paris." in contents
        assert contents["What is the capital of France?"]["role"] == "user"
        assert contents["The capital of France is Paris."]["role"] == "assistant"

    def test_empty_user_content_captures_only_assistant(
        self,
        mock_neuromem_system: NeuroMemory,
    ) -> None:
        """If user_content is missing / empty (e.g. the first agent
        run has no explicit user prompt), only the assistant turn
        gets captured."""
        capturer = build_after_agent_turn_capturer(mock_neuromem_system)
        ctx = _make_context(
            agent_name="test_agent",
            user_text="",
            events=[
                _make_event("test_agent", "Hello! How can I help?"),
            ],
        )
        capturer(ctx)

        assert mock_neuromem_system.storage.count_memories_by_status("inbox") == 1
        rows = mock_neuromem_system.storage.get_memories_by_status("inbox")
        assert rows[0]["raw_content"] == "Hello! How can I help?"
        assert rows[0]["metadata"]["role"] == "assistant"

    def test_no_events_captures_user_only(
        self,
        mock_neuromem_system: NeuroMemory,
    ) -> None:
        """If the session has no events yet (corner case, shouldn't
        happen in practice after a completed agent turn), we
        capture the user_content and skip the assistant side."""
        capturer = build_after_agent_turn_capturer(mock_neuromem_system)
        ctx = _make_context(
            agent_name="test_agent",
            user_text="standalone user message",
            events=[],
        )
        capturer(ctx)

        assert mock_neuromem_system.storage.count_memories_by_status("inbox") == 1
        rows = mock_neuromem_system.storage.get_memories_by_status("inbox")
        assert rows[0]["raw_content"] == "standalone user message"
        assert rows[0]["metadata"]["role"] == "user"

    def test_walks_backwards_to_find_latest_agent_event(
        self,
        mock_neuromem_system: NeuroMemory,
    ) -> None:
        """The capturer must find the MOST RECENT agent-authored
        event, not the first one, when the session has multiple
        prior turns."""
        capturer = build_after_agent_turn_capturer(mock_neuromem_system)
        ctx = _make_context(
            agent_name="test_agent",
            user_text="third user turn",
            events=[
                _make_event("user", "first user turn"),
                _make_event("test_agent", "first agent response"),
                _make_event("user", "second user turn"),
                _make_event("test_agent", "second agent response"),
                _make_event("user", "third user turn"),
                _make_event("test_agent", "third agent response — this is the latest"),
            ],
        )
        capturer(ctx)

        rows = mock_neuromem_system.storage.get_memories_by_status("inbox")
        assistant_rows = [r for r in rows if r["metadata"]["role"] == "assistant"]
        assert len(assistant_rows) == 1
        assert assistant_rows[0]["raw_content"] == ("third agent response — this is the latest")

    def test_skips_silent_on_unknown_agent_name(
        self,
        mock_neuromem_system: NeuroMemory,
    ) -> None:
        """If no event's author matches the agent's name, the
        assistant capture is skipped — but the user capture still
        runs."""
        capturer = build_after_agent_turn_capturer(mock_neuromem_system)
        ctx = _make_context(
            agent_name="test_agent",
            user_text="my turn",
            events=[
                _make_event("some_other_agent", "this is a different agent"),
            ],
        )
        capturer(ctx)

        assert mock_neuromem_system.storage.count_memories_by_status("inbox") == 1
        rows = mock_neuromem_system.storage.get_memories_by_status("inbox")
        assert rows[0]["metadata"]["role"] == "user"

    def test_turn_index_tracks_event_count(
        self,
        mock_neuromem_system: NeuroMemory,
    ) -> None:
        """The ``turn`` metadata field equals the session's event
        count at the moment of capture — this lets downstream
        consumers order memories chronologically within a session."""
        capturer = build_after_agent_turn_capturer(mock_neuromem_system)
        events = [
            _make_event("user", "1"),
            _make_event("test_agent", "1"),
            _make_event("user", "2"),
            _make_event("test_agent", "2"),
        ]
        ctx = _make_context(
            agent_name="test_agent",
            user_text="4th event makes it 4 total",
            events=events,
        )
        capturer(ctx)

        rows = mock_neuromem_system.storage.get_memories_by_status("inbox")
        for row in rows:
            assert row["metadata"]["turn"] == 4


# ---------------------------------------------------------------------------
# T011 — US3 passive context injection
# ---------------------------------------------------------------------------


class TestBeforeModelContextInjector:
    """Before each model call, the relevant memory tree gets
    prepended to the system instruction."""

    def _make_llm_request(
        self,
        *,
        instruction: str | None = None,
    ) -> SimpleNamespace:
        """Synthesise an LlmRequest-shaped object whose config has a
        mutable ``system_instruction`` attribute."""
        config = SimpleNamespace(system_instruction=instruction)
        return SimpleNamespace(config=config)

    def test_empty_graph_leaves_request_unchanged(
        self,
        mock_neuromem_system: NeuroMemory,
    ) -> None:
        """No memories yet → build_prompt_context returns empty
        string → no injection → request unchanged."""
        injector = build_before_model_context_injector(mock_neuromem_system)
        ctx = _make_context(user_text="what should I do?")
        request = self._make_llm_request(instruction="You are a helpful assistant.")

        result = injector(ctx, request)

        assert result is None
        assert request.config.system_instruction == "You are a helpful assistant."

    def test_injects_tree_for_matching_query(
        self,
        mock_neuromem_system: NeuroMemory,
    ) -> None:
        """Seed a memory, force_dream it into consolidated state,
        then call the injector with a matching query. The tree
        should be prepended."""
        # Seed and consolidate.
        mock_neuromem_system.enqueue("python sqlite async memory discussion")
        mock_neuromem_system.force_dream(block=True)

        injector = build_before_model_context_injector(mock_neuromem_system)
        ctx = _make_context(user_text="python sqlite")
        request = self._make_llm_request(instruction="base instruction")

        injector(ctx, request)

        # The instruction has been mutated to include the memory tree.
        # We don't hard-code the ASCII tree content — the MockLLMProvider's
        # tag extraction is non-deterministic in detail — we just
        # assert the base instruction is still present and SOMETHING
        # was prepended.
        assert request.config.system_instruction is not None
        assert "base instruction" in request.config.system_instruction
        assert len(request.config.system_instruction) > len("base instruction")

    def test_no_user_content_falls_back_to_latest_user_event(
        self,
        mock_neuromem_system: NeuroMemory,
    ) -> None:
        """If ctx.user_content is missing, the injector walks the
        session events backwards to find the latest user turn and
        uses that as the query."""
        mock_neuromem_system.enqueue("python sqlite async memory")
        mock_neuromem_system.force_dream(block=True)

        injector = build_before_model_context_injector(mock_neuromem_system)
        ctx = _make_context(
            user_text="",  # No current user_content
            events=[
                _make_event("user", "python sqlite"),
                _make_event("test_agent", "response"),
            ],
        )
        request = self._make_llm_request(instruction="base")

        injector(ctx, request)

        # The fallback found "python sqlite" from events and did
        # the lookup.
        assert len(request.config.system_instruction) > len("base")

    def test_no_query_at_all_leaves_request_unchanged(
        self,
        mock_neuromem_system: NeuroMemory,
    ) -> None:
        """If neither user_content nor any user-authored event has
        text, the injector bails silently."""
        injector = build_before_model_context_injector(mock_neuromem_system)
        ctx = _make_context(user_text="", events=[])
        request = self._make_llm_request(instruction="base")

        injector(ctx, request)

        assert request.config.system_instruction == "base"

    def test_missing_config_bails_silently(
        self,
        mock_neuromem_system: NeuroMemory,
    ) -> None:
        """If the LlmRequest has no config attr, the injector bails
        silently — we don't want callback errors to break the
        agent turn."""
        mock_neuromem_system.enqueue("python sqlite")
        mock_neuromem_system.force_dream(block=True)

        injector = build_before_model_context_injector(mock_neuromem_system)
        ctx = _make_context(user_text="python sqlite")
        # No config attribute at all.
        request = SimpleNamespace()

        result = injector(ctx, request)

        assert result is None

    def test_empty_existing_instruction_replaced_by_tree(
        self,
        mock_neuromem_system: NeuroMemory,
    ) -> None:
        """If the existing system_instruction is None or empty, the
        tree becomes the new instruction (not prepended to nothing
        with two empty newlines — it's just set outright)."""
        mock_neuromem_system.enqueue("python sqlite async")
        mock_neuromem_system.force_dream(block=True)

        injector = build_before_model_context_injector(mock_neuromem_system)
        ctx = _make_context(user_text="python")
        request = self._make_llm_request(instruction=None)

        injector(ctx, request)

        # Must not start with newlines — the tree should be the full
        # content, not a prepend-to-None.
        assert request.config.system_instruction is not None
        assert not request.config.system_instruction.startswith("\n")

    def test_returns_none_always(
        self,
        mock_neuromem_system: NeuroMemory,
    ) -> None:
        """ADK callback contract: returning None tells ADK to
        proceed with the (possibly mutated) request. Returning a
        specific object short-circuits the model call. We always
        return None."""
        injector = build_before_model_context_injector(mock_neuromem_system)
        ctx = _make_context(user_text="anything")
        request = self._make_llm_request()

        result = injector(ctx, request)
        assert result is None


# ---------------------------------------------------------------------------
# Smoke test — enable_memory wires both callbacks correctly
# ---------------------------------------------------------------------------


class TestEnableMemoryWiring:
    """enable_memory attaches both callbacks to the agent in the
    correct slots, preserving any pre-existing callback."""

    def test_both_callback_slots_set_after_enable(
        self,
        mock_adk_agent: Any,
    ) -> None:
        from conftest import MockEmbeddingProvider, MockLLMProvider  # noqa: PLC0415
        from neuromem_adk import enable_memory  # noqa: PLC0415

        assert mock_adk_agent.before_model_callback is None
        assert mock_adk_agent.after_agent_callback is None

        enable_memory(
            mock_adk_agent,
            ":memory:",
            llm=MockLLMProvider(),
            embedder=MockEmbeddingProvider(),
        )

        # Both slots now have a callable attached.
        assert mock_adk_agent.before_model_callback is not None
        assert mock_adk_agent.after_agent_callback is not None
        assert callable(mock_adk_agent.before_model_callback) or isinstance(
            mock_adk_agent.before_model_callback, list
        )
        assert callable(mock_adk_agent.after_agent_callback) or isinstance(
            mock_adk_agent.after_agent_callback, list
        )

    def test_preserves_existing_callback_via_list_chaining(
        self,
        mock_adk_agent: Any,
    ) -> None:
        """If the agent already has a callback in a slot, enable_memory
        preserves it by converting to a list that ADK will iterate."""
        from conftest import MockEmbeddingProvider, MockLLMProvider  # noqa: PLC0415
        from neuromem_adk import enable_memory  # noqa: PLC0415

        def pre_existing_cb(*args: Any, **kwargs: Any) -> None:
            return None

        mock_adk_agent.before_model_callback = pre_existing_cb

        enable_memory(
            mock_adk_agent,
            ":memory:",
            llm=MockLLMProvider(),
            embedder=MockEmbeddingProvider(),
        )

        # The slot is now a list containing the pre-existing callback
        # as the first element (runs first) and neuromem's injector
        # as the second (runs after).
        assert isinstance(mock_adk_agent.before_model_callback, list)
        assert mock_adk_agent.before_model_callback[0] is pre_existing_cb
        assert callable(mock_adk_agent.before_model_callback[1])


# Silence unused import complaint if pytest's auto-fixture discovery
# doesn't count the direct imports as usage.
_ = pytest
