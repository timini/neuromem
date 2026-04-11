"""Unit tests for neuromem_adk.memory_service.NeuromemMemoryService.

Covers T013 (US5 native BaseMemoryService integration). Uses
synthetic session / event objects built with stdlib
``types.SimpleNamespace`` rather than real ADK session construction.

The ``SearchMemoryResponse`` and ``MemoryEntry`` classes ARE from
real ADK — we need to verify our output matches their shape.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from google.adk.memory.base_memory_service import (
    BaseMemoryService,
    MemoryEntry,
    SearchMemoryResponse,
)
from neuromem import NeuroMemory
from neuromem_adk import NeuromemMemoryService


def _make_content(text: str) -> SimpleNamespace:
    """Minimal Content-shaped synthetic object with a text Part."""
    return SimpleNamespace(parts=[SimpleNamespace(text=text)])


def _make_event(author: str, text: str) -> SimpleNamespace:
    return SimpleNamespace(author=author, content=_make_content(text))


def _make_session(
    events: list[SimpleNamespace],
    *,
    session_id: str = "test-session",
) -> SimpleNamespace:
    return SimpleNamespace(id=session_id, events=events)


# ---------------------------------------------------------------------------
# Subclassing + abstract method implementation
# ---------------------------------------------------------------------------


class TestNeuromemMemoryServiceInstantiation:
    def test_subclasses_base_memory_service(
        self,
        mock_neuromem_system: NeuroMemory,
    ) -> None:
        """NeuromemMemoryService IS a BaseMemoryService — the two
        abstract methods have been implemented so `__init__` does
        not raise TypeError at instantiation."""
        service = NeuromemMemoryService(mock_neuromem_system)
        assert isinstance(service, BaseMemoryService)

    def test_stores_memory_reference(
        self,
        mock_neuromem_system: NeuroMemory,
    ) -> None:
        """The service holds a reference to the underlying memory so
        downstream code can inspect or override it."""
        service = NeuromemMemoryService(mock_neuromem_system)
        assert service._memory is mock_neuromem_system


# ---------------------------------------------------------------------------
# add_session_to_memory
# ---------------------------------------------------------------------------


class TestAddSessionToMemory:
    """Session ingestion — walk events, enqueue text, force dream."""

    def test_ingests_three_turn_session(
        self,
        mock_neuromem_system: NeuroMemory,
    ) -> None:
        """A 3-event session (user, assistant, user) lands as 3
        consolidated memories after add_session_to_memory returns."""
        service = NeuromemMemoryService(mock_neuromem_system)
        session = _make_session(
            events=[
                _make_event("user", "hello there"),
                _make_event("test_agent", "hi! how can I help?"),
                _make_event("user", "tell me about python"),
            ],
        )
        service.add_session_to_memory(session)

        # All three were enqueued, then force_dream'd → consolidated.
        assert mock_neuromem_system.storage.count_memories_by_status("inbox") == 0
        assert mock_neuromem_system.storage.count_memories_by_status("consolidated") == 3

    def test_session_metadata_attached_to_memories(
        self,
        mock_neuromem_system: NeuroMemory,
    ) -> None:
        """Every enqueued memory's metadata includes the session id,
        turn index, author, and normalised role."""
        service = NeuromemMemoryService(mock_neuromem_system)
        session = _make_session(
            events=[
                _make_event("user", "hello"),
                _make_event("test_agent", "hi"),
            ],
            session_id="session-abc",
        )
        service.add_session_to_memory(session)

        rows = mock_neuromem_system.storage.get_memories_by_status("consolidated")
        assert len(rows) == 2
        for row in rows:
            md = row["metadata"]
            assert md["session_id"] == "session-abc"
            assert md["author"] in ("user", "test_agent")
            assert md["role"] in ("user", "assistant")
            assert md["turn"] in (0, 1)

    def test_skips_events_without_text_content(
        self,
        mock_neuromem_system: NeuroMemory,
    ) -> None:
        """Events with empty / missing content (tool calls, function
        responses in a real session) are silently skipped."""
        service = NeuromemMemoryService(mock_neuromem_system)
        session = _make_session(
            events=[
                _make_event("user", "real content"),
                SimpleNamespace(author="tool", content=None),  # no content
                SimpleNamespace(
                    author="test_agent",
                    content=SimpleNamespace(parts=[]),
                ),  # empty parts
                _make_event("test_agent", "another real one"),
            ],
        )
        service.add_session_to_memory(session)

        # Only the two real-content events.
        assert mock_neuromem_system.storage.count_memories_by_status("consolidated") == 2

    def test_empty_session_is_noop(
        self,
        mock_neuromem_system: NeuroMemory,
    ) -> None:
        """A session with no events doesn't enqueue anything and
        doesn't force a dream (performance optimisation)."""
        service = NeuromemMemoryService(mock_neuromem_system)
        session = _make_session(events=[])
        service.add_session_to_memory(session)

        assert mock_neuromem_system.storage.count_memories_by_status("inbox") == 0
        assert mock_neuromem_system.storage.count_memories_by_status("consolidated") == 0


# ---------------------------------------------------------------------------
# search_memory
# ---------------------------------------------------------------------------


class TestSearchMemory:
    """Search delegation — returns a SearchMemoryResponse in ADK's
    native shape with the right MemoryEntry content."""

    def test_returns_search_memory_response(
        self,
        mock_neuromem_system: NeuroMemory,
    ) -> None:
        """Empty graph → SearchMemoryResponse with empty memories list.
        This is the correct ADK return shape even when nothing matches.
        """
        service = NeuromemMemoryService(mock_neuromem_system)
        result = service.search_memory(
            app_name="test-app",
            user_id="test-user",
            query="anything",
        )
        assert isinstance(result, SearchMemoryResponse)
        assert result.memories == []

    def test_returns_matching_memories(
        self,
        mock_neuromem_system: NeuroMemory,
    ) -> None:
        """Seed memories, force consolidation, search for a matching
        query. The response contains MemoryEntry objects whose content
        is the stored raw_content."""
        mem_id = mock_neuromem_system.enqueue("python sqlite async discussion")
        mock_neuromem_system.force_dream(block=True)

        service = NeuromemMemoryService(mock_neuromem_system)
        result = service.search_memory(
            app_name="test-app",
            user_id="test-user",
            query="python sqlite",
        )

        assert isinstance(result, SearchMemoryResponse)
        assert len(result.memories) >= 1

        entry = result.memories[0]
        assert isinstance(entry, MemoryEntry)
        # Content has the right shape.
        assert entry.content is not None
        assert len(entry.content.parts) == 1
        assert entry.content.parts[0].text == "python sqlite async discussion"
        # The seeded memory's id round-trips through the search.
        ids_returned = {e.id for e in result.memories}
        assert mem_id in ids_returned

    def test_app_name_and_user_id_are_accepted_but_not_used(
        self,
        mock_neuromem_system: NeuroMemory,
    ) -> None:
        """v0.1 doesn't implement multi-tenant isolation — app_name
        and user_id arguments are accepted (ADK's BaseMemoryService
        contract requires them) but don't filter anything. Confirmed
        by a smoke call with clearly-distinct values."""
        mock_neuromem_system.enqueue("shared memory content")
        mock_neuromem_system.force_dream(block=True)

        service = NeuromemMemoryService(mock_neuromem_system)

        r1 = service.search_memory(
            app_name="app-one",
            user_id="user-one",
            query="memory",
        )
        r2 = service.search_memory(
            app_name="app-two",
            user_id="user-two",
            query="memory",
        )

        # Same hits for both — no tenant isolation applied.
        ids_r1 = {e.id for e in r1.memories}
        ids_r2 = {e.id for e in r2.memories}
        assert ids_r1 == ids_r2

    def test_search_ignores_archived_memories(
        self,
        mock_neuromem_system: NeuroMemory,
    ) -> None:
        """Archived memories are filtered out of the ContextHelper
        tree, so the search response never contains them. This matches
        the neuromem-core contract — archived memories are retrievable
        only via retrieve_memories (by explicit id), not via search."""
        import time as time_module  # noqa: PLC0415

        mem_id = mock_neuromem_system.enqueue("will be archived")
        mock_neuromem_system.force_dream(block=True)

        # Age the memory so it gets archived on next decay.
        now = int(time_module.time())
        year_ago = now - 365 * 86400
        mock_neuromem_system.storage.spike_access_weight([mem_id], year_ago)
        mock_neuromem_system.storage.apply_decay_and_archive(
            decay_lambda=1e-6,
            archive_threshold=0.1,
            current_timestamp=now,
        )
        assert mock_neuromem_system.storage.get_memory_by_id(mem_id)["status"] == "archived"

        # Search for the topic — archived memory must not appear.
        service = NeuromemMemoryService(mock_neuromem_system)
        result = service.search_memory(
            app_name="test",
            user_id="test",
            query="archived",
        )
        ids = {e.id for e in result.memories}
        assert mem_id not in ids


# Silence unused imports if auto-fixture discovery doesn't count them.
_ = pytest
_ = Any
