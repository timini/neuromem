"""Adapter that plugs neuromem into ADK's native ``BaseMemoryService`` slot.

ADK 1.29's ``google.adk.memory.BaseMemoryService`` defines two
abstract methods:

- ``add_session_to_memory(session)`` — called when a session
  completes. Implementations ingest the session's events into
  their underlying memory store.
- ``search_memory(*, app_name, user_id, query)`` — called by
  ADK's built-in memory-search flow (e.g. via the ``LoadMemory``
  and ``PreloadMemory`` tools). Returns a ``SearchMemoryResponse``
  containing a list of ``MemoryEntry`` hits.

This module provides ``NeuromemMemoryService(BaseMemoryService)``
that delegates both to a live ``NeuroMemory`` instance. Advanced
users who need to wire ADK's runner directly (instead of going
through ``enable_memory``'s callback path) instantiate this class
explicitly and pass it to the ``Runner`` config.

**v0.1 scope note**: The ``app_name`` and ``user_id`` arguments on
``search_memory`` are accepted but not used for tenant isolation.
A single ``NeuromemMemoryService`` instance corresponds to a single
``NeuroMemory`` store, which in v0.1 is single-user by design.
Multi-tenant isolation lands in a future version by either (a)
mapping ``user_id`` onto a metadata filter inside neuromem, or
(b) instantiating one ``NeuromemMemoryService`` per user.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from google.adk.memory.base_memory_service import (
    BaseMemoryService,
    MemoryEntry,
    SearchMemoryResponse,
)
from google.genai import types as genai_types
from neuromem.context import ContextHelper

if TYPE_CHECKING:
    from google.adk.sessions import Session
    from neuromem import NeuroMemory


class NeuromemMemoryService(BaseMemoryService):
    """ADK ``BaseMemoryService`` backed by a ``NeuroMemory`` instance.

    Construct directly when you need to wire neuromem into ADK's
    native memory-service slot without going through
    ``enable_memory``'s callback-based path. For the 90% case,
    use ``enable_memory(agent, db_path=...)`` instead — it
    instantiates a service internally.
    """

    def __init__(self, memory: NeuroMemory) -> None:
        """Wrap an existing ``NeuroMemory`` instance.

        Arguments:
            memory: The ``NeuroMemory`` to delegate to. Its own
                provider and storage configuration determines how
                sessions are summarised, clustered, and persisted.
        """
        self._memory = memory
        self._helper = ContextHelper(memory)

    # ------------------------------------------------------------------
    # BaseMemoryService abstract methods
    # ------------------------------------------------------------------

    def add_session_to_memory(self, session: Session) -> None:
        """Ingest a completed ADK session into neuromem.

        Strategy:
        1. Walk ``session.events`` in order.
        2. For each event that has text content, call
           ``memory.enqueue(...)`` with role metadata derived from
           the event's author.
        3. Force a dream cycle at the end so the new content is
           consolidated into the concept graph before control
           returns to ADK.

        Events without text content (tool calls, function responses,
        images) are silently skipped — neuromem v0.1 only handles
        text. Future versions may support richer modalities.
        """
        events = getattr(session, "events", None) or []
        enqueued_any = False

        for turn_index, event in enumerate(events):
            text = self._extract_event_text(event)
            if not text:
                continue
            author = getattr(event, "author", None) or "unknown"
            role = "user" if author == "user" else "assistant"
            self._memory.enqueue(
                text,
                metadata={
                    "role": role,
                    "turn": turn_index,
                    "author": author,
                    "session_id": getattr(session, "id", None),
                },
            )
            enqueued_any = True

        if enqueued_any:
            # Force consolidation before returning so the memory
            # graph reflects the session's content before the next
            # session starts. This is the key behaviour users of
            # ADK's native memory-service flow expect.
            self._memory.force_dream(block=True)

    def search_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        query: str,
    ) -> SearchMemoryResponse:
        """Search the memory store for entries matching ``query``.

        Delegates to ``neuromem.tools.search_memory`` via the
        ``ContextHelper`` and converts the returned ASCII tree into
        ADK's ``SearchMemoryResponse`` shape. In v0.1 ``app_name``
        and ``user_id`` are accepted but not used — see the module
        docstring for the multi-tenant scope note.

        Returns a ``SearchMemoryResponse`` with one ``MemoryEntry``
        per matching memory. The ASCII tree from the ContextHelper
        is converted back into individual entries by walking the
        underlying graph; we never return the tree string itself
        because ADK's ``MemoryEntry.content`` expects a
        ``genai.types.Content`` object, not free-form text.
        """
        # Discard app_name / user_id — not used in v0.1. Reserved
        # for future multi-tenant support.
        del app_name, user_id

        # Build the tree to get the anchoring nearest-neighbour
        # traversal that would run under enable_memory's passive
        # injection path. Then convert the traversal's memory list
        # to MemoryEntry objects.
        tree = self._helper.build_prompt_context(query)
        if not tree:
            return SearchMemoryResponse(memories=[])

        # Walk the subgraph ourselves to get the structured memory
        # list. ContextHelper.build_prompt_context already renders
        # an ASCII tree — for ADK we need the raw memory records,
        # so we repeat the graph traversal at the storage layer
        # and filter down to the memories surfaced in the tree.
        memory_ids = _parse_memory_ids_from_tree(tree)
        entries: list[MemoryEntry] = []
        for mem_id in memory_ids:
            row = self._memory.storage.get_memory_by_id(mem_id)
            if row is None:
                continue
            entries.append(_memory_row_to_entry(row))

        return SearchMemoryResponse(memories=entries)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_event_text(event: Any) -> str:
        """Return the concatenated text of all text parts in an
        event's Content, or the empty string if there's no text."""
        content = getattr(event, "content", None)
        if content is None:
            return ""
        parts = getattr(content, "parts", None) or []
        texts = [getattr(p, "text", None) or "" for p in parts]
        return " ".join(t for t in texts if t).strip()


# ---------------------------------------------------------------------------
# Module-level helpers (used by NeuromemMemoryService.search_memory)
# ---------------------------------------------------------------------------


def _parse_memory_ids_from_tree(tree: str) -> list[str]:
    """Pull every ``mem_...`` id out of a rendered ContextHelper tree.

    The ASCII tree format uses ``📄 mem_<hex>: "<summary>"`` for each
    memory leaf. We extract the id with a simple scan rather than
    re-traversing the graph — this keeps ``NeuromemMemoryService``
    agnostic to the underlying storage's subgraph method and means
    changes to the storage layer can't silently break the
    MemoryService.
    """
    ids: list[str] = []
    for line in tree.splitlines():
        # Look for ``mem_<hex>`` tokens — the leaf marker format.
        idx = line.find("mem_")
        if idx == -1:
            continue
        end = idx
        while end < len(line) and (line[end].isalnum() or line[end] == "_"):
            end += 1
        mem_id = line[idx:end]
        if mem_id.startswith("mem_") and len(mem_id) > 4:
            ids.append(mem_id)
    return ids


def _memory_row_to_entry(row: dict[str, Any]) -> MemoryEntry:
    """Convert a neuromem memory-record dict into an ADK
    ``MemoryEntry``.

    Maps:
    - ``row["raw_content"]`` → ``entry.content`` as a
      ``genai.types.Content`` with a single text ``Part``.
    - ``row["metadata"]`` → ``entry.custom_metadata``.
    - ``row["id"]`` → ``entry.id``.
    - ``row["metadata"]["role"]`` (or "neuromem" fallback) → ``entry.author``.
    - ``row["last_accessed"]`` (or ``row["created_at"]``) → ``entry.timestamp``
      as an ISO-8601 UTC string (ADK's MemoryEntry.timestamp is
      ``Optional[str]``, not an int).
    """
    from datetime import datetime, timezone  # noqa: PLC0415

    metadata = row.get("metadata") or {}
    author = metadata.get("role") or "neuromem"
    ts_int = row.get("last_accessed") or row.get("created_at") or int(time.time())
    ts_str = datetime.fromtimestamp(int(ts_int), tz=timezone.utc).isoformat()

    content = genai_types.Content(
        parts=[genai_types.Part(text=row["raw_content"])],
    )

    return MemoryEntry(
        content=content,
        custom_metadata=metadata,
        id=row["id"],
        author=author,
        timestamp=ts_str,
    )
