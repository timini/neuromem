"""Agent-facing tool functions for ``neuromem``.

Two flat-signature functions that framework-wrapper packages can
bind directly as agent tools (Google ADK, Anthropic SDK, LangChain,
etc. — all expect ``fn(args...) -> result``, not class methods):

- :func:`search_memory` — explore the concept graph via a query
  string, returning an ASCII tree of relevant memories.
- :func:`retrieve_memories` — fetch full memory records by ID,
  triggering the Long-Term Potentiation (LTP) spike on each
  successfully retrieved consolidated memory.

Both functions take the ``NeuroMemory`` system as an explicit
argument rather than holding a reference — tool call interfaces
typically can't capture closures, so the system is passed through
at call time. Downstream wrappers usually ``functools.partial``
the system into the tool to hide this from the agent.

See ``specs/001-neuromem-core/contracts/public-api.md`` §tools for
the full behavioural contract.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from neuromem.context import ContextHelper

if TYPE_CHECKING:
    from neuromem.system import NeuroMemory

logger = logging.getLogger("neuromem.tools")


def search_memory(
    query: str,
    system: NeuroMemory,
    top_k: int = 5,
    depth: int = 2,
) -> str:
    """Search long-term memory for concepts related to ``query``.

    This is a thin wrapper over
    :meth:`ContextHelper.build_prompt_context`. It is provided as a
    flat function (rather than a class method) so framework wrappers
    can register it directly as an agent tool. Semantics are
    identical to ``ContextHelper.build_prompt_context``:

    - Embeds ``query`` via the system's ``EmbeddingProvider``.
    - Finds the ``top_k`` nearest nodes in the knowledge graph.
    - Traverses the sub-graph up to ``depth`` hops from those nodes.
    - Renders the result as a multi-line ASCII tree suitable for
      inclusion in an agent's prompt or tool-result message.

    Returns the empty string when the graph has no relevant content
    (Resolved Design Decision #2 — KISS, no on-demand dream cycle).

    Arguments:
        query: The search query. Must be non-empty.
        system: The ``NeuroMemory`` instance to search.
        top_k: Number of nearest nodes to anchor the traversal on.
            Defaults to 5. Must be ``>= 1``.
        depth: BFS depth for sub-graph traversal. Defaults to 2.
            Must be ``>= 0``.

    Returns:
        A formatted ASCII tree string, or ``""`` if no relevant
        content is found.

    Raises:
        ValueError: on empty query, ``top_k < 1``, or ``depth < 0``.
    """
    helper = ContextHelper(system)
    return helper.build_prompt_context(
        current_task_description=query,
        top_k=top_k,
        depth=depth,
    )


def expand_node(
    node_id: str,
    system: NeuroMemory,
    depth: int = 2,
) -> str:
    """Render the subtree rooted at ``node_id`` as an ASCII tree.

    The ``expand_node`` tool (ADR-003 F6) is the agent-facing
    "zoom in" primitive. After reading the tree returned by
    :func:`search_memory` the agent may decide that one branch
    (e.g. a ``degree`` centroid) looks relevant but the injected
    depth-3 render doesn't have enough content to answer the
    question. Calling ``expand_node(node_id)`` returns a fresh
    render rooted at that node, lazily resolving any uncached
    paragraph_summaries touched by the render along the way.

    Semantics:

    - Renders exactly the same way :meth:`ContextHelper.build_prompt_context`
      does — seeded on ``node_id``, traversed to ``depth`` hops,
      80-node cap, centroid names + paragraph summaries lazy-resolved
      and cached.
    - Does NOT trigger Long-Term Potentiation. Browsing the ontology
      is not the same action as recalling a specific memory; LTP
      fires only on :func:`retrieve_memories`.
    - Returns the empty string when ``node_id`` doesn't exist or has
      no renderable content.
    - Does NOT embed a query vector — the subgraph is seeded directly
      on the provided node id.

    Arguments:
        node_id: The centroid (or leaf) node id to root the subtree on.
        system: The ``NeuroMemory`` instance to query.
        depth: BFS depth from ``node_id``. Defaults to 2.

    Returns:
        An ASCII tree string, or ``""`` if the node doesn't exist or
        has no descendants to render.

    Raises:
        ValueError: on empty ``node_id`` or ``depth < 0``.
    """
    if not node_id:
        raise ValueError("node_id must be non-empty")
    if depth < 0:
        raise ValueError(f"depth must be >= 0, got {depth}")

    subgraph = system.storage.get_subgraph([node_id], depth=depth)
    if not subgraph.get("nodes"):
        return ""

    # Cap + resolve names + resolve summaries — same contract as
    # ContextHelper.build_prompt_context, minus the query-embedding
    # + top-k-seed step.
    from neuromem.context import _NODE_CAP, _enforce_node_cap, _render_ascii_tree  # noqa: PLC0415

    _enforce_node_cap(subgraph, seed_ids=[node_id], cap=_NODE_CAP)
    system.resolve_centroid_names(subgraph["nodes"])
    system.resolve_junction_summaries(subgraph["nodes"])
    return _render_ascii_tree(subgraph)


def retrieve_memories(
    memory_ids: list[str],
    system: NeuroMemory,
) -> list[dict[str, Any]]:
    """Fetch full memory records by ID, triggering LTP on consolidated ones.

    For each ID in ``memory_ids``:

    - If the memory exists and is ``consolidated``, its
      ``access_weight`` is reset to 1.0 and its ``last_accessed`` is
      updated to the current Unix timestamp (Long-Term Potentiation
      per FR-021). This is the agent's recall reinforcement loop.
    - If the memory exists but is ``inbox``, ``dreaming``, or
      ``archived``, it is returned as-is but NOT spiked (LTP only
      applies to consolidated memories per spec data-model.md
      V-M4 / V-M5).
    - If the memory does not exist, it is silently skipped — no
      error, no entry in the result list (per User Story 2
      acceptance scenario #3).

    The LTP spike happens via a single ``spike_access_weight`` call
    batching every consolidated memory id in the input, which is
    cheap even for large batches.

    Arguments:
        memory_ids: List of memory UUIDs to retrieve. Empty list is
            valid and returns ``[]``.
        system: The ``NeuroMemory`` instance to query.

    Returns:
        A list of memory record dicts, one per successfully found
        memory. Each dict has the shape::

            {
                "id": str,
                "raw_content": str,
                "summary": str | None,
                "status": str,       # 'inbox' | 'dreaming' |
                                     # 'consolidated' | 'archived'
                "access_weight": float,
                "created_at": int,   # Unix timestamp
                "last_accessed": int | None,
                "metadata": dict | None,
                "named_entities": list[str],  # proper nouns extracted
                                              # during the dream cycle;
                                              # empty list when NER
                                              # produced no hits or the
                                              # provider doesn't implement
                                              # extract_named_entities.
            }

    Raises:
        Propagates any ``StorageError`` from the underlying adapter.
        Missing IDs are silently skipped — they are NOT a raiseable
        condition.
    """
    if not memory_ids:
        return []

    results: list[dict[str, Any]] = []
    consolidated_ids: list[str] = []

    for mem_id in memory_ids:
        row = system.storage.get_memory_by_id(mem_id)
        if row is None:
            # Missing ID — silent skip. Not an error.
            logger.debug("retrieve_memories: id %s not found, skipping", mem_id)
            continue
        results.append(row)
        if row["status"] == "consolidated":
            consolidated_ids.append(mem_id)

    if consolidated_ids:
        # LTP spike: reset access_weight to 1.0 and last_accessed to now
        # for every consolidated memory in a single adapter call.
        now = int(time.time())
        system.storage.spike_access_weight(consolidated_ids, now)

        # The rows in `results` were fetched BEFORE the spike, so
        # their in-memory access_weight and last_accessed values are
        # stale. Update the dicts in place so callers see the
        # post-spike values without a second round-trip.
        for row in results:
            if row["id"] in consolidated_ids:
                row["access_weight"] = 1.0
                row["last_accessed"] = now

    return results
