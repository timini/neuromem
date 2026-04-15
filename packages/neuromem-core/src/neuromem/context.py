"""``ContextHelper`` — prompt-injection helper for agent LLM calls.

Given a task description, embeds it via the injected
``EmbeddingProvider``, finds the closest nodes in the knowledge
graph, traverses outward to build a contextual sub-graph, and
renders the result as an ASCII tree suitable for inclusion in an
agent's system prompt.

Intended call site: before every LLM invocation in the agent's
runtime loop::

    helper = ContextHelper(memory_system)
    ctx = helper.build_prompt_context(current_task)
    system_prompt = f"You are a helpful agent.\\n\\nRelevant memories:\\n{ctx}"

The helper is stateless — it only holds a reference to the
``NeuroMemory`` system. All state lives in ``memory_system.storage``.

See specs/001-neuromem-core/contracts/public-api.md §ContextHelper
for the full behavioural contract.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neuromem.system import NeuroMemory

logger = logging.getLogger("neuromem.context")


# Max characters of a memory summary to include inline in the rendered
# tree. Summaries are typically 1–2 sentences (~150–300 chars) coming
# out of ``LLMProvider.generate_summary``; at 80 chars we were
# truncating most of them, losing the detail the answer LLM needs to
# answer specific questions. 300 chars comfortably fits two sentences
# and still caps pathological outputs. Not yet configurable at the
# API level — can be made so if a real-world use case needs something
# different.
_MEMORY_SNIPPET_MAX_CHARS = 300

# Max characters of a per-centroid paragraph summary (ADR-003 D2) to
# include inline in the rendered tree. Summaries come out of
# ``LLMProvider.generate_junction_summary`` at 2-4 sentences each
# (~400-600 chars). Cap here is slightly higher than memory snippets
# because junctions abstract over more content, so the summary
# carries proportionally more signal per character.
_JUNCTION_SUMMARY_MAX_CHARS = 400

# ADR-003 F5/D3: hard cap on the rendered subgraph's node count.
# Depth-3 renders on a busy graph can easily touch hundreds of nodes
# and overwhelm the answer LLM's prompt window. 80 nodes × ~400 chars
# of tree-row text ≈ 32 KB, comfortably inside Gemini Flash's window.
# Seeds (originally retrieved top_k nodes) are never dropped; other
# nodes are dropped farthest-from-seed-first until the cap is met.
_NODE_CAP = 80


class ContextHelper:
    """Prompt-injection helper backed by a ``NeuroMemory`` instance."""

    def __init__(self, memory_system: NeuroMemory) -> None:
        self.system = memory_system

    def build_prompt_context(
        self,
        current_task_description: str,
        top_k: int = 5,
        depth: int = 3,
    ) -> str:
        """Embed the task, find the nearest nodes, render the sub-graph.

        Returns a formatted ASCII tree ready to drop into a system
        prompt, or an empty string if the graph has no relevant nodes
        (per Resolved Design Decision #2 — KISS, no on-demand dream
        cycle).

        Side effects:
          - one call to ``embedder.get_embeddings([current_task_description])``
          - no LLM calls
          - no ``retrieve_memories`` call, so no LTP spike — retrieval
            of specific memories is the agent's follow-up action via
            ``neuromem.tools.retrieve_memories``.
        """
        if not current_task_description:
            raise ValueError("current_task_description must be non-empty")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if depth < 0:
            raise ValueError(f"depth must be >= 0, got {depth}")

        # Embed the query. Providers return a (1, D) matrix even for
        # single texts, so index row 0 to get the 1-D query vector.
        query_matrix = self.system.embedder.get_embeddings([current_task_description])
        query = query_matrix[0]

        # Find the closest nodes in the graph.
        nearest = self.system.storage.get_nearest_nodes(query, top_k=top_k)
        if not nearest:
            return ""

        root_ids = [n["id"] for n in nearest]
        subgraph = self.system.storage.get_subgraph(root_ids, depth=depth)

        # ADR-003 F5: cap the rendered subgraph at a fixed node count
        # so depth-3 renders never blow up on pathological graphs. The
        # top-k seeds are protected (never dropped); other nodes are
        # dropped in reverse-priority order (farthest from a seed
        # first). Mutates subgraph in place.
        _enforce_node_cap(subgraph, seed_ids=root_ids, cap=_NODE_CAP)

        # ADR-002: lazily name any placeholder centroids in the rendered
        # subgraph. The dream-cycle clustering loop writes centroids
        # with placeholder labels (cluster_<hex>); this is where they
        # get LLM-generated semantic ones — but ONLY for centroids
        # actually appearing in this query's render. Mutates
        # ``subgraph["nodes"]`` in place. NEVER raises.
        self.system.resolve_centroid_names(subgraph["nodes"])

        # ADR-003 D2: lazily populate per-centroid paragraph summaries
        # for centroids whose trunk-pass eager summarisation didn't
        # cover (deeper levels). Same lazy-cache pattern: batched LLM
        # call, persist, mutate in place. NEVER raises.
        self.system.resolve_junction_summaries(subgraph["nodes"])

        # If the subgraph has no memories AND no node hierarchy above
        # the roots, the tree is degenerate. Return empty rather than
        # emit a header with no content.
        if not subgraph.get("nodes") and not subgraph.get("memories"):
            return ""

        return _render_ascii_tree(subgraph)


def _enforce_node_cap(
    subgraph: dict[str, Any],
    *,
    seed_ids: list[str],
    cap: int,
) -> None:
    """Drop far-from-seed nodes until the subgraph has ≤ ``cap`` nodes.

    Priority order for retention:
      1. Seed nodes (original top_k retrieved) — NEVER dropped.
      2. Remaining nodes, ranked by a simple BFS distance from any
         seed (closer = higher priority).

    Mutates ``subgraph`` in place. Also prunes ``edges`` referring to
    dropped nodes and ``memories`` that are only reachable via
    dropped nodes (via has_tag edges).

    No-op when the subgraph already has ≤ ``cap`` nodes.
    """
    nodes = subgraph.get("nodes", [])
    if len(nodes) <= cap:
        return

    nodes_by_id = {n["id"]: n for n in nodes}
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in subgraph.get("edges", []):
        if edge.get("relationship") != "child_of":
            continue
        src = edge.get("source_id")
        tgt = edge.get("target_id")
        if src in nodes_by_id and tgt in nodes_by_id:
            adjacency[src].add(tgt)
            adjacency[tgt].add(src)

    # BFS distance from the seed set. Unreachable nodes get infinity.
    distance: dict[str, int] = {sid: 0 for sid in seed_ids if sid in nodes_by_id}
    frontier = list(distance.keys())
    while frontier:
        next_frontier: list[str] = []
        for nid in frontier:
            for neighbour in adjacency[nid]:
                if neighbour not in distance:
                    distance[neighbour] = distance[nid] + 1
                    next_frontier.append(neighbour)
        frontier = next_frontier

    # Rank: closer first, unreachable last. Stable by id for
    # determinism.
    def _priority(node_id: str) -> tuple[int, str]:
        return (distance.get(node_id, 10_000), node_id)

    # Seed nodes are contractually NEVER dropped, even when cap is
    # pathologically small (e.g. caller passed top_k=100 with the
    # hard-coded cap of 80). Clamp the effective cap to at least the
    # seed count so the sorted-slice keeps all of them. The seeds
    # sort first (distance=0) so they occupy the head of the slice
    # regardless — the clamp just ensures the slice is long enough.
    seed_ids_in_graph = [sid for sid in seed_ids if sid in nodes_by_id]
    effective_cap = max(cap, len(seed_ids_in_graph))
    kept_ids = set(sorted(nodes_by_id.keys(), key=_priority)[:effective_cap])
    subgraph["nodes"] = [n for n in nodes if n["id"] in kept_ids]
    subgraph["edges"] = [
        e
        for e in subgraph.get("edges", [])
        if (
            # Keep child_of edges only when both endpoints are kept.
            e.get("relationship") == "child_of"
            and e.get("source_id") in kept_ids
            and e.get("target_id") in kept_ids
        )
        or (
            # Keep has_tag edges only when the target (node) is kept.
            # Memory source is a memory id, not a node id.
            e.get("relationship") == "has_tag" and e.get("target_id") in kept_ids
        )
    ]
    # Drop memories no longer reachable via kept has_tag edges.
    reachable_memory_ids = {
        e.get("source_id") for e in subgraph["edges"] if e.get("relationship") == "has_tag"
    }
    subgraph["memories"] = [
        m for m in subgraph.get("memories", []) if m["id"] in reachable_memory_ids
    ]


def _render_ascii_tree(subgraph: dict[str, Any]) -> str:
    """Render a subgraph dict into a multi-line ASCII tree string.

    Input shape (per ``StorageAdapter.get_subgraph``)::

        {
          "nodes": [{"id": ..., "label": ..., "is_centroid": ...}, ...],
          "edges": [{"source_id": ..., "target_id": ...,
                     "weight": ..., "relationship": "child_of"|"has_tag"},
                    ...],
          "memories": [{"id": ..., "summary": ..., ...}, ...],
        }

    Output shape::

        Relevant Memory Context:
        📁 Databases
        ├── 📁 SQLite
        │   ├── 📄 mem_<uuid>: "WAL mode discussion..."
        │   └── 📄 mem_<uuid>: "Massive inserts perf..."
        └── 📁 Indexing
            └── 📄 mem_<uuid>: "Composite indexes..."

    Multi-parent handling: if a node is reachable from more than one
    centroid parent, the FIRST encountered parent gets the full
    subtree; later mentions render as a compact
    ``(also under: <first-parent-label>)`` note without recursing.

    Returns the empty string if the subgraph has no nodes at all.
    """
    nodes_by_id: dict[str, dict] = {n["id"]: n for n in subgraph.get("nodes", [])}
    memories_by_id: dict[str, dict] = {m["id"]: m for m in subgraph.get("memories", [])}
    edges = subgraph.get("edges", [])

    if not nodes_by_id:
        return ""

    # Build adjacency:
    #   children_of[parent_id] = list of child node ids   (via child_of edges)
    #   parents_of[child_id]   = list of parent node ids  (reverse)
    #   memories_of[node_id]   = list of memory ids       (via has_tag edges)
    #   tags_of_memory[mem_id] = list of tag labels       (reverse has_tag,
    #                                                       used for inline
    #                                                       tag rendering)
    children_of: dict[str, list[str]] = defaultdict(list)
    parents_of: dict[str, list[str]] = defaultdict(list)
    memories_of: dict[str, list[str]] = defaultdict(list)
    tags_of_memory: dict[str, list[str]] = defaultdict(list)

    for edge in edges:
        rel = edge.get("relationship")
        src = edge.get("source_id")
        tgt = edge.get("target_id")
        if rel == "child_of":
            children_of[src].append(tgt)
            parents_of[tgt].append(src)
        elif rel == "has_tag":
            # has_tag source is a memory id; target is a node id.
            memories_of[tgt].append(src)
            node = nodes_by_id.get(tgt)
            if node is not None:
                tags_of_memory[src].append(node["label"])

    # Find root nodes: those with no incoming child_of edges within
    # the subgraph. These are the entry points for tree rendering.
    all_node_ids = set(nodes_by_id.keys())
    ids_with_parent_in_subgraph = {
        child for children in children_of.values() for child in children if child in all_node_ids
    }
    root_ids = sorted(all_node_ids - ids_with_parent_in_subgraph)

    if not root_ids:
        # Every node has a parent in the subgraph — shouldn't happen
        # for a tree, but if the data is cyclic we bail gracefully.
        logger.warning("_render_ascii_tree: no roots found; subgraph has cycles?")
        return ""

    lines: list[str] = ["Relevant Memory Context:"]
    # Tracks which nodes have already been fully rendered (so their
    # second/later appearances become compact "(also under: ...)"
    # references). Maps node_id → parent label under which the node
    # was FIRST rendered. A value of ``None`` means the first render
    # was as a top-level root (no parent). Mutated by _render_root
    # and _render_descendant.
    first_parent_label: dict[str, str | None] = {}

    # Render every root node. Roots have no parent so they sit at
    # the top level with no connector — the connector logic only
    # kicks in for descendants.
    for root_id in root_ids:
        _render_root(
            nodes_by_id=nodes_by_id,
            node_id=root_id,
            lines=lines,
            first_parent_label=first_parent_label,
            children_of=children_of,
            memories_of=memories_of,
            memories_by_id=memories_by_id,
            tags_of_memory=tags_of_memory,
        )

    return "\n".join(lines) + "\n"


def _emit_memory_row(
    mem: dict,
    mem_id: str,
    lines: list[str],
    connector_prefix: str,
    is_last_item: bool,
    tags_of_memory: dict[str, list[str]],
) -> None:
    """Emit a memory line plus optional annotation continuation line.

    Shared between ``_render_root`` and ``_render_descendant`` — both
    render memories identically given the right connector prefix.
    Extracted to keep each render function's McCabe complexity below
    the project's C901 = 10 ceiling.

    - ``connector_prefix`` is the tree-prefix that comes BEFORE the
      ``├── ``/``└── ``. Root memories pass ``""`` (no indent);
      descendants pass the accumulated ``child_prefix``.
    - ``is_last_item`` picks the connector shape: last item uses
      ``└── `` and a blank continuation gap, non-last uses ``├── ``
      and ``│   `` continuation.
    """
    connector = "└── " if is_last_item else "├── "
    snippet = _summary_snippet(mem.get("summary") or "")
    lines.append(f'{connector_prefix}{connector}📄 {mem_id}: "{snippet}"')
    annotation = _format_annotation_line(mem, tags_of_memory)
    if annotation is not None:
        cont_gap = "    " if is_last_item else "│   "
        lines.append(f"{connector_prefix}{cont_gap}{annotation}")


def _format_annotation_line(
    mem: dict,
    tags_of_memory: dict[str, list[str]],
) -> str | None:
    """Build the ``tags: … │ entities: …`` sub-line for a memory, or
    ``None`` if the memory has neither — in which case the caller
    should skip emitting an annotation row at all.

    Tag order: the labels are already in the order the ``has_tag``
    edges were walked, which for SQLiteAdapter is undefined. Dedupe
    while preserving first-seen order so the output is deterministic
    across runs.
    """
    raw_tags = tags_of_memory.get(mem["id"], [])
    seen: set[str] = set()
    tags: list[str] = []
    for label in raw_tags:
        if label not in seen:
            seen.add(label)
            tags.append(label)

    entities = list(mem.get("named_entities") or [])

    parts: list[str] = []
    if tags:
        parts.append(f"tags: {', '.join(tags)}")
    if entities:
        parts.append(f"entities: {', '.join(entities)}")

    if not parts:
        return None
    return " · ".join(parts)


def _format_also_under(first_parent_label_value: str | None) -> str:
    """Format the '(also under: X)' suffix for a node seen a second time.

    A value of ``None`` in ``first_parent_label`` means the node was
    first rendered as a top-level root. Both branches produce
    human-readable output — no nonsensical ``(also under: NodeOwnLabel)``.
    """
    if first_parent_label_value is None:
        return " (also under: top level)"
    return f" (also under: {first_parent_label_value})"


def _emit_paragraph_summary_line(
    node: dict,
    lines: list[str],
    connector_prefix: str,
    is_last_sibling: bool,
) -> None:
    """Emit a ``para: "..."`` continuation row just after a centroid's
    folder line, if the centroid has a ``paragraph_summary`` (ADR-003).

    Matches the memory annotation shape in :func:`_emit_memory_row`:
    the row uses ``connector_prefix`` (the folder's own prefix) plus
    a ``cont_gap`` that extends the tree art down to the paragraph.
    ``cont_gap`` is ``│   `` when the folder has siblings still to
    come below it, and ``    `` when the folder was the last sibling.

    No-op when the centroid has no summary — the tree still renders
    just the label, preserving the pre-ADR-003 shape.
    """
    summary = (node.get("paragraph_summary") or "").strip()
    if not summary:
        return
    if len(summary) > _JUNCTION_SUMMARY_MAX_CHARS:
        summary = summary[: _JUNCTION_SUMMARY_MAX_CHARS - 1] + "…"
    cont_gap = "    " if is_last_sibling else "│   "
    lines.append(f'{connector_prefix}{cont_gap}para: "{summary}"')


def _render_root(
    nodes_by_id: dict[str, dict],
    node_id: str,
    lines: list[str],
    first_parent_label: dict[str, str | None],
    children_of: dict[str, list[str]],
    memories_of: dict[str, list[str]],
    memories_by_id: dict[str, dict],
    tags_of_memory: dict[str, list[str]],
) -> None:
    """Render a root node (no indent) and recursively descend."""
    node = nodes_by_id.get(node_id)
    if node is None:
        return

    if node_id in first_parent_label:
        # Already rendered elsewhere — compact reference only.
        suffix = _format_also_under(first_parent_label[node_id])
        lines.append(f"📁 {node['label']}{suffix}")
        return

    # First time we render this node, and it's a root — no parent.
    # Store ``None`` as the sentinel; _format_also_under handles it.
    first_parent_label[node_id] = None
    lines.append(f"📁 {node['label']}")

    sub_children = children_of.get(node_id, [])
    raw_mem_ids = memories_of.get(node_id, [])

    seen_mems: set[str] = set()
    mem_ids: list[str] = []
    for mid in raw_mem_ids:
        if mid not in seen_mems:
            seen_mems.add(mid)
            mem_ids.append(mid)

    items: list[tuple[str, str]] = []
    for cid in sub_children:
        items.append(("node", cid))
    for mid in mem_ids:
        items.append(("memory", mid))

    # Emit the centroid's paragraph summary (ADR-003 D2) as a
    # continuation row just below the folder line, before children.
    # Roots have no sibling tree-art above them, so render the gap
    # as blank (the paragraph is visually "under" the root, not
    # indented for a deeper level).
    _emit_paragraph_summary_line(
        node=node,
        lines=lines,
        connector_prefix="",
        is_last_sibling=True,
    )

    for i, (kind, item_id) in enumerate(items):
        is_last_item = i == len(items) - 1
        if kind == "node":
            _render_descendant(
                nodes_by_id=nodes_by_id,
                node_id=item_id,
                prefix="",
                is_last_sibling=is_last_item,
                parent_label=node["label"],
                lines=lines,
                first_parent_label=first_parent_label,
                children_of=children_of,
                memories_of=memories_of,
                memories_by_id=memories_by_id,
                tags_of_memory=tags_of_memory,
            )
        else:
            mem = memories_by_id.get(item_id)
            if mem is None:
                continue
            _emit_memory_row(
                mem=mem,
                mem_id=item_id,
                lines=lines,
                connector_prefix="",
                is_last_item=is_last_item,
                tags_of_memory=tags_of_memory,
            )


def _render_descendant(
    nodes_by_id: dict[str, dict],
    node_id: str,
    prefix: str,
    is_last_sibling: bool,
    parent_label: str,
    lines: list[str],
    first_parent_label: dict[str, str | None],
    children_of: dict[str, list[str]],
    memories_of: dict[str, list[str]],
    memories_by_id: dict[str, dict],
    tags_of_memory: dict[str, list[str]],
) -> None:
    """Render a non-root node with its tree prefix + connector."""
    node = nodes_by_id.get(node_id)
    if node is None:
        return

    connector = "└── " if is_last_sibling else "├── "

    if node_id in first_parent_label:
        # Already rendered under a different parent — compact reference.
        suffix = _format_also_under(first_parent_label[node_id])
        lines.append(f"{prefix}{connector}📁 {node['label']}{suffix}")
        return

    first_parent_label[node_id] = parent_label
    lines.append(f"{prefix}{connector}📁 {node['label']}")

    sub_children = children_of.get(node_id, [])
    raw_mem_ids = memories_of.get(node_id, [])

    seen_mems: set[str] = set()
    mem_ids: list[str] = []
    for mid in raw_mem_ids:
        if mid not in seen_mems:
            seen_mems.add(mid)
            mem_ids.append(mid)

    items: list[tuple[str, str]] = []
    for cid in sub_children:
        items.append(("node", cid))
    for mid in mem_ids:
        items.append(("memory", mid))

    child_prefix = prefix + ("    " if is_last_sibling else "│   ")

    # Emit the centroid's paragraph summary (ADR-003 D2) as a
    # continuation row just below the folder line, before children.
    # Mirrors the memory annotation pattern: use the folder's prefix
    # plus a cont_gap driven by whether the folder has siblings below.
    _emit_paragraph_summary_line(
        node=node,
        lines=lines,
        connector_prefix=prefix,
        is_last_sibling=is_last_sibling,
    )

    for i, (kind, item_id) in enumerate(items):
        is_last_item = i == len(items) - 1
        if kind == "node":
            _render_descendant(
                nodes_by_id=nodes_by_id,
                node_id=item_id,
                prefix=child_prefix,
                is_last_sibling=is_last_item,
                parent_label=node["label"],
                lines=lines,
                first_parent_label=first_parent_label,
                children_of=children_of,
                memories_of=memories_of,
                memories_by_id=memories_by_id,
                tags_of_memory=tags_of_memory,
            )
        else:
            mem = memories_by_id.get(item_id)
            if mem is None:
                continue
            _emit_memory_row(
                mem=mem,
                mem_id=item_id,
                lines=lines,
                connector_prefix=child_prefix,
                is_last_item=is_last_item,
                tags_of_memory=tags_of_memory,
            )


def _summary_snippet(summary: str) -> str:
    """Truncate a memory summary for inline display in the tree."""
    if len(summary) <= _MEMORY_SNIPPET_MAX_CHARS:
        return summary
    return summary[: _MEMORY_SNIPPET_MAX_CHARS - 1] + "…"
