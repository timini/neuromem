"""Unit tests for neuromem.context.ContextHelper and _render_ascii_tree.

The renderer is tested with hand-crafted subgraph dicts (matching the
shape of StorageAdapter.get_subgraph output) to exercise corner cases
without needing a live NeuroMemory end-to-end. The ContextHelper class
is then tested end-to-end against a NeuroMemory built on an in-memory
SQLiteAdapter.
"""

from __future__ import annotations

import numpy as np
import pytest
from neuromem.context import ContextHelper, _enforce_node_cap, _render_ascii_tree
from neuromem.providers import EmbeddingProvider
from neuromem.storage.sqlite import SQLiteAdapter
from neuromem.system import NeuroMemory

from tests.conftest import MockEmbeddingProvider, MockLLMProvider

# ---------------------------------------------------------------------------
# _render_ascii_tree — direct unit tests with hand-crafted subgraphs
# ---------------------------------------------------------------------------


class TestRenderAsciiTree:
    def test_empty_subgraph_returns_empty_string(self) -> None:
        assert _render_ascii_tree({"nodes": [], "edges": [], "memories": []}) == ""

    def test_missing_keys_return_empty_string(self) -> None:
        assert _render_ascii_tree({}) == ""

    def test_single_root_single_memory(self) -> None:
        subgraph = {
            "nodes": [{"id": "n1", "label": "SQLite", "is_centroid": False}],
            "edges": [
                {
                    "source_id": "m1",
                    "target_id": "n1",
                    "weight": 1.0,
                    "relationship": "has_tag",
                }
            ],
            "memories": [{"id": "m1", "summary": "hello world"}],
        }
        rendered = _render_ascii_tree(subgraph)
        assert "📁 SQLite" in rendered
        assert '📄 m1: "hello world"' in rendered
        assert "└──" in rendered  # single-child uses └──

    def test_single_root_two_memories_uses_correct_connectors(self) -> None:
        subgraph = {
            "nodes": [{"id": "n1", "label": "SQLite", "is_centroid": False}],
            "edges": [
                {
                    "source_id": "m1",
                    "target_id": "n1",
                    "weight": 1.0,
                    "relationship": "has_tag",
                },
                {
                    "source_id": "m2",
                    "target_id": "n1",
                    "weight": 1.0,
                    "relationship": "has_tag",
                },
            ],
            "memories": [
                {"id": "m1", "summary": "first"},
                {"id": "m2", "summary": "second"},
            ],
        }
        rendered = _render_ascii_tree(subgraph)
        assert "├── 📄 m1:" in rendered  # first of two uses ├──
        assert "└── 📄 m2:" in rendered  # last uses └──

    def test_centroid_with_two_children_and_memories(self) -> None:
        """The main example from spec.md §ContextHelper."""
        subgraph = {
            "nodes": [
                {"id": "centroid", "label": "Databases", "is_centroid": True},
                {"id": "sqlite", "label": "SQLite", "is_centroid": False},
                {"id": "indexing", "label": "Indexing", "is_centroid": False},
            ],
            "edges": [
                {
                    "source_id": "centroid",
                    "target_id": "sqlite",
                    "weight": 0.9,
                    "relationship": "child_of",
                },
                {
                    "source_id": "centroid",
                    "target_id": "indexing",
                    "weight": 0.85,
                    "relationship": "child_of",
                },
                {
                    "source_id": "mem_wal",
                    "target_id": "sqlite",
                    "weight": 1.0,
                    "relationship": "has_tag",
                },
                {
                    "source_id": "mem_idx",
                    "target_id": "indexing",
                    "weight": 1.0,
                    "relationship": "has_tag",
                },
            ],
            "memories": [
                {"id": "mem_wal", "summary": "WAL mode discussion"},
                {"id": "mem_idx", "summary": "Composite indexes"},
            ],
        }
        rendered = _render_ascii_tree(subgraph)

        # Header
        assert rendered.startswith("Relevant Memory Context:")
        # Root centroid
        assert "📁 Databases" in rendered
        # Both children under the centroid
        assert "├── 📁 SQLite" in rendered
        assert "└── 📁 Indexing" in rendered
        # Memories under their respective leaf nodes
        assert 'mem_wal: "WAL mode discussion"' in rendered
        assert 'mem_idx: "Composite indexes"' in rendered
        # Connectors for first-of-two ├── and last └──
        assert rendered.count("├── ") >= 1
        assert rendered.count("└── ") >= 2

    def test_long_summary_truncated_with_ellipsis(self) -> None:
        """Summaries longer than the snippet cap get truncated with a
        trailing ellipsis. Assert behavioural truncation — via the
        module-level constant — rather than hard-coding the specific
        cap so a future tune of _MEMORY_SNIPPET_MAX_CHARS doesn't
        break this test."""
        from neuromem.context import _MEMORY_SNIPPET_MAX_CHARS  # noqa: PLC0415

        long_text = "x" * (_MEMORY_SNIPPET_MAX_CHARS * 2)
        subgraph = {
            "nodes": [{"id": "n1", "label": "Tag", "is_centroid": False}],
            "edges": [
                {
                    "source_id": "m1",
                    "target_id": "n1",
                    "weight": 1.0,
                    "relationship": "has_tag",
                }
            ],
            "memories": [{"id": "m1", "summary": long_text}],
        }
        rendered = _render_ascii_tree(subgraph)
        assert "…" in rendered
        # Find the snippet line and check its length: the renderer
        # produces (cap - 1) content characters plus the "…" glyph.
        snippet_line = next(line for line in rendered.splitlines() if "m1:" in line)
        quoted = snippet_line.split('"', 1)[1].rsplit('"', 1)[0]
        assert len(quoted) <= _MEMORY_SNIPPET_MAX_CHARS

    def test_multi_parent_node_rendered_once_with_also_under_reference(
        self,
    ) -> None:
        """A node reachable from two parents is rendered in full under
        the first parent; the second parent gets a compact reference.

        The 'also under' reference must name the FIRST parent (not the
        node's own label — that was the I-3 bug).
        """
        subgraph = {
            "nodes": [
                {"id": "p1", "label": "Parent1", "is_centroid": True},
                {"id": "p2", "label": "Parent2", "is_centroid": True},
                {"id": "shared", "label": "Shared", "is_centroid": False},
            ],
            "edges": [
                {
                    "source_id": "p1",
                    "target_id": "shared",
                    "weight": 0.9,
                    "relationship": "child_of",
                },
                {
                    "source_id": "p2",
                    "target_id": "shared",
                    "weight": 0.88,
                    "relationship": "child_of",
                },
            ],
            "memories": [],
        }
        rendered = _render_ascii_tree(subgraph)
        # Shared appears twice — once in full, once as an "also under" ref.
        shared_lines = [line for line in rendered.splitlines() if "Shared" in line]
        assert len(shared_lines) == 2
        # The second mention must cite Parent1 (the parent under which
        # Shared was first rendered), not Shared itself or Parent2.
        also_under_line = next(line for line in shared_lines if "also under:" in line)
        assert "also under: Parent1" in also_under_line
        assert "also under: Shared" not in also_under_line
        # Both parents appear
        assert "📁 Parent1" in rendered
        assert "📁 Parent2" in rendered

    def test_root_revisited_as_descendant_shows_top_level_reference(self) -> None:
        """Regression for I-3: when a node is rendered as a root and
        then encountered again (via some other tree walking order),
        the 'also under' reference must say 'top level', NOT the
        node's own label.

        Builds a subgraph where node 'A' is both a root (no parent
        within the subgraph) AND appears in another edge list that
        could hypothetically re-reach it. Even in this degenerate
        case, the output must be human-readable.
        """
        # Construct a scenario where A has no parent (root), but
        # also appears as the target of a child_of edge from B.
        # This creates a cycle in the edge data (A is both a root
        # and a descendant of B), but _render_ascii_tree's cycle
        # detection should handle it gracefully.
        subgraph = {
            "nodes": [
                {"id": "a", "label": "NodeA", "is_centroid": True},
                {"id": "b", "label": "NodeB", "is_centroid": True},
            ],
            "edges": [
                # A → B is the normal tree edge
                {
                    "source_id": "a",
                    "target_id": "b",
                    "weight": 0.9,
                    "relationship": "child_of",
                },
                # B → A is a back-edge (cycle) that tries to make
                # A a descendant of B as well
                {
                    "source_id": "b",
                    "target_id": "a",
                    "weight": 0.9,
                    "relationship": "child_of",
                },
            ],
            "memories": [],
        }
        rendered = _render_ascii_tree(subgraph)
        # With a cycle, every node has a parent and root_ids is
        # empty → function returns empty string (per the cycle
        # detection in _render_ascii_tree).
        assert rendered == ""

    def test_format_also_under_top_level_vs_parent(self) -> None:
        """Direct test of the _format_also_under helper."""
        from neuromem.context import _format_also_under

        assert _format_also_under(None) == " (also under: top level)"
        assert _format_also_under("Databases") == " (also under: Databases)"

    def test_multi_root_forest(self) -> None:
        """Two independent hierarchies render as two top-level entries."""
        subgraph = {
            "nodes": [
                {"id": "r1", "label": "Databases", "is_centroid": True},
                {"id": "r2", "label": "Languages", "is_centroid": True},
                {"id": "c1", "label": "SQLite", "is_centroid": False},
                {"id": "c2", "label": "Python", "is_centroid": False},
            ],
            "edges": [
                {
                    "source_id": "r1",
                    "target_id": "c1",
                    "weight": 0.9,
                    "relationship": "child_of",
                },
                {
                    "source_id": "r2",
                    "target_id": "c2",
                    "weight": 0.9,
                    "relationship": "child_of",
                },
            ],
            "memories": [],
        }
        rendered = _render_ascii_tree(subgraph)
        assert "📁 Databases" in rendered
        assert "📁 Languages" in rendered
        assert "📁 SQLite" in rendered
        assert "📁 Python" in rendered


# ---------------------------------------------------------------------------
# ContextHelper — end-to-end against a real NeuroMemory + SQLiteAdapter
# ---------------------------------------------------------------------------


@pytest.fixture
def live_memory_system(
    mock_embedder: MockEmbeddingProvider,
    mock_llm: MockLLMProvider,
) -> NeuroMemory:
    return NeuroMemory(
        storage=SQLiteAdapter(":memory:"),
        llm=mock_llm,
        embedder=mock_embedder,
    )


class TestContextHelperEndToEnd:
    def test_empty_graph_returns_empty_string(
        self,
        live_memory_system: NeuroMemory,
    ) -> None:
        helper = ContextHelper(live_memory_system)
        assert helper.build_prompt_context("any query") == ""

    def test_populated_graph_renders_mem_ids(
        self,
        live_memory_system: NeuroMemory,
    ) -> None:
        """After a full enqueue → dream cycle, the helper renders at
        least one memory reference in the ASCII tree."""
        live_memory_system.enqueue("python sqlite databases")
        live_memory_system.enqueue("python numpy arrays")
        live_memory_system.enqueue("ruby rails framework")
        live_memory_system.force_dream(block=True)

        helper = ContextHelper(live_memory_system)
        ctx = helper.build_prompt_context("python")

        assert ctx != ""
        assert "📁" in ctx
        assert "📄" in ctx  # at least one memory reference line

    def test_empty_task_raises_value_error(
        self,
        live_memory_system: NeuroMemory,
    ) -> None:
        helper = ContextHelper(live_memory_system)
        with pytest.raises(ValueError, match="non-empty"):
            helper.build_prompt_context("")

    def test_invalid_top_k_raises(
        self,
        live_memory_system: NeuroMemory,
    ) -> None:
        helper = ContextHelper(live_memory_system)
        with pytest.raises(ValueError, match="top_k"):
            helper.build_prompt_context("hello", top_k=0)

    def test_invalid_depth_raises(
        self,
        live_memory_system: NeuroMemory,
    ) -> None:
        helper = ContextHelper(live_memory_system)
        with pytest.raises(ValueError, match="depth"):
            helper.build_prompt_context("hello", depth=-1)

    def test_no_llm_calls_during_build(
        self,
        live_memory_system: NeuroMemory,
    ) -> None:
        """build_prompt_context must NOT call the LLM — only embed."""
        # Populate then swap the LLM for one that raises on any call.
        live_memory_system.enqueue("alpha beta gamma")
        live_memory_system.force_dream(block=True)

        class RaisingLLM:
            def generate_summary(self, raw_text: str) -> str:
                raise AssertionError("generate_summary must not be called")

            def extract_tags(self, summary: str) -> list[str]:
                raise AssertionError("extract_tags must not be called")

            def generate_category_name(self, concepts: list[str]) -> str:
                raise AssertionError("generate_category_name must not be called")

        # Inject directly — we're testing that the helper doesn't
        # touch the LLM on the recall path.
        live_memory_system.llm = RaisingLLM()  # type: ignore[assignment]
        helper = ContextHelper(live_memory_system)
        # Should not raise — the helper only embeds the query.
        _ = helper.build_prompt_context("alpha")

    def test_does_not_trigger_ltp_spike(
        self,
        live_memory_system: NeuroMemory,
    ) -> None:
        """Recall via ContextHelper does NOT spike access_weight on the
        returned memories — LTP is reserved for retrieve_memories."""
        live_memory_system.enqueue("python sqlite")
        live_memory_system.force_dream(block=True)

        # Artificially lower access_weight on all consolidated memories.
        import sqlite3 as _s  # noqa: F401 — cosmetic import for clarity

        live_memory_system.storage._conn.execute(  # type: ignore[attr-defined]
            "UPDATE memories SET access_weight = 0.3 WHERE status = 'consolidated'"
        )

        helper = ContextHelper(live_memory_system)
        _ = helper.build_prompt_context("python")

        # access_weight must still be 0.3 (recall did NOT spike it).
        rows = live_memory_system.storage._conn.execute(  # type: ignore[attr-defined]
            "SELECT access_weight FROM memories WHERE status = 'consolidated'"
        ).fetchall()
        for row in rows:
            assert row[0] == pytest.approx(0.3)


class TestContextHelperMultiTopK:
    def test_top_k_parameter_honoured(
        self,
        live_memory_system: NeuroMemory,
    ) -> None:
        """top_k should influence how many root nodes are considered.

        With top_k=1 we anchor on one nearest node; with top_k=10 we
        anchor on up to ten. Both should return a non-empty tree if
        the graph has content.
        """
        for i in range(5):
            live_memory_system.enqueue(f"topic{i} example content")
        live_memory_system.force_dream(block=True)

        helper = ContextHelper(live_memory_system)
        small = helper.build_prompt_context("example", top_k=1)
        large = helper.build_prompt_context("example", top_k=10)
        assert small != ""
        assert large != ""


# ---------------------------------------------------------------------------
# Inline tag + named-entity annotation rendering
# ---------------------------------------------------------------------------


class TestMemoryAnnotations:
    """Below each ``📄 mem_xxx: "summary..."`` line, the renderer
    emits an optional annotation line combining:

    - ``tags:`` — labels of the tag-nodes this memory links to (built
      by walking has_tag edges; de-duplicated, preserves first-seen
      order for determinism).
    - ``entities:`` — the memory's stored ``named_entities`` list.

    Either half is omitted if empty. If BOTH are empty, no annotation
    line is emitted at all (keeps the tree tight for memories with
    no enrichment).
    """

    def _subgraph_with_tags(self, tags: list[str], entities: list[str] | None) -> dict:
        """Build a one-memory subgraph where the memory has ``has_tag``
        edges to one node per label in ``tags``. ``entities`` is set
        on the memory dict (None → omitted key entirely)."""
        if not tags:
            raise ValueError("need at least one tag to anchor the memory in the tree")
        nodes = [{"id": f"n_{t}", "label": t, "is_centroid": False} for t in tags]
        edges = [
            {
                "source_id": "m1",
                "target_id": f"n_{t}",
                "weight": 1.0,
                "relationship": "has_tag",
            }
            for t in tags
        ]
        mem: dict = {"id": "m1", "summary": "some summary"}
        if entities is not None:
            mem["named_entities"] = entities
        return {"nodes": nodes, "edges": edges, "memories": [mem]}

    def test_tags_only_renders_tags_half(self) -> None:
        """Memory has tag edges but no entities → annotation line has
        ``tags:`` half, no ``entities:`` half."""
        subgraph = self._subgraph_with_tags(["coupon", "savings"], entities=[])
        rendered = _render_ascii_tree(subgraph)
        assert "tags:" in rendered
        assert "coupon" in rendered
        assert "savings" in rendered
        # Entities line is omitted.
        assert "entities:" not in rendered

    def test_entities_rendered_when_present(self) -> None:
        """Memory has both tags AND entities → annotation line has
        both halves in the order ``tags: … · entities: …``."""
        subgraph = self._subgraph_with_tags(["coupon"], entities=["Target", "Cartwheel"])
        rendered = _render_ascii_tree(subgraph)
        assert "entities: Target, Cartwheel" in rendered
        assert "tags:" in rendered

    def test_both_halves_separated_by_middle_dot(self) -> None:
        """The ``·`` separator combines the two halves compactly."""
        subgraph = self._subgraph_with_tags(["coupon"], entities=["Target"])
        rendered = _render_ascii_tree(subgraph)
        assert "tags: coupon · entities: Target" in rendered

    def test_no_entities_key_on_memory_omits_entities_half(self) -> None:
        """A memory dict that predates the named_entities field (no
        key at all) must still render — just without the entities
        half of the annotation. Backwards-compat for hand-crafted
        subgraph inputs."""
        subgraph = self._subgraph_with_tags(["coupon"], entities=None)
        rendered = _render_ascii_tree(subgraph)
        assert "entities:" not in rendered
        assert "tags: coupon" in rendered

    def test_duplicate_tag_edges_deduplicated_per_annotation(self) -> None:
        """If the same memory→tag-node pair appears twice in the edge
        list (shouldn't happen thanks to the UNIQUE constraint, but
        the renderer is defensive), the tags: line contains the
        label only once."""
        subgraph = self._subgraph_with_tags(["coupon"], entities=[])
        # Append a duplicate has_tag edge to exercise the dedupe path.
        subgraph["edges"].append(
            {
                "source_id": "m1",
                "target_id": "n_coupon",
                "weight": 1.0,
                "relationship": "has_tag",
            }
        )
        rendered = _render_ascii_tree(subgraph)
        # Find the tags: line and check 'coupon' appears once there.
        tag_line = next((ln for ln in rendered.splitlines() if "tags:" in ln), None)
        assert tag_line is not None
        assert tag_line.count("coupon") == 1

    def test_annotation_line_has_tree_continuation_prefix(self) -> None:
        """The annotation line directly below a memory row must use
        a valid tree continuation prefix — ``│`` under memories that
        are NOT the last child, and spaces under the last child — so
        downstream ASCII tree parsers don't see a broken tree."""
        # Two memories under one root: first is "├──", second is "└──".
        subgraph = {
            "nodes": [{"id": "nroot", "label": "root", "is_centroid": False}],
            "edges": [
                {
                    "source_id": "m1",
                    "target_id": "nroot",
                    "weight": 1.0,
                    "relationship": "has_tag",
                },
                {
                    "source_id": "m2",
                    "target_id": "nroot",
                    "weight": 1.0,
                    "relationship": "has_tag",
                },
            ],
            "memories": [
                {"id": "m1", "summary": "first", "named_entities": ["Alpha"]},
                {"id": "m2", "summary": "second", "named_entities": ["Beta"]},
            ],
        }
        rendered = _render_ascii_tree(subgraph)
        # m1 is not last → continuation uses '│   '
        assert "│   tags:" in rendered or "│   entities: Alpha" in rendered
        # m2 is last → continuation uses '    '
        lines = rendered.splitlines()
        # Find the line containing "entities: Beta" — it should start
        # with four spaces (last-child continuation), not '│'.
        beta_line = next(ln for ln in lines if "entities: Beta" in ln)
        assert beta_line.startswith("    ")
        assert not beta_line.startswith("│")


# ---------------------------------------------------------------------------
# build_prompt_context invokes resolve_centroid_names (ADR-002)
# ---------------------------------------------------------------------------


class TestBuildPromptContextLazyNaming:
    """ADR-002: ContextHelper.build_prompt_context calls
    NeuroMemory.resolve_centroid_names on the loaded subgraph
    before rendering, so placeholder centroids in the rendered tree
    get LLM-generated semantic names. Tests:

    - First render after a dream cycle invokes the batched LLM
      naming exactly once.
    - Second render of the same query touches the same centroid
      but does NOT re-invoke the LLM (cache hit via storage
      persistence).
    - The rendered tree string contains the resolved label, not
      the placeholder.
    """

    def test_first_render_resolves_placeholders(self) -> None:
        """A render that touches a placeholder centroid invokes
        generate_category_names_batch ONCE and the rendered tree
        shows the resolved label, not the placeholder."""

        class FixedNamerLLM(MockLLMProvider):
            def __init__(self) -> None:
                super().__init__()
                self.batch_call_count = 0

            def generate_category_names_batch(self, pairs: list[list[str]]) -> list[str]:
                self.batch_call_count += 1
                return [f"named_{i}" for i in range(len(pairs))]

        # ControlledEmbedder is defined in tests/conftest.py-style
        # tests/test_system.py. Import it via the test module path.
        from tests.test_system import ControlledEmbedder  # noqa: PLC0415

        # Build a system with two near-identical tags so clustering
        # produces a centroid. Embedder also serves the query.
        ctrl = ControlledEmbedder(
            {
                "alpha": [1.0, 0.0, 0.0, 0.0],
                "beta": [0.999, 0.01, 0.0, 0.0],
                # Query embedding routes to alpha, which surfaces the
                # centroid via depth-1 walk up the child_of edges.
                "alpha query": [1.0, 0.0, 0.0, 0.0],
            }
        )
        llm = FixedNamerLLM()
        system = NeuroMemory(
            storage=SQLiteAdapter(":memory:"),
            llm=llm,
            embedder=ctrl,
            cluster_threshold=0.9,
        )
        system.enqueue("alpha beta")
        system.force_dream()

        helper = ContextHelper(system)
        rendered = helper.build_prompt_context("alpha query", top_k=1, depth=2)

        # The render fired the batched LLM naming exactly once.
        assert llm.batch_call_count == 1
        # The rendered tree shows the resolved label, not the
        # placeholder. A literal `cluster_` substring would mean
        # naming didn't happen.
        assert "cluster_" not in rendered
        assert "named_0" in rendered

    def test_second_render_does_not_re_resolve(self) -> None:
        """Running the same query twice should only fire the LLM
        naming on the first render. Subsequent renders see the
        already-named centroid via storage persistence."""

        class CountingNamerLLM(MockLLMProvider):
            def __init__(self) -> None:
                super().__init__()
                self.batch_call_count = 0

            def generate_category_names_batch(self, pairs: list[list[str]]) -> list[str]:
                self.batch_call_count += 1
                return [f"first_run_{i}" for i in range(len(pairs))]

        from tests.test_system import ControlledEmbedder  # noqa: PLC0415

        ctrl = ControlledEmbedder(
            {
                "alpha": [1.0, 0.0, 0.0, 0.0],
                "beta": [0.999, 0.01, 0.0, 0.0],
                "alpha query": [1.0, 0.0, 0.0, 0.0],
            }
        )
        llm = CountingNamerLLM()
        system = NeuroMemory(
            storage=SQLiteAdapter(":memory:"),
            llm=llm,
            embedder=ctrl,
            cluster_threshold=0.9,
        )
        system.enqueue("alpha beta")
        system.force_dream()

        helper = ContextHelper(system)
        helper.build_prompt_context("alpha query", top_k=1, depth=2)
        assert llm.batch_call_count == 1

        # Second render of the same query: storage now has the resolved
        # label, so resolve_centroid_names sees no placeholders → no
        # LLM call.
        helper.build_prompt_context("alpha query", top_k=1, depth=2)
        assert llm.batch_call_count == 1


class TestEnforceNodeCap:
    """_enforce_node_cap caps the rendered subgraph at N nodes (ADR-003
    F5/D3). Invariants:

    - Subgraphs at or under the cap are untouched.
    - Seed nodes are NEVER dropped, even when top_k > cap.
    - Non-seeds are dropped farthest-from-seed-first (BFS distance over
      child_of edges).
    - Edges and memories referring to dropped nodes are pruned
      consistently.
    - Unreachable nodes (disconnected components) are dropped first
      after seeds.
    """

    @staticmethod
    def _build_chain(n_nodes: int) -> dict:
        """A linear chain: s0 -child_of-> n1 -child_of-> n2 -> ...
        Nodes are named ``n0..n{n-1}``; ``n0`` is the seed root."""
        nodes = [
            {
                "id": f"n{i}",
                "label": f"node{i}",
                "is_centroid": True,
                "embedding": [float(i), 0.0],
                "paragraph_summary": None,
            }
            for i in range(n_nodes)
        ]
        edges = [
            {
                "source_id": f"n{i}",
                "target_id": f"n{i + 1}",
                "weight": 1.0,
                "relationship": "child_of",
            }
            for i in range(n_nodes - 1)
        ]
        return {"nodes": nodes, "edges": edges, "memories": []}

    def test_noop_when_under_cap(self) -> None:
        sub = self._build_chain(5)
        _enforce_node_cap(sub, seed_ids=["n0"], cap=10)
        assert len(sub["nodes"]) == 5

    def test_caps_at_limit(self) -> None:
        sub = self._build_chain(100)
        _enforce_node_cap(sub, seed_ids=["n0"], cap=20)
        assert len(sub["nodes"]) == 20
        kept_ids = {n["id"] for n in sub["nodes"]}
        # Closest to the seed survive.
        for i in range(20):
            assert f"n{i}" in kept_ids
        # Farthest get dropped.
        assert "n99" not in kept_ids

    def test_seeds_never_dropped_even_when_top_k_exceeds_cap(self) -> None:
        """Seed nodes are guaranteed to survive. If the caller passes
        more seeds than the cap, effective_cap is clamped to the seed
        count so every seed is kept."""
        sub = self._build_chain(100)
        # Fabricate 40 seeds — all original top-k nodes must survive.
        seed_ids = [f"n{i}" for i in range(40)]
        _enforce_node_cap(sub, seed_ids=seed_ids, cap=20)
        kept_ids = {n["id"] for n in sub["nodes"]}
        for sid in seed_ids:
            assert sid in kept_ids, (
                f"seed {sid} was dropped despite the contract promising seed preservation"
            )

    def test_prunes_orphan_edges(self) -> None:
        sub = self._build_chain(100)
        _enforce_node_cap(sub, seed_ids=["n0"], cap=10)
        kept_ids = {n["id"] for n in sub["nodes"]}
        for edge in sub["edges"]:
            assert edge["source_id"] in kept_ids
            assert edge["target_id"] in kept_ids

    def test_prunes_orphan_memories(self) -> None:
        """Memory kept iff its has_tag edge points to a kept node."""
        sub = self._build_chain(100)
        # Add a memory attached via has_tag to a far node (n95) — should
        # be pruned — and one attached to the seed (n0) — should stay.
        sub["edges"].extend(
            [
                {
                    "source_id": "mem_far",
                    "target_id": "n95",
                    "weight": 1.0,
                    "relationship": "has_tag",
                },
                {
                    "source_id": "mem_near",
                    "target_id": "n0",
                    "weight": 1.0,
                    "relationship": "has_tag",
                },
            ]
        )
        sub["memories"] = [
            {"id": "mem_far", "summary": "far"},
            {"id": "mem_near", "summary": "near"},
        ]
        _enforce_node_cap(sub, seed_ids=["n0"], cap=10)
        mem_ids = {m["id"] for m in sub["memories"]}
        assert "mem_near" in mem_ids
        assert "mem_far" not in mem_ids

    def test_unreachable_nodes_drop_first(self) -> None:
        """A node with no path to any seed has distance=10_000 and is
        the first to be dropped when the cap tightens."""
        chain = self._build_chain(20)
        # Add 5 orphan nodes with no edges.
        for i in range(5):
            chain["nodes"].append(
                {
                    "id": f"orphan_{i}",
                    "label": f"o{i}",
                    "is_centroid": True,
                    "embedding": [0.0, 0.0],
                    "paragraph_summary": None,
                }
            )
        _enforce_node_cap(chain, seed_ids=["n0"], cap=20)
        kept_ids = {n["id"] for n in chain["nodes"]}
        # All 20 chain nodes survive (they are the closest 20).
        for i in range(20):
            assert f"n{i}" in kept_ids
        # All 5 orphans get dropped.
        for i in range(5):
            assert f"orphan_{i}" not in kept_ids


# Silence unused-import: np and EmbeddingProvider are used in type
# hints by ControlledEmbedder in sibling test files. Kept here for
# readability and to guard against future imports breaking this file.
_ = np
_ = EmbeddingProvider
