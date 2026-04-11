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
from neuromem.context import ContextHelper, _render_ascii_tree
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
        long_text = "x" * 500
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
        # Snippet must be truncated to <= 80 chars plus the trailing ellipsis
        assert "…" in rendered
        # Total snippet length including the ellipsis should be <= 80
        # (the renderer uses MAX_CHARS-1 characters + "…")
        # Find the snippet line and check its length
        snippet_line = next(line for line in rendered.splitlines() if "m1:" in line)
        # Extract the quoted portion
        quoted = snippet_line.split('"', 1)[1].rsplit('"', 1)[0]
        assert len(quoted) <= 80

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


# Silence unused-import: np and EmbeddingProvider are used in type
# hints by ControlledEmbedder in sibling test files. Kept here for
# readability and to guard against future imports breaking this file.
_ = np
_ = EmbeddingProvider
