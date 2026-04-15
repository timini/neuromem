"""ADR-004 — hot-path / ingest split invariants.

After the background dream cycle completes, storage is fully baked
(every centroid has a real label + a paragraph summary) and the
hot-path retrieval methods (build_prompt_context, expand_node,
retrieve_memories) must make ZERO LLM calls in the happy path.

This test file locks in both sides of that contract:

1. The dream cycle's new full-sweep steps 8 and 9 populate every
   centroid's label and paragraph_summary.
2. Hot-path methods post-dream call no LLM methods.
3. `enqueue` is itself zero-LLM (per ADR-004 D1).
4. The safety-net warning fires when invariants are broken.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from neuromem.context import ContextHelper
from neuromem.providers import EmbeddingProvider, LLMProvider
from neuromem.storage.sqlite import SQLiteAdapter
from neuromem.system import NeuroMemory
from neuromem.tools import expand_node, retrieve_memories, search_memory

from tests.conftest import MockEmbeddingProvider, MockLLMProvider


class _LLMCallCounter(MockLLMProvider):
    """Records every LLM method invocation so a test can assert zero."""

    def __init__(self) -> None:
        super().__init__()
        self.summary_calls = 0
        self.summary_batch_calls = 0
        self.tags_calls = 0
        self.tags_batch_calls = 0
        self.ner_calls = 0
        self.ner_batch_calls = 0
        self.name_calls = 0
        self.name_batch_calls = 0
        self.junction_summary_calls = 0
        self.junction_summary_batch_calls = 0

    def generate_summary(self, raw_text: str) -> str:
        self.summary_calls += 1
        return raw_text[:80]

    def generate_summary_batch(self, raw_texts: list[str]) -> list[str]:
        self.summary_batch_calls += 1
        return [t[:80] for t in raw_texts]

    def extract_tags(self, summary: str) -> list[str]:
        self.tags_calls += 1
        return summary.split()[:3]

    def extract_tags_batch(self, summaries: list[str]) -> list[list[str]]:
        self.tags_batch_calls += 1
        return [s.split()[:3] for s in summaries]

    def extract_named_entities(self, summary: str) -> list[str]:
        self.ner_calls += 1
        return []

    def extract_named_entities_batch(self, summaries: list[str]) -> list[list[str]]:
        self.ner_batch_calls += 1
        return [[] for _ in summaries]

    def generate_category_name(
        self, concepts: list[str], *, avoid_names: set[str] | None = None
    ) -> str:
        self.name_calls += 1
        return "category"

    def generate_category_names_batch(
        self, pairs: list[list[str]], *, avoid_names: set[str] | None = None
    ) -> list[str]:
        self.name_batch_calls += 1
        return [f"c{i}" for i in range(len(pairs))]

    def generate_junction_summary(self, children_summaries: list[str]) -> str:
        self.junction_summary_calls += 1
        return "junction summary"

    def generate_junction_summaries_batch(self, groups: list[list[str]]) -> list[str]:
        self.junction_summary_batch_calls += 1
        return [f"g{i}" for i in range(len(groups))]

    def total_calls(self) -> int:
        return (
            self.summary_calls
            + self.summary_batch_calls
            + self.tags_calls
            + self.tags_batch_calls
            + self.ner_calls
            + self.ner_batch_calls
            + self.name_calls
            + self.name_batch_calls
            + self.junction_summary_calls
            + self.junction_summary_batch_calls
        )


def _seeded_system(
    llm: LLMProvider | None = None,
    embedder: EmbeddingProvider | None = None,
) -> NeuroMemory:
    return NeuroMemory(
        storage=SQLiteAdapter(":memory:"),
        llm=llm or _LLMCallCounter(),
        embedder=embedder or MockEmbeddingProvider(),
    )


# ---------------------------------------------------------------------------
# ADR-004 D1 — enqueue non-blocking
# ---------------------------------------------------------------------------


class TestEnqueueNonBlocking:
    def test_enqueue_makes_no_llm_call(self) -> None:
        llm = _LLMCallCounter()
        system = _seeded_system(llm=llm)
        system.enqueue("some memory content that would have been summarised")
        assert llm.total_calls() == 0, (
            f"enqueue made {llm.total_calls()} LLM calls — ADR-004 D1 requires zero"
        )

    def test_enqueue_stores_empty_summary(self) -> None:
        system = _seeded_system()
        mem_id = system.enqueue("long text " * 50)
        row = system.storage.get_memory_by_id(mem_id)
        assert row is not None
        assert row["summary"] == ""

    def test_enqueue_session_is_also_non_blocking(self) -> None:
        llm = _LLMCallCounter()
        system = _seeded_system(llm=llm)
        system.enqueue_session(
            [
                {"role": "user", "text": "turn 1"},
                {"role": "assistant", "text": "turn 2"},
            ]
        )
        assert llm.total_calls() == 0


# ---------------------------------------------------------------------------
# ADR-004 D2 — eager full-sweep in dream cycle
# ---------------------------------------------------------------------------


class _ControlledEmbedder(EmbeddingProvider):
    """Small deterministic embedder producing clusters HDBSCAN can
    resolve without flakiness."""

    def get_embeddings(self, texts: list[str]) -> np.ndarray:
        rng = np.random.default_rng(seed=hash(" ".join(texts)) & 0xFFFF)
        return rng.normal(size=(len(texts), 8)).astype(np.float32)


class TestEagerConsolidation:
    def test_all_centroids_named_after_dream(self) -> None:
        """Step 8 full-sweep: no centroid retains a 'cluster_' placeholder."""
        system = _seeded_system(embedder=_ControlledEmbedder())
        for phrase in [
            "alpha beta gamma",
            "delta epsilon zeta",
            "eta theta iota",
        ]:
            system.enqueue(phrase)
        system.force_dream(block=True)

        centroids = [n for n in system.storage.get_all_nodes() if n["is_centroid"]]
        for c in centroids:
            assert not c["label"].startswith("cluster_"), (
                f"centroid {c['id']} still carries placeholder label "
                f"{c['label']!r} post-dream — ADR-004 step 8 didn't run"
            )

    def test_all_centroids_summarised_after_dream(self) -> None:
        """Step 9 full-sweep: no centroid has paragraph_summary = NULL."""
        system = _seeded_system(embedder=_ControlledEmbedder())
        for phrase in [
            "alpha beta gamma",
            "delta epsilon zeta",
            "eta theta iota",
        ]:
            system.enqueue(phrase)
        system.force_dream(block=True)

        centroids = [n for n in system.storage.get_all_nodes() if n["is_centroid"]]
        for c in centroids:
            assert c.get("paragraph_summary"), (
                f"centroid {c['id']} has NULL paragraph_summary post-dream "
                f"— ADR-004 step 9 didn't run"
            )

    def test_all_memories_summarised_after_dream(self) -> None:
        """Step 2 full-sweep: no consolidated memory has summary = ''."""
        system = _seeded_system(embedder=_ControlledEmbedder())
        for phrase in ["alpha beta", "gamma delta"]:
            system.enqueue(phrase)
        system.force_dream(block=True)

        consolidated = system.storage.get_memories_by_status("consolidated")
        for m in consolidated:
            assert m["summary"], (
                f"memory {m['id']} has empty summary post-dream — step 2 didn't run"
            )


# ---------------------------------------------------------------------------
# ADR-004 D3 — hot-path zero-LLM invariant
# ---------------------------------------------------------------------------


class TestHotPathZeroLLM:
    def _prepare(self) -> tuple[NeuroMemory, _LLMCallCounter]:
        llm = _LLMCallCounter()
        system = NeuroMemory(
            storage=SQLiteAdapter(":memory:"),
            llm=llm,
            embedder=_ControlledEmbedder(),
        )
        for phrase in [
            "alpha beta gamma",
            "delta epsilon zeta",
            "eta theta iota",
            "kappa lambda mu",
        ]:
            system.enqueue(phrase)
        system.force_dream(block=True)
        return system, llm

    def test_build_prompt_context_makes_no_llm_call(self) -> None:
        system, llm = self._prepare()
        pre = llm.total_calls()
        helper = ContextHelper(system)
        helper.build_prompt_context("alpha gamma")
        # Only the embedder should have been called; no LLM methods.
        assert llm.total_calls() == pre, (
            f"build_prompt_context made {llm.total_calls() - pre} LLM "
            f"calls — ADR-004 D3 requires zero"
        )

    def test_expand_node_makes_no_llm_call(self) -> None:
        system, llm = self._prepare()
        nodes = system.storage.get_all_nodes()
        centroid = next((n for n in nodes if n["is_centroid"]), nodes[0])
        pre = llm.total_calls()
        expand_node(centroid["id"], system)
        assert llm.total_calls() == pre, f"expand_node made {llm.total_calls() - pre} LLM calls"

    def test_retrieve_memories_makes_no_llm_call(self) -> None:
        system, llm = self._prepare()
        mems = system.storage.get_memories_by_status("consolidated")
        pre = llm.total_calls()
        retrieve_memories([mems[0]["id"]], system)
        assert llm.total_calls() == pre, (
            f"retrieve_memories made {llm.total_calls() - pre} LLM calls"
        )

    def test_search_memory_makes_no_llm_call(self) -> None:
        system, llm = self._prepare()
        pre = llm.total_calls()
        search_memory("alpha", system)
        assert llm.total_calls() == pre


# ---------------------------------------------------------------------------
# Safety-net warnings
# ---------------------------------------------------------------------------


class TestSafetyNetWarnings:
    def test_resolve_centroid_names_warns_when_placeholders_survive(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """If somehow a placeholder centroid reaches the hot path,
        the resolver logs a WARNING and still does the lazy work."""
        system = _seeded_system(embedder=_ControlledEmbedder())
        # Construct a fake placeholder centroid node and hand it to
        # the resolver directly (bypasses the dream cycle).
        fake_nodes = [
            {
                "id": "fake_node",
                "label": "cluster_fakehexsuffix",
                "is_centroid": True,
                "embedding": np.array([0.1, 0.2], dtype=np.float32),
            }
        ]
        with caplog.at_level(logging.WARNING, logger="neuromem.system"):
            system.resolve_centroid_names(fake_nodes)
        assert any("placeholder centroid(s) on hot path" in rec.message for rec in caplog.records)

    def test_resolve_junction_summaries_warns_when_null_summaries_survive(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        system = _seeded_system(embedder=_ControlledEmbedder())
        fake_nodes = [
            {
                "id": "fake_node",
                "label": "topic",
                "is_centroid": True,
                "embedding": np.array([0.1, 0.2], dtype=np.float32),
                "paragraph_summary": None,
            }
        ]
        with caplog.at_level(logging.WARNING, logger="neuromem.system"):
            system.resolve_junction_summaries(fake_nodes)
        assert any("unsummarised centroid(s) on hot path" in rec.message for rec in caplog.records)


# Silence unused-import lint.
_ = EmbeddingProvider
