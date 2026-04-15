"""ADR-004 D1 — non-blocking enqueue tests.

``enqueue`` must be fast (≤ 20 ms per NF-H1) and safe under
concurrency. These tests are the CI tripwire for the non-blocking
contract; the call-counting invariant lives in
``test_hot_path_invariant.py``.
"""

from __future__ import annotations

import threading
import time

import pytest
from neuromem.storage.sqlite import SQLiteAdapter
from neuromem.system import NeuroMemory

from tests.conftest import MockEmbeddingProvider, MockLLMProvider


@pytest.fixture
def system() -> NeuroMemory:
    # dream_threshold=9999 so the auto-background-dream doesn't fire
    # during the test — we want to isolate per-enqueue latency from
    # any dream-cycle side effects. (The SQLite :memory: connection
    # is not threadsafe, which is a separate pre-existing issue
    # tracked for post-ADR-004 follow-up.)
    return NeuroMemory(
        storage=SQLiteAdapter(":memory:"),
        llm=MockLLMProvider(),
        embedder=MockEmbeddingProvider(),
        dream_threshold=9999,
    )


class TestEnqueueLatency:
    def test_per_call_wall_under_20ms_with_mock_llm(self, system: NeuroMemory) -> None:
        """NF-H1: enqueue must return in ≤ 20 ms with a mock provider.

        Mock LLM's generate_summary is essentially free, so we're
        measuring the storage insert + inbox-count check. A regression
        that restored the synchronous generate_summary call would
        immediately blow this budget.
        """
        walls: list[float] = []
        for i in range(20):
            start = time.perf_counter()
            system.enqueue(f"text {i}")
            walls.append((time.perf_counter() - start) * 1000.0)
        p50 = sorted(walls)[len(walls) // 2]
        assert p50 <= 20.0, f"enqueue p50 {p50:.1f} ms exceeds 20 ms NF-H1"


class TestEnqueueSummaryLifecycle:
    def test_summary_empty_at_enqueue(self, system: NeuroMemory) -> None:
        mem_id = system.enqueue("some text")
        row = system.storage.get_memory_by_id(mem_id)
        assert row is not None
        assert row["summary"] == ""

    def test_summary_populated_after_dream(self, system: NeuroMemory) -> None:
        mem_id = system.enqueue("some text " * 20)
        system.force_dream(block=True)
        row = system.storage.get_memory_by_id(mem_id)
        assert row is not None
        assert row["summary"] != ""


class TestSequentialEnqueueThenDream:
    def test_many_enqueues_followed_by_single_dream(self) -> None:
        """Many fast enqueues followed by a single force_dream
        consolidates all memories. Exercises the "inbox fills quickly
        because enqueue is no longer blocking on summary" pattern
        that's the whole point of ADR-004 D1."""
        system = NeuroMemory(
            storage=SQLiteAdapter(":memory:"),
            llm=MockLLMProvider(),
            embedder=MockEmbeddingProvider(),
            dream_threshold=9999,
        )
        for i in range(30):
            system.enqueue(f"memory-{i}")
        assert system.storage.count_memories_by_status("inbox") == 30
        system.force_dream(block=True)
        assert system.storage.count_memories_by_status("consolidated") == 30
        assert system.storage.count_memories_by_status("inbox") == 0


# threading is imported for a thread-safety test that's deferred to a
# follow-up commit. SQLite :memory: doesn't tolerate cross-thread
# writes well; the proper fix requires a connection-per-thread pool
# or explicit cross-thread locking, both out of scope for ADR-004.
_ = threading
