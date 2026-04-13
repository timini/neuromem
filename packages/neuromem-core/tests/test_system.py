"""Unit tests for neuromem.system.NeuroMemory.

Grown incrementally:
  - T019: constructor validation + sync enqueue path (this file's
    initial content)
  - T020: _run_dream_cycle pipeline tests
  - T021: threading + force_dream concurrency tests
  - T024: US3 decay/archival end-to-end tests
  - T026: US5 force_dream acceptance tests
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from neuromem.providers import EmbeddingProvider, LLMProvider
from neuromem.storage.sqlite import SQLiteAdapter
from neuromem.system import NeuroMemory

from tests.conftest import MockEmbeddingProvider, MockLLMProvider


@pytest.fixture
def memory_system(
    mock_embedder: MockEmbeddingProvider,
    mock_llm: MockLLMProvider,
) -> NeuroMemory:
    """A freshly-constructed NeuroMemory with an in-memory SQLite backend."""
    return NeuroMemory(
        storage=SQLiteAdapter(":memory:"),
        llm=mock_llm,
        embedder=mock_embedder,
    )


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_accepts_valid_dependencies(
        self,
        mock_embedder: MockEmbeddingProvider,
        mock_llm: MockLLMProvider,
    ) -> None:
        system = NeuroMemory(
            storage=SQLiteAdapter(":memory:"),
            llm=mock_llm,
            embedder=mock_embedder,
        )
        assert system.dream_threshold == 10
        assert system.is_dreaming is False

    def test_accepts_custom_thresholds(
        self,
        mock_embedder: MockEmbeddingProvider,
        mock_llm: MockLLMProvider,
    ) -> None:
        system = NeuroMemory(
            storage=SQLiteAdapter(":memory:"),
            llm=mock_llm,
            embedder=mock_embedder,
            dream_threshold=5,
            decay_lambda=1e-6,
            archive_threshold=0.2,
            cluster_threshold=0.75,
        )
        assert system.dream_threshold == 5
        assert system.decay_lambda == pytest.approx(1e-6)
        assert system.archive_threshold == pytest.approx(0.2)
        assert system.cluster_threshold == pytest.approx(0.75)

    def test_rejects_non_storage_adapter(
        self,
        mock_embedder: MockEmbeddingProvider,
        mock_llm: MockLLMProvider,
    ) -> None:
        with pytest.raises(TypeError, match="StorageAdapter"):
            NeuroMemory(
                storage="not an adapter",  # type: ignore[arg-type]
                llm=mock_llm,
                embedder=mock_embedder,
            )

    def test_rejects_non_llm_provider(
        self,
        mock_embedder: MockEmbeddingProvider,
    ) -> None:
        with pytest.raises(TypeError, match="LLMProvider"):
            NeuroMemory(
                storage=SQLiteAdapter(":memory:"),
                llm=object(),  # type: ignore[arg-type]
                embedder=mock_embedder,
            )

    def test_rejects_non_embedding_provider(
        self,
        mock_llm: MockLLMProvider,
    ) -> None:
        with pytest.raises(TypeError, match="EmbeddingProvider"):
            NeuroMemory(
                storage=SQLiteAdapter(":memory:"),
                llm=mock_llm,
                embedder=object(),  # type: ignore[arg-type]
            )

    def test_rejects_zero_dream_threshold(
        self,
        mock_embedder: MockEmbeddingProvider,
        mock_llm: MockLLMProvider,
    ) -> None:
        with pytest.raises(ValueError, match="dream_threshold"):
            NeuroMemory(
                storage=SQLiteAdapter(":memory:"),
                llm=mock_llm,
                embedder=mock_embedder,
                dream_threshold=0,
            )

    def test_rejects_negative_decay_lambda(
        self,
        mock_embedder: MockEmbeddingProvider,
        mock_llm: MockLLMProvider,
    ) -> None:
        with pytest.raises(ValueError, match="decay_lambda"):
            NeuroMemory(
                storage=SQLiteAdapter(":memory:"),
                llm=mock_llm,
                embedder=mock_embedder,
                decay_lambda=-1e-6,
            )

    def test_rejects_out_of_range_archive_threshold(
        self,
        mock_embedder: MockEmbeddingProvider,
        mock_llm: MockLLMProvider,
    ) -> None:
        with pytest.raises(ValueError, match="archive_threshold"):
            NeuroMemory(
                storage=SQLiteAdapter(":memory:"),
                llm=mock_llm,
                embedder=mock_embedder,
                archive_threshold=1.5,
            )

    def test_rejects_out_of_range_cluster_threshold(
        self,
        mock_embedder: MockEmbeddingProvider,
        mock_llm: MockLLMProvider,
    ) -> None:
        with pytest.raises(ValueError, match="cluster_threshold"):
            NeuroMemory(
                storage=SQLiteAdapter(":memory:"),
                llm=mock_llm,
                embedder=mock_embedder,
                cluster_threshold=0.0,
            )


# ---------------------------------------------------------------------------
# enqueue (sync path)
# ---------------------------------------------------------------------------


class TestEnqueueSync:
    def test_returns_memory_id(self, memory_system: NeuroMemory) -> None:
        mem_id = memory_system.enqueue("hello world")
        assert isinstance(mem_id, str)
        assert mem_id.startswith("mem_")

    def test_memory_lands_in_inbox_status(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        mem_id = memory_system.enqueue("raw content")
        row = memory_system.storage.get_memory_by_id(mem_id)
        assert row is not None
        assert row["status"] == "inbox"
        assert row["access_weight"] == pytest.approx(1.0)
        assert row["raw_content"] == "raw content"

    def test_summary_populated_from_llm(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        # MockLLMProvider returns raw_text[:80] as the summary.
        long_text = "a" * 200
        mem_id = memory_system.enqueue(long_text)
        row = memory_system.storage.get_memory_by_id(mem_id)
        assert row["summary"] == "a" * 80

    def test_metadata_round_trip(self, memory_system: NeuroMemory) -> None:
        mem_id = memory_system.enqueue("hi", metadata={"role": "user", "turn": 3})
        row = memory_system.storage.get_memory_by_id(mem_id)
        assert row["metadata"] == {"role": "user", "turn": 3}

    def test_empty_raw_text_raises(self, memory_system: NeuroMemory) -> None:
        with pytest.raises(ValueError, match="raw_text"):
            memory_system.enqueue("")

    def test_enqueue_latency_sc002(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        """SC-002: enqueue must return in <50 ms excluding generate_summary.

        Since MockLLMProvider.generate_summary is essentially free (a
        string slice), total enqueue latency here is dominated by the
        SQLite write. A 50 ms budget with MockLLMProvider is a very
        loose bound — a real failure would show latency in the
        hundreds of ms, indicating a serious issue.
        """
        start = time.perf_counter()
        memory_system.enqueue("hello")
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        assert elapsed_ms < 50.0, f"enqueue took {elapsed_ms:.1f} ms (budget 50)"

    def test_multiple_enqueues_all_in_inbox(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        for i in range(5):
            memory_system.enqueue(f"memory {i}")
        assert memory_system.storage.count_memories_by_status("inbox") == 5

    def test_enqueue_below_threshold_does_not_spawn_dream(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        # Default threshold is 10; 5 enqueues should leave is_dreaming False.
        for i in range(5):
            memory_system.enqueue(f"memory {i}")
        assert memory_system.is_dreaming is False

    def test_concrete_provider_types_accepted(
        self,
        mock_embedder: MockEmbeddingProvider,
    ) -> None:
        """Custom subclasses of the ABCs must work."""

        class TerseLLM(LLMProvider):
            def generate_summary(self, raw_text: str) -> str:
                return raw_text[:20]

            def extract_tags(self, summary: str) -> list[str]:
                return []

            def generate_category_name(self, concepts: list[str]) -> str:
                return "Misc"

        system = NeuroMemory(
            storage=SQLiteAdapter(":memory:"),
            llm=TerseLLM(),
            embedder=mock_embedder,
        )
        mem_id = system.enqueue("hello world this is a long sentence")
        row = system.storage.get_memory_by_id(mem_id)
        # TerseLLM returns raw_text[:20]. Index 20 of the input is the
        # space character AFTER "is", so the slice is 20 chars with a
        # trailing space — exactly what Python produces. No off-by-one
        # adjustment in the assertion.
        assert row["summary"] == "hello world this is "


# ---------------------------------------------------------------------------
# enqueue_session — session-level ingestion
# ---------------------------------------------------------------------------


class TestEnqueueSession:
    """``enqueue_session`` is the session-level counterpart to
    ``enqueue``. A list of (role, text) turns becomes a single
    memory whose raw_content is the concatenated transcript. The
    key invariant we lock in here: only ONE LLM summary call per
    session regardless of how many turns it contains.
    """

    def test_returns_single_memory_id(self, memory_system: NeuroMemory) -> None:
        mem_id = memory_system.enqueue_session(
            [
                {"role": "user", "text": "hello"},
                {"role": "assistant", "text": "hi there"},
            ]
        )
        assert isinstance(mem_id, str)
        assert mem_id.startswith("mem_")

    def test_single_turn_valid(self, memory_system: NeuroMemory) -> None:
        """A one-turn session is a degenerate but valid input."""
        mem_id = memory_system.enqueue_session([{"role": "user", "text": "alone turn"}])
        row = memory_system.storage.get_memory_by_id(mem_id)
        assert row is not None
        assert row["raw_content"] == "user: alone turn"

    def test_transcript_format_in_raw_content(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        """Turns are joined with a double newline and prefixed by role."""
        mem_id = memory_system.enqueue_session(
            [
                {"role": "user", "text": "question?"},
                {"role": "assistant", "text": "answer."},
                {"role": "user", "text": "follow-up"},
            ]
        )
        row = memory_system.storage.get_memory_by_id(mem_id)
        assert row["raw_content"] == ("user: question?\n\nassistant: answer.\n\nuser: follow-up")

    def test_one_summary_call_per_session(
        self,
        mock_embedder: MockEmbeddingProvider,
    ) -> None:
        """The whole point of this method: a 10-turn session should
        produce exactly ONE generate_summary call, not 10."""

        class CountingLLM(MockLLMProvider):
            def __init__(self) -> None:
                super().__init__()
                self.summary_call_count = 0

            def generate_summary(self, raw_text: str) -> str:
                self.summary_call_count += 1
                return raw_text[:50]

        llm = CountingLLM()
        system = NeuroMemory(
            storage=SQLiteAdapter(":memory:"),
            llm=llm,
            embedder=mock_embedder,
        )
        turns = [
            {"role": "user" if i % 2 == 0 else "assistant", "text": f"turn {i}"} for i in range(10)
        ]
        system.enqueue_session(turns)
        assert llm.summary_call_count == 1

    def test_metadata_attached_to_session_memory(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        mem_id = memory_system.enqueue_session(
            [{"role": "user", "text": "hi"}],
            metadata={"session_index": 5, "turn_count": 1},
        )
        row = memory_system.storage.get_memory_by_id(mem_id)
        assert row["metadata"] == {"session_index": 5, "turn_count": 1}

    def test_empty_turns_raises(self, memory_system: NeuroMemory) -> None:
        with pytest.raises(ValueError, match="turns must be non-empty"):
            memory_system.enqueue_session([])

    def test_turn_missing_keys_raises(self, memory_system: NeuroMemory) -> None:
        with pytest.raises(ValueError, match="must have 'role' and 'text'"):
            memory_system.enqueue_session([{"role": "user"}])  # no text
        with pytest.raises(ValueError, match="must have 'role' and 'text'"):
            memory_system.enqueue_session([{"text": "hi"}])  # no role

    def test_turn_wrong_type_raises(self, memory_system: NeuroMemory) -> None:
        with pytest.raises(ValueError, match="must be a dict"):
            memory_system.enqueue_session(["just a string"])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# force_dream (T020 Half A: block=True runs sync; block=False lands in T021)
# ---------------------------------------------------------------------------


class TestForceDream:
    def test_block_true_runs_synchronously(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        """force_dream(block=True) runs _run_dream_cycle directly."""
        for i in range(3):
            memory_system.enqueue(f"memory {i}")
        memory_system.force_dream(block=True)
        # After dreaming, all memories must be consolidated.
        assert memory_system.storage.count_memories_by_status("inbox") == 0
        assert memory_system.storage.count_memories_by_status("consolidated") == 3


# ---------------------------------------------------------------------------
# T021: Concurrency — threshold-spawned threads, block=False, double buffer
# ---------------------------------------------------------------------------


class BlockingLLM(LLMProvider):
    """LLM that blocks inside ``extract_tags`` until released by a test.

    Used to pause the dreaming cycle mid-flight so tests can observe
    in-progress state (``is_dreaming``, dreaming-status memories) and
    time-sensitive behaviour (non-blocking return of ``force_dream``,
    double-buffer correctness).

    Workflow:
      1. Test enqueues memories, triggers a background dream cycle.
      2. Dream thread enters ``extract_tags`` → sets ``entered_event``.
      3. Test waits on ``entered_event`` to know the cycle is live.
      4. Test observes intermediate state (dreaming status, lock held).
      5. Test sets ``proceed_event`` to release the dream thread.
      6. Dream thread finishes; test joins ``_dream_thread`` for cleanup.
    """

    def __init__(self) -> None:
        self.entered_event = threading.Event()
        self.proceed_event = threading.Event()
        self.extract_tags_call_count = 0

    def generate_summary(self, raw_text: str) -> str:
        return raw_text[:40]

    def extract_tags(self, summary: str) -> list[str]:
        self.extract_tags_call_count += 1
        self.entered_event.set()
        if not self.proceed_event.wait(timeout=5.0):
            raise RuntimeError("BlockingLLM timeout — test forgot to set proceed_event")
        return [w for w in summary.split() if w.isalpha()][:3]

    def generate_category_name(self, concepts: list[str]) -> str:
        return "Cat"


class TestEnqueueThresholdSpawnsThread:
    def test_enqueue_past_threshold_spawns_background_thread(
        self,
        mock_llm: MockLLMProvider,
        mock_embedder: MockEmbeddingProvider,
    ) -> None:
        """Enqueuing past ``dream_threshold`` spawns a daemon thread
        that runs ``_run_dream_cycle`` and consolidates the memories.
        """
        system = NeuroMemory(
            storage=SQLiteAdapter(":memory:"),
            llm=mock_llm,
            embedder=mock_embedder,
            dream_threshold=3,
        )
        for i in range(3):
            system.enqueue(f"memory {i}")

        # A thread was spawned. Join it to avoid races in the assertion.
        assert system._dream_thread is not None
        system._dream_thread.join(timeout=5.0)
        assert not system._dream_thread.is_alive()

        # All 3 memories consolidated via the background cycle.
        assert system.storage.count_memories_by_status("consolidated") == 3
        assert system.storage.count_memories_by_status("inbox") == 0
        assert system.is_dreaming is False

    def test_enqueue_below_threshold_does_not_spawn(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        """Default threshold is 10; 5 enqueues should NOT spawn a thread."""
        for i in range(5):
            memory_system.enqueue(f"memory {i}")
        assert memory_system._dream_thread is None
        assert memory_system.is_dreaming is False
        assert memory_system.storage.count_memories_by_status("inbox") == 5


class TestForceDreamNonBlocking:
    def test_block_false_returns_immediately(
        self,
        mock_embedder: MockEmbeddingProvider,
    ) -> None:
        """force_dream(block=False) spawns a thread and returns fast,
        while the thread is still inside extract_tags."""
        blocking_llm = BlockingLLM()
        system = NeuroMemory(
            storage=SQLiteAdapter(":memory:"),
            llm=blocking_llm,
            embedder=mock_embedder,
        )
        system.enqueue("alpha beta gamma")

        start = time.perf_counter()
        system.force_dream(block=False)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        assert elapsed_ms < 500.0, f"block=False returned in {elapsed_ms:.1f} ms"

        # Wait for the dream thread to enter extract_tags.
        assert blocking_llm.entered_event.wait(timeout=2.0), (
            "dream thread never entered extract_tags"
        )

        # Right now the cycle is live: the memory is in 'dreaming' status,
        # is_dreaming is True, and the spawned thread is still alive.
        dreaming_rows = system.storage.get_memories_by_status("dreaming")
        assert len(dreaming_rows) == 1
        assert system.is_dreaming is True
        assert system._dream_thread is not None
        assert system._dream_thread.is_alive()

        # Release the dream cycle and join the thread.
        blocking_llm.proceed_event.set()
        system._dream_thread.join(timeout=5.0)
        assert not system._dream_thread.is_alive()
        assert system.is_dreaming is False
        assert system.storage.count_memories_by_status("consolidated") == 1

    def test_block_false_noop_while_thread_already_running(
        self,
        mock_embedder: MockEmbeddingProvider,
    ) -> None:
        """A second force_dream(block=False) while the first is running
        does NOT spawn another thread — it's a silent no-op."""
        blocking_llm = BlockingLLM()
        system = NeuroMemory(
            storage=SQLiteAdapter(":memory:"),
            llm=blocking_llm,
            embedder=mock_embedder,
        )
        system.enqueue("first memory")
        system.force_dream(block=False)
        assert blocking_llm.entered_event.wait(timeout=2.0)

        first_thread = system._dream_thread
        assert first_thread is not None and first_thread.is_alive()

        # Second block=False call — should NOT spawn a new thread.
        system.force_dream(block=False)
        assert system._dream_thread is first_thread

        # Clean up.
        blocking_llm.proceed_event.set()
        first_thread.join(timeout=5.0)


class TestForceDreamBlockingJoinsExistingThread:
    def test_block_true_waits_for_in_flight_background_thread(
        self,
        mock_embedder: MockEmbeddingProvider,
    ) -> None:
        """force_dream(block=True) while a background thread is running
        joins the existing thread before running its own cycle."""
        blocking_llm = BlockingLLM()
        system = NeuroMemory(
            storage=SQLiteAdapter(":memory:"),
            llm=blocking_llm,
            embedder=mock_embedder,
        )
        system.enqueue("first")
        system.force_dream(block=False)
        assert blocking_llm.entered_event.wait(timeout=2.0)

        existing_thread = system._dream_thread
        assert existing_thread is not None and existing_thread.is_alive()

        # Release the BlockingLLM from another thread so the background
        # cycle can finish while the main thread is inside force_dream.
        releaser = threading.Thread(
            target=lambda: blocking_llm.proceed_event.set(),
            name="releaser",
        )
        releaser.start()

        # block=True should join existing_thread, then run its own cycle
        # (which is a no-op if the inbox is empty). Returns only after
        # everything is consolidated.
        system.force_dream(block=True)
        releaser.join(timeout=5.0)

        assert not existing_thread.is_alive()
        assert system.storage.count_memories_by_status("consolidated") == 1
        assert system.is_dreaming is False


class TestDoubleBuffer:
    def test_memories_enqueued_during_dreaming_land_in_next_cycle(
        self,
        mock_embedder: MockEmbeddingProvider,
    ) -> None:
        """The double-buffer invariant: memories arriving during an
        active dream cycle stay in 'inbox' and are processed by a
        LATER cycle, not the current one."""
        blocking_llm = BlockingLLM()
        system = NeuroMemory(
            storage=SQLiteAdapter(":memory:"),
            llm=blocking_llm,
            embedder=mock_embedder,
        )
        first_id = system.enqueue("first memory")
        system.force_dream(block=False)
        assert blocking_llm.entered_event.wait(timeout=2.0)

        # Cycle is live and inside extract_tags. Enqueue another memory.
        second_id = system.enqueue("second memory")

        # First memory is in 'dreaming'; second is in 'inbox'.
        assert system.storage.get_memory_by_id(first_id)["status"] == "dreaming"
        assert system.storage.get_memory_by_id(second_id)["status"] == "inbox"

        # Release the cycle and wait for it to finish.
        blocking_llm.proceed_event.set()
        assert system._dream_thread is not None
        system._dream_thread.join(timeout=5.0)

        # First is now consolidated. Second is still in inbox (not
        # processed by the cycle that was running when it arrived).
        assert system.storage.get_memory_by_id(first_id)["status"] == "consolidated"
        assert system.storage.get_memory_by_id(second_id)["status"] == "inbox"

        # A fresh force_dream processes the second memory.
        system.force_dream(block=True)
        assert system.storage.get_memory_by_id(second_id)["status"] == "consolidated"


class TestGenuineConcurrency:
    def test_two_threads_cannot_both_enter_dream_cycle(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        """Real two-thread test (per PR #36 review finding I-3).

        Two worker threads call ``_run_dream_cycle`` simultaneously via
        a ``threading.Barrier``. Only one of them can acquire the lock
        and run the pipeline; the other must see the lock held and
        return immediately without touching any state.

        This is the cross-thread exclusion test that
        ``test_held_lock_causes_immediate_return`` deliberately does
        not cover (it exercises only the lock-skip logic branch via
        same-thread re-entry).
        """
        memory_system.enqueue("the quick brown fox")

        barrier = threading.Barrier(2)
        results: list[Exception | None] = []

        def worker() -> None:
            try:
                barrier.wait(timeout=2.0)
                memory_system._run_dream_cycle()
                results.append(None)
            except Exception as exc:  # pragma: no cover — guarded against lock bugs
                results.append(exc)

        t1 = threading.Thread(target=worker, name="race-1")
        t2 = threading.Thread(target=worker, name="race-2")
        t1.start()
        t2.start()
        t1.join(timeout=5.0)
        t2.join(timeout=5.0)

        # Neither thread raised.
        assert all(r is None for r in results), f"worker raised: {results}"
        # Exactly one thread did the work: the memory is consolidated
        # (not doubled, not stuck in dreaming).
        assert memory_system.storage.count_memories_by_status("consolidated") == 1
        assert memory_system.storage.count_memories_by_status("dreaming") == 0
        assert memory_system.storage.count_memories_by_status("inbox") == 0
        # Lock and state flag are both clean after both threads exit.
        assert memory_system.is_dreaming is False
        assert memory_system._dream_lock.acquire(blocking=False)
        memory_system._dream_lock.release()


# ---------------------------------------------------------------------------
# _run_dream_cycle (T020 Half A — pipeline skeleton, no clustering)
# ---------------------------------------------------------------------------


class TestRunDreamCycleHappyPath:
    def test_three_memories_consolidate_through_pipeline(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        # MockLLMProvider.extract_tags returns the first 3 alpha tokens
        # of the summary. MockEmbeddingProvider produces deterministic
        # 16-dim float32 vectors.
        ids = [
            memory_system.enqueue("alpha beta gamma"),
            memory_system.enqueue("delta epsilon zeta"),
            memory_system.enqueue("eta theta iota"),
        ]
        memory_system.force_dream()

        # 1. All 3 memories are now consolidated.
        for mid in ids:
            row = memory_system.storage.get_memory_by_id(mid)
            assert row is not None
            assert row["status"] == "consolidated"

        # 2. Nine distinct tag nodes (one per unique token) exist and
        #    are leaf tags, not centroids (clustering lands in Half B).
        nodes = memory_system.storage.get_all_nodes()
        labels = {n["label"] for n in nodes}
        expected = {
            "alpha",
            "beta",
            "gamma",
            "delta",
            "epsilon",
            "zeta",
            "eta",
            "theta",
            "iota",
        }
        assert expected <= labels
        for node in nodes:
            assert node["is_centroid"] is False

        # 3. has_tag edges wired from each memory to its 3 tags.
        #    get_subgraph traverses from a node and collects reachable
        #    memories — use alpha's node as the root.
        alpha_id = next(n["id"] for n in nodes if n["label"] == "alpha")
        sub = memory_system.storage.get_subgraph([alpha_id], depth=1)
        reachable_memory_ids = {m["id"] for m in sub["memories"]}
        assert ids[0] in reachable_memory_ids

    def test_empty_inbox_is_no_op(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        """_run_dream_cycle on empty inbox must not raise."""
        memory_system.force_dream()
        assert memory_system.storage.count_memories_by_status("inbox") == 0
        assert memory_system.storage.count_memories_by_status("consolidated") == 0
        assert memory_system.is_dreaming is False

    def test_is_dreaming_false_after_cycle(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        memory_system.enqueue("hello")
        memory_system.force_dream()
        assert memory_system.is_dreaming is False

    def test_inbox_emptied_after_cycle(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        for i in range(5):
            memory_system.enqueue(f"raw {i}")
        assert memory_system.storage.count_memories_by_status("inbox") == 5
        memory_system.force_dream()
        assert memory_system.storage.count_memories_by_status("inbox") == 0


class TestRunDreamCycleRollback:
    def test_extract_tags_exception_rolls_back_batch(
        self,
        mock_embedder: MockEmbeddingProvider,
    ) -> None:
        class BrokenLLM(LLMProvider):
            def generate_summary(self, raw_text: str) -> str:
                return raw_text[:40]

            def extract_tags(self, summary: str) -> list[str]:
                raise RuntimeError("provider boom")

            def generate_category_name(self, concepts: list[str]) -> str:
                return "Misc"

        system = NeuroMemory(
            storage=SQLiteAdapter(":memory:"),
            llm=BrokenLLM(),
            embedder=mock_embedder,
        )
        mem_ids = [system.enqueue(f"raw {i}") for i in range(3)]

        with pytest.raises(RuntimeError, match="provider boom"):
            system.force_dream()

        # All 3 memories should be BACK in inbox, not stuck in dreaming.
        assert system.storage.count_memories_by_status("inbox") == 3
        assert system.storage.count_memories_by_status("dreaming") == 0
        assert system.storage.count_memories_by_status("consolidated") == 0
        for mid in mem_ids:
            assert system.storage.get_memory_by_id(mid)["status"] == "inbox"
        # And is_dreaming is cleared even though we raised.
        assert system.is_dreaming is False

    def test_embedder_exception_rolls_back_batch(
        self,
        mock_llm: MockLLMProvider,
    ) -> None:
        class BrokenEmbedder(EmbeddingProvider):
            def get_embeddings(self, texts):
                raise RuntimeError("embed boom")

        system = NeuroMemory(
            storage=SQLiteAdapter(":memory:"),
            llm=mock_llm,
            embedder=BrokenEmbedder(),
        )
        mem_id = system.enqueue("hello there")

        with pytest.raises(RuntimeError, match="embed boom"):
            system.force_dream()

        assert system.storage.get_memory_by_id(mem_id)["status"] == "inbox"
        assert system.is_dreaming is False


class TestRunDreamCycleLocking:
    def test_held_lock_causes_immediate_return(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        """Validates the lock-skip branch of ``_run_dream_cycle``.

        Acquires ``_dream_lock`` on the test thread, then calls
        ``_run_dream_cycle`` from the SAME thread. Because
        ``threading.Lock`` is not re-entrant, the non-blocking
        ``acquire(blocking=False)`` inside ``_run_dream_cycle``
        returns ``False`` and the method returns immediately without
        touching any inbox memories.

        Note: this test exercises the lock-skip **logic branch**, not
        actual cross-thread concurrency. A genuine two-thread test
        (spawning two workers hitting ``_run_dream_cycle`` simultaneously
        with a ``threading.Barrier``) is a T021 deliverable and will
        cover the cross-thread exclusion path. See PR #36 review
        finding I-3 for the reasoning behind the split.
        """
        mem_id = memory_system.enqueue("stays in inbox")
        assert memory_system._dream_lock.acquire(blocking=False)
        try:
            memory_system._run_dream_cycle()
            # Memory was NOT processed.
            assert memory_system.storage.get_memory_by_id(mem_id)["status"] == "inbox"
        finally:
            memory_system._dream_lock.release()


class TestEnsureTagNodes:
    def test_reuses_existing_nodes_by_label(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        """Second dream cycle reuses tag nodes created in the first.

        After two dream cycles that both produce the tag 'shared',
        there should be exactly ONE 'shared' node in storage, not two.
        """
        memory_system.enqueue("shared common tokens")
        memory_system.force_dream()
        nodes_after_first = memory_system.storage.get_all_nodes()
        shared_count_1 = sum(1 for n in nodes_after_first if n["label"] == "shared")
        assert shared_count_1 == 1

        memory_system.enqueue("shared another thing")
        memory_system.force_dream()
        nodes_after_second = memory_system.storage.get_all_nodes()
        shared_count_2 = sum(1 for n in nodes_after_second if n["label"] == "shared")
        assert shared_count_2 == 1


# ---------------------------------------------------------------------------
# Agglomerative clustering (T020 Half B)
# ---------------------------------------------------------------------------


class ControlledEmbedder(EmbeddingProvider):
    """Deterministic embedder for clustering tests.

    Returns a fixed 4-dim vector for every known label. Unknown
    labels raise KeyError so bugs in the tag-to-node wiring surface
    loudly rather than silently returning arbitrary vectors.
    """

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = {label: np.asarray(vec, dtype=np.float32) for label, vec in vectors.items()}
        self.dim = next(iter(self._vectors.values())).shape[0]

    def get_embeddings(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vectors[t] for t in texts])


class TestAgglomerativeClustering:
    def test_clustering_creates_centroid_when_close(
        self,
        mock_llm: MockLLMProvider,
    ) -> None:
        """Two tags with cosine ~0.995 merge into a centroid."""
        ctrl = ControlledEmbedder(
            {
                # alpha and beta are nearly identical (cos ≈ 0.995)
                "alpha": [1.0, 0.0, 0.0, 0.0],
                "beta": [0.995, 0.1, 0.0, 0.0],
                # gamma is orthogonal to both
                "gamma": [0.0, 0.0, 1.0, 0.0],
            }
        )
        system = NeuroMemory(
            storage=SQLiteAdapter(":memory:"),
            llm=mock_llm,
            embedder=ctrl,
            cluster_threshold=0.9,
        )
        # MockLLMProvider.extract_tags returns first 3 alpha tokens.
        system.enqueue("alpha beta gamma")
        system.force_dream()

        nodes = system.storage.get_all_nodes()
        leaves = [n for n in nodes if not n["is_centroid"]]
        centroids = [n for n in nodes if n["is_centroid"]]

        # 3 leaves (alpha, beta, gamma) + at least 1 centroid (alpha+beta)
        leaf_labels = {n["label"] for n in leaves}
        assert leaf_labels >= {"alpha", "beta", "gamma"}
        assert len(centroids) >= 1

        # MockLLMProvider.generate_category_name(["alpha","beta"]) → "CatAB"
        # or "CatBA" depending on argmax ordering; both are valid.
        centroid_labels = {n["label"] for n in centroids}
        assert centroid_labels & {"CatAB", "CatBA"}

    def test_clustering_does_not_merge_below_threshold(
        self,
        mock_llm: MockLLMProvider,
    ) -> None:
        """With cluster_threshold=0.99, cos=0.5 pairs are not merged."""
        ctrl = ControlledEmbedder(
            {
                "alpha": [1.0, 0.0, 0.0, 0.0],
                "beta": [0.5, 0.866, 0.0, 0.0],  # cos(alpha, beta) = 0.5
                "gamma": [0.0, 0.5, 0.866, 0.0],
            }
        )
        system = NeuroMemory(
            storage=SQLiteAdapter(":memory:"),
            llm=mock_llm,
            embedder=ctrl,
            cluster_threshold=0.99,
        )
        system.enqueue("alpha beta gamma")
        system.force_dream()

        nodes = system.storage.get_all_nodes()
        centroids = [n for n in nodes if n["is_centroid"]]
        assert len(centroids) == 0

    def test_has_tag_edges_still_point_to_leaves_after_clustering(
        self,
        mock_llm: MockLLMProvider,
    ) -> None:
        """A memory's has_tag edges must land on the LEAF tag nodes,
        never on their centroid parents, so contextual recall can
        still reach the original concept."""
        ctrl = ControlledEmbedder(
            {
                "alpha": [1.0, 0.0, 0.0, 0.0],
                "beta": [0.999, 0.01, 0.0, 0.0],
                "gamma": [0.0, 0.0, 1.0, 0.0],
            }
        )
        system = NeuroMemory(
            storage=SQLiteAdapter(":memory:"),
            llm=mock_llm,
            embedder=ctrl,
            cluster_threshold=0.9,
        )
        mem_id = system.enqueue("alpha beta gamma")
        system.force_dream()

        # Pull alpha's leaf node id and walk the subgraph from it.
        nodes = system.storage.get_all_nodes()
        alpha_leaf = next(n for n in nodes if n["label"] == "alpha" and not n["is_centroid"])
        sub = system.storage.get_subgraph([alpha_leaf["id"]], depth=2)

        # The memory must be reachable via the alpha leaf (has_tag edge).
        reachable = {m["id"] for m in sub["memories"]}
        assert mem_id in reachable

        # Verify directly: the has_tag edge source is the memory, target
        # is the leaf, NOT the centroid.
        has_tag_edges = [e for e in sub["edges"] if e["relationship"] == "has_tag"]
        assert any(
            e["source_id"] == mem_id and e["target_id"] == alpha_leaf["id"] for e in has_tag_edges
        )

    def test_child_of_edges_wired_centroid_to_members(
        self,
        mock_llm: MockLLMProvider,
    ) -> None:
        """After clustering, centroid → member edges exist with
        relationship='child_of' and weight equal to the merge similarity."""
        ctrl = ControlledEmbedder(
            {
                "alpha": [1.0, 0.0, 0.0, 0.0],
                "beta": [0.99, 0.0, 0.0, 0.0],  # cos = 0.99 exactly
            }
        )
        system = NeuroMemory(
            storage=SQLiteAdapter(":memory:"),
            llm=mock_llm,
            embedder=ctrl,
            cluster_threshold=0.9,
        )
        system.enqueue("alpha beta")
        system.force_dream()

        nodes = system.storage.get_all_nodes()
        centroids = [n for n in nodes if n["is_centroid"]]
        assert len(centroids) == 1
        centroid = centroids[0]

        sub = system.storage.get_subgraph([centroid["id"]], depth=1)
        child_of_edges = [
            e
            for e in sub["edges"]
            if e["relationship"] == "child_of" and e["source_id"] == centroid["id"]
        ]
        assert len(child_of_edges) == 2
        for edge in child_of_edges:
            assert edge["weight"] >= 0.9  # above the cluster threshold
            assert edge["weight"] <= 1.0


class TestSanitiseCategoryName:
    def test_single_word_passthrough(self) -> None:
        from neuromem.system import _sanitise_category_name

        assert _sanitise_category_name("Databases") == "Databases"

    def test_multi_word_takes_first(self) -> None:
        from neuromem.system import _sanitise_category_name

        assert _sanitise_category_name("Large Databases") == "Large"

    def test_newline_separated_takes_first(self) -> None:
        from neuromem.system import _sanitise_category_name

        assert _sanitise_category_name("Databases\nStorage") == "Databases"

    def test_empty_returns_fallback(self) -> None:
        from neuromem.system import _sanitise_category_name

        assert _sanitise_category_name("") == "Category"

    def test_whitespace_only_returns_fallback(self) -> None:
        from neuromem.system import _sanitise_category_name

        assert _sanitise_category_name("   ") == "Category"

    def test_leading_trailing_whitespace_stripped(self) -> None:
        from neuromem.system import _sanitise_category_name

        assert _sanitise_category_name("  Databases  ") == "Databases"


# ---------------------------------------------------------------------------
# T024: US3 end-to-end decay and archival integration tests
#
# No new production code — the decay/archival behaviour was
# implemented in T018 (SQLiteAdapter.apply_decay_and_archive) and
# wired into the dreaming cycle in T020 (_run_dream_cycle step 8).
# These tests validate the full User Story 3 lifecycle end-to-end:
#
#   1. Consolidated memory with old last_accessed + force_dream →
#      gets archived, its edges stripped from the active graph.
#   2. Partially-decayed memory above the archive threshold stays
#      consolidated with a lowered access_weight.
#   3. retrieve_memories on an archived memory returns raw_content
#      but does NOT resurrect the status AND does NOT trigger LTP.
#   4. Archived memories no longer appear in search_memory results
#      for a query that would have matched them pre-archival.
# ---------------------------------------------------------------------------


class TestDecayAndArchivalEndToEnd:
    """User Story 3 — Memories Decay and Are Archived When Not Accessed."""

    def _seed_consolidated_memory(
        self,
        memory_system: NeuroMemory,
        raw_text: str = "alpha beta gamma",
    ) -> str:
        """Helper: enqueue one memory and force a dream cycle so it's
        fully consolidated with tag nodes and has_tag edges in place."""
        mem_id = memory_system.enqueue(raw_text)
        memory_system.force_dream(block=True)
        row = memory_system.storage.get_memory_by_id(mem_id)
        assert row is not None
        assert row["status"] == "consolidated"
        return mem_id

    def test_old_memory_gets_archived_after_force_dream(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        """A consolidated memory with last_accessed 60 days past gets
        archived on the next decay pass."""
        mem_id = self._seed_consolidated_memory(memory_system)

        # Simulate age via the public ABC: spike_access_weight sets
        # last_accessed to the supplied timestamp (and resets
        # access_weight to 1.0) for consolidated memories. Handing it a
        # 60-days-ago timestamp is the adapter-agnostic way to simulate
        # an old last-accessed without touching a private `._conn`.
        now = int(time.time())
        sixty_days_ago = now - 60 * 86400
        memory_system.storage.spike_access_weight([mem_id], sixty_days_ago)

        # Aggressive lambda so 60 days of decay pushes the weight from
        # 1.0 down to ~0.006, well below the 0.1 archive threshold.
        memory_system.storage.apply_decay_and_archive(
            decay_lambda=1e-6,  # half-life ~8 days
            archive_threshold=0.1,
            current_timestamp=now,
        )

        # The memory must now be archived.
        row = memory_system.storage.get_memory_by_id(mem_id)
        assert row["status"] == "archived"

    def test_recently_accessed_memory_stays_consolidated(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        """A consolidated memory with last_accessed 7 days past and a
        gentle decay lambda should stay consolidated — just with a
        slightly lower access_weight."""
        mem_id = self._seed_consolidated_memory(memory_system)

        now = int(time.time())
        seven_days_ago = now - 7 * 86400
        memory_system.storage.spike_access_weight([mem_id], seven_days_ago)

        memory_system.storage.apply_decay_and_archive(
            decay_lambda=3e-7,  # ~30-day half-life
            archive_threshold=0.1,
            current_timestamp=now,
        )

        row = memory_system.storage.get_memory_by_id(mem_id)
        assert row["status"] == "consolidated"
        # access_weight should have dropped slightly but not below 0.5
        # (7 days * 3e-7 = ~0.18 lambda-t → exp(-0.18) ≈ 0.83)
        assert 0.5 < row["access_weight"] < 1.0

    def test_archival_strips_has_tag_edges(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        """When a memory is archived, its has_tag edges must be
        removed from the active graph (but child_of edges between
        nodes must be preserved — the I-3 invariant from the T018
        contract).

        Verified through the public ABC by traversing the graph from
        every node with ``get_subgraph(depth=2)``: the memory should
        appear as a has_tag source pre-archival and should be gone
        post-archival. This is adapter-agnostic — it works for any
        ``StorageAdapter`` implementation, not just SQLite."""
        mem_id = self._seed_consolidated_memory(memory_system, "python")

        # Confirm the memory shows up via has_tag edges before archival.
        nodes = memory_system.storage.get_all_nodes()
        assert nodes, "consolidation should have produced tag nodes"
        node_ids = [n["id"] for n in nodes]

        def collect_has_tag_sources() -> set[str]:
            """Return the set of memory IDs reachable via has_tag edges
            from any currently-stored node. Uses only the public ABC."""
            sub = memory_system.storage.get_subgraph(node_ids, depth=2)
            return {e["source_id"] for e in sub["edges"] if e["relationship"] == "has_tag"}

        assert mem_id in collect_has_tag_sources()

        # Force archival through the public ABC.
        now = int(time.time())
        year_ago = now - 365 * 86400
        memory_system.storage.spike_access_weight([mem_id], year_ago)
        memory_system.storage.apply_decay_and_archive(
            decay_lambda=1e-6,
            archive_threshold=0.1,
            current_timestamp=now,
        )
        assert memory_system.storage.get_memory_by_id(mem_id)["status"] == "archived"

        # The archived memory must no longer surface via has_tag edges.
        assert mem_id not in collect_has_tag_sources()

    def test_retrieve_memories_on_archived_returns_content_no_resurrect(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        """User Story 3 acceptance #2: archived memories are still
        retrievable via retrieve_memories — the content is preserved.
        But the retrieval MUST NOT flip the status back to
        consolidated, and MUST NOT trigger the LTP spike."""
        from neuromem.tools import retrieve_memories

        mem_id = self._seed_consolidated_memory(
            memory_system, "important knowledge about sqlite wal mode"
        )

        # Archive it via the public ABC.
        now = int(time.time())
        year_ago = now - 365 * 86400
        memory_system.storage.spike_access_weight([mem_id], year_ago)
        memory_system.storage.apply_decay_and_archive(
            decay_lambda=1e-6,
            archive_threshold=0.1,
            current_timestamp=now,
        )
        assert memory_system.storage.get_memory_by_id(mem_id)["status"] == "archived"

        # Retrieve the archived memory via the tool.
        results = retrieve_memories([mem_id], memory_system)
        assert len(results) == 1
        row = results[0]

        # Content intact.
        assert row["raw_content"] == "important knowledge about sqlite wal mode"
        assert row["summary"] is not None

        # Status stays archived — NOT resurrected to consolidated.
        assert row["status"] == "archived"

        # LTP did NOT spike it — the access_weight in the result must
        # match the archived value (around 0.02 * exp(-1e-6 * 365 days)
        # which rounds to near zero), NOT 1.0.
        assert row["access_weight"] < 0.1
        assert row["access_weight"] != pytest.approx(1.0)

        # Storage row matches.
        stored = memory_system.storage.get_memory_by_id(mem_id)
        assert stored["status"] == "archived"
        assert stored["access_weight"] < 0.1

    def test_archived_memory_not_in_search_results(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        """A query that would have matched a memory before archival
        must no longer surface it after archival — the has_tag edge
        is gone, so get_subgraph can't reach the memory from any
        nearby node."""
        from neuromem.tools import search_memory

        # Seed two memories: one will get archived, the other stays.
        archived_id = self._seed_consolidated_memory(memory_system, "alpha beta gamma")
        live_id = memory_system.enqueue("delta epsilon zeta")
        memory_system.force_dream(block=True)

        # Verify the archived memory IS currently findable pre-archival.
        pre_result = search_memory("alpha", memory_system)
        assert archived_id in pre_result

        # Archive it via the public ABC.
        now = int(time.time())
        year_ago = now - 365 * 86400
        memory_system.storage.spike_access_weight([archived_id], year_ago)
        memory_system.storage.apply_decay_and_archive(
            decay_lambda=1e-6,
            archive_threshold=0.1,
            current_timestamp=now,
        )

        # The archived memory must NOT appear in the new search result.
        post_result = search_memory("alpha", memory_system)
        assert archived_id not in post_result
        # The live memory must still appear if we search for it.
        live_result = search_memory("delta", memory_system)
        assert live_id in live_result

    def test_content_preserved_through_archival(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        """Archival MUST preserve raw_content and summary — the
        memory row stays in storage with all its text intact.
        Per FR-015."""
        raw = "the quick brown fox jumps over the lazy dog"
        mem_id = self._seed_consolidated_memory(memory_system, raw)

        pre_summary = memory_system.storage.get_memory_by_id(mem_id)["summary"]

        # Archive via the public ABC.
        now = int(time.time())
        year_ago = now - 365 * 86400
        memory_system.storage.spike_access_weight([mem_id], year_ago)
        memory_system.storage.apply_decay_and_archive(
            decay_lambda=1e-6,
            archive_threshold=0.1,
            current_timestamp=now,
        )

        # Content preserved across archival.
        post = memory_system.storage.get_memory_by_id(mem_id)
        assert post["status"] == "archived"
        assert post["raw_content"] == raw
        assert post["summary"] == pre_summary


# ---------------------------------------------------------------------------
# T026 — US5 force_dream acceptance tests
# ---------------------------------------------------------------------------
#
# Four acceptance scenarios from spec.md User Story 5
# ("developer can manually flush the inbox"):
#
#   (a) 3 inbox memories + dream_threshold=10 + force_dream(block=True)
#       → all 3 become consolidated without the threshold being hit.
#   (b) empty inbox + force_dream(block=True)
#       → returns immediately, no error, no state change.
#   (c) force_dream(block=False)
#       → returns before dreaming finishes (is_dreaming still True briefly).
#   (d) force_dream while a previous cycle is in progress
#       → does NOT spawn a second thread.
#
# These overlap intentionally with the T020/T021 thread-concurrency tests
# above; they exist as a cleanly-labelled US5 acceptance checklist a
# reviewer can tick off against the spec.


class TestForceDreamAcceptance:
    def test_us5_a_force_dream_block_true_consolidates_below_threshold(
        self,
        mock_llm: MockLLMProvider,
        mock_embedder: MockEmbeddingProvider,
    ) -> None:
        """US5 (a): force_dream(block=True) processes the entire inbox
        even when its size is well below the automatic dream threshold.
        """
        system = NeuroMemory(
            storage=SQLiteAdapter(":memory:"),
            llm=mock_llm,
            embedder=mock_embedder,
            dream_threshold=10,
        )
        for i in range(3):
            system.enqueue(f"memory number {i}")

        # Pre-condition: 3 inbox, 0 consolidated, threshold not reached.
        assert system.storage.count_memories_by_status("inbox") == 3
        assert system._dream_thread is None
        assert system.is_dreaming is False

        system.force_dream(block=True)

        # Post-condition: all 3 consolidated without threshold spawn.
        assert system.storage.count_memories_by_status("inbox") == 0
        assert system.storage.count_memories_by_status("consolidated") == 3
        assert system.is_dreaming is False

    def test_us5_b_force_dream_block_true_on_empty_inbox_is_noop(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        """US5 (b): force_dream(block=True) with an empty inbox returns
        without raising and without changing any state."""
        # Pre-condition: completely empty storage.
        assert memory_system.storage.count_memories_by_status("inbox") == 0
        assert memory_system.storage.count_memories_by_status("consolidated") == 0
        assert memory_system.is_dreaming is False

        # Should return quickly and silently.
        start = time.perf_counter()
        memory_system.force_dream(block=True)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        assert elapsed_ms < 500.0, f"empty-inbox force_dream took {elapsed_ms:.1f} ms"

        # Post-condition: nothing changed.
        assert memory_system.storage.count_memories_by_status("inbox") == 0
        assert memory_system.storage.count_memories_by_status("consolidated") == 0
        assert memory_system.is_dreaming is False

    def test_us5_c_force_dream_block_false_returns_before_dreaming_finishes(
        self,
        mock_embedder: MockEmbeddingProvider,
    ) -> None:
        """US5 (c): force_dream(block=False) returns before the spawned
        dream cycle finishes — observable because ``is_dreaming`` is
        still True immediately after the call returns.
        """
        blocking_llm = BlockingLLM()
        system = NeuroMemory(
            storage=SQLiteAdapter(":memory:"),
            llm=blocking_llm,
            embedder=mock_embedder,
        )
        system.enqueue("alpha beta gamma")

        system.force_dream(block=False)

        # Wait for the dream thread to enter extract_tags — at this point
        # we know the cycle is live but CAN'T finish (BlockingLLM has not
        # been released).
        assert blocking_llm.entered_event.wait(timeout=2.0), (
            "dream thread never entered extract_tags"
        )
        assert system.is_dreaming is True
        assert system._dream_thread is not None
        assert system._dream_thread.is_alive()

        # Release and clean up.
        blocking_llm.proceed_event.set()
        system._dream_thread.join(timeout=5.0)
        assert system.is_dreaming is False
        assert system.storage.count_memories_by_status("consolidated") == 1

    def test_us5_d_force_dream_does_not_spawn_second_thread_while_running(
        self,
        mock_embedder: MockEmbeddingProvider,
    ) -> None:
        """US5 (d): calling force_dream while a previous cycle is still
        in flight does NOT spawn a second thread.

        Exercises both modes:
          - block=False → silent no-op, ``_dream_thread`` unchanged.
          - block=True  → joins the existing thread, does not spawn
            another thread for the (now-empty) leftover inbox.
        """
        blocking_llm = BlockingLLM()
        system = NeuroMemory(
            storage=SQLiteAdapter(":memory:"),
            llm=blocking_llm,
            embedder=mock_embedder,
        )
        system.enqueue("first memory")
        system.force_dream(block=False)
        assert blocking_llm.entered_event.wait(timeout=2.0)

        first_thread = system._dream_thread
        assert first_thread is not None and first_thread.is_alive()

        # Second block=False — must NOT spawn a new thread.
        system.force_dream(block=False)
        assert system._dream_thread is first_thread, (
            "force_dream(block=False) while a cycle is in flight spawned a new thread"
        )

        # Release from a side-thread so block=True can join the existing
        # thread and return.
        releaser = threading.Thread(
            target=lambda: blocking_llm.proceed_event.set(),
            name="releaser",
        )
        releaser.start()

        # block=True while a cycle is in flight → join existing, then
        # run a fresh cycle (trivially no-op on the empty leftover inbox).
        system.force_dream(block=True)
        releaser.join(timeout=5.0)

        assert not first_thread.is_alive()
        assert system.is_dreaming is False
        # Exactly one consolidation path ran, so the memory is
        # consolidated exactly once.
        assert system.storage.count_memories_by_status("consolidated") == 1


# ---------------------------------------------------------------------------
# T025 — full-loop parity against DictStorageAdapter
# ---------------------------------------------------------------------------


def test_dict_adapter_full_loop(
    mock_embedder: MockEmbeddingProvider,
    mock_llm: MockLLMProvider,
) -> None:
    """Run the US1 enqueue → force_dream → build_prompt_context flow
    against ``DictStorageAdapter`` and assert it produces the same
    shape of output as the SQLite path.

    This is the behavioural end-to-end proof that
    ``NeuroMemory``/``ContextHelper`` couple only to the
    ``StorageAdapter`` ABC — not to any sqlite-specific quirk. The 14
    contract tests already assert method-by-method parity at the
    adapter boundary; this test does so at the orchestration boundary,
    which is where Principle III actually pays off.
    """
    from neuromem.context import ContextHelper

    from tests.conftest import DictStorageAdapter

    dict_system = NeuroMemory(
        storage=DictStorageAdapter(),
        llm=mock_llm,
        embedder=mock_embedder,
    )

    sqlite_system = NeuroMemory(
        storage=SQLiteAdapter(":memory:"),
        llm=mock_llm,
        embedder=mock_embedder,
    )

    # Identical content into both systems.
    raw_texts = [
        "python sqlite async",
        "python numpy vectorized",
        "rust tokio async",
    ]
    for system in (dict_system, sqlite_system):
        for t in raw_texts:
            system.enqueue(t)
        system.force_dream(block=True)

    # Both systems must end up with 3 consolidated memories and at
    # least one concept node each.
    for system in (dict_system, sqlite_system):
        assert system.storage.count_memories_by_status("inbox") == 0
        assert system.storage.count_memories_by_status("consolidated") == 3
        assert len(system.storage.get_all_nodes()) >= 1

    # ContextHelper.build_prompt_context must produce a non-empty ASCII
    # tree containing the memory markers for both backends.
    dict_tree = ContextHelper(dict_system).build_prompt_context("python")
    sqlite_tree = ContextHelper(sqlite_system).build_prompt_context("python")

    assert dict_tree != ""
    assert sqlite_tree != ""
    assert "📁" in dict_tree and "📁" in sqlite_tree
    assert "📄" in dict_tree and "📄" in sqlite_tree

    # Structural parity check: both trees should reference the same
    # number of memory markers, since the two adapters were fed
    # identical input and the MockEmbeddingProvider is deterministic.
    # (The exact node labels can differ because the dream clustering
    # depends on MockLLMProvider.generate_category_name, which is also
    # deterministic — so the trees should in fact be structurally
    # identical.)
    assert dict_tree.count("📄") == sqlite_tree.count("📄")


# ---------------------------------------------------------------------------
# Issue #44 — dream cycle must call extract_tags_batch, not per-memory loop
# ---------------------------------------------------------------------------


class TestDreamCycleUsesExtractTagsBatch:
    """The dream cycle must use the batched ``extract_tags_batch``
    method, not a serial per-memory loop. This is the performance
    invariant from issue #44 — with 100+ memories per dream cycle
    the difference is 100 serial Gemini calls vs 1 batched call.

    Verified by a counting LLM provider that tracks both method
    call counts. After one dream cycle over N memories we expect
    batch_call_count == 1 AND per_call_count == 0 (the overridden
    batch method never falls through to the per-call path)."""

    class _CountingLLM(LLMProvider):
        """LLM provider that counts both per-call and batched
        extract_tags invocations so the test can assert which
        path the dream cycle actually takes."""

        def __init__(self) -> None:
            self.per_call_count = 0
            self.batch_call_count = 0
            self.batch_sizes: list[int] = []

        def generate_summary(self, raw_text: str) -> str:
            return raw_text[:80]

        def extract_tags(self, summary: str) -> list[str]:
            # Used only by the default fallback or a cycle that
            # incorrectly calls the per-memory path.
            self.per_call_count += 1
            return [w for w in summary.split() if w.isalpha()][:3]

        def extract_tags_batch(self, summaries: list[str]) -> list[list[str]]:
            self.batch_call_count += 1
            self.batch_sizes.append(len(summaries))
            return [[w for w in s.split() if w.isalpha()][:3] for s in summaries]

        def generate_category_name(self, concepts: list[str]) -> str:
            return "Cat"

    def test_dream_cycle_calls_batch_exactly_once_per_cycle(
        self,
        mock_embedder: MockEmbeddingProvider,
    ) -> None:
        """Enqueue 5 memories, force_dream once, assert the counting
        LLM saw exactly one batched call of size 5 and ZERO per-call
        extract_tags invocations. Proves the dream cycle uses the
        batched path, not the legacy serial loop."""
        llm = self._CountingLLM()
        system = NeuroMemory(
            storage=SQLiteAdapter(":memory:"),
            llm=llm,
            embedder=mock_embedder,
        )
        for i in range(5):
            system.enqueue(f"memory {i} alpha beta gamma")
        system.force_dream(block=True)

        assert llm.batch_call_count == 1
        assert llm.batch_sizes == [5]
        assert llm.per_call_count == 0
        # Memories all landed as consolidated, proving the pipeline
        # still works end-to-end with the batched path.
        assert system.storage.count_memories_by_status("consolidated") == 5

    def test_dream_cycle_with_default_fallback_still_works(
        self,
        mock_embedder: MockEmbeddingProvider,
        mock_llm: MockLLMProvider,
    ) -> None:
        """A provider that doesn't override ``extract_tags_batch``
        (like the shipped ``MockLLMProvider``) gets the default
        fallback, which calls ``extract_tags`` once per summary.
        The dream cycle still completes correctly — this is the
        backwards-compatibility guarantee of the #44 fix.
        """
        system = NeuroMemory(
            storage=SQLiteAdapter(":memory:"),
            llm=mock_llm,
            embedder=mock_embedder,
        )
        for i in range(4):
            system.enqueue(f"memory {i} alpha beta")
        system.force_dream(block=True)
        assert system.storage.count_memories_by_status("consolidated") == 4


# ---------------------------------------------------------------------------
# Named-entity extraction integration (dream cycle)
# ---------------------------------------------------------------------------


class TestNamedEntityExtractionInDreamCycle:
    """The dream cycle was extended to call
    ``LLMProvider.extract_named_entities_batch`` after tag extraction
    and persist the result via ``StorageAdapter.set_named_entities``.

    These tests lock in:
    1. Providers that don't override extract_named_entities (i.e.
       inherit the ABC default returning ``[]``) cause no regressions —
       consolidated memories come out with ``named_entities == []``.
    2. A provider that DOES override extract_named_entities_batch has
       its return value persisted onto each memory — readable via
       both ``get_memory_by_id`` and ``get_memories_by_status``.
    3. The length invariant enforced by ``zip(..., strict=True)`` is
       respected: a batch response of wrong length surfaces as a
       ``ValueError`` inside the dream cycle rather than silently
       mis-pairing entities with memories.
    """

    def test_default_provider_persists_empty_entities(
        self,
        memory_system: NeuroMemory,
    ) -> None:
        """MockLLMProvider doesn't override NER — after consolidation
        each memory still has ``named_entities == []``."""
        for i in range(3):
            memory_system.enqueue(f"memory {i} alpha beta")
        memory_system.force_dream(block=True)

        memories = memory_system.storage.get_memories_by_status("consolidated")
        assert len(memories) == 3
        for mem in memories:
            assert mem["named_entities"] == []

    def test_overridden_provider_populates_entities(
        self,
        mock_embedder: MockEmbeddingProvider,
    ) -> None:
        """A provider that overrides extract_named_entities_batch drives
        per-memory entity persistence through to storage."""

        class NamingLLM(MockLLMProvider):
            """Returns a deterministic entity list per summary — the
            first word of each summary, so we can assert ordering."""

            def extract_named_entities_batch(self, summaries: list[str]) -> list[list[str]]:
                return [[s.split()[0]] if s else [] for s in summaries]

        system = NeuroMemory(
            storage=SQLiteAdapter(":memory:"),
            llm=NamingLLM(),
            embedder=mock_embedder,
        )
        # MockLLMProvider.generate_summary returns raw_text truncated —
        # the first word is therefore the first word of the raw_text.
        system.enqueue("Target sells creamer")
        system.enqueue("Cartwheel saves money")
        system.enqueue("coupon expiring soon")
        system.force_dream(block=True)

        memories = sorted(
            system.storage.get_memories_by_status("consolidated"),
            key=lambda m: m["raw_content"],
        )
        assert [m["named_entities"] for m in memories] == [
            ["Cartwheel"],
            ["Target"],
            ["coupon"],
        ]

    def test_batch_length_mismatch_raises(
        self,
        mock_embedder: MockEmbeddingProvider,
    ) -> None:
        """A misbehaving provider that returns a list of the wrong
        length fails loudly via zip(strict=True). Silent mis-pairing
        would corrupt the entity→memory mapping, which is far worse
        than a dream-cycle failure."""

        class BrokenNERLLM(MockLLMProvider):
            def extract_named_entities_batch(self, summaries: list[str]) -> list[list[str]]:
                # Deliberately wrong length: one item for two summaries.
                return [["oops"]]

        system = NeuroMemory(
            storage=SQLiteAdapter(":memory:"),
            llm=BrokenNERLLM(),
            embedder=mock_embedder,
        )
        system.enqueue("first memory")
        system.enqueue("second memory")
        # _run_dream_cycle swallows errors and logs; but the broken
        # invariant means memories don't make it to consolidated.
        # Either the dream cycle raises, or it fails silently and
        # leaves the memories in 'dreaming'. Assert on the observable
        # outcome: zero consolidated memories.
        with pytest.raises(ValueError, match="shorter than argument"):
            system.force_dream(block=True)


def test_explicit_ignore_unused_embedding_provider_import() -> None:
    """Referenced for type-checker compatibility."""
    # EmbeddingProvider is referenced by type annotations in test fixtures.
    _ = EmbeddingProvider
