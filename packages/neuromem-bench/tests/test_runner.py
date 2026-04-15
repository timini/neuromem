"""Runner tests — in particular, the metadata/trace capture path.

These tests avoid touching any real LLM/embedder: they drive
``run_benchmark`` with an in-memory fake dataset and a stub agent
that sets ``last_trace`` / ``last_tool_calls`` manually. The point
is to verify the runner's plumbing (how it pulls diagnostic fields
off the agent and serialises them into the JSONL), not the agents
themselves.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from neuromem_bench import run_benchmark
from neuromem_bench.datasets.base import (
    BenchInstance,
    BenchSession,
    BenchTurn,
    Dataset,
)


class _FakeDataset(Dataset):
    """Yields two hand-built instances for fast, offline tests."""

    @property
    def name(self) -> str:
        return "fake"

    @property
    def split(self) -> str:
        return "t"

    def load(self, *, limit: int | None = None) -> Iterator[BenchInstance]:
        instances = [
            BenchInstance(
                instance_id="inst-1",
                sessions=[
                    BenchSession(
                        session_id="s1",
                        turns=[BenchTurn(role="user", text="hello")],
                    )
                ],
                question="what did I say?",
                gold_answer="hello",
                question_type="single-session-user",
            ),
            BenchInstance(
                instance_id="inst-2",
                sessions=[
                    BenchSession(
                        session_id="s2",
                        turns=[BenchTurn(role="user", text="cats")],
                    )
                ],
                question="what animal?",
                gold_answer="cat",
                question_type="single-session-user",
            ),
        ]
        for idx, inst in enumerate(instances):
            if limit is not None and idx >= limit:
                return
            yield inst


class _StubAgent:
    """Stub that sets last_trace + last_tool_calls per answer.

    Mirrors the shape ``NeuromemAgent`` / ``NeuromemAdkAgent``
    populate. The runner pulls from attrs via getattr, so we just
    need the attrs to exist on the instance at answer-return time.
    """

    def __init__(self, *, include_trace: bool = True) -> None:
        self._include_trace = include_trace
        self.last_tool_calls: dict[str, int] = {}
        # Intentionally not pre-setting last_trace so we test the
        # getattr-with-default path when include_trace=False.

    def process_session(self, turns: list[dict[str, str]]) -> None:  # noqa: ARG002
        return

    def answer(self, question: str) -> str:  # noqa: ARG002
        self.last_tool_calls = {"search_memory": 1}
        if self._include_trace:
            self.last_trace: dict[str, Any] = {
                "context_tree": "root\n  - memory: hello world",
                "context_tree_chars": 26,
                "top_k": 20,
                "force_dream_wall_s": 0.01,
                "build_context_wall_s": 0.005,
                "answer_wall_s": 0.02,
            }
        return "hello"

    def reset(self) -> None:
        return


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class TestTraceCapture:
    def test_trace_on_by_default(self, tmp_path: Path) -> None:
        out = tmp_path / "results.jsonl"
        run_benchmark(
            dataset=_FakeDataset(),
            agent=_StubAgent(include_trace=True),
            metric=lambda p, g, q: 1.0 if g in p else 0.0,  # noqa: ARG005
            metric_name="contains",
            output_jsonl=out,
            verbose=False,
            # trace defaults to True
        )
        rows = _read_jsonl(out)
        assert len(rows) == 2
        for row in rows:
            trace = row["metadata"]["trace"]
            assert trace["context_tree"].startswith("root")
            assert trace["top_k"] == 20
            assert isinstance(trace["force_dream_wall_s"], float)
            assert isinstance(trace["build_context_wall_s"], float)
            assert isinstance(trace["answer_wall_s"], float)
            assert trace["context_tree_chars"] == 26
            # Existing last_tool_calls path still populates its own key.
            assert row["metadata"]["tool_calls"] == {"search_memory": 1}

    def test_no_trace_suppresses_key(self, tmp_path: Path) -> None:
        out = tmp_path / "results.jsonl"
        run_benchmark(
            dataset=_FakeDataset(),
            agent=_StubAgent(include_trace=True),
            metric=lambda p, g, q: 1.0 if g in p else 0.0,  # noqa: ARG005
            metric_name="contains",
            output_jsonl=out,
            verbose=False,
            trace=False,
        )
        rows = _read_jsonl(out)
        for row in rows:
            assert "trace" not in row["metadata"]
            # --no-trace must not break the unrelated tool_calls field.
            assert row["metadata"]["tool_calls"] == {"search_memory": 1}
            # Score/prediction/timing fields still present.
            assert isinstance(row["score"], float)
            assert row["prediction"] == "hello"
            assert isinstance(row["wall_time_s"], float)

    def test_trace_skipped_if_agent_lacks_last_trace(self, tmp_path: Path) -> None:
        """Agents without last_trace (NullAgent, NaiveRagAgent) must
        silently fall through — trace key absent, no exception."""
        out = tmp_path / "results.jsonl"
        run_benchmark(
            dataset=_FakeDataset(),
            agent=_StubAgent(include_trace=False),
            metric=lambda p, g, q: 0.5,  # noqa: ARG005
            metric_name="noop",
            output_jsonl=out,
            verbose=False,
            trace=True,
        )
        rows = _read_jsonl(out)
        for row in rows:
            assert "trace" not in row["metadata"]


class _FlakyAgent:
    """Agent that raises on every second answer() call. Exists to
    prove instance-level resilience: the runner must emit an error
    row and keep going, not tear down the whole bench."""

    def __init__(self) -> None:
        self._call = 0
        self.last_tool_calls: dict[str, int] = {}

    def process_session(self, turns: list[dict[str, str]]) -> None:  # noqa: ARG002
        return

    def answer(self, question: str) -> str:
        self._call += 1
        if self._call % 2 == 0:
            raise RuntimeError(f"simulated flake on call {self._call}")
        return "ok"

    def reset(self) -> None:
        return


class _CountingFactoryDataset(Dataset):
    """Yields ``n`` distinct instances for parallelism tests."""

    def __init__(self, n: int) -> None:
        self._n = n

    @property
    def name(self) -> str:
        return "count"

    @property
    def split(self) -> str:
        return "t"

    def load(self, *, limit: int | None = None) -> Iterator[BenchInstance]:
        count = self._n if limit is None else min(self._n, limit)
        for i in range(count):
            yield BenchInstance(
                instance_id=f"inst-{i:03d}",
                sessions=[
                    BenchSession(
                        session_id=f"s-{i}",
                        turns=[BenchTurn(role="user", text=f"hi {i}")],
                    )
                ],
                question="?",
                gold_answer="answer",
                question_type="single-session-user",
            )


class TestResilience:
    def test_flaky_agent_produces_error_rows(self, tmp_path: Path) -> None:
        """One bad instance must not kill the whole run."""
        out = tmp_path / "results.jsonl"
        run_benchmark(
            dataset=_CountingFactoryDataset(n=6),
            agent=_FlakyAgent(),
            metric=lambda p, g, q: 1.0 if p == "ok" else 0.0,  # noqa: ARG005
            metric_name="eq",
            output_jsonl=out,
            verbose=False,
        )
        rows = _read_jsonl(out)
        # All 6 rows emitted (none swallowed despite exceptions).
        assert len(rows) == 6
        # Even-index calls raised → 3 error rows; odd-index calls
        # succeeded → 3 ok rows.
        error_rows = [r for r in rows if "error" in r["metadata"]]
        ok_rows = [r for r in rows if "error" not in r["metadata"]]
        assert len(error_rows) == 3
        assert len(ok_rows) == 3
        # Error rows are shaped correctly.
        for r in error_rows:
            assert r["score"] == 0.0
            assert r["prediction"] == ""
            assert r["metadata"]["error_type"] == "RuntimeError"
            assert "simulated flake" in r["metadata"]["error"]
        # OK rows have real predictions.
        for r in ok_rows:
            assert r["prediction"] == "ok"
            assert r["score"] == 1.0


class _ParallelAgent:
    """Trivial agent for parallelism testing. Tracks which thread
    built it so a test can assert workers get distinct instances."""

    def __init__(self) -> None:
        import threading  # noqa: PLC0415

        self.built_by_thread = threading.get_ident()
        self.last_tool_calls: dict[str, int] = {}

    def process_session(self, turns: list[dict[str, str]]) -> None:  # noqa: ARG002
        return

    def answer(self, question: str) -> str:  # noqa: ARG002
        # Small sleep so threads overlap meaningfully.
        import time as _t  # noqa: PLC0415

        _t.sleep(0.02)
        return "x"

    def reset(self) -> None:
        return


class TestParallelism:
    def test_workers_gt_1_dispatches_across_threads(self, tmp_path: Path) -> None:
        out = tmp_path / "results.jsonl"
        built_agents: list[_ParallelAgent] = []
        build_lock: Any = __import__("threading").Lock()

        def factory() -> _ParallelAgent:
            a = _ParallelAgent()
            with build_lock:
                built_agents.append(a)
            return a

        run_benchmark(
            dataset=_CountingFactoryDataset(n=20),
            agent_factory=factory,
            metric=lambda p, g, q: 1.0 if p == "x" else 0.0,  # noqa: ARG005
            metric_name="eq",
            output_jsonl=out,
            verbose=False,
            workers=4,
        )
        rows = _read_jsonl(out)
        assert len(rows) == 20
        # All instance_ids appear exactly once (no dupes, none dropped).
        ids = sorted(r["instance_id"] for r in rows)
        assert ids == sorted(f"inst-{i:03d}" for i in range(20))
        # At least 2 distinct worker threads actually built agents
        # (not a strict N=4 — ThreadPool may reuse a warm thread if
        # tasks come back fast enough — but parallelism must be real).
        distinct_threads = {a.built_by_thread for a in built_agents}
        assert len(distinct_threads) >= 2, f"expected ≥2 workers, saw {len(distinct_threads)}"

    def test_workers_gt_1_requires_factory(self, tmp_path: Path) -> None:
        import pytest  # noqa: PLC0415

        with pytest.raises(ValueError, match="workers > 1"):
            run_benchmark(
                dataset=_CountingFactoryDataset(n=1),
                agent=_StubAgent(include_trace=False),
                workers=2,
                output_jsonl=tmp_path / "x.jsonl",
                verbose=False,
            )

    def test_mutually_exclusive_agent_and_factory(self, tmp_path: Path) -> None:
        import pytest  # noqa: PLC0415

        with pytest.raises(ValueError, match="exactly one"):
            run_benchmark(
                dataset=_CountingFactoryDataset(n=1),
                agent=_StubAgent(include_trace=False),
                agent_factory=_StubAgent,
                output_jsonl=tmp_path / "x.jsonl",
                verbose=False,
            )

    def test_missing_both_agent_and_factory(self, tmp_path: Path) -> None:
        import pytest  # noqa: PLC0415

        with pytest.raises(ValueError, match="agent or agent_factory"):
            run_benchmark(
                dataset=_CountingFactoryDataset(n=1),
                output_jsonl=tmp_path / "x.jsonl",
                verbose=False,
            )
