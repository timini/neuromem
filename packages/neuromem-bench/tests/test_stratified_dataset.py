"""Tests for the ``_StratifiedDataset`` helper in the CLI script.

The helper is defined inside ``scripts/run_longmemeval.py`` rather
than the package so that it's trivially accessible to whoever is
tweaking the CLI. These tests import it by file path to exercise
the contract: N of each question_type, in file order, with early-
exit once saturation is reached.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Iterator
from pathlib import Path

from neuromem_bench.datasets.base import BenchInstance, BenchSession, BenchTurn, Dataset

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_longmemeval.py"
_spec = importlib.util.spec_from_file_location("run_longmemeval_script", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
StratifiedDataset = _module._StratifiedDataset


def _make_instance(qid: str, qtype: str) -> BenchInstance:
    return BenchInstance(
        instance_id=qid,
        sessions=[
            BenchSession(
                session_id="s",
                turns=[BenchTurn(role="user", text="hi")],
            )
        ],
        question="?",
        gold_answer="answer",
        question_type=qtype,
    )


class _ListDataset(Dataset):
    def __init__(self, instances: list[BenchInstance]) -> None:
        self._instances = instances

    @property
    def name(self) -> str:
        return "list"

    @property
    def split(self) -> str:
        return "t"

    def load(self, *, limit: int | None = None) -> Iterator[BenchInstance]:
        for i, inst in enumerate(self._instances):
            if limit is not None and i >= limit:
                return
            yield inst


class TestStratifiedDataset:
    def test_caps_each_type_at_per_type(self) -> None:
        # 5 of type A, 3 of type B, 4 of type C. per_type=2 →
        # 2+2+2 = 6 instances out.
        instances = (
            [_make_instance(f"a{i}", "A") for i in range(5)]
            + [_make_instance(f"b{i}", "B") for i in range(3)]
            + [_make_instance(f"c{i}", "C") for i in range(4)]
        )
        strat = StratifiedDataset(_ListDataset(instances), per_type=2)
        got = list(strat.load())
        assert len(got) == 6
        types = [i.question_type for i in got]
        assert types.count("A") == 2
        assert types.count("B") == 2
        assert types.count("C") == 2

    def test_preserves_file_order_within_type(self) -> None:
        instances = [_make_instance(f"a{i}", "A") for i in range(5)]
        strat = StratifiedDataset(_ListDataset(instances), per_type=3)
        got = list(strat.load())
        assert [i.instance_id for i in got] == ["a0", "a1", "a2"]

    def test_passthrough_name_and_split(self) -> None:
        strat = StratifiedDataset(_ListDataset([]), per_type=1)
        assert strat.name == "list"
        assert strat.split == "t"

    def test_limit_cap_still_respected(self) -> None:
        # per_type=10 would emit everything, but limit=3 caps total.
        instances = [_make_instance(f"a{i}", "A") for i in range(5)]
        strat = StratifiedDataset(_ListDataset(instances), per_type=10)
        got = list(strat.load(limit=3))
        assert len(got) == 3

    def test_missing_question_type_buckets_as_unknown(self) -> None:
        # An instance with None question_type should bucket into
        # the "unknown" category and be subject to the cap like any
        # other category.
        instances = [
            _make_instance("x0", None),  # type: ignore[arg-type]
            _make_instance("x1", None),  # type: ignore[arg-type]
            _make_instance("x2", None),  # type: ignore[arg-type]
        ]
        strat = StratifiedDataset(_ListDataset(instances), per_type=2)
        got = list(strat.load())
        assert len(got) == 2
        assert all(i.question_type is None for i in got)

    def test_early_exit_on_saturation(self) -> None:
        # 2 of type A in positions 0-1, then 200 of type A in
        # positions 2-201. per_type=2 fills A after instance 1.
        # Subsequent 200 instances all get skipped; after 100
        # consecutive skips, the stratifier should return WITHOUT
        # pulling the last 100 from the inner dataset.
        instances = [_make_instance(f"a{i}", "A") for i in range(202)]
        # Count how many the inner dataset actually yielded.
        pulled_count = 0

        class CountingDataset(Dataset):
            @property
            def name(self) -> str:
                return "count"

            @property
            def split(self) -> str:
                return "t"

            def load(self, *, limit: int | None = None) -> Iterator[BenchInstance]:
                nonlocal pulled_count
                for inst in instances:
                    pulled_count += 1
                    yield inst

        strat = StratifiedDataset(CountingDataset(), per_type=2)
        got = list(strat.load())
        assert len(got) == 2
        # Saturation threshold is 100 consecutive skips. Starting
        # pulling at instance 3, we skip 100 instances (positions 2
        # to 101 inclusive), then return. Actual pulled ≤ 2 emitted + 100 skipped.
        assert pulled_count <= 102, f"early-exit didn't fire; pulled {pulled_count}"
