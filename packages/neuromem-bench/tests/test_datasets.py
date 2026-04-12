"""Unit tests for neuromem_bench.datasets.

No network calls. These tests exercise the format converters and
dataclass shapes with synthetic raw input, so CI on every PR
validates the schema mapping is intact even though a real
LongMemEval run costs money and takes minutes.
"""

from __future__ import annotations

from neuromem_bench.datasets.base import (
    BenchInstance,
    BenchSession,
    BenchTurn,
    Dataset,
)
from neuromem_bench.datasets.longmemeval import LongMemEval


def test_bench_turn_dataclass_shape() -> None:
    turn = BenchTurn(role="user", text="hello")
    assert turn.role == "user"
    assert turn.text == "hello"
    assert turn.metadata == {}


def test_bench_session_dataclass_shape() -> None:
    session = BenchSession(
        session_id="s1",
        turns=[BenchTurn(role="user", text="hi")],
        timestamp="2026-04-01",
    )
    assert session.session_id == "s1"
    assert len(session.turns) == 1
    assert session.timestamp == "2026-04-01"


def test_bench_instance_dataclass_shape() -> None:
    instance = BenchInstance(
        instance_id="q1",
        sessions=[],
        question="what?",
        gold_answer="answer",
        question_type="single-session-user",
    )
    assert instance.instance_id == "q1"
    assert instance.sessions == []
    assert instance.question == "what?"
    assert instance.gold_answer == "answer"
    assert instance.question_type == "single-session-user"


def test_longmemeval_subclasses_dataset() -> None:
    ds = LongMemEval(split="s")
    assert isinstance(ds, Dataset)
    assert ds.name == "longmemeval"
    assert ds.split == "s"


def test_longmemeval_rejects_unknown_split() -> None:
    import pytest  # noqa: PLC0415

    with pytest.raises(ValueError, match="unknown split"):
        LongMemEval(split="xl")


def test_longmemeval_convert_instance_full_roundtrip() -> None:
    """Feed a synthetic raw instance through _convert_instance
    and verify the resulting BenchInstance matches expectations."""
    ds = LongMemEval(split="s")
    raw = {
        "question_id": "q_abc123",
        "question_type": "multi-session",
        "question": "What did the user say about Rex?",
        "answer": "Rex is a golden retriever.",
        "question_date": "2026-04-15",
        "haystack_session_ids": ["sess-1", "sess-2"],
        "haystack_dates": ["2026-04-10", "2026-04-12"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "My dog is named Rex."},
                {"role": "assistant", "content": "Got it, Rex.", "has_answer": True},
            ],
            [
                {"role": "user", "content": "Rex is a golden retriever."},
                {"role": "assistant", "content": "Noted."},
            ],
        ],
        "answer_session_ids": ["sess-1", "sess-2"],
    }
    instance = ds._convert_instance(raw)

    assert instance.instance_id == "q_abc123"
    assert instance.question == "What did the user say about Rex?"
    assert instance.gold_answer == "Rex is a golden retriever."
    assert instance.question_type == "multi-session"

    assert len(instance.sessions) == 2
    assert instance.sessions[0].session_id == "sess-1"
    assert instance.sessions[0].timestamp == "2026-04-10"
    assert len(instance.sessions[0].turns) == 2
    assert instance.sessions[0].turns[0].role == "user"
    assert instance.sessions[0].turns[0].text == "My dog is named Rex."
    assert instance.sessions[0].turns[1].metadata["has_answer"] is True

    assert instance.sessions[1].session_id == "sess-2"
    assert instance.sessions[1].timestamp == "2026-04-12"

    assert instance.metadata["question_date"] == "2026-04-15"
    assert instance.metadata["answer_session_ids"] == ["sess-1", "sess-2"]


def test_longmemeval_convert_instance_skips_empty_turns() -> None:
    """Turns with empty content are skipped — this matches the
    BenchSession contract that every session has at least one turn."""
    ds = LongMemEval(split="s")
    raw = {
        "question_id": "q",
        "question": "?",
        "answer": "!",
        "haystack_session_ids": ["s"],
        "haystack_dates": [None],
        "haystack_sessions": [
            [
                {"role": "user", "content": ""},
                {"role": "assistant", "content": "real content"},
                {"role": "user", "content": "   "},  # whitespace-only kept (has content)
            ]
        ],
    }
    instance = ds._convert_instance(raw)
    # Only the two content-bearing turns; the empty-string one drops.
    assert len(instance.sessions) == 1
    texts = [t.text for t in instance.sessions[0].turns]
    assert "real content" in texts
    assert "" not in texts
