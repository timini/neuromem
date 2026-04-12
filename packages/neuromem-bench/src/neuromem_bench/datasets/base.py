"""Abstract base class for benchmark dataset loaders.

Every concrete dataset loader (LongMemEval, MemoryAgentBench, etc.)
subclasses ``Dataset`` and implements ``load()`` to yield
``BenchInstance`` objects.

A ``BenchInstance`` is the unit of work the benchmark runner drives
an agent through: a sequence of input sessions (chat history the
agent should ingest into memory) + a question + a gold answer.

The shape is deliberately neutral — LongMemEval, MemBench,
MemoryAgentBench, and GraphRAG-Bench can all map onto it, even
though their native formats differ. Loaders normalise to this shape
so the runner doesn't need dataset-specific code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BenchTurn:
    """One turn in a conversation session.

    A turn is a single (role, text) pair — either a user message
    or an assistant response. Roles are lowercase strings to
    avoid enum churn across datasets.
    """

    role: str
    """Typically ``"user"`` or ``"assistant"``. Dataset-specific
    extensions (e.g. ``"tool"``, ``"system"``) are allowed."""

    text: str
    """The raw text content of the turn."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Optional per-turn metadata (timestamps, session index,
    source document ids, etc.). Datasets set this freely; the
    runner passes it through to the agent as-is."""


@dataclass
class BenchSession:
    """A grouped sequence of turns that belong to one conversation.

    LongMemEval instances typically have 40–500 sessions per
    instance, each with a handful of turns. The runner ingests
    each session's turns in order, then moves on to the next
    session. Session boundaries let the agent (or the memory
    system) reason about "when did this conversation happen" —
    which matters for temporal-reasoning questions.
    """

    session_id: str
    """Unique identifier for this session within the instance."""

    turns: list[BenchTurn]
    """Turns in order. Always non-empty — a session with no turns
    is never emitted by a well-formed loader."""

    timestamp: str | None = None
    """ISO-8601 timestamp or other dataset-specific date string,
    if the dataset provides one. ``None`` if the dataset doesn't
    track session timing."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Session-level metadata (topic tags, user id, etc.)."""


@dataclass
class BenchInstance:
    """One benchmark evaluation unit.

    Corresponds to a single "question" in the dataset. The agent
    is given the instance's full ``sessions`` history as context
    (ingested via repeated ``process_turn`` calls), then asked
    the ``question``, and its answer is scored against the
    ``gold_answer``.
    """

    instance_id: str
    """Unique identifier for this instance within the dataset."""

    sessions: list[BenchSession]
    """Ordered list of sessions to ingest. For LongMemEval,
    anywhere from 40 to 500 sessions per instance."""

    question: str
    """The question to answer after all sessions have been
    ingested."""

    gold_answer: str
    """The ground-truth answer. For exact-match scoring, matched
    as a string (case-insensitive, normalised whitespace). For
    LLM-as-judge scoring, passed as the reference."""

    question_type: str | None = None
    """Optional dataset-specific question type tag (e.g. LongMemEval's
    'single-session-user', 'multi-session', 'temporal-reasoning',
    'knowledge-update', 'abstention'). Used for per-category
    scoring breakdown in the results."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Instance-level metadata (evidence session ids, question
    date, etc.)."""


class Dataset(ABC):
    """Abstract dataset loader contract.

    Concrete subclasses implement ``load()`` to yield
    ``BenchInstance`` objects. Loaders are responsible for:

    1. Downloading the dataset's raw files (from GitHub,
       HuggingFace, or wherever the canonical source lives).
    2. Caching the download so subsequent runs don't re-download.
    3. Parsing the raw format and yielding normalised
       ``BenchInstance`` objects.

    Loaders should be **lazy** — they yield instances one at a
    time rather than loading the whole dataset into memory, so
    the runner can stream through large datasets without
    blowing up memory.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this dataset (e.g. ``"longmemeval"``)."""

    @property
    @abstractmethod
    def split(self) -> str:
        """Which split of the dataset this loader is configured for.
        Most datasets have multiple splits of different sizes
        (LongMemEval has ``s``, ``m`` for short and medium)."""

    @abstractmethod
    def load(self, *, limit: int | None = None) -> Iterator[BenchInstance]:
        """Yield benchmark instances from this dataset.

        Arguments:
            limit: Optional max number of instances to yield.
                Useful for smoke tests that want to burn through
                a small sample without downloading gigabytes.

        Yields:
            ``BenchInstance`` objects, one per benchmark question.
        """
