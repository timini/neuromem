"""Unit tests for ``NeuromemAdkAgent``.

Full end-to-end coverage of the ADK + enable_memory path already lives
in ``neuromem-adk``'s own integration test (``test_adk_integration.py``)
against real Gemini. These tests here cover the ``BaseAgent``-shaped
wiring we put on TOP of that: process_session ingestion bypassing the
Runner, reset rebuilding state, answer dispatching through an ADK
Runner-shaped mock.

No real Gemini, no real ADK — everything below ``enable_memory`` is
patched out.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# The google-adk package emits an "EXPERIMENTAL feature ... is enabled"
# UserWarning on first construction of several of its classes. The
# workspace-root ``filterwarnings=['error']`` would otherwise escalate
# this to a test failure even though it's upstream noise we can't
# meaningfully suppress at import time. Scope the filter to this module
# only — core unit tests stay strict.
pytestmark = [
    pytest.mark.filterwarnings(r"ignore:.*\[EXPERIMENTAL\].*is enabled:UserWarning"),
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    pytest.mark.filterwarnings("ignore::ResourceWarning"),
]


class _FakeMemory:
    """Stand-in for NeuroMemory. Records ``enqueue_session`` calls
    (the path NeuromemAdkAgent uses) plus any stray ``enqueue``
    calls (which shouldn't happen on this agent — enqueue is only
    used by the ADK callback path at answer time)."""

    def __init__(self) -> None:
        self.enqueues: list[tuple[str, dict]] = []
        self.session_enqueues: list[tuple[list, dict]] = []
        self.force_dream_count = 0
        self.cluster_threshold = 0.0
        self.dream_threshold = 0

    def enqueue(self, text: str, metadata: dict | None = None) -> str:
        self.enqueues.append((text, metadata or {}))
        return f"mem_{len(self.enqueues)}"

    def enqueue_session(self, turns: list, metadata: dict | None = None) -> str:
        self.session_enqueues.append((list(turns), metadata or {}))
        return f"mem_session_{len(self.session_enqueues)}"

    def force_dream(self, block: bool = True) -> None:
        self.force_dream_count += 1


def _fake_event(text: str, *, author: str = "benchmark_agent"):
    """Build an ADK-event-shaped SimpleNamespace with a single
    text-bearing part and a configurable author.

    ``answer()`` filters by author — events whose author isn't the
    agent's name are skipped (e.g. tool-result events, system
    events). Tests that want to exercise the filter pass
    ``author="some_other_thing"``. Default is ``"benchmark_agent"``,
    which matches the hard-coded agent name in NeuromemAdkAgent."""
    return SimpleNamespace(
        author=author,
        content=SimpleNamespace(parts=[SimpleNamespace(text=text)]),
    )


@pytest.fixture
def patched_adk():
    """Patch every external dep NeuromemAdkAgent._build reaches for.

    Yields a dict of mocks so individual tests can assert on how the
    agent interacted with each.
    """
    with (
        patch("google.adk.agents.Agent") as agent_cls,
        patch("google.adk.runners.Runner") as runner_cls,
        patch("google.adk.sessions.InMemorySessionService") as sess_cls,
        patch("neuromem_adk.enable_memory") as enable_memory,
        patch("neuromem_gemini.GeminiLLMProvider") as llm_cls,
        patch("neuromem_gemini.GeminiEmbeddingProvider") as embed_cls,
    ):
        fake_agent = MagicMock(name="agent_instance")
        agent_cls.return_value = fake_agent

        fake_runner = MagicMock(name="runner_instance")
        runner_cls.return_value = fake_runner

        fake_memory = _FakeMemory()
        enable_memory.return_value = fake_memory

        yield {
            "agent_cls": agent_cls,
            "runner_cls": runner_cls,
            "session_service_cls": sess_cls,
            "enable_memory": enable_memory,
            "llm_cls": llm_cls,
            "embed_cls": embed_cls,
            "fake_agent": fake_agent,
            "fake_runner": fake_runner,
            "fake_memory": fake_memory,
        }


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_constructor_builds_memory_and_runner(self, patched_adk) -> None:
        """Post-__init__, all wiring should be in place: enable_memory
        called once (with the threshold kwargs from PR #49), a Runner
        built, and the session_id populated."""
        from neuromem_bench.agent import NeuromemAdkAgent  # noqa: PLC0415

        agent = NeuromemAdkAgent(api_key="fake-key", model="gemini-flash-latest")

        assert patched_adk["enable_memory"].call_count == 1
        assert patched_adk["runner_cls"].call_count == 1

        # Thresholds are passed as enable_memory kwargs (not post-hoc
        # mutation). Inspect the call-kwargs to verify the config
        # matches NeuromemAgent's tuning for apples-to-apples
        # benchmark comparison.
        _, kwargs = patched_adk["enable_memory"].call_args
        assert kwargs["cluster_threshold"] == 0.55
        assert kwargs["dream_threshold"] == 9999

        # Session id is populated and has the expected prefix.
        assert agent._session_id.startswith("instance-")

    def test_custom_cluster_threshold_passes_through(self, patched_adk) -> None:
        """A non-default cluster_threshold on the agent flows through
        to enable_memory's cluster_threshold kwarg unchanged."""
        from neuromem_bench.agent import NeuromemAdkAgent  # noqa: PLC0415

        NeuromemAdkAgent(api_key="fake-key", cluster_threshold=0.7)
        _, kwargs = patched_adk["enable_memory"].call_args
        assert kwargs["cluster_threshold"] == 0.7


# ---------------------------------------------------------------------------
# process_session
# ---------------------------------------------------------------------------


class TestProcessSession:
    def test_enqueues_session_with_metadata(self, patched_adk) -> None:
        """Ingestion bypasses ADK — process_session goes straight to
        memory.enqueue_session. Metadata includes session_index
        (1-indexed) and turn_count. No Runner calls during ingestion."""
        from neuromem_bench.agent import NeuromemAdkAgent  # noqa: PLC0415

        agent = NeuromemAdkAgent(api_key="fake-key")
        agent.process_session(
            [
                {"role": "user", "text": "hello"},
                {"role": "assistant", "text": "hi there"},
            ]
        )

        session_enqueues = patched_adk["fake_memory"].session_enqueues
        assert len(session_enqueues) == 1
        turns, metadata = session_enqueues[0]
        assert len(turns) == 2
        assert metadata == {"session_index": 1, "turn_count": 2}

        # Per-turn enqueue should never have been hit — ingestion is
        # session-level.
        assert patched_adk["fake_memory"].enqueues == []

        # Runner was constructed in __init__ but never invoked during
        # ingestion — that's the whole point of the bypass.
        patched_adk["fake_runner"].run_debug.assert_not_called()

    def test_session_index_resets_on_rebuild(self, patched_adk) -> None:
        """reset() rebuilds state, including the session counter — so
        the next instance's first session metadata reads
        ``session_index=1``, not ``session_index=4``."""
        from neuromem_bench.agent import NeuromemAdkAgent  # noqa: PLC0415

        agent = NeuromemAdkAgent(api_key="fake-key")
        for i in range(3):
            agent.process_session([{"role": "user", "text": f"session {i}"}])
        agent.reset()
        agent.process_session([{"role": "user", "text": "new instance session"}])

        _, last_metadata = patched_adk["fake_memory"].session_enqueues[-1]
        assert last_metadata == {"session_index": 1, "turn_count": 1}

    def test_many_sessions_all_flow_through(self, patched_adk) -> None:
        from neuromem_bench.agent import NeuromemAdkAgent  # noqa: PLC0415

        agent = NeuromemAdkAgent(api_key="fake-key")
        for i in range(50):
            agent.process_session(
                [
                    {"role": "user", "text": f"q {i}"},
                    {"role": "assistant", "text": f"a {i}"},
                ]
            )

        assert len(patched_adk["fake_memory"].session_enqueues) == 50

    def test_empty_turns_list_skipped(self, patched_adk) -> None:
        """Defensive: a session with zero turns (shouldn't happen from
        LongMemEval but could from custom callers) is a no-op."""
        from neuromem_bench.agent import NeuromemAdkAgent  # noqa: PLC0415

        agent = NeuromemAdkAgent(api_key="fake-key")
        agent.process_session([])
        assert patched_adk["fake_memory"].session_enqueues == []


# ---------------------------------------------------------------------------
# answer
# ---------------------------------------------------------------------------


class TestAnswer:
    def test_answer_forces_dream_then_drives_runner(self, patched_adk) -> None:
        """The answer path:
        1. Forces a dream cycle so the graph is ready for injection.
        2. Calls runner.run_debug with the correct user_id + session_id.
        3. Returns the last text part from the event list."""
        from neuromem_bench.agent import NeuromemAdkAgent  # noqa: PLC0415

        async def fake_run_debug(*args, **kwargs):
            return [
                _fake_event("tool call noise"),
                _fake_event("Final answer with the gold detail."),
            ]

        patched_adk["fake_runner"].run_debug = fake_run_debug

        agent = NeuromemAdkAgent(api_key="fake-key")
        agent.process_session([{"role": "user", "text": "prior turn"}])
        result = agent.answer("What did I say?")

        # Dream cycle fired before the runner's first turn.
        assert patched_adk["fake_memory"].force_dream_count == 1

        # Last text-bearing part is what we return.
        assert result == "Final answer with the gold detail."

    def test_answer_returns_empty_string_on_no_text_parts(self, patched_adk) -> None:
        """Defensive: an event list with no text-bearing parts (e.g.
        only tool-call events) yields an empty string rather than
        crashing. The benchmark metric treats empty strings as 0-score."""
        from neuromem_bench.agent import NeuromemAdkAgent  # noqa: PLC0415

        async def fake_run_debug(*args, **kwargs):
            return [SimpleNamespace(author="benchmark_agent", content=None)]

        patched_adk["fake_runner"].run_debug = fake_run_debug

        agent = NeuromemAdkAgent(api_key="fake-key")
        assert agent.answer("Q?") == ""

    def test_answer_filters_events_by_author(self, patched_adk) -> None:
        """CRITICAL: events authored by anything other than the agent
        (e.g. tool-result events from retrieve_memories) MUST NOT be
        treated as the LLM's answer. If the LLM calls
        ``retrieve_memories(...)`` mid-answer and the tool returns a
        text-rich JSON result, that text would otherwise be picked
        up as the final answer — silently scoring 0 on the benchmark.

        Set up an event list where a tool-result event with a long
        text part comes AFTER the agent's actual answer. The
        author-filter must skip it and return the agent's text."""
        from neuromem_bench.agent import NeuromemAdkAgent  # noqa: PLC0415

        async def fake_run_debug(*args, **kwargs):
            return [
                _fake_event("The answer is 42.", author="benchmark_agent"),
                _fake_event(
                    '{"raw_content": "tool result blob..."}',
                    author="retrieve_memories",
                ),
            ]

        patched_adk["fake_runner"].run_debug = fake_run_debug

        agent = NeuromemAdkAgent(api_key="fake-key")
        result = agent.answer("Q?")
        assert result == "The answer is 42."
        assert "tool result blob" not in result

    def test_answer_returns_empty_when_only_tool_events_present(self, patched_adk) -> None:
        """If the LLM only emits tool-call events (no text from the
        agent itself), answer returns ''. Benchmark scores it 0; the
        empty result is a visible-enough signal in the JSONL output
        that something went wrong."""
        from neuromem_bench.agent import NeuromemAdkAgent  # noqa: PLC0415

        async def fake_run_debug(*args, **kwargs):
            return [
                _fake_event("tool blob 1", author="some_tool"),
                _fake_event("tool blob 2", author="another_tool"),
            ]

        patched_adk["fake_runner"].run_debug = fake_run_debug

        agent = NeuromemAdkAgent(api_key="fake-key")
        assert agent.answer("Q?") == ""


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_rebuilds_memory_and_rolls_session_id(self, patched_adk) -> None:
        """reset() must:
        1. Call enable_memory a second time (fresh memory).
        2. Construct a second Runner.
        3. Produce a different session_id so prior session events
           don't leak into the new instance.
        """
        from neuromem_bench.agent import NeuromemAdkAgent  # noqa: PLC0415

        agent = NeuromemAdkAgent(api_key="fake-key")
        first_session_id = agent._session_id
        first_enable_count = patched_adk["enable_memory"].call_count

        agent.reset()

        assert patched_adk["enable_memory"].call_count == first_enable_count + 1
        assert patched_adk["runner_cls"].call_count >= 2
        assert agent._session_id != first_session_id
        assert agent._session_id.startswith("instance-")
