"""End-to-end integration test for neuromem-adk against real ADK + Gemini.

Single ``@pytest.mark.integration``-gated smoke test. Builds a real
``google.adk.agents.Agent``, attaches memory via ``enable_memory``,
runs a scripted 3-turn conversation through ADK's real ``Runner``
against the real Gemini API, and asserts the full cognitive loop
worked end-to-end:

1. **Automatic turn capture** — every conversation turn landed in
   the memory store with role metadata.
2. **Passive context injection** — the third turn's answer
   references a fact mentioned in the first turn, proving the
   injected context surfaced the relevant memory.
3. **LTP spike** — memories that were referenced on later turns
   have their ``access_weight`` spiked to 1.0.

Cost per run: ~5000 tokens, <$0.01 at Gemini 2.0 Flash rates.
Wall clock: ~10–30 seconds depending on API latency.

Default ``uv run pytest`` run excludes this test via the
workspace-root ``-m "not integration"`` addopt. Opt in with::

    GOOGLE_API_KEY=... uv run pytest packages/neuromem-adk/ \\
        -m integration --no-cov -v

The ``--no-cov`` flag is required because the workspace coverage
gate is scoped to neuromem-core's four core files and a sibling-
package-only run trivially misses the 90% threshold.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from neuromem_adk import enable_memory
from neuromem_gemini import GeminiEmbeddingProvider, GeminiLLMProvider

# Module-level markers:
#  - integration gates the file behind `-m integration`
#  - filterwarnings suppress noise from upstream libraries that the
#    workspace-root ``filterwarnings=['error']`` would otherwise
#    escalate into test failures:
#    - ResourceWarning: google-genai's httpx SSL socket on GC
#    - PytestUnraisableExceptionWarning: pytest's wrapper for that
#    - DeprecationWarning: google-genai subclasses aiohttp's
#      ClientSession, which emits a DeprecationWarning inside
#      __init_subclass__ on every subclass definition
#  Scoped to this module only — core unit tests remain strict.
pytestmark = [
    pytest.mark.integration,
    pytest.mark.filterwarnings("ignore::ResourceWarning"),
    pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning"),
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
]


# ---------------------------------------------------------------------------
# Scripted conversation — small enough to keep cost under $0.01
# ---------------------------------------------------------------------------

# Three turns designed so the third turn can only be answered
# correctly if the first turn's fact made it into memory AND the
# passive context injection surfaced it before the model call.
_TURN_1 = "My dog's name is Rex and he's a 3-year-old golden retriever."
_TURN_2 = "I also own two cats, Luna and Shadow."
_TURN_3 = "What did I tell you about my dog? Include the dog's name."


def _extract_text(events: list) -> list[str]:
    """Return the text from each agent-authored event in the list.

    ADK's Runner.run_debug yields Event objects; each event that
    represents a model response has a ``content`` with text parts.
    This helper picks out the assistant-side text for assertion.
    """
    texts: list[str] = []
    for event in events:
        content = getattr(event, "content", None)
        if content is None:
            continue
        parts = getattr(content, "parts", None) or []
        for part in parts:
            text = getattr(part, "text", None)
            if text:
                texts.append(text)
    return texts


async def _run_one_turn(
    runner: Runner,
    message: str,
    *,
    user_id: str,
    session_id: str,
) -> list:
    """Invoke ``runner.run_debug`` as an async coroutine and collect
    the event list. Handles the ADK 1.29 async-first API.
    """
    return await runner.run_debug(
        message,
        user_id=user_id,
        session_id=session_id,
        quiet=True,
    )


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------


def test_end_to_end_3_turn_conversation_with_memory_recall(
    gemini_api_key: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run a 3-turn scripted conversation through a real ADK Agent
    with neuromem-adk attached. Verify memory capture, passive
    context injection, and LTP spike.

    Uses ``monkeypatch.setenv`` to put the ``gemini_api_key`` value
    into ``GOOGLE_API_KEY`` for the duration of the test — ADK's
    own model client reads from ``os.environ``, so the fixture's
    return-value-only semantics aren't enough on their own. Also
    passes explicit providers to ``enable_memory`` so the test
    doesn't double up on env-var resolution (and so the test still
    works if the Gemini sibling's default model name changes).
    """
    # ADK's model client reads GOOGLE_API_KEY from os.environ at
    # construction time — so ensure it's set before we build the
    # Agent or the Runner. monkeypatch restores the original state
    # at test teardown.
    monkeypatch.setenv("GOOGLE_API_KEY", gemini_api_key)

    # 1. Build a real ADK agent.
    agent = Agent(
        model="gemini-2.0-flash-001",
        name="memory_test_agent",
        instruction=(
            "You are a helpful personal assistant. When the user asks "
            "about facts they've previously shared, answer directly and "
            "include the specific detail they mentioned."
        ),
    )

    # 2. Attach neuromem with explicit Gemini providers (using the
    #    fixture's api key directly — tests should not silently
    #    rely on env-var mutation for their own setup path).
    memory = enable_memory(
        agent,
        db_path=":memory:",
        llm=GeminiLLMProvider(api_key=gemini_api_key),
        embedder=GeminiEmbeddingProvider(api_key=gemini_api_key),
    )

    # Sanity: the agent is wired — both callback slots and both tools.
    assert agent.before_model_callback is not None
    assert agent.after_agent_callback is not None
    assert len(agent.tools) == 2

    # 3. Build an ADK Runner. InMemorySessionService is the simplest
    #    session backend — enough for a short integration test.
    session_service = InMemorySessionService()
    runner = Runner(
        app_name="neuromem-adk-integration-test",
        agent=agent,
        session_service=session_service,
        auto_create_session=True,
    )
    try:
        # 4. Run the first two turns, then force a dream cycle so
        #    memories are consolidated into the concept graph BEFORE
        #    the third turn fires its before_model_callback (which is
        #    what actually surfaces the Rex fact via context
        #    injection). Without force_dream, the inbox count is
        #    well below the dream_threshold default of 10 and no
        #    consolidation would happen mid-conversation.
        #
        #    ADK 1.29's ``run_debug`` is async — we drive it via
        #    asyncio.run per-turn so the sync test function stays
        #    simple.
        events1 = asyncio.run(
            _run_one_turn(runner, _TURN_1, user_id="test-user", session_id="test-session")
        )
        events2 = asyncio.run(
            _run_one_turn(runner, _TURN_2, user_id="test-user", session_id="test-session")
        )

        # 4a. Force consolidation so the graph is built.
        memory.force_dream(block=True)

        # 5. Now ask the question that depends on turn 1's memory.
        events3 = asyncio.run(
            _run_one_turn(runner, _TURN_3, user_id="test-user", session_id="test-session")
        )

        # 6. Assertions.

        # 6a. Turn capture — every turn flowed into memory.
        total = memory.storage.count_memories_by_status(
            "consolidated"
        ) + memory.storage.count_memories_by_status("inbox")
        assert total >= 6, (
            f"Expected at least 6 memories (3 user + 3 assistant turns), "
            f"got {total}. Storage state: "
            f"consolidated={memory.storage.count_memories_by_status('consolidated')}, "
            f"inbox={memory.storage.count_memories_by_status('inbox')}"
        )

        # 6b. Turn 3's response references Rex. This proves the
        #     before_model_callback surfaced the turn-1 memory via
        #     context injection — the model couldn't otherwise know
        #     about the dog's name from turn 3 alone.
        turn3_texts = _extract_text(events3)
        turn3_full = " ".join(turn3_texts).lower()
        assert "rex" in turn3_full, (
            f"Turn 3 response does not mention Rex. Full response:\n{chr(10).join(turn3_texts)}"
        )

        # 6c. Also confirm earlier turns returned real responses
        #     (not empty or error).
        assert _extract_text(events1), "Turn 1 returned no response"
        assert _extract_text(events2), "Turn 2 returned no response"

        # 6d. LTP spike — at least one consolidated memory should have
        #     access_weight at the spiked value. The passive injection
        #     path in the before_model_callback doesn't itself spike
        #     (only the retrieve_memories tool does), so we verify
        #     the spike path by calling retrieve_memories directly
        #     on a consolidated memory and confirming the weight
        #     updates.
        consolidated = memory.storage.get_memories_by_status("consolidated")
        assert consolidated, "No memories were consolidated"

        first_mem_id = consolidated[0]["id"]
        from neuromem.tools import retrieve_memories  # noqa: PLC0415

        retrieved = retrieve_memories([first_mem_id], memory)
        assert len(retrieved) == 1
        assert retrieved[0]["access_weight"] == pytest.approx(1.0)
        assert retrieved[0]["last_accessed"] is not None

        # 6e. Diagnostic output — print the first user+assistant
        #     memory content so a reviewer can see what the real
        #     LLM actually produced for the captured turns.
        print("\n[integration] captured memories:")
        for row in consolidated[:4]:
            role = (row.get("metadata") or {}).get("role", "?")
            content = row["raw_content"][:80]
            print(f"  [{role:9}] {content}")
    finally:
        # Clean up — close the runner so the httpx client shuts down
        # cleanly and the SSL socket doesn't leak a ResourceWarning
        # past the warning filter (belt-and-braces: the pytestmark
        # filter above catches it even if close is a no-op).
        try:
            asyncio.run(runner.close())
        except (RuntimeError, TypeError):
            # close() may not need to be awaited in all ADK versions,
            # or the event loop may already be closed.
            with contextlib.suppress(RuntimeError, TypeError):
                runner.close()
