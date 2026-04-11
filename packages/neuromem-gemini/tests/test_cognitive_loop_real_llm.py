"""End-to-end integration tests for the neuromem cognitive loop against real Gemini.

These are the only tests in the whole monorepo that hit a live API.
They are marked ``@pytest.mark.integration`` and the workspace-root
pytest config excludes that marker from the default run, so the
pre-commit pytest hook and normal ``uv run pytest`` invocations never
trigger a network call. Opt in with::

    uv run pytest packages/neuromem-gemini/tests/ -m integration -v

Cost per run: roughly 5000 tokens through ``gemini-2.0-flash-001`` +
a handful of 3072-dim ``gemini-embedding-001`` calls. <$0.01 total.

Scope is deliberately tight (3 tests):

1. **Tag quality.** Prove the "how can `is` / `a` be a tag?" behaviour
   from the deterministic mock was a mock-only artifact — a real LLM
   never returns stopwords as concept tags.
2. **End-to-end recall.** Prove that after a real dream cycle,
   ``search_memory`` returns a non-empty tree with relevant content.
3. **LTP spike.** Prove that ``retrieve_memories`` spikes
   ``access_weight=1.0`` on consolidated memories.

The corpus is a handful of realistic software-engineering
conversational turns — not synthetic, not lorem ipsum — so the
cluster graph that forms has some hope of looking semantically
meaningful.
"""

from __future__ import annotations

import pytest
from neuromem import ContextHelper, NeuroMemory, SQLiteAdapter
from neuromem.tools import retrieve_memories, search_memory
from neuromem_gemini import GeminiEmbeddingProvider, GeminiLLMProvider

# Module-level markers:
#   - `integration` gates the whole file behind `-m integration`.
#   - The two `filterwarnings` entries suppress the ResourceWarning
#     that google-genai's underlying ``httpx.Client`` emits when its
#     SSL socket is GC'd without an explicit close. google-genai
#     1.72 doesn't expose a close() on ``genai.Client``, so there's
#     nothing we can call in teardown. The workspace-root
#     ``filterwarnings=['error']`` config would otherwise escalate
#     the warning into a hard test failure — scoping the ignore to
#     this file keeps the core unit tests strict while letting the
#     network-gated integration tests coexist peacefully with the
#     upstream SDK's cleanup behaviour.
pytestmark = [
    pytest.mark.integration,
    pytest.mark.filterwarnings("ignore::ResourceWarning"),
    pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning"),
]


# ---------------------------------------------------------------------------
# Corpus — realistic conversational turns about a small set of topics
# (SQLite internals, Python async, numpy vector math). Small enough to
# dream in one cycle, diverse enough that clustering has something to do.
# ---------------------------------------------------------------------------

CORPUS: list[str] = [
    "The user is debugging a WAL-mode SQLite locking issue where writers block readers under load.",
    "The agent suggested setting PRAGMA synchronous=NORMAL to mitigate fsync-induced latency spikes.",
    "Python's asyncio event loop can starve under CPU-bound work if you forget to run blocking calls in a thread pool.",
    "numpy broadcasting lets you subtract a 1-D vector from a 2-D matrix row-wise without writing an explicit for loop.",
    "Cosine similarity between two vectors is the dot product divided by the product of their norms.",
]


# ---------------------------------------------------------------------------
# Fixture: a freshly-built NeuroMemory wired to real Gemini providers,
# pre-populated with the corpus and fully dream-cycled so tests start
# from "consolidated state" without paying the API bill for the setup
# more than once per session.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def consolidated_system(gemini_api_key: str) -> NeuroMemory:
    """Build a NeuroMemory + real Gemini providers and run one full dream cycle.

    Scoped to ``module`` so all three tests share a single consolidation
    — one ~5000-token run, not three. Each test still operates on a
    fresh query / retrieval, so there is no state leakage.
    """
    memory = NeuroMemory(
        storage=SQLiteAdapter(":memory:"),
        llm=GeminiLLMProvider(api_key=gemini_api_key),
        embedder=GeminiEmbeddingProvider(api_key=gemini_api_key),
    )
    memory_ids: list[str] = []
    for turn in CORPUS:
        memory_ids.append(memory.enqueue(turn))

    memory.force_dream(block=True)

    # Sanity: all corpus memories should be consolidated post-dream.
    # If this fails we have a bigger problem than any individual test.
    for mem_id in memory_ids:
        row = memory.storage.get_memory_by_id(mem_id)
        assert row is not None, f"memory {mem_id} disappeared during dream cycle"
        assert row["status"] == "consolidated", (
            f"memory {mem_id} status is {row['status']!r}, expected 'consolidated'"
        )

    # Attach the ID list so tests can use them without re-deriving.
    memory._test_corpus_ids = memory_ids  # type: ignore[attr-defined]
    return memory


# ---------------------------------------------------------------------------
# 1. Tag quality — the answer to "how can `is` / `a` be a tag?"
# ---------------------------------------------------------------------------

# Stopwords a real LLM should never emit as concept tags. Case-
# insensitive match. We don't try to be linguistically exhaustive —
# this list is just the ones the mock provider exhibited + a few
# obvious neighbours.
STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "of",
        "to",
        "for",
        "in",
        "on",
        "at",
        "by",
        "and",
        "or",
        "but",
        "if",
        "it",
        "this",
        "that",
        "these",
        "those",
        "with",
        "from",
    }
)


def test_real_llm_produces_meaningful_tags(
    consolidated_system: NeuroMemory,
) -> None:
    """Every node label emitted by the real LLM must be a real concept,
    not a stopword. This directly answers the confusion the user raised
    about the mock provider returning 'is' and 'a' as tag nodes.
    """
    nodes = consolidated_system.storage.get_all_nodes()
    assert nodes, "dream cycle should have produced at least one node"

    offending: list[str] = []
    for node in nodes:
        label = node["label"].lower().strip()
        if label in STOPWORDS:
            offending.append(node["label"])

    assert not offending, (
        f"Real Gemini emitted {len(offending)} stopword tags: {offending}. "
        "This is a prompt regression — the extract_tags prompt in "
        "GeminiLLMProvider should filter stopwords, articles, and "
        "prepositions. Investigate the prompt in "
        "`packages/neuromem-gemini/src/neuromem_gemini/llm.py::extract_tags`."
    )

    # Diagnostic: print the actual tag set so the test output tells a
    # reviewer what the real LLM produced without them having to
    # rerun the suite.
    labels = sorted(n["label"] for n in nodes)
    print(f"\n[integration] real Gemini produced {len(labels)} tag nodes:")
    for label in labels:
        print(f"  - {label}")


# ---------------------------------------------------------------------------
# 2. End-to-end recall — ContextHelper / search_memory return real content
# ---------------------------------------------------------------------------


def test_real_llm_end_to_end_recall(
    consolidated_system: NeuroMemory,
) -> None:
    """After a real dream cycle, both ContextHelper.build_prompt_context
    and search_memory should return non-empty trees that contain at
    least one of the enqueued raw memories for a topical query.
    """
    query = "sqlite performance tuning"

    # Helper path.
    helper = ContextHelper(consolidated_system)
    ctx_block = helper.build_prompt_context(query)
    assert ctx_block, "build_prompt_context should return a non-empty tree"
    assert "📁" in ctx_block, "tree should contain at least one category marker"
    assert "📄" in ctx_block, "tree should contain at least one memory leaf"

    # Tool path.
    tool_output = search_memory(query, consolidated_system)
    assert tool_output, "search_memory should return a non-empty tree"
    assert "📁" in tool_output
    assert "📄" in tool_output

    # Content relevance: at least one of the SQLite-related memories
    # from the corpus should surface verbatim-ish in the tree. Case-
    # insensitive substring match against a few diagnostic keywords
    # drawn directly from the corpus text.
    sqlite_keywords = ["wal", "sqlite", "pragma", "synchronous"]
    hits = [kw for kw in sqlite_keywords if kw.lower() in tool_output.lower()]
    assert hits, (
        f"search_memory({query!r}) returned a tree with no sqlite keywords. "
        f"Tree was:\n{tool_output}"
    )
    print(f"\n[integration] search_memory({query!r}) tree:\n{tool_output}")


# ---------------------------------------------------------------------------
# 3. LTP spike — retrieve_memories on a real consolidated row
# ---------------------------------------------------------------------------


def test_real_llm_retrieve_memories_spikes_ltp(
    consolidated_system: NeuroMemory,
) -> None:
    """``retrieve_memories`` must spike ``access_weight`` back to 1.0
    and populate ``last_accessed`` on every consolidated row it
    returns. This is the Long-Term Potentiation recall-reinforcement
    loop from FR-021 — the real LLM has no bearing on the
    arithmetic, but running it against a live graph is the end-to-end
    smoke test that the wiring is intact.
    """
    corpus_ids: list[str] = consolidated_system._test_corpus_ids  # type: ignore[attr-defined]
    assert corpus_ids, "fixture didn't attach the corpus id list"

    # Pick the first corpus memory and verify it's consolidated.
    target_id = corpus_ids[0]
    pre = consolidated_system.storage.get_memory_by_id(target_id)
    assert pre is not None and pre["status"] == "consolidated"

    results = retrieve_memories([target_id], consolidated_system)
    assert len(results) == 1
    row = results[0]

    assert row["id"] == target_id
    assert row["status"] == "consolidated"
    assert row["access_weight"] == pytest.approx(1.0)
    assert row["last_accessed"] is not None
    assert row["raw_content"] == CORPUS[0]  # content round-trips cleanly

    # And the database row was updated in place, not just the return
    # dict — confirm by re-reading.
    post = consolidated_system.storage.get_memory_by_id(target_id)
    assert post["access_weight"] == pytest.approx(1.0)
    assert post["last_accessed"] is not None
