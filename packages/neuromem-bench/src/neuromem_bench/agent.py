"""Agent wrappers for benchmark runs.

Every agent satisfies the :class:`BaseAgent` protocol:

    class BaseAgent(Protocol):
        def process_turn(self, text: str, *, role: str) -> None: ...
        def answer(self, question: str) -> str: ...
        def reset(self) -> None: ...

Three concrete implementations:

- :class:`NullAgent`          — no-memory baseline. Feeds the full
                                conversation history to the LLM on
                                every answer() call, capped to a
                                configurable turn window. The
                                "how well does the base LLM alone
                                do" baseline.
- :class:`NaiveRagAgent`      — vector-only baseline. Embeds each
                                turn, answers by retrieving top-k
                                by cosine similarity. No cognitive
                                loop (no tag extraction, no
                                clustering, no decay, no LTP).
                                The "how well does a plain
                                retrieval setup do" baseline.
- :class:`NeuromemAgent`      — direct-NeuroMemory baseline with the
                                full cognitive loop (hippocampus +
                                dreaming + decay + LTP). One-shot
                                answer path: context tree injected as
                                system prompt, no tool calls. Fast
                                per-instance.
- :class:`NeuromemAdkAgent`   — the thing we actually ship. Real
                                Google ADK ``Agent`` with
                                ``neuromem_adk.enable_memory`` wired
                                up. The answer LLM has ``search_memory``
                                and ``retrieve_memories`` as function
                                tools and can look up ``raw_content``
                                mid-answer when the injected context
                                tree's summaries aren't specific enough.
                                Measures the full product, not the
                                handicapped one-shot variant.

All agents take an ``api_key`` constructor arg and use real Gemini
under the hood. For unit-testable agents, use mock providers.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from neuromem import NeuroMemory, SQLiteAdapter
from neuromem_gemini import GeminiEmbeddingProvider, GeminiLLMProvider

from neuromem_bench._client import GeminiAnsweringClient


class BaseAgent(Protocol):
    """Protocol every benchmark agent satisfies.

    The runner calls ``process_turn`` for each turn in each
    session of the benchmark instance, then ``answer`` once to
    get the predicted response for the benchmark question, then
    ``reset`` before the next instance.
    """

    def process_turn(self, text: str, *, role: str) -> None:
        """Ingest one turn of conversation history.

        Called once per turn per session. The agent accumulates
        whatever state it needs — text buffer, vector store,
        neuromem graph — so that a subsequent ``answer`` call
        can use it.
        """
        ...

    def answer(self, question: str) -> str:
        """Return the agent's answer to the benchmark question.

        Called once per instance, after all ``process_turn``
        calls for the instance's sessions have been made.
        """
        ...

    def reset(self) -> None:
        """Clear all per-instance state.

        Called between instances so a previous instance's
        memories don't leak into the next. Implementations
        should rebuild storage, reinstantiate the memory system,
        etc.
        """
        ...


# ---------------------------------------------------------------------------
# NullAgent — no memory baseline
# ---------------------------------------------------------------------------


class NullAgent(BaseAgent):
    """Baseline: feed the full conversation history to the LLM on
    every answer, capped to a configurable max-turn window.

    This is the "what does the LLM alone do with as much history
    as its context window allows" control. Without memory, the
    best it can do is stuff N turns into the prompt and hope the
    question's answer is in the last N.

    On LongMemEval_S (40 sessions × ~4 turns each = ~160 turns,
    ~115K tokens), the full history won't fit in Gemini Flash's
    context for free — this baseline truncates to the most
    recent ``max_turns`` turns, which is the realistic "no
    memory" failure mode.
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "gemini-2.0-flash-001",
        max_turns: int = 50,
    ) -> None:
        self._client = GeminiAnsweringClient(api_key=api_key, model=model)
        self._max_turns = max_turns
        self._history: list[tuple[str, str]] = []

    def process_turn(self, text: str, *, role: str) -> None:
        self._history.append((role, text))
        # Cap the history so we never build a prompt bigger than
        # the model's context window. Oldest turns drop first.
        if len(self._history) > self._max_turns:
            self._history = self._history[-self._max_turns :]

    def answer(self, question: str) -> str:
        if not self._history:
            return self._client.generate(
                system_instruction=None,
                user_message=question,
            )
        history_block = "\n".join(f"{role}: {text}" for role, text in self._history)
        system_instruction = (
            "You are a helpful assistant. The following is a history "
            "of a prior conversation. Use it to answer the user's "
            "question as directly as possible.\n\n"
            "Prior conversation:\n"
            f"{history_block}"
        )
        return self._client.generate(
            system_instruction=system_instruction,
            user_message=question,
        )

    def reset(self) -> None:
        self._history = []


# ---------------------------------------------------------------------------
# NaiveRagAgent — vector-only baseline
# ---------------------------------------------------------------------------


class NaiveRagAgent(BaseAgent):
    """Baseline: embed each turn, retrieve top-k by cosine
    similarity at answer time, stuff into prompt.

    No cognitive loop — no tag extraction, no clustering, no
    decay, no LTP. Just a flat vector store with a similarity
    search. Represents "what you'd get if you wired a vector DB
    to your agent and called it memory".

    The comparison against ``NeuromemAgent`` is the point of
    this baseline: if neuromem's cognitive loop produces
    meaningfully better scores than naive vector retrieval, the
    decay + clustering + LTP machinery is earning its complexity.
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "gemini-2.0-flash-001",
        embedder_model: str = "gemini-embedding-001",
        top_k: int = 5,
    ) -> None:
        self._client = GeminiAnsweringClient(api_key=api_key, model=model)
        self._embedder = GeminiEmbeddingProvider(api_key=api_key, model=embedder_model)
        self._top_k = top_k
        self._texts: list[tuple[str, str]] = []  # (role, text)
        self._embeddings: list[np.ndarray] = []

    def process_turn(self, text: str, *, role: str) -> None:
        # Embed one turn. We do this per-turn rather than batching
        # because the runner feeds turns sequentially and the
        # alternative would be an extra ingestion phase.
        vector = self._embedder.get_embeddings([text])[0]
        self._texts.append((role, text))
        self._embeddings.append(np.asarray(vector, dtype=np.float32))

    def answer(self, question: str) -> str:
        if not self._texts:
            return self._client.generate(
                system_instruction=None,
                user_message=question,
            )

        q_vec = np.asarray(self._embedder.get_embeddings([question])[0], dtype=np.float32)
        matrix = np.stack(self._embeddings).astype(np.float64)
        q_norm = float(np.linalg.norm(q_vec))
        row_norms = np.linalg.norm(matrix, axis=1)
        denom = row_norms * q_norm
        safe_denom = np.where(denom == 0.0, 1.0, denom)
        sims = (matrix @ q_vec) / safe_denom
        sims[row_norms == 0.0] = 0.0

        effective_k = min(self._top_k, len(self._texts))
        top_idx = np.argpartition(-sims, effective_k - 1)[:effective_k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        context_block = "\n".join(
            f"{self._texts[int(i)][0]}: {self._texts[int(i)][1]}" for i in top_idx
        )
        system_instruction = (
            "You are a helpful assistant. The following excerpts from "
            "a prior conversation are the most semantically relevant "
            "to the user's current question. Use them to answer.\n\n"
            "Relevant prior excerpts:\n"
            f"{context_block}"
        )
        return self._client.generate(
            system_instruction=system_instruction,
            user_message=question,
        )

    def reset(self) -> None:
        self._texts = []
        self._embeddings = []


# ---------------------------------------------------------------------------
# NeuromemAgent — the cognitive-loop path
# ---------------------------------------------------------------------------


class NeuromemAgent(BaseAgent):
    """The agent we're validating. Uses neuromem's full cognitive
    loop under the hood: hippocampal capture, neocortex dreaming
    (tag extraction + agglomerative clustering), synaptic decay,
    long-term potentiation on retrieval.

    **Does NOT go through neuromem-adk.** This agent uses the
    cognitive loop directly via ``NeuroMemory.enqueue`` +
    ``force_dream`` + ``ContextHelper.build_prompt_context``, NOT
    via ``neuromem_adk.enable_memory``. Going through a real ADK
    ``Runner`` per turn would add latency orthogonal to what we're
    measuring here (the quality of the memory graph, not ADK's
    runtime wiring). The T015 integration test inside neuromem-adk
    already proves the ADK wiring works end-to-end against real
    Gemini; this agent exercises the same underlying NeuroMemory
    machinery with a simpler harness.

    If you want to measure the full ADK integration path specifically
    (latency, tool-call behaviour, session-end consolidation via
    ADK's BaseMemoryService slot), that's a separate agent class
    worth adding — probably named ``NeuromemAdkAgent`` — that
    actually calls ``enable_memory``. It would be slower per turn
    but would measure a different property.

    **Tuning knobs** passed to ``NeuroMemory``:
      - ``dream_threshold=9999`` so auto-dream never fires during
        per-turn enqueue. We force a single dream cycle at the end
        of each instance's ingestion. This batches all the tag
        extraction + clustering into one pass, which is faster
        and produces a cleaner graph than per-turn dreams.
      - ``cluster_threshold=0.55`` (lower than the 0.82 default)
        to encourage meaningful hierarchy formation. See issue #42
        for why the default is too strict for small corpora.
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "gemini-2.0-flash-001",
        embedder_model: str = "gemini-embedding-001",
        cluster_threshold: float = 0.55,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._embedder_model = embedder_model
        self._cluster_threshold = cluster_threshold
        self._client = GeminiAnsweringClient(api_key=api_key, model=model)
        self._memory: NeuroMemory | None = None
        self._build_memory()

    def _build_memory(self) -> None:
        """(Re-)instantiate a fresh NeuroMemory with in-memory storage
        backed by real Gemini providers."""
        self._memory = NeuroMemory(
            storage=SQLiteAdapter(":memory:"),
            llm=GeminiLLMProvider(api_key=self._api_key, model=self._model),
            embedder=GeminiEmbeddingProvider(api_key=self._api_key, model=self._embedder_model),
            dream_threshold=9999,
            cluster_threshold=self._cluster_threshold,
        )

    def process_turn(self, text: str, *, role: str) -> None:
        assert self._memory is not None
        self._memory.enqueue(text, metadata={"role": role})

    def answer(self, question: str) -> str:
        assert self._memory is not None
        # Force consolidation so the concept graph is built before
        # we query it. This is a single expensive call — roughly
        # proportional to the number of enqueued memories — that
        # compensates for us having never hit the dream_threshold.
        self._memory.force_dream(block=True)

        # Build the ContextHelper tree for the question. This is
        # the same machinery enable_memory's before_model_callback
        # uses.
        from neuromem.context import ContextHelper  # noqa: PLC0415

        helper = ContextHelper(self._memory)
        context_tree = helper.build_prompt_context(question)

        system_instruction = (
            "You are a helpful assistant. The following is a tree of "
            "relevant long-term memories retrieved by a cognitive "
            "memory system. Use them to answer the user's question "
            "directly and specifically.\n\n"
            "Relevant memories:\n"
            f"{context_tree or '(no relevant memories found)'}"
        )
        return self._client.generate(
            system_instruction=system_instruction,
            user_message=question,
        )

    def reset(self) -> None:
        # Drop the old memory and build a fresh one. The GC
        # cleanup path for SQLiteAdapter closes the file handle
        # (see its __del__).
        self._memory = None
        self._build_memory()


# ---------------------------------------------------------------------------
# NeuromemAdkAgent — the real ADK-backed path with tool access
# ---------------------------------------------------------------------------


class NeuromemAdkAgent(BaseAgent):
    """Real Google ADK ``Agent`` with ``neuromem_adk.enable_memory``.

    The counterpart to :class:`NeuromemAgent`. The key behavioural
    differences that matter for benchmark scoring:

    - **Tools exposed to the answer LLM**: ``enable_memory`` registers
      ``search_memory`` and ``retrieve_memories`` as ADK function
      tools. The LLM can decide mid-answer to call them when the
      passively injected context tree's 300-char summary snippets
      aren't enough. In particular, ``retrieve_memories(ids)``
      returns the full ``raw_content`` of a memory, so the LLM can
      drill into specifics that were compressed out of the summary.
      The direct-NeuroMemory path in :class:`NeuromemAgent` does NOT
      have this — it's strictly the one-shot context-tree prompt.

    - **Passive context injection via ADK callback**: ADK's
      ``before_model_callback`` fires on every model invocation;
      ``enable_memory`` wires a context-tree renderer into it, so
      the prompt is enriched even when the LLM doesn't actively
      ask for memory.

    - **Turn capture**: ``enable_memory`` also wires an
      ``after_agent_callback`` that walks ``session.events`` to
      enqueue each new (user, assistant) pair into memory. For the
      benchmark ingestion phase we bypass this and call
      ``memory.enqueue(text, metadata={"role": role})`` directly —
      driving the ADK Runner for every historical turn would double
      the LLM spend on ingestion for no benefit. The callback-based
      capture still fires at answer time for the benchmark question
      + produced response, which is fine (reset drops it before the
      next instance anyway).

    **Design trade-offs**:

    - Per-answer call is slower than :class:`NeuromemAgent` because
      ADK's ``Runner.run_debug`` is async-only and each call goes
      through the full callback pipeline + potential tool invocations.
      Expect 2-5× the latency of the one-shot path. That's the cost
      of measuring the real product.

    - ``enable_memory`` doesn't expose ``cluster_threshold`` or
      ``dream_threshold`` kwargs; we mutate both attributes on the
      returned ``NeuroMemory`` instance after the call to keep the
      config consistent with :class:`NeuromemAgent`. A future
      enable_memory signature could accept **kwargs — flagged as a
      follow-up, not blocking.

    - Ingestion bypasses the ADK path entirely. The fully orthodox
      version would drive the Runner for every turn, but that
      doubles ingestion cost and is not what the benchmark is
      measuring (we're measuring answer-time recall, not turn
      capture fidelity — the integration test in neuromem-adk
      already covers capture correctness).
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "gemini-2.0-flash-001",
        embedder_model: str = "gemini-embedding-001",
        cluster_threshold: float = 0.55,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._embedder_model = embedder_model
        self._cluster_threshold = cluster_threshold
        # Session identifiers are stable across turns within one
        # benchmark instance; reset() re-rolls the session_id so
        # the prior instance's session.events don't leak in.
        self._app_name = "neuromem-bench"
        self._user_id = "benchmark-user"
        self._session_id: str = ""  # populated in _build
        self._memory = None  # type: ignore[assignment]
        self._runner = None  # type: ignore[assignment]
        self._build()

    def _build(self) -> None:
        """(Re-)instantiate the ADK Agent, memory, and Runner.

        Called on ``__init__`` and on every ``reset()``. Uses
        in-memory SQLite so each benchmark instance has fresh
        storage. The ``session_id`` is a fresh UUID so ADK's
        in-memory session service doesn't merge state across
        instances even though we keep the same ``SessionService``
        instance for simplicity.
        """
        import uuid  # noqa: PLC0415

        from google.adk.agents import Agent  # noqa: PLC0415
        from google.adk.runners import Runner  # noqa: PLC0415
        from google.adk.sessions import InMemorySessionService  # noqa: PLC0415
        from neuromem_adk import enable_memory  # noqa: PLC0415

        self._session_id = f"instance-{uuid.uuid4().hex[:12]}"

        # Construct the ADK agent with an instruction that hints tool
        # use. Passive context injection lands in the system prompt
        # before every model call; the instruction below nudges the
        # LLM to fall through to the tools when the injected summaries
        # aren't enough.
        agent = Agent(
            model=self._model,
            name="benchmark_agent",
            instruction=(
                "You are a helpful assistant with access to long-term "
                "memory. Before each of your responses, a tree of "
                "relevant memories from prior conversations is "
                "automatically injected into your context. Read it "
                "first. If the injected memories contain the answer, "
                "respond directly and specifically. If the summaries "
                "are too compressed to answer precisely, call the "
                "retrieve_memories tool with the specific memory IDs "
                "shown in the tree to fetch their full original text. "
                "If the tree doesn't surface the relevant topic at "
                "all, call search_memory with a focused query. Always "
                "answer the user's question directly — no hedging, no "
                "'I don't have access' responses unless the memories "
                "genuinely don't cover the topic."
            ),
        )

        memory = enable_memory(
            agent,
            db_path=":memory:",
            llm=GeminiLLMProvider(api_key=self._api_key, model=self._model),
            embedder=GeminiEmbeddingProvider(api_key=self._api_key, model=self._embedder_model),
        )

        # enable_memory doesn't expose these kwargs; tune post-hoc so
        # the config matches NeuromemAgent for apples-to-apples
        # benchmark comparison.
        memory.cluster_threshold = self._cluster_threshold
        # Prevent auto-dream during per-turn enqueue; we force one
        # consolidation at answer time, same as NeuromemAgent.
        memory.dream_threshold = 9999

        self._memory = memory

        self._runner = Runner(
            app_name=self._app_name,
            agent=agent,
            session_service=InMemorySessionService(),
            auto_create_session=True,
        )

    def process_turn(self, text: str, *, role: str) -> None:
        # Bypass ADK for ingestion. See class docstring for why.
        assert self._memory is not None
        self._memory.enqueue(text, metadata={"role": role})

    def answer(self, question: str) -> str:
        import asyncio  # noqa: PLC0415

        assert self._memory is not None
        assert self._runner is not None

        # Force consolidation so the concept graph is built before the
        # before_model_callback fires. Without this, the context tree
        # injection has nothing to render (inbox memories aren't
        # searchable — only consolidated ones are).
        self._memory.force_dream(block=True)

        events = asyncio.run(
            self._runner.run_debug(
                question,
                user_id=self._user_id,
                session_id=self._session_id,
                quiet=True,
            )
        )

        # Walk the event list in reverse — the last text-bearing part
        # on the last agent-authored event is the final response.
        # Tool-invocation events may appear earlier; we skip them by
        # taking the LAST text part.
        final_text = ""
        for event in events:
            content = getattr(event, "content", None)
            if content is None:
                continue
            parts = getattr(content, "parts", None) or []
            for part in parts:
                text = getattr(part, "text", None)
                if text:
                    final_text = text
        return final_text.strip()

    def reset(self) -> None:
        # Drop everything and rebuild. New SQLite :memory: DB, new
        # Agent, new Runner, new session_id. The GC handles the old
        # SQLiteAdapter's file-handle close via __del__.
        self._memory = None
        self._runner = None
        self._build()
