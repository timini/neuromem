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
        # Buffer turns during process_turn and flush all at once via
        # NeuroMemory.enqueue_many() at answer time. enqueue_many runs
        # generate_summary calls concurrently in a thread pool, so
        # ingestion for a 550-turn instance collapses from 550× serial
        # API latency to ceil(550/20)=28× pool-iteration latency. Big
        # win on any paid Gemini tier where the per-minute quota is
        # well above the per-second pool saturation rate.
        self._pending_turns: list[tuple[str, str]] = []
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
        # Buffer — actual enqueue happens concurrently in answer().
        assert self._memory is not None
        self._pending_turns.append((role, text))

    def _flush_pending(self) -> None:
        """Concurrently enqueue all buffered turns in one batch.

        Called from answer() right before force_dream. After the flush
        the buffer is empty. If the buffer is empty already (e.g. an
        answer() call with no prior process_turn, or a second answer()
        in the same instance — unusual but possible), this is a no-op.
        """
        assert self._memory is not None
        if not self._pending_turns:
            return
        raw_texts = [text for _role, text in self._pending_turns]
        metadatas: list[dict | None] = [{"role": role} for role, _text in self._pending_turns]
        self._memory.enqueue_many(raw_texts, metadatas=metadatas)
        self._pending_turns = []

    def answer(self, question: str) -> str:
        assert self._memory is not None
        # Flush buffered turns concurrently — one pool of N workers
        # races through all the generate_summary calls.
        self._flush_pending()
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
        # (see its __del__). Clear the pending-turns buffer too
        # so the next instance doesn't accidentally ingest the
        # previous instance's tail.
        self._memory = None
        self._pending_turns = []
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

    - Thresholds are passed to ``enable_memory`` directly via its
      ``cluster_threshold`` and ``dream_threshold`` kwargs (added in
      PR #49). Matches the tuning :class:`NeuromemAgent` uses for
      apples-to-apples benchmark comparison.

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
        # Agent name is used both when constructing the ADK Agent AND
        # when filtering events by author at answer time (the LLM's
        # text parts are authored by this string; tool-result events
        # are not).
        self._agent_name = "benchmark_agent"
        # Turn counter for metadata — matches the schema the
        # after_agent_callback in neuromem-adk uses so memories
        # captured via the ADK runner (at answer time) and memories
        # captured via our bypass path (at ingestion time) share the
        # same ``metadata["turn"]`` field.
        self._turn_index = 0
        # Same buffered-ingestion pattern as NeuromemAgent: process_turn
        # buffers, answer() flushes all turns concurrently via
        # NeuroMemory.enqueue_many. Big speedup for the ingestion
        # phase at the cost of a tiny buffer per-instance.
        self._pending_turns: list[tuple[str, str]] = []
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
        # Turn counter resets per benchmark instance so the metadata
        # ``turn`` field starts at 1 for each instance's first ingested
        # turn — matching what the ADK callback path would emit.
        self._turn_index = 0
        # Clear pending buffer too — a prior instance could have left
        # its tail behind if answer() wasn't called (shouldn't happen,
        # but cheap insurance).
        self._pending_turns = []

        # Construct the ADK agent with an instruction that hints tool
        # use. Passive context injection lands in the system prompt
        # before every model call; the instruction below nudges the
        # LLM to fall through to the tools when the injected summaries
        # aren't enough.
        agent = Agent(
            model=self._model,
            name=self._agent_name,
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

        # Thresholds go in as kwargs via enable_memory (PR #49). No
        # post-hoc attribute mutation needed. Matches NeuromemAgent's
        # tuning (cluster_threshold=0.55, dream_threshold=9999) for
        # apples-to-apples benchmark comparison. dream_threshold=9999
        # effectively disables auto-dream during ingestion; we force
        # one consolidation at answer time, same as NeuromemAgent.
        memory = enable_memory(
            agent,
            db_path=":memory:",
            llm=GeminiLLMProvider(api_key=self._api_key, model=self._model),
            embedder=GeminiEmbeddingProvider(api_key=self._api_key, model=self._embedder_model),
            cluster_threshold=self._cluster_threshold,
            dream_threshold=9999,
        )

        self._memory = memory

        self._runner = Runner(
            app_name=self._app_name,
            agent=agent,
            session_service=InMemorySessionService(),
            auto_create_session=True,
        )

    def process_turn(self, text: str, *, role: str) -> None:
        # Buffer turns; actual ingestion happens concurrently in
        # answer() via NeuroMemory.enqueue_many. See class docstring
        # for the bypass rationale. Metadata schema mirrors what
        # neuromem_adk's after_agent_turn_capturer writes so
        # downstream consumers can't tell whether a memory was
        # captured via this bypass or via the ADK callback path.
        assert self._memory is not None
        self._turn_index += 1
        self._pending_turns.append((role, text))

    def _flush_pending(self) -> None:
        """Concurrently enqueue all buffered turns in one batch."""
        assert self._memory is not None
        if not self._pending_turns:
            return
        # Rebuild turn indices as sequential so the "turn" metadata
        # field lines up with position in the buffer regardless of
        # what _turn_index counter looked like while buffering.
        start_idx = self._turn_index - len(self._pending_turns) + 1
        raw_texts = [text for _role, text in self._pending_turns]
        metadatas: list[dict | None] = [
            {"role": role, "turn": start_idx + i}
            for i, (role, _text) in enumerate(self._pending_turns)
        ]
        self._memory.enqueue_many(raw_texts, metadatas=metadatas)
        self._pending_turns = []

    def answer(self, question: str) -> str:
        import asyncio  # noqa: PLC0415

        assert self._memory is not None
        assert self._runner is not None

        # Flush buffered turns concurrently before we build the graph.
        self._flush_pending()

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

        # CRITICAL: filter by author. ADK emits events for tool calls
        # AND tool RESULTS, both of which can have text-bearing
        # ``content.parts``. If we naively grab the last text part
        # across all events, a tool-result JSON blob (e.g. from
        # ``retrieve_memories``) becomes the "answer" — wrong, and a
        # silent failure that scores 0 on llm_judge.
        #
        # Only events authored by our agent represent the LLM's own
        # text output. The agent's name is set in _build, mirroring
        # the same author-filter pattern neuromem_adk's
        # after_agent_turn_capturer uses.
        final_text = ""
        for event in events:
            if getattr(event, "author", None) != self._agent_name:
                continue
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
