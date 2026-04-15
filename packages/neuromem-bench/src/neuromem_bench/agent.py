"""Agent wrappers for benchmark runs.

Every agent satisfies the :class:`BaseAgent` protocol:

    class BaseAgent(Protocol):
        def process_session(self, turns: list[dict[str, str]]) -> None: ...
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

import logging
import time
from typing import Any, Protocol

import numpy as np
from neuromem import NeuroMemory, SQLiteAdapter
from neuromem_gemini import GeminiEmbeddingProvider

from neuromem_bench._client import GeminiAnsweringClient

logger = logging.getLogger("neuromem_bench.agent")

# Cap the rendered context tree at 20 KB per instance when writing to
# trace output. Deep graphs + top_k=20 can produce 50K+ char strings
# that bloat the JSONL without adding diagnostic value — 20K keeps
# enough structure to grep for gold-answer tokens during failure
# analysis.
_TRACE_CONTEXT_CHAR_CAP = 20_000
# Snippet of each tool call's result kept in the ADK trace. Enough to
# eyeball whether the tool returned anything coherent; full result
# would dominate the file.
_TRACE_TOOL_RESULT_CHAR_CAP = 400


def _cap_text(text: str, limit: int) -> str:
    """Truncate ``text`` to ``limit`` chars with a marker suffix."""
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"...[truncated: {len(text) - limit} chars]"


def _walk_adk_tool_events(events: list[Any]) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """Walk ADK events, returning (tool_calls_tally, tool_trace).

    ADK emits call events and response events in strict order; we
    pair each response with the most recent open call of the same
    tool name. Response payloads for neuromem's tools are dicts with
    a ``result`` string, but we fall back to ``str(...)`` gracefully.
    """
    tool_calls: dict[str, int] = {}
    tool_trace: list[dict[str, Any]] = []
    for event in events:
        content = getattr(event, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            fn_call = getattr(part, "function_call", None)
            if fn_call is not None:
                name = getattr(fn_call, "name", "") or "<unnamed>"
                tool_calls[name] = tool_calls.get(name, 0) + 1
                args = getattr(fn_call, "args", None) or {}
                tool_trace.append(
                    {
                        "tool": name,
                        "args": dict(args) if isinstance(args, dict) else str(args),
                        "result_chars": 0,
                        "result_snippet": "",
                    }
                )
            fn_resp = getattr(part, "function_response", None)
            if fn_resp is not None and tool_trace:
                _fill_tool_response(fn_resp, tool_trace)
    return tool_calls, tool_trace


def _fill_tool_response(fn_resp: Any, tool_trace: list[dict[str, Any]]) -> None:
    """Fill the most recent open trace entry matching fn_resp's name."""
    resp_name = getattr(fn_resp, "name", "")
    resp_payload = getattr(fn_resp, "response", None)
    if isinstance(resp_payload, dict):
        result_str = str(resp_payload.get("result", resp_payload))
    else:
        result_str = str(resp_payload or "")
    for entry in reversed(tool_trace):
        if entry["tool"] == resp_name and entry["result_chars"] == 0:
            entry["result_chars"] = len(result_str)
            entry["result_snippet"] = _cap_text(result_str, _TRACE_TOOL_RESULT_CHAR_CAP)
            return


def _extract_adk_final_text(events: list[Any], agent_name: str) -> tuple[str, int]:
    """Return (last_text, count_of_text_events) authored by agent_name.

    Only events authored by our agent carry the LLM's own text output;
    tool-result events can also have text-bearing parts, which would
    otherwise pollute the answer (e.g. a JSON blob from retrieve_memories
    silently becoming the "answer").
    """
    final_text = ""
    count = 0
    for event in events:
        if getattr(event, "author", None) != agent_name:
            continue
        content = getattr(event, "content", None)
        if content is None:
            continue
        parts = getattr(content, "parts", None) or []
        for part in parts:
            text = getattr(part, "text", None)
            if text:
                final_text = text
                count += 1
    return final_text, count


class BaseAgent(Protocol):
    """Protocol every benchmark agent satisfies.

    The runner calls ``process_session`` once per session in the
    benchmark instance, then ``answer`` once to get the predicted
    response for the benchmark question, then ``reset`` before the
    next instance.

    **Sessions, not turns.** Earlier versions of this protocol used
    ``process_turn(text, role)`` once per turn — which meant the
    ``NeuromemAgent`` made one ``generate_summary`` LLM call per
    turn (200+ calls per LongMemEval instance, tripping Gemini's
    rate limits). Since LongMemEval's data model is explicitly
    session-structured and the neuroscience we're modelling encodes
    episodes rather than individual utterances, we now ingest at
    the session level. Agents that internally care about
    turn-granularity (NaiveRagAgent embeds each turn) can still
    loop over ``turns`` inside their ``process_session``.
    """

    def process_session(self, turns: list[dict[str, str]]) -> None:
        """Ingest one session of conversation history.

        Called once per session per instance. ``turns`` is a list
        of ``{"role": ..., "text": ...}`` dicts in conversation
        order. Agents accumulate whatever state they need so a
        subsequent ``answer`` can use it.
        """
        ...

    def answer(self, question: str) -> str:
        """Return the agent's answer to the benchmark question.

        Called once per instance, after all ``process_session``
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

    def process_session(self, turns: list[dict[str, str]]) -> None:
        # NullAgent's "memory" is just the most-recent N turns
        # flattened, so we append each turn individually and trim at
        # the tail. Session boundaries have no semantic meaning for
        # this baseline — we're deliberately measuring what the raw
        # LLM does with a sliding window of context.
        for turn in turns:
            self._history.append((turn["role"], turn["text"]))
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

    def process_session(self, turns: list[dict[str, str]]) -> None:
        # NaiveRag intentionally keeps per-turn retrieval granularity
        # — that's the whole point of vector-store baselines: match
        # specific utterances, not session summaries. So within a
        # session we still embed each turn separately. The only
        # session-level thing here is that we batch the embedding
        # call, cutting the per-session API round-trips from N to 1.
        if not turns:
            return
        texts = [t["text"] for t in turns]
        # Embedder handles batches natively (up to 100 per Gemini
        # call); one API call per session here is optimal.
        vectors = self._embedder.get_embeddings(texts)
        for turn, vector in zip(turns, vectors, strict=True):
            self._texts.append((turn["role"], turn["text"]))
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
        llm_provider: str = "gemini",
        embedder_provider: str | None = None,
        llm_api_key: str | None = None,
        embedder_api_key: str | None = None,
        memory_model: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        # memory_model decouples the memory-layer LLM model from the
        # answer-LLM model. Used when --llm-provider is something
        # other than gemini (e.g. gemma via Ollama) but the answer
        # LLM is still Gemini (because GeminiAnsweringClient is the
        # only answer client wired in the one-shot agent path).
        self._memory_model = memory_model or model
        self._embedder_model = embedder_model
        self._cluster_threshold = cluster_threshold
        self._llm_provider = llm_provider
        self._embedder_provider = embedder_provider or llm_provider
        self._llm_api_key = llm_api_key or api_key
        self._embedder_api_key = embedder_api_key or llm_api_key or api_key
        self._client = GeminiAnsweringClient(api_key=api_key, model=model)
        self._memory: NeuroMemory | None = None
        self._session_index = 0
        self._build_memory()

    def _build_memory(self) -> None:
        """(Re-)instantiate a fresh NeuroMemory with in-memory storage.

        LLM + embedder come from the provider factory so callers can
        pick openai/anthropic/gemma as well as gemini.
        """
        from neuromem_bench._providers import build_pair  # noqa: PLC0415

        pair = build_pair(
            llm_provider=self._llm_provider,
            llm_api_key=self._llm_api_key,
            llm_model=self._memory_model,
            embedder_provider=self._embedder_provider,
            embedder_api_key=self._embedder_api_key,
            embedder_model=self._embedder_model,
        )
        assert pair.embedder is not None
        self._memory = NeuroMemory(
            storage=SQLiteAdapter(":memory:"),
            llm=pair.llm,
            embedder=pair.embedder,
            dream_threshold=9999,
            cluster_threshold=self._cluster_threshold,
        )

    def process_session(self, turns: list[dict[str, str]]) -> None:
        # One memory per session — the whole rationale for the
        # session-level ingestion shift. A 4-turn session becomes
        # a single NeuroMemory with the concatenated transcript as
        # raw_content and a single generate_summary call capturing
        # the session's narrative arc. Cuts the per-instance API
        # call count 4-5× and produces denser, more coherent
        # memories that preserve relationships between adjacent
        # turns (which per-turn summarisation would miss).
        assert self._memory is not None
        if not turns:
            return
        self._session_index += 1
        self._memory.enqueue_session(
            turns,
            metadata={
                "session_index": self._session_index,
                "turn_count": len(turns),
            },
        )

    def answer(self, question: str) -> str:
        assert self._memory is not None
        # Force consolidation so the concept graph is built before
        # we query it. This is a single expensive call — roughly
        # proportional to the number of enqueued memories — that
        # compensates for us having never hit the dream_threshold.
        t0 = time.perf_counter()
        self._memory.force_dream(block=True)
        force_dream_wall_s = time.perf_counter() - t0

        # Build the ContextHelper tree for the question. This is
        # the same machinery enable_memory's before_model_callback
        # uses.
        from neuromem.context import ContextHelper  # noqa: PLC0415

        helper = ContextHelper(self._memory)
        # top_k=5 (the default) is too tight: when the question's
        # nearest tag node ("degree") falls behind 5 generic centroids
        # in cosine ranking, the relevant memory becomes invisible to
        # the answer LLM. Bumping to 20 widens the seed set so the
        # depth-2 subgraph walk reaches the right memory more reliably.
        top_k = 20
        t1 = time.perf_counter()
        context_tree = helper.build_prompt_context(question, top_k=top_k)
        build_context_wall_s = time.perf_counter() - t1

        system_instruction = (
            "You are a helpful assistant. The following is a tree of "
            "relevant long-term memories retrieved by a cognitive "
            "memory system. Use them to answer the user's question "
            "directly and specifically.\n\n"
            "Relevant memories:\n"
            f"{context_tree or '(no relevant memories found)'}"
        )
        t2 = time.perf_counter()
        answer_text = self._client.generate(
            system_instruction=system_instruction,
            user_message=question,
        )
        answer_wall_s = time.perf_counter() - t2

        context_str = context_tree or ""
        self.last_trace: dict[str, Any] = {
            "context_tree": _cap_text(context_str, _TRACE_CONTEXT_CHAR_CAP),
            "context_tree_chars": len(context_str),
            "top_k": top_k,
            "force_dream_wall_s": force_dream_wall_s,
            "build_context_wall_s": build_context_wall_s,
            "answer_wall_s": answer_wall_s,
        }
        return answer_text

    def reset(self) -> None:
        # Drop the old memory and build a fresh one. The GC
        # cleanup path for SQLiteAdapter closes the file handle
        # (see its __del__).
        self._memory = None
        self._session_index = 0
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
        llm_provider: str = "gemini",
        embedder_provider: str | None = None,
        llm_api_key: str | None = None,
        embedder_api_key: str | None = None,
        memory_model: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._memory_model = memory_model or model
        self._embedder_model = embedder_model
        self._cluster_threshold = cluster_threshold
        self._llm_provider = llm_provider
        self._embedder_provider = embedder_provider or (
            "openai" if llm_provider == "anthropic" else llm_provider
        )
        self._llm_api_key = llm_api_key or api_key
        self._embedder_api_key = embedder_api_key or llm_api_key or api_key
        self._app_name = "neuromem-bench"
        self._user_id = "benchmark-user"
        self._session_id: str = ""  # populated in _build
        # Agent name is used both when constructing the ADK Agent AND
        # when filtering events by author at answer time (the LLM's
        # text parts are authored by this string; tool-result events
        # are not).
        self._agent_name = "benchmark_agent"
        # Session counter for metadata on ingested memories (1-indexed,
        # reset per benchmark instance via _build). ADK's own
        # after_agent_turn_capturer still writes a per-turn ``turn``
        # field for memories captured at answer time — those are
        # complementary events; ingestion goes through the bypass path
        # (one memory per whole session) while answer-time Q+A capture
        # goes through the ADK callback (one memory per turn pair).
        self._session_index = 0
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
        # Session counter resets per benchmark instance so the
        # metadata ``session_index`` field starts at 1 for each
        # instance's first ingested session.
        self._session_index = 0

        # Construct the ADK agent with an instruction that hints tool
        # use. Passive context injection lands in the system prompt
        # before every model call; the instruction below nudges the
        # LLM to fall through to the tools when the injected summaries
        # aren't enough.
        from neuromem_bench.prompts import load_prompt  # noqa: PLC0415

        agent = Agent(
            model=self._model,
            name=self._agent_name,
            instruction=load_prompt("adk_agent_instruction"),
        )

        # Thresholds go in as kwargs via enable_memory (PR #49). No
        # post-hoc attribute mutation needed. Matches NeuromemAgent's
        # tuning (cluster_threshold=0.55, dream_threshold=9999) for
        # apples-to-apples benchmark comparison. dream_threshold=9999
        # effectively disables auto-dream during ingestion; we force
        # one consolidation at answer time, same as NeuromemAgent.
        from neuromem_bench._providers import build_pair  # noqa: PLC0415

        pair = build_pair(
            llm_provider=self._llm_provider,
            llm_api_key=self._llm_api_key,
            llm_model=self._memory_model,
            embedder_provider=self._embedder_provider,
            embedder_api_key=self._embedder_api_key,
            embedder_model=self._embedder_model,
        )
        assert pair.embedder is not None
        memory = enable_memory(
            agent,
            db_path=":memory:",
            llm=pair.llm,
            embedder=pair.embedder,
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

    def process_session(self, turns: list[dict[str, str]]) -> None:
        # Bypass ADK for ingestion and ingest at the session level.
        # One memory per session, one generate_summary call per
        # session — matches NeuromemAgent's new behaviour so the
        # two benchmark arms differ only in their answer-time paths
        # (the whole point of NeuromemAdkAgent is tool access at
        # answer time, not a different ingestion strategy).
        assert self._memory is not None
        if not turns:
            return
        self._session_index += 1
        self._memory.enqueue_session(
            turns,
            metadata={
                "session_index": self._session_index,
                "turn_count": len(turns),
            },
        )

    def answer(self, question: str) -> str:
        import asyncio  # noqa: PLC0415

        assert self._memory is not None
        assert self._runner is not None

        # Force consolidation so the concept graph is built before the
        # before_model_callback fires. Without this, the context tree
        # injection has nothing to render (inbox memories aren't
        # searchable — only consolidated ones are).
        t0 = time.perf_counter()
        self._memory.force_dream(block=True)
        force_dream_wall_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        events = asyncio.run(
            self._runner.run_debug(
                question,
                user_id=self._user_id,
                session_id=self._session_id,
                quiet=True,
            )
        )
        run_wall_s = time.perf_counter() - t1

        # Walk events once to collect BOTH the per-tool call tally
        # (historic `last_tool_calls` shape, kept for backward compat)
        # AND the per-call trace {tool, args, result_chars, snippet}
        # for post-run failure analysis.
        tool_calls, tool_trace = _walk_adk_tool_events(events)
        self.last_tool_calls: dict[str, int] = tool_calls
        if tool_calls:
            logger.info(
                "adk tool calls: %s",
                ", ".join(f"{k}×{v}" for k, v in sorted(tool_calls.items())),
            )
        else:
            logger.info("adk tool calls: none")

        final_text, final_answer_events = _extract_adk_final_text(events, self._agent_name)

        self.last_trace: dict[str, Any] = {
            "tool_calls": dict(tool_calls),
            "tool_trace": tool_trace,
            "final_answer_events": final_answer_events,
            "force_dream_wall_s": force_dream_wall_s,
            "run_wall_s": run_wall_s,
        }
        return final_text.strip()

    def reset(self) -> None:
        # Drop everything and rebuild. New SQLite :memory: DB, new
        # Agent, new Runner, new session_id. The GC handles the old
        # SQLiteAdapter's file-handle close via __del__.
        self._memory = None
        self._runner = None
        self._build()
