"""The ``enable_memory`` one-line wire-up function.

Takes an existing Google ADK ``Agent`` and attaches persistent
long-term memory to it. Mutates the agent in place (appends to
``tools``, sets or chains the two callbacks) and returns the
underlying ``NeuroMemory`` handle so callers can inspect or
manipulate state.

Default provider path reads ``GOOGLE_API_KEY`` / ``GEMINI_API_KEY``
from the environment and auto-instantiates the Gemini provider pair
from the ``neuromem-gemini`` sibling package. Override via the
``llm=`` / ``embedder=`` keyword arguments.

See ``specs/003-neuromem-adk/contracts/public-api.md`` for the full
public-API contract and stability guarantees.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from neuromem import NeuroMemory, SQLiteAdapter
from neuromem.tools import retrieve_memories, search_memory

from neuromem_adk.callbacks import (
    build_after_agent_turn_capturer,
    build_before_model_context_injector,
)

if TYPE_CHECKING:
    from google.adk.agents import Agent
    from neuromem.providers import EmbeddingProvider, LLMProvider

# Marker attribute set on the agent after successful wiring. Checked on
# re-entry to prevent double-attachment, which would double-wire tools
# and callbacks and produce silently-wrong behaviour.
_ENABLED_MARKER = "_neuromem_adk_enabled"

# The env-var names the default provider path reads. ``GOOGLE_API_KEY``
# is the canonical ADK name; ``GEMINI_API_KEY`` is the existing
# neuromem-gemini convention. Either works.
_API_KEY_ENV_NAMES = ("GOOGLE_API_KEY", "GEMINI_API_KEY")


def _resolve_api_key() -> str:
    """Return the Gemini API key from the environment or raise KeyError.

    Tries ``GOOGLE_API_KEY`` first, then ``GEMINI_API_KEY``. Raises
    ``KeyError`` with an actionable message naming both candidates if
    neither is set.
    """
    for name in _API_KEY_ENV_NAMES:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    raise KeyError(
        f"neuromem-adk: no Gemini API key found in the environment. "
        f"Set one of {_API_KEY_ENV_NAMES} before calling "
        f"enable_memory() with default providers, or pass explicit "
        f"llm= and embedder= arguments to skip the default path. "
        f"See packages/neuromem-adk/README.md for credential setup."
    )


def _default_providers() -> tuple[LLMProvider, EmbeddingProvider]:
    """Instantiate the default Gemini provider pair.

    Imported lazily so ``enable_memory`` callers who pass explicit
    providers never pay the import cost of ``neuromem_gemini`` (which
    transitively imports ``google.genai``).
    """
    from neuromem_gemini import GeminiEmbeddingProvider, GeminiLLMProvider  # noqa: PLC0415

    api_key = _resolve_api_key()
    return GeminiLLMProvider(api_key=api_key), GeminiEmbeddingProvider(api_key=api_key)


def enable_memory(
    agent: Agent,
    db_path: str | os.PathLike[str],
    *,
    llm: LLMProvider | None = None,
    embedder: EmbeddingProvider | None = None,
    cluster_threshold: float | None = None,
    dream_threshold: int | None = None,
) -> NeuroMemory:
    """Attach persistent long-term memory to an existing ADK agent.

    This is the one-line wire-up helper. Typical usage::

        from google.adk.agents import Agent
        from neuromem_adk import enable_memory

        agent = Agent(model="gemini-2.0-flash-001", name="assistant",
                      instruction="You are a helpful assistant.")
        memory = enable_memory(agent, db_path="memory.db")

    The function mutates ``agent`` in place — this is intentional and
    matches ADK's attribute-based configuration style. Returns the
    underlying ``NeuroMemory`` instance so callers can inspect memory
    state, force a dream cycle, or override behaviour during tests.

    Arguments:
        agent: The ``google.adk.agents.Agent`` instance to wire
            memory onto. MUST NOT already have memory attached — a
            second call on the same agent raises ``ValueError``.
        db_path: Filesystem path to the SQLite database. Pass
            ``":memory:"`` for an in-memory database (useful in tests).
        llm: Optional ``LLMProvider`` override. If ``None``, the
            default ``GeminiLLMProvider`` is auto-instantiated from
            ``neuromem-gemini`` reading ``GOOGLE_API_KEY`` from the
            environment.
        embedder: Optional ``EmbeddingProvider`` override. Same
            default fallback as ``llm``.
        cluster_threshold: Optional ``NeuroMemory`` clustering-similarity
            threshold override. When omitted, the ``NeuroMemory``
            default is used. Lower values (e.g. 0.55) encourage more
            hierarchical clustering of tags and are useful for small
            corpora. See ``NeuroMemory`` docstring for the full
            behaviour of this knob.
        dream_threshold: Optional override for the inbox size at
            which an auto-dream-cycle fires. Pass a very high value
            (e.g. 9999) to effectively disable auto-dreaming when the
            caller intends to drive ``force_dream`` explicitly —
            typical for benchmark harnesses and tests that want
            deterministic consolidation boundaries.

    Returns:
        The ``NeuroMemory`` instance wired to the agent. Callers can
        use this to inspect state (``memory.storage.get_all_nodes()``),
        force consolidation (``memory.force_dream(block=True)``), or
        introspect for debugging.

    Raises:
        ValueError: The agent already has memory attached, or a
            supplied ``cluster_threshold`` / ``dream_threshold`` is
            outside the range accepted by ``NeuroMemory.__init__``.
        KeyError: ``llm`` or ``embedder`` was not supplied and neither
            ``GOOGLE_API_KEY`` nor ``GEMINI_API_KEY`` is set.
        StorageError: The SQLite database at ``db_path`` is corrupt
            or unreadable.
    """
    # Check the marker first, BEFORE instantiating any providers or
    # touching storage — otherwise a double-attachment would waste API
    # calls creating providers that then get thrown away.
    if getattr(agent, _ENABLED_MARKER, False):
        raise ValueError(
            f"neuromem-adk: agent {agent.name!r} already has memory "
            f"attached. enable_memory() must be called exactly once "
            f"per Agent instance. Create a fresh Agent if you need a "
            f"second memory attachment."
        )

    # Resolve providers. If either is missing, we need both halves of
    # the default pair instantiated and then we inject the override
    # where the caller supplied one. This preserves the "pass only one"
    # ergonomics: the omitted slot falls back to Gemini.
    if llm is None or embedder is None:
        default_llm, default_embedder = _default_providers()
        llm = llm or default_llm
        embedder = embedder or default_embedder

    # Build the memory system. SQLiteAdapter raises StorageError with
    # the db_path in the message on I/O failure, so we don't need to
    # wrap it ourselves — propagation is cleanest here.
    #
    # cluster_threshold + dream_threshold: only forwarded if supplied
    # so callers who don't care inherit NeuroMemory's own defaults.
    # Range validation lives in NeuroMemory.__init__; we pass the
    # raw value through and let the core's ValueError propagate.
    nm_kwargs: dict[str, Any] = {}
    if cluster_threshold is not None:
        nm_kwargs["cluster_threshold"] = cluster_threshold
    if dream_threshold is not None:
        nm_kwargs["dream_threshold"] = dream_threshold
    memory = NeuroMemory(
        storage=SQLiteAdapter(str(db_path)),
        llm=llm,
        embedder=embedder,
        **nm_kwargs,
    )

    # Attach the two callbacks (T010 + T011). ADK's callback fields
    # accept either a single callable or a list of callables — when
    # a list is provided, ADK invokes them in order. Chain-preserving
    # assignment: if the agent already has a callback in the slot,
    # keep it AND add ours by wrapping both in a list.
    _chain_callback(
        agent,
        "before_model_callback",
        build_before_model_context_injector(memory),
    )
    _chain_callback(
        agent,
        "after_agent_callback",
        build_after_agent_turn_capturer(memory),
    )

    # Register the memory function tools (T012). We use functools.partial
    # to pre-bind the internal ``system`` handle, so ADK's schema
    # generator (which reads the partial's signature via
    # inspect.signature) sees only the user-facing args the LLM should
    # pass. The LLM never has to construct or reference a NeuroMemory
    # object — that's internal plumbing.
    _register_memory_tools(agent, memory)

    # TODO(T013): wire NeuromemMemoryService into the runner hook.

    # Set the marker LAST so a mid-wiring failure doesn't leave the
    # agent in a half-attached state where double-attachment detection
    # trips on a failed attach.
    setattr(agent, _ENABLED_MARKER, True)

    return memory


def _register_memory_tools(agent: Agent, memory: NeuroMemory) -> None:
    """Append the two memory function tools to ``agent.tools``.

    Defines local wrapper functions that close over ``memory`` and
    expose ONLY the user-facing arguments to ADK's schema generator.
    The LLM sees:

    - ``search_memory``: ``{query, top_k, depth}``
    - ``retrieve_memories``: ``{memory_ids}``

    **Why wrapper functions instead of functools.partial**: Python's
    ``inspect.signature`` on a partial object keeps pre-bound keyword
    args in the reported parameter list (just marked with the bound
    value as the default). Since ADK's schema generator reads that
    signature to build the function-call schema it hands to the LLM,
    the internal ``system`` handle would leak into the visible args.
    Wrapper functions with the exact right signature solve this
    cleanly — ``inspect.signature`` reports only the three arguments
    we want the LLM to see, and the schema generator never sees the
    internal plumbing.

    Wrapper functions also produce cleaner tracebacks (the wrapper's
    module and line show up in the traceback) and preserve docstrings
    the LLM actually uses — ADK reads the docstring as the tool's
    description.

    Any pre-existing tools on the agent are preserved — the new
    tools are appended to the end.
    """

    def search_memory_tool(query: str, top_k: int = 5, depth: int = 2) -> str:
        """Search long-term memory for concepts related to the query.

        Returns an ASCII tree showing the most relevant memories
        organised by concept category. Each memory leaf includes its
        ID (``mem_...``) which can be passed to ``retrieve_memories``
        to fetch the full content.

        Arguments:
            query: The search query text.
            top_k: Number of nearest concept nodes to anchor the
                search on. Defaults to 5.
            depth: Graph traversal depth from the nearest nodes.
                Defaults to 2.

        Returns:
            An ASCII tree string, or an empty string if nothing
            relevant is in memory.
        """
        return search_memory(query=query, system=memory, top_k=top_k, depth=depth)

    def retrieve_memories_tool(memory_ids: list[str]) -> list[dict[str, Any]]:
        """Fetch full memory records by ID and reinforce them.

        Use this after ``search_memory`` returns memory IDs of
        interest, to get the full content and metadata. This call
        also reinforces each memory's recency weight (Long-Term
        Potentiation) so frequently-recalled memories survive
        future decay cycles.

        Arguments:
            memory_ids: List of memory IDs (each starts with ``mem_``)
                to fetch. Unknown IDs are silently skipped.

        Returns:
            A list of memory record dicts, one per found memory.
        """
        return retrieve_memories(memory_ids=memory_ids, system=memory)

    agent.tools.append(search_memory_tool)
    agent.tools.append(retrieve_memories_tool)


def _chain_callback(agent: Agent, slot_name: str, new_cb: Any) -> None:
    """Append ``new_cb`` to the agent's named callback slot, preserving
    any existing callback.

    ADK 1.29's callback slots (``before_model_callback``,
    ``after_agent_callback``, etc.) accept three shapes:
    - ``None`` — no callback
    - ``callable`` — a single callback
    - ``list[callable]`` — multiple callbacks invoked in order

    This helper normalises the shape to a list and appends the new
    callback to the end, so user-registered callbacks fire BEFORE
    neuromem's capture/injection. That ordering matters if the user
    has, e.g., a before_model_callback that mutates the system
    instruction — we want our memory injection to run last so it
    can see the final instruction text the user wanted.
    """
    existing = getattr(agent, slot_name, None)
    if existing is None:
        new_value: Any = new_cb
    elif isinstance(existing, list):
        new_value = [*existing, new_cb]
    else:
        # Single callable; wrap into a two-element list.
        new_value = [existing, new_cb]
    setattr(agent, slot_name, new_value)
