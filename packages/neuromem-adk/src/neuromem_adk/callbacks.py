"""ADK callback closures wired by ``enable_memory``.

Two callbacks, one per direction of the cognitive loop:

- **``build_after_agent_turn_capturer(memory)``** — returns a closure
  used as ADK's ``after_agent_callback``. Runs after every completed
  agent turn. Extracts the (user, assistant) pair from the ADK
  ``Context`` and calls ``memory.enqueue(...)`` on each, so turns
  flow into the hippocampal inbox automatically without the app
  ever calling a save method (spec US2, FR-004).

- **``build_before_model_context_injector(memory)``** — returns a
  closure used as ADK's ``before_model_callback``. Runs before every
  model call. Reads the user's current message, asks
  ``ContextHelper.build_prompt_context`` for a relevant memory tree,
  and prepends the rendered tree to the ``LlmRequest``'s
  ``config.system_instruction`` so the model sees relevant history
  "for free" before reasoning (spec US3, FR-005).

Both closures are **closures over ``memory``** because ADK registers
callbacks as plain callables on the agent, and the agent's callback
invocation doesn't pass a handle to whatever configured memory is.
Binding the ``memory`` reference in the closure is the cleanest way
to thread state through the callback path.

Both closures are **sync** even though ADK also accepts async
callbacks. Sync is sufficient for v0.1 because:
- ``memory.enqueue`` returns in under 50 ms (SC-002) and is non-
  blocking with respect to the dream thread.
- ``ContextHelper.build_prompt_context`` is CPU-bound (cosine
  similarity + graph traversal) and already fast (<500 ms for 10K
  nodes, SC-003). Async would add overhead without saving time.

Unit tests in ``tests/test_callbacks.py`` use stdlib
``types.SimpleNamespace`` to build synthetic ``Context`` / ``LlmRequest``
objects — cheaper than constructing real ADK objects, which require
a full ``InvocationContext`` tree. Integration tests in
``tests/test_adk_integration.py`` exercise the real ADK objects
against real Gemini.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from neuromem.context import ContextHelper

if TYPE_CHECKING:
    from neuromem import NeuroMemory


# ---------------------------------------------------------------------------
# Small text-extraction helper
# ---------------------------------------------------------------------------


def _extract_text_from_content(content: Any) -> str:
    """Return the concatenated text of all text parts in a Content.

    ADK content objects (``google.genai.types.Content``) have a
    ``.parts`` list where each element is either a text-carrying
    ``Part`` (with a ``.text`` attribute) or some other modality
    (image, function call, etc.). This helper picks out every part
    that has a non-empty ``.text`` string and concatenates them with
    a space separator.

    Returns an empty string if ``content`` is ``None`` or has no
    text parts. We never raise — the callback path must be robust
    to weird content shapes (tool call results, function
    responses, image inputs) and just skip anything it can't
    interpret.
    """
    if content is None:
        return ""
    parts = getattr(content, "parts", None) or []
    texts = [getattr(p, "text", None) or "" for p in parts]
    return " ".join(t for t in texts if t).strip()


# ---------------------------------------------------------------------------
# After-agent callback — automatic turn capture (US2, T010)
# ---------------------------------------------------------------------------


def build_after_agent_turn_capturer(memory: NeuroMemory) -> Any:
    """Return an ``after_agent_callback`` closure that captures the
    turn that just completed into the supplied ``NeuroMemory``.

    Extraction strategy:

    1. The user's message is the callback context's ``user_content``.
    2. The assistant's response is the most recent event in
       ``ctx.session.events`` whose ``author`` matches the agent's
       name. ADK produces one Event per turn for each speaker, so
       the latest assistant Event is the one that the current
       turn's model call just emitted.

    Both pieces flow through ``memory.enqueue`` as two separate
    memories, each tagged with a ``role`` metadata field (``"user"``
    or ``"assistant"``) so downstream consumers can filter by role
    if they care. The ``turn`` field in the metadata is the
    session's current event count at the moment of capture —
    monotonic within a session, reset between sessions.

    The closure returns ``None`` on every call (ADK interprets
    ``None`` as "keep the agent's normal response"). The closure
    intentionally does NOT mutate the ``Content`` the agent is
    about to return — we're a side-effect observer, not a
    transformer.

    Errors inside ``memory.enqueue`` propagate to ADK's callback
    runner, which logs them and (in v1 ADK) does not cancel the
    agent turn. A failed capture is visible in the logs but
    doesn't break the conversation.
    """

    def after_agent_turn_capturer(ctx: Any) -> None:
        session = getattr(ctx, "session", None)
        events = getattr(session, "events", None) or []
        agent_name = getattr(ctx, "agent_name", None)

        user_text = _extract_text_from_content(getattr(ctx, "user_content", None))

        # Walk events from the end to find the latest one whose
        # author is the current agent. That's the response the
        # model just produced.
        assistant_text = ""
        for event in reversed(events):
            if getattr(event, "author", None) == agent_name:
                assistant_text = _extract_text_from_content(getattr(event, "content", None))
                if assistant_text:
                    break

        turn_index = len(events)

        if user_text:
            memory.enqueue(
                user_text,
                metadata={"role": "user", "turn": turn_index},
            )
        if assistant_text:
            memory.enqueue(
                assistant_text,
                metadata={"role": "assistant", "turn": turn_index},
            )

        return None

    return after_agent_turn_capturer


# ---------------------------------------------------------------------------
# Before-model callback — passive context injection (US3, T011)
# ---------------------------------------------------------------------------


def build_before_model_context_injector(memory: NeuroMemory) -> Any:
    """Return a ``before_model_callback`` closure that prepends the
    relevant memory tree to the ``LlmRequest``'s system instruction.

    Strategy:

    1. Extract the query text. Prefer the callback context's
       ``user_content`` (what the user just said). Fall back to the
       latest user-authored event in ``ctx.session.events`` if
       ``user_content`` is missing or text-less.
    2. Ask ``ContextHelper(memory).build_prompt_context(query)`` for
       a rendered ASCII tree of the most relevant memories.
    3. If the tree is non-empty, prepend it to
       ``llm_request.config.system_instruction`` (or set the
       instruction outright if none was set).
    4. Return ``None`` so ADK proceeds with the mutated request.

    Empty queries, missing config, or an empty graph all short-
    circuit: the request passes through unchanged. The callback is
    strictly additive — it never removes or reorders existing
    system-instruction content.

    The mutation targets ``llm_request.config.system_instruction``
    rather than prepending a new element to ``llm_request.contents``
    because the latter would become part of the chat history (and
    get re-prepended on every subsequent turn), which is wrong.
    System instructions are the designated slot for static prompt
    content that sits outside the chat history.
    """
    helper = ContextHelper(memory)

    def before_model_context_injector(ctx: Any, llm_request: Any) -> None:
        query = _resolve_query_text(ctx)
        if not query:
            return None
        tree = helper.build_prompt_context(query)
        if not tree:
            return None
        _inject_into_system_instruction(llm_request, tree)
        return None

    return before_model_context_injector


def _resolve_query_text(ctx: Any) -> str:
    """Extract the query text the context injector should search on.

    Prefer ``ctx.user_content`` (the current turn's user input). Fall
    back to the latest user-authored event in ``ctx.session.events``
    if ``user_content`` is missing or text-less. Returns an empty
    string if neither source yields text.
    """
    query = _extract_text_from_content(getattr(ctx, "user_content", None))
    if query:
        return query

    session = getattr(ctx, "session", None)
    events = getattr(session, "events", None) or []
    for event in reversed(events):
        if getattr(event, "author", None) == "user":
            candidate = _extract_text_from_content(getattr(event, "content", None))
            if candidate:
                return candidate
    return ""


def _inject_into_system_instruction(llm_request: Any, tree: str) -> None:
    """Prepend ``tree`` to ``llm_request.config.system_instruction``.

    Bails silently if:
    - The request has no ``config`` attribute.
    - The config's ``system_instruction`` slot is immutable / frozen.

    These error paths must be graceful — the callback must never
    break an agent turn.
    """
    config = getattr(llm_request, "config", None)
    if config is None:
        return
    existing = getattr(config, "system_instruction", None) or ""
    merged = f"{existing}\n\n{tree}" if existing else tree
    try:
        config.system_instruction = merged
    except (AttributeError, TypeError):
        # Some config shapes are immutable. Silent bail — caller's
        # prompt is unchanged.
        return
