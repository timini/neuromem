"""Abstract base classes for the neuromem provider contract.

``EmbeddingProvider`` and ``LLMProvider`` are the injection points
through which ``NeuroMemory`` talks to external services. Concrete
implementations live in downstream framework-wrapper packages
(``neuromem-openai``, ``neuromem-anthropic``, ``neuromem-langchain``,
etc.) — never inside ``neuromem-core``.

Callers are responsible for constructing a concrete instance and
injecting it into the ``NeuroMemory`` constructor. The core library
imports ``EmbeddingProvider`` / ``LLMProvider`` only as type names —
it never instantiates them.

See specs/001-neuromem-core/contracts/providers.md for the full
behavioural contract per method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

# numpy is already a mandatory runtime dependency per Constitution v2.0.0
# Principle II. Importing NDArray at module top-level (rather than under
# TYPE_CHECKING) lets typing.get_type_hints() work correctly for downstream
# tools that introspect the ABC signatures.


# Cap on how many existing-centroid labels to list in the naming
# prompts' "avoid these" section. More than ~30 and the prompt starts
# to dilute the instruction; the most recently added centroids are the
# most likely clash candidates anyway.
_AVOID_NAMES_MAX = 30


def render_avoid_section(avoid_names: set[str] | None) -> str:
    """Build the optional "don't reuse these labels" snippet for the
    category-naming prompts (issue #42 part 1).

    Returns an empty string when ``avoid_names`` is falsy so the
    prompt templates still format cleanly — they reserve an
    ``{avoid_section}`` placeholder at the natural spot and the empty
    default produces a prompt identical to the pre-#42 behaviour.

    Caps the label list at ``_AVOID_NAMES_MAX`` to keep prompt size
    bounded. Labels are sorted for determinism, so the same avoid set
    always produces the same prompt (helps test fixtures and caches).
    """
    if not avoid_names:
        return ""
    labels = sorted(n for n in avoid_names if n)
    if not labels:
        return ""
    if len(labels) > _AVOID_NAMES_MAX:
        labels = labels[:_AVOID_NAMES_MAX]
    return "Avoid reusing these already-taken labels: " + ", ".join(labels) + "."


class EmbeddingProvider(ABC):
    """Convert batches of strings into dense float vectors.

    Implementations wrap a vendor API (OpenAI, Cohere, Google, Voyage,
    local sentence-transformers, etc.). The core calls this ABC from
    two places:

    - ``NeuroMemory._run_dream_cycle`` (the dreaming thread), to embed
      freshly extracted tag labels.
    - ``ContextHelper.build_prompt_context`` (the caller's thread),
      to embed an incoming query for nearest-neighbour lookup.

    Implementations MUST therefore be safe for concurrent calls from
    these two contexts.
    """

    @abstractmethod
    def get_embeddings(self, texts: list[str]) -> NDArray[np.floating]:
        """Return one embedding vector per input text.

        The return value is a 2-D numpy array of shape
        ``(len(texts), D)``, where ``D`` is the provider's fixed
        embedding dimensionality. Row order MUST match input order
        exactly — the dreaming pipeline relies on zipping the result
        back against the input tag list.

        ``D`` is constant for the lifetime of a provider instance.
        Pure-Python ``list[list[float]]`` return values are also
        accepted (the core wraps them with ``np.asarray``), but
        numpy-native is preferred to avoid an unnecessary conversion.
        """


class LLMProvider(ABC):
    """Provide LLM-backed text transformations used by the cognitive loop.

    The three abstract methods below correspond to the three distinct
    LLM call-sites in the neuromem pipeline:

    - ``generate_summary`` is called on every ``enqueue()`` on the
      caller's thread. Keep it fast.
    - ``extract_tags`` is called per-memory from the dreaming thread.
    - ``generate_category_name`` is called per-cluster merge from the
      dreaming thread.
    """

    @abstractmethod
    def generate_summary(self, raw_text: str) -> str:
        """Return a 1–2 sentence episodic summary of ``raw_text``.

        Called from the dream-cycle worker thread (ADR-004: summary
        generation moved off the caller thread). If a sibling caller
        invokes this directly they get the old synchronous behaviour
        — but ``NeuroMemory.enqueue`` no longer does.
        """

    def generate_summary_batch(self, raw_texts: list[str]) -> list[str]:
        """Summarise many raw texts in a single operation (ADR-004).

        Called from the dream-cycle worker once per cycle to
        backfill the ``summary`` column on every inbox memory. The
        default implementation loops over :meth:`generate_summary`;
        concrete providers SHOULD override with a chunked + parallel
        batched LLM call for real speedup.

        **Contract**:

        - Returns a list of strings, one per input text, in the
          same order. ``len(return_value) == len(raw_texts)`` is a
          hard invariant.
        - Empty input → empty output. No LLM call.
        - Each returned string is a 1–2 sentence episodic summary
          of the corresponding raw text.
        """
        return [self.generate_summary(t) for t in raw_texts]

    @abstractmethod
    def extract_tags(self, summary: str) -> list[str]:
        """Extract discrete concept labels from a memory summary.

        Returns a list of non-empty strings. Labels MAY be multi-word
        (e.g., ``"machine learning"``). 3–7 tags per call is the
        sweet spot; more than 15 is unusual and may overwhelm the
        clustering pass.
        """

    def extract_tags_batch(self, summaries: list[str]) -> list[list[str]]:
        """Extract tags for many summaries in a single operation.

        Concrete providers SHOULD override this with a **batched LLM
        call** so the dream cycle's per-memory tag extraction collapses
        from N serial requests into O(1) (or a small number of chunks
        for very large batches). For typical dream-cycle workloads
        (~100 memories per cycle), the batched form is 50–100× faster
        than the serial equivalent.

        The default implementation provided here is a correctness
        fallback — it loops over :meth:`extract_tags`, one call per
        summary, so every existing provider that doesn't override keeps
        working without any change. It's just slow.

        **Contract**:

        - Returns a list of lists, one per input summary, in the same
          order as the input.
        - ``len(return_value) == len(summaries)`` is a hard invariant.
          The dream cycle uses ``zip(..., strict=True)`` to pair
          summaries with their tag lists and will raise if the
          lengths don't match.
        - Empty input → empty output. No LLM call needed.
        - A batched override that fails mid-call (parse error, token
          limit, etc.) SHOULD fall back to the per-memory loop so the
          dream cycle still succeeds — see ``GeminiLLMProvider``'s
          implementation for the canonical defensive pattern.

        Added in neuromem-core 0.1.x as a performance fix for issue
        #44. Backwards-compatible with every provider from 0.1.0 —
        the default implementation matches the old serial dream-cycle
        behaviour exactly.
        """
        return [self.extract_tags(s) for s in summaries]

    def extract_named_entities(self, summary: str) -> list[str]:
        """Extract named entities (proper nouns) from a memory summary.

        Named entities are things the text specifically *refers to* —
        brand names, places, people, organisations, products — as
        opposed to the *concepts* that :meth:`extract_tags` returns.
        For a summary like "The user redeemed a Target coupon on
        coffee creamer", tags would be something like
        ``["coupon", "redemption", "savings"]`` and named entities
        would be ``["Target"]`` (coffee creamer is a product category,
        not a specific branded entity).

        The two lists are deliberately allowed to overlap at the edges
        — e.g., the Target brand may show up in both — because the
        clustering machinery uses tags and the context-tree renderer
        uses named entities, and it's fine for a salient proper noun
        to do double duty.

        **Default implementation**: returns ``[]``. Providers that
        don't implement NER (mocks, simple offline providers) get
        graceful degradation — the dream cycle still runs, the tree
        just renders without an "Entities:" line. Concrete providers
        that call out to an LLM SHOULD override this with a real
        extraction.

        Returns a list of strings; empty list is valid (the text
        mentions no named entities, which is common for short
        conceptual discussions).
        """
        return []

    def extract_named_entities_batch(self, summaries: list[str]) -> list[list[str]]:
        """Batched variant of :meth:`extract_named_entities`.

        Same contract as :meth:`extract_tags_batch`:

        - Returns a list of lists, one per input summary, in order.
        - ``len(return_value) == len(summaries)`` — hard invariant.
        - Empty input → empty output, no LLM call.
        - Concrete providers SHOULD override with a batched LLM call
          to collapse N serial requests into one per dream cycle.
        - The batched override SHOULD fall back to the per-memory loop
          on parse failure so the dream cycle still succeeds.

        Default implementation loops over :meth:`extract_named_entities`
        so providers without a batched override (or without NER at all)
        keep working — they'll just return a list of empty lists.
        """
        return [self.extract_named_entities(s) for s in summaries]

    @abstractmethod
    def generate_category_name(
        self,
        concepts: list[str],
        *,
        avoid_names: set[str] | None = None,
    ) -> str:
        """Name the category that encompasses a cluster of concepts.

        MUST return a single one-word string (no spaces). If the LLM
        returns a multi-word phrase, ``NeuroMemory`` will take the
        first word and log a warning — but implementations SHOULD
        honour the contract on their own.

        ``avoid_names`` (issue #42): optional set of labels the caller
        wants to stay away from — typically the labels of centroids
        already present in the graph, so the LLM doesn't pick the
        same word for a second independent cluster. Implementations
        SHOULD pass this hint into their prompt but MAY ignore it
        (it's a soft preference, not a hard constraint). The caller
        does NOT re-roll if the returned name is in ``avoid_names``;
        the parameter is purely advisory.
        """

    def generate_category_names_batch(
        self,
        pairs: list[list[str]],
        *,
        avoid_names: set[str] | None = None,
    ) -> list[str]:
        """Name many independent clusters in one operation.

        Used by the lazy centroid naming flow (ADR-002): at render
        time, ``ContextHelper`` collects all centroids in the
        rendered subgraph that still have placeholder labels,
        gathers each centroid's children's labels into a list, and
        asks the provider to name them all in one round-trip.

        Each ``pairs[i]`` is the labels of the children that were
        merged into cluster ``i``. Historically (ADR-001 era) each
        sub-list had exactly 2 entries — binary agglomerative.
        Under ADR-003, clusters may have 2–20 children so
        ``pairs[i]`` can be a variable-length list. The groups are
        independent: each name depends only on its own slot, not on
        cross-group context.

        **Contract**:

        - Returns a list of one-word strings, one per input group, in
          the same order. ``len(return_value) == len(pairs)`` is a
          hard invariant — the resolver uses ``zip(..., strict=True)``
          and will raise on length mismatch (which falls through to
          numpy nearest-neighbour, by design).
        - Empty input → empty output, no LLM call.

        ``avoid_names`` (issue #42): same semantics as on
        :meth:`generate_category_name` — an advisory set of labels
        the caller would prefer the LLM stay away from. Threaded into
        the batched prompt so the LLM sees the existing graph's
        centroid labels and doesn't duplicate them within the batch
        OR vs the existing graph.

        **Default implementation**: loops over :meth:`generate_category_name`,
        once per group, forwarding ``avoid_names``. Same correctness
        fallback pattern as :meth:`extract_tags_batch`. Concrete
        providers SHOULD override with a single batched LLM call
        (one prompt asking for N one-word names) — the speedup
        matters for the dream-cycle render path where dozens of
        names may need resolving in one query.
        """
        return [self.generate_category_name(group, avoid_names=avoid_names) for group in pairs]

    def generate_junction_summary(self, children_summaries: list[str]) -> str:
        """Return a paragraph summary of a centroid's subtree (ADR-003).

        Called by both the dream cycle's trunk-summarisation step and
        ``NeuroMemory.resolve_junction_summaries`` at render time.
        ``children_summaries`` is the list of child summaries that feed
        into this junction — leaves' memory summaries, or other
        junctions' paragraph summaries (floor-up recursion).

        Expected output: 2-4 sentences that fairly represent the union
        of the children. Proper nouns, numbers, and user-specific facts
        are MORE important than abstractions — summaries feed retrieval.

        **Default implementation**: concatenate the first ~400 chars of
        the children, truncate. A correctness fallback so providers
        without paragraph-summary support keep working; concrete
        providers SHOULD override with a real LLM call.
        """
        if not children_summaries:
            return ""
        # Truncate each child item to a per-item budget BEFORE joining,
        # so the composite fallback summary doesn't cut a single child
        # in half when the first item is already long (review finding).
        # ~120 chars per item × 3 items = ~400 chars total. Items
        # stripped of surrounding whitespace, ellipsised when truncated.
        per_item_cap = 120
        per_item_trimmed: list[str] = []
        for raw in children_summaries:
            if not raw:
                continue
            s = raw.strip()
            if not s:
                continue
            if len(s) > per_item_cap:
                s = s[: per_item_cap - 1] + "…"
            per_item_trimmed.append(s)
        return " ".join(per_item_trimmed)

    def generate_junction_summaries_batch(self, groups: list[list[str]]) -> list[str]:
        """Batch variant of :meth:`generate_junction_summary` (ADR-003).

        Same shape as :meth:`generate_category_names_batch`: each
        ``groups[i]`` is an independent set of child summaries to
        roll up into one junction summary. Returns ``len(groups)``
        strings in the same order.

        **Default implementation**: loops, one call per group.
        Concrete providers SHOULD override with a batched LLM call
        — this is called once per dream cycle (for the trunk) and
        once per deep-render (lazy), both performance-sensitive.

        Empty input → empty output, no LLM call.
        """
        return [self.generate_junction_summary(g) for g in groups]


# ---------------------------------------------------------------------------
# ClusteringProvider — ADR-003 D1
# ---------------------------------------------------------------------------


class Cluster:
    """One centroid produced by a ``ClusteringProvider``.

    Pure data record: no behaviour. Not a dataclass to avoid adding a
    stdlib-imports-per-module cost to downstream consumers who only
    need the type. Equivalent to a ``(id, embedding, child_ids,
    cohesion)`` tuple but named-access for readability.

    **id**: a provider-minted unique string. The caller (NeuroMemory's
    clustering step) treats these as opaque identifiers and will
    rewrite them to real node UUIDs as it persists the centroids.
    Other entries in the same ``cluster()`` return list MAY reference
    this id in their ``child_ids`` — the caller uses the return-list
    order to resolve forward references (children appear BEFORE
    parents that reference them).

    **embedding**: the centroid's embedding vector. Usually the
    element-wise mean of the children's embeddings; different
    providers may use alternatives (medoid, weighted mean).

    **child_ids**: list of ids that are this centroid's direct
    children. Each id is either (a) the original leaf id passed into
    ``cluster()``, or (b) the ``id`` of an earlier entry in the same
    returned list. Length typically 2–8 (ADR-003 F2 target), capped
    at 20 by provider policy.

    **cohesion**: the mean pairwise cosine among the children's
    embeddings, in ``[0, 1]``. Used as the ``child_of`` edge weight
    so the graph preserves the similarity semantics ADR-001
    established. Providers that cannot compute this cheaply should
    return ``1.0`` as a neutral placeholder.
    """

    __slots__ = ("id", "embedding", "child_ids", "cohesion")

    def __init__(
        self,
        *,
        id: str,  # noqa: A002 — name matches the concept; shadowing builtin is fine here
        embedding: NDArray[np.floating],
        child_ids: list[str],
        cohesion: float,
    ) -> None:
        self.id = id
        self.embedding = embedding
        self.child_ids = child_ids
        self.cohesion = cohesion


class ClusteringProvider(ABC):
    """Cluster a set of (id, embedding) pairs into a hierarchy (ADR-003).

    The dream cycle calls this ABC once per cycle: input is all tag
    nodes emitted by the current cycle's NER + tag extraction; output
    is a flat list of ``Cluster`` records that the caller persists as
    centroid nodes with ``child_of`` edges.

    Providers do NOT touch storage. They are pure: given the same
    input, the same implementation with the same configuration MUST
    return a functionally equivalent result.

    Two providers ship with neuromem-core:

    - ``HDBSCANClusteringProvider`` (default, recommended) — density-
      based clustering with natural noise-point handling. Recursive
      in the sense that the returned list may include centroids-of-
      centroids forming a multi-level hierarchy.
    - ``StubClusteringProvider`` — test-only deterministic stub that
      groups nodes in pairs. Used by unit tests so cluster-dependent
      assertions don't depend on HDBSCAN's internal randomness.
    """

    @abstractmethod
    def cluster(
        self,
        nodes: list[tuple[str, NDArray[np.floating]]],
    ) -> list[Cluster]:
        """Return the centroids that cluster ``nodes`` into a hierarchy.

        ``nodes`` is a list of ``(node_id, embedding)`` pairs. Node
        ids are opaque strings owned by the caller — the provider
        references them in ``Cluster.child_ids`` but does not
        interpret them.

        **Output ordering**: the returned list is in dependency order.
        If cluster ``c_5`` appears in ``c_9``'s ``child_ids``, then
        ``c_5`` appears BEFORE ``c_9`` in the list. The caller
        walks the list once and resolves forward references as it
        creates nodes in storage.

        **Empty output is valid**: if ``nodes`` has < 2 entries, or
        the provider's algorithm decides everything is noise, return
        an empty list. The caller leaves the leaves as-is and the
        render renders them directly without any centroid above them.
        """
