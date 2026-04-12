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

        Called from the caller's thread inside ``enqueue()``. If the
        caller cannot afford real LLM latency on the hot path, they
        should inject a fast/local/no-op provider here and do the
        real summarisation upstream.
        """

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
    def generate_category_name(self, concepts: list[str]) -> str:
        """Name the category that encompasses a cluster of concepts.

        MUST return a single one-word string (no spaces). If the LLM
        returns a multi-word phrase, ``NeuroMemory`` will take the
        first word and log a warning — but implementations SHOULD
        honour the contract on their own.
        """
