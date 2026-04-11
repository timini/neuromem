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

    @abstractmethod
    def generate_category_name(self, concepts: list[str]) -> str:
        """Name the category that encompasses a cluster of concepts.

        MUST return a single one-word string (no spaces). If the LLM
        returns a multi-word phrase, ``NeuroMemory`` will take the
        first word and log a warning — but implementations SHOULD
        honour the contract on their own.
        """
