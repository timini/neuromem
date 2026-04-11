"""neuromem — neuroscience-inspired long-term memory for AI agents.

Public API surface. Importers can get everything they need from the
top-level namespace::

    from neuromem import (
        NeuroMemory, ContextHelper, SQLiteAdapter,
        StorageAdapter, StorageError,
        search_memory, retrieve_memories,
    )
    from neuromem.providers import LLMProvider, EmbeddingProvider

See ``packages/neuromem-core/README.md`` for the quickstart and
``specs/001-neuromem-core/spec.md`` for the full feature specification.
"""

import logging

from neuromem.context import ContextHelper
from neuromem.providers import EmbeddingProvider, LLMProvider
from neuromem.storage.base import StorageAdapter, StorageError
from neuromem.storage.sqlite import SQLiteAdapter
from neuromem.system import NeuroMemory
from neuromem.tools import retrieve_memories, search_memory
from neuromem.vectors import batch_cosine_similarity, compute_centroid, cosine_similarity

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Logger configuration (T029)
# ---------------------------------------------------------------------------
#
# Every log call inside the library goes through the ``neuromem`` logger
# namespace (e.g. ``neuromem.system``, ``neuromem.tools``, ``neuromem.
# storage.sqlite``). We attach a ``NullHandler`` to the top-level
# ``neuromem`` logger so that importers who haven't configured any
# logging don't see the stdlib's "No handlers could be found for
# logger ..." warning (Python 2 legacy behaviour that still shows up
# as silent message-dropping in 3.x).
#
# We deliberately DO NOT call ``logger.setLevel(...)``. Keeping the
# level at ``NOTSET`` means the effective level inherits from the
# ``root`` logger, which defaults to ``WARNING`` — matching the
# recommended behaviour for libraries (see Python logging HOWTO:
# "Configuring Logging for a Library"). Callers who want more detail
# configure the root logger (or ``neuromem`` specifically) from their
# application code and everything Just Works.

logger = logging.getLogger("neuromem")
logger.addHandler(logging.NullHandler())

__all__ = [
    "ContextHelper",
    "EmbeddingProvider",
    "LLMProvider",
    "NeuroMemory",
    "SQLiteAdapter",
    "StorageAdapter",
    "StorageError",
    "__version__",
    "batch_cosine_similarity",
    "compute_centroid",
    "cosine_similarity",
    "logger",
    "retrieve_memories",
    "search_memory",
]
