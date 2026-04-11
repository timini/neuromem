"""neuromem — neuroscience-inspired long-term memory for AI agents.

Public API surface is populated incrementally by downstream tasks.
See specs/001-neuromem-core/tasks.md for the sequence.
"""

__version__ = "0.1.0"

# Re-exports will be uncommented as downstream modules land:
# from neuromem.context import ContextHelper
# from neuromem.providers import EmbeddingProvider, LLMProvider
# from neuromem.storage.base import StorageAdapter, StorageError
# from neuromem.storage.sqlite import SQLiteAdapter
# from neuromem.system import NeuroMemory
# from neuromem.tools import retrieve_memories, search_memory
# from neuromem.vectors import batch_cosine_similarity, compute_centroid, cosine_similarity

__all__ = ["__version__"]
