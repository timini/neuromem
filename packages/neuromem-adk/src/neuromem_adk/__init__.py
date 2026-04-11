"""neuromem-adk — Google Agent Development Kit integration for neuromem.

Public API surface (uncommented incrementally as downstream tasks land):

    from neuromem_adk import enable_memory, NeuromemMemoryService

See `packages/neuromem-adk/README.md` and
`specs/003-neuromem-adk/quickstart.md` for the full usage guide.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Re-exports are uncommented as the underlying modules land:
# - enable_memory lands in T008 (US1 happy path)
# - NeuromemMemoryService lands in T013 (US5)
#
# from neuromem_adk.enable import enable_memory
# from neuromem_adk.memory_service import NeuromemMemoryService

__all__ = ["__version__"]
