"""neuromem-bench — benchmark harness for the neuromem memory library.

Public API:

    from neuromem_bench import run_benchmark, RunSummary
    from neuromem_bench.agent import (
        BaseAgent, NullAgent, NaiveRagAgent, NeuromemAdkAgent,
    )
    from neuromem_bench.datasets import LongMemEval
    from neuromem_bench.metrics import exact_match, contains_match, llm_judge

See the package README for usage and the current state of
benchmark coverage.
"""

from __future__ import annotations

from neuromem_bench.runner import RunSummary, run_benchmark

__version__ = "0.1.0"

__all__ = [
    "RunSummary",
    "__version__",
    "run_benchmark",
]
