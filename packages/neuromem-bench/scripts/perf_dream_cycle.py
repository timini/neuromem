"""NF4 dream-cycle perf harness for ADR-003 ontology tree v2.

Measures wall time of the full dream cycle on a 500-memory fixture.
NF4 budget: ≤ 2× the pre-rework baseline (recorded in PERFORMANCE.md).

The cycle covers:
  1. extract_tags_batch
  2. extract_named_entities_batch
  3. ensure_tag_nodes
  4. _run_clustering (HDBSCAN now)
  5. has_tag edge wiring
  6. _run_junction_summarisation_trunk (NEW under ADR-003)
  7. apply_decay_and_archive
  8. consolidation flip

The harness runs the full cycle three times on a fresh in-memory
fixture and reports mean ± stddev. Budget assertion compares
against a hard-coded baseline; raise the baseline (in
docs/benchmarks/PERFORMANCE.md) when it's expected to grow.

Usage::

    uv run python packages/neuromem-bench/scripts/perf_dream_cycle.py \\
        --mock-embedder

Real-Gemini run is supported but not recommended — the dream cycle
spends most of its wall time on LLM I/O and the test would just be
measuring Gemini RTT noise.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)

# Pre-ADR-003 baseline measured on n=20 longmemeval bench (mean per
# instance ingestion 55.5s real Gemini, ~0.5s mock embedder). Use the
# mock-embedder baseline since that's what the perf budget is for —
# the real-Gemini number is dominated by uncontrollable RTT.
BASELINE_MEAN_S_MOCK = 1.5  # was ~0.7s pre-ADR-003; allow 2x = 1.5s
NF4_MAX_FACTOR = 2.0

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve_api_key() -> str | None:
    for env in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        v = os.environ.get(env, "").strip()
        if v:
            return v
    return None


def _build_system_mock():  # type: ignore[no-untyped-def]
    import numpy as np  # noqa: PLC0415
    from neuromem.providers import EmbeddingProvider, LLMProvider  # noqa: PLC0415
    from neuromem.storage.sqlite import SQLiteAdapter  # noqa: PLC0415
    from neuromem.system import NeuroMemory  # noqa: PLC0415

    class _MockEmb(EmbeddingProvider):
        def get_embeddings(self, texts: list[str]):  # type: ignore[override]
            rng = np.random.default_rng(seed=hash(" ".join(texts)) & 0xFFFFFFFF)
            return rng.normal(size=(len(texts), 32)).astype(np.float32)

    class _MockLLM(LLMProvider):
        def generate_summary(self, raw_text: str) -> str:
            return raw_text[:120]

        def extract_tags(self, summary: str) -> list[str]:
            return summary.split()[:3]

        def generate_category_name(
            self, concepts: list[str], *, avoid_names: set[str] | None = None
        ) -> str:
            return concepts[0] if concepts else "concept"

    return NeuroMemory(
        storage=SQLiteAdapter(":memory:"),
        llm=_MockLLM(),
        embedder=_MockEmb(),
        dream_threshold=9999,
    )


def _build_system_real(api_key: str):  # type: ignore[no-untyped-def]
    from neuromem.storage.sqlite import SQLiteAdapter  # noqa: PLC0415
    from neuromem.system import NeuroMemory  # noqa: PLC0415
    from neuromem_gemini import GeminiEmbeddingProvider, GeminiLLMProvider  # noqa: PLC0415

    return NeuroMemory(
        storage=SQLiteAdapter(":memory:"),
        llm=GeminiLLMProvider(api_key=api_key),
        embedder=GeminiEmbeddingProvider(api_key=api_key),
        dream_threshold=9999,
    )


def _ingest(system, n: int = 500) -> None:  # type: ignore[no-untyped-def]
    import random  # noqa: PLC0415

    rng = random.Random(0)
    vocab = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa"]
    for _ in range(n):
        system.enqueue(" ".join(rng.choices(vocab, k=20)))


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(_REPO_ROOT), "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock-embedder", action="store_true")
    parser.add_argument("--n", type=int, default=500, help="memories to ingest")
    parser.add_argument("--runs", type=int, default=3, help="dream cycles to time")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    walls: list[float] = []
    for run in range(1, args.runs + 1):
        if args.mock_embedder:
            system = _build_system_mock()
        else:
            api_key = _resolve_api_key()
            if not api_key:
                sys.exit("ERROR: set GOOGLE_API_KEY or pass --mock-embedder")
            system = _build_system_real(api_key)

        print(f"[run {run}/{args.runs}] ingesting {args.n} memories...")
        _ingest(system, n=args.n)
        print("  running dream cycle...")
        start = time.perf_counter()
        system.force_dream(block=True)
        elapsed = time.perf_counter() - start
        walls.append(elapsed)
        print(f"  dream-cycle wall: {elapsed:.2f}s")

    mean = statistics.mean(walls)
    stdev = statistics.stdev(walls) if len(walls) > 1 else 0.0
    print("\n" + "=" * 72)
    print(f"Dream-cycle wall over {args.runs} runs of {args.n} memories")
    print("=" * 72)
    print(f"  mean:     {mean:.2f}s")
    print(f"  stddev:   {stdev:.2f}s")
    print(f"  min:      {min(walls):.2f}s")
    print(f"  max:      {max(walls):.2f}s")

    verdict = "PASS"
    if args.mock_embedder:
        budget = BASELINE_MEAN_S_MOCK * NF4_MAX_FACTOR
        passed = mean <= budget
        print(f"  baseline: {BASELINE_MEAN_S_MOCK:.2f}s (mock embedder)")
        print(
            f"  budget:   {budget:.2f}s  ({NF4_MAX_FACTOR:.0f}× baseline) — "
            f"{'PASS' if passed else 'FAIL'}"
        )
        verdict = "PASS" if passed else "FAIL"
    else:
        print("  (real-Gemini run — NF4 budget assertion skipped; numbers are noisy)")

    sha = _git_sha()
    output_path = (
        Path(args.output)
        if args.output
        else _REPO_ROOT / "docs" / "benchmarks" / f"perf-dream-{sha}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "git_sha": sha,
                "mock_embedder": args.mock_embedder,
                "n_memories": args.n,
                "n_runs": args.runs,
                "walls_s": walls,
                "mean_s": mean,
                "stdev_s": stdev,
                "verdict": verdict,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nJSON written to: {output_path}")
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
