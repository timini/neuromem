"""ADR-004 ingestion perf harness — MEASUREMENT, NOT A GATE.

Ingestion runs in the background between sessions, so its wall time
is not on any user-visible critical path. This harness records it so
regressions are visible, but does NOT fail CI on regression. Use
``--fail-on-regression`` to flip it to a gate once we're happy with
the pinned baseline.

Reports:
  - Total wall for a 500-memory dream cycle.
  - Per-component breakdown: generate_summary_batch, extract_tags_batch,
    extract_named_entities_batch, clustering, _run_all_centroid_naming,
    _run_all_junction_summarisation, decay + consolidate.
  - Storage footprint (reuses perf_storage_footprint.py logic).

Usage::

    # Cheap baseline (no network):
    uv run python packages/neuromem-bench/scripts/perf_ingestion.py \\
        --mock-embedder

    # Real Gemini run (expensive, ad-hoc):
    GOOGLE_API_KEY=... uv run python \\
        packages/neuromem-bench/scripts/perf_ingestion.py
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

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve_api_key() -> str | None:
    for env in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        v = os.environ.get(env, "").strip()
        if v:
            return v
    return None


class _InstrumentedLLM:
    """Wraps an LLMProvider and measures per-method wall time."""

    def __init__(self, inner) -> None:  # type: ignore[no-untyped-def]
        self._inner = inner
        self.walls: dict[str, list[float]] = {}

    def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
        target = getattr(self._inner, name)
        if not callable(target) or not name.startswith(("generate_", "extract_")):
            return target

        def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
            start = time.perf_counter()
            result = target(*args, **kwargs)
            elapsed = time.perf_counter() - start
            self.walls.setdefault(name, []).append(elapsed)
            return result

        return wrapper

    def total_per_method(self) -> dict[str, float]:
        return {name: sum(walls) for name, walls in self.walls.items()}


def _build_system_mock(instr: _InstrumentedLLM | None = None):  # type: ignore[no-untyped-def]
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

        def generate_category_name(self, concepts: list[str]) -> str:
            return concepts[0] if concepts else "concept"

    inner_llm = _MockLLM()
    llm = instr._inner if instr else inner_llm  # type: ignore[union-attr]
    return NeuroMemory(
        storage=SQLiteAdapter(":memory:"),
        llm=instr if instr else inner_llm,
        embedder=_MockEmb(),
        dream_threshold=9999,
    ), inner_llm if instr is None else llm


def _build_system_real(api_key: str):  # type: ignore[no-untyped-def]
    from neuromem.storage.sqlite import SQLiteAdapter  # noqa: PLC0415
    from neuromem.system import NeuroMemory  # noqa: PLC0415
    from neuromem_gemini import GeminiEmbeddingProvider, GeminiLLMProvider  # noqa: PLC0415

    inner = GeminiLLMProvider(api_key=api_key)
    return NeuroMemory(
        storage=SQLiteAdapter(":memory:"),
        llm=inner,
        embedder=GeminiEmbeddingProvider(api_key=api_key),
        dream_threshold=9999,
    ), inner


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
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="(Opt-in) exit non-zero if mean wall exceeds the hard target. "
        "Default is measurement-only.",
    )
    args = parser.parse_args()

    walls: list[float] = []
    all_breakdown: list[dict[str, float]] = []
    for run in range(1, args.runs + 1):
        if args.mock_embedder:
            system, inner_llm = _build_system_mock()
        else:
            api_key = _resolve_api_key()
            if not api_key:
                sys.exit("ERROR: set GOOGLE_API_KEY or pass --mock-embedder")
            system, inner_llm = _build_system_real(api_key)

        # Wrap post-construction so enqueue's zero-LLM path isn't
        # counted (enqueue is always instant under ADR-004).
        instr = _InstrumentedLLM(inner_llm)
        system.llm = instr  # type: ignore[assignment]

        print(f"[run {run}/{args.runs}] ingesting {args.n} memories...")
        _ingest(system, n=args.n)
        print("  running dream cycle...")
        start = time.perf_counter()
        system.force_dream(block=True)
        elapsed = time.perf_counter() - start
        walls.append(elapsed)
        per_method = instr.total_per_method()
        all_breakdown.append(per_method)
        print(f"  dream-cycle wall: {elapsed:.2f}s")
        print("  per-method LLM time breakdown:")
        for name, total in sorted(per_method.items(), key=lambda kv: -kv[1]):
            print(f"    {name:<40} {total:>6.2f}s  (calls={len(instr.walls[name])})")

    mean = statistics.mean(walls)
    stdev = statistics.stdev(walls) if len(walls) > 1 else 0.0

    print("\n" + "=" * 72)
    print(f"ADR-004 ingestion — measurement only ({args.runs} runs, {args.n} memories)")
    print("=" * 72)
    print(f"  mean:     {mean:.2f}s")
    print(f"  stddev:   {stdev:.2f}s")
    print(f"  min:      {min(walls):.2f}s")
    print(f"  max:      {max(walls):.2f}s")

    verdict = "MEASURED"
    if args.fail_on_regression:
        hard_target_mock = 10.0  # sec — generous; tune once we have a pinned baseline
        hard_target_real = 300.0
        target = hard_target_mock if args.mock_embedder else hard_target_real
        verdict = "PASS" if mean <= target else "FAIL"
        print(f"  gate:     ≤ {target:.1f}s — {verdict}")

    sha = _git_sha()
    output_path = (
        Path(args.output)
        if args.output
        else _REPO_ROOT / "docs" / "benchmarks" / f"perf-ingestion-{sha}.json"
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
                "per_run_method_breakdown_s": all_breakdown,
                "verdict": verdict,
                "fail_on_regression": args.fail_on_regression,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nJSON written to: {output_path}")

    # Exit zero unless regression-gate was explicitly requested and failed.
    sys.exit(0 if verdict != "FAIL" else 1)


if __name__ == "__main__":
    main()
