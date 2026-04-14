"""NF1/NF2 render-latency harness for ADR-003 ontology tree v2.

Measures ``ContextHelper.build_prompt_context`` wall time on a fixed
500-memory corpus for 50 distinct queries. Runs two passes:

  - **Cold pass**: fresh SQLite, first-time queries. Every centroid
    in the rendered subgraph that the eager trunk pass didn't cover
    pays the lazy summary LLM call. NF1 budget: p50 ≤ 1000 ms,
    p95 ≤ 1500 ms.
  - **Warm pass**: same queries re-issued. Every centroid name and
    paragraph summary should now be cached in storage — zero LLM
    calls expected. NF2 budget: p50 ≤ 500 ms.

Usage::

    # Quick sanity check (mock embedder, no Gemini needed):
    uv run python packages/neuromem-bench/scripts/perf_render_latency.py \\
        --mock-embedder

    # Full-stack measurement (real Gemini, costs money):
    GOOGLE_API_KEY=... uv run python \\
        packages/neuromem-bench/scripts/perf_render_latency.py

Output JSON lands at ``docs/benchmarks/perf-render-<git-sha>.json``.
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

# NF targets from ADR-003.
NF1_P50_COLD_MS = 1000
NF1_P95_COLD_MS = 1500
NF2_P50_WARM_MS = 500

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve_api_key() -> str | None:
    for env in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        v = os.environ.get(env, "").strip()
        if v:
            return v
    env_file = _REPO_ROOT / ".env"
    if env_file.is_file():
        for raw in env_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, val = line.partition("=")
            if k.strip() in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
                return val.strip().strip("'\"")
    return None


def _build_fixture_mock() -> tuple:
    """Build a fixture without hitting Gemini — deterministic mock
    embeddings and trivial summaries. For CI / cheap regression
    runs."""
    import numpy as np  # noqa: PLC0415
    from neuromem.providers import EmbeddingProvider, LLMProvider  # noqa: PLC0415
    from neuromem.storage.sqlite import SQLiteAdapter  # noqa: PLC0415
    from neuromem.system import NeuroMemory  # noqa: PLC0415

    class _MockEmb(EmbeddingProvider):
        def __init__(self) -> None:
            self._dim = 32

        def get_embeddings(self, texts: list[str]):  # type: ignore[override]
            rng = np.random.default_rng(seed=hash(" ".join(texts)) & 0xFFFFFFFF)
            return rng.normal(size=(len(texts), self._dim)).astype(np.float32)

    class _MockLLM(LLMProvider):
        def generate_summary(self, raw_text: str) -> str:
            return raw_text[:120]

        def extract_tags(self, summary: str) -> list[str]:
            return summary.split()[:3]

        def generate_category_name(self, concepts: list[str]) -> str:
            return concepts[0] if concepts else "concept"

    # dream_threshold=9999 prevents the auto-dream background thread
    # from firing during enqueue. SQLite's :memory: connection is not
    # threadsafe and the bench wants a deterministic single-thread
    # measurement; the caller invokes force_dream(block=True) once.
    system = NeuroMemory(
        storage=SQLiteAdapter(":memory:"),
        llm=_MockLLM(),
        embedder=_MockEmb(),
        dream_threshold=9999,
    )
    return system, _MockEmb(), _MockLLM()


def _build_fixture_real(api_key: str) -> tuple:
    """Build a fixture using real Gemini providers."""
    from neuromem.storage.sqlite import SQLiteAdapter  # noqa: PLC0415
    from neuromem.system import NeuroMemory  # noqa: PLC0415
    from neuromem_gemini import (  # noqa: PLC0415
        GeminiEmbeddingProvider,
        GeminiLLMProvider,
    )

    embedder = GeminiEmbeddingProvider(api_key=api_key)
    llm = GeminiLLMProvider(api_key=api_key)
    system = NeuroMemory(
        storage=SQLiteAdapter(":memory:"),
        llm=llm,
        embedder=embedder,
        dream_threshold=9999,
    )
    return system, embedder, llm


def _ingest_500(system) -> None:  # type: ignore[no-untyped-def]
    """Enqueue 500 small memories and force one dream cycle."""
    import random  # noqa: PLC0415

    rng = random.Random(42)
    vocab = [
        "degree",
        "business",
        "administration",
        "mortgage",
        "loan",
        "rate",
        "coffee",
        "creamer",
        "target",
        "coupon",
        "commute",
        "minutes",
        "yoga",
        "class",
        "studio",
        "bedroom",
        "wall",
        "paint",
        "blue",
        "volunteer",
        "animal",
        "shelter",
        "dinner",
        "fundraise",
        "tennis",
        "racket",
        "handbag",
        "designer",
        "playlist",
        "spotify",
        "summer",
        "vibes",
        "last",
        "name",
        "change",
        "marriage",
        "occupation",
        "software",
        "engineer",
        "graduate",
        "university",
        "research",
        "travel",
        "costa",
        "rica",
        "pack",
        "shirt",
        "bookshelf",
        "ikea",
        "internet",
        "plan",
        "speed",
        "gbps",
        "spirituality",
        "meditation",
        "dog",
        "cat",
        "pet",
        "walk",
        "park",
        "family",
        "sister",
        "brother",
        "birthday",
        "gift",
        "personalised",
        "blanket",
        "lucas",
        "rachel",
        "craft",
        "fair",
        "market",
        "international",
        "handmade",
        "candle",
        "soap",
        "jewellery",
        "ghost",
        "kitchen",
        "steak",
        "case",
        "study",
        "portfolio",
        "senior",
        "designer",
        "adobe",
        "figma",
        "prototype",
    ]
    for _ in range(500):
        n_tokens = rng.randint(12, 28)
        text = " ".join(rng.choices(vocab, k=n_tokens))
        system.enqueue(text)
    system.force_dream(block=True)


def _build_queries() -> list[str]:
    """50 distinct search-style queries anchored on the fixture vocab."""
    bases = [
        "What degree did I graduate with?",
        "How much is my mortgage rate?",
        "Where did I redeem a coupon?",
        "How long is my commute?",
        "Where do I take yoga?",
        "What colour are my bedroom walls?",
        "When did I volunteer at the shelter?",
        "Where did I buy my tennis racket?",
        "How much was the designer handbag?",
        "What's on my Spotify playlist?",
    ]
    out: list[str] = []
    for base in bases:
        for suffix in (
            "",
            " last year",
            " recently",
            " during the weekend",
            " for my friend",
        ):
            out.append(base + suffix)
    return out[:50]


def _measure_pass(helper, queries: list[str]) -> list[float]:  # type: ignore[no-untyped-def]
    walls: list[float] = []
    for q in queries:
        start = time.perf_counter()
        _ = helper.build_prompt_context(q)
        walls.append((time.perf_counter() - start) * 1000.0)
    return walls


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = int(len(sorted_values) * p / 100.0)
    idx = max(0, min(idx, len(sorted_values) - 1))
    return sorted_values[idx]


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
    parser.add_argument(
        "--mock-embedder",
        action="store_true",
        help="Use deterministic mock providers instead of Gemini",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: docs/benchmarks/perf-render-<sha>.json)",
    )
    args = parser.parse_args()

    if args.mock_embedder:
        print("Building fixture with MOCK providers (no network I/O)...")
        system, _, _ = _build_fixture_mock()
    else:
        api_key = _resolve_api_key()
        if not api_key:
            sys.exit(
                "ERROR: GOOGLE_API_KEY/GEMINI_API_KEY not set. Use --mock-embedder for a cheap run."
            )
        print("Building fixture with REAL Gemini providers (costs money)...")
        system, _, _ = _build_fixture_real(api_key)

    print("Ingesting 500 memories + running dream cycle...")
    t0 = time.perf_counter()
    _ingest_500(system)
    ingest_wall = time.perf_counter() - t0
    print(f"  ingestion + dream: {ingest_wall:.1f}s")

    from neuromem.context import ContextHelper  # noqa: PLC0415

    helper = ContextHelper(system)
    queries = _build_queries()

    print(f"\nCold-cache pass over {len(queries)} queries...")
    cold = _measure_pass(helper, queries)
    print("Warm-cache pass over same queries...")
    warm = _measure_pass(helper, queries)

    cold_p50 = statistics.median(cold)
    cold_p95 = _pct(cold, 95)
    warm_p50 = statistics.median(warm)
    warm_p95 = _pct(warm, 95)

    print("\n" + "=" * 72)
    print(f"{'Pass':<12}{'p50 (ms)':>12}{'p95 (ms)':>12}{'min':>10}{'max':>10}{'target':>16}")
    print("-" * 72)
    print(
        f"{'cold':<12}{cold_p50:>12.1f}{cold_p95:>12.1f}"
        f"{min(cold):>10.1f}{max(cold):>10.1f}"
        f"{f'p50≤{NF1_P50_COLD_MS}, p95≤{NF1_P95_COLD_MS}':>16}"
    )
    print(
        f"{'warm':<12}{warm_p50:>12.1f}{warm_p95:>12.1f}"
        f"{min(warm):>10.1f}{max(warm):>10.1f}"
        f"{f'p50≤{NF2_P50_WARM_MS}':>16}"
    )
    print("=" * 72)

    # Budget evaluation.
    cold_pass = cold_p50 <= NF1_P50_COLD_MS and cold_p95 <= NF1_P95_COLD_MS
    warm_pass = warm_p50 <= NF2_P50_WARM_MS
    verdict = "PASS" if (cold_pass and warm_pass) else "FAIL"
    print(f"\nNF1 (cold-cache render): {'PASS' if cold_pass else 'FAIL'}")
    print(f"NF2 (warm-cache render): {'PASS' if warm_pass else 'FAIL'}")
    print(f"OVERALL: {verdict}")

    sha = _git_sha()
    output_path = (
        Path(args.output)
        if args.output
        else _REPO_ROOT / "docs" / "benchmarks" / f"perf-render-{sha}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "git_sha": sha,
                "mock_embedder": args.mock_embedder,
                "ingestion_wall_s": ingest_wall,
                "cold_walls_ms": cold,
                "warm_walls_ms": warm,
                "cold_p50_ms": cold_p50,
                "cold_p95_ms": cold_p95,
                "warm_p50_ms": warm_p50,
                "warm_p95_ms": warm_p95,
                "nf1_pass": cold_pass,
                "nf2_pass": warm_pass,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nJSON written to: {output_path}")

    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
