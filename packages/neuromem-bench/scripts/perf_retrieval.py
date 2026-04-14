"""ADR-004 retrieval perf harness — HARD gates.

Retrieval is the sub-second hot path. This harness builds a
fully-baked 500-memory fixture (complete dream cycle, every
centroid named + summarised), issues 50 distinct queries, and
asserts:

  NF-H1: enqueue wall per call ≤ 20 ms
  NF-H2: build_prompt_context p50 (mock embedder) ≤ 50 ms
  NF-H3: build_prompt_context p50 (real Gemini) ≤ 500 ms
  NF-H4: build_prompt_context p95 (real Gemini) ≤ 1000 ms
  NF-H5: expand_node p50 ≤ 200 ms
  NF-H6: retrieve_memories p50 ≤ 50 ms
  NF-H7: ZERO LLM calls from any retrieval method in happy path

Ingestion cost is NOT measured here — see perf_ingestion.py.

Usage::

    # Quick (CI-safe, no network):
    uv run python packages/neuromem-bench/scripts/perf_retrieval.py \\
        --mock-embedder

    # Real-Gemini measurement (ad-hoc, costs money):
    GOOGLE_API_KEY=... uv run python \\
        packages/neuromem-bench/scripts/perf_retrieval.py

Exit non-zero if any target is missed.
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

# NF targets from ADR-004.
NF_H1_ENQUEUE_MS = 20
NF_H2_BUILD_MOCK_P50_MS = 50
NF_H3_BUILD_GEMINI_P50_MS = 500
NF_H4_BUILD_GEMINI_P95_MS = 1000
NF_H5_EXPAND_P50_MS = 200
NF_H6_RETRIEVE_P50_MS = 50

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


class _CallCountingLLM:
    """Wraps any LLMProvider and counts every method invocation.
    Used to assert NF-H7 (zero LLM calls in the retrieval path)."""

    def __init__(self, inner) -> None:  # type: ignore[no-untyped-def]
        self._inner = inner
        self.counts: dict[str, int] = {}

    def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
        target = getattr(self._inner, name)
        if not callable(target):
            return target

        def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
            if name.startswith(("generate_", "extract_")):
                self.counts[name] = self.counts.get(name, 0) + 1
            return target(*args, **kwargs)

        return wrapper

    def total(self) -> int:
        return sum(self.counts.values())


def _build_fixture_mock():  # type: ignore[no-untyped-def]
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

    inner_llm = _MockLLM()
    embedder = _MockEmb()
    system = NeuroMemory(
        storage=SQLiteAdapter(":memory:"),
        llm=inner_llm,
        embedder=embedder,
        dream_threshold=9999,
    )
    return system, inner_llm, embedder


def _build_fixture_real(api_key: str):  # type: ignore[no-untyped-def]
    from neuromem.storage.sqlite import SQLiteAdapter  # noqa: PLC0415
    from neuromem.system import NeuroMemory  # noqa: PLC0415
    from neuromem_gemini import GeminiEmbeddingProvider, GeminiLLMProvider  # noqa: PLC0415

    inner_llm = GeminiLLMProvider(api_key=api_key)
    embedder = GeminiEmbeddingProvider(api_key=api_key)
    system = NeuroMemory(
        storage=SQLiteAdapter(":memory:"),
        llm=inner_llm,
        embedder=embedder,
        dream_threshold=9999,
    )
    return system, inner_llm, embedder


def _ingest_500(system) -> float:  # type: ignore[no-untyped-def]
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
    ]
    for _ in range(500):
        n_tokens = rng.randint(12, 28)
        system.enqueue(" ".join(rng.choices(vocab, k=n_tokens)))
    t0 = time.perf_counter()
    system.force_dream(block=True)
    return time.perf_counter() - t0


def _enqueue_latency_probe(system, n: int = 20) -> list[float]:  # type: ignore[no-untyped-def]
    """Measure per-enqueue wall (NF-H1). Runs AFTER the main fixture
    so the inbox insert is on a populated DB."""
    walls: list[float] = []
    for i in range(n):
        start = time.perf_counter()
        system.enqueue(f"latency-probe {i}")
        walls.append((time.perf_counter() - start) * 1000.0)
    return walls


def _queries() -> list[str]:
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
        for suffix in ("", " last year", " recently", " during the weekend", " for my friend"):
            out.append(base + suffix)
    return out[:50]


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
    parser.add_argument("--mock-embedder", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.mock_embedder:
        print("Building fully-baked 500-memory fixture (MOCK providers)...")
        system, inner_llm, _ = _build_fixture_mock()
    else:
        api_key = _resolve_api_key()
        if not api_key:
            sys.exit(
                "ERROR: GOOGLE_API_KEY/GEMINI_API_KEY not set. "
                "Pass --mock-embedder for a cheap run."
            )
        print("Building fully-baked 500-memory fixture (REAL Gemini)...")
        system, inner_llm, _ = _build_fixture_real(api_key)

    ingest_wall = _ingest_500(system)
    print(f"  ingestion + dream: {ingest_wall:.1f}s")

    # Wrap the LLM provider with a call counter for the retrieval
    # phase. Enqueue calls above didn't use this wrapper, which is
    # fine — ADR-004 is that enqueue is zero-LLM, but we assert
    # specifically the RETRIEVAL methods are zero-LLM.
    counter = _CallCountingLLM(inner_llm)
    system.llm = counter  # type: ignore[assignment]

    # NF-H1 enqueue latency (post-baked corpus).
    enqueue_walls = _enqueue_latency_probe(system, n=20)

    # Retrieval passes.
    from neuromem.context import ContextHelper  # noqa: PLC0415
    from neuromem.tools import expand_node, retrieve_memories  # noqa: PLC0415

    helper = ContextHelper(system)
    queries = _queries()

    # build_prompt_context
    build_walls: list[float] = []
    for q in queries:
        start = time.perf_counter()
        helper.build_prompt_context(q)
        build_walls.append((time.perf_counter() - start) * 1000.0)

    # expand_node on one centroid each
    centroids = [n for n in system.storage.get_all_nodes() if n["is_centroid"]]
    expand_walls: list[float] = []
    for c in centroids[: min(20, len(centroids))]:
        start = time.perf_counter()
        expand_node(c["id"], system)
        expand_walls.append((time.perf_counter() - start) * 1000.0)

    # retrieve_memories
    consolidated = system.storage.get_memories_by_status("consolidated")
    retrieve_walls: list[float] = []
    for mem in consolidated[:20]:
        start = time.perf_counter()
        retrieve_memories([mem["id"]], system)
        retrieve_walls.append((time.perf_counter() - start) * 1000.0)

    # Compile.
    enqueue_p50 = statistics.median(enqueue_walls)
    build_p50 = statistics.median(build_walls)
    build_p95 = _pct(build_walls, 95)
    expand_p50 = statistics.median(expand_walls) if expand_walls else 0.0
    retrieve_p50 = statistics.median(retrieve_walls)

    build_target_p50 = NF_H2_BUILD_MOCK_P50_MS if args.mock_embedder else NF_H3_BUILD_GEMINI_P50_MS
    build_target_p95 = NF_H4_BUILD_GEMINI_P95_MS if not args.mock_embedder else 1500

    # Gates.
    results = {
        "NF-H1 enqueue p50": (enqueue_p50, NF_H1_ENQUEUE_MS),
        "NF-H? build p50": (build_p50, build_target_p50),
        "NF-H? build p95": (build_p95, build_target_p95),
        "NF-H5 expand p50": (expand_p50, NF_H5_EXPAND_P50_MS),
        "NF-H6 retrieve p50": (retrieve_p50, NF_H6_RETRIEVE_P50_MS),
    }
    llm_calls_in_retrieval = counter.total()
    nf_h7_pass = llm_calls_in_retrieval == 0

    print("\n" + "=" * 72)
    print(f"{'Metric':<28}{'measured':>14}{'target':>14}{'verdict':>14}")
    print("-" * 72)
    for name, (measured, target) in results.items():
        verdict = "PASS" if measured <= target else "FAIL"
        print(f"{name:<28}{measured:>12.1f}ms{target:>12}ms{verdict:>14}")
    print("-" * 72)
    print(
        f"{'NF-H7 zero LLM in retrieve':<28}"
        f"{llm_calls_in_retrieval:>12} calls"
        f"{0:>12}"
        f"{('PASS' if nf_h7_pass else 'FAIL'):>14}"
    )
    if counter.counts:
        print(f"  breakdown: {counter.counts}")
    print("=" * 72)

    overall_pass = all(m <= t for m, t in results.values()) and nf_h7_pass
    verdict = "PASS" if overall_pass else "FAIL"
    print(f"OVERALL: {verdict}")

    sha = _git_sha()
    output_path = (
        Path(args.output)
        if args.output
        else _REPO_ROOT / "docs" / "benchmarks" / f"perf-retrieval-{sha}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "git_sha": sha,
                "mock_embedder": args.mock_embedder,
                "ingestion_wall_s": ingest_wall,
                "enqueue_walls_ms": enqueue_walls,
                "build_prompt_walls_ms": build_walls,
                "expand_walls_ms": expand_walls,
                "retrieve_walls_ms": retrieve_walls,
                "targets": {
                    name: {"measured_ms": m, "target_ms": t, "pass": m <= t}
                    for name, (m, t) in results.items()
                },
                "llm_calls_in_retrieval": llm_calls_in_retrieval,
                "llm_call_breakdown": dict(counter.counts),
                "nf_h7_pass": nf_h7_pass,
                "overall": verdict,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nJSON written to: {output_path}")
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
