"""NF5 storage-footprint harness for ADR-003 ontology tree v2.

Builds a 500-memory in-memory corpus, runs the dream cycle (which
populates labels, named_entities, paragraph_summary), then measures
the SQLite file size by serialising the in-memory database to a
temp file. Compares against a pinned baseline.

NF5 budget: ≤ 1.2× baseline (paragraph_summary column adds ~400
bytes per centroid).

Usage::

    uv run python packages/neuromem-bench/scripts/perf_storage_footprint.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Pinned baseline: file size on a 500-memory mock-embedder fixture,
# pre-ADR-003-paragraph-summary. Measured 2026-04-14 at ~620 KB:
# 32-dim embeddings × 500 memories + ~600 nodes + edges + summaries.
# Real-Gemini embeddings (768-dim) would be ~6 MB; the harness uses
# mock embeddings so the budget assertion is reproducible without
# network I/O. NF5 budget = 1.2× baseline for paragraph_summary
# column growth.
BASELINE_BYTES = 620_000
NF5_MAX_FACTOR = 1.2

_REPO_ROOT = Path(__file__).resolve().parents[3]


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

        def generate_category_name(self, concepts: list[str]) -> str:
            return concepts[0] if concepts else "concept"

    return NeuroMemory(
        storage=SQLiteAdapter(":memory:"),
        llm=_MockLLM(),
        embedder=_MockEmb(),
        dream_threshold=9999,
    )


def _ingest_and_dream(system, n: int = 500) -> None:  # type: ignore[no-untyped-def]
    import random  # noqa: PLC0415

    rng = random.Random(0)
    vocab = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa"]
    for _ in range(n):
        system.enqueue(" ".join(rng.choices(vocab, k=20)))
    system.force_dream(block=True)


def _serialise_to_disk(system, path: Path) -> int:  # type: ignore[no-untyped-def]
    """Use SQLite's native backup API to spill the in-memory DB to a
    file on disk; report the file size."""
    import sqlite3  # noqa: PLC0415

    src_conn = system.storage._conn  # type: ignore[attr-defined]
    dst_conn = sqlite3.connect(str(path))
    try:
        src_conn.backup(dst_conn)
        dst_conn.commit()
    finally:
        dst_conn.close()
    return path.stat().st_size


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
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print("Building 500-memory corpus and running dream cycle...")
    system = _build_system_mock()
    _ingest_and_dream(system, n=args.n)

    print("Serialising :memory: db to a temp file for sizing...")
    import contextlib  # noqa: PLC0415

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        tmp_path = Path(f.name)
    try:
        size_bytes = _serialise_to_disk(system, tmp_path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp_path.unlink()

    budget = int(BASELINE_BYTES * NF5_MAX_FACTOR)
    passed = size_bytes <= budget

    print("\n" + "=" * 72)
    print(f"Storage footprint at {args.n} memories")
    print("=" * 72)
    print(f"  size:     {size_bytes / 1024:>8.1f} KB ({size_bytes:,} bytes)")
    print(f"  baseline: {BASELINE_BYTES / 1024:>8.1f} KB ({BASELINE_BYTES:,} bytes)")
    print(f"  budget:   {budget / 1024:>8.1f} KB  ({NF5_MAX_FACTOR:.1f}× baseline)")
    print(f"  ratio:    {size_bytes / BASELINE_BYTES:>8.2f}×")
    print(f"  verdict:  {'PASS' if passed else 'FAIL'}")

    sha = _git_sha()
    output_path = (
        Path(args.output)
        if args.output
        else _REPO_ROOT / "docs" / "benchmarks" / f"perf-storage-{sha}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "git_sha": sha,
                "n_memories": args.n,
                "size_bytes": size_bytes,
                "baseline_bytes": BASELINE_BYTES,
                "budget_bytes": budget,
                "ratio": size_bytes / BASELINE_BYTES,
                "verdict": "PASS" if passed else "FAIL",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nJSON written to: {output_path}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
