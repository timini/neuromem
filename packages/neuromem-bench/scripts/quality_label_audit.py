"""ADR-003 F3 — centroid-label quality audit.

Samples N centroids post-dream, asks a judge LLM to rate each
label against its children on a 1–5 scale, and prints the
distribution. Also asserts the generic-noun blocklist holds (no
"thing" / "aspect" / etc. labels survive the dream cycle).

Acceptance gate (per the plan's success metrics): ≥ 90% of labels
rated 4 or 5; 0 labels in the generic blocklist.

Usage::

    GOOGLE_API_KEY=... uv run python \\
        packages/neuromem-bench/scripts/quality_label_audit.py \\
        --sample 50 \\
        --model gemini-2.0-flash-001
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

_REPO_ROOT = Path(__file__).resolve().parents[3]

# ADR-003 F3 generic-noun blocklist — must never appear as a label
# after dream-cycle naming.
GENERIC_BLOCKLIST = frozenset(
    {
        "thing",
        "things",
        "stuff",
        "misc",
        "topic",
        "topics",
        "entity",
        "entities",
        "aspect",
        "aspects",
        "factor",
        "factors",
        "concept",
        "concepts",
        "item",
        "items",
        "various",
        "general",
        "category",
        "categories",
        "element",
        "elements",
        "other",
    }
)

# Acceptance thresholds.
MIN_PCT_HIGH_QUALITY = 0.90  # >= 4/5 rating
MAX_GENERIC_HITS = 0


def _resolve_api_key() -> str:
    for env in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        v = os.environ.get(env, "").strip()
        if v:
            return v
    sys.exit("ERROR: GOOGLE_API_KEY/GEMINI_API_KEY not set")


def _build_corpus(api_key: str, model: str):  # type: ignore[no-untyped-def]
    """Build a moderately-sized fixture so we have centroids to audit."""
    from neuromem.storage.sqlite import SQLiteAdapter  # noqa: PLC0415
    from neuromem.system import NeuroMemory  # noqa: PLC0415
    from neuromem_gemini import GeminiEmbeddingProvider, GeminiLLMProvider  # noqa: PLC0415

    system = NeuroMemory(
        storage=SQLiteAdapter(":memory:"),
        llm=GeminiLLMProvider(api_key=api_key, model=model),
        embedder=GeminiEmbeddingProvider(api_key=api_key),
        dream_threshold=9999,
    )
    # Use a small but realistic LongMemEval-style ingestion.
    sample_texts = [
        "I graduated with a Business Administration degree last year.",
        "My morning commute is 45 minutes each way on the train.",
        "I redeemed a $5 coupon on coffee creamer at Target.",
        "Yoga classes at the studio downtown three times a week.",
        "Repainted the bedroom walls a deep blue called 'midnight'.",
        "Volunteered at the animal shelter's fundraising dinner.",
        "Bought a new tennis racket from the pro shop.",
        "Designer handbag cost me $850 — worth it though.",
        "Created a 'Summer Vibes' playlist on Spotify.",
        "Internet plan is 1 Gbps fiber, perfect for video calls.",
    ] * 5  # 50 memories — enough to produce many centroids
    for text in sample_texts:
        system.enqueue(text)
    system.force_dream(block=True)
    return system


def _judge_prompt(label: str, child_labels: list[str]) -> str:
    children_block = "\n".join(f"  - {c}" for c in child_labels)
    return (
        "Rate how well a category label describes its members on a "
        "1-5 scale (1=meaningless, 5=excellent abstraction).\n\n"
        f"Label: {label!r}\n"
        "Members:\n"
        f"{children_block}\n\n"
        "Reply with a single integer 1-5. No explanation, no markdown."
    )


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
    parser.add_argument("--sample", type=int, default=50)
    parser.add_argument("--model", default="gemini-2.0-flash-001")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    api_key = _resolve_api_key()
    print(f"Building corpus and dreaming with model {args.model}...")
    system = _build_corpus(api_key, args.model)

    # Render once on a generic query to force lazy-name resolution
    # of every centroid in the trunk.
    from neuromem.context import ContextHelper  # noqa: PLC0415

    helper = ContextHelper(system)
    helper.build_prompt_context("anything")

    nodes = system.storage.get_all_nodes()
    centroids = [n for n in nodes if n["is_centroid"]]
    if not centroids:
        sys.exit("No centroids produced — corpus too small for HDBSCAN")

    rng = random.Random(0)
    sample = rng.sample(centroids, k=min(args.sample, len(centroids)))
    print(f"Auditing {len(sample)} centroids out of {len(centroids)} total.")

    # For each, fetch its children's labels and ask the judge LLM.
    from neuromem_gemini.llm import GeminiLLMProvider, _generate_with_retry  # noqa: PLC0415

    judge = GeminiLLMProvider(api_key=api_key, model=args.model)
    rows: list[dict] = []
    blocklist_hits: list[str] = []
    for centroid in sample:
        sub = system.storage.get_subgraph([centroid["id"]], depth=1)
        children = [n["label"] for n in sub.get("nodes", []) if n["id"] != centroid["id"]]
        # Blocklist check first.
        label = (centroid["label"] or "").strip().lower()
        if label in GENERIC_BLOCKLIST:
            blocklist_hits.append(label)
        # Judge prompt.
        try:
            resp = _generate_with_retry(
                judge._client,  # type: ignore[attr-defined]
                judge._model,  # type: ignore[attr-defined]
                _judge_prompt(centroid["label"], children),
                bucket=judge._bucket,  # type: ignore[attr-defined]
            )
            text = (resp.text or "").strip()
            score = int(text.split()[0]) if text else 0
        except Exception as exc:  # pragma: no cover — network noise
            print(f"  judge call failed for {centroid['id']}: {exc}")
            score = 0
        rows.append(
            {
                "centroid_id": centroid["id"],
                "label": centroid["label"],
                "children": children,
                "score": score,
            }
        )

    high_quality = sum(1 for r in rows if r["score"] >= 4)
    pct_high = high_quality / len(rows) if rows else 0.0
    pass_quality = pct_high >= MIN_PCT_HIGH_QUALITY
    pass_blocklist = len(blocklist_hits) <= MAX_GENERIC_HITS
    overall = pass_quality and pass_blocklist

    print("\n" + "=" * 72)
    print(f"{len(rows)} centroids audited")
    print(
        f"  ≥4/5 ratings: {high_quality}/{len(rows)} ({pct_high:.0%}) — "
        f"target ≥ {MIN_PCT_HIGH_QUALITY:.0%}: "
        f"{'PASS' if pass_quality else 'FAIL'}"
    )
    print(
        f"  blocklist hits: {len(blocklist_hits)} — "
        f"target {MAX_GENERIC_HITS}: "
        f"{'PASS' if pass_blocklist else 'FAIL'}"
    )
    if blocklist_hits:
        print(f"    offenders: {sorted(set(blocklist_hits))}")
    print(f"  OVERALL: {'PASS' if overall else 'FAIL'}")

    sha = _git_sha()
    output_path = (
        Path(args.output)
        if args.output
        else (_REPO_ROOT / "docs" / "benchmarks" / f"quality-label-audit-{sha}.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "git_sha": sha,
                "model": args.model,
                "n_audited": len(rows),
                "high_quality": high_quality,
                "pct_high": pct_high,
                "blocklist_hits": blocklist_hits,
                "rows": rows,
                "overall": "PASS" if overall else "FAIL",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nJSON written to: {output_path}")
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
