"""ADR-003 F4 — paragraph-summary faithfulness audit.

Samples N centroids that have a paragraph_summary populated, asks
a judge LLM to rate each summary on a 1–5 scale for "does this
summary fairly represent the children", and reports the
distribution.

Acceptance gate: mean ≥ 4.0, p10 ≥ 3.0. Any centroid scoring ≤ 2
is flagged as a HUMAN-REVIEW row in the JSON output.

Usage::

    GOOGLE_API_KEY=... uv run python \\
        packages/neuromem-bench/scripts/quality_summary_audit.py \\
        --sample 50
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Acceptance thresholds.
MIN_MEAN_SCORE = 4.0
MIN_P10_SCORE = 3.0
HUMAN_REVIEW_THRESHOLD = 2  # rows scoring <= this flagged for review


def _resolve_api_key() -> str:
    for env in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        v = os.environ.get(env, "").strip()
        if v:
            return v
    sys.exit("ERROR: GOOGLE_API_KEY/GEMINI_API_KEY not set")


def _build_corpus(api_key: str, model: str):  # type: ignore[no-untyped-def]
    from neuromem.storage.sqlite import SQLiteAdapter  # noqa: PLC0415
    from neuromem.system import NeuroMemory  # noqa: PLC0415
    from neuromem_gemini import GeminiEmbeddingProvider, GeminiLLMProvider  # noqa: PLC0415

    system = NeuroMemory(
        storage=SQLiteAdapter(":memory:"),
        llm=GeminiLLMProvider(api_key=api_key, model=model),
        embedder=GeminiEmbeddingProvider(api_key=api_key),
        dream_threshold=9999,
    )
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
    ] * 5
    for text in sample_texts:
        system.enqueue(text)
    system.force_dream(block=True)
    return system


def _judge_prompt(summary: str, child_descriptions: list[str]) -> str:
    children_block = "\n".join(f"  - {c}" for c in child_descriptions)
    return (
        "Rate how faithfully a summary describes its source content "
        "on a 1-5 scale (1=misleading or omits key facts, "
        "5=accurately represents every fact mentioned).\n\n"
        f"Summary: {summary}\n\n"
        "Source content:\n"
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

    # Force render so lazy summaries get filled.
    from neuromem.context import ContextHelper  # noqa: PLC0415

    ContextHelper(system).build_prompt_context("anything")

    nodes = system.storage.get_all_nodes()
    summarised = [n for n in nodes if n["is_centroid"] and n.get("paragraph_summary")]
    if not summarised:
        sys.exit("No centroids have paragraph_summary populated")

    rng = random.Random(0)
    sample = rng.sample(summarised, k=min(args.sample, len(summarised)))
    print(f"Auditing {len(sample)} summarised centroids out of {len(summarised)}.")

    from neuromem_gemini.llm import GeminiLLMProvider, _generate_with_retry  # noqa: PLC0415

    judge = GeminiLLMProvider(api_key=api_key, model=args.model)
    rows: list[dict] = []
    for centroid in sample:
        sub = system.storage.get_subgraph([centroid["id"]], depth=1)
        # Gather per-child descriptions: the paragraph_summary if a
        # centroid child, the leaf label otherwise.
        child_descs: list[str] = []
        for n in sub.get("nodes", []):
            if n["id"] == centroid["id"]:
                continue
            desc = n.get("paragraph_summary") or n.get("label", "")
            if desc:
                child_descs.append(str(desc))
        # Append memory summaries attached via has_tag to children.
        memories_by_id = {m["id"]: m for m in sub.get("memories", [])}
        for edge in sub.get("edges", []):
            if edge.get("relationship") != "has_tag":
                continue
            mem = memories_by_id.get(edge.get("source_id", ""))
            if mem and mem.get("summary"):
                child_descs.append(str(mem["summary"]))

        try:
            resp = _generate_with_retry(
                judge._client,  # type: ignore[attr-defined]
                judge._model,  # type: ignore[attr-defined]
                _judge_prompt(centroid["paragraph_summary"], child_descs),
                bucket=judge._bucket,  # type: ignore[attr-defined]
            )
            text = (resp.text or "").strip()
            score = int(text.split()[0]) if text else 0
        except Exception as exc:  # pragma: no cover
            print(f"  judge call failed for {centroid['id']}: {exc}")
            score = 0
        rows.append(
            {
                "centroid_id": centroid["id"],
                "label": centroid["label"],
                "summary": centroid["paragraph_summary"],
                "n_children": len(child_descs),
                "score": score,
            }
        )

    scores = [r["score"] for r in rows if r["score"] > 0]
    mean_score = statistics.mean(scores) if scores else 0.0
    p10 = statistics.quantiles(scores, n=10)[0] if len(scores) >= 10 else min(scores or [0])
    flagged = [r for r in rows if r["score"] <= HUMAN_REVIEW_THRESHOLD]

    pass_mean = mean_score >= MIN_MEAN_SCORE
    pass_p10 = p10 >= MIN_P10_SCORE
    overall = pass_mean and pass_p10

    print("\n" + "=" * 72)
    print(f"{len(rows)} summaries audited (scores 1-5)")
    print(
        f"  mean: {mean_score:.2f} — target ≥ {MIN_MEAN_SCORE}: {'PASS' if pass_mean else 'FAIL'}"
    )
    print(f"  p10:  {p10:.1f} — target ≥ {MIN_P10_SCORE}: {'PASS' if pass_p10 else 'FAIL'}")
    print(f"  flagged (≤{HUMAN_REVIEW_THRESHOLD}/5): {len(flagged)}")
    print(f"  OVERALL: {'PASS' if overall else 'FAIL'}")

    sha = _git_sha()
    output_path = (
        Path(args.output)
        if args.output
        else (_REPO_ROOT / "docs" / "benchmarks" / f"quality-summary-audit-{sha}.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "git_sha": sha,
                "model": args.model,
                "n_audited": len(rows),
                "mean_score": mean_score,
                "p10": p10,
                "flagged_for_review": [r["centroid_id"] for r in flagged],
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
