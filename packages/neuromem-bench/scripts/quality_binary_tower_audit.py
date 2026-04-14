"""ADR-003 F5 — binary-tower elimination audit.

Walks the rendered tree from build_prompt_context output and counts
the longest single-child chain. Pre-ADR-003 (binary agglomerative)
trees routinely had 4+ deep chains where every middle centroid had
exactly one rendered child — meaningless tower hierarchies. The
ADR-003 success metric is: 0 trees in a 20-question sample contain
a single-child chain of length ≥ 4.

Pure analysis on the rendered ASCII output — no judge LLM needed.
Cheap and runs without Gemini.

Usage::

    GOOGLE_API_KEY=... uv run python \\
        packages/neuromem-bench/scripts/quality_binary_tower_audit.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Acceptance gate.
MAX_TOWER_LENGTH = 3  # chains of length >= 4 are forbidden


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


# Match a rendered folder/leaf row at any indent depth. The character
# at the start of the row tells us its depth (count of indent units).
# Indent unit is 4 chars: either "    " (no parent below) or "│   "
# (parent below). The connector "├── " or "└── " marks a child item.
_INDENT_PATTERN = re.compile(r"^(?:│   |    )*([├└]── )?(📁|📄)")


def _depth_of_line(line: str) -> int:
    """Return indent depth (in 4-char units) of a tree row, or -1 if
    the line isn't a render row (e.g. the header, blank line, para)."""
    if not _INDENT_PATTERN.match(line):
        return -1
    # Count leading "│   " or "    " 4-char units up to the connector.
    depth = 0
    i = 0
    while i + 4 <= len(line) and line[i : i + 4] in ("│   ", "    "):
        depth += 1
        i += 4
    return depth


def _immediate_children(parsed: list[tuple[int, bool]], parent_idx: int) -> list[tuple[int, bool]]:
    """Return the immediate children (depth = parent_depth + 1) of the
    centroid at ``parent_idx``, stopping when we exit the subtree.
    """
    parent_depth = parsed[parent_idx][0]
    children: list[tuple[int, bool]] = []
    for j in range(parent_idx + 1, len(parsed)):
        d_j, is_c_j = parsed[j]
        if d_j <= parent_depth:
            break
        if d_j == parent_depth + 1:
            children.append((d_j, is_c_j))
    return children


def _next_only_centroid_child_idx(parsed: list[tuple[int, bool]], parent_idx: int) -> int | None:
    """If the centroid at parent_idx has exactly one centroid child,
    return that child's index; otherwise None."""
    children = _immediate_children(parsed, parent_idx)
    if len(children) != 1 or not children[0][1]:
        return None
    parent_depth = parsed[parent_idx][0]
    for j in range(parent_idx + 1, len(parsed)):
        if parsed[j][0] == parent_depth + 1:
            return j
    return None


def _count_single_child_chains(tree: str) -> int:
    """Longest chain of centroids each with exactly one centroid child."""
    lines = [line for line in tree.splitlines() if _INDENT_PATTERN.match(line)]
    parsed = [(_depth_of_line(line), "📁" in line) for line in lines]

    max_chain = 0
    for i, (_, is_centroid) in enumerate(parsed):
        if not is_centroid:
            continue
        chain = 0
        cur = i
        while True:
            nxt = _next_only_centroid_child_idx(parsed, cur)
            if nxt is None:
                break
            chain += 1
            cur = nxt
        max_chain = max(max_chain, chain)
    return max_chain


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
    parser.add_argument("--n-queries", type=int, default=20)
    parser.add_argument("--model", default="gemini-2.0-flash-001")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    api_key = _resolve_api_key()
    print(f"Building corpus with {args.model}...")
    system = _build_corpus(api_key, args.model)

    from neuromem.context import ContextHelper  # noqa: PLC0415

    helper = ContextHelper(system)
    queries = [
        "What degree did I graduate with?",
        "How long is my commute?",
        "Where did I redeem coupons?",
        "Where do I take yoga?",
        "What colour are my walls?",
        "When did I volunteer?",
        "Where did I buy tennis gear?",
        "How much was the handbag?",
        "What's on my playlist?",
        "What's my internet speed?",
    ][: args.n_queries]
    rows: list[dict] = []
    for q in queries:
        tree = helper.build_prompt_context(q)
        max_chain = _count_single_child_chains(tree)
        rows.append(
            {
                "query": q,
                "max_single_child_chain": max_chain,
                "fail": max_chain > MAX_TOWER_LENGTH,
            }
        )

    failures = [r for r in rows if r["fail"]]
    overall = len(failures) == 0

    print("\n" + "=" * 72)
    print(f"Binary-tower audit: {len(rows)} queries")
    print(
        f"  trees with chain >{MAX_TOWER_LENGTH}: "
        f"{len(failures)} — target 0: "
        f"{'PASS' if overall else 'FAIL'}"
    )
    if failures:
        print("  offenders:")
        for r in failures:
            print(f"    {r['max_single_child_chain']}-deep on: {r['query']!r}")

    sha = _git_sha()
    output_path = (
        Path(args.output)
        if args.output
        else (_REPO_ROOT / "docs" / "benchmarks" / f"quality-binary-tower-{sha}.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "git_sha": sha,
                "model": args.model,
                "n_queries": len(rows),
                "max_chain_threshold": MAX_TOWER_LENGTH,
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
