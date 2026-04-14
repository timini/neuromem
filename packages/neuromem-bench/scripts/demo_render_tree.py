"""Render-tree preview — what the ADK agent actually sees injected.

Uses the same 5-memory fixture as demo_adk_agent.py but skips the
agent entirely and just prints the ASCII tree ContextHelper.build_prompt_context
produces for a given question.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", message=r".*\[EXPERIMENTAL\].*", category=UserWarning)

_REPO_ROOT = Path(__file__).resolve().parents[3]

MEMORIES = [
    (
        "user: I'm thinking of taking up a new hobby, any recommendations?\n"
        "A: Sure! What are you into?\n"
        "user: I like being outside. I was thinking rock climbing, but actually "
        "I just signed up for bouldering classes at The Crag — it's about a "
        "15 minute walk from my apartment in Shoreditch."
    ),
    (
        'user: I need some laptop advice. I have a 2019 MacBook Pro 13" and '
        "the battery's dying.\n"
        "A: How much do you want to spend?\n"
        'user: Budget around £2000. I ended up ordering the M3 MacBook Air 15" '
        "in space grey, 24GB RAM, 1TB SSD. Arrived yesterday and I'm loving it."
    ),
    (
        "user: Trying to remember the name of that great restaurant we went to "
        "for Sarah's birthday last month.\n"
        "A: There were a few options, which area?\n"
        "user: It was Sabor on Heddon Street — the Spanish place with the "
        "suckling pig. March 14th, cost about £180 for the two of us."
    ),
    (
        "user: Quick question, what was my MOT expiry date on the Golf?\n"
        "A: I don't have that on me — check the V5.\n"
        "user: Oh I just remembered, it's October 12th 2026. I'll book the "
        "service at Hendy VW in Southampton next week."
    ),
    (
        "user: My eye doctor said my prescription hasn't changed. Still -2.75 "
        "in the right, -3.00 in the left with slight astigmatism. Ordered two "
        "new pairs from Cubitts, £320 total."
    ),
]


def _resolve_api_key() -> str:
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
                value = val.strip().strip("'\"")
                os.environ.setdefault("GOOGLE_API_KEY", value)
                os.environ.setdefault("GEMINI_API_KEY", value)
                return value
    sys.exit("ERROR: GOOGLE_API_KEY / GEMINI_API_KEY not set")


def main() -> None:
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemini-2.0-flash-001")
    parser.add_argument("--query", default="Where do I go bouldering?")
    args = parser.parse_args()

    api_key = _resolve_api_key()

    from neuromem import NeuroMemory, SQLiteAdapter  # noqa: PLC0415
    from neuromem.context import ContextHelper  # noqa: PLC0415
    from neuromem_gemini import GeminiEmbeddingProvider, GeminiLLMProvider  # noqa: PLC0415

    print(f"Building NeuroMemory with model {args.model}...")
    memory = NeuroMemory(
        storage=SQLiteAdapter(":memory:"),
        llm=GeminiLLMProvider(api_key=api_key, model=args.model),
        embedder=GeminiEmbeddingProvider(api_key=api_key),
        dream_threshold=9999,
    )

    print("Enqueueing 5 memories (non-blocking)...")
    for i, raw in enumerate(MEMORIES, 1):
        memory.enqueue(raw, metadata={"demo_idx": i})

    print("Running dream cycle (eager full-sweep bake)...")
    memory.force_dream(block=True)

    print(f"\nQuery: {args.query!r}\n")
    print("=" * 72)
    print("Rendered tree (what ADK's before_model_callback injects):")
    print("=" * 72)
    helper = ContextHelper(memory)
    tree = helper.build_prompt_context(args.query)
    print(tree)


if __name__ == "__main__":
    main()
