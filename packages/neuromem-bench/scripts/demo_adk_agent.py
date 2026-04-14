"""Live demo: ADK agent with neuromem long-term memory (ADR-004).

Ingests 5 hand-crafted memories about a fictional user, runs one
dream cycle to bake them into the ontology, then asks three
questions and measures:

  - injected-tree render wall (NF-H2 hot-path metric)
  - tool-call sequence
  - full answer wall (includes Gemini's own answer-LLM + any tool
    round-trips, which dominate)

Run::

    GOOGLE_API_KEY=... uv run python \\
        packages/neuromem-bench/scripts/demo_adk_agent.py \\
        --model gemini-2.0-flash-001
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", message=r".*\[EXPERIMENTAL\].*", category=UserWarning)

_REPO_ROOT = Path(__file__).resolve().parents[3]

MEMORIES = [
    # Fictional user history — the kind of content that would
    # accumulate across multiple past sessions.
    (
        "user: I'm thinking of taking up a new hobby, any recommendations?\n"
        "A: Sure! What are you into?\n"
        "user: I like being outside. I was thinking rock climbing, "
        "but actually I just signed up for bouldering classes at The "
        "Crag — it's about a 15 minute walk from my apartment in "
        "Shoreditch."
    ),
    (
        'user: I need some laptop advice. I have a 2019 MacBook Pro 13" '
        "and the battery's dying.\n"
        "A: How much do you want to spend?\n"
        "user: Budget around £2000. I ended up ordering the M3 MacBook "
        'Air 15" in space grey, 24GB RAM, 1TB SSD. Arrived yesterday '
        "and I'm loving it."
    ),
    (
        "user: Trying to remember the name of that great restaurant "
        "we went to for Sarah's birthday last month.\n"
        "A: There were a few options, which area?\n"
        "user: It was Sabor on Heddon Street — the Spanish place with "
        "the suckling pig. March 14th, cost about £180 for the two of "
        "us. Worth every penny."
    ),
    (
        "user: Quick question, what was my MOT expiry date on the Golf?\n"
        "A: I don't have that on me — check the V5.\n"
        "user: Oh I just remembered, it's October 12th 2026. I'll book "
        "the service at Hendy VW in Southampton next week."
    ),
    (
        "user: My eye doctor said my prescription hasn't changed. "
        "Still -2.75 in the right, -3.00 in the left with slight "
        "astigmatism. Ordered two new pairs from Cubitts, £320 total."
    ),
]

QUESTIONS = [
    "Where do I go bouldering?",
    "What laptop do I have?",
    "When does my MOT expire?",
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


def _configure_logging() -> None:
    h = logging.StreamHandler(stream=sys.stderr)
    h.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    bl = logging.getLogger("neuromem_bench")
    bl.setLevel(logging.INFO)
    bl.addHandler(h)
    bl.propagate = False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemini-2.0-flash-001")
    args = parser.parse_args()

    _configure_logging()
    api_key = _resolve_api_key()

    print("=" * 72)
    print(f"neuromem-adk demo — model={args.model}")
    print("=" * 72)

    print("\n1. Agent instruction (the prompt):")
    print("-" * 72)
    from neuromem_bench.prompts import load_prompt  # noqa: PLC0415

    print(load_prompt("adk_agent_instruction"))
    print("-" * 72)

    print("\n2. Ingesting 5 memories...")
    from neuromem_bench.agent import NeuromemAdkAgent  # noqa: PLC0415

    agent = NeuromemAdkAgent(api_key=api_key, model=args.model)
    t0 = time.perf_counter()
    for i, raw in enumerate(MEMORIES, 1):
        # Each "memory" is a whole session transcript. enqueue is
        # non-blocking (ADR-004 D1) — each call returns in a few ms.
        agent._memory.enqueue(raw, metadata={"demo_idx": i})  # type: ignore[attr-defined]
    ingest_wall = (time.perf_counter() - t0) * 1000.0
    print(f"   wall: {ingest_wall:.1f} ms (pure SQL inserts — ADR-004 D1)")

    print("\n3. Running dream cycle (bakes every centroid's label + summary)...")
    t0 = time.perf_counter()
    agent._memory.force_dream(block=True)  # type: ignore[attr-defined]
    dream_wall = time.perf_counter() - t0
    print(f"   wall: {dream_wall:.1f} s (background work — not on retrieval hot path)")

    # Measure the retrieval hot path by itself (no ADK, just the
    # injected-tree render that before_model_callback does).
    print("\n4. Measuring retrieval (what the agent sees injected each turn):")
    from neuromem.context import ContextHelper  # noqa: PLC0415

    helper = ContextHelper(agent._memory)  # type: ignore[attr-defined]
    for q in QUESTIONS:
        t0 = time.perf_counter()
        tree = helper.build_prompt_context(q)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        lines = tree.count("\n") if tree else 0
        print(f"   Q: {q!r}")
        print(f"     retrieval wall: {wall_ms:.1f} ms   tree: {lines} lines")

    print("\n5. Running the agent end-to-end on each question (tool calls tallied):")
    for q in QUESTIONS:
        print(f"\n   > {q}")
        agent._session_id = ""  # force a fresh ADK session per question
        agent._build()  # reload runner
        # Re-ingest the memories into the new instance, bake.
        for raw in MEMORIES:
            agent._memory.enqueue(raw)  # type: ignore[attr-defined]
        agent._memory.force_dream(block=True)  # type: ignore[attr-defined]

        t0 = time.perf_counter()
        answer = agent.answer(q)
        wall = time.perf_counter() - t0
        tool_calls = getattr(agent, "last_tool_calls", {})
        tc_str = (
            ", ".join(f"{k}×{v}" for k, v in sorted(tool_calls.items())) if tool_calls else "none"
        )
        print(f"     answer: {answer}")
        print(f"     tool calls: [{tc_str}]   full-turn wall: {wall:.1f} s")


if __name__ == "__main__":
    main()
