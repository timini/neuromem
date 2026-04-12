"""Reproducer: trace where "Target" is lost on LongMemEval Instance 3.

Instance 3 of LongMemEval_s asks: "Where did I redeem a $5 coupon on
coffee creamer?" — gold answer is "Target".

On the real benchmark run, NeuromemAgent's prediction was:

    "You unexpectedly saved $5 on coffee creamer by redeeming a
     coupon you found."

The cognitive loop captured *that* the user redeemed a coupon on
creamer, but lost *where* (Target). This script isolates the
exact stage where the token "Target" drops out so you can see
it with your own eyes.

The script runs two complementary pipelines on the one session
that actually contains the evidence (session_id=answer_d61669c7,
turns 0–7):

  1. **Inspection pipeline** — for each turn, feed the raw
     conversation text through:
       a. ``generate_summary``     (1–2 sentence compression)
       b. ``extract_tags``         (3–5 comma-separated concepts)
     and print both artefacts so you can see where "Target" / "Cartwheel"
     survive or vanish.

  2. **End-to-end pipeline** — spin up a real ``NeuroMemory`` (SQLite in
     memory + Gemini providers), enqueue the 8 turns, force a dream
     cycle, run ``ContextHelper.build_prompt_context`` with the
     benchmark question, print the rendered context tree, then run
     the final answer call exactly like ``NeuromemAgent.answer``
     does. This is the "faithful reproduction" of the failure.

Both pipelines share one ``GeminiLLMProvider`` / ``GeminiEmbeddingProvider``
pair so you're measuring the real prompts against the real model,
not a mock.

Usage::

    GOOGLE_API_KEY=... uv run python \\
        packages/neuromem-bench/scripts/repro_instance3_target.py

Cost: ~8 generate_summary calls + ~8 extract_tags calls + ~5
generate_category_name calls + ~10 embeddings + 1 final answer call.
Well under $0.01 at Gemini 2.0 Flash rates. Takes ~2–3 minutes
wall-clock.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings(
    "ignore",
    message=r".*\[EXPERIMENTAL\].*is enabled",
    category=UserWarning,
)

INSTANCE_QUESTION_ID_PREFIX = "51a45a95"
QUESTION = "Where did I redeem a $5 coupon on coffee creamer?"
GOLD_ANSWER = "Target"


def _parse_dotenv(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if not path.is_file():
        return result
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        result[key] = value
    return result


def _resolve_api_key() -> str:
    for env_name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    repo_root = Path(__file__).resolve().parents[3]
    env_vars = _parse_dotenv(repo_root / ".env")
    for env_name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        value = env_vars.get(env_name, "").strip()
        if value:
            return value
    sys.exit("ERROR: no GOOGLE_API_KEY / GEMINI_API_KEY in env or .env")


def _load_evidence_session() -> list[dict]:
    """Return the 8-turn session with the coupon evidence.

    Instance 51a45a95's ``answer_session_ids`` is
    ``['answer_d61669c7']`` — the session ID that carries
    every mention of Target + Cartwheel + coffee creamer + $5
    coupon.
    """
    cache_path = (
        Path.home() / ".cache" / "neuromem-bench" / "longmemeval" / "longmemeval_s_cleaned.json"
    )
    if not cache_path.is_file():
        sys.exit(
            f"ERROR: LongMemEval cache not found at {cache_path}. "
            f"Run run_longmemeval.py at least once to populate it."
        )
    data = json.loads(cache_path.read_text(encoding="utf-8"))

    for inst in data:
        if not inst.get("question_id", "").startswith(INSTANCE_QUESTION_ID_PREFIX):
            continue
        answer_sids = set(inst["answer_session_ids"])
        for sid, session in zip(
            inst["haystack_session_ids"], inst["haystack_sessions"], strict=True
        ):
            if sid in answer_sids:
                return session
        sys.exit("ERROR: instance found but answer session missing")

    sys.exit(f"ERROR: no instance matching {INSTANCE_QUESTION_ID_PREFIX}")


def _has_target(text: str) -> str:
    """Render a visible marker next to text showing Target/Cartwheel presence."""
    hits = []
    if "Target" in text or "target" in text.lower().split():
        hits.append("Target")
    if "Cartwheel" in text or "cartwheel" in text.lower():
        hits.append("Cartwheel")
    return f" [{'+'.join(hits)}]" if hits else " [—]"


# ---------------------------------------------------------------------------
# Pipeline 1 — summary + tag inspection per turn
# ---------------------------------------------------------------------------


def inspect_pipeline(session: list[dict], api_key: str) -> None:
    from neuromem_gemini import GeminiLLMProvider  # noqa: PLC0415

    llm = GeminiLLMProvider(api_key=api_key)

    print("\n" + "=" * 78)
    print("PIPELINE 1 — per-turn summary + tag inspection")
    print("=" * 78)
    print(
        "Watch for the [Target] / [Cartwheel] markers. If they vanish between\n"
        "the raw turn and the summary — the summariser lost the entity. If they\n"
        "survive the summary but vanish in the tags — the tagger lost the entity.\n"
    )

    for i, turn in enumerate(session):
        raw = turn["content"]
        role = turn["role"]
        print(f"--- turn {i} [{role}] -------------------------------------------------")
        print(f"RAW  ({len(raw):>5} chars){_has_target(raw)}:")
        preview = raw if len(raw) <= 260 else raw[:260] + "…"
        print(f"     {preview}")

        summary = llm.generate_summary(raw)
        print(f"SUM  ({len(summary):>5} chars){_has_target(summary)}:")
        print(f"     {summary}")

        tags = llm.extract_tags(summary)
        tag_blob = ", ".join(tags)
        has_target = "Target" in tag_blob or "Cartwheel" in tag_blob
        marker = " [Target/Cartwheel in tags]" if has_target else " [Target/Cartwheel NOT in tags]"
        print(f"TAGS {marker}:")
        print(f"     {tags}")
        print()


# ---------------------------------------------------------------------------
# Pipeline 2 — real NeuroMemory end-to-end
# ---------------------------------------------------------------------------


def e2e_pipeline(session: list[dict], api_key: str) -> None:
    from google import genai  # noqa: PLC0415
    from google.genai import types as genai_types  # noqa: PLC0415
    from neuromem import NeuroMemory, SQLiteAdapter  # noqa: PLC0415
    from neuromem.context import ContextHelper  # noqa: PLC0415
    from neuromem_gemini import GeminiEmbeddingProvider, GeminiLLMProvider  # noqa: PLC0415

    print("\n" + "=" * 78)
    print("PIPELINE 2 — end-to-end NeuroMemory (the real failure path)")
    print("=" * 78)
    print(
        "Mirrors NeuromemAgent exactly: enqueue each turn → force_dream →\n"
        "build_prompt_context → final answer. The rendered context tree is\n"
        "what the LLM actually sees.\n"
    )

    memory = NeuroMemory(
        storage=SQLiteAdapter(":memory:"),
        llm=GeminiLLMProvider(api_key=api_key),
        embedder=GeminiEmbeddingProvider(api_key=api_key),
        dream_threshold=9999,  # Never auto-dream; we force once.
        cluster_threshold=0.55,  # Same as NeuromemAgent.
    )

    print(f"[enqueueing {len(session)} turns…]")
    for turn in session:
        memory.enqueue(turn["content"], metadata={"role": turn["role"]})

    print("[force_dream…]")
    memory.force_dream(block=True)

    helper = ContextHelper(memory)
    context_tree = helper.build_prompt_context(QUESTION)

    print("\n--- rendered ContextHelper tree (what the answer LLM sees) ---")
    print(context_tree or "(empty — no relevant memories found)")
    print("--- end tree ---\n")

    target_in_tree = "Target" in (context_tree or "")
    print(f"→ 'Target' present in context tree? {target_in_tree}")

    client = genai.Client(api_key=api_key)
    system = (
        "You are a helpful assistant. The following is a tree of "
        "relevant long-term memories retrieved by a cognitive "
        "memory system. Use them to answer the user's question "
        "directly and specifically.\n\n"
        "Relevant memories:\n"
        f"{context_tree or '(no relevant memories found)'}"
    )
    resp = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=QUESTION,
        config=genai_types.GenerateContentConfig(system_instruction=system),
    )
    answer = (resp.text or "").strip()

    print("\n--- final answer ---")
    print(f"Q: {QUESTION}")
    print(f"Gold: {GOLD_ANSWER}")
    print(f"Pred: {answer}")
    print("--- end ---")

    correct = GOLD_ANSWER.lower() in answer.lower()
    print(f"\n→ Contains 'Target'? {correct}")


def main() -> None:
    api_key = _resolve_api_key()
    session = _load_evidence_session()

    print(f"Question: {QUESTION}")
    print(f"Gold answer: {GOLD_ANSWER}")
    print(f"Evidence session: {len(session)} turns")

    inspect_pipeline(session, api_key)
    e2e_pipeline(session, api_key)


if __name__ == "__main__":
    main()
