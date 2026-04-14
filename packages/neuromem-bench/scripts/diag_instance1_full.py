"""Diagnostic: run Instance 1 EXACTLY like the benchmark.

The bench runner returned score 0.0 with prediction "I do not have
specific information about the degree you graduated with" — but a
sibling diagnostic that mirrors the same flow rendered a tree
containing "Business Administration". This script narrows the gap:

  - Uses NeuromemAgent end-to-end (same code path the benchmark uses).
  - After ingestion+force_dream, scans every memory for the degree
    info → confirms whether the per-session summarisation step
    preserved it THIS run.
  - Captures the rendered context tree the answer LLM will see.
  - Sends the answer prompt and prints the prediction.
  - Reports: did the summary preserve the fact? did the tree expose
    the memory? did the LLM answer correctly?

Run multiple times to see how flaky each stage is.
"""

from __future__ import annotations

import argparse
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

INSTANCE_QUESTION_ID_PREFIX = "e47becba"


def _parse_dotenv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        out[key] = value
    return out


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


def _load_instance1() -> dict:
    cache_path = (
        Path.home() / ".cache" / "neuromem-bench" / "longmemeval" / "longmemeval_s_cleaned.json"
    )
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    for inst in data:
        if inst.get("question_id", "").startswith(INSTANCE_QUESTION_ID_PREFIX):
            return inst
    sys.exit(f"ERROR: no instance matching {INSTANCE_QUESTION_ID_PREFIX}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemini-2.0-flash-001")
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()

    api_key = _resolve_api_key()
    instance = _load_instance1()
    question = instance["question"]
    gold = instance["answer"]

    from neuromem.context import ContextHelper  # noqa: PLC0415
    from neuromem_bench.agent import NeuromemAgent  # noqa: PLC0415

    for run in range(1, args.runs + 1):
        print(f"\n{'#' * 72}\n# RUN {run}/{args.runs} model={args.model}\n{'#' * 72}")
        agent = NeuromemAgent(api_key=api_key, model=args.model)

        for session in instance["haystack_sessions"]:
            turns = [{"role": t["role"], "text": t["content"]} for t in session]
            agent.process_session(turns)

        memory = agent._memory  # type: ignore[attr-defined]
        memory.force_dream(block=True)

        # === Stage A: did per-session summarisation preserve the fact? ===
        all_mems = memory.storage.get_memories_by_status("consolidated")
        preserving_mems = []
        for mem in all_mems:
            summary = (mem.get("summary") or "").lower()
            if "business administration" in summary or (
                "graduat" in summary and "degree" in summary
            ):
                preserving_mems.append(mem)
        print(f"\n[A] Memories with 'Business Administration' in summary: {len(preserving_mems)}")
        for mem in preserving_mems[:3]:
            print(f"  - {mem['id']}: {mem.get('summary')[:280]}")

        # === Stage B: render tree the LLM will see ===
        helper = ContextHelper(memory)
        tree = helper.build_prompt_context(question, top_k=20) or ""
        tree_has_ba = "business administration" in tree.lower()
        tree_has_degree = "degree" in tree.lower()
        print(f"\n[B] Tree contains 'Business Administration'? {tree_has_ba}")
        print(f"    Tree contains 'degree'? {tree_has_degree}")
        if not tree_has_ba and preserving_mems:
            print("    !!! Memory exists but is NOT in the rendered tree (retrieval miss)")
            print(f"    Tree size: {len(tree)} chars")
            print(f"    Tree:\n{tree}\n")
            # Inspect: what tags/entities does the missing memory have?
            mem = preserving_mems[0]
            print(f"    Missing memory id: {mem['id']}")
            sub = memory.storage.get_subgraph([mem["id"]], depth=2)
            print(f"    Its subgraph nodes: {len(sub.get('nodes', []))}")
            for node in sub.get("nodes", []):
                print(
                    f"      node {node['id'][:16]} label={node.get('label')!r} centroid={node.get('is_centroid')}"
                )
            for edge in sub.get("edges", []):
                print(
                    f"      edge {edge.get('source_id', '')[:16]} --{edge.get('relationship')}--> {edge.get('target_id', '')[:16]} w={edge.get('weight')}"
                )
            # Now try direct vector search for the question
            from neuromem_gemini import GeminiEmbeddingProvider  # noqa: PLC0415

            embedder = GeminiEmbeddingProvider(api_key=api_key)
            q_vec = embedder.get_embeddings([question])[0]
            nearest = memory.storage.get_nearest_nodes(q_vec, top_k=10)
            print("    Top-10 nearest nodes to question vector:")
            for n in nearest:
                print(
                    f"      {n.get('id', '')[:16]} label={n.get('label')!r} centroid={n.get('is_centroid')} score={n.get('score', n.get('similarity'))}"
                )

        # === Stage C: actually call the answer LLM ===
        prediction = agent.answer(question)
        match = "business administration" in prediction.lower()
        print(f"\n[C] Prediction: {prediction}")
        print(f"    Match? {match}  (gold: {gold})")


if __name__ == "__main__":
    main()
