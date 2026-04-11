"""Demo: show the full hierarchical memory tree neuromem builds
from a rich multi-topic corpus.

Run manually (not via pytest — this is a visual diagnostic):

    GOOGLE_API_KEY=... uv run python packages/neuromem-adk/scripts/demo_rich_tree.py

What it does:
1. Instantiates a NeuroMemory with real Gemini providers and a
   tuned cluster_threshold (0.55, more aggressive than the 0.82
   default) so agglomerative clustering actually merges related
   tags into centroid nodes.
2. Enqueues 16 memories spanning 4 distinct topic areas
   (Databases / Python async / Machine learning / Cooking).
3. Forces a dream cycle to consolidate everything.
4. Prints:
   - The full list of nodes (marking centroids vs leaves)
   - The parent/child edge structure
   - ContextHelper.build_prompt_context output for three different
     queries, one targeting each of three topic areas.

The goal is to see (a) a multi-level tree with named centroid
roots, (b) leaf tag nodes clustered under meaningful parents, and
(c) the ASCII rendering that agents see when they search memory.

Cost: <$0.01 at Gemini 2.0 Flash rates (~8000 tokens total).
Runtime: ~20–30 seconds wall clock.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path


def _parse_dotenv(path: Path) -> dict[str, str]:
    """Minimal stdlib .env parser (same as neuromem-gemini / -adk conftest)."""
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
    """Load GOOGLE_API_KEY from env or repo-root .env, exit if missing."""
    for env_name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        value = os.environ.get(env_name, "").strip()
        if value:
            return value

    repo_root = Path(__file__).resolve().parents[3]
    env_path = repo_root / ".env"
    env_vars = _parse_dotenv(env_path)
    for env_name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        value = env_vars.get(env_name, "").strip()
        if value:
            return value

    print(
        "ERROR: no GOOGLE_API_KEY / GEMINI_API_KEY in environment or .env.",
        file=sys.stderr,
    )
    sys.exit(1)


# Suppress google-adk experimental-feature warnings + google-genai
# DeprecationWarning from aiohttp ClientSession subclassing. This is
# a standalone script so we use warnings.filterwarnings, not pytest
# markers.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings(
    "ignore",
    message=r".*\[EXPERIMENTAL\].*is enabled",
    category=UserWarning,
)


def main() -> None:  # noqa: C901 — demo script, single-file simplicity
    api_key = _resolve_api_key()

    # Import AFTER setting the warning filters so the google-adk
    # import-time UserWarning is suppressed. (This script doesn't
    # actually use ADK — it goes through neuromem directly — but
    # neuromem-gemini is imported and pulls in google-genai, which
    # is fine.)
    from neuromem import ContextHelper, NeuroMemory, SQLiteAdapter
    from neuromem_gemini import GeminiEmbeddingProvider, GeminiLLMProvider

    # Tuned parameters for a visible hierarchy:
    # - dream_threshold=50 means auto-dream never fires during
    #   enqueue; we force_dream manually at the end so all 16
    #   memories consolidate in one pass (needed for clustering
    #   to see them all).
    # - cluster_threshold=0.55 (vs default 0.82) encourages
    #   agglomerative clustering to merge more aggressively. At
    #   0.82 you effectively get a flat tree of leaf tags; at
    #   0.55 you get named centroid parents.
    memory = NeuroMemory(
        storage=SQLiteAdapter(":memory:"),
        llm=GeminiLLMProvider(api_key=api_key),
        embedder=GeminiEmbeddingProvider(api_key=api_key),
        dream_threshold=50,
        cluster_threshold=0.55,
    )

    # Corpus: 16 memories across 4 topic areas. Each topic has 4
    # thematically-related memories so tag extraction produces
    # overlapping concepts that clustering can merge.
    corpus: dict[str, list[str]] = {
        "Databases": [
            "SQLite in WAL mode allows concurrent reads during writes, reducing locking conflicts.",
            "PostgreSQL B-tree indexes speed up range queries on ordered "
            "columns by avoiding full table scans.",
            "Redis uses in-memory data structures like hashes and sorted "
            "sets for sub-millisecond key-value access.",
            "MongoDB stores documents as BSON and supports aggregation "
            "pipelines for complex server-side transformations.",
        ],
        "Python async": [
            "asyncio event loops schedule coroutines cooperatively "
            "without threads, using a single-threaded scheduler.",
            "The async/await keywords let you write non-blocking code "
            "in sequential style without callback pyramids.",
            "Python's GIL prevents true parallelism for CPU-bound code "
            "across multiple threads in the same process.",
            "Trio is an alternative async framework with structured "
            "concurrency and nursery-based task management.",
        ],
        "Machine learning": [
            "Transformer models use self-attention to weigh token "
            "relationships across sequences without recurrence.",
            "Gradient descent optimizes neural network weights by "
            "iteratively following the negative loss gradient.",
            "Convolutional neural networks share weights spatially for "
            "translation-invariant image recognition features.",
            "Recurrent neural networks process sequences by maintaining "
            "a hidden state that updates at each timestep.",
        ],
        "Cooking": [
            "Sourdough bread uses wild yeast fermented over many hours "
            "for complex sour flavor and chewy crumb.",
            "Searing a steak at high heat creates the Maillard reaction, "
            "producing brown crust and umami flavor compounds.",
            "Pasta dough is a simple mix of flour and eggs, kneaded for "
            "ten minutes until smooth and elastic.",
            "Risotto is made by slowly adding warm broth to arborio rice "
            "while stirring to release starch.",
        ],
    }

    print(f"\n{'=' * 72}")
    print("neuromem-adk rich-tree demo")
    print(f"{'=' * 72}")
    print(f"Corpus: {sum(len(v) for v in corpus.values())} memories across {len(corpus)} topics")
    print(f"cluster_threshold: {memory.cluster_threshold}")
    print(f"dream_threshold:   {memory.dream_threshold} (auto-dream disabled)")
    print()

    # Enqueue everything. This fires generate_summary on the caller
    # thread (that's the hot path). ~16 LLM calls, ~50ms–500ms each.
    print("Enqueuing 16 memories (one generate_summary LLM call each)...")
    for topic, items in corpus.items():
        for item in items:
            memory.enqueue(item, metadata={"topic": topic})
    print(f"  inbox count: {memory.storage.count_memories_by_status('inbox')}")

    # Force consolidation. Runs the entire dream pipeline:
    # - Tag extraction (16 more LLM calls)
    # - Batch embed of unique tags
    # - Agglomerative clustering with the tuned threshold
    # - Per-cluster LLM-named centroids (a few more LLM calls)
    # - has_tag edge wiring
    print("\nRunning force_dream(block=True) — this is the expensive step...")
    memory.force_dream(block=True)
    print(f"  consolidated count: {memory.storage.count_memories_by_status('consolidated')}")

    # ---- The actual tree structure ----
    print(f"\n{'=' * 72}")
    print("Node inventory — centroids vs leaves")
    print(f"{'=' * 72}")
    nodes = memory.storage.get_all_nodes()
    centroids = [n for n in nodes if n["is_centroid"]]
    leaves = [n for n in nodes if not n["is_centroid"]]
    print(f"\n🌳 {len(centroids)} centroid (root/interior) nodes:")
    for node in sorted(centroids, key=lambda n: n["label"]):
        print(f"    {node['label']:30}  (id={node['id'][:12]})")
    print(f"\n🍃 {len(leaves)} leaf (tag) nodes:")
    for node in sorted(leaves, key=lambda n: n["label"]):
        print(f"    {node['label']:30}  (id={node['id'][:12]})")

    # ---- Graph structure — child_of edges ----
    print(f"\n{'=' * 72}")
    print("Graph edges (child_of — the tree backbone)")
    print(f"{'=' * 72}")
    if centroids:
        subgraph = memory.storage.get_subgraph(
            [c["id"] for c in centroids],
            depth=3,
        )
        child_of_edges = [e for e in subgraph["edges"] if e["relationship"] == "child_of"]
        if child_of_edges:
            print(f"\n{len(child_of_edges)} child_of edges:")
            id_to_label = {n["id"]: n["label"] for n in nodes}
            for edge in sorted(
                child_of_edges,
                key=lambda e: id_to_label.get(e["source_id"], ""),
            ):
                parent = id_to_label.get(edge["source_id"], "?")
                child = id_to_label.get(edge["target_id"], "?")
                print(f"    {parent} ──▶ {child}")
        else:
            print("\n(No child_of edges — clustering produced a flat tag set)")
    else:
        print("\n(No centroids — clustering produced only leaf nodes)")

    # ---- Rendered ASCII trees for different queries ----
    helper = ContextHelper(memory)
    for query in [
        "sqlite performance tuning",
        "asynchronous python event loops",
        "making bread at home",
    ]:
        print(f"\n{'=' * 72}")
        print(f"build_prompt_context({query!r})")
        print(f"{'=' * 72}")
        tree = helper.build_prompt_context(query)
        if tree:
            print(tree)
        else:
            print("(empty — no matching concepts for this query)")

    print(f"\n{'=' * 72}")
    print("Demo complete.")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()
