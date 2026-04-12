"""Rescore an existing JSONL of benchmark results using LLM-as-judge.

Useful when a run was scored with ``contains_match`` (cheap,
deterministic, strict) and you want the semantic verdict without
re-running the agent. The predictions are already in the JSONL;
we just feed them to the llm_judge metric and write a new JSONL
alongside with updated scores.

Usage:
    GOOGLE_API_KEY=... uv run python \\
        packages/neuromem-bench/scripts/rescore_with_llm_judge.py \\
        docs/benchmarks/longmemeval-s-neuromem-adk-n3.jsonl
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
    sys.exit("ERROR: GOOGLE_API_KEY / GEMINI_API_KEY not set")


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <input.jsonl>")

    input_path = Path(sys.argv[1])
    output_path = input_path.with_suffix(".llm_judge.jsonl")

    api_key = _resolve_api_key()

    from neuromem_bench.metrics import llm_judge  # noqa: PLC0415

    rescored: list[dict] = []
    print(f"[rescore] reading {input_path}")
    with input_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            old_score = row["score"]
            new_score = llm_judge(
                prediction=row["prediction"],
                gold=row["gold_answer"],
                question=row["question"],
                api_key=api_key,
            )
            row["score"] = new_score
            row["metric_name"] = "llm_judge"
            row["old_score_contains_match"] = old_score
            rescored.append(row)
            print(
                f"  [{row['instance_id'][:8]}] "
                f"contains_match={old_score:.2f} → llm_judge={new_score:.2f}  "
                f"Q: {row['question'][:60]!r}"
            )

    with output_path.open("w", encoding="utf-8") as fh:
        for row in rescored:
            fh.write(json.dumps(row) + "\n")

    print(f"[rescore] wrote {output_path}")
    mean = sum(r["score"] for r in rescored) / len(rescored) if rescored else 0.0
    print(f"[rescore] new mean score: {mean:.3f}")


if __name__ == "__main__":
    main()
