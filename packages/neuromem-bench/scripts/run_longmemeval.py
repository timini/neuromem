"""CLI entrypoint: run a LongMemEval split against one or more agents.

Usage:

    GOOGLE_API_KEY=... uv run python \\
        packages/neuromem-bench/scripts/run_longmemeval.py \\
        --split s \\
        --sample-size 5 \\
        --agent neuromem \\
        --output docs/benchmarks/longmemeval-s-sample.jsonl

Options:
    --split          Which LongMemEval split to run (s | m | oracle).
                     Defaults to 's' — the smallest, cheapest to run.
    --sample-size    Max instances to score. Defaults to 5 (smoke test).
                     None/0 = full split.
    --agent          Which agent backend(s) to run against. One or more
                     of: null, naive-rag, neuromem. Pass multiple
                     times to run a comparison. Defaults to 'neuromem'.
                     (Legacy alias ``neuromem-adk`` is still accepted.)
    --output         Path to write JSONL results to. A sibling .md file
                     with the aggregate summary is written alongside.
                     Defaults to docs/benchmarks/longmemeval-<split>-<agent>.jsonl
    --metric         Which metric to score with: contains | exact | llm-judge.
                     Defaults to contains (cheap, deterministic,
                     tolerant of phrasing differences).
    --model          Gemini model for the agents. Defaults to
                     gemini-flash-latest. Avoid preview-tier models
                     for long runs — their rate limits are punishing.

Cost note: a 5-instance smoke run against LongMemEval_s burns
roughly 15–30K tokens through the neuromem agent (each instance
has ~40 sessions × ~4 turns to process, each running a
generate_summary LLM call on the hot path). Estimated cost per
5-instance smoke run: ~$0.02 at Gemini 2.0 Flash rates. A full
500-instance run would be ~$2.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

# Force unbuffered stdout/stderr so progress shows up in real time even
# when piped through ``tee`` or redirected to a file. Python otherwise
# block-buffers stdout when it isn't attached to a TTY, which meant
# a 10+ hour benchmark run produced ZERO visible progress before we
# killed it (see docs/investigations for the PR writeup). Setting
# reconfigure(line_buffering=True) takes effect immediately and is
# cheap; the alternative (PYTHONUNBUFFERED=1 env var) requires the
# caller to remember to set it, which nobody does.
try:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
except (AttributeError, OSError):
    # Python < 3.7 or a stdout replacement that doesn't support
    # reconfigure — fall back silently, the explicit flush=True calls
    # in the runner still cover the critical progress lines.
    pass

# Suppress the usual upstream warnings from google-adk / google-genai
# that the workspace filterwarnings=['error'] config would otherwise
# escalate. (This is a standalone script, not a pytest test, so we
# use warnings.filterwarnings directly.)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings(
    "ignore",
    message=r".*\[EXPERIMENTAL\].*is enabled",
    category=UserWarning,
)


def _parse_dotenv(path: Path) -> dict[str, str]:
    """Minimal stdlib .env parser — same pattern the other sibling
    packages use."""
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
    """Load GOOGLE_API_KEY (or GEMINI_API_KEY) from env or .env."""
    for env_name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    # Walk up from this script to find the repo root.
    repo_root = Path(__file__).resolve().parents[3]
    env_vars = _parse_dotenv(repo_root / ".env")
    for env_name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        value = env_vars.get(env_name, "").strip()
        if value:
            return value
    print(
        "ERROR: no GOOGLE_API_KEY / GEMINI_API_KEY in env or repo-root .env",
        file=sys.stderr,
        flush=True,
    )
    sys.exit(1)


def _build_agent(name: str, api_key: str, model: str):
    """Construct an agent by name. Imports lazily so baselines that
    don't need ADK don't pay the import cost."""
    from neuromem_bench.agent import (  # noqa: PLC0415
        NaiveRagAgent,
        NeuromemAgent,
        NullAgent,
    )

    if name == "null":
        return NullAgent(api_key=api_key, model=model)
    if name == "naive-rag":
        return NaiveRagAgent(api_key=api_key, model=model)
    if name in ("neuromem", "neuromem-adk"):
        # Accept the legacy --agent neuromem-adk name too so any
        # docs or scripts referencing the pre-rename CLI keep working.
        return NeuromemAgent(api_key=api_key, model=model)
    raise ValueError(f"unknown agent name: {name!r}. Valid: null / naive-rag / neuromem")


def _build_metric(name: str, api_key: str):
    """Return a (metric_fn, metric_name) pair for the given name."""
    from neuromem_bench.metrics import (  # noqa: PLC0415
        contains_match,
        exact_match,
        llm_judge,
    )

    if name == "exact":
        return (lambda p, g, q: exact_match(p, g), "exact_match")  # noqa: ARG005
    if name == "contains":
        return (lambda p, g, q: contains_match(p, g), "contains_match")  # noqa: ARG005
    if name == "llm-judge":

        def judge(p: str, g: str, q: str) -> float:
            return llm_judge(p, g, q, api_key=api_key)

        return (judge, "llm_judge")
    raise ValueError(f"unknown metric name: {name!r}. Valid: exact / contains / llm-judge")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LongMemEval against one or more neuromem agents."
    )
    parser.add_argument(
        "--split",
        choices=("s", "m", "oracle"),
        default="s",
        help="LongMemEval split (default: s)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="Number of instances to run (default: 5; 0 = full split)",
    )
    parser.add_argument(
        "--agent",
        action="append",
        default=None,
        help="Agent backend (repeatable). null / naive-rag / neuromem. Default: neuromem. Legacy 'neuromem-adk' still accepted.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path (default: docs/benchmarks/<auto>.jsonl)",
    )
    parser.add_argument(
        "--metric",
        choices=("exact", "contains", "llm-judge"),
        default="contains",
        help="Scoring metric (default: contains)",
    )
    parser.add_argument(
        "--model",
        default="gemini-flash-latest",
        help=(
            "Gemini model name (default: gemini-flash-latest — rolling "
            "alias for the current stable flash tier). Preview models "
            "(e.g. gemini-3.1-flash-lite-preview) have aggressive "
            "rate limits and are NOT recommended for long benchmark "
            "runs — they can stall indefinitely under sustained load."
        ),
    )
    args = parser.parse_args()

    agents = args.agent or ["neuromem"]
    limit = args.sample_size if args.sample_size > 0 else None

    api_key = _resolve_api_key()

    # Imports deferred until after API-key check so a missing key
    # doesn't load google-adk for nothing.
    from neuromem_bench import run_benchmark  # noqa: PLC0415
    from neuromem_bench.datasets import LongMemEval  # noqa: PLC0415

    metric_fn, metric_name = _build_metric(args.metric, api_key=api_key)

    repo_root = Path(__file__).resolve().parents[3]

    for agent_name in agents:
        dataset = LongMemEval(split=args.split)
        agent = _build_agent(agent_name, api_key=api_key, model=args.model)

        if args.output is not None:
            output_path = args.output
        else:
            output_path = (
                repo_root
                / "docs"
                / "benchmarks"
                / f"longmemeval-{args.split}-{agent_name}-n{limit or 'full'}.jsonl"
            )

        print(
            f"\n{'=' * 72}\n"
            f"Running LongMemEval split={args.split} "
            f"agent={agent_name} metric={metric_name} "
            f"limit={limit} model={args.model}\n"
            f"Output: {output_path}\n"
            f"{'=' * 72}",
            flush=True,
        )

        run_benchmark(
            dataset=dataset,
            agent=agent,
            metric=metric_fn,
            metric_name=metric_name,
            limit=limit,
            output_jsonl=output_path,
            verbose=True,
        )


if __name__ == "__main__":
    main()
