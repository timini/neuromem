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
import logging
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
    """Load GOOGLE_API_KEY (or GEMINI_API_KEY) from env or .env.

    Also EXPORTS the discovered key back to ``os.environ`` so child
    libraries that read the env directly (notably the Google ADK
    Runner, which instantiates its own genai client from the env
    rather than accepting a kwarg) find it. Without this, the ADK
    agent path failed immediately on every answer with
    "No API key was provided".
    """
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
            # Export so downstream ADK / genai clients pick it up.
            os.environ.setdefault("GOOGLE_API_KEY", value)
            os.environ.setdefault("GEMINI_API_KEY", value)
            return value
    print(
        "ERROR: no GOOGLE_API_KEY / GEMINI_API_KEY in env or repo-root .env.\n"
        "\n"
        "The benchmark's answer LLM and llm-judge metric currently use Gemini "
        "(via GeminiAnsweringClient), so a Google key is required even when "
        "--llm-provider / --embedder-provider point at OpenAI, Anthropic, or "
        "a local Gemma model (those flags only swap the memory-layer LLM/\n"
        "embedder). Set GOOGLE_API_KEY=... (or GEMINI_API_KEY=...) in your\n"
        "environment or in a repo-root .env file and rerun.",
        file=sys.stderr,
        flush=True,
    )
    sys.exit(1)


def _build_agent(
    name: str,
    api_key: str,
    model: str,
    *,
    llm_provider: str = "gemini",
    embedder_provider: str | None = None,
    llm_api_key: str | None = None,
    embedder_api_key: str | None = None,
    memory_model: str | None = None,
):
    """Construct an agent by name. Imports lazily so baselines that
    don't need the google-adk stack don't pay its import cost.

    Provider-aware agents (neuromem + neuromem-adk) accept kwargs
    for the memory-layer LLM/embedder choice. Baselines (null,
    naive-rag) stay Gemini-only for now.
    """
    if name == "null":
        from neuromem_bench.agent import NullAgent  # noqa: PLC0415

        return NullAgent(api_key=api_key, model=model)
    if name == "naive-rag":
        from neuromem_bench.agent import NaiveRagAgent  # noqa: PLC0415

        return NaiveRagAgent(api_key=api_key, model=model)
    if name == "neuromem":
        from neuromem_bench.agent import NeuromemAgent  # noqa: PLC0415

        return NeuromemAgent(
            api_key=api_key,
            model=model,
            llm_provider=llm_provider,
            embedder_provider=embedder_provider,
            llm_api_key=llm_api_key,
            embedder_api_key=embedder_api_key,
            memory_model=memory_model,
        )
    if name == "neuromem-adk":
        from neuromem_bench.agent import NeuromemAdkAgent  # noqa: PLC0415

        return NeuromemAdkAgent(
            api_key=api_key,
            model=model,
            llm_provider=llm_provider,
            embedder_provider=embedder_provider,
            llm_api_key=llm_api_key,
            embedder_api_key=embedder_api_key,
            memory_model=memory_model,
        )
    raise ValueError(
        f"unknown agent name: {name!r}. Valid: null / naive-rag / neuromem / neuromem-adk"
    )


def _resolve_provider_api_key(
    provider: str,
    default: str | None,
) -> str:
    """Look up the API key for a given provider from common env vars,
    falling back to the default (usually GOOGLE_API_KEY)."""
    env_names = {
        "openai": ("OPENAI_API_KEY",),
        "anthropic": ("ANTHROPIC_API_KEY",),
        "gemini": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        "gemma": (),  # Ollama — no key needed; empty string is fine.
    }.get(provider.lower(), ())
    for env in env_names:
        v = os.environ.get(env, "").strip()
        if v:
            return v
    if default:
        return default
    if provider.lower() == "gemma":
        return ""
    sys.exit(
        f"ERROR: no API key for provider {provider!r}. "
        f"Set {'/'.join(env_names)} or pass --{provider}-api-key."
    )


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


def _configure_logging() -> None:
    """Wire ``neuromem_bench.*`` loggers up to stderr at INFO.

    Used for per-instance observability events (e.g. ADK tool-call
    tallies) without drowning the runner's own timestamped stdout
    stream. Other library loggers (neuromem, google.adk, httpx) stay
    at WARNING so their internals don't spam the transcript.
    """
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    bench_logger = logging.getLogger("neuromem_bench")
    bench_logger.setLevel(logging.INFO)
    bench_logger.addHandler(handler)
    bench_logger.propagate = False


def main() -> None:
    _configure_logging()
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
        help=(
            "Agent backend (repeatable). One of: null, naive-rag, "
            "neuromem, neuromem-adk. Default: neuromem-adk — the "
            "real ADK-backed variant with search_memory / "
            "retrieve_memories / expand_node tools. Pass "
            "--agent neuromem for the one-shot baseline that "
            "injects the tree as a system prompt and cannot use "
            "tools."
        ),
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
            "Model name passed to the answer-LLM and to the memory "
            "layer (default: gemini-flash-latest). For OpenAI/Anthropic "
            "use the native model id (e.g. gpt-4.1-mini, "
            "claude-sonnet-4-6); for Gemma via Ollama use e.g. gemma3."
        ),
    )
    parser.add_argument(
        "--llm-provider",
        choices=("gemini", "openai", "anthropic", "gemma"),
        default="gemini",
        help=(
            "Memory-layer LLM provider (default: gemini). Selects "
            "which SDK drives generate_summary / extract_tags / "
            "generate_category_names / generate_junction_summaries. "
            "Does NOT change which model the ADK answer-LLM uses — "
            "that's --model."
        ),
    )
    parser.add_argument(
        "--embedder-provider",
        choices=("gemini", "openai", "gemma"),
        default=None,
        help=(
            "Memory-layer embedder provider. Defaults to match "
            "--llm-provider, except anthropic→openai (since Anthropic "
            "has no native embedder)."
        ),
    )
    parser.add_argument(
        "--embedder-model",
        default=None,
        help="Override the embedder model (default: provider-specific).",
    )
    parser.add_argument(
        "--memory-model",
        default=None,
        help=(
            "Override the memory-layer LLM model independently of "
            "--model. Defaults to --model if unset. Useful when the "
            "answer LLM and memory LLM should be different providers "
            "(e.g. Gemma for memory, Gemini for answering)."
        ),
    )
    parser.add_argument(
        "--trace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Capture per-instance diagnostic trace into metadata.trace "
            "on every JSONL row (rendered context tree + per-phase "
            "timings for one-shot agents; tool-call transcript for ADK "
            "agents). On by default so every bench run is "
            "self-diagnosable; pass --no-trace to suppress."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of parallel worker threads (default: 1, serial). "
            "Each worker owns its own agent instance. N=4 on a full "
            "LongMemEval-s run takes ~1.5h vs ~6h serial. The API "
            "rate-limit bucket is shared across workers so quota "
            "stays honest. Preview-tier models may throttle hard; "
            "retries (5x exp backoff) + instance-level error rows "
            "keep the run alive through it."
        ),
    )
    args = parser.parse_args()

    # Default changed to neuromem-adk (ADR-003): the one-shot `neuromem`
    # agent cannot call expand_node / search_memory / retrieve_memories
    # and so doesn't exercise the ontology-tree-v2 work. The ADK agent
    # is slower per instance but measures what the rework actually
    # delivers. Pass `--agent neuromem` explicitly for the one-shot
    # baseline.
    agents = args.agent or ["neuromem-adk"]
    limit = args.sample_size if args.sample_size > 0 else None

    api_key = _resolve_api_key()

    # Resolve provider API keys per provider. LLM provider determines
    # llm_api_key; embedder provider determines embedder_api_key. Both
    # fall back to api_key (Google's) if a specific env var isn't set.
    llm_api_key = _resolve_provider_api_key(args.llm_provider, default=api_key)
    effective_embedder_provider = args.embedder_provider or (
        "openai" if args.llm_provider == "anthropic" else args.llm_provider
    )
    embedder_api_key = _resolve_provider_api_key(effective_embedder_provider, default=api_key)

    # Imports deferred until after API-key check so a missing key
    # doesn't load google-adk for nothing.
    from neuromem_bench import run_benchmark  # noqa: PLC0415
    from neuromem_bench.datasets import LongMemEval  # noqa: PLC0415

    metric_fn, metric_name = _build_metric(args.metric, api_key=api_key)

    repo_root = Path(__file__).resolve().parents[3]

    for agent_name in agents:
        dataset = LongMemEval(split=args.split)

        # Factory pattern: _build_agent is called fresh per worker
        # thread when workers>1, or once (at the top of the serial
        # loop) when workers==1. Either way the agent is fully
        # constructed before the bench enters any hot path.
        def _make_agent(_name: str = agent_name):
            return _build_agent(
                _name,
                api_key=api_key,
                model=args.model,
                llm_provider=args.llm_provider,
                embedder_provider=args.embedder_provider,
                llm_api_key=llm_api_key,
                embedder_api_key=embedder_api_key,
                memory_model=args.memory_model,
            )

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
            f"limit={limit} model={args.model} workers={args.workers}\n"
            f"Output: {output_path}\n"
            f"{'=' * 72}",
            flush=True,
        )

        # Pass agent_name explicitly so run_benchmark doesn't probe
        # the factory to discover it. For NeuromemAdkAgent the probe
        # would spin up an ADK Runner just to read type(agent).__name__.
        agent_class_name = {
            "null": "NullAgent",
            "naive-rag": "NaiveRagAgent",
            "neuromem": "NeuromemAgent",
            "neuromem-adk": "NeuromemAdkAgent",
        }.get(agent_name, agent_name)

        run_benchmark(
            dataset=dataset,
            agent_factory=_make_agent,
            agent_name=agent_class_name,
            metric=metric_fn,
            metric_name=metric_name,
            limit=limit,
            output_jsonl=output_path,
            verbose=True,
            trace=args.trace,
            workers=args.workers,
        )


if __name__ == "__main__":
    main()
