"""Benchmark runner — orchestrates dataset → agent → metrics.

The core loop:

    for instance in dataset.load(limit=N):
        agent.reset()
        for session in instance.sessions:
            agent.process_session([
                {"role": t.role, "text": t.text} for t in session.turns
            ])
        prediction = agent.answer(instance.question)
        score = metric(prediction, instance.gold_answer)
        record_result(...)

Results are written one JSONL line per instance to a caller-
supplied output path. A markdown summary is written to a sibling
``.md`` file with aggregate statistics.

No async, no parallel runs in v0.1. Agents vary in how much
state they carry between process_session and answer, so sequential
is the simplest correct thing. Future versions can parallelise
at the instance level.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from neuromem_bench.agent import BaseAgent
from neuromem_bench.datasets.base import BenchInstance, Dataset
from neuromem_bench.metrics import contains_match

# Type alias for a metric function: takes prediction + gold + question,
# returns a float score in [0.0, 1.0]. Not every metric uses ``question``
# (exact_match doesn't) but we pass it so llm_judge works transparently.
MetricFn = Callable[[str, str, str], float]

# Type alias for a factory that builds a fresh agent on demand. Used
# by the parallel path so each worker thread can own its own agent
# (agents are not safe to share across threads — each holds its own
# SQLite :memory: DB, consolidation lock, and ADK Runner).
AgentFactory = Callable[[], BaseAgent]


def _log(verbose: bool, message: str) -> None:
    """Emit a progress line with UTC timestamp and immediate flush.

    ``flush=True`` is critical: when the runner is invoked under
    ``tee`` or a similar pipe, Python's default block-buffered stdout
    swallows prints until the buffer fills (typically 8 KB) or the
    process exits. For a benchmark where instances take 20+ minutes,
    that means zero visible signal for hours — which is exactly the
    hang-detection regression that motivated this helper.

    The UTC timestamp lets a log-tail user see the cadence of progress
    markers and spot stalls directly (if two markers are 10 min apart
    on something that should take 10 s, something's wrong).
    """
    if not verbose:
        return
    stamp = time.strftime("%H:%M:%S", time.gmtime())
    print(f"[{stamp}Z] {message}", flush=True, file=sys.stdout)


@dataclass
class InstanceResult:
    """One instance's benchmark result.

    Serialised as one JSONL line in the results file.
    """

    instance_id: str
    question: str
    question_type: str | None
    gold_answer: str
    prediction: str
    score: float
    wall_time_s: float
    agent_name: str
    dataset_name: str
    dataset_split: str
    metric_name: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunSummary:
    """Aggregate statistics across a full run."""

    dataset_name: str
    dataset_split: str
    agent_name: str
    metric_name: str
    instance_count: int
    mean_score: float
    wall_time_s: float
    per_question_type: dict[str, dict[str, float]] = field(default_factory=dict)
    """Per-type breakdown. Each entry: ``{"count": int, "mean": float}``."""


def run_benchmark(
    *,
    dataset: Dataset,
    agent: BaseAgent | None = None,
    agent_factory: AgentFactory | None = None,
    metric: MetricFn = lambda p, g, q: contains_match(p, g),  # noqa: ARG005
    metric_name: str = "contains_match",
    limit: int | None = None,
    output_jsonl: Path | None = None,
    verbose: bool = True,
    trace: bool = True,
    workers: int = 1,
) -> RunSummary:
    """Run one benchmark end-to-end and return an aggregate summary.

    Arguments:
        dataset: The :class:`Dataset` to pull instances from.
        agent: The :class:`BaseAgent` to drive. Mutually exclusive
            with ``agent_factory``. Required when ``workers == 1``.
        agent_factory: Zero-arg callable that builds a fresh agent.
            Required when ``workers > 1`` — each worker thread owns
            its own agent (shared state / :memory: SQLite / ADK
            Runner per agent means sharing is not safe).
        metric: Scoring function ``(prediction, gold, question) -> float``.
            Defaults to :func:`contains_match` — cheap and noisy but
            better than exact-match for loose string answers.
        metric_name: Human-readable name for the metric, recorded
            in the results.
        limit: Optional max number of instances to run. Useful for
            smoke tests.
        output_jsonl: Optional path to stream per-instance results
            to, one JSONL line per instance. The parent directory
            is created if it doesn't exist. A ``.md`` sibling file
            with the run summary is written alongside.
        verbose: If True, print per-instance progress to stdout.
        trace: If True (the default), capture each agent's
            ``last_trace`` diagnostic dict into ``metadata["trace"]``
            on the JSONL row. Includes the rendered context tree for
            one-shot agents and the tool-call transcript for ADK
            agents. Opt out with ``--no-trace`` for size-sensitive
            runs; default-on means every bench run is self-diagnosable
            without needing a follow-up re-run to capture context.
        workers: Number of parallel worker threads. ``1`` (default)
            runs serially with the single ``agent``. ``N > 1`` requires
            ``agent_factory``; each worker lazily builds its own agent
            and processes a share of instances. Since instances are
            ~45s of API wait time, threads escape the GIL on I/O and
            deliver near-linear speedup up to whatever the API-key
            rate-limit bucket tolerates.

    Returns:
        A :class:`RunSummary` with aggregate statistics. Also
        written to ``output_jsonl.with_suffix(".md")`` if a path
        was supplied.
    """
    _validate_run_config(agent=agent, agent_factory=agent_factory, workers=workers)

    results: list[InstanceResult] = []
    agent_name = type(agent).__name__ if agent is not None else _factory_agent_name(agent_factory)

    jsonl_handle = None
    if output_jsonl is not None:
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        jsonl_handle = output_jsonl.open("w", encoding="utf-8")
    write_lock = threading.Lock()

    def _emit(result: InstanceResult, instance_number: int) -> None:
        """Thread-safe result sink: append + JSONL write + DONE log."""
        with write_lock:
            results.append(result)
            if jsonl_handle is not None:
                jsonl_handle.write(json.dumps(asdict(result)) + "\n")
                jsonl_handle.flush()
            _log(
                verbose,
                f"[{instance_number}] DONE   score={result.score:.2f}  t={result.wall_time_s:.1f}s",
            )

    start_all = time.perf_counter()
    try:
        if workers == 1:
            serial_agent = agent if agent is not None else agent_factory()  # type: ignore[misc]
            _run_serial(
                dataset=dataset,
                agent=serial_agent,
                metric=metric,
                metric_name=metric_name,
                agent_name=agent_name,
                limit=limit,
                verbose=verbose,
                trace=trace,
                emit=_emit,
            )
        else:
            _run_parallel(
                dataset=dataset,
                agent_factory=agent_factory,  # type: ignore[arg-type]
                metric=metric,
                metric_name=metric_name,
                agent_name=agent_name,
                limit=limit,
                verbose=verbose,
                trace=trace,
                workers=workers,
                emit=_emit,
            )
    finally:
        if jsonl_handle is not None:
            jsonl_handle.close()

    total_time = time.perf_counter() - start_all
    summary = _summarise(
        results=results,
        dataset=dataset,
        agent_name=agent_name,
        metric_name=metric_name,
        total_time=total_time,
    )

    if output_jsonl is not None:
        summary_md = output_jsonl.with_suffix(".md")
        _write_summary_markdown(summary, summary_md)
        _log(verbose, f"JSONL results:  {output_jsonl}")
        _log(verbose, f"markdown summary: {summary_md}")

    _log(
        verbose,
        f"FINAL mean={summary.mean_score:.3f} "
        f"n={summary.instance_count} "
        f"wall_time={summary.wall_time_s:.1f}s",
    )

    return summary


def _run_one_instance(
    *,
    instance: BenchInstance,
    agent: BaseAgent,
    metric: MetricFn,
    metric_name: str,
    agent_name: str,
    dataset_name: str,
    dataset_split: str,
    verbose: bool = True,
    instance_number: int | None = None,
    trace: bool = True,
) -> InstanceResult:
    """Drive one instance through the agent and score the result.

    When ``verbose`` is True, emits progress markers at the natural
    phase boundaries (reset complete, ingestion complete, answer
    complete, scored) so a piped run shows *some* signal of progress
    instead of going silent for 20+ minutes between instance-end
    lines. Each marker includes elapsed seconds since instance start
    so a stalled phase is obvious from the log.
    """
    prefix = f"[{instance_number}]   " if instance_number is not None else "    "

    agent.reset()
    start = time.perf_counter()
    _log(verbose, f"{prefix}- reset done, starting ingestion")

    # Ingest all sessions in order. One ``process_session`` call per
    # session — agents decide how to handle the turns internally.
    # For neuromem agents this means one generate_summary call per
    # session (vs per turn historically), which cuts API pressure
    # 4-5× and preserves turn-level semantics via the transcript
    # in raw_content.
    turn_count = 0
    for session in instance.sessions:
        turns_payload = [{"role": turn.role, "text": turn.text} for turn in session.turns]
        agent.process_session(turns_payload)
        turn_count += len(turns_payload)
    _log(
        verbose,
        f"{prefix}- ingestion done "
        f"({turn_count} turns across {len(instance.sessions)} sessions, "
        f"elapsed={time.perf_counter() - start:.1f}s)",
    )

    prediction = agent.answer(instance.question)
    elapsed = time.perf_counter() - start
    _log(
        verbose,
        f"{prefix}- answer done (elapsed={elapsed:.1f}s), scoring…",
    )
    score = metric(prediction, instance.gold_answer, instance.question)

    # ADR-003 observability: if the agent recorded a per-answer
    # tool-call tally (NeuromemAdkAgent does; the one-shot agents
    # don't), attach it to the result's metadata so the JSONL output
    # preserves the ADK tool-use signal for post-run analysis.
    metadata: dict[str, Any] = {}
    tool_calls = getattr(agent, "last_tool_calls", None)
    if isinstance(tool_calls, dict) and tool_calls:
        metadata["tool_calls"] = dict(tool_calls)

    # Trace capture: pull the agent's last_trace dict into metadata so
    # every JSONL row carries enough diagnostic info (rendered context
    # tree, per-phase timings, tool-call transcript) to split
    # retrieval-miss vs confabulation failures post-run. Mirrors the
    # last_tool_calls pattern above; agents that don't populate
    # last_trace (e.g. NullAgent, NaiveRagAgent) simply get skipped.
    if trace:
        last_trace = getattr(agent, "last_trace", None)
        if isinstance(last_trace, dict) and last_trace:
            metadata["trace"] = dict(last_trace)

    return InstanceResult(
        instance_id=instance.instance_id,
        question=instance.question,
        question_type=instance.question_type,
        gold_answer=instance.gold_answer,
        prediction=prediction,
        score=score,
        wall_time_s=elapsed,
        agent_name=agent_name,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        metric_name=metric_name,
        metadata=metadata,
    )


def _validate_run_config(
    *,
    agent: BaseAgent | None,
    agent_factory: AgentFactory | None,
    workers: int,
) -> None:
    """Guard against unrunnable combinations before any side effects.

    Mirrors the contract the docstring describes: exactly one of
    agent/agent_factory must be provided; workers must be >= 1;
    workers > 1 requires agent_factory because agents hold per-instance
    SQLite + consolidation + ADK state that isn't thread-safe.
    """
    if agent is None and agent_factory is None:
        raise ValueError("run_benchmark requires agent or agent_factory")
    if agent is not None and agent_factory is not None:
        raise ValueError("run_benchmark: pass exactly one of agent or agent_factory")
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    if workers > 1 and agent_factory is None:
        raise ValueError("workers > 1 requires agent_factory (agents are not thread-safe)")


def _factory_agent_name(factory: AgentFactory) -> str:
    """Best-effort agent-class-name lookup for a factory.

    Tries factory.__wrapped__ (functools.partial-like wrappers), then
    falls back to building one agent just to read type(...).__name__.
    Run once at the top of run_benchmark so results[*].agent_name is
    populated correctly even in the parallel path where the
    run-benchmark caller only passes a factory.
    """
    wrapped = getattr(factory, "__wrapped__", None)
    if wrapped is not None:
        name = getattr(wrapped, "__name__", None)
        if isinstance(name, str):
            return name
    try:
        probe = factory()
    except Exception:
        return "UnknownAgent"
    return type(probe).__name__


def _run_serial(
    *,
    dataset: Dataset,
    agent: BaseAgent,
    metric: MetricFn,
    metric_name: str,
    agent_name: str,
    limit: int | None,
    verbose: bool,
    trace: bool,
    emit: Callable[[InstanceResult, int], None],
) -> None:
    """Serial instance dispatch. One agent, one instance at a time,
    same order as the dataset yields them. This is the
    backward-compatible path (``workers == 1`` default).
    """
    for idx, instance in enumerate(dataset.load(limit=limit)):
        _log(
            verbose,
            f"[{idx + 1}] START  instance={instance.instance_id[:8]} "
            f"type={instance.question_type or '?'} "
            f"sessions={len(instance.sessions)} "
            f"Q: {instance.question[:60]!r}",
        )
        result = _safe_run_one_instance(
            instance=instance,
            agent=agent,
            metric=metric,
            metric_name=metric_name,
            agent_name=agent_name,
            dataset_name=dataset.name,
            dataset_split=dataset.split,
            verbose=verbose,
            instance_number=idx + 1,
            trace=trace,
        )
        emit(result, idx + 1)


def _run_parallel(
    *,
    dataset: Dataset,
    agent_factory: AgentFactory,
    metric: MetricFn,
    metric_name: str,
    agent_name: str,
    limit: int | None,
    verbose: bool,
    trace: bool,
    workers: int,
    emit: Callable[[InstanceResult, int], None],
) -> None:
    """Dispatch instances across a ``ThreadPoolExecutor`` of ``workers``
    threads. Each worker lazily builds its own agent via
    ``agent_factory`` on first use (persisted across the rest of that
    worker's instances — matching the serial path's reuse pattern,
    just one-agent-per-thread instead of one-agent-total).

    Each instance is wrapped in :func:`_safe_run_one_instance` so a
    crash in one thread produces an error-shaped result instead of
    tearing down the whole pool.
    """
    thread_local = threading.local()

    def _get_or_build_agent() -> BaseAgent:
        agent = getattr(thread_local, "agent", None)
        if agent is None:
            agent = agent_factory()
            thread_local.agent = agent
        return agent

    def _process(instance: BenchInstance, instance_number: int) -> InstanceResult:
        _log(
            verbose,
            f"[{instance_number}] START  instance={instance.instance_id[:8]} "
            f"type={instance.question_type or '?'} "
            f"sessions={len(instance.sessions)} "
            f"Q: {instance.question[:60]!r}",
        )
        worker_agent = _get_or_build_agent()
        return _safe_run_one_instance(
            instance=instance,
            agent=worker_agent,
            metric=metric,
            metric_name=metric_name,
            agent_name=agent_name,
            dataset_name=dataset.name,
            dataset_split=dataset.split,
            verbose=verbose,
            instance_number=instance_number,
            trace=trace,
        )

    instances = list(dataset.load(limit=limit))
    _log(verbose, f"dispatching {len(instances)} instances across {workers} workers")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_process, inst, idx + 1): idx + 1 for idx, inst in enumerate(instances)
        }
        for fut in as_completed(futures):
            instance_number = futures[fut]
            # _safe_run_one_instance never raises, but .result() can
            # still propagate non-bench errors (e.g. a KeyboardInterrupt
            # from the main thread). Let those bubble; they're meant
            # to abort the run.
            emit(fut.result(), instance_number)


def _safe_run_one_instance(
    *,
    instance: BenchInstance,
    agent: BaseAgent,
    metric: MetricFn,
    metric_name: str,
    agent_name: str,
    dataset_name: str,
    dataset_split: str,
    verbose: bool = True,
    instance_number: int | None = None,
    trace: bool = True,
) -> InstanceResult:
    """Run one instance, but never raise — convert any exception into
    an error-shaped :class:`InstanceResult` so the calling loop (serial
    or parallel) can keep going.

    This is the containment boundary that lets an overnight run survive
    transient flakiness after the provider-level retry budget is already
    exhausted. A single instance going bad produces a 0-score row with
    ``metadata["error"] = repr(exc)`` and the next instance still runs.

    Before this wrapper existed, a single ``httpx.ReadTimeout`` after
    5 retries killed the whole overnight bench at instance 17/200.
    """
    start = time.perf_counter()
    try:
        return _run_one_instance(
            instance=instance,
            agent=agent,
            metric=metric,
            metric_name=metric_name,
            agent_name=agent_name,
            dataset_name=dataset_name,
            dataset_split=dataset_split,
            verbose=verbose,
            instance_number=instance_number,
            trace=trace,
        )
    except Exception as exc:
        wall = time.perf_counter() - start
        prefix = f"[{instance_number}]   " if instance_number is not None else "    "
        _log(
            verbose,
            f"{prefix}!! FAILED after {wall:.1f}s: {type(exc).__name__}: {exc}",
        )
        return InstanceResult(
            instance_id=instance.instance_id,
            question=instance.question,
            question_type=instance.question_type,
            gold_answer=instance.gold_answer,
            prediction="",
            score=0.0,
            wall_time_s=wall,
            agent_name=agent_name,
            dataset_name=dataset_name,
            dataset_split=dataset_split,
            metric_name=metric_name,
            metadata={
                "error": repr(exc),
                "error_type": type(exc).__name__,
            },
        )


def _summarise(
    *,
    results: list[InstanceResult],
    dataset: Dataset,
    agent_name: str,
    metric_name: str,
    total_time: float,
) -> RunSummary:
    """Build a :class:`RunSummary` from per-instance results."""
    n = len(results)
    if n == 0:
        mean = 0.0
    else:
        mean = sum(r.score for r in results) / n

    per_type: dict[str, dict[str, float]] = {}
    for r in results:
        key = r.question_type or "unknown"
        bucket = per_type.setdefault(key, {"count": 0, "sum": 0.0})
        bucket["count"] += 1
        bucket["sum"] += r.score
    # Convert sums to means.
    per_type_final = {
        k: {"count": v["count"], "mean": (v["sum"] / v["count"]) if v["count"] else 0.0}
        for k, v in per_type.items()
    }

    return RunSummary(
        dataset_name=dataset.name,
        dataset_split=dataset.split,
        agent_name=agent_name,
        metric_name=metric_name,
        instance_count=n,
        mean_score=mean,
        wall_time_s=total_time,
        per_question_type=per_type_final,
    )


def _write_summary_markdown(summary: RunSummary, path: Path) -> None:
    """Write a human-readable summary of a run as markdown."""
    lines = [
        f"# Benchmark run: {summary.dataset_name} / {summary.dataset_split}",
        "",
        f"- **Agent**: `{summary.agent_name}`",
        f"- **Metric**: `{summary.metric_name}`",
        f"- **Instances**: {summary.instance_count}",
        f"- **Mean score**: **{summary.mean_score:.3f}**",
        f"- **Wall-clock time**: {summary.wall_time_s:.1f}s",
        "",
    ]
    if summary.per_question_type:
        lines.append("## Per-question-type breakdown")
        lines.append("")
        lines.append("| Question type | Count | Mean score |")
        lines.append("|---|---|---|")
        for q_type in sorted(summary.per_question_type.keys()):
            row = summary.per_question_type[q_type]
            lines.append(f"| `{q_type}` | {int(row['count'])} | {row['mean']:.3f} |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
