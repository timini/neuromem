"""Benchmark runner — orchestrates dataset → agent → metrics.

The core loop:

    for instance in dataset.load(limit=N):
        agent.reset()
        for session in instance.sessions:
            for turn in session.turns:
                agent.process_turn(turn.text, role=turn.role)
        prediction = agent.answer(instance.question)
        score = metric(prediction, instance.gold_answer)
        record_result(...)

Results are written one JSONL line per instance to a caller-
supplied output path. A markdown summary is written to a sibling
``.md`` file with aggregate statistics.

No async, no parallel runs in v0.1. Agents vary in how much
state they carry between process_turn and answer, so sequential
is the simplest correct thing. Future versions can parallelise
at the instance level.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
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
    agent: BaseAgent,
    metric: MetricFn = lambda p, g, q: contains_match(p, g),  # noqa: ARG005
    metric_name: str = "contains_match",
    limit: int | None = None,
    output_jsonl: Path | None = None,
    verbose: bool = True,
) -> RunSummary:
    """Run one benchmark end-to-end and return an aggregate summary.

    Arguments:
        dataset: The :class:`Dataset` to pull instances from.
        agent: The :class:`BaseAgent` to drive.
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

    Returns:
        A :class:`RunSummary` with aggregate statistics. Also
        written to ``output_jsonl.with_suffix(".md")`` if a path
        was supplied.
    """
    results: list[InstanceResult] = []
    agent_name = type(agent).__name__

    jsonl_handle = None
    if output_jsonl is not None:
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        jsonl_handle = output_jsonl.open("w", encoding="utf-8")

    start_all = time.perf_counter()
    try:
        for idx, instance in enumerate(dataset.load(limit=limit)):
            _log(
                verbose,
                f"[{idx + 1}] START  instance={instance.instance_id[:8]} "
                f"type={instance.question_type or '?'} "
                f"sessions={len(instance.sessions)} "
                f"Q: {instance.question[:60]!r}",
            )
            result = _run_one_instance(
                instance=instance,
                agent=agent,
                metric=metric,
                metric_name=metric_name,
                agent_name=agent_name,
                dataset_name=dataset.name,
                dataset_split=dataset.split,
                verbose=verbose,
                instance_number=idx + 1,
            )
            results.append(result)

            if jsonl_handle is not None:
                jsonl_handle.write(json.dumps(asdict(result)) + "\n")
                jsonl_handle.flush()

            _log(
                verbose,
                f"[{idx + 1}] DONE   score={result.score:.2f}  t={result.wall_time_s:.1f}s",
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

    # Ingest all sessions in order.
    turn_count = 0
    for session in instance.sessions:
        for turn in session.turns:
            agent.process_turn(turn.text, role=turn.role)
            turn_count += 1
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
