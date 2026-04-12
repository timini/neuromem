"""LongMemEval dataset loader (ICLR 2025, arXiv 2410.10813).

Canonical source:
- Paper:     https://arxiv.org/abs/2410.10813
- Repo:      https://github.com/xiaowu0162/LongMemEval
- Dataset:   https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned

LongMemEval evaluates long-term memory in chat assistants across
five competency dimensions: information extraction, multi-session
reasoning, temporal reasoning, knowledge updates, and abstention.
Each instance is a multi-session chat history (40 sessions for
the ``s`` split, ~500 for ``m``) followed by a question whose
answer depends on facts the assistant must have retained across
sessions.

Scoring: LongMemEval uses GPT-4o-as-judge as the canonical metric.
v0.1 of this loader ships the raw instances; the metrics module
offers both exact-match (cheap, fast, rough signal) and a future
LLM-judge path that matches the upstream protocol.

File format: each split is a single JSON file (not JSONL) —
a list of instance dicts with the schema documented in the
repo's README. Per-instance shape (as of early 2026):

    {
      "question_id": "...",
      "question_type": "single-session-user" | "multi-session" |
                       "temporal-reasoning" | "knowledge-update" |
                       "abstention",
      "question": "...",
      "answer": "...",
      "question_date": "...",
      "haystack_session_ids": ["...", ...],
      "haystack_dates": ["...", ...],
      "haystack_sessions": [
          [  # one session = list of turns
              {"role": "user", "content": "...", "has_answer": bool},
              {"role": "assistant", "content": "..."},
              ...
          ],
          ...
      ],
      "answer_session_ids": ["...", ...]  # evidence sessions
    }

Download strategy: stdlib ``urllib`` with a local disk cache under
``~/.cache/neuromem-bench/longmemeval/`` so repeat runs don't
re-fetch. The files are a few hundred MB so we don't want to
download them on every CI invocation.
"""

from __future__ import annotations

import json
import os
import shutil
import urllib.request
from collections.abc import Iterator
from pathlib import Path

from neuromem_bench.datasets.base import (
    BenchInstance,
    BenchSession,
    BenchTurn,
    Dataset,
)

# Canonical HuggingFace URLs for the three LongMemEval splits. We
# hit the raw download endpoint on the HF dataset, which serves
# the files without authentication for public datasets.
_HF_BASE = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main"

_SPLIT_FILES: dict[str, str] = {
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
    "oracle": "longmemeval_oracle.json",
}


def _default_cache_dir() -> Path:
    """Return the local cache directory for downloaded benchmark data.

    Honors ``NEUROMEM_BENCH_CACHE`` if set; otherwise uses
    ``~/.cache/neuromem-bench/``. Creates the directory lazily.
    """
    base = os.environ.get("NEUROMEM_BENCH_CACHE", "").strip()
    if base:
        return Path(base).expanduser().resolve()
    return Path.home() / ".cache" / "neuromem-bench"


class LongMemEval(Dataset):
    """LongMemEval loader. Lazily yields :class:`BenchInstance` objects.

    Construct with a split identifier:

        ds = LongMemEval(split="s")       # 40-session variant
        ds = LongMemEval(split="m")       # ~500-session variant
        ds = LongMemEval(split="oracle")  # oracle (answer session only)

    Then iterate:

        for instance in ds.load(limit=10):
            ...

    First call to ``load`` downloads the dataset file to the
    local cache (a few hundred MB for s/m). Subsequent calls read
    the cached copy.
    """

    def __init__(
        self,
        *,
        split: str = "s",
        cache_dir: Path | None = None,
    ) -> None:
        if split not in _SPLIT_FILES:
            raise ValueError(
                f"LongMemEval: unknown split {split!r}. Valid: {sorted(_SPLIT_FILES.keys())}"
            )
        self._split = split
        self._cache_dir = (cache_dir or _default_cache_dir()) / "longmemeval"

    # ------------------------------------------------------------------
    # Dataset ABC implementation
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "longmemeval"

    @property
    def split(self) -> str:
        return self._split

    def load(self, *, limit: int | None = None) -> Iterator[BenchInstance]:
        """Yield benchmark instances from the configured split.

        Downloads and caches the dataset file on first call.
        Yields lazily so large splits don't blow up memory.
        """
        path = self._ensure_downloaded()
        with path.open("r", encoding="utf-8") as fh:
            # LongMemEval ships as a single JSON array, not JSONL.
            # We load it once (~100–500 MB) and then iterate.
            # Streaming JSON parsing would be nicer but the files
            # are small enough that whole-load is fine.
            raw_instances = json.load(fh)

        for count, raw in enumerate(raw_instances):
            if limit is not None and count >= limit:
                return
            yield self._convert_instance(raw)

    # ------------------------------------------------------------------
    # Download + cache
    # ------------------------------------------------------------------

    def _ensure_downloaded(self) -> Path:
        """Return the local path to the dataset file, downloading
        it on first call. Caches under ``_cache_dir``.
        """
        filename = _SPLIT_FILES[self._split]
        url = f"{_HF_BASE}/{filename}"
        target = self._cache_dir / filename

        if target.exists() and target.stat().st_size > 0:
            return target

        self._cache_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"[neuromem-bench] downloading LongMemEval split={self._split} from {url} -> {target}"
        )
        try:
            with urllib.request.urlopen(url) as response, target.open("wb") as out:  # noqa: S310
                shutil.copyfileobj(response, out)
        except Exception:
            # Remove partial file so the next call retries cleanly.
            if target.exists():
                target.unlink()
            raise
        size_mb = target.stat().st_size / (1024 * 1024)
        print(f"[neuromem-bench] downloaded {size_mb:.1f} MB")
        return target

    # ------------------------------------------------------------------
    # Raw → normalised instance conversion
    # ------------------------------------------------------------------

    def _convert_instance(self, raw: dict) -> BenchInstance:
        """Convert one LongMemEval raw instance dict into a
        ``BenchInstance``."""
        haystack_sessions = raw.get("haystack_sessions") or []
        haystack_session_ids = raw.get("haystack_session_ids") or []
        haystack_dates = raw.get("haystack_dates") or []

        sessions: list[BenchSession] = []
        for i, raw_session in enumerate(haystack_sessions):
            session_id = (
                haystack_session_ids[i] if i < len(haystack_session_ids) else f"session-{i}"
            )
            timestamp = haystack_dates[i] if i < len(haystack_dates) else None
            turns = [
                BenchTurn(
                    role=turn.get("role", "user"),
                    text=turn.get("content", ""),
                    metadata={
                        "has_answer": bool(turn.get("has_answer", False)),
                    },
                )
                for turn in raw_session
                if turn.get("content")
            ]
            if turns:
                sessions.append(
                    BenchSession(
                        session_id=session_id,
                        turns=turns,
                        timestamp=timestamp,
                    )
                )

        return BenchInstance(
            instance_id=raw.get("question_id", ""),
            sessions=sessions,
            question=raw.get("question", ""),
            gold_answer=raw.get("answer", ""),
            question_type=raw.get("question_type"),
            metadata={
                "question_date": raw.get("question_date"),
                "answer_session_ids": raw.get("answer_session_ids") or [],
            },
        )
