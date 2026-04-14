"""Anthropic (Claude) implementation of ``neuromem.providers.LLMProvider``.

Uses the official ``anthropic`` Python SDK's ``messages.create``
endpoint. Claude models don't have a native JSON mode like OpenAI,
but they honour explicit "return only JSON" instructions very
reliably — we rely on that + a defensive markdown-fence strip.

Every method the cognitive loop consumes is implemented:
- generate_summary + generate_summary_batch (ADR-004)
- extract_tags + extract_tags_batch
- extract_named_entities + extract_named_entities_batch
- generate_category_name + generate_category_names_batch (with
  ADR-003 F3 generic-noun blocklist)
- generate_junction_summary + generate_junction_summaries_batch
"""

from __future__ import annotations

import json
import re
from typing import Any

from anthropic import Anthropic
from neuromem.providers import LLMProvider

from neuromem_anthropic.prompts import load_prompt

# Generic single-word nouns rejected by centroid naming (ADR-003 F3).
_GENERIC_LABEL_BLOCKLIST: frozenset[str] = frozenset(
    {
        "thing",
        "things",
        "aspect",
        "aspects",
        "factor",
        "factors",
        "element",
        "elements",
        "item",
        "items",
        "entity",
        "entities",
        "topic",
        "topics",
        "concept",
        "concepts",
        "category",
        "categories",
        "stuff",
        "misc",
        "other",
        "general",
        "various",
    }
)

_FENCE_RE = re.compile(r"^\s*```(?:json)?\n?(.*?)\n?```\s*$", re.DOTALL)


def _strip_markdown_fence(text: str) -> str:
    match = _FENCE_RE.match(text)
    return match.group(1) if match else text


def _sanitise_snippet(text: str) -> str:
    return text.replace("\n#", "\n ")


# Per-method max_tokens budgets. Anthropic REQUIRES max_tokens on
# every messages.create call; these defaults are sized to comfortably
# fit the expected outputs without truncating Claude mid-answer.
_MAX_TOKENS_SUMMARY = 400
_MAX_TOKENS_TAGS = 400
_MAX_TOKENS_NER = 400
_MAX_TOKENS_CATEGORY_NAME = 20
_MAX_TOKENS_JUNCTION_SUMMARY = 600
# Batched variants need wider budgets — roughly chunk_size × per-item
# budget.
_MAX_TOKENS_BATCH_SUMMARY = 4096
_MAX_TOKENS_BATCH_TAGS = 2048
_MAX_TOKENS_BATCH_NER = 2048
_MAX_TOKENS_BATCH_CATEGORY_NAMES = 600
_MAX_TOKENS_BATCH_JUNCTION = 6144


class AnthropicLLMProvider(LLMProvider):
    """Claude-backed ``LLMProvider``.

    Defaults to ``claude-sonnet-4-6`` — good quality at moderate cost.
    Other sensible model IDs: ``claude-opus-4-6`` (top quality),
    ``claude-haiku-4-5`` (fast + cheap).

    Thread-safe: the anthropic SDK is a synchronous HTTP client that
    can be called from multiple threads.
    """

    # Chunk sizes mirror the gemini / openai providers.
    _BATCH_CHUNK_SIZE = 30
    _JUNCTION_BATCH_CHUNK_SIZE = 8
    _MEMORY_SUMMARY_BATCH_CHUNK_SIZE = 6
    _BATCH_WORKERS = 10

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-6",
        *,
        base_url: str | None = None,
        request_timeout_s: float = 60.0,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must be non-empty")
        if not model:
            raise ValueError("model must be non-empty")
        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": request_timeout_s,
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = Anthropic(**client_kwargs)
        self._model = model

    # ------------------------------------------------------------------
    # Internal: one messages.create call
    # ------------------------------------------------------------------

    def _chat(self, prompt: str, *, max_tokens: int) -> str:
        """Send one user-turn message, return the text content.

        Anthropic REQUIRES ``max_tokens`` per call (unlike OpenAI
        where it's optional). The caller picks a budget per method.
        """
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        # resp.content is a list of content blocks; text is in
        # .text on each. Concatenate (in practice there's one block).
        parts: list[str] = []
        for block in resp.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "".join(parts).strip()

    # ------------------------------------------------------------------
    # Name helpers (ADR-003 F3)
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_name(raw: str) -> str:
        stripped = (raw or "").strip()
        if not stripped:
            return ""
        return stripped.split()[0].strip(".,!?:;\"'").lower()

    def _is_blocked(self, name: str) -> bool:
        return bool(name) and name in _GENERIC_LABEL_BLOCKLIST

    def _first_concept_fallback(self, concepts: list[str]) -> str:
        for c in concepts:
            if c and c.strip():
                return self._clean_name(c)
        return "concept"

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def generate_summary(self, raw_text: str) -> str:
        prompt = load_prompt("generate_summary").format(raw_text=raw_text)
        return self._chat(prompt, max_tokens=_MAX_TOKENS_SUMMARY)

    def generate_summary_batch(self, raw_texts: list[str]) -> list[str]:
        if not raw_texts:
            return []
        if len(raw_texts) == 1:
            return [self.generate_summary(raw_texts[0])]
        if len(raw_texts) <= self._MEMORY_SUMMARY_BATCH_CHUNK_SIZE:
            return self._generate_summary_one_chunk(raw_texts)

        import concurrent.futures  # noqa: PLC0415

        chunks = [
            raw_texts[i : i + self._MEMORY_SUMMARY_BATCH_CHUNK_SIZE]
            for i in range(0, len(raw_texts), self._MEMORY_SUMMARY_BATCH_CHUNK_SIZE)
        ]
        workers = min(self._BATCH_WORKERS, len(chunks))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            chunk_results = list(pool.map(self._generate_summary_one_chunk, chunks))
        out: list[str] = []
        for r in chunk_results:
            out.extend(r)
        return out

    def _generate_summary_one_chunk(self, raw_texts: list[str]) -> list[str]:
        if not raw_texts:
            return []
        if len(raw_texts) == 1:
            return [self.generate_summary(raw_texts[0])]
        numbered_blocks = [f"[{i + 1}]\n{_sanitise_snippet(t)}" for i, t in enumerate(raw_texts)]
        prompt = load_prompt("generate_summary_batch").format(
            n=len(raw_texts), numbered="\n\n".join(numbered_blocks)
        )
        raw = self._chat(prompt, max_tokens=_MAX_TOKENS_BATCH_SUMMARY)
        return self._parse_json_array(raw, raw_texts, self.generate_summary)

    # ------------------------------------------------------------------
    # Tag extraction
    # ------------------------------------------------------------------

    def extract_tags(self, summary: str) -> list[str]:
        prompt = load_prompt("extract_tags").format(summary=summary)
        raw = self._chat(prompt, max_tokens=_MAX_TOKENS_TAGS)
        tags = [t.strip().strip("\"'").strip() for t in raw.split(",")]
        return [t for t in tags if t][:12]

    def extract_tags_batch(self, summaries: list[str]) -> list[list[str]]:
        if not summaries:
            return []
        if len(summaries) == 1:
            return [self.extract_tags(summaries[0])]
        numbered = "\n".join(f"[{i + 1}] {s}" for i, s in enumerate(summaries))
        prompt = load_prompt("extract_tags_batch").format(n=len(summaries), numbered=numbered)
        raw = self._chat(prompt, max_tokens=_MAX_TOKENS_BATCH_TAGS)
        try:
            parsed = json.loads(_strip_markdown_fence(raw))
        except json.JSONDecodeError:
            return [self.extract_tags(s) for s in summaries]
        if not isinstance(parsed, list) or len(parsed) != len(summaries):
            return [self.extract_tags(s) for s in summaries]
        out: list[list[str]] = []
        for i, tags in enumerate(parsed):
            if not isinstance(tags, list):
                out.append(self.extract_tags(summaries[i]))
                continue
            out.append([str(t).strip().strip("\"'").strip() for t in tags if str(t).strip()][:12])
        return out

    # ------------------------------------------------------------------
    # Named-entity extraction
    # ------------------------------------------------------------------

    def extract_named_entities(self, summary: str) -> list[str]:
        prompt = load_prompt("extract_named_entities").format(summary=summary)
        raw = self._chat(prompt, max_tokens=_MAX_TOKENS_NER)
        if not raw or raw.upper().strip(".,!?\"'") == "NONE":
            return []
        entities = [e.strip().strip("\"'").strip() for e in raw.split(",")]
        return [e for e in entities if e][:8]

    def extract_named_entities_batch(self, summaries: list[str]) -> list[list[str]]:
        if not summaries:
            return []
        if len(summaries) == 1:
            return [self.extract_named_entities(summaries[0])]
        numbered = "\n".join(f"[{i + 1}] {s}" for i, s in enumerate(summaries))
        prompt = load_prompt("extract_named_entities_batch").format(
            n=len(summaries), numbered=numbered
        )
        raw = self._chat(prompt, max_tokens=_MAX_TOKENS_BATCH_NER)
        try:
            parsed = json.loads(_strip_markdown_fence(raw))
        except json.JSONDecodeError:
            return [self.extract_named_entities(s) for s in summaries]
        if not isinstance(parsed, list) or len(parsed) != len(summaries):
            return [self.extract_named_entities(s) for s in summaries]
        out: list[list[str]] = []
        for i, ents in enumerate(parsed):
            if not isinstance(ents, list):
                out.append(self.extract_named_entities(summaries[i]))
                continue
            out.append([str(e).strip() for e in ents if str(e).strip()][:8])
        return out

    # ------------------------------------------------------------------
    # Category naming (ADR-002 + ADR-003 F3)
    # ------------------------------------------------------------------

    def generate_category_name(self, concepts: list[str]) -> str:
        if not concepts:
            raise ValueError("concepts must be non-empty")
        prompt = load_prompt("generate_category_name").format(concepts=", ".join(concepts))
        cleaned = self._clean_name(self._chat(prompt, max_tokens=_MAX_TOKENS_CATEGORY_NAME))
        if not cleaned or self._is_blocked(cleaned):
            return self._first_concept_fallback(concepts)
        return cleaned

    def generate_category_names_batch(self, pairs: list[list[str]]) -> list[str]:
        if not pairs:
            return []
        if len(pairs) == 1:
            return [self.generate_category_name(pairs[0])]
        if len(pairs) <= self._BATCH_CHUNK_SIZE:
            return self._generate_category_names_one_chunk(pairs)

        import concurrent.futures  # noqa: PLC0415

        chunks = [
            pairs[i : i + self._BATCH_CHUNK_SIZE]
            for i in range(0, len(pairs), self._BATCH_CHUNK_SIZE)
        ]
        workers = min(self._BATCH_WORKERS, len(chunks))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            chunk_results = list(pool.map(self._generate_category_names_one_chunk, chunks))
        out: list[str] = []
        for r in chunk_results:
            out.extend(r)
        return out

    def _generate_category_names_one_chunk(self, pairs: list[list[str]]) -> list[str]:
        if not pairs:
            return []
        if len(pairs) == 1:
            return [self.generate_category_name(pairs[0])]
        numbered = "\n".join(f"[{i + 1}] {', '.join(pair)}" for i, pair in enumerate(pairs))
        prompt = load_prompt("generate_category_names_batch").format(
            n=len(pairs), numbered=numbered
        )
        raw = self._chat(prompt, max_tokens=_MAX_TOKENS_BATCH_CATEGORY_NAMES)
        try:
            parsed = json.loads(_strip_markdown_fence(raw))
        except json.JSONDecodeError:
            return [self.generate_category_name(p) for p in pairs]
        if not isinstance(parsed, list) or len(parsed) != len(pairs):
            return [self.generate_category_name(p) for p in pairs]
        out: list[str] = []
        for i, name in enumerate(parsed):
            if not isinstance(name, str):
                out.append(self.generate_category_name(pairs[i]))
                continue
            cleaned = self._clean_name(name)
            if not cleaned or self._is_blocked(cleaned):
                out.append(self.generate_category_name(pairs[i]))
                continue
            out.append(cleaned)
        return out

    # ------------------------------------------------------------------
    # Junction paragraph summaries (ADR-003 D2)
    # ------------------------------------------------------------------

    def generate_junction_summary(self, children_summaries: list[str]) -> str:
        if not children_summaries:
            return ""
        joined = "\n- ".join(_sanitise_snippet(s.strip()) for s in children_summaries if s)
        prompt = load_prompt("generate_junction_summary").format(joined=joined)
        return self._chat(prompt, max_tokens=_MAX_TOKENS_JUNCTION_SUMMARY)

    def generate_junction_summaries_batch(self, groups: list[list[str]]) -> list[str]:
        if not groups:
            return []
        if len(groups) == 1:
            return [self.generate_junction_summary(groups[0])]
        if len(groups) <= self._JUNCTION_BATCH_CHUNK_SIZE:
            return self._generate_junction_summaries_one_chunk(groups)

        import concurrent.futures  # noqa: PLC0415

        chunks = [
            groups[i : i + self._JUNCTION_BATCH_CHUNK_SIZE]
            for i in range(0, len(groups), self._JUNCTION_BATCH_CHUNK_SIZE)
        ]
        workers = min(self._BATCH_WORKERS, len(chunks))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            chunk_results = list(pool.map(self._generate_junction_summaries_one_chunk, chunks))
        out: list[str] = []
        for r in chunk_results:
            out.extend(r)
        return out

    def _generate_junction_summaries_one_chunk(self, groups: list[list[str]]) -> list[str]:
        if not groups:
            return []
        if len(groups) == 1:
            return [self.generate_junction_summary(groups[0])]
        numbered_blocks: list[str] = []
        for i, group in enumerate(groups):
            bullets = "\n  - ".join(_sanitise_snippet(s.strip()) for s in group if s)
            numbered_blocks.append(f"[{i + 1}]\n  - {bullets}")
        numbered = "\n\n".join(numbered_blocks)
        prompt = load_prompt("generate_junction_summaries_batch").format(
            n=len(groups), numbered=numbered
        )
        raw = self._chat(prompt, max_tokens=_MAX_TOKENS_BATCH_JUNCTION)
        return self._parse_json_array(raw, groups, self.generate_junction_summary)

    # ------------------------------------------------------------------
    # Shared JSON-parse helper for batched summary paths
    # ------------------------------------------------------------------

    def _parse_json_array(
        self,
        raw: str,
        inputs: list,
        per_item_fallback: Any,
    ) -> list[str]:
        """Parse a bare JSON array of strings from ``raw``; fall back
        per-item on parse error or length mismatch. Claude's bare-array
        output is easier than OpenAI's (which requires an object
        wrapper due to JSON mode)."""
        try:
            parsed = json.loads(_strip_markdown_fence(raw))
        except json.JSONDecodeError:
            return [per_item_fallback(x) for x in inputs]
        if not isinstance(parsed, list) or len(parsed) != len(inputs):
            return [per_item_fallback(x) for x in inputs]
        out: list[str] = []
        for i, v in enumerate(parsed):
            if not isinstance(v, str) or not v.strip():
                out.append(per_item_fallback(inputs[i]))
                continue
            out.append(v.strip())
        return out
