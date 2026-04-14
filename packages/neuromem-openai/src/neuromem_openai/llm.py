"""OpenAI implementation of ``neuromem.providers.LLMProvider``.

Implements every abstract + non-abstract method the cognitive loop
consumes:

- ``generate_summary`` + ``generate_summary_batch`` (ADR-004 step 2)
- ``extract_tags`` + ``extract_tags_batch`` (dream cycle)
- ``extract_named_entities`` + ``extract_named_entities_batch``
- ``generate_category_name`` + ``generate_category_names_batch``
  (ADR-002 lazy centroid naming)
- ``generate_junction_summary`` + ``generate_junction_summaries_batch``
  (ADR-003 D2 junction summaries)

All methods share a tight ``_chat`` helper that calls
``openai.chat.completions.create``; batched methods additionally set
``response_format={"type": "json_object"}`` to get reliable JSON
output.

Also drives any OpenAI-compatible endpoint via ``base_url`` — point
at Ollama / vLLM / Azure / LM Studio / OpenRouter by passing the URL
at construction time.
"""

from __future__ import annotations

import json
import re
from typing import Any

from neuromem.providers import LLMProvider
from openai import OpenAI

from neuromem_openai.prompts import load_prompt

# Generic single-word nouns that centroid-naming must reject (ADR-003
# F3). Mirrors the list in GeminiLLMProvider exactly — a generic noun
# is a generic noun regardless of which LLM emitted it.
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


# Fence-stripping regex: match an optional "```" or "```json" opener,
# capture the body, tolerate a trailing "```" closer with optional
# trailing whitespace. Used as a defensive pass on JSON outputs — the
# OpenAI JSON mode is reliable enough that we rarely need this, but
# some older models and Ollama-hosted models still wrap their output.
_FENCE_RE = re.compile(r"^\s*```(?:json)?\n?(.*?)\n?```\s*$", re.DOTALL)


def _strip_markdown_fence(text: str) -> str:
    match = _FENCE_RE.match(text)
    return match.group(1) if match else text


def _sanitise_snippet(text: str) -> str:
    """Defang ``\\n##`` markers in user-supplied snippet text so memory
    content can't shadow our prompt's section headers. Same helper
    present on GeminiLLMProvider (ADR-003 review finding)."""
    return text.replace("\n#", "\n ")


class OpenAILLMProvider(LLMProvider):
    """OpenAI-backed ``LLMProvider``.

    Defaults to ``gpt-4.1-mini`` — fast, inexpensive, reliable on
    JSON output. Use ``gpt-4.1`` / ``gpt-4o`` for higher quality at
    2-3x cost; use ``o4-mini`` / ``o3`` for reasoning-heavy tasks.

    ``base_url`` drives any OpenAI-compatible endpoint (Ollama at
    ``http://localhost:11434/v1``, vLLM, Azure, OpenRouter, LM
    Studio, etc.).

    The constructor is stateless beyond the SDK client — all work
    happens inside per-method calls. Thread-safe by construction
    (openai SDK is synchronous but reentrant).
    """

    # Same chunk-size tuning as GeminiLLMProvider.
    _BATCH_CHUNK_SIZE = 30
    _JUNCTION_BATCH_CHUNK_SIZE = 8
    _MEMORY_SUMMARY_BATCH_CHUNK_SIZE = 6
    _BATCH_WORKERS = 10

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4.1-mini",
        *,
        base_url: str | None = None,
        request_timeout_s: float = 60.0,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must be non-empty")
        if not model:
            raise ValueError("model must be non-empty")
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=request_timeout_s)
        self._model = model

    # ------------------------------------------------------------------
    # Internal: one chat.completions call
    # ------------------------------------------------------------------

    def _chat(self, prompt: str, *, json_mode: bool = False) -> str:
        """Send one user-turn chat completion, return the text content.

        ``json_mode=True`` adds ``response_format={"type": "json_object"}``
        which forces the model to produce syntactically valid JSON
        (OpenAI models strongly enforce this, reducing the parse-fail
        fallback rate vs Gemini's text-mode JSON).
        """
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._client.chat.completions.create(**kwargs)
        return (resp.choices[0].message.content or "").strip()

    # ------------------------------------------------------------------
    # Generic name helpers (mirrors GeminiLLMProvider)
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
    # Summary — dream-cycle worker (ADR-004 moved off the caller thread)
    # ------------------------------------------------------------------

    def generate_summary(self, raw_text: str) -> str:
        prompt = load_prompt("generate_summary").format(raw_text=raw_text)
        return self._chat(prompt)

    def generate_summary_batch(self, raw_texts: list[str]) -> list[str]:
        """ADR-004 step 2. Chunked + JSON-mode batched summariser."""
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
        for chunk_result in chunk_results:
            out.extend(chunk_result)
        return out

    def _generate_summary_one_chunk(self, raw_texts: list[str]) -> list[str]:
        if not raw_texts:
            return []
        if len(raw_texts) == 1:
            return [self.generate_summary(raw_texts[0])]

        numbered_blocks = [f"[{i + 1}]\n{_sanitise_snippet(t)}" for i, t in enumerate(raw_texts)]
        # OpenAI JSON mode requires the prompt to mention "json". We
        # also ask explicitly for an object with a "summaries" key,
        # since JSON mode returns an object, not a bare array.
        prompt = (
            "Return a JSON object with a single key 'summaries' whose "
            "value is an array of exactly "
            f"{len(raw_texts)} strings, in order.\n\n"
            + load_prompt("generate_summary_batch").format(
                n=len(raw_texts), numbered="\n\n".join(numbered_blocks)
            )
        )
        raw = self._chat(prompt, json_mode=True)
        return self._parse_json_array(raw, "summaries", raw_texts, self.generate_summary)

    # ------------------------------------------------------------------
    # Tag extraction (dream cycle)
    # ------------------------------------------------------------------

    def extract_tags(self, summary: str) -> list[str]:
        prompt = load_prompt("extract_tags").format(summary=summary)
        raw = self._chat(prompt)
        tags = [t.strip().strip("\"'").strip() for t in raw.split(",")]
        return [t for t in tags if t][:12]

    def extract_tags_batch(self, summaries: list[str]) -> list[list[str]]:
        if not summaries:
            return []
        if len(summaries) == 1:
            return [self.extract_tags(summaries[0])]

        numbered = "\n".join(f"[{i + 1}] {s}" for i, s in enumerate(summaries))
        prompt = (
            "Return a JSON object with a single key 'tags' whose value "
            "is an array of arrays of strings (one inner array per "
            f"numbered text, {len(summaries)} in total).\n\n"
            + load_prompt("extract_tags_batch").format(n=len(summaries), numbered=numbered)
        )
        raw = self._chat(prompt, json_mode=True)
        try:
            parsed = json.loads(_strip_markdown_fence(raw))
        except json.JSONDecodeError:
            return [self.extract_tags(s) for s in summaries]
        inner = parsed.get("tags") if isinstance(parsed, dict) else None
        if not isinstance(inner, list) or len(inner) != len(summaries):
            return [self.extract_tags(s) for s in summaries]
        out: list[list[str]] = []
        for i, tags in enumerate(inner):
            if not isinstance(tags, list):
                out.append(self.extract_tags(summaries[i]))
                continue
            out.append([str(t).strip().strip("\"'").strip() for t in tags if str(t).strip()][:12])
        return out

    # ------------------------------------------------------------------
    # Named-entity extraction (dream cycle)
    # ------------------------------------------------------------------

    def extract_named_entities(self, summary: str) -> list[str]:
        prompt = load_prompt("extract_named_entities").format(summary=summary)
        raw = self._chat(prompt).strip()
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
        prompt = (
            "Return a JSON object with a single key 'entities' whose "
            "value is an array of arrays of strings (one inner array "
            f"per numbered text, {len(summaries)} in total). Use an "
            "empty array for texts with no named entities.\n\n"
            + load_prompt("extract_named_entities_batch").format(
                n=len(summaries), numbered=numbered
            )
        )
        raw = self._chat(prompt, json_mode=True)
        try:
            parsed = json.loads(_strip_markdown_fence(raw))
        except json.JSONDecodeError:
            return [self.extract_named_entities(s) for s in summaries]
        inner = parsed.get("entities") if isinstance(parsed, dict) else None
        if not isinstance(inner, list) or len(inner) != len(summaries):
            return [self.extract_named_entities(s) for s in summaries]
        out: list[list[str]] = []
        for i, ents in enumerate(inner):
            if not isinstance(ents, list):
                out.append(self.extract_named_entities(summaries[i]))
                continue
            out.append([str(e).strip() for e in ents if str(e).strip()][:8])
        return out

    # ------------------------------------------------------------------
    # Category naming (ADR-002 lazy naming)
    # ------------------------------------------------------------------

    def generate_category_name(self, concepts: list[str]) -> str:
        if not concepts:
            raise ValueError("concepts must be non-empty")
        prompt = load_prompt("generate_category_name").format(concepts=", ".join(concepts))
        cleaned = self._clean_name(self._chat(prompt))
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
        for chunk_result in chunk_results:
            out.extend(chunk_result)
        return out

    def _generate_category_names_one_chunk(self, pairs: list[list[str]]) -> list[str]:
        if not pairs:
            return []
        if len(pairs) == 1:
            return [self.generate_category_name(pairs[0])]
        numbered = "\n".join(f"[{i + 1}] {', '.join(pair)}" for i, pair in enumerate(pairs))
        prompt = (
            "Return a JSON object with a single key 'names' whose value "
            f"is an array of exactly {len(pairs)} strings.\n\n"
            + load_prompt("generate_category_names_batch").format(n=len(pairs), numbered=numbered)
        )
        raw = self._chat(prompt, json_mode=True)
        try:
            parsed = json.loads(_strip_markdown_fence(raw))
        except json.JSONDecodeError:
            return [self.generate_category_name(p) for p in pairs]
        names = parsed.get("names") if isinstance(parsed, dict) else None
        if not isinstance(names, list) or len(names) != len(pairs):
            return [self.generate_category_name(p) for p in pairs]
        out: list[str] = []
        for i, name in enumerate(names):
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
        return self._chat(prompt)

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
        for chunk_result in chunk_results:
            out.extend(chunk_result)
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
        prompt = (
            "Return a JSON object with a single key 'summaries' whose "
            f"value is an array of exactly {len(groups)} strings.\n\n"
            + load_prompt("generate_junction_summaries_batch").format(
                n=len(groups), numbered=numbered
            )
        )
        raw = self._chat(prompt, json_mode=True)
        return self._parse_json_array(
            raw, "summaries", groups, lambda g: self.generate_junction_summary(g)
        )

    # ------------------------------------------------------------------
    # Shared JSON-parse helper for batched-summary paths
    # ------------------------------------------------------------------

    def _parse_json_array(
        self,
        raw: str,
        key: str,
        inputs: list,
        per_item_fallback: Any,
    ) -> list[str]:
        """Parse {"<key>": [...]} from ``raw``; fall back per-item on
        parse error or length mismatch. ``per_item_fallback`` is called
        with one input at a time and must return a string."""
        try:
            parsed = json.loads(_strip_markdown_fence(raw))
        except json.JSONDecodeError:
            return [per_item_fallback(x) for x in inputs]
        values = parsed.get(key) if isinstance(parsed, dict) else None
        if not isinstance(values, list) or len(values) != len(inputs):
            return [per_item_fallback(x) for x in inputs]
        out: list[str] = []
        for i, v in enumerate(values):
            if not isinstance(v, str) or not v.strip():
                out.append(per_item_fallback(inputs[i]))
                continue
            out.append(v.strip())
        return out
