"""Scoring metrics for neuromem-bench.

v0.1 ships two metrics:

- :func:`exact_match` — normalise both strings (lowercase, strip
  punctuation, collapse whitespace) and return 1.0 on equality,
  0.0 otherwise. Cheap, fast, free. Produces a noisy lower bound
  on the real accuracy — a correct answer phrased differently
  will score 0 — but it's a useful first signal and it's
  deterministic.

- :func:`llm_judge` — call an LLM to score the answer against
  the gold answer semantically. Returns 1.0 if the LLM says
  "correct", 0.0 if "incorrect". Matches LongMemEval's canonical
  GPT-4o-as-judge protocol. Slower and costs tokens, but the
  real accuracy signal.

v0.1 uses exact_match in the runner by default. llm_judge is
wired up but opt-in via a runner flag.

Future metrics (deferred):
- token-level F1 (for partial-credit ranked answers)
- BERTScore (for generation-quality spot checks)
- per-question-type breakdown (LongMemEval's five skill
  categories)
"""

from __future__ import annotations

import re

from neuromem_bench._client import GeminiAnsweringClient

# Stripping pattern: keep letters, digits, spaces, hyphens.
# Collapses everything else (punctuation, quotes, newlines) to a
# single space before whitespace-normalisation.
_NORMALISE_RE = re.compile(r"[^\w\s-]+")
_WHITESPACE_RE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace.

    Used by both metric implementations so the comparison is
    consistent.
    """
    text = text.lower().strip()
    text = _NORMALISE_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def exact_match(prediction: str, gold: str) -> float:
    """Return 1.0 if the normalised prediction equals the
    normalised gold answer, else 0.0.

    Case-insensitive, punctuation-insensitive, whitespace-
    normalised. Still a strict metric — "Paris, France" vs
    "Paris" scores 0.0. For fuzzier semantic equivalence use
    :func:`llm_judge` instead.
    """
    return 1.0 if _normalise(prediction) == _normalise(gold) else 0.0


def contains_match(prediction: str, gold: str) -> float:
    """Return 1.0 if the normalised gold answer appears as a
    substring of the normalised prediction, else 0.0.

    A looser alternative to exact_match — answers "Paris, which
    is the capital of France" will score 1.0 against gold
    "Paris". Still deterministic, still cheap, and catches the
    common "correct but verbose" failure mode of exact_match.

    Use this as the default noisy baseline — it has much better
    signal than exact_match without needing an LLM call.
    """
    norm_pred = _normalise(prediction)
    norm_gold = _normalise(gold)
    if not norm_gold:
        return 0.0
    return 1.0 if norm_gold in norm_pred else 0.0


def llm_judge(
    prediction: str,
    gold: str,
    question: str,
    *,
    api_key: str,
    model: str = "gemini-2.0-flash-001",
) -> float:
    """LLM-as-judge scoring.

    Asks an LLM whether ``prediction`` is a correct answer to
    ``question`` given ``gold`` as the reference. Returns 1.0
    for "correct" and 0.0 for "incorrect".

    Matches the protocol LongMemEval uses (canonically with
    GPT-4o). Here we default to Gemini 2.0 Flash for cost — a
    Flash judge is noisier than GPT-4o but roughly 20x cheaper
    and "good enough" for v0.1 baselining. The judge model is
    configurable.

    Note: this is an LLM call, so it counts against the API
    budget. A 500-instance LongMemEval run scores 500 times,
    which adds up. Use the deterministic metrics for per-PR
    signal and the llm_judge metric for end-of-milestone
    validation.
    """
    client = GeminiAnsweringClient(api_key=api_key, model=model)
    prompt = (
        "You are grading an answer. Given the question, the reference "
        "(correct) answer, and the submitted prediction, respond with "
        "exactly one word: 'correct' if the prediction is a correct "
        "answer to the question (even if phrased differently from the "
        "reference), or 'incorrect' otherwise.\n\n"
        f"Question: {question}\n"
        f"Reference answer: {gold}\n"
        f"Submitted prediction: {prediction}\n\n"
        "Respond with exactly one word: correct or incorrect."
    )
    verdict = (
        client.generate(
            system_instruction=None,
            user_message=prompt,
        )
        .lower()
        .strip()
        .strip(".,!?\"'")
    )
    return 1.0 if verdict.startswith("correct") else 0.0
