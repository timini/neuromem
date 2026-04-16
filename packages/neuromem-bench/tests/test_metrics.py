"""Metrics unit tests — focus on the int-gold coercion path.

LongMemEval ships some gold answers as bare ints (counting
questions like "how many tanks?" → 3). Before this guard landed,
``_normalise`` crashed with ``AttributeError: 'int' object has no
attribute 'lower'`` on those rows, killing the instance with an
unhelpful stack trace instead of scoring cleanly.
"""

from __future__ import annotations

from neuromem_bench.metrics import contains_match, exact_match


class TestNormaliseIntGold:
    def test_int_gold_does_not_crash_contains_match(self) -> None:
        # Prediction "the answer is 3" SHOULD contain the normalised gold "3".
        assert contains_match("The answer is 3.", 3) == 1.0

    def test_int_gold_not_in_prediction_scores_zero(self) -> None:
        assert contains_match("The answer is 5.", 3) == 0.0

    def test_none_gold_scores_zero_not_str_none_match(self) -> None:
        # Without the None guard, ``_normalise(None)`` would coerce to
        # "none" and spuriously match any prediction containing "none".
        # With the guard it returns "", and contains_match short-
        # circuits on empty gold.
        assert contains_match("None of the above.", None) == 0.0

    def test_float_gold_handled(self) -> None:
        # Defensive: floats shouldn't crash either.
        assert contains_match("The value is 3.14 exactly.", 3.14) == 1.0

    def test_int_gold_exact_match(self) -> None:
        # exact_match goes through the same _normalise — also shouldn't crash.
        assert exact_match("3", 3) == 1.0
        assert exact_match("three", 3) == 0.0
