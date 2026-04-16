"""Unit tests for the token-bucket rate limiter.

These tests use short real-time waits to verify timing behaviour.
Nothing here talks to Gemini — the module is pure Python, so the
fast/deterministic tests here are enough to cover the contract.
The integration between this module and the provider classes is
exercised implicitly by every network test that uses Gemini
providers; a mis-wired bucket would surface there.
"""

from __future__ import annotations

import threading
import time

import pytest
from neuromem_gemini._rate_limit import (
    BucketAcquireTimeout,
    TokenBucket,
    _reset_registry,
    get_bucket,
)


@pytest.fixture(autouse=True)
def reset_registry():
    """Clear the module-level bucket registry before each test so
    tests can't leak buckets into each other via shared state.

    autouse so every test in this file starts with a clean slate.
    """
    _reset_registry()
    yield
    _reset_registry()


# ---------------------------------------------------------------------------
# TokenBucket — direct behaviour
# ---------------------------------------------------------------------------


class TestTokenBucket:
    def test_invalid_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="rate_per_minute"):
            TokenBucket(rate_per_minute=0)
        with pytest.raises(ValueError, match="rate_per_minute"):
            TokenBucket(rate_per_minute=-5)

    def test_rate_property_is_read_only(self) -> None:
        b = TokenBucket(rate_per_minute=60)
        assert b.rate_per_minute == 60
        with pytest.raises(AttributeError):
            b.rate_per_minute = 120  # type: ignore[misc]

    def test_starts_at_capacity(self) -> None:
        """A fresh bucket lets the first N acquires through without
        meaningful wait — that's the burst allowance."""
        bucket = TokenBucket(rate_per_minute=600)  # 10 per second
        start = time.monotonic()
        for _ in range(50):
            bucket.acquire()
        elapsed = time.monotonic() - start
        # All 50 should be near-instant (within tens of ms).
        assert elapsed < 0.2, f"acquire of 50 tokens took {elapsed:.3f}s"

    def test_blocks_once_capacity_exhausted(self) -> None:
        """At 600 rpm (10 tokens/sec), draining 600 tokens and then
        asking for one more should block for ~0.1s (one token's
        refill interval)."""
        bucket = TokenBucket(rate_per_minute=600)
        # Drain the initial capacity.
        for _ in range(600):
            bucket.acquire()
        # Now time the 601st acquire — it should wait for refill.
        start = time.monotonic()
        bucket.acquire()
        elapsed = time.monotonic() - start
        # Refill is 10 per second → each token takes 0.1s. Give a
        # generous ±50% window to stay reliable on loaded CI.
        assert 0.05 < elapsed < 0.25, f"expected ~0.1s wait, got {elapsed:.3f}s"

    def test_sustained_acquires_respect_rate(self) -> None:
        """Over a longer window, the effective throughput converges
        to the configured rate regardless of burst shape."""
        # 1200 rpm = 20 tokens/sec. Measure 40 acquires — should take
        # roughly (40 - capacity) / rate_per_sec seconds once the
        # initial burst is spent. 1200 capacity means no wait for
        # any of 40 acquires, so this is actually just a sanity test.
        bucket = TokenBucket(rate_per_minute=120)  # 2 tokens/sec
        # Drain initial capacity (120 tokens).
        for _ in range(120):
            bucket.acquire()
        # Now 10 more acquires should take ~5 seconds at 2/sec.
        start = time.monotonic()
        for _ in range(10):
            bucket.acquire()
        elapsed = time.monotonic() - start
        # Expect ~5s. Tolerances: the first post-drain acquire might
        # already be waiting, so duration is between 4.5 and 5.5
        # roughly. Keep the window generous for CI noise.
        assert 4.0 < elapsed < 6.5, f"expected ~5s for 10 acquires at 2/sec, got {elapsed:.2f}s"

    def test_acquire_timeout_raises_after_budget_elapsed(self) -> None:
        """Starvation should fail fast, not deadlock. A call to
        ``acquire(timeout_s=0.2)`` on a drained 60-rpm bucket (which
        refills 1 token per second) must raise after ~0.2s, not wait
        the full 1s for the token."""
        bucket = TokenBucket(rate_per_minute=60)  # 1 token/sec refill
        # Drain the burst capacity.
        for _ in range(60):
            bucket.acquire()
        start = time.monotonic()
        with pytest.raises(BucketAcquireTimeout, match="within"):
            bucket.acquire(timeout_s=0.2)
        elapsed = time.monotonic() - start
        # Should raise within ~200ms + a small tolerance for wake-up
        # latency. Much less than the 1s refill interval.
        assert elapsed < 0.6, f"expected timeout within 0.6s, got {elapsed:.3f}s"

    def test_acquire_timeout_none_is_unbounded_wait(self) -> None:
        """Passing ``timeout_s=None`` opts out of the safety cap and
        restores the pre-fix unbounded wait. Verifies we don't raise
        spuriously on a slow-but-legitimate wait."""
        bucket = TokenBucket(rate_per_minute=600)  # 10/sec
        # Drain burst capacity (600 tokens).
        for _ in range(600):
            bucket.acquire()
        # 1 more acquire at 10/sec with timeout_s=None should block
        # for ~0.1s and return cleanly — no timeout exception.
        start = time.monotonic()
        bucket.acquire(timeout_s=None)
        elapsed = time.monotonic() - start
        assert elapsed < 0.5  # well within the default 300s cap too

    def test_refill_does_not_exceed_capacity(self) -> None:
        """A bucket sitting idle can't accumulate more than one
        minute of tokens. Bursty traffic is bounded."""
        bucket = TokenBucket(rate_per_minute=60)
        # Simulate being idle for "a long time" — we can't fast-
        # forward time so we just wait briefly and verify the bucket
        # hasn't exceeded capacity by checking we can do exactly 60
        # quick acquires and then the 61st blocks.
        time.sleep(0.5)  # half a second of "idle time"
        # Should still only have at most ``capacity`` tokens (==60).
        # Drain them fast.
        for _ in range(60):
            bucket.acquire()
        # The next acquire must block (at 60rpm = 1/sec refill).
        start = time.monotonic()
        bucket.acquire()
        elapsed = time.monotonic() - start
        # Should wait close to 1s (one refill interval at 60rpm).
        # Generous window for CI noise.
        assert elapsed > 0.5, f"expected 61st acquire to block, got {elapsed:.3f}s"

    def test_thread_safety_basic(self) -> None:
        """Multiple threads hitting the same bucket must collectively
        respect the rate. Here we drain the bucket from N threads
        concurrently and verify the total wall time is ~= the total
        tokens consumed over the rate.
        """
        bucket = TokenBucket(rate_per_minute=600)  # 10/sec
        # 600 tokens capacity + 100 more at 10/sec = ~10s total.
        # We'll drain 600 quickly then ask for 30 more concurrently.
        for _ in range(600):
            bucket.acquire()

        barrier = threading.Barrier(3)

        def worker():
            barrier.wait()
            for _ in range(10):
                bucket.acquire()

        threads = [threading.Thread(target=worker) for _ in range(3)]
        start = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.monotonic() - start
        # 30 tokens at 10/sec = ~3s. Loose window — thread scheduling
        # on CI can add jitter, but we should still be in the 2-5s
        # range if the bucket is correctly serialising access.
        assert 2.0 < elapsed < 6.0, (
            f"3 threads × 10 acquires at 10/sec took {elapsed:.2f}s "
            "— bucket is not correctly serialising concurrent access"
        )


# ---------------------------------------------------------------------------
# get_bucket registry
# ---------------------------------------------------------------------------


class TestGetBucket:
    def test_empty_api_key_raises(self) -> None:
        with pytest.raises(ValueError, match="api_key"):
            get_bucket("", rate_per_minute=60)

    def test_same_key_returns_same_bucket(self) -> None:
        """The whole point of the module-level registry: providers
        sharing an API key must share the bucket."""
        b1 = get_bucket("key-a", rate_per_minute=60)
        b2 = get_bucket("key-a", rate_per_minute=60)
        assert b1 is b2

    def test_different_keys_get_different_buckets(self) -> None:
        """Different API keys have independent quotas."""
        b1 = get_bucket("key-a", rate_per_minute=60)
        b2 = get_bucket("key-b", rate_per_minute=60)
        assert b1 is not b2

    def test_second_caller_with_different_rate_keeps_original(self, caplog) -> None:
        """Second caller for the same key requesting a different
        rate gets the original bucket (first caller wins) AND a
        warning is logged."""
        import logging

        caplog.set_level(logging.WARNING, logger="neuromem_gemini.rate_limit")

        first = get_bucket("key-a", rate_per_minute=60)
        second = get_bucket("key-a", rate_per_minute=120)

        assert first is second
        assert first.rate_per_minute == 60
        assert any("already exists at rpm=60" in r.message for r in caplog.records)

    def test_registry_reset_clears_buckets(self) -> None:
        """Tests that need a clean slate can call _reset_registry."""
        first = get_bucket("key-a", rate_per_minute=60)
        _reset_registry()
        second = get_bucket("key-a", rate_per_minute=60)
        assert first is not second
