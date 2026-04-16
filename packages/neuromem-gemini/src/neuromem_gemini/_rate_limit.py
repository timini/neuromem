"""Token-bucket rate limiter for Gemini API calls.

Gemini's quotas are per-API-key RPM (requests per minute), with a
burst allowance equal to the RPM itself. Our cognitive loop makes
bursts during ingestion (~N serial ``generate_summary`` calls, one
per turn in each session) and during dream cycles (batched
``extract_tags`` + ``extract_named_entities`` + ~20-40 serial
``generate_category_name`` merges). Without coordination, sustained
pressure either trips 429 RESOURCE_EXHAUSTED on free tiers or —
worse — 504 DEADLINE_EXCEEDED on paid tiers as Gemini internally
queues our requests past its patience. The 504 is what killed the
3-instance LongMemEval run after 16 minutes of ingestion.

This module provides a simple token-bucket rate limiter shared
across every ``GeminiLLMProvider`` / ``GeminiEmbeddingProvider`` /
``GeminiAnsweringClient`` that uses the same API key. The bucket
is created lazily on first use and refilled continuously at the
configured RPM. ``acquire()`` blocks until a token is available,
so callers don't need to think about retries — they just call and
the limiter paces them.

Design notes:

- Keyed by ``api_key`` only. If two providers for the same key are
  constructed with different RPM values, the **first provider wins**
  and a warning is logged. Coordinating providers should agree on
  the rate.
- Thread-safe. Dream cycles run on a background thread; ingestion
  runs on the caller thread. Both paths acquire from the same
  bucket, which serialises their API calls inside the RPM budget.
- Continuous refill (not interval-based): the bucket gains tokens
  proportional to elapsed monotonic time since last refill. This
  smooths bursts — 60 RPM doesn't mean "60 at t=0 then 0 for 60
  seconds", it means "1 token per second available on demand".
- Capacity == rate. Burst allowance matches Gemini's own. No point
  pre-buying more than a minute of tokens; we'd just spend them
  all at once and starve the following minute.

Integration: providers construct a bucket via :func:`get_bucket` at
``__init__`` and call ``self._bucket.acquire()`` before every
generate_content / embed_content call. See ``llm.py`` /
``embedder.py`` for the callsite.
"""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger("neuromem_gemini.rate_limit")


# Default safety cap on how long a caller is willing to wait for a
# token before giving up. 5 minutes is long enough to absorb a
# full minute of thundering-herd contention at low RPM, but short
# enough that a genuinely rate-limit-deadlocked run fails fast
# rather than silently hanging at 0% CPU indefinitely (seen in
# the Apr 2026 n=120 bench on gemini-3-flash-preview: 40+ threads
# contending for a preview-tier RPM that was effectively under 5,
# bench stuck for 46 min with no progress and no error).
_DEFAULT_ACQUIRE_TIMEOUT_S = 300.0


class BucketAcquireTimeout(TimeoutError):
    """Raised when ``TokenBucket.acquire()`` couldn't obtain a token
    within its configured wait budget.

    Subclass of ``TimeoutError`` so existing code that catches
    ``TimeoutError`` handles it naturally. The dedicated type lets
    callers distinguish rate-limit starvation from other timeouts
    (httpx transport timeouts, request deadlines) when they need to
    treat them differently.
    """


class TokenBucket:
    """Thread-safe token-bucket rate limiter.

    Capacity equals rate (one-minute burst). Tokens refill
    continuously at ``rate_per_minute / 60`` per second. ``acquire``
    blocks until a token is available or the timeout elapses.
    """

    def __init__(self, rate_per_minute: int) -> None:
        if rate_per_minute <= 0:
            raise ValueError(f"rate_per_minute must be > 0, got {rate_per_minute}")
        self._rate = rate_per_minute
        self._capacity = float(rate_per_minute)
        self._tokens = float(rate_per_minute)  # start full
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    @property
    def rate_per_minute(self) -> int:
        """The configured rate — read-only after construction."""
        return self._rate

    def acquire(self, *, timeout_s: float | None = None) -> None:
        """Block until one token is available, then consume it.

        Uses a simple loop with a sleep interval tuned to refill a
        single token from the current token count. That means callers
        wake up right as the next token is available rather than
        polling, so there's no busy-wait.

        ``timeout_s`` (default :data:`_DEFAULT_ACQUIRE_TIMEOUT_S`)
        caps the total wait. When exceeded, raises
        :class:`BucketAcquireTimeout` — a subclass of
        ``TimeoutError`` — so callers can distinguish rate-limit
        starvation from other timeout sources. Pass ``None`` to
        opt out of the cap (discouraged in production: a genuinely
        rate-limited call path can starve threads forever).
        """
        budget = _DEFAULT_ACQUIRE_TIMEOUT_S if timeout_s is None else timeout_s
        deadline = time.monotonic() + budget if budget is not None else None
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                refill = elapsed * (self._rate / 60.0)
                self._tokens = min(self._capacity, self._tokens + refill)
                self._last_refill = now

                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return

                # How long until we have one token?
                deficit = 1.0 - self._tokens
                wait_s = deficit / (self._rate / 60.0)

            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise BucketAcquireTimeout(
                        f"TokenBucket.acquire: could not get a token within "
                        f"{budget:.0f}s (rate={self._rate} rpm). This usually "
                        f"means the effective RPM is much lower than configured "
                        f"— preview-tier models and quota-exhausted keys are "
                        f"typical causes. Lower --workers or raise the model's "
                        f"quota."
                    )
                wait_s = min(wait_s, remaining)

            # Release the lock while sleeping so other threads can
            # also enter the waiting state. We do NOT hand out tokens
            # during the sleep — any caller that acquires the lock
            # next runs the refill math itself.
            time.sleep(wait_s)


# ---------------------------------------------------------------------------
# Module-level bucket registry
# ---------------------------------------------------------------------------

_buckets: dict[str, TokenBucket] = {}
_buckets_lock = threading.Lock()


def get_bucket(api_key: str, rate_per_minute: int) -> TokenBucket:
    """Return the shared bucket for ``api_key``, creating it lazily.

    If a bucket for this key already exists with a different rate,
    the existing bucket is returned unchanged and a warning is
    logged. The philosophy here is "first caller wins the rate
    config" — providers sharing a key should agree on the rate, but
    getting a rate mismatch shouldn't crash the app.

    The registry is process-wide and not cleaned up — buckets are
    lightweight and the number of distinct API keys in a process
    is small. Tests that need isolation should call
    :func:`_reset_registry` in a fixture.
    """
    if not api_key:
        raise ValueError("api_key must be non-empty")
    with _buckets_lock:
        bucket = _buckets.get(api_key)
        if bucket is None:
            bucket = TokenBucket(rate_per_minute)
            _buckets[api_key] = bucket
            logger.debug(
                "created rate-limit bucket api_key=***%s rpm=%d",
                api_key[-4:] if len(api_key) >= 4 else "?",
                rate_per_minute,
            )
        elif bucket.rate_per_minute != rate_per_minute:
            logger.warning(
                "rate-limit bucket for api_key=***%s already exists at "
                "rpm=%d; ignoring requested rpm=%d. Providers sharing an "
                "API key should agree on a rate.",
                api_key[-4:] if len(api_key) >= 4 else "?",
                bucket.rate_per_minute,
                rate_per_minute,
            )
        return bucket


def _reset_registry() -> None:
    """Test-only: clear the module-level bucket registry.

    Prefix is ``_`` because it's not part of the public API; tests
    can import it via the module path. Not intended for production
    use — buckets are deliberately process-lifetime singletons so
    separate providers can't accidentally coordinate through
    parallel registries.
    """
    with _buckets_lock:
        _buckets.clear()
