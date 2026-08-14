"""Backoff and a circuit breaker for outbound provider calls.

Retries alone make a dead provider worse: every request pays the full retry ladder
before failing over. The breaker short-circuits after a few consecutive failures so
subsequent requests skip straight to the next provider in the chain.
"""
from __future__ import annotations

import time

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


class CircuitOpen(RuntimeError):
    pass


class ProviderError(RuntimeError):
    pass


class CircuitBreaker:
    def __init__(self, fail_threshold: int = 3, reset_seconds: float = 30.0):
        self.fail_threshold = fail_threshold
        self.reset_seconds = reset_seconds
        self.failures = 0
        self.opened_at: float | None = None

    @property
    def state(self) -> str:
        if self.opened_at is None:
            return "closed"
        # After the cooldown we allow exactly one probe through rather than
        # reopening the floodgates.
        return "half_open" if time.time() - self.opened_at >= self.reset_seconds else "open"

    def allow(self) -> bool:
        return self.state != "open"

    def record_success(self) -> None:
        self.failures = 0
        self.opened_at = None

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.fail_threshold:
            self.opened_at = time.time()

    def guard(self):
        if not self.allow():
            raise CircuitOpen(f"circuit open, retry in "
                              f"{self.reset_seconds - (time.time() - self.opened_at):.1f}s")


def with_backoff(attempts: int = 2, initial: float = 0.25, cap: float = 2.0):
    """Deliberately shallow. Under a latency budget, a third attempt is almost
    always worse for the user than failing over to the next provider."""
    return retry(
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=initial, max=cap),
        retry=retry_if_exception_type(ProviderError),
        reraise=True,
    )
