# src/baselines/hybrid/scheduler.py
"""
nprobe schedulers for hybrid search.

Simple nprobe scheduler (start → linear grow until K valid)".
This module yields a sequence of nprobe values; the search loop decides when to stop.
"""

from typing import Iterator


def linear_nprobe_scheduler(
    start: int = 4,
    step: int = 4,
    max_nprobe: int = 64,
) -> Iterator[int]:
    """
    Yield nprobe values linearly increasing from `start` to `max_nprobe`.

    Args:
        start: first nprobe to try (must be >= 1).
        step: amount to increase each round (must be >= 1).
        max_nprobe: upper cap on nprobe (inclusive).

    Yields:
        int: next nprobe value to try.
    """
    if start < 1:
        raise ValueError("start must be >= 1")
    if step < 1:
        raise ValueError("step must be >= 1")
    if max_nprobe < start:
        # nothing to yield
        return

    current = start
    while current <= max_nprobe:
        yield current
        current += step

__all__ = [
    "linear_nprobe_scheduler",
]
