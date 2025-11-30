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

def geometric_nprobe_scheduler(
    start: int = 4,
    factor: float = 2.0,
    max_nprobe: int = 64,
) -> Iterator[int]:
    """
    Yield nprobe values that grow geometrically from `start` up to `max_nprobe`.

    Args:
        start: first nprobe to try (must be >= 1).
        factor: multiplicative growth factor per round (must be > 1.0).
        max_nprobe: upper cap on nprobe (inclusive).

    Yields:
        int: next nprobe value to try.
    """
    if start < 1:
        raise ValueError("start must be >= 1")
    if factor <= 1.0:
        raise ValueError("factor must be > 1.0")
    if max_nprobe < start:
        # nothing to yield
        return

    current = start
    while current <= max_nprobe:
        yield current
        # multiply then clamp to avoid infinite loops if we overshoot
        next_val = int(current * factor)
        if next_val <= current:
            # safeguard: fall back to linear increment of 1
            next_val = current + 1
        current = next_val

__all__ = [
    "linear_nprobe_scheduler",
    "geometric_nprobe_scheduler",
    "linear_list_budget_scheduler",
]

def linear_list_budget_scheduler(
    n_lists: int,
    start_lists: int = 4,
    step_lists: int = 4,
    max_lists: int | None = None,
) -> Iterator[int]:
    """
    Yield list-budget values for probing IVF lists in order.

    This is conceptually similar to `linear_nprobe_scheduler`, but operates in
    terms of *how many IVF lists from a precomputed probe_order* should be used
    in each round. A hybrid search loop can interpret the yielded value `m` as:
        "probe the first `m` lists from probe_order in this round".

    Args:
        n_lists: total number of IVF lists in the index (L).
        start_lists: number of lists to include in the first round (>= 1).
        step_lists: increment in list budget per round (>= 1).
        max_lists: optional cap on lists per round; if None, defaults to n_lists.

    Yields:
        int: list budget for the current round (clamped to [1, n_lists]).
    """
    if n_lists <= 0:
        return

    if start_lists < 1:
        raise ValueError("start_lists must be >= 1")
    if step_lists < 1:
        raise ValueError("step_lists must be >= 1")

    if max_lists is None:
        max_lists = n_lists

    if max_lists < start_lists:
        # nothing to yield
        return

    current = start_lists
    while current <= max_lists:
        # never exceed total number of lists
        yield min(current, n_lists)
        current += step_lists