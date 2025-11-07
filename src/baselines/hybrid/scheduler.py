# baselines/hybrid/scheduler.py

from typing import Iterator


def linear_nprobe_scheduler(
    start: int = 4,
    step: int = 4,
    max_nprobe: int = 64,
) -> Iterator[int]:
    """
    Simple scheduler: start at `start`, grow by `step` until `max_nprobe`.

    Yields:
        int: next nprobe value to try.
    """
    # TODO: yield start, start+step, ... capped at max_nprobe
    n = start 
    while n <= max_nprobe:
        yield n
        n += step
