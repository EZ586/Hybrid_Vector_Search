from __future__ import annotations
import time

def time_ms(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    ms = (time.perf_counter() - t0) * 1000.0
    return out, ms
