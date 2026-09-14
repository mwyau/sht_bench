"""Timing calibration shared by the benchmark executables."""

from __future__ import annotations

import gc
import math
import time
from collections.abc import Callable


def _time_block(fn: Callable[[], object], iterations: int) -> float:
    start = time.perf_counter_ns()
    for _ in range(iterations):
        fn()
    return (time.perf_counter_ns() - start) / 1e9


def measure(
    fn: Callable[[], object], *, warmup: int, repeat: int, min_time: float
) -> tuple[int, list[float]]:
    """Warm up and repeatedly time *fn* using the established SHT method.

    The returned sample values are seconds per invocation.  Calibration and
    warm-up happen before the repeated samples; garbage collection is disabled
    only while those samples are collected.
    """

    for _ in range(warmup):
        fn()
    iterations = 1
    while True:
        elapsed = _time_block(fn, iterations)
        if elapsed >= min_time or iterations >= 1 << 20:
            break
        scale = max(2, min(16, math.ceil(min_time / max(elapsed, 1e-12))))
        iterations *= scale
    samples: list[float] = []
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeat):
            elapsed = _time_block(fn, iterations)
            samples.append(elapsed / iterations)
    finally:
        if was_enabled:
            gc.enable()
    return iterations, samples
