"""Strict real-valued time-axis validation shared by dataset contracts."""

from __future__ import annotations

import math
from numbers import Real
from typing import Iterable


def canonical_time_axis(values: Iterable[object], *, name: str) -> tuple[float, ...]:
    """Return finite, strictly increasing real timestamps without coercing aliases."""
    try:
        raw_times = tuple(values)
    except TypeError as error:
        raise TypeError(f"{name} must be an iterable of real timestamps") from error
    if not raw_times:
        raise ValueError(f"{name} must be a non-empty finite time axis")
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in raw_times):
        raise TypeError(f"{name} values must be real numeric timestamps")
    try:
        times = tuple(float(value) for value in raw_times)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} values must be finite real timestamps") from error
    if any(not math.isfinite(value) for value in times):
        raise ValueError(f"{name} values must be finite real timestamps")
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError(f"{name} must be strictly increasing")
    return times
