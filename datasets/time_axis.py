"""Strict real-valued time-axis validation shared by dataset contracts."""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class TimeAxisAlignment:
    """One-to-one timestamp matches from a reference axis to an observed axis."""

    reference_times_s: tuple[float, ...]
    observed_times_s: tuple[float, ...]
    tolerance_s: float
    observed_indices: tuple[int | None, ...]
    residuals_s: tuple[float | None, ...]
    unmatched_observed_indices: tuple[int, ...]

    @property
    def complete(self) -> bool:
        """Whether every reference timestamp has an observed match."""
        return all(index is not None for index in self.observed_indices)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible report retaining unmatched samples and residuals."""
        return {
            "schema": "umcd.time_axis_alignment.v1",
            "method": "unique_match_within_absolute_tolerance_no_interpolation",
            "reference_times_s": list(self.reference_times_s),
            "observed_times_s": list(self.observed_times_s),
            "tolerance_s": self.tolerance_s,
            "observed_indices": list(self.observed_indices),
            "residuals_s": list(self.residuals_s),
            "unmatched_reference_indices": [
                index
                for index, observed_index in enumerate(self.observed_indices)
                if observed_index is None
            ],
            "unmatched_observed_indices": list(self.unmatched_observed_indices),
            "complete": self.complete,
        }


def align_time_axes(
    reference_times_s: Iterable[object],
    observed_times_s: Iterable[object],
    *,
    tolerance_s: Real,
    require_complete: bool = False,
) -> TimeAxisAlignment:
    """Match two monotone time axes without interpolation or sample reuse.

    Each reference timestamp must have at most one observed timestamp within the
    caller-declared absolute tolerance. Missing matches are reported unless
    ``require_complete`` is true. An observed sample can satisfy only one
    reference timestamp; extra observed samples are retained as unmatched rows.
    The returned residual for a match is ``observed_time - reference_time``.
    """
    reference = canonical_time_axis(reference_times_s, name="reference_times_s")
    observed = canonical_time_axis(observed_times_s, name="observed_times_s")
    if isinstance(tolerance_s, bool) or not isinstance(tolerance_s, Real):
        raise TypeError("tolerance_s must be a real number")
    tolerance = float(tolerance_s)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance_s must be finite and non-negative")
    if not isinstance(require_complete, bool):
        raise TypeError("require_complete must be a boolean")

    observed_indices: list[int | None] = []
    residuals: list[float | None] = []
    used: set[int] = set()
    for reference_time in reference:
        start = bisect_left(observed, reference_time - tolerance)
        stop = bisect_right(observed, reference_time + tolerance)
        if stop - start > 1:
            raise ValueError(
                "timestamp tolerance matches multiple observed samples for a reference sample"
            )
        if start == stop:
            observed_indices.append(None)
            residuals.append(None)
            continue
        observed_index = start
        if observed_index in used:
            raise ValueError("one observed timestamp cannot align to multiple reference samples")
        used.add(observed_index)
        observed_indices.append(observed_index)
        residuals.append(observed[observed_index] - reference_time)

    if require_complete and any(index is None for index in observed_indices):
        missing = [
            index for index, observed_index in enumerate(observed_indices) if observed_index is None
        ]
        raise ValueError(f"time-axis alignment is missing reference samples: {missing}")
    return TimeAxisAlignment(
        reference_times_s=reference,
        observed_times_s=observed,
        tolerance_s=tolerance,
        observed_indices=tuple(observed_indices),
        residuals_s=tuple(residuals),
        unmatched_observed_indices=tuple(
            index for index in range(len(observed)) if index not in used
        ),
    )
