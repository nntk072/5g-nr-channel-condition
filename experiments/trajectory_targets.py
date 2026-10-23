"""Build timestamp-aligned forecast targets for NR trajectories."""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from numbers import Real

import numpy as np
from datasets.nr import NRTrajectory
from datasets.split_policy import SplitPolicyRegistry


@dataclass(frozen=True, slots=True)
class TrajectoryForecastTarget:
    """One eligible origin and target index for an elapsed-time horizon."""

    trajectory_id: str
    origin_index: int
    target_index: int
    origin_sample_id: str
    target_sample_id: str
    origin_time_s: float
    target_time_s: float
    requested_horizon_s: float
    actual_horizon_s: float
    split: str | None


def build_trajectory_horizon_targets(
    trajectory: NRTrajectory,
    horizons_s: Sequence[float],
    *,
    gap_after_sample_ids: Collection[str] = (),
    split_registry: SplitPolicyRegistry | None = None,
    timestamp_tolerance_s: float = 1e-9,
) -> tuple[TrajectoryForecastTarget, ...]:
    """Index exact elapsed-time targets without crossing gaps or split borders.

    Irregular timestamps are supported: each horizon is added to an origin
    timestamp, and only a sample at that resulting time (within the absolute
    tolerance) is eligible. No row-offset assumption, interpolation, or
    nearest-sample substitution is made. A gap marker names the sample just
    before a declared discontinuity and blocks intervals crossing that edge.
    Targets with no matching timestamp and pairs blocked by a gap or split
    boundary are omitted. When a split registry is provided, each sample ID
    must be registered as a child and pairs are emitted only when the full
    origin-to-target interval remains in one split. This builder indexes
    target records but does not validate their target masks or validity codes.

    Args:
        trajectory: Strictly time-ordered observations and targets.
        horizons_s: Positive elapsed-time forecast horizons.
        gap_after_sample_ids: Sample IDs after which no forecast may continue.
        split_registry: Frozen registry that resolves sample child IDs to splits.
        timestamp_tolerance_s: Absolute tolerance for floating-point timestamp
            matching; must be finite and non-negative.

    Returns:
        Eligible forecast target indices ordered by origin time then ascending
        requested horizon.

    Raises:
        TypeError: If inputs have invalid types.
        ValueError: If horizons, gaps, or timestamp tolerance are invalid.
        KeyError: If a sample is absent from the supplied frozen split registry.
    """
    if not isinstance(trajectory, NRTrajectory):
        raise TypeError("trajectory must be an NRTrajectory")
    if isinstance(timestamp_tolerance_s, bool) or not isinstance(timestamp_tolerance_s, Real):
        raise TypeError("timestamp_tolerance_s must be a real number")
    tolerance = float(timestamp_tolerance_s)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("timestamp_tolerance_s must be finite and non-negative")

    horizons = tuple(horizons_s)
    if not horizons:
        raise ValueError("at least one forecast horizon is required")
    normalized_horizons: list[float] = []
    for horizon in horizons:
        if isinstance(horizon, bool) or not isinstance(horizon, Real):
            raise TypeError("forecast horizons must be real numbers")
        value = float(horizon)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("forecast horizons must be finite and positive")
        normalized_horizons.append(value)
    if len(set(normalized_horizons)) != len(normalized_horizons):
        raise ValueError("forecast horizons must be unique")
    normalized_horizons.sort()

    gaps = frozenset(gap_after_sample_ids)
    if any(not isinstance(sample_id, str) or not sample_id.strip() for sample_id in gaps):
        raise TypeError("gap markers must be non-empty sample IDs")
    unknown_gaps = gaps.difference(trajectory.sample_ids)
    if unknown_gaps:
        raise ValueError(f"gap markers reference unknown samples: {sorted(unknown_gaps)}")
    gap_indices = {trajectory.sample_ids.index(sample_id) for sample_id in gaps}

    sample_splits: tuple[str, ...] | None = None
    if split_registry is not None:
        if not isinstance(split_registry, SplitPolicyRegistry):
            raise TypeError("split_registry must be a SplitPolicyRegistry")
        sample_splits = tuple(
            split_registry.split_for_child(sample_id) for sample_id in trajectory.sample_ids
        )

    times = np.asarray(trajectory.sample_times_s, dtype=np.float64)
    rows: list[TrajectoryForecastTarget] = []
    for origin_index, origin_time in enumerate(times):
        for horizon in normalized_horizons:
            requested_target_time = float(origin_time + horizon)
            first_candidate = int(
                np.searchsorted(times, requested_target_time - tolerance, side="left")
            )
            candidate_stop = int(
                np.searchsorted(times, requested_target_time + tolerance, side="right")
            )
            matching = [
                index for index in range(max(first_candidate, origin_index + 1), candidate_stop)
            ]
            if not matching:
                continue
            if len(matching) > 1:
                raise ValueError(
                    "timestamp_tolerance_s matches multiple target samples; reduce the tolerance"
                )
            target_index = matching[0]
            if any(index in gap_indices for index in range(origin_index, target_index)):
                continue
            interval_splits = (
                set(sample_splits[origin_index : target_index + 1])
                if sample_splits is not None
                else set()
            )
            if len(interval_splits) > 1:
                continue
            rows.append(
                TrajectoryForecastTarget(
                    trajectory_id=trajectory.trajectory_id,
                    origin_index=origin_index,
                    target_index=target_index,
                    origin_sample_id=trajectory.sample_ids[origin_index],
                    target_sample_id=trajectory.sample_ids[target_index],
                    origin_time_s=float(times[origin_index]),
                    target_time_s=float(times[target_index]),
                    requested_horizon_s=horizon,
                    actual_horizon_s=float(times[target_index] - times[origin_index]),
                    split=(sample_splits[origin_index] if sample_splits is not None else None),
                )
            )
    return tuple(rows)
