"""Prespecified receiver-only one-step forecasts for stored NR RT trajectories."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

import numpy as np
from datasets.nr import NRTrajectory
from datasets.split_policy import SplitPolicyRegistry


def evaluate_trajectory_forecasts(
    trajectory: NRTrajectory,
    *,
    split: str,
    gap_after_sample_ids: Collection[str] = (),
    split_registry: SplitPolicyRegistry | None = None,
) -> list[dict[str, Any]]:
    """Compare persistence and complex linear extrapolation to future RT CFR.

    Predictor inputs are only past DMRS channel estimates. The full RT CFR is
    used as an offline target. The linear predictor requires two prior samples;
    its timestamps are used explicitly and need not be equally spaced.

    A declared gap after a sample prevents forecasts that use that transition,
    including linear extrapolation whose history crosses the gap. Metrics use
    only target elements selected by the record's ``channel_cfr`` mask; targets
    with no valid elements produce no metric row.
    """
    if split not in {"train", "validation", "calibration", "test"}:
        raise ValueError("split must be train, validation, calibration, or test")
    if split_registry is not None:
        if not isinstance(split_registry, SplitPolicyRegistry):
            raise TypeError("split_registry must be a SplitPolicyRegistry")
        sample_splits = tuple(
            split_registry.split_for_child(sample_id) for sample_id in trajectory.sample_ids
        )
        if any(sample_split != split for sample_split in sample_splits):
            raise ValueError(
                f"trajectory samples do not all belong to the requested {split!r} split"
            )
    try:
        gaps = frozenset(gap_after_sample_ids)
    except TypeError as error:
        raise TypeError("gap_after_sample_ids must be a collection of sample IDs") from error
    if any(not isinstance(value, str) or not value.strip() for value in gaps):
        raise TypeError("gap_after_sample_ids must contain non-empty sample IDs")
    unknown_gaps = gaps.difference(trajectory.sample_ids)
    if unknown_gaps:
        raise ValueError(f"gap markers reference unknown samples: {sorted(unknown_gaps)}")
    if len(trajectory.records) < 2:
        raise ValueError("at least two trajectory samples are required")

    observations: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for record in trajectory.records:
        estimate = record.observation.channel_estimate.h_hat
        if hasattr(estimate, "detach"):
            estimate = estimate.detach().cpu().numpy()
        estimate_array = np.asarray(estimate)
        if estimate_array.ndim != 7 or estimate_array.shape[0] != 1:
            raise ValueError(
                "each estimate must be one [batch, rx, ant, tx, stream, symbol, tone] sample"
            )
        observed_cfr = estimate_array[0, 0, 0, 0, 0].mean(axis=0)
        target_cfr = np.asarray(record.targets.get("channel_cfr"))
        if (
            target_cfr.shape != observed_cfr.shape
            or not np.iscomplexobj(target_cfr)
            or not np.isfinite(target_cfr).all()
            or not np.isfinite(observed_cfr).all()
        ):
            raise ValueError(
                "receiver estimates and full-channel targets must be aligned finite CFRs"
            )
        observations.append(observed_cfr.astype(np.complex128, copy=False))
        targets.append(target_cfr.astype(np.complex128, copy=False))

    rows: list[dict[str, Any]] = []
    for index in range(1, len(trajectory.records)):
        # A gap marker on the origin means the next stored sample is beyond a
        # discontinuity and must not be used as a forecast transition.
        if trajectory.sample_ids[index - 1] in gaps:
            continue
        observation = observations[index - 1]
        target = targets[index]
        target_mask = trajectory.records[index].target_masks.get("channel_cfr")
        if target_mask is None:
            selected = np.ones(target.shape, dtype=bool)
        else:
            mask_detach = getattr(target_mask, "detach", None)
            if callable(mask_detach):
                target_mask = mask_detach().cpu().numpy()
            raw_mask = np.asarray(target_mask)
            if raw_mask.dtype.kind != "b":
                raise TypeError("channel_cfr target mask must have boolean dtype")
            try:
                selected = np.broadcast_to(raw_mask, target.shape)
            except ValueError as error:
                raise ValueError("channel_cfr target mask does not match its target") from error
        valid_count = int(np.count_nonzero(selected))
        if valid_count == 0:
            continue
        denominator = float(np.mean(np.abs(target[selected]) ** 2))
        if denominator <= np.finfo(np.float64).tiny:
            raise ValueError("zero-power valid full-channel targets need an explicit outage metric")
        persistence_nmse = float(
            np.mean(np.abs(observation[selected] - target[selected]) ** 2) / denominator
        )
        rows.append(
            {
                "trajectory_id": trajectory.trajectory_id,
                "sample_id": trajectory.records[index].observation.sample_id,
                "split": split,
                "forecast_origin_s": trajectory.sample_times_s[index - 1],
                "forecast_target_s": trajectory.sample_times_s[index],
                "horizon_s": trajectory.sample_times_s[index]
                - trajectory.sample_times_s[index - 1],
                "valid_target_element_count": valid_count,
                "method": "persistence",
                "nmse_linear": persistence_nmse,
                "nmse_db": float(10.0 * np.log10(max(persistence_nmse, np.finfo(float).tiny))),
            }
        )
        if index >= 2 and trajectory.sample_ids[index - 2] not in gaps:
            prior_interval = (
                trajectory.sample_times_s[index - 1] - trajectory.sample_times_s[index - 2]
            )
            if prior_interval <= 0:
                raise ValueError("trajectory timestamps must be strictly increasing")
            horizon = trajectory.sample_times_s[index] - trajectory.sample_times_s[index - 1]
            prediction = observation + (observation - observations[index - 2]) * (
                horizon / prior_interval
            )
            linear_nmse = float(
                np.mean(np.abs(prediction[selected] - target[selected]) ** 2) / denominator
            )
            rows.append(
                {
                    "trajectory_id": trajectory.trajectory_id,
                    "sample_id": trajectory.records[index].observation.sample_id,
                    "split": split,
                    "forecast_origin_s": trajectory.sample_times_s[index - 1],
                    "forecast_target_s": trajectory.sample_times_s[index],
                    "horizon_s": horizon,
                    "valid_target_element_count": valid_count,
                    "method": "linear_complex_extrapolation",
                    "nmse_linear": linear_nmse,
                    "nmse_db": float(10.0 * np.log10(max(linear_nmse, np.finfo(float).tiny))),
                }
            )
    return rows


def summarize_forecast_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize per-transition errors without treating transitions as independent sites."""
    summaries: dict[str, Any] = {}
    methods = sorted({str(row["method"]) for row in rows})
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        per_trajectory: dict[str, list[float]] = {}
        for row in selected:
            per_trajectory.setdefault(str(row["trajectory_id"]), []).append(
                float(row["nmse_linear"])
            )
        trajectory_mean_nmse = {
            trajectory_id: float(np.mean(values))
            for trajectory_id, values in sorted(per_trajectory.items())
        }
        summaries[method] = {
            "transition_count": len(selected),
            "trajectory_count": len(trajectory_mean_nmse),
            "mean_transition_nmse_linear": float(np.mean([row["nmse_linear"] for row in selected])),
            "mean_transition_nmse_db": float(np.mean([row["nmse_db"] for row in selected])),
            "median_transition_nmse_db": float(np.median([row["nmse_db"] for row in selected])),
            "per_trajectory_mean_nmse_linear": trajectory_mean_nmse,
        }
    return summaries
