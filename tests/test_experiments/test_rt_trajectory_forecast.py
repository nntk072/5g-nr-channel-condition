"""Tests for fixed receiver-only RT trajectory forecast baselines."""

from dataclasses import replace

import numpy as np
import pytest
from datasets.nr import NRChannelRecord, NRObservation, NRTrajectory
from datasets.split_policy import (
    SplitChildBinding,
    SplitParentType,
    SplitPolicy,
    SplitSubject,
    bind_split_children,
    make_split_policy_registry,
)
from datasets.targets import TargetValidity
from experiments.rt_trajectory_forecast import (
    evaluate_trajectory_forecasts,
    summarize_forecast_rows,
)
from receivers.nr import NRChannelEstimate, NRPUSCHConfig


def _record(sample_id: str, observed_level: float, target_level: float) -> NRChannelRecord:
    config = NRPUSCHConfig(n_size_bwp=1, subcarrier_spacing_khz=30)
    observed = np.full((1, 1, 1, 1, 1, 2, 3), observed_level, dtype=np.complex64)
    estimate = NRChannelEstimate(
        h_hat=observed,
        error_variance=np.ones((1, 1, 1, 1, 1, 2, 3), dtype=np.float32),
        pilot_mask=np.ones((1, 1, 2, 3), dtype=np.bool_),
        metadata={"receiver": "fixture"},
    )
    observation = NRObservation(
        sample_id=sample_id,
        received_grid=np.ones((1, 1, 1, 2, 3), dtype=np.complex64),
        channel_estimate=estimate,
        fft_bin_indices=(0, 1, 2),
        receiver_config=config,
    )
    return NRChannelRecord(
        observation=observation,
        targets={"channel_cfr": np.full(3, target_level, dtype=np.complex64)},
        target_masks={"channel_cfr": np.ones(3, dtype=np.bool_)},
        provenance={"fixture": "forecast-test"},
    )


def test_persistence_and_timestamp_scaled_linear_forecasts_use_only_past_estimates() -> None:
    trajectory = NRTrajectory(
        trajectory_id="forecast-fixture",
        sample_times_s=(0.0, 1.0, 3.0),
        records=(
            _record("t0", 1.0, 1.0),
            _record("t1", 3.0, 3.0),
            _record("t3", 7.0, 7.0),
        ),
    )

    rows = evaluate_trajectory_forecasts(trajectory, split="test")
    summary = summarize_forecast_rows(rows)

    assert summary["persistence"]["transition_count"] == 2
    assert summary["linear_complex_extrapolation"]["transition_count"] == 1
    linear = next(row for row in rows if row["method"] == "linear_complex_extrapolation")
    persistence = next(
        row for row in rows if row["method"] == "persistence" and row["sample_id"] == "t3"
    )
    assert linear["horizon_s"] == pytest.approx(2.0)
    assert linear["nmse_linear"] == pytest.approx(0.0, abs=1e-14)
    assert persistence["nmse_linear"] > linear["nmse_linear"]
    assert all(row["split"] == "test" for row in rows)


def test_forecast_baselines_reject_unrecognized_split_or_single_snapshot() -> None:
    with pytest.raises(ValueError, match="split"):
        evaluate_trajectory_forecasts(
            NRTrajectory("one", (0.0, 1.0), (_record("a", 1.0, 1.0), _record("b", 1.0, 1.0))),
            split="development",
        )


def test_forecast_baselines_accept_calibration_partition() -> None:
    trajectory = NRTrajectory(
        "calibration-split",
        (0.0, 1.0),
        (_record("cal-a", 1.0, 1.0), _record("cal-b", 1.0, 1.0)),
    )

    rows = evaluate_trajectory_forecasts(trajectory, split="calibration")

    assert rows
    assert {row["split"] for row in rows} == {"calibration"}


def test_forecast_metrics_use_only_valid_target_elements() -> None:
    first = _record("t0", 1.0, 1.0)
    target = _record("t1", 1.0, 1.0)
    target = replace(
        target,
        targets={"channel_cfr": np.asarray([2.0, 0.0, 0.0], dtype=np.complex64)},
        target_masks={"channel_cfr": np.asarray([True, False, False])},
        target_reason_codes={
            "channel_cfr": (
                TargetValidity.VALID,
                TargetValidity.NOT_OBSERVABLE,
                TargetValidity.NOT_OBSERVABLE,
            )
        },
    )
    trajectory = NRTrajectory("masked", (0.0, 1.0), (first, target))

    rows = evaluate_trajectory_forecasts(trajectory, split="test")

    assert len(rows) == 1
    assert rows[0]["valid_target_element_count"] == 1
    assert rows[0]["nmse_linear"] == pytest.approx(0.25)


def test_forecast_metrics_skip_empty_support_and_declared_time_gaps() -> None:
    first = _record("t0", 1.0, 1.0)
    empty_target = replace(
        _record("t1", 1.0, 1.0),
        targets={"channel_cfr": np.zeros(3, dtype=np.complex64)},
        target_masks={"channel_cfr": np.zeros(3, dtype=np.bool_)},
        target_reason_codes={"channel_cfr": (TargetValidity.NOT_OBSERVABLE,) * 3},
    )
    last = replace(
        _record("t2", 1.0, 1.0),
        targets={"channel_cfr": np.zeros(3, dtype=np.complex64)},
        target_masks={"channel_cfr": np.zeros(3, dtype=np.bool_)},
        target_reason_codes={"channel_cfr": (TargetValidity.NOT_OBSERVABLE,) * 3},
    )
    trajectory = NRTrajectory("gapped", (0.0, 1.0, 2.0), (first, empty_target, last))

    no_support = evaluate_trajectory_forecasts(trajectory, split="test")
    across_gap = evaluate_trajectory_forecasts(
        trajectory, split="test", gap_after_sample_ids={"t0"}
    )

    assert no_support == []
    assert across_gap == []
    with pytest.raises(ValueError, match="unknown samples"):
        evaluate_trajectory_forecasts(
            trajectory, split="test", gap_after_sample_ids={"not-in-trajectory"}
        )


def test_forecast_evaluator_enforces_frozen_split_assignment() -> None:
    trajectory = NRTrajectory(
        "split-bound",
        (0.0, 1.0),
        (_record("sample-a", 1.0, 1.0), _record("sample-b", 1.0, 1.0)),
    )
    parents = tuple(f"parent-{index}" for index in range(64))
    subjects = tuple(
        SplitSubject(
            subject_id=parent_id,
            parent_type=SplitParentType.NETWORK_EPISODE,
            source_family="synthetic-test",
        )
        for parent_id in parents
    )
    parent_registry = make_split_policy_registry(
        subjects, policy=SplitPolicy.release_v2(seed="forecast-registry-test")
    )
    registry = bind_split_children(
        parent_registry,
        tuple(
            SplitChildBinding(sample_id, parents[index], "nr_trajectory_sample")
            for index, sample_id in enumerate(trajectory.sample_ids)
        ),
    )
    assigned_splits = {registry.split_for_child(sample_id) for sample_id in trajectory.sample_ids}
    wrong_split = next(
        candidate
        for candidate in ("train", "validation", "test")
        if candidate not in assigned_splits
    )

    with pytest.raises(ValueError, match="do not all belong"):
        evaluate_trajectory_forecasts(trajectory, split=wrong_split, split_registry=registry)
