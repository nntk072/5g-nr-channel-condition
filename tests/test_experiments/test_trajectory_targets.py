"""Tests for timestamp-aligned trajectory forecast target indices."""

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
from experiments.trajectory_targets import build_trajectory_horizon_targets
from receivers.nr import NRChannelEstimate, NRPUSCHConfig


def _record(sample_id: str) -> NRChannelRecord:
    config = NRPUSCHConfig(n_size_bwp=1, subcarrier_spacing_khz=30)
    estimate = NRChannelEstimate(
        h_hat=np.ones((1, 1, 1, 1, 1, 1, 1), dtype=np.complex64),
        error_variance=np.ones((1, 1, 1, 1, 1, 1, 1), dtype=np.float32),
        pilot_mask=np.ones((1, 1, 1, 1), dtype=np.bool_),
        metadata={"fixture": True},
    )
    observation = NRObservation(
        sample_id=sample_id,
        received_grid=np.ones((1, 1, 1, 1, 1), dtype=np.complex64),
        channel_estimate=estimate,
        fft_bin_indices=(0,),
        receiver_config=config,
    )
    return NRChannelRecord(
        observation=observation,
        targets={"channel_cfr": np.ones(1, dtype=np.complex64)},
        target_masks={"channel_cfr": np.ones(1, dtype=np.bool_)},
        provenance={"fixture": "trajectory-targets"},
    )


def _trajectory(sample_ids: tuple[str, ...], times: tuple[float, ...]) -> NRTrajectory:
    return NRTrajectory(
        trajectory_id="trajectory-fixture",
        sample_times_s=times,
        records=tuple(_record(sample_id) for sample_id in sample_ids),
    )


def test_horizons_use_elapsed_time_and_exact_targets_for_irregular_sampling() -> None:
    trajectory = _trajectory(("t0", "t1", "t25", "t45"), (0.0, 1.0, 2.5, 4.5))

    targets = build_trajectory_horizon_targets(trajectory, (1.5, 2.0))

    assert [(row.origin_sample_id, row.target_sample_id) for row in targets] == [
        ("t1", "t25"),
        ("t25", "t45"),
    ]
    assert [row.requested_horizon_s for row in targets] == [1.5, 2.0]
    assert [row.actual_horizon_s for row in targets] == [1.5, 2.0]
    assert all(row.split is None for row in targets)


def test_horizon_does_not_substitute_a_nearby_timestamp() -> None:
    trajectory = _trajectory(("t0", "t1", "t25"), (0.0, 1.0, 2.5))

    assert build_trajectory_horizon_targets(trajectory, (1.4,)) == ()


def test_tolerance_that_matches_multiple_samples_is_rejected() -> None:
    trajectory = _trajectory(("t0", "t1", "t11"), (0.0, 1.0, 1.1))

    with pytest.raises(ValueError, match="multiple target samples"):
        build_trajectory_horizon_targets(trajectory, (1.05,), timestamp_tolerance_s=0.06)


def test_declared_gap_blocks_pairs_that_cross_it() -> None:
    trajectory = _trajectory(("t0", "t1", "t2"), (0.0, 1.0, 2.0))

    targets = build_trajectory_horizon_targets(trajectory, (1.0, 2.0), gap_after_sample_ids={"t0"})

    assert [(row.origin_sample_id, row.target_sample_id) for row in targets] == [("t1", "t2")]


def test_frozen_maximum_inter_sample_gap_blocks_only_crossing_pairs() -> None:
    trajectory = _trajectory(("t0", "t1", "t25", "t35"), (0.0, 1.0, 2.5, 3.5))

    targets = build_trajectory_horizon_targets(
        trajectory,
        (1.0, 2.5),
        maximum_inter_sample_gap_s=1.0,
    )

    assert [(row.origin_sample_id, row.target_sample_id) for row in targets] == [
        ("t0", "t1"),
        ("t25", "t35"),
    ]


@pytest.mark.parametrize("maximum_gap", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_maximum_inter_sample_gap_is_rejected(maximum_gap: float) -> None:
    trajectory = _trajectory(("t0", "t1"), (0.0, 1.0))

    with pytest.raises(ValueError, match="finite and positive"):
        build_trajectory_horizon_targets(
            trajectory,
            (1.0,),
            maximum_inter_sample_gap_s=maximum_gap,
        )


@pytest.mark.parametrize("maximum_gap", [True, "1.0", 1 + 0j])
def test_maximum_inter_sample_gap_rejects_coercion(maximum_gap: object) -> None:
    trajectory = _trajectory(("t0", "t1"), (0.0, 1.0))

    with pytest.raises(TypeError, match="real number or None"):
        build_trajectory_horizon_targets(
            trajectory,
            (1.0,),
            maximum_inter_sample_gap_s=maximum_gap,  # type: ignore[arg-type]
        )


def test_gap_marker_must_reference_a_trajectory_sample() -> None:
    trajectory = _trajectory(("t0", "t1"), (0.0, 1.0))

    with pytest.raises(ValueError, match="unknown samples"):
        build_trajectory_horizon_targets(trajectory, (1.0,), gap_after_sample_ids={"missing"})


def test_split_registry_boundary_blocks_cross_split_targets() -> None:
    parent_ids = tuple(f"parent-{index}" for index in range(64))
    subjects = tuple(
        SplitSubject(
            subject_id=parent_id,
            parent_type=SplitParentType.NETWORK_EPISODE,
            source_family="synthetic-test",
        )
        for parent_id in parent_ids
    )
    registry = make_split_policy_registry(subjects, policy=SplitPolicy.release_v2(seed="test"))
    adjacent = next(
        (left, right)
        for left, right in zip(parent_ids, parent_ids[1:])
        if registry.split_for_parent(left) != registry.split_for_parent(right)
    )
    sample_ids = ("left", "right")
    registry = bind_split_children(
        registry,
        tuple(
            SplitChildBinding(sample_id, parent_id, "trajectory_sample")
            for sample_id, parent_id in zip(sample_ids, adjacent, strict=True)
        ),
    )
    trajectory = _trajectory(sample_ids, (0.0, 1.0))

    targets = build_trajectory_horizon_targets(trajectory, (1.0,), split_registry=registry)

    assert targets == ()


@pytest.mark.parametrize("horizon", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_horizons_are_rejected(horizon: float) -> None:
    trajectory = _trajectory(("t0", "t1"), (0.0, 1.0))

    with pytest.raises(ValueError, match="finite and positive"):
        build_trajectory_horizon_targets(trajectory, (horizon,))
