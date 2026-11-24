"""Trajectory horizon targets: typed inputs, horizon roster rules and gap-marker identities."""

from __future__ import annotations

import pytest
from experiments.trajectory_targets import build_trajectory_horizon_targets
from tests.test_experiments.test_trajectory_targets import _trajectory


def test_trajectory_argument_must_be_an_nr_trajectory() -> None:
    with pytest.raises(TypeError, match="trajectory must be an NRTrajectory"):
        build_trajectory_horizon_targets("not-a-trajectory", (1.0,))


@pytest.mark.parametrize("tolerance", [True, "0.1"])
def test_timestamp_tolerance_must_be_real(tolerance) -> None:
    trajectory = _trajectory(("t0", "t1"), (0.0, 1.0))
    with pytest.raises(TypeError, match="timestamp_tolerance_s must be a real number"):
        build_trajectory_horizon_targets(trajectory, (1.0,), timestamp_tolerance_s=tolerance)


@pytest.mark.parametrize("tolerance", [float("nan"), -1e-9])
def test_timestamp_tolerance_must_be_finite_and_nonnegative(tolerance) -> None:
    trajectory = _trajectory(("t0", "t1"), (0.0, 1.0))
    with pytest.raises(ValueError, match="finite and non-negative"):
        build_trajectory_horizon_targets(trajectory, (1.0,), timestamp_tolerance_s=tolerance)


def test_horizons_must_be_nonempty_real_and_unique() -> None:
    trajectory = _trajectory(("t0", "t1"), (0.0, 1.0))
    with pytest.raises(ValueError, match="at least one forecast horizon"):
        build_trajectory_horizon_targets(trajectory, ())
    with pytest.raises(TypeError, match="forecast horizons must be real numbers"):
        build_trajectory_horizon_targets(trajectory, (True,))
    with pytest.raises(ValueError, match="forecast horizons must be unique"):
        build_trajectory_horizon_targets(trajectory, (1.0, 1.0))


@pytest.mark.parametrize("gap", [("",), (3,)])
def test_gap_markers_must_be_nonempty_sample_identifiers(gap) -> None:
    trajectory = _trajectory(("t0", "t1"), (0.0, 1.0))
    with pytest.raises(TypeError, match="gap markers must be non-empty sample IDs"):
        build_trajectory_horizon_targets(trajectory, (1.0,), gap_after_sample_ids=gap)
