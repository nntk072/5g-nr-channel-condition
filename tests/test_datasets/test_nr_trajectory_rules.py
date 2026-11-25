"""NR trajectories and target-support audits: timing, identity, target roster and coverage rules."""

from __future__ import annotations

import numpy as np
import pytest
from datasets.nr import NRTrajectory, audit_nr_trajectory_target_support
from tests.test_datasets.test_nr import make_single_record


def _record(sample_id: str, value: float = 1.0):
    return make_single_record(sample_id, targets={"scalar": np.asarray(value)}, target_masks={})


def test_trajectory_requires_one_time_per_record_and_canonical_identity() -> None:
    first, second = _record("a"), _record("b")
    with pytest.raises(TypeError, match="trajectory_id must be a string"):
        NRTrajectory(7, (0.0, 1.0), (first, second))
    with pytest.raises(ValueError, match="non-empty canonical string"):
        NRTrajectory(" traj", (0.0, 1.0), (first, second))
    with pytest.raises(ValueError, match="one time for every trajectory record"):
        NRTrajectory("traj", (0.0,), (first, second))
    with pytest.raises(ValueError, match="one time for every trajectory record"):
        NRTrajectory("traj", (), ())


def test_trajectory_sample_ids_must_be_unique() -> None:
    record = _record("same")
    with pytest.raises(ValueError, match="sample IDs must be unique"):
        NRTrajectory("traj", (0.0, 1.0), (record, record))


def test_trajectory_target_series_and_unknown_targets() -> None:
    trajectory = NRTrajectory("traj", (0.0, 1.0), (_record("a", 1.0), _record("b", 2.0)))
    values, masks = trajectory.target_series("scalar")
    assert len(values) == 2 and len(masks) == 2
    with pytest.raises(KeyError, match="unknown trajectory target"):
        trajectory.target_series("absent")
    with pytest.raises(KeyError, match="unknown trajectory target"):
        trajectory.target_reason_series("absent")
    with pytest.raises(KeyError, match="unknown trajectory target"):
        trajectory.target_support_counts("absent")
    assert len(trajectory.model_inputs()) == 2


def test_audit_requires_expected_cells_and_well_formed_assignments() -> None:
    trajectory = NRTrajectory("traj", (0.0,), (_record("a"),))
    with pytest.raises(ValueError, match="expected_cells must declare"):
        audit_nr_trajectory_target_support((("train", "fam", trajectory),), ())
    with pytest.raises(ValueError, match="each expected cell must be a"):
        audit_nr_trajectory_target_support((), ("train", "fam"))
    with pytest.raises(ValueError, match="expected split, family, and target IDs"):
        audit_nr_trajectory_target_support((), (("train", "", "scalar"),))
    with pytest.raises(ValueError, match="duplicate expected NR target cell"):
        audit_nr_trajectory_target_support(
            (), (("train", "fam", "scalar"), ("train", "fam", "scalar"))
        )
    with pytest.raises(ValueError, match="each assignment must be a"):
        audit_nr_trajectory_target_support(("not-a-tuple",), (("train", "fam", "scalar"),))
    with pytest.raises(ValueError, match="assignment split and family IDs must be nonempty"):
        audit_nr_trajectory_target_support(
            ((" ", "fam", trajectory),), (("train", "fam", "scalar"),)
        )
    with pytest.raises(TypeError, match="assignment trajectory must be an NRTrajectory"):
        audit_nr_trajectory_target_support(
            (("train", "fam", "not-trajectory"),), (("train", "fam", "scalar"),)
        )


def test_audit_reports_trajectories_without_expected_coverage() -> None:
    trajectory = NRTrajectory("traj", (0.0,), (_record("a"),))
    with pytest.raises(ValueError, match="has no expected coverage cells"):
        audit_nr_trajectory_target_support(
            (("train", "fam", trajectory),), (("train", "other", "scalar"),)
        )
