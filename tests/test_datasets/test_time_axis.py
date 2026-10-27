import numpy as np
import pytest
from datasets.network import NodePoseTimeline
from datasets.time_axis import align_time_axes, canonical_time_axis


def test_time_axis_accepts_numpy_real_scalars_and_normalizes_to_float():
    assert canonical_time_axis((np.int64(0), np.float64(0.5)), name="times") == (0.0, 0.5)


@pytest.mark.parametrize("value", ["0.5", 0.5 + 0j, True])
def test_time_axis_rejects_coercible_nonreal_values(value):
    with pytest.raises(TypeError, match="real numeric timestamps"):
        canonical_time_axis((value,), name="times")


@pytest.mark.parametrize("values", [(), (0.0, 0.0), (0.0, float("inf"))])
def test_time_axis_requires_nonempty_finite_strictly_increasing_values(values):
    with pytest.raises(ValueError):
        canonical_time_axis(values, name="times")


def test_time_axis_alignment_reports_matches_residuals_and_unmatched_rows():
    alignment = align_time_axes(
        (0.0, 1.0, 2.0),
        (0.0001, 1.0002, 3.0),
        tolerance_s=0.001,
    )

    assert alignment.observed_indices == (0, 1, None)
    assert alignment.residuals_s == pytest.approx((0.0001, 0.0002, None))
    assert alignment.unmatched_observed_indices == (2,)
    assert alignment.complete is False
    assert alignment.to_dict()["method"] == "unique_match_within_absolute_tolerance_no_interpolation"


def test_time_axis_alignment_requires_unique_one_to_one_matches():
    with pytest.raises(ValueError, match="matches multiple observed samples"):
        align_time_axes((1.0,), (0.99, 1.01), tolerance_s=0.02)
    with pytest.raises(ValueError, match="cannot align to multiple reference samples"):
        align_time_axes((0.0, 0.1), (0.05,), tolerance_s=0.06)
    with pytest.raises(ValueError, match="missing reference samples"):
        align_time_axes((0.0, 1.0), (0.0,), tolerance_s=0.0, require_complete=True)


def test_time_axis_alignment_rejects_invalid_tolerance_and_completion_flag():
    with pytest.raises(TypeError, match="tolerance_s must be a real number"):
        align_time_axes((0.0,), (0.0,), tolerance_s=True)
    with pytest.raises(ValueError, match="finite and non-negative"):
        align_time_axes((0.0,), (0.0,), tolerance_s=-1.0)
    with pytest.raises(TypeError, match="require_complete must be a boolean"):
        align_time_axes((0.0,), (0.0,), tolerance_s=0.0, require_complete=1)


def test_pose_timeline_aligns_on_reference_grid_and_returns_provenance():
    positions = np.asarray([[[0.0, 0.0, 1.0]], [[1.0, 0.0, 1.0]], [[2.0, 0.0, 1.0]]])
    orientations = np.broadcast_to(np.eye(3), (3, 1, 3, 3)).copy()
    poses = NodePoseTimeline(
        node_ids=("rx",),
        frame_times_s=(0.0, 1.0001, 2.0),
        positions_world_m=positions,
        orientations_local_to_world=orientations,
    )

    aligned, report = poses.align_to((0.0, 1.0, 2.0), tolerance_s=0.001)

    assert aligned.frame_times_s == (0.0, 1.0, 2.0)
    assert aligned.position("rx", 1).tolist() == [1.0, 0.0, 1.0]
    assert report.observed_indices == (0, 1, 2)
    assert report.residuals_s == pytest.approx((0.0, 0.0001, 0.0))
