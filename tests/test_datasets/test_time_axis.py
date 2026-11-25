import numpy as np
import pytest
from datasets.network import NodePoseTimeline
from datasets.time_axis import (
    MultiSourceTimeAlignment,
    TimeAxisAlignment,
    align_named_time_axes,
    align_time_axes,
    canonical_time_axis,
)


def test_time_axis_accepts_numpy_real_scalars_and_normalizes_to_float():
    assert canonical_time_axis((np.int64(0), np.float64(0.5)), name="times") == (0.0, 0.5)


def test_time_axis_rejects_noniterable_and_unrepresentable_real_values():
    class OverflowingReal(float):
        def __float__(self):
            raise OverflowError("too large")

    with pytest.raises(TypeError, match="must be an iterable"):
        canonical_time_axis(1.0, name="times")
    with pytest.raises(ValueError, match="finite real timestamps"):
        canonical_time_axis((OverflowingReal(1.0),), name="times")


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
    assert (
        alignment.to_dict()["method"] == "unique_match_within_absolute_tolerance_no_interpolation"
    )


def test_time_axis_alignment_report_replays_and_rejects_claim_drift():
    report = align_time_axes((0.0, 1.0), (0.0, 1.0001), tolerance_s=0.001).to_dict()
    assert TimeAxisAlignment.from_dict(report).complete
    changed_residual = {**report, "residuals_s": [0.0, 0.0]}
    with pytest.raises(ValueError, match="differs from deterministic replay"):
        TimeAxisAlignment.from_dict(changed_residual)

    boolean_alias = {**report, "complete": 1}
    with pytest.raises(ValueError, match="differs from deterministic replay"):
        TimeAxisAlignment.from_dict(boolean_alias)


def test_time_axis_alignment_report_rejects_invalid_contract_fields():
    report = align_time_axes((0.0,), (0.0,), tolerance_s=0.0).to_dict()
    with pytest.raises(ValueError, match="fields do not match"):
        TimeAxisAlignment.from_dict({**report, "extra": True})
    with pytest.raises(ValueError, match="unsupported time-axis alignment schema"):
        TimeAxisAlignment.from_dict({**report, "schema": "future"})
    with pytest.raises(ValueError, match="unsupported time-axis alignment method"):
        TimeAxisAlignment.from_dict({**report, "method": "interpolate"})
    with pytest.raises(TypeError, match="tolerance_s must be real"):
        TimeAxisAlignment.from_dict({**report, "tolerance_s": True})


def test_named_source_alignment_reports_each_clock_without_interpolation():
    report = align_named_time_axes(
        (0.0, 1.0, 2.0),
        {
            "tx": (0.0, 1.0, 2.0),
            "rx": (0.001, 1.001, 3.0),
            "object-7": (0.0, 2.0),
        },
        tolerance_s=0.002,
    )

    assert tuple(source_id for source_id, _ in report.source_alignments) == (
        "object-7",
        "rx",
        "tx",
    )
    assert report.complete is False
    assert dict(report.source_alignments)["rx"].observed_indices == (0, 1, None)
    assert dict(report.source_alignments)["rx"].unmatched_observed_indices == (2,)
    assert dict(report.source_alignments)["object-7"].observed_indices == (0, None, 1)


def test_named_source_alignment_replays_and_rejects_source_or_match_drift():
    report = align_named_time_axes(
        (0.0, 1.0), {"rx": (0.0, 1.0), "tx": (0.0, 1.0)}, tolerance_s=0.0
    )
    assert MultiSourceTimeAlignment.from_dict(report.to_dict()) == report
    changed = report.to_dict()
    changed["sources"][0]["alignment"]["observed_indices"] = [1, 0]
    with pytest.raises(ValueError, match="deterministic replay"):
        MultiSourceTimeAlignment.from_dict(changed)


def test_named_source_alignment_rejects_invalid_report_shapes_and_claims():
    report = align_named_time_axes((0.0,), {"rx": (0.0,)}, tolerance_s=0.0).to_dict()
    with pytest.raises(ValueError, match="fields do not match"):
        MultiSourceTimeAlignment.from_dict({**report, "extra": True})
    with pytest.raises(ValueError, match="unsupported multi-source"):
        MultiSourceTimeAlignment.from_dict({**report, "schema": "future"})
    with pytest.raises(ValueError, match="non-empty list"):
        MultiSourceTimeAlignment.from_dict({**report, "sources": []})
    malformed = {**report, "sources": [{"source_id": "rx"}]}
    with pytest.raises(ValueError, match="invalid field roster"):
        MultiSourceTimeAlignment.from_dict(malformed)
    changed = {**report, "complete": False}
    with pytest.raises(ValueError, match="deterministic replay"):
        MultiSourceTimeAlignment.from_dict(changed)


@pytest.mark.parametrize(
    ("rows", "error", "message"),
    [
        ((), ValueError, "at least one named source"),
        ((("rx",),), TypeError, "entries must be"),
        (((" rx ", align_time_axes((0.0,), (0.0,), tolerance_s=0.0)),), ValueError, "canonical strings"),
        ((("rx", object()),), TypeError, "must be a TimeAxisAlignment"),
        ((("rx", align_time_axes((1.0,), (1.0,), tolerance_s=0.0)),), ValueError, "shared reference axis"),
        (
            (
                ("rx", align_time_axes((0.0,), (0.0,), tolerance_s=0.0)),
                ("rx", align_time_axes((0.0,), (0.0,), tolerance_s=0.0)),
            ),
            ValueError,
            "must be unique",
        ),
    ],
)
def test_multi_source_alignment_rejects_invalid_source_entries(rows, error, message):
    with pytest.raises(error, match=message):
        MultiSourceTimeAlignment((0.0,), rows)


def test_named_source_alignment_rejects_invalid_source_mappings():
    with pytest.raises(ValueError, match="non-empty source mapping"):
        align_named_time_axes((0.0,), {}, tolerance_s=0.0)
    with pytest.raises(ValueError, match="non-empty source mapping"):
        align_named_time_axes((0.0,), [], tolerance_s=0.0)
    with pytest.raises(TypeError, match="keys must be source ID strings"):
        align_named_time_axes((0.0,), {1: (0.0,)}, tolerance_s=0.0)
    assert len(align_named_time_axes((0.0,), {"rx": (0.0,)}, tolerance_s=0.0).sha256) == 64


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
