from __future__ import annotations

import numpy as np
import pytest
from propagation import visibility
from propagation.visibility import segment_mesh_visibility

TRIANGLE = np.asarray(
    [[[0.0, -1.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]]],
    dtype=np.float64,
)


def test_segment_visibility_distinguishes_clear_and_triangle_interior_hit():
    clear = segment_mesh_visibility((-1, 2, 0), (1, 2, 0), TRIANGLE)
    blocked = segment_mesh_visibility((-1, 0, 0), (1, 0, 0), TRIANGLE)

    assert clear.outcome == "visible"
    assert clear.intersected_triangle_indices == ()
    assert blocked.outcome == "blocked"
    assert blocked.intersected_triangle_indices == (0,)


def test_segment_visibility_marks_edge_endpoint_and_coplanar_contact_ambiguous():
    edge = segment_mesh_visibility((-1, 0, 1), (1, 0, 1), TRIANGLE)
    endpoint = segment_mesh_visibility((0, 0, 0), (1, 2, 0), TRIANGLE)
    coplanar = segment_mesh_visibility((0, -2, 0), (0, 2, 0), TRIANGLE)

    assert edge.outcome == "ambiguous"
    assert endpoint.outcome == "ambiguous"
    assert coplanar.outcome == "ambiguous"
    assert endpoint.ambiguous_triangle_indices == (0,)


def test_segment_visibility_applies_affine_transform_and_unit_scale():
    translate = np.eye(4)
    translate[:3, :3] = np.asarray(
        [[2.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.5]], dtype=np.float64
    )
    translate[0, 3] = 2.0
    shifted = segment_mesh_visibility(
        (0, 0, 0),
        (4, 0, 0),
        TRIANGLE,
        transform_to_world=translate,
        meters_per_unit=0.5,
    )

    assert shifted.outcome == "blocked"
    assert shifted.segment_length_m == pytest.approx(2.0)


def test_segment_visibility_rejects_coincident_and_tolerance_short_segments():
    with pytest.raises(ValueError, match="separated by more than tolerance"):
        segment_mesh_visibility((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), TRIANGLE)
    with pytest.raises(ValueError, match="separated by more than tolerance"):
        segment_mesh_visibility((0.0, 0.0, 0.0), (0.5e-8, 0.0, 0.0), TRIANGLE, tolerance_m=1e-8)


def test_nearly_grazing_crossing_and_degenerate_face_are_ambiguous():
    broad_triangle = np.asarray(
        [[[0.0, -100.0, -100.0], [0.0, 100.0, -100.0], [0.0, 0.0, 100.0]]],
        dtype=np.float64,
    )
    grazing = segment_mesh_visibility(
        (-5e-9, -1.0, 0.0), (5e-9, 1.0, 0.0), broad_triangle, tolerance_m=1e-8
    )
    degenerate_triangle = np.asarray(
        [[[0.0, -1e-10, 0.0], [0.0, 1e-10, 0.0], [0.0, 0.0, 1e-10]]],
        dtype=np.float64,
    )
    degenerate = segment_mesh_visibility((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), degenerate_triangle)

    assert grazing.outcome == "ambiguous"
    assert grazing.ambiguous_triangle_indices == (0,)
    assert degenerate.outcome == "ambiguous"
    assert degenerate.ambiguous_triangle_indices == (0,)


def test_adding_and_removing_blocker_changes_geometry_visibility():
    clear = segment_mesh_visibility((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), np.empty((0, 3, 3)))
    blocked = segment_mesh_visibility((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), TRIANGLE)
    reopened = segment_mesh_visibility((-1.0, 2.0, 0.0), (1.0, 2.0, 0.0), TRIANGLE)

    assert clear.outcome == "visible"
    assert blocked.outcome == "blocked"
    assert reopened.outcome == "visible"


def test_zero_thickness_wall_is_geometrically_blocking_without_rf_claims():
    wall = segment_mesh_visibility((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), TRIANGLE)

    assert wall.outcome == "blocked"
    # This API takes only endpoints and geometry; transmission is outside its model.
    assert not hasattr(wall, "transmission_coefficient")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"triangles": np.zeros((3, 3))},
        {"triangles": np.full((1, 3, 3), np.nan)},
        {"triangles": TRIANGLE, "meters_per_unit": 0.0},
        {"triangles": TRIANGLE, "tolerance_m": -1.0},
        {"triangles": TRIANGLE, "transform_to_world": np.zeros((4, 4))},
    ],
)
def test_segment_visibility_rejects_malformed_geometry(kwargs):
    call = {"start": (-1, 0, 0), "end": (1, 0, 0), **kwargs}
    with pytest.raises(ValueError):
        segment_mesh_visibility(**call)


def test_triangle_geometry_helpers_cover_degenerate_and_outside_cases():
    degenerate = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    assert visibility._barycentric(np.zeros(3), degenerate) is None
    assert not visibility._point_in_triangle(np.zeros(3), degenerate, 1e-8)
    assert (
        visibility._intersect_open_segment_triangle(
            np.asarray([-1.0, 0.0, 0.0]),
            np.asarray([2.0, 0.0, 0.0]),
            degenerate,
            tolerance_m=1e-8,
        )
        == "ambiguous"
    )
    numerically_degenerate = np.asarray([[0.0, 0.0, 0.0], [1e-6, 0.0, 0.0], [0.0, 1e-6, 0.0]])
    assert (
        visibility._intersect_open_segment_triangle(
            np.asarray([0.0, 0.0, -1.0]),
            np.asarray([0.0, 0.0, 2.0]),
            numerically_degenerate,
            tolerance_m=1e-8,
        )
        == "ambiguous"
    )


def test_coplanar_segment_without_contact_is_visible_and_collinear_edge_is_ambiguous():
    planar_triangle = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
    )
    assert not visibility._coplanar_segment_overlaps_triangle(
        np.asarray([2.0, 2.0, 0.0]), np.asarray([1.0, 0.0, 0.0]), planar_triangle, 1e-8
    )
    assert visibility._coplanar_segment_overlaps_triangle(
        np.asarray([-1.0, 0.0, 0.0]), np.asarray([3.0, 0.0, 0.0]), planar_triangle, 1e-8
    )
    clear = segment_mesh_visibility((2, 2, 0), (3, 2, 0), planar_triangle.reshape(1, 3, 3))
    assert clear.outcome == "visible"


def test_endpoint_contact_outside_triangle_is_not_an_intersection():
    classification = visibility._intersect_open_segment_triangle(
        np.asarray([0.0, 2.0, 0.0]),
        np.asarray([1.0, 0.0, 0.0]),
        TRIANGLE[0],
        tolerance_m=1e-8,
    )
    assert classification == "none"
    beyond = visibility._intersect_open_segment_triangle(
        np.asarray([-1.0, 0.0, 0.0]),
        np.asarray([0.5, 0.0, 0.0]),
        TRIANGLE[0],
        tolerance_m=1e-8,
    )
    assert beyond == "none"


def test_near_parallel_plane_crossing_reports_ambiguity_only_near_the_face():
    broad_triangle = np.asarray([[0.0, -100.0, -100.0], [0.0, 100.0, -100.0], [0.0, 0.0, 100.0]])
    near = visibility._intersect_open_segment_triangle(
        np.asarray([-2e-9, -1.0, 0.0]),
        np.asarray([1.3e-8, 2.0, 0.0]),
        broad_triangle,
        tolerance_m=1e-8,
    )
    far = visibility._intersect_open_segment_triangle(
        np.asarray([-2e-9, -1000.0, 0.0]),
        np.asarray([1.3e-8, 2000.0, 0.0]),
        broad_triangle,
        tolerance_m=1e-8,
    )
    outside_parameter = visibility._intersect_open_segment_triangle(
        np.asarray([2e-8, -1.0, 0.0]),
        np.asarray([1e-8, 2.0, 0.0]),
        broad_triangle,
        tolerance_m=1e-8,
    )
    assert near == "ambiguous"
    assert far == "none"
    assert outside_parameter == "none"


def test_coplanar_overlap_skips_zero_length_triangle_edge():
    triangle_with_duplicate_vertex = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert not visibility._coplanar_segment_overlaps_triangle(
        np.asarray([2.0, 2.0, 0.0]),
        np.asarray([1.0, 0.0, 0.0]),
        triangle_with_duplicate_vertex,
        1e-8,
    )


def test_parallel_plane_without_crossing_is_clear_and_disjoint_collinear_edge_is_clear():
    parallel = visibility._intersect_open_segment_triangle(
        np.asarray([1.0, -1.0, 0.0]),
        np.asarray([0.0, 2.0, 0.0]),
        TRIANGLE[0],
        tolerance_m=1e-8,
    )
    planar_triangle = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
    )
    disjoint_collinear = visibility._coplanar_segment_overlaps_triangle(
        np.asarray([2.0, 0.0, 0.0]),
        np.asarray([1.0, 0.0, 0.0]),
        planar_triangle,
        1e-8,
    )
    assert parallel == "none"
    assert not disjoint_collinear
