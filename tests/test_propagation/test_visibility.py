from __future__ import annotations

import numpy as np
import pytest
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
    translate[0, 3] = 2.0
    shifted = segment_mesh_visibility(
        (0, 0, 0), (4, 0, 0), TRIANGLE,
        transform_to_world=translate,
        meters_per_unit=0.5,
    )

    assert shifted.outcome == "blocked"
    assert shifted.segment_length_m == pytest.approx(2.0)


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
