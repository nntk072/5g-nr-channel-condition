"""Segment-mesh visibility: transform, point and tolerance admission rules."""

from __future__ import annotations

import numpy as np
import pytest
from propagation.visibility import segment_mesh_visibility

TRIANGLE = np.asarray([[[0.0, -1.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]]], dtype=np.float64)


@pytest.mark.parametrize(
    ("transform", "message"),
    [
        pytest.param(np.zeros((3, 3)), "finite 4x4 matrix", id="shape"),
        pytest.param(np.eye(4) * np.nan, "finite 4x4 matrix", id="nan"),
        pytest.param(
            np.vstack([np.eye(3, 4), [0.0, 0.0, 0.0, 2.0]]), "must be affine", id="projective"
        ),
        pytest.param(np.diag([0.0, 1.0, 1.0, 1.0]), "invertible linear part", id="singular"),
    ],
)
def test_transform_must_be_an_invertible_affine_map(transform, message) -> None:
    with pytest.raises(ValueError, match=message):
        segment_mesh_visibility((-1, 0, 0), (1, 0, 0), TRIANGLE, transform_to_world=transform)


@pytest.mark.parametrize(
    ("start", "end"),
    [
        pytest.param((0, 0), (1, 0, 0), id="start-shape"),
        pytest.param((0, 0, 0), (np.nan, 0, 0), id="end-nan"),
    ],
)
def test_segment_endpoints_must_be_finite_three_dimensional_points(start, end) -> None:
    with pytest.raises(ValueError, match="finite three-dimensional point"):
        segment_mesh_visibility(start, end, TRIANGLE)


def test_identity_transform_is_accepted() -> None:
    result = segment_mesh_visibility((-1, 2, 0), (1, 2, 0), TRIANGLE, transform_to_world=np.eye(4))
    assert result.outcome == "visible"
