"""Spherical angular power mapping: input, grid and orientation contracts."""

from __future__ import annotations

import math

import numpy as np
import pytest
from evaluation.covariance.spherical_angular_power import (
    global_to_element_local_directions,
    map_element_frame_spherical_power,
    map_spherical_power,
    spherical_cell_solid_angles,
)

FULL_ZENITH = np.array([0.0, math.pi / 2, math.pi])
FULL_AZIMUTH = np.array([-math.pi, 0.0, math.pi])


@pytest.mark.parametrize(
    ("orientation", "message"),
    [
        pytest.param([0.0, 0.0], "three finite Euler angles", id="two-angles"),
        pytest.param([0.0, float("nan"), 0.0], "only finite values", id="nan"),
        pytest.param([True, 0.0, 0.0], "boolean|finite|Euler", id="bool"),
    ],
)
def test_orientation_must_be_three_finite_angles(orientation, message) -> None:
    with pytest.raises((ValueError, TypeError), match=message):
        global_to_element_local_directions(
            np.array([0.5]), np.array([0.1]), orientation_rad=orientation
        )


def test_local_directions_require_equal_finite_shapes() -> None:
    with pytest.raises(ValueError, match="equal shapes"):
        global_to_element_local_directions(
            np.array([0.5, 0.6]), np.array([0.1]), orientation_rad=[0, 0, 0]
        )
    with pytest.raises(ValueError, match="zenith_rad must lie in"):
        global_to_element_local_directions(
            np.array([4.0]), np.array([0.1]), orientation_rad=[0, 0, 0]
        )


def test_identity_orientation_leaves_directions_unchanged() -> None:
    zenith, azimuth = global_to_element_local_directions(
        np.array([0.3, 1.2]), np.array([0.4, -0.7]), orientation_rad=[0.0, 0.0, 0.0]
    )
    np.testing.assert_allclose(zenith, [0.3, 1.2], atol=1e-12)
    np.testing.assert_allclose(azimuth, [0.4, -0.7], atol=1e-12)


def test_cell_solid_angles_form_the_full_sphere() -> None:
    angles = spherical_cell_solid_angles(np.array([0.0, math.pi / 2, math.pi]), FULL_AZIMUTH)
    assert angles.sum() == pytest.approx(4 * math.pi)


@pytest.mark.parametrize(
    ("zenith_edges", "azimuth_edges", "message"),
    [
        pytest.param([0.0], [-math.pi, math.pi], "at least two values", id="short-zenith"),
        pytest.param(
            [0.0, math.pi / 2, math.pi / 2],
            FULL_AZIMUTH,
            "zenith edges must span",
            id="zenith-order",
        ),
        pytest.param(
            FULL_ZENITH,
            [-math.pi, 0.0, math.pi / 2],
            "azimuth edges must span exactly 2\\*pi",
            id="azimuth-order",
        ),
        pytest.param(
            FULL_ZENITH,
            [0.0, math.pi / 2],
            "azimuth edges must span exactly 2\\*pi",
            id="partial-azimuth",
        ),
    ],
)
def test_angular_grid_must_be_a_full_sphere(zenith_edges, azimuth_edges, message) -> None:
    with pytest.raises(ValueError, match=message):
        spherical_cell_solid_angles(
            np.asarray(zenith_edges, float), np.asarray(azimuth_edges, float)
        )


@pytest.mark.parametrize(
    ("zenith", "azimuth", "weights", "message"),
    [
        pytest.param(
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            np.ones((2, 2)),
            "equal-length nonempty vectors",
            id="matrix",
        ),
        pytest.param(
            np.array([0.5]),
            np.array([0.1, 0.2]),
            np.array([1.0]),
            "equal-length nonempty vectors",
            id="length",
        ),
        pytest.param(
            np.array([0.5]),
            np.array([0.1]),
            np.array([1.0, 2.0]),
            "power_weights must match",
            id="weights-length",
        ),
        pytest.param(
            np.array([4.0]), np.array([0.1]), np.array([1.0]), "on-sphere", id="off-sphere"
        ),
        pytest.param(
            np.array([0.5]), np.array([0.1]), np.array([-1.0]), "nonnegative", id="negative-power"
        ),
        pytest.param(
            np.array([0.5]),
            np.array([0.1]),
            np.array([0.0]),
            "finite positive total",
            id="zero-power",
        ),
    ],
)
def test_power_mapping_validates_directions_and_weights(zenith, azimuth, weights, message) -> None:
    with pytest.raises(ValueError, match=message):
        map_spherical_power(
            zenith,
            azimuth,
            weights,
            zenith_edges_rad=FULL_ZENITH,
            azimuth_edges_rad=FULL_AZIMUTH,
        )


def test_power_mapping_reports_density_per_steradian() -> None:
    zenith = np.array([0.3, 1.2, 2.5])
    azimuth = np.array([0.1, -0.5, 2.0])
    result = map_spherical_power(
        zenith,
        azimuth,
        np.array([1.0, 2.0, 3.0]),
        zenith_edges_rad=FULL_ZENITH,
        azimuth_edges_rad=FULL_AZIMUTH,
    )
    assert result is not None


def test_element_frame_mapping_composes_orientation_and_binning() -> None:
    result = map_element_frame_spherical_power(
        np.array([0.3, 1.2]),
        np.array([0.1, -0.5]),
        np.array([1.0, 2.0]),
        orientation_rad=[0.0, 0.0, 0.0],
        zenith_edges_rad=FULL_ZENITH,
        azimuth_edges_rad=FULL_AZIMUTH,
    )
    assert result is not None
