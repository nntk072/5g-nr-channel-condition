from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from evaluation.spherical_angular_power import (
    global_to_element_local_directions,
    map_element_frame_spherical_power,
    map_spherical_power,
    spherical_cell_solid_angles,
)


def test_equal_solid_angle_quadrature_has_constant_full_sphere_density() -> None:
    zenith_edges = np.asarray([0.0, math.pi / 2.0, math.pi])
    azimuth_edges = np.linspace(-math.pi, math.pi, 5)
    areas = spherical_cell_solid_angles(zenith_edges, azimuth_edges)
    theta_centers = np.arccos((np.cos(zenith_edges[:-1]) + np.cos(zenith_edges[1:])) / 2.0)
    phi_centers = (azimuth_edges[:-1] + azimuth_edges[1:]) / 2.0
    theta, phi = np.meshgrid(theta_centers, phi_centers, indexing="ij")

    result = map_spherical_power(
        theta.ravel(),
        phi.ravel(),
        areas.ravel(),
        zenith_edges_rad=zenith_edges,
        azimuth_edges_rad=azimuth_edges,
    )

    fractions = np.asarray(result["integrated_power_fractions"])
    density = np.asarray(result["power_density_per_sr"])
    np.testing.assert_allclose(areas.sum(), 4.0 * math.pi, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(fractions, areas / (4.0 * math.pi), rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(density, 1.0 / (4.0 * math.pi), rtol=0.0, atol=1e-14)


def test_full_sphere_binning_handles_poles_and_wrapped_azimuth_boundary() -> None:
    result = map_spherical_power(
        [0.0, math.pi, math.pi / 2.0],
        [-math.pi, math.pi, 0.0],
        [1.0, 2.0, 3.0],
        zenith_edges_rad=[0.0, math.pi / 2.0, math.pi],
        azimuth_edges_rad=[-math.pi, 0.0, math.pi],
    )

    fractions = np.asarray(result["integrated_power_fractions"])
    assert fractions[0, 0] == pytest.approx(1.0 / 6.0)
    assert fractions[1, 0] == pytest.approx(2.0 / 6.0)
    assert fractions[1, 1] == pytest.approx(3.0 / 6.0)


def test_global_direction_is_inverse_rotated_into_element_frame() -> None:
    theta, phi = global_to_element_local_directions(
        np.asarray([math.pi / 2.0]),
        np.asarray([0.0]),
        orientation_rad=(math.pi / 2.0, 0.0, 0.0),
    )

    assert theta[0] == pytest.approx(math.pi / 2.0)
    assert phi[0] == pytest.approx(-math.pi / 2.0)


def test_element_frame_power_map_uses_inverse_orientation_and_preserves_power() -> None:
    zenith_edges = np.asarray([0.0, math.pi / 2.0, math.pi])
    azimuth_edges = np.linspace(-math.pi, math.pi, 5)
    result = map_element_frame_spherical_power(
        [math.pi / 2.0],
        [0.0],
        [2.5],
        orientation_rad=(math.pi / 2.0, 0.0, 0.0),
        zenith_edges_rad=zenith_edges,
        azimuth_edges_rad=azimuth_edges,
    )

    fractions = np.asarray(result["integrated_power_fractions"])
    assert result["schema"] == "umcd.element_frame_spherical_power_map.v1"
    assert result["direction_frame"] == "element_local"
    assert result["orientation_rad"] == [math.pi / 2.0, 0.0, 0.0]
    assert result["total_power"] == pytest.approx(2.5)
    assert fractions.sum() == pytest.approx(1.0)
    assert fractions[1, 1] == pytest.approx(1.0)


def test_element_frame_power_map_matches_versioned_json_schema() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    result = map_element_frame_spherical_power(
        [math.pi / 2.0],
        [0.0],
        [2.5],
        orientation_rad=(0.2, -0.1, 0.3),
        zenith_edges_rad=[0.0, math.pi / 2.0, math.pi],
        azimuth_edges_rad=np.linspace(-math.pi, math.pi, 5),
    )
    schema_path = (
        Path(__file__).parents[2]
        / "datasets"
        / "schema"
        / "element-frame-spherical-angular-power-map.v1.schema.json"
    )
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(result)


@pytest.mark.parametrize(
    ("zenith", "azimuth", "weights"),
    [
        ([True], [0.0], [1.0]),
        ([True, 0.5], [0.0, 0.0], [1.0, 1.0]),
        ([0.0], [False], [1.0]),
        ([0.0], [0.0], [True]),
        ([0.0], [0.0], [1.0 + 0.0j]),
        (["0.0"], [0.0], [1.0]),
    ],
)
def test_spherical_mapper_rejects_non_real_or_boolean_inputs(
    zenith: list[object], azimuth: list[object], weights: list[object]
) -> None:
    with pytest.raises(ValueError, match="real numeric"):
        map_spherical_power(
            zenith,
            azimuth,
            weights,
            zenith_edges_rad=[0.0, math.pi],
            azimuth_edges_rad=[-math.pi, math.pi],
        )


def test_direction_transform_rejects_boolean_orientation() -> None:
    with pytest.raises(ValueError, match="real numeric"):
        global_to_element_local_directions(
            [math.pi / 2.0],
            [0.0],
            orientation_rad=(True, 0.0, 0.0),
        )


@pytest.mark.parametrize(
    ("theta", "phi", "weights", "theta_edges", "phi_edges"),
    [
        ([0.0], [0.0], [1.0], [0.1, math.pi], [-math.pi, math.pi]),
        ([0.0], [0.0], [1.0], [0.0, math.pi], [-math.pi, 0.0]),
        ([0.0], [0.0], [-1.0], [0.0, math.pi], [-math.pi, math.pi]),
        ([math.nan], [0.0], [1.0], [0.0, math.pi], [-math.pi, math.pi]),
    ],
)
def test_spherical_power_map_rejects_invalid_grid_or_samples(
    theta: list[float],
    phi: list[float],
    weights: list[float],
    theta_edges: list[float],
    phi_edges: list[float],
) -> None:
    with pytest.raises(ValueError):
        map_spherical_power(
            theta,
            phi,
            weights,
            zenith_edges_rad=theta_edges,
            azimuth_edges_rad=phi_edges,
        )
