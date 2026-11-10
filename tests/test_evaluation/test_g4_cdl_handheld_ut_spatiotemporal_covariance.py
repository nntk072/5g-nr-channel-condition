"""Portable checks for the physical handheld-UT CDL covariance reference."""

from __future__ import annotations

import math

import pytest
from evaluation.g4_cdl_handheld_ut_spatiotemporal_covariance import (
    cdl_handheld_ut_dual_spatiotemporal_covariance_reference,
    cdl_handheld_ut_spatiotemporal_covariance_reference,
    handheld_ut_dual_geometry,
    handheld_ut_single_v_field_gcs,
    handheld_ut_single_v_geometry,
)
from evaluation.g4_cdl_patterned_rotated_dual_tx_rx_spatiotemporal_covariance import (
    _local_axes,
)

_ORIENTATION = (0.2, -0.1, 0.3)
_VELOCITY = (10.0, 20.0, 20.0)


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_handheld_four_corner_geometry_matches_tr38901_device_dimensions() -> None:
    positions, bases = handheld_ut_single_v_geometry("tr38901-4")
    expected = (
        (-0.075, -0.035, 0.0),
        (-0.075, 0.035, 0.0),
        (0.075, -0.035, 0.0),
        (0.075, 0.035, 0.0),
    )
    for observed, target in zip(positions, expected, strict=True):
        assert observed == pytest.approx(target)
    assert len(bases) == 4


@pytest.mark.parametrize("roster", ("tr38901-4", "tr38901"))
def test_handheld_port_bases_are_right_handed_orthonormal(roster: str) -> None:
    _, bases = handheld_ut_single_v_geometry(roster)
    for x_axis, y_axis, z_axis in bases:
        assert _dot(x_axis, x_axis) == pytest.approx(1.0, abs=1e-15)
        assert _dot(y_axis, y_axis) == pytest.approx(1.0, abs=1e-15)
        assert _dot(z_axis, z_axis) == pytest.approx(1.0, abs=1e-15)
        assert _dot(x_axis, y_axis) == pytest.approx(0.0, abs=1e-15)
        assert _dot(x_axis, z_axis) == pytest.approx(0.0, abs=1e-15)
        assert _dot(y_axis, z_axis) == pytest.approx(0.0, abs=1e-15)


def test_handheld_port_boresight_field_has_table_7_3_2_gain() -> None:
    _, bases = handheld_ut_single_v_geometry((1,))
    endpoint_axes = _local_axes(_ORIENTATION)
    boresight_device = bases[0][0]
    boresight_gcs = tuple(
        sum(endpoint_axes[column][row] * boresight_device[column] for column in range(3))
        for row in range(3)
    )
    field = handheld_ut_single_v_field_gcs(
        boresight_gcs,
        endpoint_orientation_rad=_ORIENTATION,
        port_basis_device=bases[0],
    )
    power = field[0] ** 2 + field[1] ** 2
    assert power == pytest.approx(10.0 ** (5.3 / 10.0), rel=1e-12)


@pytest.mark.parametrize(("roster", "port_count"), (("tr38901-4", 4), ("tr38901", 8)))
def test_handheld_covariance_reference_has_expected_shape_and_finite_values(
    roster: str,
    port_count: int,
) -> None:
    matrix = cdl_handheld_ut_spatiotemporal_covariance_reference(
        "D",
        _VELOCITY,
        antenna_locations=roster,
        ut_orientation_rad=_ORIENTATION,
    )
    assert len(matrix) == port_count
    assert all(len(row) == port_count for row in matrix)
    assert all(
        math.isfinite(value.real) and math.isfinite(value.imag) for row in matrix for value in row
    )


@pytest.mark.parametrize("bad_roster", ("", "corners", (1, 1), (0, 2)))
def test_handheld_reference_rejects_invalid_rosters(bad_roster) -> None:
    with pytest.raises(ValueError):
        cdl_handheld_ut_spatiotemporal_covariance_reference(
            "A",
            _VELOCITY,
            antenna_locations=bad_roster,
            ut_orientation_rad=_ORIENTATION,
        )


@pytest.mark.parametrize("polarization_type", ("VH", "cross"))
@pytest.mark.parametrize(("roster", "location_count"), (("tr38901-4", 4), ("tr38901", 8)))
def test_dual_handheld_geometry_duplicates_locations_and_orders_components(
    polarization_type: str, roster: str, location_count: int
) -> None:
    positions, bases, components = handheld_ut_dual_geometry(
        roster, polarization_type=polarization_type
    )
    assert len(positions) == 2 * location_count
    assert len(bases) == 2 * location_count
    assert positions[:location_count] == positions[location_count:]
    assert bases[:location_count] == bases[location_count:]
    assert components == tuple([0] * location_count + [1] * location_count)


@pytest.mark.parametrize(("roster", "port_count"), (("tr38901-4", 8), ("tr38901", 16)))
def test_dual_handheld_covariance_reference_has_expected_shape_and_finite_values(
    roster: str, port_count: int
) -> None:
    matrix = cdl_handheld_ut_dual_spatiotemporal_covariance_reference(
        "D",
        _VELOCITY,
        antenna_locations=roster,
        ut_orientation_rad=_ORIENTATION,
        polarization_type="cross",
    )
    assert len(matrix) == port_count
    assert all(len(row) == port_count for row in matrix)
    assert all(
        math.isfinite(value.real) and math.isfinite(value.imag) for row in matrix for value in row
    )


def test_dual_handheld_reference_rejects_invalid_polarization_type() -> None:
    with pytest.raises(ValueError, match="polarization_type"):
        cdl_handheld_ut_dual_spatiotemporal_covariance_reference(
            "A",
            _VELOCITY,
            antenna_locations="tr38901-4",
            ut_orientation_rad=_ORIENTATION,
            polarization_type="slant",
        )
