"""Portable checks for the patterned rotated CDL dual-endpoint covariance oracle."""

from __future__ import annotations

import math

import pytest
from evaluation.g4_cdl_patterned_rotated_dual_tx_rx_spatiotemporal_covariance import (
    _local_axes,
    _pattern_power,
    cdl_patterned_rotated_dual_tx_rx_spatiotemporal_covariance_reference,
)
from evaluation.g4_cdl_rotated_dual_tx_rx_spatiotemporal_covariance import (
    cdl_rotated_dual_tx_rx_spatiotemporal_covariance_reference,
)

_ORIENTATION = ((-0.22, 0.31, 0.47), (0.37, 0.28, -0.41))
_VELOCITY = (10.0, 20.0, 20.0)


@pytest.mark.parametrize(("pattern", "gain_db"), (("38.901", 8.0), ("38.901-handheld", 5.3)))
def test_pattern_boresight_matches_table_gain(pattern: str, gain_db: float) -> None:
    orientation = (0.41, -0.27, 0.19)
    local_x, _, _ = _local_axes(orientation)
    assert _pattern_power(pattern, orientation, local_x) == pytest.approx(10.0 ** (gain_db / 10.0))


def test_pattern_omni_is_unit_power() -> None:
    assert _pattern_power("omni", (0.4, -0.2, 0.7), (0.3, 0.4, math.sqrt(0.75))) == 1.0


@pytest.mark.parametrize("model", tuple("ABCDE"))
def test_omni_patterned_oracle_reduces_to_frozen_omni_oracle(model: str) -> None:
    tx_orientation, rx_orientation = _ORIENTATION
    patterned = cdl_patterned_rotated_dual_tx_rx_spatiotemporal_covariance_reference(
        model,
        _VELOCITY,
        tx_pattern="omni",
        rx_pattern="omni",
        tx_orientation_rad=tx_orientation,
        rx_orientation_rad=rx_orientation,
    )
    frozen = cdl_rotated_dual_tx_rx_spatiotemporal_covariance_reference(
        model,
        _VELOCITY,
        tx_orientation_rad=tx_orientation,
        rx_orientation_rad=rx_orientation,
    )
    for patterned_row, frozen_row in zip(patterned, frozen, strict=True):
        for patterned_value, frozen_value in zip(patterned_row, frozen_row, strict=True):
            assert patterned_value == pytest.approx(frozen_value, abs=1e-15, rel=1e-15)


def test_patterned_reference_has_expected_shape_and_finite_values() -> None:
    tx_orientation, rx_orientation = _ORIENTATION
    matrix = cdl_patterned_rotated_dual_tx_rx_spatiotemporal_covariance_reference(
        "D",
        _VELOCITY,
        tx_pattern="38.901",
        rx_pattern="38.901-handheld",
        tx_orientation_rad=tx_orientation,
        rx_orientation_rad=rx_orientation,
    )
    assert len(matrix) == 16
    assert all(len(row) == 16 for row in matrix)
    assert all(math.isfinite(value.real) and math.isfinite(value.imag) for row in matrix for value in row)


@pytest.mark.parametrize("bad_pattern", ("", "invalid", "38.902"))
def test_patterned_reference_rejects_unknown_patterns(bad_pattern: str) -> None:
    tx_orientation, rx_orientation = _ORIENTATION
    with pytest.raises(ValueError, match="supported TR 38.901 element patterns"):
        cdl_patterned_rotated_dual_tx_rx_spatiotemporal_covariance_reference(
            "A",
            _VELOCITY,
            tx_pattern=bad_pattern,
            rx_pattern="38.901",
            tx_orientation_rad=tx_orientation,
            rx_orientation_rad=rx_orientation,
        )
