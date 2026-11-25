"""Patterned rotated dual-polarised CDL references: pattern, profile, velocity and count rules."""

from __future__ import annotations

import pytest
from evaluation.covariance.cdl.patterned_rotated_dual_tx_rx import (
    _pattern_power,
)
from evaluation.covariance.cdl.patterned_rotated_dual_tx_rx import (
    cdl_patterned_rotated_dual_tx_rx_spatiotemporal_covariance_reference as reference,
)

ORIENTATION = (0.0, 0.0, 0.0)
DIRECTION = (1.0, 0.0, 0.0)


def test_pattern_power_is_defined_for_omni_and_rejects_unknown_patterns() -> None:
    assert _pattern_power("omni", ORIENTATION, DIRECTION) == 1.0
    with pytest.raises(ValueError, match="unsupported antenna pattern"):
        _pattern_power("bogus", ORIENTATION, DIRECTION)


@pytest.mark.parametrize(
    ("model", "velocity", "options", "message"),
    [
        pytest.param("Z", (0.0, 0.0, 0.0), {}, "unsupported CDL profile", id="profile"),
        pytest.param("A", (1.0, 2.0), {}, "exactly three Cartesian", id="velocity"),
        pytest.param(
            "A",
            (0.0, 0.0, 0.0),
            {"tx_element_count": 0},
            "tx_element_count must be a positive integer",
            id="tx-count",
        ),
        pytest.param(
            "A",
            (0.0, 0.0, 0.0),
            {"rx_element_count": True},
            "rx_element_count must be a positive integer",
            id="rx-count",
        ),
        pytest.param(
            "A",
            (0.0, 0.0, 0.0),
            {"carrier_frequency_hz": 0.0},
            "carrier must be positive",
            id="carrier",
        ),
        pytest.param(
            "A",
            (0.0, 0.0, 0.0),
            {"tx_pattern": "bogus"},
            "must be supported TR 38.901 element patterns",
            id="pattern",
        ),
    ],
)
def test_reference_rejects_invalid_arguments(model, velocity, options, message) -> None:
    arguments = {
        "tx_pattern": "omni",
        "rx_pattern": "omni",
        "tx_orientation_rad": ORIENTATION,
        "rx_orientation_rad": ORIENTATION,
    }
    arguments.update(options)
    with pytest.raises(ValueError, match=message):
        reference(model, velocity, **arguments)
