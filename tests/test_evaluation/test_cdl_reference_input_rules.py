"""CDL spatiotemporal covariance references: shared carrier, lag, spacing and count rules."""

from __future__ import annotations

from typing import Any

import pytest
from evaluation.covariance.cdl import joint_tx_rx as joint
from evaluation.covariance.cdl import joint_tx_rx_dual_pol as dual_pol
from evaluation.covariance.cdl import joint_tx_rx_dual_tx_rx as dual_tx_rx
from evaluation.covariance.cdl import rotated_dual_tx_rx as rotated
from evaluation.covariance.cdl import spatiotemporal as single


def _reference(module: Any, name: str):
    return getattr(module, name)


CASES = [
    pytest.param(
        joint,
        "cdl_joint_tx_rx_spatiotemporal_covariance_reference",
        {"tx_count": 2, "rx_count": 2},
        id="joint",
    ),
    pytest.param(
        dual_tx_rx,
        "cdl_joint_tx_rx_dual_tx_rx_spatiotemporal_covariance_reference",
        {"tx_element_count": 2, "rx_element_count": 2},
        id="dual-tx-rx",
    ),
    pytest.param(
        dual_pol,
        "cdl_joint_tx_rx_dual_pol_spatiotemporal_covariance_reference",
        {"tx_count": 2, "rx_element_count": 2},
        id="dual-pol",
    ),
    pytest.param(
        single, "cdl_spatiotemporal_covariance_reference", {"element_count": 2}, id="single"
    ),
]


@pytest.mark.parametrize(("module", "name", "counts"), CASES)
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        pytest.param({"carrier_frequency_hz": 0.0}, "carrier must be positive", id="carrier-zero"),
        pytest.param({"lag_s": -1e-6}, "carrier must be positive", id="lag-negative"),
        pytest.param(
            {"element_spacing_wavelengths": -0.5}, "carrier must be positive", id="spacing-negative"
        ),
        pytest.param({"carrier_frequency_hz": float("nan")}, "must be finite", id="carrier-nan"),
    ],
)
def test_reference_checks_radio_parameters(module, name, counts, overrides, message) -> None:
    function = _reference(module, name)
    arguments: dict[str, Any] = {**counts, **overrides}
    with pytest.raises(ValueError, match=message):
        function("A", (0.0, 0.0, 0.0), **arguments)


@pytest.mark.parametrize(("module", "name", "counts"), CASES)
def test_reference_counts_must_be_positive_integers(module, name, counts) -> None:
    function = _reference(module, name)
    for key in counts:
        with pytest.raises(ValueError, match="positive integer"):
            function("A", (0.0, 0.0, 0.0), **{**counts, key: 0})
        with pytest.raises(ValueError, match="positive integer"):
            function("A", (0.0, 0.0, 0.0), **{**counts, key: True})


def test_rotated_reference_checks_orientations_and_radio_parameters() -> None:
    function = rotated.cdl_rotated_dual_tx_rx_spatiotemporal_covariance_reference
    with pytest.raises(ValueError, match="three finite Euler angles|orientation"):
        function(
            "A",
            (0.0, 0.0, 0.0),
            tx_orientation_rad=(0.0, 0.0),
            rx_orientation_rad=(0.0, 0.0, 0.0),
        )
    with pytest.raises(ValueError, match="carrier must be positive"):
        function(
            "A",
            (0.0, 0.0, 0.0),
            tx_orientation_rad=(0.0, 0.0, 0.0),
            rx_orientation_rad=(0.0, 0.0, 0.0),
            carrier_frequency_hz=0.0,
        )
