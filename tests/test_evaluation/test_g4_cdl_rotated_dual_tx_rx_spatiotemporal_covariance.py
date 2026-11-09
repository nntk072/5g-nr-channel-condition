from __future__ import annotations

import math

import pytest
from evaluation.g4_cdl_joint_tx_rx_dual_tx_rx_spatiotemporal_covariance import (
    cdl_joint_tx_rx_dual_tx_rx_spatiotemporal_covariance_reference,
)
from evaluation.g4_cdl_rotated_dual_tx_rx_spatiotemporal_covariance import (
    cdl_rotated_dual_tx_rx_spatiotemporal_covariance_reference,
)

_ZERO = (0.0, 0.0, 0.0)
_RX_ROT = (0.37, 0.28, -0.41)
_TX_ROT = (-0.22, 0.31, 0.47)


@pytest.mark.parametrize("model", tuple("ABCDE"))
def test_zero_orientation_reduces_to_existing_dual_endpoint_reference(model: str) -> None:
    velocity = (10.0, 20.0, 20.0)
    rotated = cdl_rotated_dual_tx_rx_spatiotemporal_covariance_reference(
        model,
        velocity,
        tx_orientation_rad=_ZERO,
        rx_orientation_rad=_ZERO,
    )
    baseline = cdl_joint_tx_rx_dual_tx_rx_spatiotemporal_covariance_reference(model, velocity)
    assert len(rotated) == len(baseline) == 16
    for row in range(16):
        for column in range(16):
            assert rotated[row][column] == pytest.approx(baseline[row][column], abs=2e-12)


@pytest.mark.parametrize("model", tuple("ABCDE"))
def test_rotated_zero_lag_stationary_reference_is_hermitian(model: str) -> None:
    reference = cdl_rotated_dual_tx_rx_spatiotemporal_covariance_reference(
        model,
        (0.0, 0.0, 0.0),
        tx_orientation_rad=_TX_ROT,
        rx_orientation_rad=_RX_ROT,
        lag_s=0.0,
    )
    for row in range(len(reference)):
        assert reference[row][row].imag == pytest.approx(0.0, abs=2e-12)
        assert reference[row][row].real > 0.0
        for column in range(len(reference)):
            assert reference[row][column] == pytest.approx(
                reference[column][row].conjugate(), abs=2e-12
            )


def test_rotation_mixes_nlos_polarization_blocks() -> None:
    reference = cdl_rotated_dual_tx_rx_spatiotemporal_covariance_reference(
        "A",
        (0.0, 0.0, 0.0),
        tx_orientation_rad=_TX_ROT,
        rx_orientation_rad=_RX_ROT,
        lag_s=0.0,
    )
    # Channel indices 0 and 8 select different RX polarizations while keeping
    # the first TX V port. At zero orientation their covariance is zero; after
    # receive-basis rotation the diffuse Jones variances mix them.
    assert abs(reference[0][8]) > 1e-4


@pytest.mark.parametrize("model", tuple("ABCDE"))
def test_rotated_oblique_reference_is_finite(model: str) -> None:
    reference = cdl_rotated_dual_tx_rx_spatiotemporal_covariance_reference(
        model,
        (10.0, 20.0, 20.0),
        tx_orientation_rad=_TX_ROT,
        rx_orientation_rad=_RX_ROT,
    )
    assert all(
        math.isfinite(value.real) and math.isfinite(value.imag)
        for row in reference
        for value in row
    )


@pytest.mark.parametrize(
    ("model", "velocity", "tx_orientation", "rx_orientation", "tx_count", "rx_count"),
    [
        ("F", (10.0, 20.0, 20.0), _TX_ROT, _RX_ROT, 2, 2),
        ("A", (10.0, 20.0), _TX_ROT, _RX_ROT, 2, 2),
        ("A", (10.0, 20.0, float("inf")), _TX_ROT, _RX_ROT, 2, 2),
        ("A", (10.0, 20.0, 20.0), (0.1, 0.2), _RX_ROT, 2, 2),
        ("A", (10.0, 20.0, 20.0), _TX_ROT, (0.1, 0.2), 2, 2),
        ("A", (10.0, 20.0, 20.0), _TX_ROT, _RX_ROT, 0, 2),
        ("A", (10.0, 20.0, 20.0), _TX_ROT, _RX_ROT, 2, 0),
    ],
)
def test_reference_rejects_out_of_contract_inputs(
    model: str,
    velocity: tuple[float, ...],
    tx_orientation: tuple[float, ...],
    rx_orientation: tuple[float, ...],
    tx_count: int,
    rx_count: int,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        cdl_rotated_dual_tx_rx_spatiotemporal_covariance_reference(
            model,
            velocity,
            tx_orientation_rad=tx_orientation,
            rx_orientation_rad=rx_orientation,
            tx_element_count=tx_count,
            rx_element_count=rx_count,
        )
