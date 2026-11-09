from __future__ import annotations

import math

import pytest
from evaluation.g4_cdl_joint_tx_rx_spatiotemporal_covariance import (
    cdl_joint_tx_rx_spatiotemporal_covariance_reference,
)
from evaluation.g4_cdl_spatiotemporal_covariance import cdl_spatiotemporal_covariance_reference


def test_zero_lag_reference_is_unit_power_hermitian() -> None:
    reference = cdl_joint_tx_rx_spatiotemporal_covariance_reference("B", (0.0, 0.0, 0.0), lag_s=0.0)
    assert len(reference) == 4
    for index in range(4):
        assert reference[index][index] == pytest.approx(1.0 + 0.0j, abs=1e-12)
        for column in range(4):
            assert reference[column][index] == pytest.approx(
                reference[index][column].conjugate(), abs=1e-12
            )


def test_single_tx_port_reduces_to_receive_only_reference() -> None:
    velocity = (10.0, 20.0, 20.0)
    joint = cdl_joint_tx_rx_spatiotemporal_covariance_reference(
        "D", velocity, tx_count=1, rx_count=2
    )
    receive_only = cdl_spatiotemporal_covariance_reference("D", velocity)
    for row in range(2):
        for column in range(2):
            assert joint[row][column] == pytest.approx(receive_only[row][column], abs=1e-12)


@pytest.mark.parametrize("model", tuple("ABCDE"))
def test_frozen_oblique_reference_is_finite(model: str) -> None:
    reference = cdl_joint_tx_rx_spatiotemporal_covariance_reference(model, (10.0, 20.0, 20.0))
    assert all(
        math.isfinite(value.real) and math.isfinite(value.imag)
        for row in reference
        for value in row
    )


@pytest.mark.parametrize(
    ("model", "velocity", "tx_count", "rx_count"),
    [
        ("F", (10.0, 20.0, 20.0), 2, 2),
        ("A", (10.0, 20.0), 2, 2),
        ("A", (10.0, 20.0, float("inf")), 2, 2),
        ("A", (10.0, 20.0, 20.0), 0, 2),
        ("A", (10.0, 20.0, 20.0), 2, 0),
    ],
)
def test_reference_rejects_out_of_contract_inputs(
    model: str, velocity: tuple[float, ...], tx_count: int, rx_count: int
) -> None:
    with pytest.raises((TypeError, ValueError)):
        cdl_joint_tx_rx_spatiotemporal_covariance_reference(
            model, velocity, tx_count=tx_count, rx_count=rx_count
        )
