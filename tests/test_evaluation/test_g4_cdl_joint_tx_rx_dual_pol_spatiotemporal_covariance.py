from __future__ import annotations

import math

import pytest
from evaluation.g4_cdl_joint_tx_rx_dual_pol_spatiotemporal_covariance import (
    cdl_joint_tx_rx_dual_pol_spatiotemporal_covariance_reference,
)
from evaluation.g4_cdl_joint_tx_rx_spatiotemporal_covariance import (
    cdl_joint_tx_rx_spatiotemporal_covariance_reference,
)


@pytest.mark.parametrize("model", tuple("ABCDE"))
def test_vertical_block_reduces_to_existing_single_v_reference(model: str) -> None:
    velocity = (10.0, 20.0, 20.0)
    dual = cdl_joint_tx_rx_dual_pol_spatiotemporal_covariance_reference(model, velocity)
    single = cdl_joint_tx_rx_spatiotemporal_covariance_reference(model, velocity)
    assert len(dual) == 8
    assert all(len(row) == 8 for row in dual)
    for row in range(4):
        for column in range(4):
            assert dual[row][column] == pytest.approx(single[row][column], abs=1e-12)


@pytest.mark.parametrize("model", tuple("ABCDE"))
def test_cross_polarization_blocks_are_zero_and_horizontal_power_is_bounded(model: str) -> None:
    reference = cdl_joint_tx_rx_dual_pol_spatiotemporal_covariance_reference(
        model, (0.0, 0.0, 0.0), lag_s=0.0
    )
    for index in range(4):
        assert reference[index][index] == pytest.approx(1.0 + 0.0j, abs=1e-12)
        assert 0.0 < reference[4 + index][4 + index].real < 1.0
        assert reference[4 + index][4 + index].imag == pytest.approx(0.0, abs=1e-12)
    for row in range(4):
        for column in range(4, 8):
            assert reference[row][column] == 0.0j
            assert reference[column][row] == 0.0j


@pytest.mark.parametrize("model", tuple("ABC"))
def test_nlos_horizontal_block_is_vertical_block_scaled_by_xpr(model: str) -> None:
    xpr_db = {"A": 10.0, "B": 8.0, "C": 7.0}[model]
    scale = 10.0 ** (-xpr_db / 10.0)
    reference = cdl_joint_tx_rx_dual_pol_spatiotemporal_covariance_reference(
        model, (10.0, 20.0, 20.0)
    )
    for row in range(4):
        for column in range(4):
            assert reference[4 + row][4 + column] == pytest.approx(
                scale * reference[row][column], abs=1e-12
            )


@pytest.mark.parametrize("model", tuple("ABCDE"))
def test_frozen_oblique_reference_is_finite(model: str) -> None:
    reference = cdl_joint_tx_rx_dual_pol_spatiotemporal_covariance_reference(
        model, (10.0, 20.0, 20.0)
    )
    assert all(
        math.isfinite(value.real) and math.isfinite(value.imag)
        for row in reference
        for value in row
    )


@pytest.mark.parametrize(
    ("model", "velocity", "tx_count", "rx_element_count"),
    [
        ("F", (10.0, 20.0, 20.0), 2, 2),
        ("A", (10.0, 20.0), 2, 2),
        ("A", (10.0, 20.0, float("inf")), 2, 2),
        ("A", (10.0, 20.0, 20.0), 0, 2),
        ("A", (10.0, 20.0, 20.0), 2, 0),
    ],
)
def test_reference_rejects_out_of_contract_inputs(
    model: str,
    velocity: tuple[float, ...],
    tx_count: int,
    rx_element_count: int,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        cdl_joint_tx_rx_dual_pol_spatiotemporal_covariance_reference(
            model,
            velocity,
            tx_count=tx_count,
            rx_element_count=rx_element_count,
        )
