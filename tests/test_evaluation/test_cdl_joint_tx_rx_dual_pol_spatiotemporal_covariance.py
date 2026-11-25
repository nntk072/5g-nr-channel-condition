from __future__ import annotations

import math

import pytest
from evaluation.covariance.cdl import joint_tx_rx_dual_pol as dual_pol_module
from evaluation.covariance.cdl.joint_tx_rx import (
    cdl_joint_tx_rx_spatiotemporal_covariance_reference,
)
from evaluation.covariance.cdl.joint_tx_rx_dual_pol import (
    cdl_joint_tx_rx_dual_pol_spatiotemporal_covariance_reference,
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


def test_horizontal_reference_rejects_unsupported_component_kind(monkeypatch: pytest.MonkeyPatch):
    document = {
        "profiles": [
            {
                "profile_model": "A",
                "components": [{"component": "unsupported", "power_db": 0.0}],
                "xpr_db": 0.0,
                "cluster_spreads_deg": {"asa": 0.0, "zsa": 0.0, "asd": 0.0, "zsd": 0.0},
            }
        ]
    }
    dual_pol_module._horizontal_diffuse_block.cache_clear()
    monkeypatch.setattr(dual_pol_module, "_reference_document", lambda: document)

    with pytest.raises(ValueError, match="unsupported CDL component kind"):
        dual_pol_module._horizontal_diffuse_block(
            "A",
            (123.0, 4.0, 5.0),
            carrier_frequency_hz=3.7e9,
            lag_s=0.13,
            tx_count=1,
            rx_element_count=1,
            element_spacing_wavelengths=0.5,
        )
