from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from evaluation.g4_cdl_joint_tx_rx_dual_pol_spatiotemporal_covariance import (
    cdl_joint_tx_rx_dual_pol_spatiotemporal_covariance_reference,
)
from evaluation.g4_cdl_joint_tx_rx_dual_tx_rx_spatiotemporal_covariance import (
    cdl_joint_tx_rx_dual_tx_rx_spatiotemporal_covariance_reference,
)
from evaluation.g4_cdl_joint_tx_rx_spatiotemporal_covariance import (
    cdl_joint_tx_rx_spatiotemporal_covariance_reference,
)

_REFERENCE_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "research"
    / "G4_CDL_TAP_ANGLE_REFERENCE_2026-10-26.json"
)


def _channel_index(
    rx_element: int,
    rx_pol: int,
    tx_element: int,
    tx_pol: int,
    *,
    rx_element_count: int = 2,
    tx_element_count: int = 2,
) -> int:
    rx_port = rx_pol * rx_element_count + rx_element
    tx_port = tx_pol * tx_element_count + tx_element
    return rx_port * (2 * tx_element_count) + tx_port


@pytest.mark.parametrize("model", tuple("ABCDE"))
def test_vv_selection_reduces_to_existing_single_v_reference(model: str) -> None:
    velocity = (10.0, 20.0, 20.0)
    dual = cdl_joint_tx_rx_dual_tx_rx_spatiotemporal_covariance_reference(model, velocity)
    single = cdl_joint_tx_rx_spatiotemporal_covariance_reference(model, velocity)
    selected = [_channel_index(rx, 0, tx, 0) for rx in range(2) for tx in range(2)]
    assert len(dual) == 16
    assert all(len(row) == 16 for row in dual)
    for row, dual_row in enumerate(selected):
        for column, dual_column in enumerate(selected):
            assert dual[dual_row][dual_column] == pytest.approx(single[row][column], abs=1e-12)


@pytest.mark.parametrize("model", tuple("ABCDE"))
def test_tx_v_selection_reduces_to_existing_dual_rx_reference(model: str) -> None:
    velocity = (10.0, 20.0, 20.0)
    dual_tx_rx = cdl_joint_tx_rx_dual_tx_rx_spatiotemporal_covariance_reference(model, velocity)
    dual_rx = cdl_joint_tx_rx_dual_pol_spatiotemporal_covariance_reference(model, velocity)
    selected = [
        _channel_index(rx, rx_pol, tx, 0)
        for rx_pol in range(2)
        for rx in range(2)
        for tx in range(2)
    ]
    for row, full_row in enumerate(selected):
        for column, full_column in enumerate(selected):
            assert dual_tx_rx[full_row][full_column] == pytest.approx(
                dual_rx[row][column], abs=1e-12
            )


@pytest.mark.parametrize("model", tuple("ABC"))
def test_nlos_jones_blocks_have_expected_xpr_scaling_and_zero_cross_covariance(
    model: str,
) -> None:
    document = json.loads(_REFERENCE_PATH.read_text(encoding="utf-8"))
    profile = next(row for row in document["profiles"] if row["profile_model"] == model)
    xpr_linear = 10.0 ** (float(profile["xpr_db"]) / 10.0)
    reference = cdl_joint_tx_rx_dual_tx_rx_spatiotemporal_covariance_reference(
        model, (0.0, 0.0, 0.0), lag_s=0.0
    )
    pairs = ((0, 0), (0, 1), (1, 0), (1, 1))
    for rx_pol, tx_pol in pairs:
        index = _channel_index(0, rx_pol, 0, tx_pol)
        expected_power = 1.0 if rx_pol == tx_pol else 1.0 / xpr_linear
        assert reference[index][index] == pytest.approx(expected_power + 0.0j, abs=1e-12)
    for row_pair in pairs:
        row = _channel_index(0, *row_pair[:1], 0, row_pair[1])
        for column_pair in pairs:
            if row_pair == column_pair:
                continue
            column = _channel_index(0, *column_pair[:1], 0, column_pair[1])
            assert reference[row][column] == pytest.approx(0.0j, abs=1e-12)


@pytest.mark.parametrize("model", tuple("DE"))
def test_los_vv_hh_cross_covariance_matches_signed_specular_power(model: str) -> None:
    document = json.loads(_REFERENCE_PATH.read_text(encoding="utf-8"))
    profile = next(row for row in document["profiles"] if row["profile_model"] == model)
    raw = [10.0 ** (float(row["power_db"]) / 10.0) for row in profile["components"]]
    specular_index = next(
        index
        for index, component in enumerate(profile["components"])
        if component["component"] == "specular"
    )
    specular_power = raw[specular_index] / math.fsum(raw)
    reference = cdl_joint_tx_rx_dual_tx_rx_spatiotemporal_covariance_reference(
        model, (0.0, 0.0, 0.0), lag_s=0.0
    )
    vv = _channel_index(0, 0, 0, 0)
    hh = _channel_index(0, 1, 0, 1)
    vh = _channel_index(0, 0, 0, 1)
    hv = _channel_index(0, 1, 0, 0)
    assert reference[vv][hh] == pytest.approx(-specular_power + 0.0j, abs=1e-12)
    assert reference[hh][vv] == pytest.approx(-specular_power + 0.0j, abs=1e-12)
    assert reference[vh][hv] == pytest.approx(0.0j, abs=1e-12)
    assert reference[hv][vh] == pytest.approx(0.0j, abs=1e-12)


@pytest.mark.parametrize("model", tuple("ABCDE"))
def test_zero_lag_stationary_reference_is_hermitian(model: str) -> None:
    reference = cdl_joint_tx_rx_dual_tx_rx_spatiotemporal_covariance_reference(
        model, (0.0, 0.0, 0.0), lag_s=0.0
    )
    for row in range(len(reference)):
        for column in range(len(reference)):
            assert reference[row][column] == pytest.approx(
                reference[column][row].conjugate(), abs=1e-12
            )


@pytest.mark.parametrize("model", tuple("ABCDE"))
def test_frozen_oblique_reference_is_finite(model: str) -> None:
    reference = cdl_joint_tx_rx_dual_tx_rx_spatiotemporal_covariance_reference(
        model, (10.0, 20.0, 20.0)
    )
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
    model: str,
    velocity: tuple[float, ...],
    tx_count: int,
    rx_count: int,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        cdl_joint_tx_rx_dual_tx_rx_spatiotemporal_covariance_reference(
            model,
            velocity,
            tx_element_count=tx_count,
            rx_element_count=rx_count,
        )
