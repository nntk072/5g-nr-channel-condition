from __future__ import annotations

import numpy as np
import pytest
from evaluation.g4_tdl_covariance_reference import (
    fixed_tdl_first_tap_power_components,
    scalable_tdl_first_tap_covariance_reference,
    scalable_tdl_first_tap_power_components,
)


@pytest.mark.parametrize("model", tuple("ABCDE"))
def test_first_tap_component_powers_come_from_scalable_etsi_rows(model: str) -> None:
    diffuse, specular = scalable_tdl_first_tap_power_components(model)

    assert diffuse > 0.0
    if model in "DE":
        assert specular > 0.0
    else:
        assert specular == 0.0
    assert diffuse + specular <= 1.0


@pytest.mark.parametrize("model", ("A10", "A30", "B100", "C60", "C300", "D10", "D30"))
def test_fixed_first_tap_component_powers_come_from_annex_b_rows(model: str) -> None:
    diffuse, specular = fixed_tdl_first_tap_power_components(model)

    assert diffuse > 0.0
    assert 0.0 <= specular < 1.0
    assert (specular > 0.0) == (model in {"D10", "D30"})
    assert diffuse + specular <= 1.0


@pytest.mark.parametrize("model", tuple("ABCDE"))
def test_first_tap_covariance_is_hermitian_psd_and_normalized(model: str) -> None:
    rho = 0.45
    phase = 0.23
    correlation = np.array(
        [
            [
                rho ** abs(row - column) * np.exp(1j * phase * (row - column))
                for column in range(3)
            ]
            for row in range(3)
        ],
        dtype=np.complex128,
    )
    covariance = scalable_tdl_first_tap_covariance_reference(model, correlation, 2)

    assert covariance.shape == (6, 6)
    np.testing.assert_allclose(covariance, covariance.conj().T, rtol=0.0, atol=1e-12)
    assert np.linalg.eigvalsh(covariance).min() >= -1e-12
    diffuse_power, specular_power = scalable_tdl_first_tap_power_components(model)
    los_vector = np.linalg.cholesky(correlation) @ np.ones(3)
    assert np.trace(covariance).real / 6 == pytest.approx(
        diffuse_power + specular_power * np.vdot(los_vector, los_vector).real / 3,
        abs=1e-12,
    )


@pytest.mark.parametrize(
    ("model", "correlation", "num_tx_ant"),
    (
        ("F", [[1.0]], 1),
        ("A", [[1.0, 0.2], [0.1, 1.0]], 1),
        ("A", [[1.0, 1.0], [1.0, 1.0]], 1),
        ("A", [[1.0, 0.2], [0.2, 1.0]], 0),
    ),
)
def test_first_tap_covariance_rejects_invalid_inputs(
    model: str, correlation: list[list[float]], num_tx_ant: int
) -> None:
    with pytest.raises(ValueError):
        scalable_tdl_first_tap_covariance_reference(model, correlation, num_tx_ant)
