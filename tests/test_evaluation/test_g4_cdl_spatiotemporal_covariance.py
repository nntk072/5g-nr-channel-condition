from __future__ import annotations

import math

import pytest
from evaluation.g4_cdl_spatiotemporal_covariance import cdl_spatiotemporal_covariance_reference


def test_zero_lag_reference_is_unit_power_hermitian_covariance() -> None:
    reference = cdl_spatiotemporal_covariance_reference("A", (0.0, 0.0, 0.0), lag_s=0.0)
    assert reference[0][0] == pytest.approx(1.0 + 0.0j, abs=1e-12)
    assert reference[1][1] == pytest.approx(1.0 + 0.0j, abs=1e-12)
    assert reference[1][0] == pytest.approx(reference[0][1].conjugate(), abs=1e-12)


def test_single_element_reference_has_unit_magnitude_at_zero_lag() -> None:
    reference = cdl_spatiotemporal_covariance_reference(
        "E", (10.0, 20.0, 20.0), lag_s=0.0, element_count=1
    )
    assert reference == ((pytest.approx(1.0 + 0.0j, abs=1e-12),),)


@pytest.mark.parametrize("model", tuple("ABCDE"))
def test_reference_is_finite_for_frozen_motion_cases(model: str) -> None:
    for velocity in ((30.0, 0.0, 0.0), (10.0, 20.0, 20.0)):
        reference = cdl_spatiotemporal_covariance_reference(model, velocity)
        assert len(reference) == 2
        assert all(len(row) == 2 for row in reference)
        assert all(
            math.isfinite(value.real) and math.isfinite(value.imag)
            for row in reference
            for value in row
        )


@pytest.mark.parametrize(
    ("model", "velocity"),
    [("F", (30.0, 0.0, 0.0)), ("A", (30.0, 0.0)), ("A", (30.0, 0.0, float("nan")))],
)
def test_reference_rejects_out_of_contract_inputs(model: str, velocity: tuple[float, ...]) -> None:
    with pytest.raises((TypeError, ValueError)):
        cdl_spatiotemporal_covariance_reference(model, velocity)
