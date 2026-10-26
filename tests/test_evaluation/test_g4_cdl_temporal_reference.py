from __future__ import annotations

import pytest
from evaluation.g4_cdl_temporal_reference import cdl_nlos_temporal_correlation_reference


@pytest.mark.parametrize(
    ("model", "expected"),
    (
        ("A", 0.43167972755566486 - 0.3320457616186679j),
        ("B", 0.4906163197757858 - 0.623515300201259j),
        ("C", 0.5269118181639711 - 0.4374108572450214j),
    ),
)
def test_cdl_nlos_temporal_reference_uses_frozen_etsi_rows(model: str, expected: complex) -> None:
    observed = cdl_nlos_temporal_correlation_reference(
        model,
        carrier_frequency_hz=3.5e9,
        speed_mps=30.0,
        lag_s=0.0007,
    )

    assert observed.real == pytest.approx(expected.real, abs=1e-14)
    assert observed.imag == pytest.approx(expected.imag, abs=1e-14)
    assert abs(observed) <= 1.0


@pytest.mark.parametrize(
    ("model", "carrier", "speed", "lag"),
    (
        ("D", 3.5e9, 30.0, 0.0007),
        ("A", True, 30.0, 0.0007),
        ("B", 3.5e9, -1.0, 0.0007),
        ("C", 3.5e9, 30.0, float("nan")),
    ),
)
def test_cdl_nlos_temporal_reference_rejects_unsupported_or_invalid_inputs(
    model: str, carrier: float, speed: float, lag: float
) -> None:
    with pytest.raises(ValueError):
        cdl_nlos_temporal_correlation_reference(
            model,
            carrier_frequency_hz=carrier,
            speed_mps=speed,
            lag_s=lag,
        )
