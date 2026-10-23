"""Standards-based tests for the native Case-A SSB timing utility."""

from __future__ import annotations

import numpy as np
import pytest
from integrations.sionna import ssb_case_a_candidate_symbols as public_ssb_case_a_candidate_symbols
from integrations.sionna.ssb_timing import ssb_case_a_candidate_symbols


@pytest.mark.parametrize(
    ("carrier_frequency_hz", "expected"),
    [
        (2.1e9, (2, 8, 16, 22)),
        (3.0e9, (2, 8, 16, 22)),
        (3_000_000_001.0, (2, 8, 16, 22, 30, 36, 44, 50)),
        (3.5e9, (2, 8, 16, 22, 30, 36, 44, 50)),
    ],
)
def test_case_a_candidates_match_fr1_frequency_branch(
    carrier_frequency_hz: float,
    expected: tuple[int, ...],
) -> None:
    actual = ssb_case_a_candidate_symbols(carrier_frequency_hz)

    assert actual == expected
    assert len(actual) == len(expected)
    assert tuple(sorted(actual)) == actual


def test_case_a_returns_candidates_in_ascending_order_for_each_n() -> None:
    symbols = ssb_case_a_candidate_symbols(3.5e9)

    assert symbols == tuple(symbol for n in range(4) for symbol in (2 + 14 * n, 8 + 14 * n))


@pytest.mark.parametrize("carrier_frequency_hz", [410_000_000.0, 7_125_000_000.0])
def test_case_a_accepts_inclusive_fr1_frequency_endpoints(carrier_frequency_hz: float) -> None:
    assert ssb_case_a_candidate_symbols(carrier_frequency_hz)


def test_case_a_timing_is_exported_from_sionna_integration_package() -> None:
    assert public_ssb_case_a_candidate_symbols is ssb_case_a_candidate_symbols


@pytest.mark.parametrize("carrier_frequency_hz", [np.nan, np.inf, -np.inf, 0.0, -1.0])
def test_case_a_rejects_nonfinite_or_non_fr1_carrier(
    carrier_frequency_hz: object,
) -> None:
    with pytest.raises(ValueError, match="carrier_frequency_hz"):
        ssb_case_a_candidate_symbols(carrier_frequency_hz)


@pytest.mark.parametrize("carrier_frequency_hz", [409_999_999.0, 7_125_000_001.0])
def test_case_a_rejects_frequencies_outside_fr1(
    carrier_frequency_hz: float,
) -> None:
    with pytest.raises(ValueError, match="within FR1"):
        ssb_case_a_candidate_symbols(carrier_frequency_hz)


@pytest.mark.parametrize("subcarrier_spacing_hz", [0.0, 30_000.0, np.nan, np.inf])
def test_case_a_rejects_unsupported_subcarrier_spacing(
    subcarrier_spacing_hz: float,
) -> None:
    with pytest.raises(ValueError, match="subcarrier_spacing_hz"):
        ssb_case_a_candidate_symbols(3.5e9, subcarrier_spacing_hz=subcarrier_spacing_hz)


def test_case_a_rejects_shared_spectrum_and_other_frequency_range() -> None:
    with pytest.raises(ValueError, match="shared-spectrum"):
        ssb_case_a_candidate_symbols(3.5e9, shared_spectrum=True)
    with pytest.raises(ValueError, match='frequency_range must be "FR1"'):
        ssb_case_a_candidate_symbols(3.5e9, frequency_range="FR2")


@pytest.mark.parametrize(
    ("carrier_frequency_hz", "subcarrier_spacing_hz"),
    [(True, 15_000), (3.5e9, True), ("3.5GHz", 15_000)],
)
def test_case_a_rejects_wrong_numeric_types(
    carrier_frequency_hz: object,
    subcarrier_spacing_hz: object,
) -> None:
    with pytest.raises(TypeError):
        ssb_case_a_candidate_symbols(
            carrier_frequency_hz,  # type: ignore[arg-type]
            subcarrier_spacing_hz=subcarrier_spacing_hz,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("carrier_frequency_hz", "subcarrier_spacing_hz", "message"),
    [(10**1000, 15_000, "carrier_frequency_hz"), (3.5e9, 10**1000, "subcarrier_spacing_hz")],
)
def test_case_a_rejects_unrepresentable_finite_inputs(
    carrier_frequency_hz: object,
    subcarrier_spacing_hz: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ssb_case_a_candidate_symbols(
            carrier_frequency_hz,  # type: ignore[arg-type]
            subcarrier_spacing_hz=subcarrier_spacing_hz,  # type: ignore[arg-type]
        )


def test_case_a_rejects_non_boolean_shared_spectrum_flag() -> None:
    with pytest.raises(TypeError, match="shared_spectrum must be a bool"):
        ssb_case_a_candidate_symbols(3.5e9, shared_spectrum=1)  # type: ignore[arg-type]
