"""Standards-based tests for the native Case-A SSB timing utility."""

from __future__ import annotations

import numpy as np
import pytest
from integrations.sionna import (
    ssb_candidate_symbols as public_ssb_candidate_symbols,
)
from integrations.sionna import (
    ssb_case_a_candidate_symbols as public_ssb_case_a_candidate_symbols,
)
from integrations.sionna.ssb_timing import ssb_candidate_symbols, ssb_case_a_candidate_symbols


@pytest.mark.parametrize(
    ("case", "frequency", "frequency_range", "scs", "kwargs", "expected"),
    [
        ("A", 3e9, "FR1", 15_000, {}, (2, 8, 16, 22)),
        ("B", 3e9, "FR1", 30_000, {}, (4, 8, 16, 20)),
        ("B", 3.5e9, "FR1", 30_000, {}, (4, 8, 16, 20, 32, 36, 44, 48)),
        ("C", 3e9, "FR1", 30_000, {"paired_spectrum": True}, (2, 8, 16, 22)),
        (
            "C",
            1.88e9,
            "FR1",
            30_000,
            {"paired_spectrum": False},
            (2, 8, 16, 22, 30, 36, 44, 50),
        ),
        (
            "D",
            28e9,
            "FR2",
            120_000,
            {},
            tuple(
                sorted(
                    offset + 28 * n
                    for n in (0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18)
                    for offset in (4, 8, 16, 20)
                )
            ),
        ),
        (
            "E",
            28e9,
            "FR2-1",
            240_000,
            {},
            tuple(
                sorted(
                    offset + 56 * n
                    for n in (0, 1, 2, 3, 5, 6, 7, 8)
                    for offset in (8, 12, 16, 20, 32, 36, 40, 44)
                )
            ),
        ),
        (
            "F",
            60e9,
            "FR2-2",
            480_000,
            {},
            tuple(sorted(offset + 14 * n for n in range(32) for offset in (2, 9))),
        ),
        (
            "G",
            60e9,
            "FR2-2",
            960_000,
            {},
            tuple(sorted(offset + 14 * n for n in range(32) for offset in (2, 9))),
        ),
    ],
)
def test_cases_a_through_g_match_release_19_candidate_positions(
    case: str,
    frequency: float,
    frequency_range: str,
    scs: int,
    kwargs: dict[str, bool],
    expected: tuple[int, ...],
) -> None:
    actual = ssb_candidate_symbols(
        case,
        frequency,
        subcarrier_spacing_hz=scs,
        frequency_range=frequency_range,
        **kwargs,
    )

    assert actual == expected
    assert actual == tuple(sorted(actual))
    if case in "DEFG":
        assert len(actual) == 64


def test_generic_case_a_api_is_exported_and_legacy_wrapper_matches() -> None:
    assert public_ssb_candidate_symbols is ssb_candidate_symbols
    assert ssb_case_a_candidate_symbols(3.5e9) == ssb_candidate_symbols(
        "A", 3.5e9, subcarrier_spacing_hz=15_000, frequency_range="FR1"
    )


@pytest.mark.parametrize(
    ("case", "frequency", "frequency_range", "scs", "kwargs", "message"),
    [
        ("H", 3.5e9, "FR1", 15_000, {}, "A through G"),
        ("A", 3.5e9, "FR1", 30_000, {}, "requires subcarrier_spacing_hz=15000"),
        ("D", 28e9, "FR1", 120_000, {}, 'frequency_range="FR2"'),
        ("F", 50e9, "FR2-2", 480_000, {}, "outside the supported FR2-2"),
        ("A", 3.5e9, "FR1", 15_000, {"shared_spectrum": True}, "shared-spectrum"),
    ],
)
def test_ssb_case_api_rejects_unsupported_combinations(
    case: str,
    frequency: float,
    frequency_range: str,
    scs: int,
    kwargs: dict[str, bool],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ssb_candidate_symbols(
            case,
            frequency,
            subcarrier_spacing_hz=scs,
            frequency_range=frequency_range,
            **kwargs,
        )


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
    with pytest.raises(ValueError, match="outside the supported FR1"):
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
    with pytest.raises(ValueError, match='requires frequency_range="FR1"'):
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
