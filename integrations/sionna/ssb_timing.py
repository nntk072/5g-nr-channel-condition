"""NR SSB candidate symbol positions from 3GPP TS 38.213 clause 4.1."""

from __future__ import annotations

import math
from collections.abc import Iterable
from numbers import Real
from typing import SupportsFloat

import numpy as np

_FR1_MIN_HZ = 410e6
_FR1_MAX_HZ = 7.125e9
_FR2_MIN_HZ = 24.25e9
_FR2_MAX_HZ = 71e9
_FR2_1_MAX_HZ = 52.6e9
_FR2_2_MIN_HZ = 52.6e9
_THREE_GHZ_HZ = 3e9
_CASE_SCS_HZ = {
    "A": 15_000,
    "B": 30_000,
    "C": 30_000,
    "D": 120_000,
    "E": 240_000,
    "F": 480_000,
    "G": 960_000,
}
_FR2_D_N = (0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18)
_FR2_E_N = (0, 1, 2, 3, 5, 6, 7, 8)


def ssb_candidate_symbols(
    case: str,
    carrier_frequency_hz: SupportsFloat,
    *,
    subcarrier_spacing_hz: SupportsFloat,
    frequency_range: str,
    shared_spectrum: bool = False,
    paired_spectrum: bool = True,
) -> tuple[int, ...]:
    """Return zero-based SSB candidate start symbols within one half-frame.

    Cases A-G follow TS 38.213 V19.1.0, clause 4.1. The applicable case is
    explicit because its selection is band-specific (TS 38.101); this function
    does not infer a case from frequency. FR1 cases A-C use the licensed
    spectrum branches, and Case C additionally distinguishes paired and
    unpaired spectrum. Shared-spectrum operation is rejected because its
    discovery-burst index mapping needs a separate interface.

    ``frequency_range`` is ``FR1`` for A-C, ``FR2`` for D, ``FR2-1`` for E,
    and ``FR2-2`` for F-G. This returns candidate timing positions only; it
    does not infer which positions are transmitted or map them to PBCH payload
    candidate indices.
    """
    if not isinstance(case, str):
        raise TypeError("case must be one of A through G")
    case = case.upper()
    if case not in _CASE_SCS_HZ:
        raise ValueError("case must be one of A through G")
    if isinstance(carrier_frequency_hz, (bool, np.bool_)) or not isinstance(
        carrier_frequency_hz, Real
    ):
        raise TypeError("carrier_frequency_hz must be a real scalar in hertz")
    if isinstance(subcarrier_spacing_hz, (bool, np.bool_)) or not isinstance(
        subcarrier_spacing_hz, Real
    ):
        raise TypeError("subcarrier_spacing_hz must be a real scalar in hertz")
    if not isinstance(shared_spectrum, bool):
        raise TypeError("shared_spectrum must be a bool")
    if shared_spectrum:
        raise ValueError("shared-spectrum timing/index mapping is not supported")
    if not isinstance(paired_spectrum, bool):
        raise TypeError("paired_spectrum must be a bool")

    try:
        carrier_hz = float(carrier_frequency_hz)
    except OverflowError:
        raise ValueError("carrier_frequency_hz must be finite") from None
    try:
        scs_hz = float(subcarrier_spacing_hz)
    except OverflowError:
        raise ValueError("subcarrier_spacing_hz must be finite") from None
    if not math.isfinite(carrier_hz):
        raise ValueError("carrier_frequency_hz must be finite")
    if not math.isfinite(scs_hz):
        raise ValueError("subcarrier_spacing_hz must be finite")
    expected_range = (
        "FR1" if case in "ABC" else "FR2" if case == "D" else "FR2-1" if case == "E" else "FR2-2"
    )
    if frequency_range != expected_range:
        raise ValueError(f'Case {case} requires frequency_range="{expected_range}"')
    if case in "ABC":
        within_range = _FR1_MIN_HZ <= carrier_hz <= _FR1_MAX_HZ
    elif case == "D":
        within_range = _FR2_MIN_HZ <= carrier_hz <= _FR2_MAX_HZ
    elif case == "E":
        within_range = _FR2_MIN_HZ <= carrier_hz <= _FR2_1_MAX_HZ
    else:
        within_range = _FR2_2_MIN_HZ <= carrier_hz <= _FR2_MAX_HZ
    if not within_range:
        raise ValueError(f"carrier_frequency_hz is outside the supported {expected_range} range")
    if scs_hz != _CASE_SCS_HZ[case]:
        raise ValueError(f"Case {case} requires subcarrier_spacing_hz={_CASE_SCS_HZ[case]}")

    n_values: Iterable[int]
    offsets: tuple[int, ...]
    if case == "A":
        n_values = range(2 if carrier_hz <= _THREE_GHZ_HZ else 4)
        offsets, stride = (2, 8), 14
    elif case == "B":
        n_values = range(1 if carrier_hz <= _THREE_GHZ_HZ else 2)
        offsets, stride = (4, 8, 16, 20), 28
    elif case == "C":
        two_n_limit = carrier_hz <= 3e9 if paired_spectrum else carrier_hz < 1.88e9
        n_values = range(2 if two_n_limit else 4)
        offsets, stride = (2, 8), 14
    elif case == "D":
        n_values, offsets, stride = _FR2_D_N, (4, 8, 16, 20), 28
    elif case == "E":
        n_values, offsets, stride = _FR2_E_N, (8, 12, 16, 20, 32, 36, 40, 44), 56
    else:
        n_values, offsets, stride = range(32), (2, 9), 14
    return tuple(sorted(offset + stride * n for n in n_values for offset in offsets))


def ssb_case_a_candidate_symbols(
    carrier_frequency_hz: SupportsFloat,
    *,
    subcarrier_spacing_hz: float = 15_000.0,
    frequency_range: str = "FR1",
    shared_spectrum: bool = False,
) -> tuple[int, ...]:
    """Backward-compatible Case-A wrapper for licensed FR1 timing."""
    return ssb_candidate_symbols(
        "A",
        carrier_frequency_hz,
        subcarrier_spacing_hz=subcarrier_spacing_hz,
        frequency_range=frequency_range,
        shared_spectrum=shared_spectrum,
    )
