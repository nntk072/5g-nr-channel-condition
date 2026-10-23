"""NR SSB Case-A candidate symbol positions for the supported FR1 scope.

This module implements the 15 kHz Case-A candidate positions in 3GPP
TS 38.213 clause 4.1 for FR1 operation without shared spectrum. It returns
zero-based OFDM symbol indices measured from the start of each half-frame.
The input carrier frequency is in hertz. This utility does not map SSB
waveform contents, PBCH, or PBCH DM-RS, and it does not select a half-frame.
"""

from __future__ import annotations

import math
from numbers import Real

import numpy as np

_FR1_MIN_HZ = 410e6
_FR1_MAX_HZ = 7.125e9
_CASE_A_SUBCARRIER_SPACING_HZ = 15_000
_THREE_GHZ_HZ = 3e9


def ssb_case_a_candidate_symbols(
    carrier_frequency_hz: Real,
    *,
    subcarrier_spacing_hz: float = 15_000.0,
    frequency_range: str = "FR1",
    shared_spectrum: bool = False,
) -> tuple[int, ...]:
    """Return 15 kHz Case-A SSB candidate start symbols within one half-frame.

    The result is a sorted tuple of zero-based OFDM symbol indices measured
    from the start of a half-frame. For FR1 without shared spectrum, the
    candidate indices are ``2 + 14*n`` and ``8 + 14*n``. The allowed ``n``
    values are 0 and 1 at carrier frequencies up to and including 3 GHz,
    and 0 through 3 above 3 GHz.

    Args:
        carrier_frequency_hz: Positive FR1 carrier frequency, in hertz.
        subcarrier_spacing_hz: SSB subcarrier spacing, in hertz. Case A in
            this implementation requires exactly 15,000 Hz.
        frequency_range: Must be ``"FR1"``. FR2 is outside this API's scope.
        shared_spectrum: Must be false; shared-spectrum candidate rules are
            outside this API's scope.

    Returns:
        Candidate SSB start-symbol indices, in ascending order.

    Raises:
        TypeError: If a numeric setting is not a real scalar or a boolean is
            supplied as a numeric value.
        ValueError: If a value is non-finite, outside FR1, or describes a
            configuration not supported by this Case-A implementation.

    This reports candidate positions only. It does not map PBCH or other
    SSB resource elements, nor does it infer which candidates are transmitted.
    """
    if isinstance(carrier_frequency_hz, (bool, np.bool_)) or not isinstance(
        carrier_frequency_hz, Real
    ):
        raise TypeError("carrier_frequency_hz must be a real scalar in hertz")
    if isinstance(subcarrier_spacing_hz, (bool, np.bool_)) or not isinstance(
        subcarrier_spacing_hz, Real
    ):
        raise TypeError("subcarrier_spacing_hz must be a real scalar in hertz")

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
    if not _FR1_MIN_HZ <= carrier_hz <= _FR1_MAX_HZ:
        raise ValueError(
            f"carrier_frequency_hz must be within FR1 ({_FR1_MIN_HZ:g} to {_FR1_MAX_HZ:g} Hz)"
        )
    if not math.isfinite(scs_hz):
        raise ValueError("subcarrier_spacing_hz must be finite")
    if scs_hz != _CASE_A_SUBCARRIER_SPACING_HZ:
        raise ValueError("Case A requires subcarrier_spacing_hz=15000")
    if frequency_range != "FR1":
        raise ValueError('frequency_range must be "FR1" for this API')
    if not isinstance(shared_spectrum, bool):
        raise TypeError("shared_spectrum must be a bool")
    if shared_spectrum:
        raise ValueError("shared-spectrum operation is outside this API's scope")

    n_values = range(2 if carrier_hz <= _THREE_GHZ_HZ else 4)
    return tuple(symbol for n in n_values for symbol in (2 + 14 * n, 8 + 14 * n))
