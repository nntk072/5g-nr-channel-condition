"""3GPP NR normal-cyclic-prefix OFDM symbol timing."""

from __future__ import annotations

import math


def nr_normal_cp_symbol_start_times(
    subcarrier_spacing_hz: float,
    num_symbols: int,
    *,
    slot_index_in_subframe: int = 0,
) -> tuple[float, ...]:
    """Return symbol-start times within one normal-CP NR slot.

    The schedule follows 3GPP TS 38.211 clause 5.3.1 using its normal-CP
    ``N_CP,l^mu`` equation and symbol-start recurrence. SCS must be one of the
    NR numerologies supported by this repository (15--960 kHz). The time
    origin is the start of the requested slot; the first returned value is 0.
    """
    if (
        isinstance(subcarrier_spacing_hz, bool)
        or not isinstance(subcarrier_spacing_hz, (int, float))
        or not math.isfinite(subcarrier_spacing_hz)
        or subcarrier_spacing_hz <= 0
    ):
        raise ValueError("subcarrier_spacing_hz must be finite and positive")
    if isinstance(num_symbols, bool) or not isinstance(num_symbols, int):
        raise TypeError("num_symbols must be an integer")
    if num_symbols < 1 or num_symbols > 14:
        raise ValueError("one NR normal-CP slot must contain 1 to 14 symbols")
    numerology = math.log2(subcarrier_spacing_hz / 15_000.0)
    mu = round(numerology)
    if mu not in range(7) or not math.isclose(numerology, mu, abs_tol=1e-12):
        raise ValueError("subcarrier spacing must be 15 kHz times 2**mu for mu in [0, 6]")
    if isinstance(slot_index_in_subframe, bool) or not isinstance(slot_index_in_subframe, int):
        raise TypeError("slot_index_in_subframe must be an integer")
    slots_per_subframe = 2**mu
    if not 0 <= slot_index_in_subframe < slots_per_subframe:
        raise ValueError(
            f"slot_index_in_subframe must be in [0, {slots_per_subframe - 1}] for this SCS"
        )

    kappa = 64.0
    t_c_s = 1.0 / (480_000.0 * 4096.0)
    useful_symbol_samples = 2048.0 * kappa * 2.0**-mu
    long_cp_symbols = {0, 7 * slots_per_subframe}
    start_times: list[float] = []
    elapsed_s = 0.0
    for symbol_index in range(num_symbols):
        start_times.append(elapsed_s)
        subframe_symbol_index = 14 * slot_index_in_subframe + symbol_index
        cp_samples = 144.0 * kappa * 2.0**-mu
        if subframe_symbol_index in long_cp_symbols:
            cp_samples += 16.0 * kappa
        elapsed_s += (useful_symbol_samples + cp_samples) * t_c_s
    return tuple(start_times)
