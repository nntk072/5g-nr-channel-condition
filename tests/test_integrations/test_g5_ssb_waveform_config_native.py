"""TS 38.211 normal-CP and sample-rate configuration for NR SSB waveforms."""

from __future__ import annotations

import numpy as np
import pytest
from integrations.sionna import (
    nr_ssb_ofdm_demodulate,
    nr_ssb_ofdm_modulate,
    nr_ssb_waveform_config,
    ssb_candidate_symbols,
)


@pytest.mark.parametrize(
    ("scs_hz", "numerology", "long_cp"),
    [
        (15_000, 0, 20),
        (30_000, 1, 22),
        (60_000, 2, 26),
        (120_000, 3, 34),
        (240_000, 4, 50),
        (480_000, 5, 82),
        (960_000, 6, 146),
    ],
)
def test_normal_cp_and_sample_rate_match_38_211_formula(
    scs_hz: int, numerology: int, long_cp: int
) -> None:
    config = nr_ssb_waveform_config(
        subcarrier_spacing_hz=scs_hz,
        candidate_start_symbol=0,
        fft_size=256,
    )

    assert config.numerology == numerology
    assert config.sample_rate_hz == 256 * scs_hz
    assert config.cyclic_prefix_lengths == (long_cp, 18, 18, 18)


@pytest.mark.parametrize("numerology", range(7))
def test_cp_boundaries_use_absolute_half_frame_symbol_index(numerology: int) -> None:
    scs_hz = 15_000 * (2**numerology)
    boundary = 7 * (2**numerology)
    regular_cp = 18
    long_cp = (144 + 16 * (2**numerology)) // 8

    before = nr_ssb_waveform_config(
        subcarrier_spacing_hz=scs_hz,
        candidate_start_symbol=boundary - 1,
        fft_size=256,
    )
    at_boundary = nr_ssb_waveform_config(
        subcarrier_spacing_hz=scs_hz,
        candidate_start_symbol=boundary,
        fft_size=256,
    )
    next_subframe = nr_ssb_waveform_config(
        subcarrier_spacing_hz=scs_hz,
        candidate_start_symbol=14 * (2**numerology),
        fft_size=256,
    )

    assert before.cyclic_prefix_lengths == (regular_cp, long_cp, regular_cp, regular_cp)
    assert at_boundary.cyclic_prefix_lengths == (long_cp, regular_cp, regular_cp, regular_cp)
    assert next_subframe.cyclic_prefix_lengths == (long_cp, regular_cp, regular_cp, regular_cp)


@pytest.mark.parametrize(
    ("case", "carrier_hz", "frequency_range", "scs_hz"),
    [
        ("A", 3.5e9, "FR1", 15_000),
        ("B", 3.5e9, "FR1", 30_000),
        ("C", 3.5e9, "FR1", 30_000),
        ("D", 28e9, "FR2", 120_000),
        ("E", 28e9, "FR2-1", 240_000),
        ("F", 60e9, "FR2-2", 480_000),
        ("G", 60e9, "FR2-2", 960_000),
    ],
)
def test_release_19_ssb_candidate_positions_derive_four_short_cp_lengths(
    case: str, carrier_hz: float, frequency_range: str, scs_hz: int
) -> None:
    candidates = ssb_candidate_symbols(
        case,
        carrier_hz,
        subcarrier_spacing_hz=scs_hz,
        frequency_range=frequency_range,
    )
    configs = tuple(
        nr_ssb_waveform_config(
            subcarrier_spacing_hz=scs_hz,
            candidate_start_symbol=candidate,
            fft_size=256,
        )
        for candidate in candidates
    )

    assert tuple(config.candidate_start_symbol for config in configs) == candidates
    assert all(config.cyclic_prefix_lengths == (18, 18, 18, 18) for config in configs)


@pytest.mark.parametrize("scs_hz", [15_000, 30_000, 120_000, 240_000, 480_000, 960_000])
def test_derived_waveform_config_modulates_and_roundtrips_grid(scs_hz: int) -> None:
    config = nr_ssb_waveform_config(
        subcarrier_spacing_hz=scs_hz,
        candidate_start_symbol=2,
        fft_size=256,
    )
    rng = np.random.default_rng(scs_hz)
    grid = rng.standard_normal((4, 240)) + 1j * rng.standard_normal((4, 240))
    waveform = nr_ssb_ofdm_modulate(
        grid,
        fft_size=config.fft_size,
        cyclic_prefix_lengths=config.cyclic_prefix_lengths,
    )
    recovered = nr_ssb_ofdm_demodulate(
        waveform,
        fft_size=config.fft_size,
        cyclic_prefix_lengths=config.cyclic_prefix_lengths,
    )

    np.testing.assert_allclose(recovered, grid, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("scs_hz", "start_symbol", "fft_size", "error", "message"),
    [
        (True, 0, 256, TypeError, "subcarrier_spacing_hz"),
        (float("nan"), 0, 256, ValueError, "finite"),
        (45_000, 0, 256, ValueError, "powers of two"),
        (15_000, True, 256, TypeError, "candidate_start_symbol"),
        (15_000, -1, 256, ValueError, "fit a four-symbol block"),
        (15_000, 67, 256, ValueError, "fit a four-symbol block"),
        (15_000, 0, 258, ValueError, "integer normal-CP lengths"),
    ],
)
def test_waveform_configuration_rejects_invalid_values(
    scs_hz: object,
    start_symbol: object,
    fft_size: int,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        nr_ssb_waveform_config(  # type: ignore[arg-type]
            subcarrier_spacing_hz=scs_hz,
            candidate_start_symbol=start_symbol,
            fft_size=fft_size,
        )
