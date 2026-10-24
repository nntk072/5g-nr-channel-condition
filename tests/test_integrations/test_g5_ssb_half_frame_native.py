"""Candidate-indexed SSB placement in a full 5 ms half-frame waveform."""

from __future__ import annotations

import numpy as np
import pytest
from integrations.sionna import (
    map_nr_ssb_resource_grid,
    nr_ssb_ofdm_modulate,
    nr_ssb_ofdm_modulate_half_frame,
    nr_ssb_waveform_config,
    ssb_candidate_symbols,
)


def _block(candidate_index: int, candidate_count: int, *, half_frame: int = 1):
    return map_nr_ssb_resource_grid(
        17,
        2,
        np.ones(432, dtype=np.complex128),
        ssb_index_in_half_frame=candidate_index,
        half_frame_index=half_frame,
        max_candidate_blocks_per_half_frame=candidate_count,
    )


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
def test_sparse_candidate_grids_land_at_exact_release_19_half_frame_samples(
    case: str, carrier_hz: float, frequency_range: str, scs_hz: int
) -> None:
    fft_size = 256
    candidate_positions = ssb_candidate_symbols(
        case,
        carrier_hz,
        subcarrier_spacing_hz=scs_hz,
        frequency_range=frequency_range,
    )
    candidate_count = len(candidate_positions)
    chosen = (0, candidate_count - 1) if candidate_count > 1 else (0,)
    blocks = {index: _block(index, candidate_count) for index in reversed(chosen)}

    half_frame = nr_ssb_ofdm_modulate_half_frame(
        blocks,
        case=case,
        carrier_frequency_hz=carrier_hz,
        subcarrier_spacing_hz=scs_hz,
        frequency_range=frequency_range,
        fft_size=fft_size,
    )

    assert half_frame.subcarrier_spacing_hz == scs_hz
    assert half_frame.sample_rate_hz == fft_size * scs_hz
    assert half_frame.half_frame_index == 1
    assert half_frame.transmitted_candidate_indices == tuple(sorted(chosen))
    assert half_frame.candidate_start_symbols == candidate_positions
    assert half_frame.waveform.shape == (half_frame.sample_rate_hz // 200,)
    assert not half_frame.waveform.flags.writeable

    mu = int(np.log2(scs_hz / 15_000))
    long_cp = (144 + 16 * (2**mu)) * fft_size // 2048

    def symbol_sample_offset(symbol_index: int) -> int:
        total = 0
        for absolute_symbol in range(symbol_index):
            local_symbol = absolute_symbol % (14 * (2**mu))
            cp = long_cp if local_symbol in (0, 7 * (2**mu)) else 18
            total += fft_size + cp
        return total

    occupied = np.zeros(half_frame.waveform.size, dtype=bool)
    for candidate_index in chosen:
        start_symbol = candidate_positions[candidate_index]
        config = nr_ssb_waveform_config(
            subcarrier_spacing_hz=scs_hz,
            candidate_start_symbol=start_symbol,
            fft_size=fft_size,
        )
        expected = nr_ssb_ofdm_modulate(
            blocks[candidate_index].grid,
            fft_size=fft_size,
            cyclic_prefix_lengths=config.cyclic_prefix_lengths,
        )
        start = symbol_sample_offset(start_symbol)
        end = start + expected.size
        np.testing.assert_array_equal(half_frame.waveform[start:end], expected)
        occupied[start:end] = True
    np.testing.assert_array_equal(half_frame.waveform[~occupied], 0.0)


def test_half_frame_builder_rejects_grid_candidate_metadata_mismatch() -> None:
    block = _block(1, 8)
    with pytest.raises(ValueError, match="mapping key must match"):
        nr_ssb_ofdm_modulate_half_frame(
            {0: block},
            case="A",
            carrier_frequency_hz=3.5e9,
            subcarrier_spacing_hz=15_000,
            frequency_range="FR1",
            fft_size=256,
        )


def test_half_frame_builder_rejects_mismatched_candidate_count() -> None:
    block = _block(0, 4)
    with pytest.raises(ValueError, match="candidate count must match"):
        nr_ssb_ofdm_modulate_half_frame(
            {0: block},
            case="A",
            carrier_frequency_hz=3.5e9,
            subcarrier_spacing_hz=15_000,
            frequency_range="FR1",
            fft_size=256,
        )
