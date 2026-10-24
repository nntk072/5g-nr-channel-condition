"""Known-occasion PBCH receiver under CP-contained frequency-selective channels."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sionna.phy.fec.polar")

from integrations.sionna import (  # noqa: E402
    PBCHMIB,
    map_nr_ssb_resource_grid,
    nr_pbch_acquire_and_receive,
    nr_pbch_bch_encode,
    nr_pbch_mib_serialize,
    nr_pbch_payload_generate,
    nr_pbch_qpsk_modulate,
    nr_pbch_receive_ofdm,
    nr_pbch_scramble_bits,
)

_FFT_SIZE = 256
_CP_LENGTHS = (18, 18, 18, 18)
_CELL_ID = 431
_L_MAX = 4
_SSB_INDEX = 2
_HALF_FRAME = 1


def _make_ssb_waveform(sfn: int) -> tuple[np.ndarray, PBCHMIB]:
    mib = PBCHMIB(
        system_frame_number_msb=sfn >> 4,
        subcarrier_spacing_common="scs15or60",
        ssb_subcarrier_offset=7,
        dmrs_type_a_position="pos2",
        control_resource_set_zero=5,
        search_space_zero=11,
        cell_barred="notBarred",
        intra_freq_reselection="allowed",
        spare=0,
    )
    transport_block = nr_pbch_payload_generate(
        nr_pbch_mib_serialize(mib),
        _CELL_ID,
        sfn,
        _HALF_FRAME,
        _SSB_INDEX,
        max_candidate_blocks_per_half_frame=_L_MAX,
        k_ssb_msb=1,
        pdcch_repetition=0,
    )
    coded_bits = nr_pbch_bch_encode(transport_block)
    physical_bits = nr_pbch_scramble_bits(
        coded_bits,
        _CELL_ID,
        _SSB_INDEX,
        max_candidate_blocks_per_half_frame=_L_MAX,
    )
    pbch_symbols = nr_pbch_qpsk_modulate(physical_bits)
    ssb = map_nr_ssb_resource_grid(
        _CELL_ID // 3,
        _CELL_ID % 3,
        pbch_symbols,
        ssb_index_in_half_frame=_SSB_INDEX,
        half_frame_index=_HALF_FRAME,
        max_candidate_blocks_per_half_frame=_L_MAX,
    )
    waveform_symbols: list[np.ndarray] = []
    for grid_symbol, cp_length in zip(ssb.grid, _CP_LENGTHS, strict=True):
        centered = np.zeros(_FFT_SIZE, dtype=np.complex128)
        centered[8:248] = grid_symbol
        useful = np.fft.ifft(np.fft.ifftshift(centered), norm="ortho")
        waveform_symbols.append(np.concatenate((useful[-cp_length:], useful)))
    return np.concatenate(waveform_symbols), mib


def _apply_channel_and_noise(
    waveform: np.ndarray, *, seed: int, delay: int, snr_db: float
) -> tuple[np.ndarray, float, np.ndarray]:
    rng = np.random.default_rng(seed)
    taps = np.zeros(delay + 1, dtype=np.complex128)
    taps[0] = 1.0
    taps[delay] = 0.32 * np.exp(1j * rng.uniform(-np.pi, np.pi))
    clean = np.convolve(waveform, taps)[: waveform.size]
    noise_variance = float(np.mean(np.abs(waveform) ** 2) * 10.0 ** (-snr_db / 10.0))
    noise = np.sqrt(noise_variance / 2.0) * (
        rng.standard_normal(waveform.size) + 1j * rng.standard_normal(waveform.size)
    )
    return clean + noise, noise_variance, taps


@pytest.mark.native_sionna
def test_pbch_receiver_decodes_seeded_awgn_and_cp_contained_multipath() -> None:
    sfn = 0b1011010110
    waveform, expected_mib = _make_ssb_waveform(sfn)

    for draw, delay in enumerate((1, 2, 3, 4, 5, 2, 4, 1, 5, 3, 1, 4, 2, 5, 3, 2)):
        received, noise_variance, taps = _apply_channel_and_noise(
            waveform, seed=8100 + draw, delay=delay, snr_db=24.0
        )
        result = nr_pbch_receive_ofdm(
            received,
            fft_size=_FFT_SIZE,
            cyclic_prefix_lengths=_CP_LENGTHS,
            physical_cell_id=_CELL_ID,
            ssb_index_in_half_frame=_SSB_INDEX,
            half_frame_index=_HALF_FRAME,
            max_candidate_blocks_per_half_frame=_L_MAX,
            noise_variance=noise_variance,
        )

        assert result.crc_valid
        assert result.mib == expected_mib
        assert result.payload_recovery is not None
        assert result.payload_recovery.system_frame_number == sfn
        assert result.payload_recovery.half_frame_index == _HALF_FRAME
        assert result.payload_recovery.k_ssb_msb == 1
        assert result.channel_estimates.shape == (3, 240)
        assert result.equalized_pbch_symbols.shape == (432,)
        assert result.descrambled_llrs.shape == (864,)
        subcarrier_offsets = np.arange(240) - 120
        channel_reference = sum(
            tap * np.exp(-2j * np.pi * subcarrier_offsets * tap_index / _FFT_SIZE)
            for tap_index, tap in enumerate(taps)
        )
        data_region = np.ones((3, 240), dtype=bool)
        data_region[1, 48:192] = False
        channel_error = result.channel_estimates - channel_reference[None, :]
        relative_channel_error = np.linalg.norm(channel_error[data_region]) / np.linalg.norm(
            np.broadcast_to(channel_reference, data_region.shape)[data_region]
        )
        assert relative_channel_error < 0.06


@pytest.mark.native_sionna
def test_pbch_receiver_decodes_noiseless_flat_channel() -> None:
    sfn = 93
    waveform, expected_mib = _make_ssb_waveform(sfn)
    variance = 1e-12
    rng = np.random.default_rng(9901)
    noise = np.sqrt(variance / 2.0) * (
        rng.standard_normal(waveform.size) + 1j * rng.standard_normal(waveform.size)
    )
    result = nr_pbch_receive_ofdm(
        waveform + noise,
        fft_size=_FFT_SIZE,
        cyclic_prefix_lengths=_CP_LENGTHS,
        physical_cell_id=_CELL_ID,
        ssb_index_in_half_frame=_SSB_INDEX,
        half_frame_index=_HALF_FRAME,
        max_candidate_blocks_per_half_frame=_L_MAX,
        noise_variance=variance,
    )

    assert result.crc_valid
    assert result.mib == expected_mib
    assert result.payload_recovery is not None
    assert result.payload_recovery.system_frame_number == sfn


@pytest.mark.native_sionna
def test_blind_pss_sss_and_candidate_search_decode_pbch_under_cfo_multipath_awgn() -> None:
    sfn = 0b1100101011
    waveform, expected_mib = _make_ssb_waveform(sfn)
    timing_offset = 23
    sample_rate_hz = 3_840_000.0
    cfo_hz = 2_500.0
    channelled, noise_variance, _ = _apply_channel_and_noise(
        waveform, seed=7221, delay=5, snr_db=24.0
    )
    capture = np.concatenate(
        (
            np.zeros(timing_offset, dtype=np.complex128),
            channelled,
            np.zeros(_FFT_SIZE + _CP_LENGTHS[-1], dtype=np.complex128),
        )
    )
    time = np.arange(capture.size) / sample_rate_hz
    capture *= np.exp(2j * np.pi * cfo_hz * time)

    result = nr_pbch_acquire_and_receive(
        capture,
        fft_size=_FFT_SIZE,
        cyclic_prefix_lengths=_CP_LENGTHS,
        sampling_frequency_hz=sample_rate_hz,
        max_candidate_blocks_per_half_frame=_L_MAX,
        noise_variance=noise_variance,
        cfo_hypotheses_hz=(-4_000.0, -2_000.0, 0.0, 2_000.0, 4_000.0),
        cfo_guard_samples=5,
    )

    assert result.cell_detection.sample_offset == timing_offset
    assert result.cell_detection.physical_cell_id == _CELL_ID
    assert result.cell_detection.cfo_hz == pytest.approx(cfo_hz, abs=100.0)
    assert result.cell_detection.pss_score > 0.7
    assert result.cell_detection.sss_score > 0.7
    assert result.ssb_index_in_half_frame == _SSB_INDEX
    assert result.half_frame_index == _HALF_FRAME
    assert result.pbch_reception.crc_valid
    assert result.pbch_reception.mib == expected_mib
    assert result.pbch_reception.payload_recovery is not None
    assert result.pbch_reception.payload_recovery.system_frame_number == sfn
