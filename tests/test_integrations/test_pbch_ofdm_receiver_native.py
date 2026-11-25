"""Known-occasion PBCH receiver under CP-contained frequency-selective channels."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("sionna.phy.fec.polar")

from integrations.sionna import (  # noqa: E402
    PBCHMIB,
    NRSSBResourceGrid,
    PBCHMessageClassExtension,
    PBCHOFDMReception,
    map_nr_ssb_resource_grid,
    nr_bcch_bch_message_serialize,
    nr_pbch_acquire_and_receive,
    nr_pbch_bch_encode,
    nr_pbch_mib_serialize,
    nr_pbch_payload_generate,
    nr_pbch_qpsk_modulate,
    nr_pbch_receive_ofdm,
    nr_pbch_scramble_bits,
    nr_ssb_ofdm_demodulate,
    nr_ssb_ofdm_modulate,
    nr_ssb_ofdm_modulate_half_frame,
    nr_ssb_ofdm_modulate_radio_frame,
    nr_ssb_waveform_config,
    ssb_candidate_symbols,
)

_FFT_SIZE = 256
_CP_LENGTHS = (18, 18, 18, 18)
_CELL_ID = 431
_L_MAX = 4
_SSB_INDEX = 2
_HALF_FRAME = 1


def _make_ssb_waveform(
    sfn: int,
    *,
    candidate_count: int = _L_MAX,
    ssb_index: int = _SSB_INDEX,
    half_frame: int = _HALF_FRAME,
) -> tuple[np.ndarray, PBCHMIB]:
    ssb, mib = _make_ssb_resource_grid(
        sfn,
        candidate_count=candidate_count,
        ssb_index=ssb_index,
        half_frame=half_frame,
    )
    return (
        nr_ssb_ofdm_modulate(ssb.grid, fft_size=_FFT_SIZE, cyclic_prefix_lengths=_CP_LENGTHS),
        mib,
    )


def _make_ssb_resource_grid(
    sfn: int,
    *,
    candidate_count: int = _L_MAX,
    ssb_index: int = _SSB_INDEX,
    half_frame: int = _HALF_FRAME,
) -> tuple[NRSSBResourceGrid, PBCHMIB]:
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
        half_frame,
        ssb_index,
        max_candidate_blocks_per_half_frame=candidate_count,
        k_ssb_msb=1,
        pdcch_repetition=0,
    )
    coded_bits = nr_pbch_bch_encode(transport_block)
    physical_bits = nr_pbch_scramble_bits(
        coded_bits,
        _CELL_ID,
        ssb_index,
        max_candidate_blocks_per_half_frame=candidate_count,
    )
    pbch_symbols = nr_pbch_qpsk_modulate(physical_bits)
    ssb = map_nr_ssb_resource_grid(
        _CELL_ID // 3,
        _CELL_ID % 3,
        pbch_symbols,
        ssb_index_in_half_frame=ssb_index,
        half_frame_index=half_frame,
        max_candidate_blocks_per_half_frame=candidate_count,
    )
    return ssb, mib


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
def test_pbch_receiver_preserves_message_class_extension_choice() -> None:
    sfn = 0b1011010110
    message = PBCHMessageClassExtension()
    transport_block = nr_pbch_payload_generate(
        nr_bcch_bch_message_serialize(message),
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
    waveform = nr_ssb_ofdm_modulate(ssb.grid, fft_size=_FFT_SIZE, cyclic_prefix_lengths=_CP_LENGTHS)

    result = nr_pbch_receive_ofdm(
        waveform,
        fft_size=_FFT_SIZE,
        cyclic_prefix_lengths=_CP_LENGTHS,
        physical_cell_id=_CELL_ID,
        ssb_index_in_half_frame=_SSB_INDEX,
        half_frame_index=_HALF_FRAME,
        max_candidate_blocks_per_half_frame=_L_MAX,
        noise_variance=1.0,
    )

    assert result.crc_valid
    assert result.mib is None
    assert result.bcch_bch_message == message
    assert result.payload_recovery is not None
    assert result.payload_recovery.system_frame_number is None


@pytest.mark.native_sionna
def test_pbch_receiver_mrc_matches_sum_of_independent_receive_chain_llrs() -> None:
    sfn = 0b1011010110
    waveform, expected_mib = _make_ssb_waveform(sfn)
    reference_noise_variance = float(np.mean(np.abs(waveform) ** 2) * 10.0 ** (-20.0 / 10.0))
    noise_variances = np.asarray([0.4 * reference_noise_variance, 2.5 * reference_noise_variance])
    receive_gains = (0.28 * np.exp(0.7j), 0.91 * np.exp(-1.2j))
    channel_taps = (
        np.asarray([1.0, 0.22 * np.exp(0.4j), 0.0, 0.0, 0.0, 0.10j]),
        np.asarray([1.0, 0.0, 0.0, 0.18 * np.exp(-0.8j), 0.0, 0.26 * np.exp(0.3j)]),
    )
    receive_chains = []
    for chain, (gain, taps) in enumerate(zip(receive_gains, channel_taps, strict=True)):
        clean = gain * np.convolve(waveform, taps)[: waveform.size]
        rng = np.random.default_rng(27_400 + chain)
        noise = np.sqrt(noise_variances[chain] / 2.0) * (
            rng.standard_normal(waveform.size) + 1j * rng.standard_normal(waveform.size)
        )
        receive_chains.append(clean + noise)
    received = np.stack(receive_chains)

    def receive(samples: np.ndarray, variance: float | np.ndarray) -> PBCHOFDMReception:
        return nr_pbch_receive_ofdm(
            samples,
            fft_size=_FFT_SIZE,
            cyclic_prefix_lengths=_CP_LENGTHS,
            physical_cell_id=_CELL_ID,
            ssb_index_in_half_frame=_SSB_INDEX,
            half_frame_index=_HALF_FRAME,
            max_candidate_blocks_per_half_frame=_L_MAX,
            noise_variance=variance,
        )

    branch_results = [
        receive(chain, float(noise_variances[index])) for index, chain in enumerate(received)
    ]
    combined = receive(received, noise_variances)

    assert combined.crc_valid
    assert combined.mib == expected_mib
    assert combined.channel_estimates.shape == (2, 3, 240)
    assert combined.equalized_pbch_symbols.shape == (432,)
    np.testing.assert_allclose(
        combined.descrambled_llrs,
        branch_results[0].descrambled_llrs + branch_results[1].descrambled_llrs,
        rtol=1e-11,
        atol=1e-8,
    )


@pytest.mark.native_sionna
@pytest.mark.parametrize(
    ("noise_variance", "exception", "message"),
    [
        ([1.0], ValueError, "one value per receive chain"),
        ([1.0, 0.0], ValueError, "finite and positive"),
        ([1.0, np.nan], ValueError, "finite and positive"),
        ([True, 1.0], TypeError, "not booleans"),
        ([1.0 + 1.0j, 1.0], TypeError, "real numbers"),
    ],
)
def test_pbch_receiver_rejects_invalid_per_chain_noise_variance(
    noise_variance: Any,
    exception: type[Exception],
    message: str,
) -> None:
    samples = np.zeros((2, sum(_FFT_SIZE + cp for cp in _CP_LENGTHS)), dtype=np.complex128)
    with pytest.raises(exception, match=message):
        nr_pbch_receive_ofdm(
            samples,
            fft_size=_FFT_SIZE,
            cyclic_prefix_lengths=_CP_LENGTHS,
            physical_cell_id=_CELL_ID,
            ssb_index_in_half_frame=_SSB_INDEX,
            half_frame_index=_HALF_FRAME,
            max_candidate_blocks_per_half_frame=_L_MAX,
            noise_variance=noise_variance,
        )


@pytest.mark.native_sionna
@pytest.mark.parametrize(
    ("fft_size", "cyclic_prefix_lengths"),
    [(256, (18, 20, 18, 20)), (512, (36, 40, 36, 40)), (1024, (72, 80, 72, 80))],
)
def test_ssb_ofdm_waveform_matches_sionna_22_and_roundtrips(
    fft_size: int, cyclic_prefix_lengths: tuple[int, int, int, int]
) -> None:
    import torch
    from sionna.phy.ofdm import OFDMModulator

    rng = np.random.default_rng(4100 + fft_size)
    ssb_grid = rng.standard_normal((4, 240)) + 1j * rng.standard_normal((4, 240))
    waveform = nr_ssb_ofdm_modulate(
        ssb_grid, fft_size=fft_size, cyclic_prefix_lengths=cyclic_prefix_lengths
    )
    recovered = nr_ssb_ofdm_demodulate(
        waveform, fft_size=fft_size, cyclic_prefix_lengths=cyclic_prefix_lengths
    )

    full_grid = np.zeros((1, 4, fft_size), dtype=np.complex64)
    active_start = fft_size // 2 - 120
    full_grid[0, :, active_start : active_start + 240] = ssb_grid
    native = OFDMModulator(
        cyclic_prefix_length=np.asarray(cyclic_prefix_lengths, dtype=np.int32),
        device="cpu",
    )(torch.as_tensor(full_grid))

    np.testing.assert_allclose(waveform, native[0].detach().cpu().numpy(), rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(recovered, ssb_grid, rtol=1e-12, atol=1e-12)
    assert waveform.shape == (sum(fft_size + cp for cp in cyclic_prefix_lengths),)
    assert not waveform.flags.writeable
    assert not recovered.flags.writeable


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
@pytest.mark.parametrize(("candidate_count", "ssb_index"), [(8, 7), (10, 9), (20, 19), (64, 63)])
def test_pbch_receiver_decodes_high_candidate_indices_for_all_supported_counts(
    candidate_count: int, ssb_index: int
) -> None:
    sfn = 0b1011010110
    half_frame = 1
    waveform, expected_mib = _make_ssb_waveform(
        sfn,
        candidate_count=candidate_count,
        ssb_index=ssb_index,
        half_frame=half_frame,
    )
    result = nr_pbch_receive_ofdm(
        waveform,
        fft_size=_FFT_SIZE,
        cyclic_prefix_lengths=_CP_LENGTHS,
        physical_cell_id=_CELL_ID,
        ssb_index_in_half_frame=ssb_index,
        half_frame_index=half_frame,
        max_candidate_blocks_per_half_frame=candidate_count,
        noise_variance=1e-12,
    )

    assert result.crc_valid
    assert result.mib == expected_mib
    assert result.payload_recovery is not None
    assert result.payload_recovery.system_frame_number == sfn
    assert result.payload_recovery.half_frame_index == half_frame
    index_bits = (candidate_count - 1).bit_length() - 3
    assert result.payload_recovery.ssb_index_msb_bits == tuple(
        (ssb_index >> bit) & 1 for bit in range((candidate_count - 1).bit_length() - 1, 2, -1)
    )
    assert len(result.payload_recovery.ssb_index_msb_bits) == index_bits


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


@pytest.mark.native_sionna
def test_blind_acquisition_decodes_ssb_in_second_half_of_full_radio_frame() -> None:
    sfn = 0b1010011101
    candidate_positions = ssb_candidate_symbols(
        "A", 3.5e9, subcarrier_spacing_hz=15_000, frequency_range="FR1"
    )
    candidate_count = len(candidate_positions)
    ssb, expected_mib = _make_ssb_resource_grid(sfn, candidate_count=candidate_count)
    candidate_symbol = candidate_positions[_SSB_INDEX]
    config = nr_ssb_waveform_config(
        subcarrier_spacing_hz=15_000,
        candidate_start_symbol=candidate_symbol,
        fft_size=_FFT_SIZE,
    )
    radio_frame = nr_ssb_ofdm_modulate_radio_frame(
        {1: {_SSB_INDEX: ssb}},
        case="A",
        carrier_frequency_hz=3.5e9,
        subcarrier_spacing_hz=15_000,
        frequency_range="FR1",
        fft_size=_FFT_SIZE,
    )
    timing_offset = 37
    cfo_hz = -1_750.0
    channelled, noise_variance, _ = _apply_channel_and_noise(
        radio_frame.waveform, seed=106_430, delay=4, snr_db=25.0
    )
    capture = np.concatenate(
        (
            np.zeros(timing_offset, dtype=np.complex128),
            channelled,
            np.zeros(_FFT_SIZE + config.cyclic_prefix_lengths[-1], dtype=np.complex128),
        )
    )
    time = np.arange(capture.size) / config.sample_rate_hz
    capture *= np.exp(2j * np.pi * cfo_hz * time)

    result = nr_pbch_acquire_and_receive(
        capture,
        fft_size=_FFT_SIZE,
        cyclic_prefix_lengths=config.cyclic_prefix_lengths,
        sampling_frequency_hz=config.sample_rate_hz,
        max_candidate_blocks_per_half_frame=candidate_count,
        noise_variance=noise_variance,
        cfo_hypotheses_hz=(-3_000.0, -1_500.0, 0.0, 1_500.0, 3_000.0),
        cfo_guard_samples=5,
    )

    prior_samples = sum(
        _FFT_SIZE + (20 if symbol % 14 in (0, 7) else 18) for symbol in range(candidate_symbol)
    )
    expected_offset = timing_offset + radio_frame.sample_rate_hz // 200 + prior_samples
    assert result.cell_detection.sample_offset == expected_offset
    assert result.cell_detection.physical_cell_id == _CELL_ID
    assert result.cell_detection.cfo_hz == pytest.approx(cfo_hz, abs=100.0)
    assert result.ssb_index_in_half_frame == _SSB_INDEX
    assert result.half_frame_index == 1
    assert result.pbch_reception.crc_valid
    assert result.pbch_reception.mib == expected_mib
    assert result.pbch_reception.payload_recovery is not None
    assert result.pbch_reception.payload_recovery.system_frame_number == sfn


@pytest.mark.native_sionna
def test_full_radio_frame_blind_pbch_acquisition_combines_two_receive_chains() -> None:
    sfn = 0b0101101011
    candidate_positions = ssb_candidate_symbols(
        "A", 3.5e9, subcarrier_spacing_hz=15_000, frequency_range="FR1"
    )
    candidate_count = len(candidate_positions)
    ssb, expected_mib = _make_ssb_resource_grid(sfn, candidate_count=candidate_count)
    candidate_symbol = candidate_positions[_SSB_INDEX]
    config = nr_ssb_waveform_config(
        subcarrier_spacing_hz=15_000,
        candidate_start_symbol=candidate_symbol,
        fft_size=_FFT_SIZE,
    )
    radio_frame = nr_ssb_ofdm_modulate_radio_frame(
        {1: {_SSB_INDEX: ssb}},
        case="A",
        carrier_frequency_hz=3.5e9,
        subcarrier_spacing_hz=15_000,
        frequency_range="FR1",
        fft_size=_FFT_SIZE,
    )
    timing_offset = 29
    cfo_hz = 1_250.0
    sample_rate_hz = float(radio_frame.sample_rate_hz)
    branches: list[np.ndarray] = []
    noise_variances: list[float] = []
    for chain, (gain, delay, snr_db) in enumerate(
        ((0.31 * np.exp(0.7j), 3, 19.0), (0.88 * np.exp(-1.2j), 5, 26.0))
    ):
        rng = np.random.default_rng(106_500 + chain)
        taps = np.zeros(delay + 1, dtype=np.complex128)
        taps[0] = 1.0
        taps[delay] = 0.22 * np.exp(1j * rng.uniform(-np.pi, np.pi))
        clean = gain * np.convolve(radio_frame.waveform, taps)[: radio_frame.waveform.size]
        noise_variance = float(np.mean(np.abs(clean) ** 2) * 10.0 ** (-snr_db / 10.0))
        noise = np.sqrt(noise_variance / 2.0) * (
            rng.standard_normal(clean.size) + 1j * rng.standard_normal(clean.size)
        )
        capture = np.concatenate(
            (
                np.zeros(timing_offset, dtype=np.complex128),
                clean + noise,
                np.zeros(_FFT_SIZE + config.cyclic_prefix_lengths[-1], dtype=np.complex128),
            )
        )
        time = np.arange(capture.size) / sample_rate_hz
        branches.append(capture * np.exp(2j * np.pi * cfo_hz * time))
        noise_variances.append(noise_variance)

    result = nr_pbch_acquire_and_receive(
        np.stack(branches),
        fft_size=_FFT_SIZE,
        cyclic_prefix_lengths=config.cyclic_prefix_lengths,
        sampling_frequency_hz=sample_rate_hz,
        max_candidate_blocks_per_half_frame=candidate_count,
        noise_variance=np.asarray(noise_variances),
        cfo_hypotheses_hz=(-2_500.0, -1_250.0, 0.0, 1_250.0, 2_500.0),
        cfo_guard_samples=5,
    )

    prior_samples = sum(
        _FFT_SIZE + (20 if symbol % 14 in (0, 7) else 18) for symbol in range(candidate_symbol)
    )
    expected_offset = timing_offset + radio_frame.sample_rate_hz // 200 + prior_samples
    assert result.cell_detection.sample_offset == expected_offset
    assert result.cell_detection.physical_cell_id == _CELL_ID
    assert result.cell_detection.cfo_hz == pytest.approx(cfo_hz, abs=100.0)
    assert result.ssb_index_in_half_frame == _SSB_INDEX
    assert result.half_frame_index == 1
    assert result.pbch_reception.crc_valid
    assert result.pbch_reception.mib == expected_mib
    assert result.pbch_reception.channel_estimates.shape == (2, 3, 240)
    assert result.pbch_reception.payload_recovery is not None
    assert result.pbch_reception.payload_recovery.system_frame_number == sfn


@pytest.mark.native_sionna
def test_blind_acquisition_and_pbch_decode_combine_two_receive_chains() -> None:
    sfn = 0b1100101011
    waveform, expected_mib = _make_ssb_waveform(sfn)
    timing_offset = 23
    sample_rate_hz = 3_840_000.0
    cfo_hz = 2_500.0
    snr_db = 22.0
    reference_noise_variance = float(np.mean(np.abs(waveform) ** 2) * 10.0 ** (-snr_db / 10.0))
    noise_variances = np.asarray([0.5 * reference_noise_variance, 2.0 * reference_noise_variance])
    branches = []
    for chain, (gain, delay) in enumerate(((0.34 * np.exp(0.8j), 3), (0.91 * np.exp(-1.1j), 5))):
        rng = np.random.default_rng(91_200 + chain)
        taps = np.zeros(delay + 1, dtype=np.complex128)
        taps[0] = 1.0
        taps[delay] = 0.28 * np.exp(1j * rng.uniform(-np.pi, np.pi))
        clean = gain * np.convolve(waveform, taps)[: waveform.size]
        noise = np.sqrt(noise_variances[chain] / 2.0) * (
            rng.standard_normal(waveform.size) + 1j * rng.standard_normal(waveform.size)
        )
        capture = np.concatenate(
            (
                np.zeros(timing_offset, dtype=np.complex128),
                clean + noise,
                np.zeros(_FFT_SIZE + _CP_LENGTHS[-1], dtype=np.complex128),
            )
        )
        time = np.arange(capture.size) / sample_rate_hz
        branches.append(capture * np.exp(2j * np.pi * cfo_hz * time))

    result = nr_pbch_acquire_and_receive(
        np.stack(branches),
        fft_size=_FFT_SIZE,
        cyclic_prefix_lengths=_CP_LENGTHS,
        sampling_frequency_hz=sample_rate_hz,
        max_candidate_blocks_per_half_frame=_L_MAX,
        noise_variance=noise_variances,
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
    assert result.pbch_reception.channel_estimates.shape == (2, 3, 240)


@pytest.mark.native_sionna
def test_blind_acquisition_recovers_ssb_from_candidate_timed_half_frame() -> None:
    sfn = 0b1010011010
    candidate_count = 8
    ssb_index = 3
    half_frame_index = 1
    ssb, expected_mib = _make_ssb_resource_grid(
        sfn,
        candidate_count=candidate_count,
        ssb_index=ssb_index,
        half_frame=half_frame_index,
    )
    burst = nr_ssb_ofdm_modulate_half_frame(
        {ssb_index: ssb},
        case="A",
        carrier_frequency_hz=3.5e9,
        subcarrier_spacing_hz=15_000,
        frequency_range="FR1",
        fft_size=_FFT_SIZE,
    )
    start_symbol = burst.candidate_start_symbols[ssb_index]
    start_config = nr_ssb_waveform_config(
        subcarrier_spacing_hz=15_000,
        candidate_start_symbol=start_symbol,
        fft_size=_FFT_SIZE,
    )
    expected_symbol_offset = sum(
        _FFT_SIZE + (20 if symbol % 14 in (0, 7) else 18) for symbol in range(start_symbol)
    )
    assert start_config.cyclic_prefix_lengths == _CP_LENGTHS

    sample_rate_hz = float(burst.sample_rate_hz)
    timing_offset = 23
    channelled, noise_variance, _ = _apply_channel_and_noise(
        burst.waveform, seed=8123, delay=5, snr_db=24.0
    )
    capture = np.concatenate(
        (
            np.zeros(timing_offset, dtype=np.complex128),
            channelled,
            np.zeros(_FFT_SIZE + _CP_LENGTHS[-1], dtype=np.complex128),
        )
    )
    time = np.arange(capture.size) / sample_rate_hz
    capture *= np.exp(2j * np.pi * 2_500.0 * time)

    result = nr_pbch_acquire_and_receive(
        capture,
        fft_size=_FFT_SIZE,
        cyclic_prefix_lengths=_CP_LENGTHS,
        sampling_frequency_hz=sample_rate_hz,
        max_candidate_blocks_per_half_frame=candidate_count,
        noise_variance=noise_variance,
        cfo_hypotheses_hz=(-4_000.0, -2_000.0, 0.0, 2_000.0, 4_000.0),
        cfo_guard_samples=5,
    )

    assert result.cell_detection.sample_offset == timing_offset + expected_symbol_offset
    assert result.cell_detection.physical_cell_id == _CELL_ID
    assert result.cell_detection.cfo_hz == pytest.approx(2_500.0, abs=100.0)
    assert result.ssb_index_in_half_frame == ssb_index
    assert result.half_frame_index == half_frame_index
    assert result.pbch_reception.crc_valid
    assert result.pbch_reception.mib == expected_mib


@pytest.mark.native_sionna
@pytest.mark.parametrize(
    ("case", "carrier_hz", "frequency_range", "scs_hz", "candidate_index"),
    [
        ("B", 3.5e9, "FR1", 30_000, 7),
        ("C", 3.5e9, "FR1", 30_000, 7),
        ("D", 28e9, "FR2", 120_000, 63),
        ("E", 28e9, "FR2-1", 240_000, 63),
        ("F", 60e9, "FR2-2", 480_000, 63),
        ("G", 60e9, "FR2-2", 960_000, 63),
    ],
)
def test_blind_acquisition_recovers_high_candidate_across_ssb_numerologies(
    case: str,
    carrier_hz: float,
    frequency_range: str,
    scs_hz: int,
    candidate_index: int,
) -> None:
    fft_size = _FFT_SIZE
    half_frame_index = 0
    sfn = 0b0110101101
    candidate_positions = ssb_candidate_symbols(
        case,
        carrier_hz,
        subcarrier_spacing_hz=scs_hz,
        frequency_range=frequency_range,
    )
    candidate_count = len(candidate_positions)
    ssb, expected_mib = _make_ssb_resource_grid(
        sfn,
        candidate_count=candidate_count,
        ssb_index=candidate_index,
        half_frame=half_frame_index,
    )
    burst = nr_ssb_ofdm_modulate_half_frame(
        {candidate_index: ssb},
        case=case,
        carrier_frequency_hz=carrier_hz,
        subcarrier_spacing_hz=scs_hz,
        frequency_range=frequency_range,
        fft_size=fft_size,
    )
    start_symbol = candidate_positions[candidate_index]
    config = nr_ssb_waveform_config(
        subcarrier_spacing_hz=scs_hz,
        candidate_start_symbol=start_symbol,
        fft_size=fft_size,
    )
    mu = int(np.log2(scs_hz / 15_000))
    short_cp = 144 * fft_size // 2048
    long_cp = (144 + 16 * (2**mu)) * fft_size // 2048
    symbols_per_subframe = 14 * (2**mu)
    long_cp_symbol = 7 * (2**mu)
    symbol_offset = sum(
        fft_size + (long_cp if symbol % symbols_per_subframe in (0, long_cp_symbol) else short_cp)
        for symbol in range(start_symbol)
    )
    cp_lengths = config.cyclic_prefix_lengths

    leading_samples = 11
    channelled, noise_variance, _ = _apply_channel_and_noise(
        burst.waveform, seed=9300 + candidate_index, delay=5, snr_db=24.0
    )
    capture = np.concatenate(
        (
            np.zeros(leading_samples, dtype=np.complex128),
            channelled,
            np.zeros(fft_size + cp_lengths[-1], dtype=np.complex128),
        )
    )
    sample_rate_hz = float(fft_size * scs_hz)
    cfo_hz = -1_500.0
    capture *= np.exp(2j * np.pi * cfo_hz * np.arange(capture.size) / sample_rate_hz)

    result = nr_pbch_acquire_and_receive(
        capture,
        fft_size=fft_size,
        cyclic_prefix_lengths=cp_lengths,
        sampling_frequency_hz=sample_rate_hz,
        max_candidate_blocks_per_half_frame=candidate_count,
        noise_variance=noise_variance,
        cfo_hypotheses_hz=(-3_000.0, -1_500.0, 0.0),
        cfo_guard_samples=5,
    )

    assert result.cell_detection.sample_offset == leading_samples + symbol_offset
    assert result.cell_detection.physical_cell_id == _CELL_ID
    assert result.cell_detection.cfo_hz == pytest.approx(cfo_hz, abs=100.0)
    assert result.ssb_index_in_half_frame == candidate_index
    assert result.half_frame_index == half_frame_index
    assert result.pbch_reception.crc_valid
    assert result.pbch_reception.mib == expected_mib


@pytest.mark.native_sionna
@pytest.mark.parametrize(("candidate_count", "ssb_index"), [(10, 9), (20, 19), (64, 63)])
def test_blind_acquisition_reconstructs_high_ssb_index_from_pbch_payload(
    candidate_count: int, ssb_index: int
) -> None:
    sfn = 0b1100101011
    half_frame = 1
    waveform, expected_mib = _make_ssb_waveform(
        sfn,
        candidate_count=candidate_count,
        ssb_index=ssb_index,
        half_frame=half_frame,
    )
    timing_offset = 23
    sample_rate_hz = 3_840_000.0
    channelled, noise_variance, _ = _apply_channel_and_noise(
        waveform, seed=7300 + candidate_count, delay=5, snr_db=24.0
    )
    capture = np.concatenate(
        (
            np.zeros(timing_offset, dtype=np.complex128),
            channelled,
            np.zeros(_FFT_SIZE + _CP_LENGTHS[-1], dtype=np.complex128),
        )
    )

    result = nr_pbch_acquire_and_receive(
        capture,
        fft_size=_FFT_SIZE,
        cyclic_prefix_lengths=_CP_LENGTHS,
        sampling_frequency_hz=sample_rate_hz,
        max_candidate_blocks_per_half_frame=candidate_count,
        noise_variance=noise_variance,
    )

    assert result.cell_detection.sample_offset == timing_offset
    assert result.cell_detection.physical_cell_id == _CELL_ID
    assert result.ssb_index_in_half_frame == ssb_index
    assert result.half_frame_index == half_frame
    assert result.pbch_reception.crc_valid
    assert result.pbch_reception.mib == expected_mib
    assert result.pbch_reception.payload_recovery is not None
    assert result.pbch_reception.payload_recovery.system_frame_number == sfn
