"""Standards-sequence and controlled blind NR cell-acquisition tests."""

from typing import cast

import numpy as np
import pytest
from integrations.sionna.nr_sync import (
    NRSyncNotFoundError,
    acquire_nr_cell,
    decode_nr_sss_symbol,
    detect_nr_pss,
    make_nr_sync_burst,
    modulate_nr_sync_symbol,
    nr_pss_sequence,
    nr_sss_sequence,
)
from numpy.typing import NDArray


def test_pss_sequences_are_distinct_and_have_m_sequence_correlation() -> None:
    sequences = [nr_pss_sequence(nid2) for nid2 in range(3)]
    assert all(sequence.shape == (127,) for sequence in sequences)
    assert all(np.isin(sequence, (-1, 1)).all() for sequence in sequences)
    assert len({sequence.tobytes() for sequence in sequences}) == 3
    for sequence in sequences:
        correlations = [int(np.dot(sequence, np.roll(sequence, shift))) for shift in range(127)]
        assert correlations[0] == 127
        assert set(correlations[1:]) == {-1}


def test_pss_detection_whitens_a_high_energy_noisy_receive_chain() -> None:
    fft_size = 256
    cp_length = 32
    wrong_nid2 = 0
    expected_nid2 = 2
    noisy_interferer = 10.0 * modulate_nr_sync_symbol(
        nr_pss_sequence(wrong_nid2), fft_size=fft_size, cyclic_prefix_length=cp_length
    )
    clean_branch = modulate_nr_sync_symbol(
        nr_pss_sequence(expected_nid2), fft_size=fft_size, cyclic_prefix_length=cp_length
    )
    capture = np.stack((noisy_interferer, clean_branch))

    unweighted = detect_nr_pss(
        capture,
        fft_size=fft_size,
        cyclic_prefix_length=cp_length,
        sampling_frequency_hz=3_840_000.0,
    )
    weighted = detect_nr_pss(
        capture,
        fft_size=fft_size,
        cyclic_prefix_length=cp_length,
        sampling_frequency_hz=3_840_000.0,
        noise_variance=(1e6, 1.0),
    )

    assert unweighted.nid2 == wrong_nid2
    assert weighted.nid2 == expected_nid2


def test_sss_detection_whitens_a_high_energy_noisy_receive_chain() -> None:
    fft_size = 256
    cp_length = 32
    nid2 = 1
    wrong_nid1 = 17
    expected_nid1 = 251
    noisy_interferer = 10.0 * modulate_nr_sync_symbol(
        nr_sss_sequence(wrong_nid1, nid2), fft_size=fft_size, cyclic_prefix_length=cp_length
    )
    clean_branch = modulate_nr_sync_symbol(
        nr_sss_sequence(expected_nid1, nid2), fft_size=fft_size, cyclic_prefix_length=cp_length
    )
    symbol = np.stack((noisy_interferer, clean_branch))
    channel_estimate = np.stack((np.full(127, 10.0), np.ones(127))).astype(np.complex128)

    unweighted, _ = decode_nr_sss_symbol(
        symbol,
        nid2=nid2,
        fft_size=fft_size,
        cyclic_prefix_length=cp_length,
        channel_estimate=channel_estimate,
    )
    weighted, _ = decode_nr_sss_symbol(
        symbol,
        nid2=nid2,
        fft_size=fft_size,
        cyclic_prefix_length=cp_length,
        noise_variance=(1e6, 1.0),
        channel_estimate=channel_estimate,
    )

    assert unweighted == wrong_nid1
    assert weighted == expected_nid1


def test_sss_generates_distinct_sequences_for_all_nid1_values() -> None:
    sequences = [nr_sss_sequence(nid1, 1).tobytes() for nid1 in range(336)]
    assert len(set(sequences)) == 336
    assert all(
        len(nr_sss_sequence(nid1, nid2)) == 127 for nid1 in (0, 111, 112, 335) for nid2 in range(3)
    )


def test_pss_sss_sequences_and_ssb_mapping_match_ts_38211_v19_1() -> None:
    def reference_m_sequence(initial: tuple[int, ...], tap_offset: int) -> list[int]:
        sequence = list(initial)
        for index in range(127):
            sequence.append((sequence[index + tap_offset] + sequence[index]) % 2)
        return sequence

    pss_binary = reference_m_sequence((0, 1, 1, 0, 1, 1, 1), 4)
    sss_x0 = reference_m_sequence((1, 0, 0, 0, 0, 0, 0), 4)
    sss_x1 = reference_m_sequence((1, 0, 0, 0, 0, 0, 0), 1)
    indexes = np.arange(127)

    for nid2 in range(3):
        expected_pss = np.asarray(
            [1 - 2 * pss_binary[(int(n) + 43 * nid2) % 127] for n in indexes],
            dtype=np.int8,
        )
        np.testing.assert_array_equal(nr_pss_sequence(nid2), expected_pss)
        for nid1 in range(336):
            m0 = 15 * (nid1 // 112) + 5 * nid2
            m1 = nid1 % 112
            expected_sss = np.asarray(
                [
                    (1 - 2 * sss_x0[(int(n) + m0) % 127]) * (1 - 2 * sss_x1[(int(n) + m1) % 127])
                    for n in indexes
                ],
                dtype=np.int8,
            )
            np.testing.assert_array_equal(nr_sss_sequence(nid1, nid2), expected_sss)

    fft_size = 256
    cp_length = 32
    nid1, nid2 = 335, 2
    burst = make_nr_sync_burst(
        nid1,
        nid2,
        fft_size=fft_size,
        cyclic_prefix_length=cp_length,
    ).reshape(4, fft_size + cp_length)
    subcarriers = np.arange(56, 183)
    fft_bins = (subcarriers - 120) % fft_size
    for symbol_index, sequence in (
        (0, nr_pss_sequence(nid2)),
        (2, nr_sss_sequence(nid1, nid2)),
    ):
        expected = np.zeros(fft_size, dtype=np.complex128)
        expected[fft_bins] = sequence
        actual = np.fft.fft(burst[symbol_index, cp_length:])
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0.0)
    np.testing.assert_array_equal(burst[[1, 3]], 0.0)


def test_blind_pss_sss_acquisition_recovers_timing_cfo_and_cell_id() -> None:
    nid1 = 237
    nid2 = 2
    fft_size = 256
    cp_length = 32
    sample_rate_hz = 7_680_000.0
    timing_offset = 23
    cfo_hz = 3_000.0
    burst = make_nr_sync_burst(
        nid1,
        nid2,
        fft_size=fft_size,
        cyclic_prefix_length=cp_length,
    )
    rng = np.random.default_rng(9521)
    gain = 0.7 - 0.4j
    capture = np.concatenate((np.zeros(timing_offset), burst, np.zeros(48)))
    time = np.arange(capture.size) / sample_rate_hz
    capture = gain * capture * np.exp(2j * np.pi * cfo_hz * time)
    noise_power = np.mean(np.abs(capture) ** 2) / 10 ** (18.0 / 10.0)
    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(capture.size) + 1j * rng.standard_normal(capture.size)
    )

    result = acquire_nr_cell(
        capture + noise,
        fft_size=fft_size,
        cyclic_prefix_length=cp_length,
        sampling_frequency_hz=sample_rate_hz,
        cfo_hypotheses_hz=tuple(float(value) for value in range(-6_000, 6_001, 1_000)),
    )

    assert result.sample_offset == timing_offset
    assert result.nid1 == nid1
    assert result.nid2 == nid2
    assert result.physical_cell_id == 3 * nid1 + nid2
    assert result.cfo_hz == pytest.approx(cfo_hz, abs=100.0)
    assert result.pss_score > 0.9
    assert result.sss_score > 0.9


def test_nr_sync_rejects_invalid_identity_and_capture() -> None:
    with pytest.raises(ValueError, match="nid2"):
        nr_pss_sequence(3)
    with pytest.raises(ValueError, match="nid1"):
        nr_sss_sequence(336, 0)
    with pytest.raises(ValueError, match="too short to contain both"):
        acquire_nr_cell(
            np.ones(16, dtype=np.complex128),
            fft_size=256,
            cyclic_prefix_length=32,
            sampling_frequency_hz=7_680_000.0,
        )


def test_pss_search_rejects_noise_below_detection_threshold() -> None:
    rng = np.random.default_rng(361)
    noise = rng.standard_normal(4096) + 1j * rng.standard_normal(4096)
    with pytest.raises(NRSyncNotFoundError, match="below minimum"):
        acquire_nr_cell(
            noise,
            fft_size=256,
            cyclic_prefix_length=32,
            sampling_frequency_hz=7_680_000.0,
            cfo_hypotheses_hz=(-3_000.0, 0.0, 3_000.0),
        )


def test_cell_search_excludes_late_pss_candidates_without_room_for_sss() -> None:
    fft_size = 256
    cp_length = 32
    symbol_length = fft_size + cp_length
    pss = modulate_nr_sync_symbol(
        nr_pss_sequence(1), fft_size=fft_size, cyclic_prefix_length=cp_length
    )
    capture = np.zeros(4 * symbol_length, dtype=np.complex128)
    capture[3 * symbol_length :] = pss

    with pytest.raises(NRSyncNotFoundError, match="below minimum"):
        acquire_nr_cell(
            capture,
            fft_size=fft_size,
            cyclic_prefix_length=cp_length,
            sampling_frequency_hz=7_680_000.0,
            minimum_pss_score=0.99,
            minimum_sss_score=0.0,
        )


def test_cell_search_requires_room_for_pss_and_later_sss_symbols() -> None:
    with pytest.raises(ValueError, match="too short to contain both"):
        acquire_nr_cell(
            np.zeros(2 * (256 + 32), dtype=np.complex128),
            fft_size=256,
            cyclic_prefix_length=32,
            sampling_frequency_hz=7_680_000.0,
        )


def test_pss_sss_acquisition_uses_pss_channel_estimate_under_in_cp_multipath() -> None:
    nid1 = 335
    nid2 = 1
    fft_size = 256
    cp_length = 32
    sample_rate_hz = 7_680_000.0
    timing_offset = 17
    cfo_hz = 3_000.0
    burst = make_nr_sync_burst(
        nid1,
        nid2,
        fft_size=fft_size,
        cyclic_prefix_length=cp_length,
    )
    channel = np.zeros(13, dtype=np.complex128)
    channel[0] = 1.0
    channel[-1] = 0.9 + 0.1j
    capture = np.convolve(burst, channel)
    capture = np.concatenate((np.zeros(timing_offset), capture, np.zeros(64)))
    time = np.arange(capture.size) / sample_rate_hz
    capture *= np.exp(2j * np.pi * cfo_hz * time)
    rng = np.random.default_rng(41)
    noise_power = np.mean(np.abs(capture) ** 2) / 10 ** (12.0 / 10.0)
    capture += np.sqrt(noise_power / 2) * (
        rng.standard_normal(capture.size) + 1j * rng.standard_normal(capture.size)
    )

    result = acquire_nr_cell(
        capture,
        fft_size=fft_size,
        cyclic_prefix_length=cp_length,
        sampling_frequency_hz=sample_rate_hz,
        cfo_hypotheses_hz=(0.0, 1_000.0, 2_000.0, 3_000.0, 4_000.0),
        cfo_guard_samples=12,
    )

    symbol_length = fft_size + cp_length
    sss_start = result.sample_offset + 2 * symbol_length
    corrected_sss = capture[sss_start : sss_start + symbol_length] * np.exp(
        -2j * np.pi * result.cfo_hz * np.arange(symbol_length) / sample_rate_hz
    )
    flat_nid1, flat_score = decode_nr_sss_symbol(
        corrected_sss,
        nid2=nid2,
        fft_size=fft_size,
        cyclic_prefix_length=cp_length,
    )

    assert result.sample_offset == timing_offset
    assert result.nid1 == nid1
    assert result.nid2 == nid2
    assert result.physical_cell_id == 3 * nid1 + nid2
    assert result.cfo_hz == pytest.approx(cfo_hz, abs=150.0)
    assert flat_nid1 == nid1
    assert result.pss_score > 0.7
    assert result.sss_score > flat_score + 0.15
    assert result.sss_score > 0.97


def test_acquisition_refines_off_grid_cfo_and_fractional_sample_timing() -> None:
    nid1 = 81
    nid2 = 2
    fft_size = 256
    cp_length = 32
    sample_rate_hz = 7_680_000.0
    integer_timing = 17
    fractional_timing = 0.375
    cfo_hz = 3_500.0
    burst = make_nr_sync_burst(
        nid1,
        nid2,
        fft_size=fft_size,
        cyclic_prefix_length=cp_length,
        fractional_delay_samples=fractional_timing,
    )
    rng = np.random.default_rng(638)
    gain = 0.7 - 0.4j
    capture = np.concatenate((np.zeros(integer_timing), burst, np.zeros(48)))
    time = np.arange(capture.size) / sample_rate_hz
    capture = gain * capture * np.exp(2j * np.pi * cfo_hz * time)
    noise_power = np.mean(np.abs(capture) ** 2) / 10 ** (18.0 / 10.0)
    capture += np.sqrt(noise_power / 2) * (
        rng.standard_normal(capture.size) + 1j * rng.standard_normal(capture.size)
    )

    result = acquire_nr_cell(
        capture,
        fft_size=fft_size,
        cyclic_prefix_length=cp_length,
        sampling_frequency_hz=sample_rate_hz,
        cfo_hypotheses_hz=(-6_000.0, -3_000.0, 0.0, 3_000.0, 6_000.0),
        fractional_timing_hypotheses_samples=(
            -0.375,
            -0.25,
            -0.125,
            0.0,
            0.125,
            0.25,
            0.375,
        ),
    )

    assert result.sample_offset == integer_timing
    assert result.fractional_sample_offset == pytest.approx(fractional_timing, abs=0.01)
    assert result.nid1 == nid1
    assert result.nid2 == nid2
    assert result.physical_cell_id == 3 * nid1 + nid2
    assert result.cfo_hz == pytest.approx(cfo_hz, abs=100.0)
    assert result.pss_score > 0.9
    assert result.sss_score > 0.9


@pytest.mark.parametrize(
    "fractional_timing",
    (-0.44, -0.31, -0.073, 0.041, 0.284, 0.463),
)
def test_pss_fractional_timing_refinement_is_continuous(fractional_timing: float) -> None:
    fft_size = 256
    cp_length = 32
    integer_timing = 11
    sample_rate_hz = 7_680_000.0
    burst = make_nr_sync_burst(
        119,
        1,
        fft_size=fft_size,
        cyclic_prefix_length=cp_length,
        fractional_delay_samples=fractional_timing,
    )
    capture = np.concatenate(
        (np.zeros(integer_timing), burst, np.zeros(2 * (fft_size + cp_length)))
    )
    capture = cast(NDArray[np.complexfloating], np.asarray(capture, dtype=np.complex128))

    result = acquire_nr_cell(
        capture,
        fft_size=fft_size,
        cyclic_prefix_length=cp_length,
        sampling_frequency_hz=sample_rate_hz,
        fractional_timing_hypotheses_samples=(-0.375, -0.1875, 0.0, 0.1875, 0.375),
        minimum_pss_score=0.0,
        minimum_sss_score=0.0,
    )

    assert result.sample_offset == integer_timing
    assert result.fractional_sample_offset == pytest.approx(fractional_timing, abs=0.005)
    assert result.nid1 == 119
    assert result.nid2 == 1


def test_fractional_timing_acquisition_recovers_cell_under_multipath_and_awgn() -> None:
    fft_size = 256
    cp_length = 32
    sample_rate_hz = 7_680_000.0
    nid1 = 119
    nid2 = 1
    fractional_offsets = (-0.44, -0.31, -0.17, -0.06, 0.08, 0.19, 0.33, 0.44)
    timing_errors = []
    cfo_errors = []

    for seed in range(32):
        rng = np.random.default_rng(9000 + seed)
        fractional_offset = fractional_offsets[seed % len(fractional_offsets)]
        echo_delay = int(rng.integers(3, 15))
        echo_gain = rng.uniform(0.2, 0.65) * np.exp(1j * rng.uniform(-np.pi, np.pi))
        channel = np.zeros(echo_delay + 1, dtype=np.complex128)
        channel[0] = 1.0
        channel[-1] = echo_gain
        burst = make_nr_sync_burst(
            nid1,
            nid2,
            fft_size=fft_size,
            cyclic_prefix_length=cp_length,
            fractional_delay_samples=fractional_offset,
        )
        received = np.convolve(burst, channel)
        integer_offset = int(rng.integers(7, 24))
        capture = np.concatenate(
            (
                np.zeros(integer_offset, dtype=np.complex128),
                received,
                np.zeros(2 * (fft_size + cp_length), dtype=np.complex128),
            )
        )
        cfo_hz = float(rng.uniform(-4_500.0, 4_500.0))
        time_s = np.arange(capture.size) / sample_rate_hz
        capture *= np.exp(2j * np.pi * cfo_hz * time_s)
        # Define SNR over the complete captured buffer, including its guard zeros.
        signal_power = np.mean(np.abs(capture) ** 2)
        noise_power = signal_power / 10 ** (15.0 / 10.0)
        capture += np.sqrt(noise_power / 2) * (
            rng.standard_normal(capture.size) + 1j * rng.standard_normal(capture.size)
        )

        result = acquire_nr_cell(
            capture,
            fft_size=fft_size,
            cyclic_prefix_length=cp_length,
            sampling_frequency_hz=sample_rate_hz,
            cfo_hypotheses_hz=tuple(float(value) for value in range(-6_000, 6_001, 1_000)),
            fractional_timing_hypotheses_samples=(-0.375, -0.1875, 0.0, 0.1875, 0.375),
            cfo_guard_samples=echo_delay,
        )

        timing_errors.append(
            abs(
                result.sample_offset
                + result.fractional_sample_offset
                - integer_offset
                - fractional_offset
            )
        )
        cfo_errors.append(abs(result.cfo_hz - cfo_hz))
        assert result.nid1 == nid1
        assert result.nid2 == nid2
        assert result.physical_cell_id == 3 * nid1 + nid2
        assert result.pss_score > 0.7
        assert result.sss_score > 0.9

    assert np.quantile(timing_errors, 0.95) < 0.14
    assert max(timing_errors) < 0.15
    assert np.quantile(cfo_errors, 0.95) < 200.0
    assert max(cfo_errors) < 300.0


@pytest.mark.parametrize("hypotheses", [(), (-0.5, 0.5), (0.0, 0.0), (np.nan,)])
def test_pss_search_rejects_invalid_fractional_timing_hypotheses(hypotheses) -> None:
    with pytest.raises(ValueError, match="fractional timing hypotheses"):
        acquire_nr_cell(
            np.ones(4 * (256 + 32), dtype=np.complex128),
            fft_size=256,
            cyclic_prefix_length=32,
            sampling_frequency_hz=7_680_000.0,
            fractional_timing_hypotheses_samples=hypotheses,
        )
