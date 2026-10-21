"""Standards-sequence and controlled blind NR cell-acquisition tests."""

import numpy as np
import pytest
from integrations.sionna.nr_sync import (
    NRSyncNotFoundError,
    acquire_nr_cell,
    make_nr_sync_burst,
    nr_pss_sequence,
    nr_sss_sequence,
)


def test_pss_sequences_are_distinct_and_have_m_sequence_correlation() -> None:
    sequences = [nr_pss_sequence(nid2) for nid2 in range(3)]
    assert all(sequence.shape == (127,) for sequence in sequences)
    assert all(np.isin(sequence, (-1, 1)).all() for sequence in sequences)
    assert len({sequence.tobytes() for sequence in sequences}) == 3
    for sequence in sequences:
        correlations = [int(np.dot(sequence, np.roll(sequence, shift))) for shift in range(127)]
        assert correlations[0] == 127
        assert set(correlations[1:]) == {-1}


def test_sss_generates_distinct_sequences_for_all_nid1_values() -> None:
    sequences = [nr_sss_sequence(nid1, 1).tobytes() for nid1 in range(336)]
    assert len(set(sequences)) == 336
    assert all(
        len(nr_sss_sequence(nid1, nid2)) == 127 for nid1 in (0, 111, 112, 335) for nid2 in range(3)
    )


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
    assert result.cfo_hz == cfo_hz
    assert result.pss_score > 0.9
    assert result.sss_score > 0.9


def test_nr_sync_rejects_invalid_identity_and_capture() -> None:
    with pytest.raises(ValueError, match="nid2"):
        nr_pss_sequence(3)
    with pytest.raises(ValueError, match="nid1"):
        nr_sss_sequence(336, 0)
    with pytest.raises(ValueError, match="shorter than one PSS"):
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
