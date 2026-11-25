"""PBCH OFDM admission rules and interpolation guards without PHY execution."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from integrations.sionna import pbch_ofdm_receiver as receiver


def _receive(samples, **overrides):
    arguments = {
        "received_samples": samples,
        "fft_size": 256,
        "cyclic_prefix_lengths": (1, 1, 1, 1),
        "physical_cell_id": 1,
        "ssb_index_in_half_frame": 0,
        "half_frame_index": 0,
        "max_candidate_blocks_per_half_frame": 4,
        "noise_variance": 1.0,
    }
    arguments.update(overrides)
    return receiver.nr_pbch_receive_ofdm(**arguments)


@pytest.mark.parametrize("fft_size", [240, 255])
def test_pbch_ofdm_rejects_small_or_odd_fft_sizes(fft_size):
    with pytest.raises(ValueError, match="even and greater than 240"):
        _receive(np.zeros(4 * 257, dtype=np.complex128), fft_size=fft_size)


@pytest.mark.parametrize(
    "samples",
    [
        np.zeros(10),
        np.zeros((2, 3, 4), dtype=np.complex128),
        np.asarray([complex(float("inf"), 0.0)]),
    ],
)
def test_pbch_ofdm_rejects_nonfinite_or_wrong_rank_samples(samples):
    with pytest.raises(ValueError, match="finite complex vector or receive-chain matrix"):
        _receive(samples)


def test_pbch_ofdm_rejects_sample_count_mismatch_before_demodulation():
    with pytest.raises(ValueError, match="must have shape"):
        _receive(np.zeros(8, dtype=np.complex128))


@pytest.mark.parametrize(
    ("value", "exception", "message"),
    [
        (True, TypeError, "must be an integer"),
        (1.5, TypeError, "must be an integer"),
        (0, ValueError, "must be positive"),
    ],
)
def test_positive_integer_helper_rejects_invalid_values(value, exception, message):
    with pytest.raises(exception, match=message):
        receiver._positive_integer(value, "probe")


@pytest.mark.parametrize(
    ("value", "exception", "message"),
    [
        (True, TypeError, "real number"),
        (1 + 0j, TypeError, "real number"),
        (float("inf"), ValueError, "finite and positive"),
        (0.0, ValueError, "finite and positive"),
    ],
)
def test_positive_real_helper_rejects_invalid_values(value, exception, message):
    with pytest.raises(exception, match=message):
        receiver._positive_real(value, "probe")


def test_noise_variance_expands_scalar_and_checks_per_chain_values():
    np.testing.assert_array_equal(receiver._noise_variance_by_chain(2, 2), [2.0, 2.0])
    with pytest.raises(TypeError, match="not booleans"):
        receiver._noise_variance_by_chain(True, 1)
    with pytest.raises(TypeError, match="not booleans"):
        receiver._noise_variance_by_chain([1.0, True], 2)
    with pytest.raises(ValueError, match="one value per receive chain"):
        receiver._noise_variance_by_chain([1.0], 2)
    with pytest.raises(TypeError, match="must be real numbers"):
        receiver._noise_variance_by_chain([1.0 + 0j, 2.0 + 0j], 2)
    with pytest.raises(ValueError, match="finite and positive"):
        receiver._noise_variance_by_chain([1.0, 0.0], 2)


@pytest.mark.parametrize(
    ("values", "exception", "message"),
    [
        ("1234", ValueError, "four lengths"),
        ((1, 2, 3), ValueError, "four lengths"),
        ((1, 1, True, 1), TypeError, "must be integers"),
        ((1, 1, 0, 1), ValueError, r"in \[1, fft_size\)"),
        ((1, 1, 256, 1), ValueError, r"in \[1, fft_size\)"),
    ],
)
def test_cyclic_prefix_validation_rejects_invalid_lengths(values, exception, message):
    with pytest.raises(exception, match=message):
        receiver._validate_cp_lengths(values, 256)


def test_pbch_receiver_rejects_dmrs_symbol_with_too_few_pilots(monkeypatch):
    monkeypatch.setattr(
        receiver,
        "nr_ssb_ofdm_demodulate",
        lambda *_args, **_kwargs: np.zeros((4, 240), dtype=np.complex128),
    )
    monkeypatch.setattr(
        receiver,
        "nr_pbch_dmrs_ssb",
        lambda *_args, **_kwargs: SimpleNamespace(
            ofdm_symbols=np.asarray([0]),
            subcarriers=np.asarray([0]),
            sequence=np.asarray([1 + 0j]),
        ),
    )
    with pytest.raises(RuntimeError, match="at least two DM-RS pilots"):
        _receive(np.zeros(4 * 257, dtype=np.complex128))


def test_pbch_receiver_rejects_zero_combined_channel_estimate(monkeypatch):
    monkeypatch.setattr(
        receiver,
        "nr_ssb_ofdm_demodulate",
        lambda *_args, **_kwargs: np.zeros((4, 240), dtype=np.complex128),
    )
    monkeypatch.setattr(
        receiver,
        "nr_pbch_dmrs_ssb",
        lambda *_args, **_kwargs: SimpleNamespace(
            ofdm_symbols=np.asarray([1, 1, 2, 2, 3, 3]),
            subcarriers=np.asarray([0, 1, 0, 1, 0, 1]),
            sequence=np.ones(6, dtype=np.complex128),
        ),
    )
    monkeypatch.setattr(
        receiver,
        "_pbch_data_coordinates",
        lambda _dmrs: (np.zeros(432, dtype=np.intp), np.ones(432, dtype=np.intp)),
    )
    with pytest.raises(ValueError, match="combined channel estimate is zero"):
        _receive(np.zeros(4 * 257, dtype=np.complex128))
