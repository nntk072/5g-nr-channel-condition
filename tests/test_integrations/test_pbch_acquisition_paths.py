"""Blind PBCH acquisition contracts using deterministic sync/decoder doubles."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from integrations.sionna import pbch_acquisition as acquisition


def _detection(*, offset=0, cfo=0.0):
    return SimpleNamespace(sample_offset=offset, cfo_hz=cfo, physical_cell_id=42)


def _run(monkeypatch, receiver, *, count=4, capture=None, detection=None):
    observed = []
    monkeypatch.setattr(
        acquisition,
        "acquire_nr_cell",
        lambda samples, **kwargs: detection or _detection(),
    )

    def receive(samples, **kwargs):
        observed.append((samples.copy(), kwargs))
        return receiver(**kwargs)

    monkeypatch.setattr(acquisition, "nr_pbch_receive_ofdm", receive)
    return (
        acquisition.nr_pbch_acquire_and_receive(
            np.ones(48, dtype=np.complex64) if capture is None else capture,
            fft_size=8,
            cyclic_prefix_lengths=(2, 3, 2, 3),
            sampling_frequency_hz=1000.0,
            max_candidate_blocks_per_half_frame=count,
            noise_variance=0.1,
        ),
        observed,
    )


def test_acquisition_decodes_unique_crc_valid_half_frame_and_restores_receive_shape(monkeypatch):
    def receiver(**options):
        valid = options["ssb_index_in_half_frame"] == 2 and options["half_frame_index"] == 1
        return SimpleNamespace(
            crc_valid=valid,
            payload_recovery=SimpleNamespace(half_frame_index=1, ssb_index_msb_bits=())
            if valid
            else None,
        )

    result, calls = _run(monkeypatch, receiver, capture=np.ones((2, 48), dtype=np.complex64))
    assert result.ssb_index_in_half_frame == 2
    assert result.half_frame_index == 1
    assert result.cell_detection.physical_cell_id == 42
    assert len(calls) == 8
    assert calls[0][0].shape == (2, 42)
    assert calls[0][1]["physical_cell_id"] == 42


def test_acquisition_reconstructs_high_candidate_index_bits(monkeypatch):
    def receiver(**options):
        valid = options["ssb_index_in_half_frame"] == 3 and options["half_frame_index"] == 0
        return SimpleNamespace(
            crc_valid=valid,
            payload_recovery=SimpleNamespace(half_frame_index=0, ssb_index_msb_bits=(0, 1))
            if valid
            else None,
        )

    result, calls = _run(monkeypatch, receiver, count=20)
    assert result.ssb_index_in_half_frame == 11
    assert result.half_frame_index == 0
    assert len(calls) == 16


@pytest.mark.parametrize(
    "receiver",
    [
        lambda **_kwargs: SimpleNamespace(crc_valid=False, payload_recovery=None),
        lambda **kwargs: SimpleNamespace(
            crc_valid=kwargs["ssb_index_in_half_frame"] == 0,
            payload_recovery=None,
        ),
        lambda **kwargs: SimpleNamespace(
            crc_valid=kwargs["ssb_index_in_half_frame"] == 0,
            payload_recovery=SimpleNamespace(half_frame_index=2, ssb_index_msb_bits=()),
        ),
    ],
)
def test_acquisition_requires_one_crc_valid_consistent_occasion(monkeypatch, receiver):
    with pytest.raises(acquisition.PBCHAcquisitionError, match="no candidate"):
        _run(monkeypatch, receiver)


def test_acquisition_rejects_multiple_crc_valid_occasion_hypotheses(monkeypatch):
    def receiver(**kwargs):
        return SimpleNamespace(
            crc_valid=kwargs["ssb_index_in_half_frame"] in (0, 1)
            and kwargs["half_frame_index"] == 0,
            payload_recovery=SimpleNamespace(half_frame_index=0, ssb_index_msb_bits=()),
        )

    with pytest.raises(acquisition.PBCHAcquisitionError, match="ambiguous"):
        _run(monkeypatch, receiver)


def test_acquisition_rejects_crc_valid_payload_with_invalid_candidate_bits(monkeypatch):
    def receiver(**kwargs):
        valid = kwargs["ssb_index_in_half_frame"] == 0 and kwargs["half_frame_index"] == 0
        return SimpleNamespace(
            crc_valid=valid,
            payload_recovery=SimpleNamespace(half_frame_index=0, ssb_index_msb_bits=(1,))
            if valid
            else None,
        )

    with pytest.raises(acquisition.PBCHAcquisitionError, match="no candidate"):
        _run(monkeypatch, receiver, count=4)


def test_acquisition_rejects_truncated_ssb_after_cell_detection(monkeypatch):
    monkeypatch.setattr(acquisition, "acquire_nr_cell", lambda *_a, **_k: _detection(offset=5))
    with pytest.raises(ValueError, match="complete SS/PBCH block"):
        acquisition.nr_pbch_acquire_and_receive(
            np.ones(24, dtype=np.complex64),
            fft_size=8,
            cyclic_prefix_lengths=(2, 3, 2, 3),
            sampling_frequency_hz=1000.0,
            max_candidate_blocks_per_half_frame=4,
            noise_variance=0.1,
        )


@pytest.mark.parametrize(
    ("candidate_lsb", "bits", "count", "expected"),
    [
        (2, (), 8, 2),
        (2, (1,), 8, None),
        (1, (1,), 10, 9),
        (1, (1, 0), 10, None),
        (1, (2,), 10, None),
        (7, (1,), 10, None),
    ],
)
def test_candidate_index_reconstruction_enforces_payload_high_bits(
    candidate_lsb, bits, count, expected
):
    assert acquisition._reconstruct_candidate_index(candidate_lsb, bits, count) == expected


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"fft_size": True}, TypeError, "fft_size must be an integer"),
        ({"fft_size": 0}, ValueError, "fft_size must be positive"),
        ({"cyclic_prefix_lengths": (2, 3, 4, 3)}, ValueError, "PSS and SSS"),
        ({"cyclic_prefix_lengths": (True, 3, True, 3)}, TypeError, "must be integers"),
        ({"cyclic_prefix_lengths": (2, 3, 2)}, ValueError, "contain four lengths"),
        ({"cyclic_prefix_lengths": (0, 3, 0, 3)}, ValueError, r"in \[1, fft_size\)"),
        ({"max_candidate_blocks_per_half_frame": True}, TypeError, "must be an integer"),
        ({"max_candidate_blocks_per_half_frame": 6}, ValueError, "one of 4, 8"),
        (
            {"capture": np.ones(4, dtype=np.float32)},
            ValueError,
            "capture must be a nonempty complex",
        ),
        (
            {"capture": np.ones((2, 0), dtype=np.complex64)},
            ValueError,
            "capture must be a nonempty complex",
        ),
    ],
)
def test_acquisition_validates_geometry_candidate_and_capture_inputs(
    kwargs, error, message, monkeypatch
):
    values = {
        "fft_size": 8,
        "capture": np.ones(48, dtype=np.complex64),
        "cyclic_prefix_lengths": (2, 3, 2, 3),
        "sampling_frequency_hz": 1000.0,
        "max_candidate_blocks_per_half_frame": 4,
        "noise_variance": 0.1,
    }
    values.update(kwargs)
    with pytest.raises(error, match=message):
        acquisition.nr_pbch_acquire_and_receive(**values)
