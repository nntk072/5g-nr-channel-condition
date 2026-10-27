from __future__ import annotations

import numpy as np
import pytest
from integrations.sionna import pbch_acquisition


@pytest.mark.parametrize("sample_rate", [True, np.bool_(True), "30.72e6", 30.72e6 + 0j])
def test_pbch_acquisition_rejects_nonreal_sample_rate_before_cell_search(monkeypatch, sample_rate):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("cell acquisition must not run for invalid receiver configuration")

    monkeypatch.setattr(pbch_acquisition, "acquire_nr_cell", fail_if_called)
    with pytest.raises(TypeError, match="sampling_frequency_hz must be a real number"):
        pbch_acquisition.nr_pbch_acquire_and_receive(
            np.zeros(1, dtype=np.complex128),
            fft_size=512,
            cyclic_prefix_lengths=(36, 36, 36, 36),
            sampling_frequency_hz=sample_rate,
            max_candidate_blocks_per_half_frame=4,
            noise_variance=1.0,
        )


@pytest.mark.parametrize("sample_rate", [0.0, float("inf"), float("nan"), 10**400])
def test_pbch_acquisition_rejects_nonfinite_or_unrepresentable_sample_rate(
    monkeypatch, sample_rate
):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("cell acquisition must not run for invalid receiver configuration")

    monkeypatch.setattr(pbch_acquisition, "acquire_nr_cell", fail_if_called)
    with pytest.raises(ValueError, match="sampling_frequency_hz must be finite and positive"):
        pbch_acquisition.nr_pbch_acquire_and_receive(
            np.zeros(1, dtype=np.complex128),
            fft_size=512,
            cyclic_prefix_lengths=(36, 36, 36, 36),
            sampling_frequency_hz=sample_rate,
            max_candidate_blocks_per_half_frame=4,
            noise_variance=1.0,
        )


@pytest.mark.parametrize(
    ("device", "list_size", "exception", "message"),
    [
        ("cpu", True, TypeError, "list_size must be an integer"),
        ("cpu", 0, ValueError, "list_size must be positive"),
        ("cpu", 3, ValueError, "list_size must be a power of two"),
        ("cpu", 128, ValueError, "list_size must be at most 64"),
        (None, 8, TypeError, "device must be a string"),
        ("", 8, ValueError, "device must not be empty"),
    ],
)
def test_pbch_acquisition_preflights_decoder_configuration_before_cell_search(
    monkeypatch, device, list_size, exception, message
):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("cell acquisition must not run for invalid receiver configuration")

    monkeypatch.setattr(pbch_acquisition, "acquire_nr_cell", fail_if_called)
    with pytest.raises(exception, match=message):
        pbch_acquisition.nr_pbch_acquire_and_receive(
            np.zeros(1, dtype=np.complex128),
            fft_size=512,
            cyclic_prefix_lengths=(36, 36, 36, 36),
            sampling_frequency_hz=30.72e6,
            max_candidate_blocks_per_half_frame=4,
            noise_variance=1.0,
            device=device,
            list_size=list_size,
        )


@pytest.mark.parametrize(
    ("list_size", "exception", "message"),
    [
        (True, TypeError, "list_size must be an integer"),
        (3, ValueError, "list_size must be a power of two"),
        (128, ValueError, "list_size must be at most 64"),
    ],
)
def test_pbch_bch_decoder_bounds_scl_list_size_before_decoder_construction(
    monkeypatch, list_size, exception, message
):
    from integrations.sionna import pbch_bch

    monkeypatch.setattr(
        pbch_bch,
        "_polar_decoder",
        lambda *args: (_ for _ in ()).throw(AssertionError("decoder must not be constructed")),
    )
    with pytest.raises(exception, match=message):
        pbch_bch.nr_pbch_bch_decode(np.zeros(864, dtype=np.float32), list_size=list_size)


def test_pbch_acquisition_applies_the_pss_search_work_limit_before_capture_decode():
    with pytest.raises(ValueError, match="exceeds max_search_cells"):
        pbch_acquisition.nr_pbch_acquire_and_receive(
            np.ones(2000, dtype=np.complex64),
            fft_size=512,
            cyclic_prefix_lengths=(36, 36, 36, 36),
            sampling_frequency_hz=30.72e6,
            max_candidate_blocks_per_half_frame=4,
            noise_variance=1.0,
            max_search_cells=1,
        )
