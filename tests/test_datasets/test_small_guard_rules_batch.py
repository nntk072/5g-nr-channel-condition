"""Small input guards across analytical samples, observations, path powers, PBCH scrambling and identifiability."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from datasets import observations
from datasets.analytical import generate_analytical_multipath
from integrations.sionna import path_power_threshold, pbch_data
from propagation import identifiability


def test_analytical_samples_need_an_identity_and_metadata() -> None:
    sample = generate_analytical_multipath(seed=4, num_paths=2)
    with pytest.raises(ValueError, match="sample_id must be non-empty"):
        dataclasses.replace(sample, sample_id="   ")
    with pytest.raises(ValueError, match="metadata must not be empty"):
        dataclasses.replace(sample, metadata={})


def test_real_scalar_rejects_values_that_overflow_float() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        observations._real_scalar(10**400, name="snr_db")
    assert observations._real_scalar(2, name="snr_db") == 2.0


def test_path_power_mask_rejects_nonfinite_threshold_and_negative_powers() -> None:
    gains = np.asarray([1.0 + 0j, 0.5 + 0j])
    valid = np.asarray([True, True])
    with pytest.raises(ValueError, match="threshold_db must be finite and nonpositive"):
        path_power_threshold.relative_power_threshold_mask(gains, valid, 1.0)
    with pytest.raises(ValueError, match="threshold_db must be finite and nonpositive"):
        path_power_threshold.relative_power_threshold_mask(gains, valid, float("nan"))
    keep = path_power_threshold.relative_power_threshold_mask(gains, valid, -10.0)
    assert keep.tolist() == [True, True]


def test_pbch_scrambling_requires_an_integer_candidate_count() -> None:
    bits = np.zeros(864, dtype=np.uint8)
    with pytest.raises(TypeError, match="max_candidate_blocks_per_half_frame must be an integer"):
        pbch_data.nr_pbch_scramble_bits(bits, 0, 0, max_candidate_blocks_per_half_frame=4.0)


def test_identifiability_rank_tolerance_must_be_finite_and_positive() -> None:
    delays = np.asarray([0.0, 1e-7])
    frequencies = np.linspace(0.0, 1e6, 8)
    with pytest.raises(ValueError, match="rank_tolerance must be finite and positive"):
        identifiability.path_steering_diagnostics(delays, frequencies, rank_tolerance=0.0)
