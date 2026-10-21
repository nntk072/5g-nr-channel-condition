"""Grouped data, profile estimator, and neural-pipeline contracts."""

from __future__ import annotations

import numpy as np
import pytest
from experiments.observable_multipath import (
    evaluate_profile_likelihood_sets,
    fit_two_path_decoder,
    generate_grouped_two_path_dataset,
    predict_decoder,
    profile_likelihood_delay,
    profile_likelihood_deviance_grid,
)


def test_grouped_dataset_is_deterministic_and_disjoint_by_group():
    first = generate_grouped_two_path_dataset(
        num_groups=80, samples_per_group=2, n_tones=16, seed=42
    )
    second = generate_grouped_two_path_dataset(
        num_groups=80, samples_per_group=2, n_tones=16, seed=42
    )

    assert first.source_sha256 == second.source_sha256
    np.testing.assert_array_equal(first.observed_cfr, second.observed_cfr)
    split_groups = {
        split: set(first.group_id[first.split == split])
        for split in ("train", "validation", "calibration", "test")
    }
    for left, left_groups in split_groups.items():
        for right, right_groups in split_groups.items():
            if left != right:
                assert left_groups.isdisjoint(right_groups)
    assert set(first.stratum).issubset({"close", "marginal", "separated"})
    assert np.all(first.pilot_mask.sum(axis=1) >= 2)


def test_profile_likelihood_recovers_noiseless_two_path_delay():
    frequencies = np.linspace(-4e6, 4e6, 64)
    delay = 0.23e-6
    gains = 0.9 + 0.2j, -0.35 + 0.4j
    observed = gains[0] + gains[1] * np.exp(-2j * np.pi * frequencies * delay)
    estimate, scale = profile_likelihood_delay(
        observed,
        np.ones(frequencies.size, dtype=bool),
        frequencies,
        1e-6,
        1e-8,
        grid_points=513,
    )

    assert abs(estimate - delay) < 1e-9
    assert 0 < scale <= 1e-6


def test_profile_likelihood_validates_observation_contract():
    with pytest.raises(ValueError, match="at least two"):
        profile_likelihood_delay(
            np.ones(8, dtype=complex),
            np.asarray([True, False, False, False, False, False, False, False]),
            np.arange(8, dtype=float),
            1e-6,
            1e-3,
        )


def test_profile_likelihood_deviance_is_minimized_near_noiseless_truth():
    frequencies = np.linspace(-4e6, 4e6, 64)
    delay = 0.23e-6
    gains = np.asarray([0.9 + 0.2j, -0.35 + 0.4j])
    observed = gains[0] + gains[1] * np.exp(-2j * np.pi * frequencies * delay)

    grid, deviance = profile_likelihood_deviance_grid(
        observed,
        np.ones(frequencies.size, dtype=bool),
        frequencies,
        1e-6,
        1e-8,
        grid_points=513,
    )

    assert abs(grid[int(np.argmin(deviance))] - delay) <= grid[1] - grid[0]
    assert deviance.min() == pytest.approx(0.0)


def test_profile_likelihood_sets_use_group_calibration_and_bounded_support():
    dataset = generate_grouped_two_path_dataset(
        num_groups=80, samples_per_group=1, n_tones=16, seed=231
    )

    report = evaluate_profile_likelihood_sets(dataset, coverage_level=0.8, grid_points=65)

    assert report["calibration"]["group_count"] == 12
    assert report["calibration"]["finite_sample_unbounded"] is False
    assert report["test"]["sample_count"] == dataset.indices("test").size
    assert 0.0 <= report["test"]["sample_coverage"] <= 1.0
    for row in report["test"]["rows"]:
        components = row["components_s"]
        assert components
        assert all(
            0 <= lower <= upper <= dataset.maximum_relative_delay_s for lower, upper in components
        )
        assert all(left[1] < right[0] for left, right in zip(components, components[1:]))


def test_profile_likelihood_set_retains_disconnected_delay_hypotheses():
    from experiments.observable_multipath import _profile_set_components

    components = _profile_set_components(
        np.linspace(0.0, 1.0, 5), np.asarray([0.0, 5.0, 0.0, 5.0, 0.0]), 1.0
    )

    assert len(components) == 3
    assert components[0][1] < components[1][0] < components[1][1] < components[2][0]


def test_neural_fit_and_batch_frequency_prediction_contract():
    pytest.importorskip("torch")
    dataset = generate_grouped_two_path_dataset(
        num_groups=80, samples_per_group=1, n_tones=16, seed=9
    )
    model, training = fit_two_path_decoder(
        dataset, seed=9, epochs=2, batch_size=32, patience=2, device="cpu"
    )
    delay, scale, reconstruction = predict_decoder(
        model, dataset, dataset.indices("test"), device="cpu"
    )

    assert training["epochs_completed"] == 2
    assert delay.shape == scale.shape == (dataset.indices("test").size,)
    assert np.all(np.isfinite(delay)) and np.all(np.isfinite(scale))
    assert np.all(scale > 0)
    assert reconstruction.shape == (dataset.indices("test").size, 16)
