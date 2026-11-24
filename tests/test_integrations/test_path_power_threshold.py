"""Relative path-power masks: dB levels, thresholds and input rules."""

from __future__ import annotations

import numpy as np
import pytest

from integrations.sionna.path_power_threshold import (
    relative_path_power_db,
    relative_power_threshold_mask,
)


def test_levels_are_relative_to_the_strongest_valid_path() -> None:
    gains = np.array([1.0, 0.5, 0.1, 2.0], dtype=np.complex128)
    valid = np.array([True, True, True, False])
    levels = relative_path_power_db(gains, valid)
    # The strongest valid path is the first (|a|^2 = 1); the padded path is -inf.
    assert levels[0] == pytest.approx(0.0)
    assert levels[1] == pytest.approx(10.0 * np.log10(0.25))
    assert levels[2] == pytest.approx(10.0 * np.log10(0.01))
    assert levels[3] == -np.inf


def test_zero_power_valid_paths_are_minus_infinity_and_never_selected() -> None:
    gains = np.array([0.0, 1.0j, 0.0], dtype=np.complex128)
    valid = np.array([True, True, False])
    levels = relative_path_power_db(gains, valid)
    assert levels[0] == -np.inf and levels[1] == pytest.approx(0.0)
    mask = relative_power_threshold_mask(gains, valid, -10.0)
    assert mask.tolist() == [False, True, False]


def test_threshold_keeps_valid_paths_no_weaker_than_the_bound() -> None:
    gains = np.array([1.0, 0.5, 0.1, 0.9], dtype=np.complex128)
    valid = np.array([True, True, True, True])
    mask = relative_power_threshold_mask(gains, valid, -10.0)
    # Amplitude 0.1 is -20 dB; amplitude 0.5 is -6 dB; amplitude 0.9 is about -0.9 dB.
    assert mask.tolist() == [True, True, False, True]
    tighter = relative_power_threshold_mask(gains, valid, -3.0)
    assert tighter.tolist() == [True, False, False, True]
    assert relative_power_threshold_mask(gains, valid, 0.0).tolist() == [True, False, False, False]


def test_threshold_never_admits_padded_paths() -> None:
    gains = np.array([1.0, 1.0], dtype=np.complex128)
    valid = np.array([True, False])
    assert relative_power_threshold_mask(gains, valid, -60.0).tolist() == [True, False]


@pytest.mark.parametrize(
    ("gains", "valid", "message"),
    [
        (np.ones((2, 2), dtype=np.complex128), np.ones((2, 2), dtype=bool), "one-dimensional complex"),
        (np.array([1.0, 2.0]), np.array([True, True]), "one-dimensional complex"),
        (np.ones(2, dtype=np.complex128), np.ones(3, dtype=bool), "boolean mask matching"),
        (np.ones(2, dtype=np.complex128), np.ones(2, dtype=int), "boolean mask matching"),
        (np.array([1.0 + 0j, np.nan]), np.array([True, True]), "must be finite"),
        (np.array([1e200 + 0j, 1.0 + 0j]), np.array([True, True]), "squared path powers must be finite"),
        (np.array([1.0 + 0j]), np.array([False]), "at least one valid path"),
        (np.array([0.0 + 0j]), np.array([True]), "at least one valid path"),
    ],
)
def test_levels_reject_invalid_path_vectors(gains, valid, message) -> None:
    with pytest.raises(ValueError, match=message):
        relative_path_power_db(gains, valid)


@pytest.mark.parametrize("threshold", [True, float("nan"), float("inf"), 0.5])
def test_threshold_must_be_finite_and_nonpositive(threshold) -> None:
    with pytest.raises(ValueError, match="finite and nonpositive"):
        relative_power_threshold_mask(np.ones(2, dtype=np.complex128), np.ones(2, dtype=bool), threshold)
