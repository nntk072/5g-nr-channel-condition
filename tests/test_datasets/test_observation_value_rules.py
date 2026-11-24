"""Frequency-domain observation values: the receiver view's array, grid and scalar rules."""

from __future__ import annotations

import numpy as np
import pytest
from datasets.observations import (
    FrequencyDomainModelInputs,
    FrequencyDomainObservation,
    LinkedFrequencyDomainObservation,
)


def _inputs(**overrides):
    kwargs = {
        "observed_cfr": np.ones((1, 1, 8), dtype=np.complex64),
        "noise_variance": 0.1,
        "snr_db": 10.0,
        "carrier_hz": 3.5e9,
        "subcarrier_offsets_hz": (np.arange(8) - 4) * 15e3,
    }
    kwargs.update(overrides)
    return FrequencyDomainModelInputs(**kwargs)


def test_valid_inputs_are_accepted() -> None:
    assert _inputs().carrier_hz == 3.5e9


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        pytest.param({"observed_cfr": np.ones((1, 8))}, ValueError, "finite complex", id="rank"),
        pytest.param({"observed_cfr": np.ones((1, 1, 8))}, ValueError, "finite complex", id="real"),
        pytest.param(
            {"observed_cfr": np.full((1, 1, 8), np.nan + 0j)},
            ValueError,
            "finite complex",
            id="nonfinite",
        ),
        pytest.param(
            {"subcarrier_offsets_hz": np.array(["a"] * 8)},
            TypeError,
            "real numeric dtype",
            id="offset-dtype",
        ),
        pytest.param(
            {"subcarrier_offsets_hz": np.arange(7) * 1.0},
            ValueError,
            "finite, increasing, and include DC",
            id="offset-shape",
        ),
        pytest.param(
            {"subcarrier_offsets_hz": np.arange(8) * 1.0 + 1.0},
            ValueError,
            "include DC",
            id="offset-no-dc",
        ),
        pytest.param({"noise_variance": -1.0}, ValueError, "nonnegative", id="noise-negative"),
        pytest.param({"noise_variance": True}, TypeError, "real numeric scalar", id="noise-bool"),
        pytest.param(
            {"snr_db": float("inf")}, ValueError, "snr_db must be finite", id="snr-infinite"
        ),
        pytest.param(
            {"carrier_hz": 0.0},
            ValueError,
            "carrier_hz must be finite and positive",
            id="carrier-zero",
        ),
    ],
)
def test_observation_inputs_are_checked(overrides, error, message) -> None:
    with pytest.raises(error, match=message):
        _inputs(**overrides)


def test_observation_requires_canonical_sample_and_model_inputs() -> None:
    with pytest.raises(ValueError, match="sample_id must be a non-empty canonical string"):
        FrequencyDomainObservation(" padded", _inputs())
    with pytest.raises(TypeError, match="model_inputs must be FrequencyDomainModelInputs"):
        FrequencyDomainObservation("sample.0", "not-inputs")


def test_linked_observation_checks_identity_and_frame() -> None:
    observation = FrequencyDomainObservation("sample.0", _inputs())
    with pytest.raises(ValueError, match="link_id must be a non-empty canonical string"):
        LinkedFrequencyDomainObservation(" link", 0, observation)
    with pytest.raises(TypeError, match="frame_index must be an integer"):
        LinkedFrequencyDomainObservation("link", True, observation)
    with pytest.raises(ValueError, match="frame_index must be nonnegative"):
        LinkedFrequencyDomainObservation("link", -1, observation)
    with pytest.raises(TypeError, match="observation must be a FrequencyDomainObservation"):
        LinkedFrequencyDomainObservation("link", 0, "not-observation")
