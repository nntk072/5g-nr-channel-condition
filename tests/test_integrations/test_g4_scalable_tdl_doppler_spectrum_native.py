"""Frozen Jakes-spectrum checks for scalable TDL diffuse taps."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pytest
from evaluation.g4_doppler_spectrum import (
    cluster_spectrum_cdf,
    jakes_doppler_cdf,
    simultaneous_spectrum_cdf_bounds,
)
from evaluation.g4_tdl_covariance_reference import scalable_tdl_rayleigh_tap_indices

_ROOT = Path(__file__).resolve().parents[2]
_REFERENCE_PATH = _ROOT / "docs/research/G4_SCALABLE_TDL_POWER_DELAY_REFERENCE_2026-10-26.json"
_REFERENCE_SHA256 = "38b897029bd4cc6de49eece5096e69c75291eaf6dac45bebe04767ed24a4f0bd"
_SOURCE_SHA256 = "b71460fb4978a59fbffc864652016cf161c9e17cce19f0d0e9b46bea630b1cc4"
_SEED_AUDIT_PATH = (
    _ROOT / "docs/research/G4_SCALABLE_TDL_DOPPLER_SPECTRUM_SEED_AUDIT_2026-10-27.json"
)
_SEED_AUDIT_SHA256 = "3a4c6cad865a0e5fa3ee07c166c8ebc70c8bc4bb39a1ea017d7b8400def76216"
_MODELS = tuple("ABCDE")
_SEED_START = 1_743_000
_DRAW_COUNT = 256
_TIME_COUNT = 2_048
_SINUSOID_COUNT = 20
_FAMILYWISE_ALPHA = 0.01
_MAXIMUM_CDF_ERROR = 0.05
_CUTOFFS = np.asarray((-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75), dtype=np.float64)
_CLAIM_COUNT = len(_MODELS) * len(_CUTOFFS)


def _load_json(path: Path, digest: str) -> dict[str, Any]:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError(f"{path.name} differs from its pinned digest")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} root must be an object")
    return value


def test_jakes_spectrum_protocol_is_bound_to_independent_tables_and_audited_seeds() -> None:
    reference = _load_json(_REFERENCE_PATH, _REFERENCE_SHA256)
    assert reference["schema"] == "loslab.g4_scalable_tdl_power_delay_reference.v1"
    assert reference["source_sha256"] == _SOURCE_SHA256
    assert reference["standard"] == "3GPP TR 38.901"
    assert reference["version"] == "19.2.0"
    tap_counts = {model: len(scalable_tdl_rayleigh_tap_indices(model)) for model in _MODELS}
    assert tap_counts == {"A": 23, "B": 23, "C": 24, "D": 12, "E": 13}
    assert scalable_tdl_rayleigh_tap_indices("A") == tuple(range(23))
    assert scalable_tdl_rayleigh_tap_indices("B") == tuple(range(23))
    assert scalable_tdl_rayleigh_tap_indices("C") == tuple(range(24))
    assert scalable_tdl_rayleigh_tap_indices("D") == tuple(range(1, 13))
    assert scalable_tdl_rayleigh_tap_indices("E") == tuple(range(1, 14))

    audit = _load_json(_SEED_AUDIT_PATH, _SEED_AUDIT_SHA256)
    assert audit["status"] == "CLEAR_WITHIN_SCANNED_JSON_CORPUS"
    assert audit["reserved_seeds"] == [_SEED_START + index for index in range(len(_MODELS))]
    assert audit["collisions"] == []
    assert audit["malformed_seed_fields"] == []
    assert audit["incomplete_files"] == []
    assert _CLAIM_COUNT == 35
    assert NormalDist().inv_cdf(1.0 - _FAMILYWISE_ALPHA / (2.0 * _CLAIM_COUNT)) > 0.0


def test_jakes_doppler_cdf_matches_arcsine_distribution() -> None:
    np.testing.assert_allclose(
        jakes_doppler_cdf(np.asarray((-2.0, -1.0, 0.0, 1.0, 2.0))),
        np.asarray((0.0, 0.0, 0.5, 1.0, 1.0)),
        atol=1e-15,
        rtol=0.0,
    )


def test_cluster_spectrum_cdf_recovers_equal_positive_and_negative_tones() -> None:
    sample_count = 256
    sample_rate = 4.0
    time = np.arange(sample_count) / sample_rate
    frequencies = (-0.5, 0.5)
    channels = np.stack(
        [
            np.stack([np.exp(2j * np.pi * frequency * time) for frequency in frequencies])
            for _ in range(4)
        ]
    )
    cdf = cluster_spectrum_cdf(
        channels,
        sampling_frequency_hz=sample_rate,
        maximum_doppler_hz=1.0,
        normalized_frequency_cutoffs=(-0.25, 0.0, 0.75),
    )
    assert cdf.shape == (4, 3)
    np.testing.assert_allclose(
        cdf, np.tile(np.asarray((0.5, 0.5, 1.0)), (4, 1)), atol=0.005, rtol=0.0
    )


def test_simultaneous_spectrum_cdf_bounds_keep_channel_draw_as_unit() -> None:
    expected = np.asarray((0.25, 0.5, 0.75))
    draws = np.tile(expected, (32, 1))
    result = simultaneous_spectrum_cdf_bounds(
        draws,
        expected_cdf=expected,
        familywise_alpha=0.01,
        maximum_absolute_error=0.05,
        family_claim_count=35,
    )
    assert result["draw_count"] == 32
    assert result["claim_count"] == 35
    assert result["status"] is True


@pytest.mark.native_sionna
@pytest.mark.parametrize("model", _MODELS)
def test_scalable_tdl_diffuse_doppler_spectrum_matches_jakes(model: str) -> None:
    """Check a predeclared spectrum-CDF family on independent channel draws."""
    pytest.importorskip("torch")
    pytest.importorskip("sionna.phy")
    from integrations.sionna.tdl import generate_tdl_frequency_response

    carrier_hz = 3.5e9
    speed_mps = 30.0
    maximum_doppler_hz = speed_mps * carrier_hz / 299_792_458.0
    channel = generate_tdl_frequency_response(
        model=model,
        seed=_SEED_START + _MODELS.index(model),
        carrier_frequency_hz=carrier_hz,
        subcarrier_spacing_hz=30_000.0,
        num_subcarriers=8,
        batch_size=_DRAW_COUNT,
        delay_spread_s=100e-9,
        num_time_steps=_TIME_COUNT,
        sampling_frequency_hz=4.0 * maximum_doppler_hz,
        min_speed_mps=speed_mps,
        max_speed_mps=speed_mps,
        num_sinusoids=_SINUSOID_COUNT,
        spec_version="19.2",
        device="cpu",
    )
    assert channel.metadata["batch_axis_semantics"] == "independent_stochastic_channel_draws"
    assert channel.metadata["maximum_doppler_hz"] == pytest.approx(maximum_doppler_hz)
    assert channel.metadata["num_sinusoids"] == _SINUSOID_COUNT
    path = channel.h_path.detach().cpu().numpy()[:, 0, 0, 0, 0, :, :]
    tap_indices = scalable_tdl_rayleigh_tap_indices(model)
    assert path.shape == (
        _DRAW_COUNT,
        len(tap_indices) + (1 if model in {"D", "E"} else 0),
        _TIME_COUNT,
    )
    per_draw = cluster_spectrum_cdf(
        path[:, tap_indices, :],
        sampling_frequency_hz=4.0 * maximum_doppler_hz,
        maximum_doppler_hz=maximum_doppler_hz,
        normalized_frequency_cutoffs=_CUTOFFS,
    )
    result = simultaneous_spectrum_cdf_bounds(
        per_draw,
        expected_cdf=jakes_doppler_cdf(_CUTOFFS),
        familywise_alpha=_FAMILYWISE_ALPHA,
        maximum_absolute_error=_MAXIMUM_CDF_ERROR,
        family_claim_count=_CLAIM_COUNT,
    )
    assert result["claim_count"] == _CLAIM_COUNT
    assert result["status"] is True
