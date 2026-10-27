"""Frozen Jakes-spectrum checks for fixed TDL diffuse taps."""

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
from evaluation.g4_tdl_covariance_reference import fixed_tdl_rayleigh_tap_indices

_ROOT = Path(__file__).resolve().parents[2]
_REFERENCE_PATH = _ROOT / "docs/research/G4_FIXED_TDL_TAP_PROFILE_REFERENCE_2026-10-26.json"
_REFERENCE_SHA256 = "fe22560a71f4e5683ea6780f73e84466155b26f5f97cc4b63e3250c102fafda3"
_SOURCE_SHA256 = "b3c6a73c7a40219e7270c08c7a57c84cb46f6757b303a64ec768ba04a6079e81"
_SEED_AUDIT_PATH = _ROOT / "docs/research/G4_FIXED_TDL_DOPPLER_SPECTRUM_SEED_AUDIT_2026-10-27.json"
_SEED_AUDIT_SHA256 = "b95c11501bfc0d701f4143a2194a3728d486d77ead5e49e2d1d858b2f2c5459f"
_MODELS = ("A10", "A30", "B100", "C60", "C300", "D10", "D30")
_SEED_START = 1_760_000
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


def test_fixed_jakes_spectrum_protocol_binds_annex_b_and_audited_seeds() -> None:
    reference = _load_json(_REFERENCE_PATH, _REFERENCE_SHA256)
    assert reference["schema"] == "loslab.g4_fixed_tdl_tap_profile_reference.v1"
    assert reference["source_id"] == "ts-138101-4-v19.2.2"
    assert reference["source_sha256"] == _SOURCE_SHA256
    counts = {model: len(fixed_tdl_rayleigh_tap_indices(model)) for model in _MODELS}
    assert counts == {"A10": 16, "A30": 12, "B100": 12, "C60": 12, "C300": 12, "D10": 9, "D30": 9}

    audit = _load_json(_SEED_AUDIT_PATH, _SEED_AUDIT_SHA256)
    assert audit["status"] == "CLEAR_WITHIN_SCANNED_JSON_CORPUS"
    assert audit["reserved_seeds"] == [_SEED_START + index for index in range(len(_MODELS))]
    assert audit["collisions"] == []
    assert audit["malformed_seed_fields"] == []
    assert audit["incomplete_files"] == []
    assert _CLAIM_COUNT == 49
    assert NormalDist().inv_cdf(1.0 - _FAMILYWISE_ALPHA / (2.0 * _CLAIM_COUNT)) > 0.0


@pytest.mark.native_sionna
@pytest.mark.parametrize("model", _MODELS)
def test_fixed_tdl_diffuse_doppler_spectrum_matches_jakes(model: str) -> None:
    """Check a frozen spectrum-CDF family on independent channel draws."""
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
    tap_indices = fixed_tdl_rayleigh_tap_indices(model)
    assert path.shape[0] == _DRAW_COUNT
    assert path.shape[-1] == _TIME_COUNT
    assert path.shape[1] == len(tap_indices) + (1 if model in {"D10", "D30"} else 0)
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
