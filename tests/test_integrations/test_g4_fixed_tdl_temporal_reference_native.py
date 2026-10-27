"""ETSI-referenced temporal covariance checks for fixed TDL Rayleigh taps."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_REFERENCE_PATH = _ROOT / "docs/research/G4_FIXED_TDL_TAP_PROFILE_REFERENCE_2026-10-26.json"
_REFERENCE_SHA256 = "fe22560a71f4e5683ea6780f73e84466155b26f5f97cc4b63e3250c102fafda3"
_SOURCE_SHA256 = "b3c6a73c7a40219e7270c08c7a57c84cb46f6757b303a64ec768ba04a6079e81"
_SEED_AUDIT_PATH = _ROOT / "docs/research/G4_FIXED_TDL_TEMPORAL_SEED_AUDIT_2026-10-27.json"
_SEED_AUDIT_SHA256 = "ffffcd0b526c14bd62f38a735b678c5b91311fb6918e0ac641e83fc524b32d43"
_MODELS = ("A10", "A30", "B100", "C60", "C300", "D10", "D30")
_SEED_START = 1_750_000
_SAMPLE_COUNT = 32_768
_CLAIM_COUNT = 492  # 82 ETSI Rayleigh taps x 3 lags x real/imaginary coordinates.
_FAMILYWISE_ALPHA = 0.01
_MAXIMUM_NORMALIZED_ERROR = 0.05
_NORMAL_DIST = NormalDist()


def _read_pinned_json(path: Path, expected_digest: str, *, label: str) -> dict[str, Any]:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_digest:
        raise ValueError(f"{label} does not match its frozen SHA-256")
    result = json.loads(payload)
    if not isinstance(result, dict):
        raise ValueError(f"{label} root must be an object")
    return result


def _profile_rayleigh_taps(profile: dict[str, Any]) -> tuple[int, ...]:
    components = profile.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError("profile must contain ETSI component rows")
    by_tap: dict[int, list[dict[str, Any]]] = {}
    for component in components:
        tap_index = component.get("tap_index")
        if isinstance(tap_index, bool) or not isinstance(tap_index, int):
            raise ValueError("ETSI tap indices must be integers")
        by_tap.setdefault(tap_index, []).append(component)
    expected = int(profile["tap_count"])
    if sorted(by_tap) != list(range(1, expected + 1)):
        raise ValueError("ETSI fixed TDL tap roster must be contiguous and one-based")
    return tuple(
        tap_index - 1
        for tap_index, rows in sorted(by_tap.items())
        if all(row.get("fading_distribution") == "Rayleigh" for row in rows)
    )


def _load_protocol_inputs() -> tuple[dict[str, Any], list[int]]:
    reference = _read_pinned_json(
        _REFERENCE_PATH, _REFERENCE_SHA256, label="fixed TDL ETSI component reference"
    )
    if (
        reference.get("schema") != "loslab.g4_fixed_tdl_tap_profile_reference.v1"
        or reference.get("source_id") != "ts-138101-4-v19.2.2"
        or reference.get("source_sha256") != _SOURCE_SHA256
        or reference.get("status") != "PER_TAP_REFERENCE_ONLY_NOT_CONFORMANCE_QUALIFICATION"
    ):
        raise ValueError("fixed TDL reference has an invalid normative identity")
    profiles = reference.get("profiles")
    if not isinstance(profiles, list) or {row.get("profile_model") for row in profiles} != set(
        _MODELS
    ):
        raise ValueError("reference must cover exactly the seven fixed TDL profiles")
    profile_map = {row["profile_model"]: row for row in profiles}
    expected_counts = {"A10": 16, "A30": 12, "B100": 12, "C60": 12, "C300": 12, "D10": 9, "D30": 9}
    if {model: len(_profile_rayleigh_taps(profile_map[model])) for model in _MODELS} != (
        expected_counts
    ):
        raise ValueError("ETSI Rayleigh tap roster differs from the frozen temporal family")

    seed_audit = _read_pinned_json(
        _SEED_AUDIT_PATH, _SEED_AUDIT_SHA256, label="fixed TDL temporal seed audit"
    )
    expected_seeds = [_SEED_START + index for index in range(len(_MODELS))]
    if (
        seed_audit.get("status") != "CLEAR_WITHIN_SCANNED_JSON_CORPUS"
        or seed_audit.get("reserved_seeds") != expected_seeds
        or seed_audit.get("incomplete_files")
        or seed_audit.get("collisions")
        or seed_audit.get("malformed_seed_fields")
    ):
        raise ValueError("fixed TDL temporal seed roster is not clear in its frozen audit")
    return profile_map, expected_seeds


def test_fixed_tdl_temporal_family_is_bound_to_etsi_taps_and_fresh_seeds() -> None:
    profiles, seeds = _load_protocol_inputs()
    assert seeds == list(range(1_750_000, 1_750_007))
    assert sum(len(_profile_rayleigh_taps(profiles[model])) for model in _MODELS) == 82
    assert _CLAIM_COUNT == 82 * 3 * 2
    assert _NORMAL_DIST.inv_cdf(1.0 - _FAMILYWISE_ALPHA / (2.0 * _CLAIM_COUNT)) > 0.0


@pytest.mark.native_sionna
@pytest.mark.parametrize("model", _MODELS)
def test_fixed_tdl_rayleigh_tap_temporal_covariance_matches_jakes(model: str) -> None:
    """Bound three Jakes covariance points on each independently labeled tap."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("sionna.phy")
    from integrations.sionna.tdl import generate_tdl_frequency_response

    profiles, seeds = _load_protocol_inputs()
    tap_indices = _profile_rayleigh_taps(profiles[model])
    carrier_hz = 3.5e9
    speed_mps = 30.0
    maximum_doppler_hz = speed_mps * carrier_hz / 299_792_458.0
    base_lag_s = 1.0 / (8.0 * maximum_doppler_hz)
    channel = generate_tdl_frequency_response(
        model=model,
        seed=seeds[_MODELS.index(model)],
        carrier_frequency_hz=carrier_hz,
        subcarrier_spacing_hz=30_000.0,
        num_subcarriers=8,
        batch_size=_SAMPLE_COUNT,
        num_time_steps=5,
        sampling_frequency_hz=1.0 / base_lag_s,
        min_speed_mps=speed_mps,
        max_speed_mps=speed_mps,
        num_sinusoids=20,
        spec_version="19.2",
        device="cpu",
    )
    assert channel.metadata["batch_axis_semantics"] == "independent_stochastic_channel_draws"
    assert channel.metadata["maximum_doppler_hz"] == pytest.approx(maximum_doppler_hz)
    assert channel.metadata["num_sinusoids"] == 20
    assert channel.h_path.shape[0] == _SAMPLE_COUNT
    assert channel.h_path.shape[-2] == int(profiles[model]["tap_count"])
    assert channel.h_path.shape[-1] == 5

    critical_z = _NORMAL_DIST.inv_cdf(1.0 - _FAMILYWISE_ALPHA / (2.0 * _CLAIM_COUNT))
    for tap_index in tap_indices:
        samples = channel.h_path[:, 0, 0, 0, 0, tap_index, :].to(torch.complex128)
        initial_power = samples[:, 0].abs().square()
        mean_power = initial_power.mean()
        for lag_index in (1, 2, 4):
            lag_s = lag_index * base_lag_s
            expected = torch.special.bessel_j0(
                torch.tensor(2.0 * math.pi * maximum_doppler_hz * lag_s, dtype=torch.float64)
            )
            residual = samples[:, 0].conj() * samples[:, lag_index] - expected * initial_power
            normalized_error = residual.mean() / mean_power
            half_width_real = (
                critical_z
                * residual.real.std(unbiased=True)
                / math.sqrt(_SAMPLE_COUNT)
                / mean_power
            )
            half_width_imag = (
                critical_z
                * residual.imag.std(unbiased=True)
                / math.sqrt(_SAMPLE_COUNT)
                / mean_power
            )
            assert abs(normalized_error.real) + half_width_real <= _MAXIMUM_NORMALIZED_ERROR
            assert abs(normalized_error.imag) + half_width_imag <= _MAXIMUM_NORMALIZED_ERROR
