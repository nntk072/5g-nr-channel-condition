"""ETSI-referenced proper-complex checks for scalable TDL Rayleigh taps."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

import pytest
from integrations.sionna.tdl import generate_tdl_frequency_response

_ROOT = Path(__file__).resolve().parents[2]
_REFERENCE_PATH = _ROOT / "docs/research/G4_SCALABLE_TDL_POWER_DELAY_REFERENCE_2026-10-26.json"
_REFERENCE_SHA256 = "38b897029bd4cc6de49eece5096e69c75291eaf6dac45bebe04767ed24a4f0bd"
_SOURCE_SHA256 = "b71460fb4978a59fbffc864652016cf161c9e17cce19f0d0e9b46bea630b1cc4"
_SEED_AUDIT_PATH = _ROOT / "docs/research/G4_SCALABLE_TDL_PROPERNESS_SEED_AUDIT_2026-10-27.json"
_SEED_AUDIT_SHA256 = "e2cb2de0141253d9e295cea201dd7502488931659f27ad44084922338db9142f"
_MODELS = tuple("ABCDE")
_EXPECTED_RAYLEIGH_TAPS = {"A": 23, "B": 23, "C": 24, "D": 12, "E": 13}
_SEED_START = 1_740_000
_SAMPLE_COUNT = 32_768
_FAMILYWISE_ALPHA = 0.01
_MAXIMUM_HALF_WIDTH = 0.05
_NORMAL_DIST = NormalDist()


def _load_reference() -> dict[str, Any]:
    payload = _REFERENCE_PATH.read_bytes()
    if hashlib.sha256(payload).hexdigest() != _REFERENCE_SHA256:
        raise ValueError("scalable TDL profile reference does not match its frozen digest")
    reference = json.loads(payload)
    if (
        reference.get("schema") != "loslab.g4_scalable_tdl_power_delay_reference.v1"
        or reference.get("source_id") != "tr-138901-v19.2.0"
        or reference.get("source_sha256") != _SOURCE_SHA256
        or reference.get("status") != "PROFILE_TABLE_REFERENCE_ONLY_NOT_CONFORMANCE_QUALIFICATION"
    ):
        raise ValueError("scalable TDL reference has an invalid normative identity")
    profiles = reference.get("profiles")
    if not isinstance(profiles, list) or {row.get("profile_model") for row in profiles} != set(
        _MODELS
    ):
        raise ValueError("scalable TDL reference does not cover the frozen profile roster")
    return reference


def _rayleigh_tap_indices(profile: dict[str, Any]) -> tuple[int, ...]:
    """Map ETSI component rows to adapter taps, excluding a merged LOS tap."""
    components = profile["components"]
    if [row["component_index"] for row in components] != list(range(1, len(components) + 1)):
        raise ValueError("scalable TDL component reference has a noncontiguous roster")
    has_los = any(row["fading_distribution"] == "LOS" for row in components)
    result = []
    for component in components:
        if component["fading_distribution"] != "Rayleigh":
            continue
        tap_index = int(component["component_index"]) - (2 if has_los else 1)
        if tap_index >= 0 and not (has_los and tap_index == 0):
            result.append(tap_index)
    return tuple(result)


def _rayleigh_tap_count(reference: dict[str, Any]) -> int:
    return sum(
        len(_rayleigh_tap_indices(profile))
        for profile in reference["profiles"]
        if profile["profile_model"] in _MODELS
    )


def test_scalable_tdl_circularity_family_is_bound_to_the_expected_etsi_taps() -> None:
    reference = _load_reference()
    seed_audit_bytes = _SEED_AUDIT_PATH.read_bytes()
    if hashlib.sha256(seed_audit_bytes).hexdigest() != _SEED_AUDIT_SHA256:
        raise ValueError("scalable TDL seed audit does not match its frozen digest")
    seed_audit = json.loads(seed_audit_bytes)
    expected_seeds = [_SEED_START + index for index in range(len(_MODELS))]
    if (
        seed_audit.get("status") != "CLEAR_WITHIN_SCANNED_JSON_CORPUS"
        or seed_audit.get("reserved_seeds") != expected_seeds
        or seed_audit.get("incomplete_files")
        or seed_audit.get("collisions")
        or seed_audit.get("malformed_seed_fields")
    ):
        raise ValueError("scalable TDL native seed roster is not clear in its frozen audit")
    profiles = {row["profile_model"]: row for row in reference["profiles"]}
    assert {model: len(_rayleigh_tap_indices(profiles[model])) for model in _MODELS} == (
        _EXPECTED_RAYLEIGH_TAPS
    )
    assert _rayleigh_tap_count(reference) == 95
    assert _rayleigh_tap_indices(profiles["D"])[0] == 1
    assert _rayleigh_tap_indices(profiles["E"])[0] == 1


@pytest.mark.native_sionna
@pytest.mark.parametrize("model", _MODELS)
def test_scalable_tdl_rayleigh_taps_match_proper_complex_reference(model: str) -> None:
    """Check zero mean and properness over independent channel draws per profile."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("sionna.phy")
    reference = _load_reference()
    profile = next(row for row in reference["profiles"] if row["profile_model"] == model)
    rayleigh_taps = _rayleigh_tap_indices(profile)
    claim_count = 4 * _rayleigh_tap_count(reference)
    simultaneous_z = _NORMAL_DIST.inv_cdf(1.0 - _FAMILYWISE_ALPHA / (2.0 * claim_count))
    channel = generate_tdl_frequency_response(
        model=model,
        seed=_SEED_START + _MODELS.index(model),
        carrier_frequency_hz=3.5e9,
        subcarrier_spacing_hz=30e3,
        num_subcarriers=24,
        channel_bandwidth_hz=100e6,
        batch_size=_SAMPLE_COUNT,
        delay_spread_s=100e-9,
        spec_version="19.2",
        num_time_steps=1,
        device="cpu",
    )
    profile_has_los = any(row["fading_distribution"] == "LOS" for row in profile["components"])
    assert channel.metadata["sionna_version"] == "2.2.0"
    assert channel.metadata["profile_parameter_table_source"] == f"TR 38.901:19.2:TDL-{model}.json"
    assert channel.metadata["profile_los_component"] is profile_has_los
    assert channel.metadata["batch_axis_semantics"] == "independent_stochastic_channel_draws"
    assert tuple(channel.h_path.shape) == (
        _SAMPLE_COUNT,
        1,
        1,
        1,
        1,
        len(profile["components"]) - int(profile_has_los),
        1,
    )

    for tap_index in rayleigh_taps:
        samples = channel.h_path[:, 0, 0, 0, 0, tap_index, 0].to(torch.complex128)
        power = samples.abs().square().mean()
        mean_scale = torch.sqrt(power)
        normalized_mean = samples.mean() / mean_scale
        mean_half_widths = (
            simultaneous_z
            * torch.stack((samples.real.std(unbiased=True), samples.imag.std(unbiased=True)))
            / (mean_scale * math.sqrt(_SAMPLE_COUNT))
        )
        assert bool(
            (
                torch.stack((normalized_mean.real, normalized_mean.imag)).abs() <= mean_half_widths
            ).all()
        ), (model, tap_index, normalized_mean.tolist(), mean_half_widths.tolist())

        pseudo_covariance = samples.square().mean() / power
        influence = samples.square() - pseudo_covariance * power
        pseudo_half_widths = (
            simultaneous_z
            * torch.stack((influence.real.std(unbiased=True), influence.imag.std(unbiased=True)))
            / (power * math.sqrt(_SAMPLE_COUNT))
        )
        assert bool(
            (
                torch.stack((pseudo_covariance.real, pseudo_covariance.imag)).abs()
                <= pseudo_half_widths
            ).all()
        ), (model, tap_index, pseudo_covariance.tolist(), pseudo_half_widths.tolist())
        assert float(torch.maximum(mean_half_widths.max(), pseudo_half_widths.max())) <= (
            _MAXIMUM_HALF_WIDTH
        )
