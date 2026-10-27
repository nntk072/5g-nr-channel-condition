"""ETSI-referenced first-tap Rician marginal checks for fixed TDL-D profiles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sionna.phy")

from evaluation.g4_tdl_covariance_reference import fixed_tdl_first_tap_power_components  # noqa: E402
from evaluation.g4_tdl_rician_marginal import score_tdl_rician_marginal_family  # noqa: E402
from integrations.sionna.tdl import generate_tdl_frequency_response  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
_REFERENCE_PATH = _ROOT / "docs/research/G4_FIXED_TDL_TAP_PROFILE_REFERENCE_2026-10-26.json"
_REFERENCE_SHA256 = "fe22560a71f4e5683ea6780f73e84466155b26f5f97cc4b63e3250c102fafda3"
_SOURCE_SHA256 = "b3c6a73c7a40219e7270c08c7a57c84cb46f6757b303a64ec768ba04a6079e81"
_SIONNA_TABLE_SHA256 = {
    "D10": "4ff96632bf508813492f0e34168ab1b09981afc25760cfed0031013e662c6e73",
    "D30": "6da72f15e0da4fb972cf6a2c9c4ec2d2c1330f598f15e71c83cfbb324b59353c",
}
_SEED_AUDIT_PATH = _ROOT / "docs/research/G4_FIXED_TDL_RICIAN_MARGINAL_SEED_AUDIT_2026-10-27.json"
_SEED_AUDIT_SHA256 = "3640230f34c566c72854201cf133a8ab0be70b9ab3fcdc95eab2b3c26995890e"
_MODELS = ("D10", "D30")
_SEEDS = {"D10": 1_880_010, "D30": 1_880_011}
_DRAW_COUNT = 32_768
_ALPHA = 0.01
_MAXIMUM_SIMULTANEOUS_CDF_ERROR = 0.05


def _fixed_reference() -> dict[str, Any]:
    payload = _REFERENCE_PATH.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == _REFERENCE_SHA256
    document = json.loads(payload)
    assert document["source_sha256"] == _SOURCE_SHA256
    return document


def test_fixed_tdl_rician_seed_roster_is_bound_to_clear_saved_corpus_audit() -> None:
    payload = _SEED_AUDIT_PATH.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == _SEED_AUDIT_SHA256
    report = json.loads(payload)
    assert report["status"] == "CLEAR_WITHIN_SCANNED_JSON_CORPUS"
    assert report["reserved_seeds"] == [1_880_010, 1_880_011]
    assert report["collisions"] == []
    assert report["malformed_seed_fields"] == []
    assert report["incomplete_files"] == []


@pytest.mark.native_sionna
@pytest.mark.parametrize("model", _MODELS)
def test_fixed_tdl_first_tap_matches_etsi_rician_power_distribution(model: str) -> None:
    reference = _fixed_reference()
    profile = next(row for row in reference["profiles"] if row["profile_model"] == model)
    assert any(
        component["tap_index"] == 1 and component["fading_distribution"] in {"LOS", "LOS path"}
        for component in profile["components"]
    )
    diffuse_power, specular_power = fixed_tdl_first_tap_power_components(model)
    channel = generate_tdl_frequency_response(
        model=model,
        seed=_SEEDS[model],
        carrier_frequency_hz=3.5e9,
        subcarrier_spacing_hz=30e3,
        num_subcarriers=24,
        channel_bandwidth_hz=100e6 if model == "D30" else 200_000_001.0,
        batch_size=_DRAW_COUNT,
        num_time_steps=1,
        device="cpu",
    )
    expected_source = f"TS 38.101-4:19.2.2:TDL-{model}.json"
    assert channel.metadata["sionna_version"] == "2.2.0"
    assert channel.metadata["profile_parameter_table_source"] == expected_source
    assert channel.metadata["profile_parameter_table_sha256"] == _SIONNA_TABLE_SHA256[model]
    assert channel.metadata["profile_los_component"] is True
    assert channel.metadata["batch_axis_semantics"] == "independent_stochastic_channel_draws"
    first_tap_samples = channel.h_path[:, 0, 0, 0, 0, 0, 0].to(torch.complex128).cpu().numpy()
    result = score_tdl_rician_marginal_family(
        {model: first_tap_samples},
        {model: (diffuse_power, specular_power)},
        alpha=_ALPHA / len(_MODELS),
        maximum_simultaneous_cdf_error=_MAXIMUM_SIMULTANEOUS_CDF_ERROR,
    )[0]
    assert result.draw_count == _DRAW_COUNT
    assert result.passed, (model, result)
    assert result.simultaneous_upper_bound <= _MAXIMUM_SIMULTANEOUS_CDF_ERROR
