"""Fixed-TDL LOS K-factor cross-check against pinned component powers."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_REFERENCE_PATH = _ROOT / "docs/research/G4_FIXED_TDL_LOS_K_FACTOR_REFERENCE_2026-10-26.json"
_REFERENCE_SCHEMA_PATH = (
    _ROOT / "datasets/schema/g4-fixed-tdl-los-k-factor-reference.v1.schema.json"
)
_COMPONENT_PATH = _ROOT / "docs/research/G4_FIXED_TDL_TAP_PROFILE_REFERENCE_2026-10-26.json"
_MANIFEST_PATH = _ROOT / "docs/research/G4_STANDARD_REFERENCE_MANIFEST_2026-10-26.json"


def _reference() -> dict[str, Any]:
    return json.loads(_REFERENCE_PATH.read_text(encoding="utf-8"))


def test_fixed_tdl_los_k_factor_reference_is_source_bound_and_recomputable() -> None:
    reference = _reference()
    component_bytes = _COMPONENT_PATH.read_bytes()
    assert hashlib.sha256(component_bytes).hexdigest() == reference["component_reference_sha256"]
    component_reference = json.loads(component_bytes)
    manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    source = next(row for row in manifest["sources"] if row["source_id"] == reference["source_id"])
    assert source["sha256"] == reference["source_sha256"]
    assert component_reference["source_sha256"] == reference["source_sha256"]

    rows_by_profile = {
        row["profile_model"]: row["components"] for row in component_reference["profiles"]
    }
    assert [row["profile_model"] for row in reference["profiles"]] == ["D10", "D30"]
    for row in reference["profiles"]:
        components = rows_by_profile[row["profile_model"]]
        los_power = math.fsum(
            10 ** (component["power_db"] / 10)
            for component in components
            if component["fading_distribution"] in {"LOS", "LOS path"}
        )
        diffuse_power = math.fsum(
            10 ** (component["power_db"] / 10)
            for component in components
            if component["fading_distribution"] == "Rayleigh"
        )
        assert los_power > 0.0 and diffuse_power > 0.0
        derived_linear = los_power / diffuse_power
        derived_db = 10 * math.log10(derived_linear)
        assert math.isclose(row["k_factor_linear"], derived_linear, rel_tol=1e-14)
        assert math.isclose(row["k_factor_db"], derived_db, rel_tol=0.0, abs_tol=1e-12)

    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(_REFERENCE_SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(reference)


@pytest.mark.native_sionna
@pytest.mark.parametrize(
    ("model", "carrier_frequency_hz", "channel_bandwidth_hz"),
    [("D10", 28e9, 400e6), ("D30", 3.5e9, 100e6)],
)
def test_fixed_tdl_reported_los_k_factor_matches_component_reference(
    model: str, carrier_frequency_hz: float, channel_bandwidth_hz: float
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("sionna.phy.channel.tr38901")
    from integrations.sionna.tdl import generate_tdl_frequency_response

    reference = _reference()
    expected = next(row for row in reference["profiles"] if row["profile_model"] == model)
    result = generate_tdl_frequency_response(
        model=model,
        seed=38104,
        carrier_frequency_hz=carrier_frequency_hz,
        subcarrier_spacing_hz=30e3,
        num_subcarriers=24,
        channel_bandwidth_hz=channel_bandwidth_hz,
        spec_version="19.2",
        num_time_steps=1,
        device="cpu",
    )

    actual_linear = result.metadata["k_factor_linear"]
    assert isinstance(actual_linear, (float, int)) and actual_linear > 0.0
    actual_db = 10 * math.log10(actual_linear)
    assert abs(actual_db - expected["k_factor_db"]) <= reference["rounding_tolerance_db"]
    assert result.metadata["profile_los_component"] is True
    assert result.metadata["fixed_profile_spec_version"] == reference["version"]
