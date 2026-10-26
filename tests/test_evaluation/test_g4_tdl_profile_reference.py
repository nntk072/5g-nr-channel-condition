from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jsonschema
import pytest
from evaluation.g4_tdl_profile_reference import scalable_tdl_pdp_reference

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_PATH = ROOT / "docs/research/G4_SCALABLE_TDL_POWER_DELAY_REFERENCE_2026-10-26.json"
REFERENCE_SHA256 = "38b897029bd4cc6de49eece5096e69c75291eaf6dac45bebe04767ed24a4f0bd"
SOURCE_SHA256 = "b71460fb4978a59fbffc864652016cf161c9e17cce19f0d0e9b46bea630b1cc4"


def _reference() -> dict[str, object]:
    payload = REFERENCE_PATH.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == REFERENCE_SHA256
    result = json.loads(payload)
    assert isinstance(result, dict)
    return result


def test_scalable_tdl_power_delay_reference_is_complete_and_source_bound() -> None:
    reference = _reference()
    manifest = json.loads(
        (ROOT / "docs/research/G4_STANDARD_REFERENCE_MANIFEST_2026-10-26.json").read_text(
            encoding="utf-8"
        )
    )
    source = next(
        row for row in manifest["sources"] if row["source_id"] == "tr-138901-v19.2.0"
    )
    assert source["sha256"] == reference["source_sha256"] == SOURCE_SHA256
    assert [row["profile_model"] for row in reference["profiles"]] == list("ABCDE")
    expected_counts = {"A": 23, "B": 23, "C": 24, "D": 14, "E": 15}
    for profile in reference["profiles"]:
        rows = profile["components"]
        assert len(rows) == expected_counts[profile["profile_model"]]
        assert [row["component_index"] for row in rows] == list(range(1, len(rows) + 1))
        assert all(row["fading_distribution"] == "Rayleigh" for row in rows[1:])
    schema = json.loads(
        (ROOT / "datasets/schema/g4-scalable-tdl-power-delay-reference.v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.Draft202012Validator(schema).validate(reference)


@pytest.mark.parametrize("model", tuple("ABCDE"))
def test_scalable_tdl_reference_returns_normalized_power_and_scaled_delay(model: str) -> None:
    powers, delays = scalable_tdl_pdp_reference(model, delay_spread_s=100e-9)

    assert len(powers) == len(delays) == {"A": 23, "B": 23, "C": 24, "D": 13, "E": 14}[model]
    assert sum(powers) == pytest.approx(1.0, abs=1e-14)
    assert all(power > 0.0 for power in powers)
    assert all(delay >= 0.0 for delay in delays)
    assert delays[0] == 0.0
    source_component_index = 2 if model in {"D", "E"} else 1
    assert delays[1] == pytest.approx(
        _reference()["profiles"][ord(model) - ord("A")]["components"][source_component_index][
            "normalized_delay"
        ]
        * 100e-9
    )
    if model in {"D", "E"}:
        components = _reference()["profiles"][ord(model) - ord("A")]["components"]
        linear_powers = [10.0 ** (row["power_db"] / 10.0) for row in components]
        assert powers[0] == pytest.approx(sum(linear_powers[:2]) / sum(linear_powers))


@pytest.mark.parametrize(
    ("model", "delay_spread_s", "message"),
    (("F", 100e-9, "model must be A, B, C, D, or E"), ("A", True, "finite and positive")),
)
def test_scalable_tdl_reference_rejects_unsupported_inputs(
    model: str, delay_spread_s: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        scalable_tdl_pdp_reference(model, delay_spread_s=delay_spread_s)
