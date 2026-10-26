"""Torch-independent checks for acceptance-bound G5 curve seed schemas."""

import json
from pathlib import Path

import pytest


@pytest.mark.parametrize("version", (7, 8))
def test_acceptance_curve_schemas_require_exact_inference_seed_roster(version: int) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "datasets"
        / "schema"
        / f"g5-ordered-impairments-snr-curve.v{version}.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    inference_schema = schema["properties"]["curve_inference"]
    assert "channel_seeds" in inference_schema["required"]
    assert inference_schema["properties"]["channel_seeds"] == {
        "type": "array",
        "minItems": 1,
        "uniqueItems": True,
        "items": {"type": "integer", "minimum": 0},
    }
