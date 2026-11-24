"""Paired link-curve admission: schema, evidence grade, configuration and matched-realization rules."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from artifacts.provenance import atomic_json
from evaluation.scoring.paired_link_curves import compare_curves, load_curve

SCHEMA = "loslab.tdl_pusch_time_bler_curve.v1"


def _curve(**changes: Any) -> dict[str, Any]:
    curve: dict[str, Any] = {
        "schema": SCHEMA,
        "evidence_grade": "SIMULATION_ONLY_UNVALIDATED",
        "configuration": {
            "speed_mps": 3.0,
            "matched_realization_checks": [
                {
                    "seed": 4,
                    "noise_seed": 6,
                    "transmit_bits_sha256": "bits",
                    "channel_truth_sha256": "truth",
                }
            ],
        },
        "points": [{"snr_db": 12.0, "frames": 1, "bits": 100, "frame_error_mask": [False]}],
    }
    curve.update(changes)
    return curve


def _write(path: Path, document: dict[str, Any]) -> Path:
    atomic_json(path, document)
    return path


def test_loader_requires_a_supported_schema_and_simulation_only_grade(tmp_path: Path) -> None:
    good = _write(tmp_path / "good.json", _curve())
    assert load_curve(good)["schema"] == SCHEMA
    bad_schema = _write(tmp_path / "schema.json", _curve(schema="other.v1"))
    with pytest.raises(ValueError, match="unsupported sample-domain curve schema"):
        load_curve(bad_schema)
    bad_grade = _write(tmp_path / "grade.json", _curve(evidence_grade="VALIDATED"))
    with pytest.raises(ValueError, match="unexpected evidence grade"):
        load_curve(bad_grade)


def _pair(tmp_path: Path, control: dict[str, Any], treatment: dict[str, Any]):
    return (
        _write(tmp_path / "control.json", control),
        _write(tmp_path / "treatment.json", treatment),
    )


@pytest.mark.parametrize(
    ("treatment_change", "message"),
    [
        pytest.param(
            {
                "configuration": {
                    "speed_mps": 3.0,
                    "matched_realization_checks": [
                        {
                            "seed": 4,
                            "noise_seed": 6,
                            "transmit_bits_sha256": "bits",
                            "channel_truth_sha256": "truth",
                        }
                    ],
                    "extra": 1,
                }
            },
            "configuration differs unexpectedly",
            id="extra-config",
        ),
        pytest.param(
            {"points": [{"snr_db": 99.0, "frames": 1, "bits": 100, "frame_error_mask": [False]}]},
            "SNR values or codeword counts differ|SNR grid|must differ",
            id="snr-grid",
        ),
        pytest.param({"points": []}, "SNR point counts differ|must differ", id="point-count"),
        pytest.param(
            {"configuration": {"speed_mps": 3.0, "matched_realization_checks": []}},
            "matched realization batch counts differ|must differ",
            id="batch-count",
        ),
    ],
)
def test_matched_pairs_must_differ_only_in_the_declared_field(
    tmp_path: Path, treatment_change, message
) -> None:
    treatment = _curve(**treatment_change)
    treatment["configuration"]["speed_mps"] = 30.0  # the declared varying field
    control_path, treatment_path = _pair(tmp_path, _curve(), treatment)
    with pytest.raises(ValueError, match=message):
        compare_curves(control_path, treatment_path)


def test_varying_field_must_be_declared_and_must_change(tmp_path: Path) -> None:
    control_path, treatment_path = _pair(tmp_path, _curve(), _curve())
    with pytest.raises(ValueError, match="varying_config_field must be"):
        compare_curves(control_path, treatment_path, varying_config_field="snr")
    with pytest.raises(ValueError, match="must differ at speed_mps"):
        compare_curves(control_path, treatment_path)


def test_matched_realizations_must_share_seeds_and_noise(tmp_path: Path) -> None:
    treatment = deepcopy(_curve())
    treatment["configuration"]["speed_mps"] = 30.0
    treatment["configuration"]["matched_realization_checks"][0]["seed"] = 5
    control_path, treatment_path = _pair(tmp_path, _curve(), treatment)
    with pytest.raises(ValueError, match="batch seeds differ"):
        compare_curves(control_path, treatment_path)

    treatment = deepcopy(_curve())
    treatment["configuration"]["speed_mps"] = 30.0
    treatment["configuration"]["matched_realization_checks"][0]["noise_seed"] = 7
    control_path, treatment_path = _pair(tmp_path, _curve(), treatment)
    with pytest.raises(ValueError, match="noise seeds differ"):
        compare_curves(control_path, treatment_path)
