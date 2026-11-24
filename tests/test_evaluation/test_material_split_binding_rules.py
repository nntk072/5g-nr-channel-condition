"""Material calibration: split-manifest, incidence-group and prediction bindings."""

from __future__ import annotations

import copy
from typing import Any

import pytest
from evaluation.acceptance.material_calibration import evaluate_material_calibration
from tests.test_evaluation.test_material_calibration import (
    _assessable_payload,
    _refresh_split_manifest_hash,
)


def _mutated(mutate, *, refresh: bool = True) -> dict[str, Any]:
    payload = copy.deepcopy(_assessable_payload())
    mutate(payload)
    if refresh:
        _refresh_split_manifest_hash(payload)
    return payload


def _manifest(payload: dict[str, Any]) -> dict[str, Any]:
    return payload["full_split_manifest"]


def _group(payload: dict[str, Any], index: int = 0) -> dict[str, Any]:
    return payload["incidence_groups"][index]


def _prediction(payload: dict[str, Any], index: int = 0) -> dict[str, Any]:
    return payload["test_predictions"][index]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        pytest.param(
            lambda p: _manifest(p).update(expected_strata=[]),
            "externally verified split manifest",
            id="manifest-strata",
        ),
        pytest.param(
            lambda p: _manifest(p)["groups"].append("not-an-object"),
            "each full split manifest group must be an object",
            id="group-type",
        ),
        pytest.param(
            lambda p: _manifest(p)["groups"].append(copy.deepcopy(_manifest(p)["groups"][0])),
            "duplicate split incidence group",
            id="group-duplicate",
        ),
        pytest.param(
            lambda p: _manifest(p)["groups"][0].update(split="other"),
            "invalid split assignment",
            id="group-split",
        ),
        pytest.param(
            lambda p: (
                _manifest(p)["groups"][0]["sample_ids"].append(
                    _manifest(p)["groups"][1]["sample_ids"][0]
                )
                if len(_manifest(p)["groups"]) > 1
                else None
            ),
            "sample IDs must be globally unique",
            id="sample-shared",
        ),
        pytest.param(
            lambda p: p["incidence_groups"].append(copy.deepcopy(_group(p))),
            "duplicate incidence group",
            id="ig-duplicate",
        ),
        pytest.param(
            lambda p: _group(p).update(sample_ids=[]),
            "needs its complete nonempty sample_ids",
            id="ig-sample-empty",
        ),
        pytest.param(
            lambda p: _group(p)["sample_ids"].append(_group(p)["sample_ids"][0]),
            "repeats a sample_id",
            id="ig-sample-repeat",
        ),
        pytest.param(
            lambda p: _prediction(p).update(incidence_stratum_id="other-angle"),
            "exactly cover expected_strata|incidence-group stratum",
            id="pred-stratum",
        ),
        pytest.param(
            lambda p: _prediction(p).update(incidence_angle_deg=45.0),
            "incidence-group angle",
            id="pred-angle",
        ),
        pytest.param(
            lambda p: _prediction(p).update(split="validation"),
            "must be from the test split",
            id="pred-split",
        ),
        pytest.param(
            lambda p: _prediction(p).update(material_id="other-material"),
            "exactly cover expected_strata|hashed material",
            id="pred-material",
        ),
        pytest.param(
            lambda p: _prediction(p).update(angular_array_order="phi_rows"),
            "angular_array_order must declare",
            id="pred-array-order",
        ),
        pytest.param(
            lambda p: _prediction(p).update(measured_angular_power_per_bin_w=[1.0]),
            "2-dimensional finite real array",
            id="pred-power-shape",
        ),
        pytest.param(
            lambda p: _prediction(p).update(theta_edges_deg="x"), "must be arrays", id="pred-theta"
        ),
    ],
)
def test_split_group_and_prediction_bindings_are_enforced(mutate, message) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_material_calibration(_mutated(mutate))
