"""Joint evaluation inputs: canonical values, branch fields, receiver execution and version rosters."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest
from evaluation.acceptance import joint_evaluation as joint
from tests.test_evaluation.joint_evidence_artifact_support import (
    _payload_v2,
    _payload_v4,
)


def _evaluate(payload: dict[str, Any]) -> dict[str, Any]:
    return joint.evaluate_joint_evidence(payload)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        pytest.param(np.array([1.0, np.nan]), "non-finite values", id="array-nan"),
        pytest.param({1: "x"}, "object keys must be strings", id="key-type"),
        pytest.param({"x": complex(float("inf"), 0.0)}, "non-finite complex", id="complex-inf"),
        pytest.param({"x": float("nan")}, "non-finite numbers", id="float-nan"),
        pytest.param({"x": object()}, "unsupported joint evidence value", id="object"),
    ],
)
def test_canonical_tree_rejects_nonfinite_or_untyped_values(value, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        joint._canonical_tree(value)


def test_canonical_tree_accepts_numpy_scalars_and_arrays() -> None:
    tree = joint._canonical_tree({"a": np.array([1.0, 2.0]), "b": np.float64(3.0)})
    assert "$array" in tree["a"]
    assert tree["b"] == 3.0


def test_version_specific_fields_are_gated() -> None:
    payload = copy.deepcopy(_payload_v4())
    payload["solver_budget_provenance"] = {"max_wall_seconds": 1.0}
    with pytest.raises(
        ValueError, match="joint evaluator v4 inputs must contain exactly the v4 fields"
    ):
        _evaluate(payload)


def test_receiver_execution_must_declare_its_four_objects() -> None:
    payload = copy.deepcopy(_payload_v2())
    payload["receiver_execution"].pop("candidate")
    with pytest.raises(ValueError, match="exactly its four declared objects"):
        _evaluate(payload)


def test_receiver_execution_waveform_and_budget_fields_are_checked() -> None:
    payload = copy.deepcopy(_payload_v2())
    payload["receiver_execution"]["waveform"] = {"waveform_id": "", "configuration": {}}
    with pytest.raises(ValueError, match="waveform"):
        _evaluate(payload)
