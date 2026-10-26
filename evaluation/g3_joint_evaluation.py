"""Deterministically run and bind the G3 evaluator branches under family v4.

The bundle is a numerical-evidence orchestration artifact. Its successful
recomputation never establishes measurement provenance or scientific
qualification; every returned bundle remains explicitly not qualified.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from evaluation.g3_complex_csi import evaluate_complex_csi
from evaluation.g3_joint_acceptance import (
    FAMILY_SCHEMA_V4,
    validate_g3_joint_acceptance_family,
)
from evaluation.g3_material_calibration import evaluate_g3_material_calibration
from evaluation.g3_path_equivalence_assurance import evaluate_path_equivalence_roster
from evaluation.g3_propagation_metrics import evaluate_propagation_metrics
from evaluation.g3_receiver_impact import evaluate_receiver_impact

_INPUT_SCHEMA = "loslab.g3_joint_evaluator_inputs.v1"
_INPUT_SCHEMA_V2 = "loslab.g3_joint_evaluator_inputs.v2"
_INPUT_SCHEMA_V3 = "loslab.g3_joint_evaluator_inputs.v3"
_INPUT_SCHEMA_V4 = "loslab.g3_joint_evaluator_inputs.v4"
_INPUT_SCHEMA_V5 = "loslab.g3_joint_evaluator_inputs.v5"
_INPUT_SCHEMA_V6 = "loslab.g3_joint_evaluator_inputs.v6"
_BUNDLE_SCHEMA = "loslab.g3_joint_evaluator_bundle.v2"
_BRANCHES = (
    "material_calibration",
    "complex_csi",
    "propagation_metrics",
    "receiver_impact",
    "path_equivalence",
)
_INPUT_FIELDS = {
    "schema",
    "joint_acceptance_family",
    *_BRANCHES,
}
_INPUT_FIELDS_V2 = _INPUT_FIELDS | {"receiver_execution"}
_INPUT_FIELDS_V3 = _INPUT_FIELDS_V2
_INPUT_FIELDS_V4 = _INPUT_FIELDS_V3
_INPUT_FIELDS_V5 = _INPUT_FIELDS_V4
_INPUT_FIELDS_V6 = _INPUT_FIELDS_V5
_BRANCH_REQUIRED = {
    "material_calibration": {"payload"},
    "complex_csi": {"records", "expected_strata"},
    "propagation_metrics": {
        "records",
        "expected_strata",
        "effective_bandwidth_hz",
    },
    "receiver_impact": {
        "records",
        "expected_strata",
        "candidate_config_id",
        "challenger_config_id",
        "snr_grid_db",
        "comparison_snr_db",
    },
    "path_equivalence": {"records", "expected_scenarios"},
}
_BRANCH_OPTIONAL = {
    "material_calibration": set(),
    "complex_csi": {"bootstrap_seed", "bootstrap_draws", "confidence_level"},
    "propagation_metrics": {"bootstrap_seed", "bootstrap_draws"},
    "receiver_impact": {"bootstrap_seed", "bootstrap_draws"},
    "path_equivalence": set(),
}


def _canonical_tree(value: Any) -> Any:
    """Encode JSON values and numeric arrays into an unambiguous hash tree."""

    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if array.dtype.hasobject or array.dtype.kind not in "biufc":
            raise TypeError("G3 joint evidence arrays must use non-object numeric dtypes")
        if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
            raise ValueError("G3 joint evidence arrays cannot contain non-finite values")
        contiguous = np.asarray(array, order="C")
        return {
            "$array": {
                "dtype": contiguous.dtype.str,
                "shape": list(contiguous.shape),
                "nbytes": int(contiguous.nbytes),
                "sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
            }
        }
    if isinstance(value, np.generic):
        return _canonical_tree(value.item())
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("G3 joint evidence object keys must be strings")
        return {key: _canonical_tree(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonical_tree(item) for item in value]
    if isinstance(value, complex):
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ValueError("G3 joint evidence cannot contain non-finite complex values")
        return {"$complex": [float(value.real), float(value.imag)]}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("G3 joint evidence cannot contain non-finite numbers")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"unsupported G3 joint evidence value: {type(value).__name__}")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _canonical_tree(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _branch_input(payload: dict[str, Any], branch: str) -> dict[str, Any]:
    value = payload.get(branch)
    if not isinstance(value, dict):
        raise ValueError(f"{branch} must be an object")
    keys = set(value)
    required = set(_BRANCH_REQUIRED[branch])
    if branch == "path_equivalence" and payload.get("schema") in {
        _INPUT_SCHEMA_V5,
        _INPUT_SCHEMA_V6,
    }:
        required.add("comparison_configuration")
    if branch == "path_equivalence" and payload.get("schema") == _INPUT_SCHEMA_V6:
        required.add("solver_budget_provenance")
    elif branch == "path_equivalence" and "solver_budget_provenance" in keys:
        raise ValueError("solver_budget_provenance requires joint evaluator input v6")
    if (
        branch == "path_equivalence"
        and payload.get("schema") not in {_INPUT_SCHEMA_V5, _INPUT_SCHEMA_V6}
        and "comparison_configuration" in keys
    ):
        raise ValueError("comparison_configuration requires joint evaluator input v5 or v6")
    allowed = required | _BRANCH_OPTIONAL[branch] | {"joint_acceptance_family"}
    missing = required - keys
    extra = keys - allowed
    if missing:
        raise ValueError(f"{branch} is missing required fields: {sorted(missing)}")
    if extra:
        raise ValueError(f"{branch} contains unsupported fields: {sorted(extra)}")
    declared_family = value.get("joint_acceptance_family")
    if declared_family is not None and _canonical_bytes(declared_family) != _canonical_bytes(
        payload["joint_acceptance_family"]
    ):
        raise ValueError(f"{branch} joint family differs from the bundle family")
    result = {key: value[key] for key in sorted(keys - {"joint_acceptance_family"})}
    roster_key = "expected_scenarios" if branch == "path_equivalence" else "expected_strata"
    if branch != "material_calibration":
        strata = result[roster_key]
        if not isinstance(strata, (list, tuple)):
            raise ValueError(f"{branch}.{roster_key} must be an array")
        if any(not isinstance(item, (list, tuple)) for item in strata):
            raise ValueError(f"{branch}.{roster_key} entries must be arrays")
        result[roster_key] = tuple(tuple(item) for item in strata)
    if branch != "material_calibration":
        result["joint_acceptance_family"] = payload["joint_acceptance_family"]
    return result


def _validate_receiver_execution_context(
    value: Any,
    receiver_input: dict[str, Any],
    *,
    require_row_configuration_binding: bool = False,
) -> None:
    fields = {"candidate", "challenger", "waveform", "link_budget_noise"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("receiver_execution must contain exactly its four declared objects")
    for role in ("candidate", "challenger"):
        config = value[role]
        if not isinstance(config, dict) or set(config) != {
            "config_id",
            "configuration",
            "configuration_sha256",
        }:
            raise ValueError(f"receiver_execution.{role} fields are invalid")
        if not isinstance(config["config_id"], str) or not config["config_id"].strip():
            raise ValueError(f"receiver_execution.{role}.config_id must be nonempty")
        if not isinstance(config["configuration"], dict):
            raise ValueError(f"receiver_execution.{role}.configuration must be an object")
        if config["configuration_sha256"] != _sha256(config["configuration"]):
            raise ValueError(f"receiver_execution.{role} configuration hash is inconsistent")
    if value["candidate"]["config_id"] != receiver_input["candidate_config_id"] or (
        value["challenger"]["config_id"] != receiver_input["challenger_config_id"]
    ):
        raise ValueError("receiver_execution configuration IDs differ from receiver-impact input")
    waveform = value["waveform"]
    if not isinstance(waveform, dict) or set(waveform) != {
        "waveform_id",
        "configuration",
        "configuration_sha256",
    }:
        raise ValueError("receiver_execution.waveform fields are invalid")
    if not isinstance(waveform["waveform_id"], str) or not waveform["waveform_id"].strip():
        raise ValueError("receiver_execution.waveform.waveform_id must be nonempty")
    if not isinstance(waveform["configuration"], dict) or (
        waveform["configuration_sha256"] != _sha256(waveform["configuration"])
    ):
        raise ValueError("receiver_execution.waveform configuration hash is inconsistent")
    link = value["link_budget_noise"]
    link_fields = {
        "tx_power_dbm",
        "tx_gain_dbi",
        "rx_gain_dbi",
        "noise_figure_db",
        "noise_temperature_k",
        "noise_bandwidth_hz",
    }
    if not isinstance(link, dict) or set(link) != link_fields:
        raise ValueError("receiver_execution.link_budget_noise fields are invalid")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in link.values()):
        raise ValueError("receiver_execution link-budget values must be finite numbers")
    if any(not math.isfinite(float(item)) for item in link.values()):
        raise ValueError("receiver_execution link-budget values must be finite numbers")
    if link["noise_temperature_k"] <= 0 or link["noise_bandwidth_hz"] <= 0:
        raise ValueError("receiver_execution noise temperature and bandwidth must be positive")
    execution_digest = _sha256(value)
    records = receiver_input.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("v2 receiver-impact records must be a nonempty array")
    config_digests = {
        value[role]["config_id"]: value[role]["configuration_sha256"]
        for role in ("candidate", "challenger")
    }
    for index, record in enumerate(records):
        if (
            not isinstance(record, dict)
            or record.get("receiver_execution_context_sha256") != execution_digest
        ):
            raise ValueError(
                "receiver-impact record "
                f"{index} is not bound to the declared receiver execution context"
            )
        if require_row_configuration_binding:
            configuration_id = record.get("configuration_id")
            expected_config_digest = (
                config_digests.get(configuration_id) if isinstance(configuration_id, str) else None
            )
            if (
                expected_config_digest is None
                or record.get("receiver_configuration_sha256") != expected_config_digest
            ):
                raise ValueError(
                    "receiver-impact record "
                    f"{index} is not bound to the configuration named by configuration_id"
                )


def _validate_shared_scene_condition_roster(
    family: dict[str, Any],
    branch_inputs: dict[str, dict[str, Any]],
    *,
    require_episode_binding: bool,
) -> None:
    """Require the four scene-based evaluators to cover the same family/condition cells."""

    def project_strata(branch: str, width: int) -> set[tuple[str, str]]:
        values = branch_inputs[branch]["expected_strata"]
        result: set[tuple[str, str]] = set()
        for item in values:
            if (
                len(item) != width
                or not isinstance(item[0], str)
                or not item[0].strip()
                or not isinstance(item[1], str)
                or item[1] not in {"LOS", "NLOS"}
            ):
                raise ValueError(f"{branch} expected-stratum roster is malformed")
            result.add((item[0], item[1]))
        return result

    expected_by_branch: dict[str, set[tuple[str, str]]] = {}
    for branch in ("complex_csi", "receiver_impact"):
        expected_by_branch[branch] = project_strata(branch, 2)
    expected_by_branch["propagation_metrics"] = project_strata("propagation_metrics", 3)

    scenarios = branch_inputs["path_equivalence"]["expected_scenarios"]
    path_roster: set[tuple[str, str, str]] = set()
    for item in scenarios:
        if (
            len(item) != 3
            or any(not isinstance(value, str) or not value.strip() for value in item)
            or item[1] not in {"LOS", "NLOS"}
        ):
            raise ValueError("path-equivalence expected-scenario roster is malformed")
        path_roster.add((item[0], item[1], item[2]))
    declared_path_roster = {tuple(item) for item in family["path_equivalence_expected_scenarios"]}
    if path_roster != declared_path_roster:
        raise ValueError("path-equivalence episode roster differs from the shared G3 family")

    def observed_episode_roster(branch: str) -> set[tuple[str, str, str]]:
        records = branch_inputs[branch].get("records")
        if not isinstance(records, (list, tuple)) or not records:
            raise ValueError(f"{branch} records must be a nonempty episode roster")
        result: set[tuple[str, str, str]] = set()
        for index, row in enumerate(records):
            if not isinstance(row, dict):
                raise ValueError(f"{branch} record {index} must be an object")
            family_id = row.get("family_id")
            condition = row.get("condition")
            episode_id = row.get("episode_id")
            if (
                not isinstance(family_id, str)
                or not family_id.strip()
                or not isinstance(condition, str)
                or condition not in {"LOS", "NLOS"}
                or not isinstance(episode_id, str)
                or not episode_id.strip()
            ):
                raise ValueError(
                    f"{branch} record {index} needs family_id, LOS/NLOS condition, "
                    "and episode_id for shared-roster binding"
                )
            result.add((family_id, condition, episode_id))
        return result

    if require_episode_binding:
        for branch in (
            "complex_csi",
            "propagation_metrics",
            "receiver_impact",
            "path_equivalence",
        ):
            observed_roster = observed_episode_roster(branch)
            if observed_roster != path_roster:
                raise ValueError(
                    f"{branch} observed episode roster differs from the shared G3 path roster: "
                    f"missing={sorted(path_roster - observed_roster)}, "
                    f"unexpected={sorted(observed_roster - path_roster)}"
                )

    expected_by_branch["path_equivalence"] = {
        (family_id, condition) for family_id, condition, _episode_id in path_roster
    }

    reference_branch = "path_equivalence"
    reference = expected_by_branch[reference_branch]
    for branch, observed in expected_by_branch.items():
        if observed != reference:
            raise ValueError(
                "G3 scene-family/LOS-NLOS roster must match across scene-based branches; "
                f"{branch} differs from {reference_branch}: "
                f"missing={sorted(reference - observed)}, "
                f"unexpected={sorted(observed - reference)}"
            )


def _criterion_branch_report(
    branch: str,
    outcomes: list[bool | str | None],
    *,
    outcome_kind: str,
) -> dict[str, Any]:
    """Summarize outcomes under the scorer's branch-specific result contract."""
    if outcome_kind not in {"boolean", "status", "propagation_status"}:
        raise ValueError(f"unsupported G3 criterion outcome kind: {outcome_kind!r}")
    passed = 0
    failed = 0
    for value in outcomes:
        if outcome_kind == "boolean":
            passed += value is True
            failed += value is False
        elif outcome_kind == "propagation_status":
            passed += value == "PASS" or value == "NOT_APPLICABLE"
            failed += value == "FAIL"
        else:
            passed += value == "PASS"
            failed += value == "FAIL"
    unresolved = len(outcomes) - passed - failed
    if failed:
        status = "FAIL"
    elif not outcomes or unresolved:
        status = "UNRESOLVED"
    else:
        status = "PASS"
    return {
        "status": status,
        "criterion_count": len(outcomes),
        "pass_count": passed,
        "fail_count": failed,
        "unresolved_count": unresolved,
        "basis": branch,
    }


def _numerical_criterion_rollup(outputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Aggregate frozen numerical criterion results while retaining scientific gate status."""
    material = outputs["material_calibration"].get("strata", [])
    csi = outputs["complex_csi"].get("strata", [])
    propagation = outputs["propagation_metrics"].get("strata", [])
    receiver = outputs["receiver_impact"].get("strata", [])
    path = outputs["path_equivalence"].get("scenarios", [])

    material_outcomes = (
        [row.get("pilot_margins_pass") if isinstance(row, dict) else None for row in material]
        if isinstance(material, list)
        else []
    )
    csi_outcomes = (
        [row.get("numeric_criteria_pass") if isinstance(row, dict) else None for row in csi]
        if isinstance(csi, list)
        else []
    )
    propagation_outcomes = (
        [
            metric.get("criteria_status")
            for row in propagation
            if isinstance(row, dict) and isinstance(row.get("metrics"), list)
            for metric in row["metrics"]
            if isinstance(metric, dict)
        ]
        if isinstance(propagation, list)
        else []
    )
    propagation_assessable = outputs["propagation_metrics"].get("any_metric_assessable") is True
    if not propagation_assessable:
        # NOT_APPLICABLE is neutral only when another frozen propagation
        # claim is actually assessable. The propagation evaluator exposes
        # this branch-level fact explicitly; without it, an all-NA family
        # must not be promoted to a numerical pass.
        propagation_outcomes = [
            None if outcome is None or outcome == "PASS" or outcome == "NOT_APPLICABLE" else outcome
            for outcome in propagation_outcomes
        ]
    receiver_outcomes = (
        [
            value
            for row in receiver
            if isinstance(row, dict)
            for value in (
                row.get("bler_criteria_pass"),
                row.get("snr_at_10_percent_criteria_pass"),
            )
        ]
        if isinstance(receiver, list)
        else []
    )
    path_outcomes = (
        [row.get("class_coverage_status") if isinstance(row, dict) else None for row in path]
        if isinstance(path, list)
        else []
    )

    branches = {
        "material_calibration": _criterion_branch_report(
            "pilot_margins_pass (numerical pilot criteria only)",
            material_outcomes,
            outcome_kind="boolean",
        ),
        "complex_csi": _criterion_branch_report(
            "numeric_criteria_pass", csi_outcomes, outcome_kind="boolean"
        ),
        "propagation_metrics": _criterion_branch_report(
            "per-metric criteria_status",
            propagation_outcomes,
            outcome_kind="propagation_status",
        ),
        "receiver_impact": _criterion_branch_report(
            "BLER and SNR-at-target criteria",
            receiver_outcomes,
            outcome_kind="boolean",
        ),
        "path_equivalence": _criterion_branch_report(
            "per-episode bidirectional class coverage",
            path_outcomes,
            outcome_kind="status",
        ),
    }
    statuses = [item["status"] for item in branches.values()]
    joint_status = (
        "FAIL_NUMERICAL_CRITERIA"
        if "FAIL" in statuses
        else "PASS_NUMERICAL_CRITERIA_ONLY"
        if all(status == "PASS" for status in statuses)
        else "UNRESOLVED_NUMERICAL_CRITERIA"
    )
    return {
        "schema": "loslab.g3_joint_numerical_criterion_rollup.v1",
        "status": joint_status,
        "qualification_status": "NOT_QUALIFIED",
        "branches": branches,
        "interpretation": (
            "This roll-up summarizes only the frozen numerical checks reported by each branch. "
            "A numerical pass does not verify measurement truth, independence, provenance, "
            "representativeness, reviewer authority, or scientific qualification."
        ),
    }


def _recompute(
    payload: dict[str, Any], *, coefficient_relative_tolerance: float | None = None
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("G3 joint evaluator inputs must be an object")
    schema = payload.get("schema")
    if schema == _INPUT_SCHEMA:
        if set(payload) != _INPUT_FIELDS:
            raise ValueError("G3 joint evaluator v1 inputs must contain exactly the v1 fields")
    elif schema == _INPUT_SCHEMA_V2:
        if set(payload) != _INPUT_FIELDS_V2:
            raise ValueError("G3 joint evaluator v2 inputs must contain exactly the v2 fields")
    elif schema == _INPUT_SCHEMA_V3:
        if set(payload) != _INPUT_FIELDS_V3:
            raise ValueError("G3 joint evaluator v3 inputs must contain exactly the v3 fields")
    elif schema == _INPUT_SCHEMA_V4:
        if set(payload) != _INPUT_FIELDS_V4:
            raise ValueError("G3 joint evaluator v4 inputs must contain exactly the v4 fields")
    elif schema == _INPUT_SCHEMA_V5:
        if set(payload) != _INPUT_FIELDS_V5:
            raise ValueError("G3 joint evaluator v5 inputs must contain exactly the v5 fields")
    elif schema == _INPUT_SCHEMA_V6:
        if set(payload) != _INPUT_FIELDS_V6:
            raise ValueError("G3 joint evaluator v6 inputs must contain exactly the v6 fields")
    else:
        raise ValueError(
            f"schema must be one of {_INPUT_SCHEMA!r}, {_INPUT_SCHEMA_V2!r}, "
            f"{_INPUT_SCHEMA_V3!r}, {_INPUT_SCHEMA_V4!r}, {_INPUT_SCHEMA_V5!r}, "
            f"or {_INPUT_SCHEMA_V6!r}"
        )
    family = validate_g3_joint_acceptance_family(payload["joint_acceptance_family"])
    if family["schema"] != FAMILY_SCHEMA_V4:
        raise ValueError("G3 joint evaluator bundle requires acceptance family v4")

    branch_inputs = {branch: _branch_input(payload, branch) for branch in _BRANCHES}
    material_input = branch_inputs["material_calibration"].get("payload")
    if not isinstance(material_input, dict):
        raise ValueError("material_calibration.payload must be an object")
    if _canonical_bytes(material_input.get("joint_acceptance_family")) != _canonical_bytes(family):
        raise ValueError("material calibration input must embed the exact joint family")

    csi_input = branch_inputs["complex_csi"]
    propagation_input = branch_inputs["propagation_metrics"]
    receiver_input = branch_inputs["receiver_impact"]
    _validate_shared_scene_condition_roster(
        family,
        branch_inputs,
        require_episode_binding=schema
        in {_INPUT_SCHEMA_V3, _INPUT_SCHEMA_V4, _INPUT_SCHEMA_V5, _INPUT_SCHEMA_V6},
    )
    if schema in {
        _INPUT_SCHEMA_V2,
        _INPUT_SCHEMA_V3,
        _INPUT_SCHEMA_V4,
        _INPUT_SCHEMA_V5,
        _INPUT_SCHEMA_V6,
    }:
        _validate_receiver_execution_context(
            payload["receiver_execution"],
            receiver_input,
            require_row_configuration_binding=schema
            in {_INPUT_SCHEMA_V4, _INPUT_SCHEMA_V5, _INPUT_SCHEMA_V6},
        )
    outputs = {
        "material_calibration": evaluate_g3_material_calibration(material_input),
        "complex_csi": evaluate_complex_csi(**csi_input),
        "propagation_metrics": evaluate_propagation_metrics(**propagation_input),
        "receiver_impact": evaluate_receiver_impact(**receiver_input),
        "path_equivalence": evaluate_path_equivalence_roster(
            **branch_inputs["path_equivalence"],
            coefficient_relative_tolerance=coefficient_relative_tolerance,
        ),
    }
    criterion_rollup = _numerical_criterion_rollup(outputs)
    return {
        "schema": _BUNDLE_SCHEMA,
        "status": "NUMERICAL_EVALUATION_ONLY_NOT_QUALIFIED",
        "qualification_status": "NOT_QUALIFIED",
        "joint_acceptance_family": {
            "family_id": family["family_id"],
            "family_sha256": family["family_sha256"],
            "schema": family["schema"],
            "claim_count": family["claim_count"],
        },
        "branch_inputs_sha256": {branch: _sha256(branch_inputs[branch]) for branch in _BRANCHES},
        "branch_outputs_sha256": {branch: _sha256(outputs[branch]) for branch in _BRANCHES},
        "evaluator_statuses": {
            branch: outputs[branch].get("status", "STATUS_NOT_REPORTED") for branch in _BRANCHES
        },
        "numerical_criterion_rollup": criterion_rollup,
        "evaluations": outputs,
        "interpretation": (
            "Deterministic scorer recomputation binds these numerical reports to the supplied "
            "inputs. It does not verify measurement truth, external provenance, physical-unit "
            "independence, preregistration, representativeness, reviewer authority, or legal "
            "rights; no report or aggregate status qualifies G3 or authorizes production RT."
        ),
    }


def evaluate_g3_joint_evidence(
    payload: dict[str, Any], *, coefficient_relative_tolerance: float | None = None
) -> dict[str, Any]:
    """Run all five G3 numerical evaluators under one hash-bound family v4."""

    return _recompute(payload, coefficient_relative_tolerance=coefficient_relative_tolerance)


def verify_g3_joint_evidence_bundle(
    payload: dict[str, Any],
    bundle: dict[str, Any],
    *,
    coefficient_relative_tolerance: float | None = None,
) -> dict[str, Any]:
    """Recompute a saved bundle and reject any altered evaluator output or hash."""

    if not isinstance(bundle, dict):
        raise TypeError("bundle must be an object")
    expected = _recompute(payload, coefficient_relative_tolerance=coefficient_relative_tolerance)
    if _sha256(bundle) != _sha256(expected):
        raise ValueError("G3 joint evidence bundle differs from deterministic recomputation")
    return {
        "schema": "loslab.g3_joint_evaluator_bundle_verification.v1",
        "status": "PASS_INPUT_SCOPED_RECOMPUTATION",
        "qualification_status": "NOT_QUALIFIED",
        "bundle_sha256": _sha256(bundle),
        "joint_acceptance_family_sha256": expected["joint_acceptance_family"]["family_sha256"],
        "branch_count": len(_BRANCHES),
    }
