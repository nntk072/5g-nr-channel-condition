"""Verify file-backed, replayable G3 numerical evidence runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
from pathlib import Path
from typing import Any

from evaluation.g3_acquisition_package import (
    MAX_FILE_BYTES,
    MAX_JSON_ARTIFACT_BYTES,
    MAX_PACKAGE_BYTES,
    MAX_SCHEMA_BYTES,
    _json_bytes,
    _read_bounded_regular_file,
    _read_inventory_file,
    _safe_relative_path,
)
from evaluation.g3_joint_evidence_artifact import (
    MAX_ARTIFACT_BYTES,
    MAX_INPUT_BYTES,
    _write_new_artifact,
    verify_g3_joint_evidence_bytes,
)
from evaluation.g3_qualification_contract import verify_g3_qualification_run_contract
from evaluation.g3_sample_size_planning import verify_g3_joint_sample_size_design
from evaluation.g3_signed_attestation import verify_openssh_signature_for_bytes

V1_SCHEMA_RELATIVE_PATH = "datasets/schema/g3-joint-evidence-run-manifest.v1.schema.json"
SCHEMA_RELATIVE_PATH = "datasets/schema/g3-joint-evidence-run-manifest.v2.schema.json"
SCHEMA_V3_RELATIVE_PATH = "datasets/schema/g3-joint-evidence-run-manifest.v3.schema.json"
SIGNER_POLICY_SCHEMA = "loslab.g3_run_manifest_signer_policy.v1"
SIGNER_POLICY_PURPOSE = "g3_joint_run_manifest_verification"
SIGNER_POLICY_RELATIVE_PATH = "datasets/schema/g3-run-manifest-signer-policy.v1.schema.json"
VERIFICATION_REPORT_SCHEMA_RELATIVE_PATH = (
    "datasets/schema/g3-joint-evidence-run-verification.v1.schema.json"
)
INPUT_SCHEMA_RELATIVE_PATH = "datasets/schema/g3-joint-evaluator-input.v1.schema.json"
INPUT_SCHEMA_V2_RELATIVE_PATH = "datasets/schema/g3-joint-evaluator-input.v2.schema.json"
INPUT_SCHEMA_V3_RELATIVE_PATH = "datasets/schema/g3-joint-evaluator-input.v3.schema.json"
INPUT_SCHEMA_V4_RELATIVE_PATH = "datasets/schema/g3-joint-evaluator-input.v4.schema.json"
INPUT_SCHEMA_V5_RELATIVE_PATH = "datasets/schema/g3-joint-evaluator-input.v5.schema.json"
INPUT_SCHEMA_V6_RELATIVE_PATH = "datasets/schema/g3-joint-evaluator-input.v6.schema.json"
PATH_DIAGNOSTIC_SCHEMA_V2_RELATIVE_PATH = (
    "datasets/schema/g3-path-equivalence-roster-diagnostic.v2.schema.json"
)
G3_RUN_MANIFEST_SIGNATURE_NAMESPACE = "loslab-g3-joint-run-manifest-v1@github.com"
G3_SIGNER_POLICY_SIGNATURE_NAMESPACE = "loslab-g3-run-manifest-policy-v1@github.com"
ARTIFACT_SCHEMA_V1_RELATIVE_PATH = "datasets/schema/g3-joint-evaluator-artifact.v1.schema.json"
ARTIFACT_SCHEMA_RELATIVE_PATH = "datasets/schema/g3-joint-evaluator-artifact.v2.schema.json"
QUALIFICATION_CONTRACT_SCHEMA_RELATIVE_PATH = (
    "datasets/schema/g3-qualification-run-contract.v3.schema.json"
)
QUALIFICATION_CONTRACT_SCHEMA_V2_RELATIVE_PATH = (
    "datasets/schema/g3-qualification-run-contract.v2.schema.json"
)
QUALIFICATION_CONTRACT_SCHEMA_V4_RELATIVE_PATH = (
    "datasets/schema/g3-qualification-run-contract.v4.schema.json"
)
_MAX_CAPTURE_BYTES = MAX_JSON_ARTIFACT_BYTES
_MAX_MANIFEST_TOTAL_BYTES = _MAX_CAPTURE_BYTES * 2
_MAX_SOURCE_BUNDLE_BYTES = 64 * 1024 * 1024
V1_EVALUATOR_SOURCE_PATHS = tuple(sorted((
    V1_SCHEMA_RELATIVE_PATH,
    ARTIFACT_SCHEMA_V1_RELATIVE_PATH,
    "datasets/schema/g3-joint-evaluator-input.v1.schema.json",
    "datasets/schema/g3-joint-evaluator-input.v2.schema.json",
    "datasets/schema/g3-qualification-run-contract.v2.schema.json",
    SIGNER_POLICY_RELATIVE_PATH,
    "evaluation/g3_acquisition_package.py",
    "evaluation/g3_complex_csi.py",
    "evaluation/g3_joint_acceptance.py",
    "evaluation/g3_joint_evaluation.py",
    "evaluation/g3_joint_evidence_artifact.py",
    "evaluation/g3_joint_evidence_run_manifest.py",
    "evaluation/g3_qualification_contract.py",
    "evaluation/g3_material_calibration.py",
    "evaluation/g3_numeric.py",
    "evaluation/g3_path_equivalence_assurance.py",
    "evaluation/g3_propagation_metrics.py",
    "evaluation/g3_receiver_impact.py",
    "evaluation/g3_signed_attestation.py",
)))
V2_EVALUATOR_SOURCE_PATHS = tuple(sorted((
    "pyproject.toml",
    "uv.lock",
    ARTIFACT_SCHEMA_V1_RELATIVE_PATH,
    "datasets/schema/g3-joint-evaluator-input.v1.schema.json",
    "datasets/schema/g3-joint-evaluator-input.v2.schema.json",
    "datasets/schema/g3-joint-sample-size-design-composition.v1.schema.json",
    INPUT_SCHEMA_V3_RELATIVE_PATH,
    INPUT_SCHEMA_V4_RELATIVE_PATH,
    INPUT_SCHEMA_V5_RELATIVE_PATH,
    INPUT_SCHEMA_V6_RELATIVE_PATH,
    PATH_DIAGNOSTIC_SCHEMA_V2_RELATIVE_PATH,
    SCHEMA_RELATIVE_PATH,
    "datasets/schema/g3-qualification-run-contract.v2.schema.json",
    QUALIFICATION_CONTRACT_SCHEMA_RELATIVE_PATH,
    QUALIFICATION_CONTRACT_SCHEMA_V4_RELATIVE_PATH,
    SIGNER_POLICY_RELATIVE_PATH,
    "evaluation/g3_acquisition_package.py",
    "evaluation/g3_complex_csi.py",
    "evaluation/g3_joint_acceptance.py",
    "evaluation/g3_joint_evaluation.py",
    "evaluation/g3_joint_evidence_artifact.py",
    "evaluation/g3_joint_evidence_run_manifest.py",
    "evaluation/g3_qualification_contract.py",
    "evaluation/g3_material_calibration.py",
    "evaluation/g3_numeric.py",
    "evaluation/g3_path_equivalence.py",
    "evaluation/g3_path_equivalence_assurance.py",
    "evaluation/g3_propagation_assurance.py",
    "evaluation/g3_propagation_metrics.py",
    "evaluation/g3_receiver_impact.py",
    "evaluation/g3_receiver_assurance.py",
    "evaluation/g3_sample_size_planning.py",
    "evaluation/g3_signed_attestation.py",
)))
EVALUATOR_SOURCE_PATHS = tuple(
    sorted(
        (
            *V2_EVALUATOR_SOURCE_PATHS,
            ARTIFACT_SCHEMA_RELATIVE_PATH,
            SCHEMA_V3_RELATIVE_PATH,
            "artifacts/provenance.py",
        )
    )
)
_RUN_ID_PATTERN = re.compile(r"^[a-z][a-z0-9._-]{2,63}$")


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _runtime_identity() -> dict[str, Any]:
    """Capture a self-reported environment identity without implying attestation."""
    from artifacts.provenance import versions

    observed = versions()
    package_names = (
        "numpy",
        "scikit-learn",
        "torch",
        "sionna",
        "sionna-rt",
        "sionna-phy",
    )
    return {
        "schema": "loslab.g3_execution_environment.v1",
        "python": observed["python"],
        "python_implementation": platform.python_implementation(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "machine": platform.machine(),
        "packages": {name: observed[name] for name in package_names},
        "source_revision": observed["git_sha"],
    }


def _authorize_run_manifest_signer(signer_id: str, policy_bytes: bytes) -> dict[str, str]:
    """Resolve a cryptographically verified principal to an externally governed role."""
    policy = _json_bytes(policy_bytes, "G3 run-manifest signer policy")
    if not isinstance(policy, dict) or set(policy) != {"schema", "policy_id", "signers"}:
        raise ValueError("run-manifest signer policy fields do not match v1")
    if policy_bytes != _canonical_json_bytes(policy):
        raise ValueError("run-manifest signer policy must use canonical JSON bytes")
    if policy.get("schema") != SIGNER_POLICY_SCHEMA:
        raise ValueError("unsupported run-manifest signer policy schema")
    policy_id = policy.get("policy_id")
    signers = policy.get("signers")
    if (
        not isinstance(policy_id, str)
        or not policy_id.strip()
        or not isinstance(signers, list)
        or not signers
    ):
        raise ValueError("run-manifest signer policy needs a policy ID and signer list")
    principals: set[str] = set()
    authorized: dict[str, Any] | None = None
    for entry in signers:
        fields = {"principal", "organization", "roles", "allowed_purposes"}
        if not isinstance(entry, dict) or set(entry) != fields:
            raise ValueError("run-manifest signer entries must match the v1 contract")
        principal = entry.get("principal")
        organization = entry.get("organization")
        roles = entry.get("roles")
        purposes = entry.get("allowed_purposes")
        if (
            not isinstance(principal, str)
            or not principal.strip()
            or principal != principal.strip()
            or not isinstance(organization, str)
            or not organization.strip()
        ):
            raise ValueError("signer principal and organization must be nonempty strings")
        if principal in principals:
            raise ValueError("run-manifest signer principals must be unique")
        principals.add(principal)
        if (
            not isinstance(roles, list)
            or not roles
            or any(not isinstance(role, str) or not role.strip() for role in roles)
            or len(set(roles)) != len(roles)
        ):
            raise ValueError("signer roles must be unique nonempty strings")
        if (
            not isinstance(purposes, list)
            or not purposes
            or any(not isinstance(purpose, str) or not purpose.strip() for purpose in purposes)
            or len(set(purposes)) != len(purposes)
        ):
            raise ValueError("signer allowed_purposes must be unique nonempty strings")
        if any(purpose != SIGNER_POLICY_PURPOSE for purpose in purposes):
            raise ValueError("signer policy contains an unsupported run-manifest purpose")
        if principal == signer_id:
            authorized = entry
    if authorized is None:
        raise ValueError("run-manifest signer is absent from the trusted role policy")
    if SIGNER_POLICY_PURPOSE not in authorized["allowed_purposes"]:
        raise ValueError("run-manifest signer role policy does not authorize this purpose")
    return {
        "policy_id": policy_id,
        "policy_sha256": hashlib.sha256(policy_bytes).hexdigest(),
        "signer_id": signer_id,
        "organization": authorized["organization"],
        "roles": ",".join(sorted(authorized["roles"])),
    }


def _require_independent_signer_policy_custody(
    manifest_signature: dict[str, str], policy_signature: dict[str, str]
) -> None:
    """Require a distinct principal and allowed-signers trust input for policy bytes."""
    if manifest_signature.get("signer_id") == policy_signature.get("signer_id"):
        raise ValueError("signer-policy custodian must differ from the run-manifest signer")
    if manifest_signature.get("allowed_signers_sha256") == policy_signature.get(
        "allowed_signers_sha256"
    ):
        raise ValueError("signer-policy trust keyring must differ from the run-manifest keyring")


def _source_bundle_sha256(sources: list[dict[str, Any]]) -> str:
    records = [
        {"path": item["path"], "size_bytes": item["size_bytes"], "sha256": item["sha256"]}
        for item in sources
    ]
    return hashlib.sha256(_canonical_json_bytes(records)).hexdigest()


def _validate_run_verification_report(report: dict[str, Any]) -> None:
    """Validate the public verifier result against its strict output contract."""
    source_dir = Path(__file__).resolve().parents[1]
    schema_bytes = _read_rooted_source_file(
        source_dir,
        VERIFICATION_REPORT_SCHEMA_RELATIVE_PATH,
        limit=MAX_SCHEMA_BYTES,
    )
    report["report_schema_sha256"] = hashlib.sha256(schema_bytes).hexdigest()
    schema = _json_bytes(schema_bytes, "G3 run verification report schema")
    import jsonschema

    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(report))
    if errors:
        raise RuntimeError(
            "G3 run verification report violates schema: "
            + "; ".join(error.message for error in errors[:10])
        )


def _rooted_file_identity(
    root_fd: int, root: Path, file_path: str | Path, *, limit: int, capture: bool
) -> tuple[str, bytes | None, int, str]:
    path = Path(file_path)
    try:
        relative = path.relative_to(root) if path.is_absolute() else path
    except ValueError as error:
        raise ValueError("manifest file must be located below its declared root") from error
    relative_path = relative.as_posix()
    parts = _safe_relative_path(relative_path)
    data, size, digest = _read_inventory_file(root_fd, parts, limit, capture=capture)
    return relative_path, data, size, digest


def _read_rooted_source_file(root: Path, relative_path: str, *, limit: int) -> bytes:
    root_fd = os.open(
        root, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
    )
    try:
        data, _, _ = _read_inventory_file(
            root_fd, _safe_relative_path(relative_path), limit, capture=True
        )
        if data is None:
            raise ValueError(f"source file was not captured: {relative_path}")
        return data
    finally:
        os.close(root_fd)


def _validate_joint_evaluator_input_schema(
    payload: Any, *, source_root: Path
) -> str:
    """Validate a serialized joint input against its version-pinned source schema."""
    if not isinstance(payload, dict):
        raise ValueError("G3 joint evaluator input must be an object")
    schema_by_version = {
        "loslab.g3_joint_evaluator_inputs.v1": INPUT_SCHEMA_RELATIVE_PATH,
        "loslab.g3_joint_evaluator_inputs.v2": INPUT_SCHEMA_V2_RELATIVE_PATH,
        "loslab.g3_joint_evaluator_inputs.v3": INPUT_SCHEMA_V3_RELATIVE_PATH,
        "loslab.g3_joint_evaluator_inputs.v4": INPUT_SCHEMA_V4_RELATIVE_PATH,
        "loslab.g3_joint_evaluator_inputs.v5": INPUT_SCHEMA_V5_RELATIVE_PATH,
        "loslab.g3_joint_evaluator_inputs.v6": INPUT_SCHEMA_V6_RELATIVE_PATH,
    }
    schema_path = schema_by_version.get(payload.get("schema"))
    if schema_path is None:
        raise ValueError("unsupported G3 joint evaluator input schema")
    schema = _json_bytes(
        _read_rooted_source_file(source_root, schema_path, limit=MAX_SCHEMA_BYTES),
        "G3 joint evaluator input schema",
    )
    import jsonschema

    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(payload))
    if errors:
        raise ValueError(
            "G3 joint evaluator input violates schema: "
            + "; ".join(error.message for error in errors[:10])
        )
    return schema_path


def _verify_qualification_contract_bindings(
    contract: dict[str, Any],
    input_bytes: bytes,
    *,
    require_powered_design: bool = False,
    require_raw_path_criteria: bool = False,
) -> str:
    verification = verify_g3_qualification_run_contract(contract)
    contract_schema = contract.get("schema")
    if require_powered_design:
        if contract_schema not in {
            "loslab.g3_qualification_run_contract.v3",
            "loslab.g3_qualification_run_contract.v4",
        }:
            raise ValueError(
                "new contract-bound G3 evaluation requires qualification contract v3 or v4"
            )
        design_verification = verify_g3_joint_sample_size_design(
            contract["sample_size_design"]
        )
        if design_verification["complete_precision_powered_g3_design"] is not True:
            raise ValueError("contract-bound G3 evaluation requires a complete powered design")
    if require_raw_path_criteria and contract_schema != (
        "loslab.g3_qualification_run_contract.v4"
    ):
        raise ValueError(
            "strict signed reviewer admission requires qualification contract v4 "
            "with raw path criteria and a complete powered design"
        )
    input_payload = _json_bytes(input_bytes, "G3 joint evaluator input")
    if not isinstance(input_payload, dict):
        raise ValueError("G3 joint evaluator input must be an object")
    input_schema = input_payload.get("schema")
    if contract_schema == "loslab.g3_qualification_run_contract.v3":
        allowed_input_schemas = {"loslab.g3_joint_evaluator_inputs.v4"}
    elif contract_schema == "loslab.g3_qualification_run_contract.v4":
        allowed_input_schemas = {"loslab.g3_joint_evaluator_inputs.v6"}
        if not require_powered_design:
            allowed_input_schemas.add("loslab.g3_joint_evaluator_inputs.v5")
    else:
        allowed_input_schemas = set()
    if input_schema not in allowed_input_schemas:
        raise ValueError(
            "contract-bound G3 evaluation requires input v4 for contract v3; "
            "new contract-v4 runs require input v6 with budget-bound diffuse diagnostics"
        )
    if input_payload.get("joint_acceptance_family") != contract["joint_acceptance_family"]:
        raise ValueError("qualification contract acceptance family differs from run input")
    receiver_input = input_payload.get("receiver_impact")
    if not isinstance(receiver_input, dict) or (
        receiver_input.get("candidate_config_id") != contract["candidate"]["config_id"]
        or receiver_input.get("challenger_config_id") != contract["challenger"]["config_id"]
    ):
        raise ValueError("qualification contract receiver configurations differ from run input")
    expected_execution = {
        "candidate": contract["candidate"],
        "challenger": contract["challenger"],
        "waveform": contract["waveform"],
        "link_budget_noise": contract["link_budget_noise"],
    }
    if input_payload.get("receiver_execution") != expected_execution:
        raise ValueError(
            "qualification contract waveform, receiver configs, or link budget differ from execution context"
        )
    propagation_input = input_payload.get("propagation_metrics")
    if not isinstance(propagation_input, dict):
        raise ValueError("qualification contract propagation-metric input must be an object")
    if contract_schema == "loslab.g3_qualification_run_contract.v4":
        frozen_bandwidth = contract["task_scope"].get("effective_bandwidth_hz")
        supplied_bandwidth = propagation_input.get("effective_bandwidth_hz")
        if frozen_bandwidth is None and require_powered_design:
            raise ValueError(
                "powered contract-v4 G3 evaluation requires frozen propagation effective bandwidth"
            )
        if frozen_bandwidth is not None and _canonical_json_bytes(
            frozen_bandwidth
        ) != _canonical_json_bytes(supplied_bandwidth):
            raise ValueError(
                "propagation effective bandwidth differs from frozen qualification task scope"
            )
    path_input = input_payload.get("path_equivalence")
    if not isinstance(path_input, dict) or path_input.get("expected_scenarios") != (
        contract["sampling"]["path_equivalence_scenarios"]
    ):
        raise ValueError("qualification contract path roster differs from run input")
    path_records = path_input.get("records")
    if not isinstance(path_records, list) or not path_records:
        raise ValueError("path-equivalence input must contain a nonempty comparison roster")
    if contract_schema == "loslab.g3_qualification_run_contract.v4":
        expected_path_configuration = {
            "candidate_configuration_id": contract["candidate"]["config_id"],
            "challenger_configuration_id": contract["challenger"]["config_id"],
            **{
                key: contract["path_matching"][key]
                for key in (
                    "vertex_tolerance_m",
                    "delay_tolerance_s",
                    "direction_tolerance_rad",
                    "h_floor",
                    "transmit_power_per_re_w",
                    "noise_power_per_re_w",
                )
            },
        }
        if path_input.get("comparison_configuration") != expected_path_configuration:
            raise ValueError(
                "raw path-equivalence configuration differs from frozen contract v4"
            )
        if input_schema == "loslab.g3_joint_evaluator_inputs.v6" and path_input.get(
            "solver_budget_provenance"
        ) != contract["solver_budget"]:
            raise ValueError("diffuse-path solver budget differs from frozen contract v4")
        for index, record in enumerate(path_records):
            if not isinstance(record, dict) or "comparison" in record:
                raise ValueError(
                    f"raw path-equivalence record {index} must omit supplied comparison"
                )
    else:
        expected_path_configuration = {
            "candidate_configuration_id": contract["candidate"]["config_id"],
            "challenger_configuration_id": contract["challenger"]["config_id"],
            "delay_tolerance_s": contract["path_matching"]["delay_tolerance_s"],
        }
        expected_coverage = contract["path_matching"]["minimum_bidirectional_coverage"]
        for index, record in enumerate(path_records):
            if not isinstance(record, dict) or not isinstance(record.get("comparison"), dict):
                raise ValueError(f"path-equivalence record {index} needs a comparison object")
            comparison = record["comparison"]
            configuration = comparison.get("configuration")
            if not isinstance(configuration, dict) or any(
                configuration.get(key) != expected
                for key, expected in expected_path_configuration.items()
            ) or comparison.get("coverage_criterion") != expected_coverage:
                raise ValueError(
                    f"path-equivalence comparison {index} configuration differs from frozen contract"
                )
    expected_episodes = {
        tuple(item) for item in contract["sampling"]["path_equivalence_scenarios"]
    }
    receiver_records = receiver_input.get("records")
    if not isinstance(receiver_records, list) or not receiver_records:
        raise ValueError("receiver-impact input must include a nonempty episode-bound record list")
    observed_episodes: set[tuple[str, str, str]] = set()
    config_digests = {
        contract[role]["config_id"]: contract[role]["configuration_sha256"]
        for role in ("candidate", "challenger")
    }
    for index, record in enumerate(receiver_records):
        if not isinstance(record, dict):
            raise ValueError(f"receiver-impact record {index} must be an object")
        family_id = record.get("family_id")
        condition = record.get("condition")
        episode_id = record.get("episode_id")
        if (
            not isinstance(family_id, str)
            or not family_id.strip()
            or not isinstance(condition, str)
            or condition not in {"LOS", "NLOS"}
            or not isinstance(episode_id, str)
            or not episode_id.strip()
        ):
            raise ValueError(
                f"receiver-impact record {index} needs family_id, LOS/NLOS condition, "
                "and episode_id"
            )
        configuration_id = record.get("configuration_id")
        if (
            not isinstance(configuration_id, str)
            or configuration_id not in config_digests
            or record.get("receiver_configuration_sha256")
            != config_digests.get(configuration_id)
        ):
            raise ValueError(
                f"receiver-impact record {index} configuration binding differs from frozen contract"
            )
        observed_episodes.add((family_id, condition, episode_id))
    if observed_episodes != expected_episodes:
        raise ValueError(
            "receiver-impact observed episode roster differs from the frozen path roster"
        )
    return str(verification["contract_sha256"])


def g3_evaluator_source_bundle_identity(
    *, source_root: str | Path | None = None
) -> dict[str, Any]:
    """Return the fixed G3 source roster and its digest before a run is launched."""
    root = (
        Path(source_root).resolve(strict=True)
        if source_root is not None
        else Path(__file__).resolve().parents[1]
    )
    if not root.is_dir():
        raise ValueError("source_root must be a directory")
    root_fd = os.open(
        root, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
    )
    sources: list[dict[str, Any]] = []
    total = 0
    try:
        for relative_path in EVALUATOR_SOURCE_PATHS:
            remaining = _MAX_SOURCE_BUNDLE_BYTES - total
            if remaining <= 0:
                raise ValueError("evaluator source roster exceeds its total byte limit")
            _, size, digest = _read_inventory_file(
                root_fd,
                _safe_relative_path(relative_path),
                min(MAX_FILE_BYTES, remaining),
                capture=False,
            )
            total += size
            sources.append({"path": relative_path, "size_bytes": size, "sha256": digest})
    finally:
        os.close(root_fd)
    return {
        "evaluator_sources": sources,
        "evaluator_source_bundle_sha256": _source_bundle_sha256(sources),
    }


def build_g3_joint_evidence_run_manifest(
    run_id: str,
    input_path: str | Path,
    artifact_path: str | Path,
    *,
    package_root: str | Path | None = None,
    source_root: str | Path | None = None,
    qualification_contract: dict[str, Any] | None = None,
    runtime_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe exact run files, optional preregistration, and scorer source identities."""
    if not isinstance(run_id, str) or _RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError(
            "run_id must be 3–64 lowercase letters, digits, dots, underscores, or hyphens"
        )
    input_file = Path(os.path.abspath(input_path))
    artifact_file = Path(os.path.abspath(artifact_path))
    package_dir = Path(package_root) if package_root is not None else input_file.parent
    package_dir = package_dir.resolve(strict=True)
    source_dir = (
        Path(source_root).resolve(strict=True)
        if source_root is not None
        else Path(__file__).resolve().parents[1]
    )
    if not package_dir.is_dir() or not source_dir.is_dir():
        raise ValueError("package_root and source_root must be directories")
    package_fd = os.open(
        package_dir, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
    )
    source_fd = os.open(
        source_dir, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
    )
    try:
        input_relative, input_bytes, input_size, input_digest = _rooted_file_identity(
            package_fd,
            package_dir,
            input_file,
            limit=_MAX_CAPTURE_BYTES,
            capture=True,
        )
        artifact_relative, artifact_bytes, artifact_size, artifact_digest = _rooted_file_identity(
            package_fd,
            package_dir,
            artifact_file,
            limit=_MAX_CAPTURE_BYTES,
            capture=True,
        )
        if qualification_contract is not None:
            if input_bytes is None:
                raise ValueError("joint evaluator input was not captured")
            _validate_joint_evaluator_input_schema(
                _json_bytes(input_bytes, "G3 joint evaluator input"),
                source_root=source_dir,
            )
            _verify_qualification_contract_bindings(qualification_contract, input_bytes)
        del input_bytes, artifact_bytes
        if input_relative == artifact_relative:
            raise ValueError("input and artifact paths must be distinct")
        sources: list[dict[str, Any]] = []
        source_bytes_total = 0
        for relative_path in EVALUATOR_SOURCE_PATHS:
            parts = _safe_relative_path(relative_path)
            remaining_source_budget = _MAX_SOURCE_BUNDLE_BYTES - source_bytes_total
            if remaining_source_budget <= 0:
                raise ValueError("evaluator source roster exceeds its total byte limit")
            _, size, digest = _read_inventory_file(
                source_fd,
                parts,
                min(MAX_FILE_BYTES, remaining_source_budget),
                capture=False,
            )
            source_bytes_total += size
            sources.append({"path": relative_path, "size_bytes": size, "sha256": digest})
    finally:
        os.close(package_fd)
        os.close(source_fd)
    source_bundle_sha256 = _source_bundle_sha256(sources)
    if qualification_contract is not None and qualification_contract["source_identity"][
        "source_bundle_sha256"
    ] != source_bundle_sha256:
        raise ValueError("qualification contract source bundle differs from run receipt sources")
    manifest = {
        "schema": "loslab.g3_joint_evidence_run_manifest.v3",
        "run_id": run_id,
        "qualification_status": "NOT_QUALIFIED",
        "runtime_identity": (
            _runtime_identity() if runtime_identity is None else runtime_identity
        ),
        "input": {
            "path": input_relative,
            "size_bytes": input_size,
            "sha256": input_digest,
            "role": "joint_evaluator_input",
        },
        "artifact": {
            "path": artifact_relative,
            "size_bytes": artifact_size,
            "sha256": artifact_digest,
            "role": "joint_evaluator_artifact",
        },
        "evaluator_sources": sources,
        "evaluator_source_bundle_sha256": source_bundle_sha256,
    }
    if qualification_contract is not None:
        if not isinstance(qualification_contract, dict):
            raise TypeError("qualification_contract must be an object")
        verify_g3_qualification_run_contract(qualification_contract)
        if qualification_contract["run_id"] != run_id:
            raise ValueError("qualification contract run_id differs from run receipt")
        manifest["qualification_contract"] = qualification_contract
    import jsonschema

    manifest_schema = _json_bytes(
        _read_rooted_source_file(
            source_dir, SCHEMA_V3_RELATIVE_PATH, limit=MAX_SCHEMA_BYTES
        ),
        "G3 run manifest schema",
    )
    schema_errors = list(
        jsonschema.Draft202012Validator(manifest_schema).iter_errors(manifest)
    )
    if schema_errors:
        raise ValueError(
            "G3 run manifest violates v3 schema: "
            + "; ".join(error.message for error in schema_errors[:10])
        )
    return manifest


def verify_g3_joint_evidence_run_manifest(
    manifest_path: str | Path,
    *,
    package_root: str | Path | None = None,
    source_root: str | Path | None = None,
    signature_path: str | Path | None = None,
    trusted_allowed_signers_path: str | Path | None = None,
    trusted_signer_policy_path: str | Path | None = None,
    signer_id: str | None = None,
    trusted_signer_policy_signature_path: str | Path | None = None,
    trusted_policy_allowed_signers_path: str | Path | None = None,
    policy_signer_id: str | None = None,
    require_signed_reviewer_admission: bool = False,
) -> dict[str, Any]:
    """Verify file hashes and replay the evaluator from a small standalone manifest.

    This is a numerical run receipt, not a G3.03 acquisition package or scientific qualification.
    An optional frozen contract is integrity-checked and joined to the evaluator input, but
    the result remains explicitly not qualified. When all signature arguments are supplied,
    the exact captured manifest bytes are checked against an external allowed-signers file.
    Inventory paths are opened relative to one directory descriptor without following
    symlinks. Contents are captured once and the exact captured bytes are replayed.
    """
    report: dict[str, Any] = {
        "schema": "loslab.g3_joint_evidence_run_verification.v1",
        "status": "INCONSISTENT",
        "qualification_status": "NOT_QUALIFIED",
        "run_id": None,
        "manifest_signature_status": "not_provided",
        "signer_authorization_status": "not_provided",
        "signer_policy_signature_status": "not_provided",
        "signed_reviewer_admission_status": "not_requested",
        "signer_policy_custody": "not_established_by_this_verifier",
        "runtime_identity_status": "not_recorded",
        "errors": [],
        "limitations": [
            "A verified detached signature authenticates the manifest bytes and signer principal, "
            "but does not establish trust-file custody, measurement provenance, runtime identity, "
            "or scientific qualification. Signer-role authorization requires the separate "
            "external policy input. A detached signature can bind the exact policy bytes to a "
            "policy-custodian principal, but the allowed-signers keyring's custody remains external.",
            "A v3 runtime identity is a self-reported package/platform snapshot. It does not attest "
            "the executing binaries, loaded libraries, hardware, or runtime truth.",
            "A successful replay confirms only that the retained artifact matches the current "
            "five-branch numerical evaluator for the exact input bytes.",
        ],
    }
    errors: list[str] = report["errors"]
    try:
        if not isinstance(require_signed_reviewer_admission, bool):
            raise TypeError("require_signed_reviewer_admission must be a boolean")
        if require_signed_reviewer_admission:
            report["signed_reviewer_admission_status"] = "required"
        signature_options = (signature_path, trusted_allowed_signers_path, signer_id)
        if any(value is not None for value in signature_options) and not all(
            value is not None for value in signature_options
        ):
            raise ValueError(
                "signature_path, trusted_allowed_signers_path, and signer_id must be supplied together"
            )
        if trusted_signer_policy_path is not None and signature_path is None:
            raise ValueError("trusted_signer_policy_path requires a verified manifest signature")
        policy_signature_options = (
            trusted_signer_policy_signature_path,
            trusted_policy_allowed_signers_path,
            policy_signer_id,
        )
        if any(value is not None for value in policy_signature_options) and not all(
            value is not None for value in policy_signature_options
        ):
            raise ValueError(
                "trusted_signer_policy_signature_path, trusted_policy_allowed_signers_path, "
                "and policy_signer_id must be supplied together"
            )
        if any(value is not None for value in policy_signature_options) and (
            trusted_signer_policy_path is None or signature_path is None
        ):
            raise ValueError(
                "a signer-policy signature requires both a signer policy and a signed run manifest"
            )
        if require_signed_reviewer_admission and (
            not all(value is not None for value in signature_options)
            or trusted_signer_policy_path is None
            or not all(value is not None for value in policy_signature_options)
        ):
            raise ValueError(
                "strict signed reviewer admission requires a signed run manifest, an externally "
                "signed signer policy, and both trusted allowed-signers files"
            )
        manifest_file = Path(manifest_path)
        manifest_raw = _read_bounded_regular_file(
            manifest_file, limit=MAX_PACKAGE_BYTES, description="G3 run manifest"
        )
        signature_identity: dict[str, str] | None = None
        if signature_path is not None:
            assert trusted_allowed_signers_path is not None
            assert signer_id is not None
            signature_identity = verify_openssh_signature_for_bytes(
                manifest_raw,
                signature_path,
                trusted_allowed_signers_path,
                signer_id=signer_id,
                namespace=G3_RUN_MANIFEST_SIGNATURE_NAMESPACE,
            )
            report.update(
                {
                    "manifest_signature_status": "verified",
                    "manifest_signature_namespace": signature_identity["namespace"],
                    "manifest_signer_id": signature_identity["signer_id"],
                    "manifest_signature_sha256": signature_identity["signature_sha256"],
                    "trusted_allowed_signers_sha256": signature_identity[
                        "allowed_signers_sha256"
                    ],
                    "trust_file_custody": "not_established_by_this_verifier",
                }
            )
            if trusted_signer_policy_path is not None:
                policy_bytes = _read_bounded_regular_file(
                    Path(trusted_signer_policy_path),
                    limit=MAX_SCHEMA_BYTES,
                    description="G3 run-manifest signer policy",
                )
                if trusted_signer_policy_signature_path is not None:
                    assert trusted_policy_allowed_signers_path is not None
                    assert policy_signer_id is not None
                    policy_signature_identity = verify_openssh_signature_for_bytes(
                        policy_bytes,
                        trusted_signer_policy_signature_path,
                        trusted_policy_allowed_signers_path,
                        signer_id=policy_signer_id,
                        namespace=G3_SIGNER_POLICY_SIGNATURE_NAMESPACE,
                    )
                    _require_independent_signer_policy_custody(
                        signature_identity, policy_signature_identity
                    )
                    report.update(
                        {
                            "signer_policy_signature_status": "verified",
                            "signer_policy_signature_namespace": policy_signature_identity[
                                "namespace"
                            ],
                            "signer_policy_signer_id": policy_signature_identity[
                                "signer_id"
                            ],
                            "signer_policy_detached_signature_sha256": policy_signature_identity[
                                "signature_sha256"
                            ],
                            "signer_policy_allowed_signers_sha256": policy_signature_identity[
                                "allowed_signers_sha256"
                            ],
                        }
                    )
                signer_authorization = _authorize_run_manifest_signer(
                    signature_identity["signer_id"], policy_bytes
                )
                if require_signed_reviewer_admission and "independent_reviewer" not in (
                    signer_authorization["roles"].split(",")
                ):
                    raise ValueError(
                        "strict signed reviewer admission requires the independent_reviewer role"
                    )
                report.update(
                    {
                        "signer_authorization_status": "authorized_for_run_manifest_verification",
                        "manifest_signer_organization": signer_authorization["organization"],
                        "manifest_signer_roles": signer_authorization["roles"],
                        "trusted_signer_policy_id": signer_authorization["policy_id"],
                        "trusted_signer_policy_sha256": signer_authorization["policy_sha256"],
                    }
                )
            else:
                report["signer_authorization_status"] = "not_provided"
        if require_signed_reviewer_admission:
            if report["manifest_signature_status"] != "verified":
                raise ValueError("strict signed reviewer admission requires a verified run signature")
            if report["signer_policy_signature_status"] != "verified":
                raise ValueError("strict signed reviewer admission requires a verified policy signature")
        manifest = _json_bytes(manifest_raw, "G3 run manifest")
        if not isinstance(manifest, dict):
            raise ValueError("G3 run manifest root must be an object")
        report["run_id"] = manifest.get("run_id")
        import jsonschema

        source_dir = (
            Path(source_root).resolve(strict=True)
            if source_root is not None
            else Path(__file__).resolve().parents[1]
        )
        if not source_dir.is_dir():
            raise ValueError("source_root must be a directory")

        def validate_schema(document: Any, schema_relative_path: str, label: str) -> None:
            schema = _json_bytes(
                _read_rooted_source_file(
                    source_dir, schema_relative_path, limit=MAX_SCHEMA_BYTES
                ),
                f"{label} schema",
            )
            schema_errors = list(jsonschema.Draft202012Validator(schema).iter_errors(document))
            if schema_errors:
                raise ValueError(
                    f"{label} violates schema: "
                    + "; ".join(error.message for error in schema_errors[:10])
                )

        manifest_version = manifest.get("schema")
        if manifest_version == "loslab.g3_joint_evidence_run_manifest.v1":
            manifest_schema_path = V1_SCHEMA_RELATIVE_PATH
            expected_source_paths = V1_EVALUATOR_SOURCE_PATHS
        elif manifest_version == "loslab.g3_joint_evidence_run_manifest.v2":
            manifest_schema_path = SCHEMA_RELATIVE_PATH
            expected_source_paths = V2_EVALUATOR_SOURCE_PATHS
        elif manifest_version == "loslab.g3_joint_evidence_run_manifest.v3":
            manifest_schema_path = SCHEMA_V3_RELATIVE_PATH
            expected_source_paths = EVALUATOR_SOURCE_PATHS
        else:
            raise ValueError("unsupported G3 run manifest schema")
        if require_signed_reviewer_admission and manifest_version != (
            "loslab.g3_joint_evidence_run_manifest.v3"
        ):
            raise ValueError(
                "strict signed reviewer admission requires run-manifest v3 runtime provenance"
            )
        validate_schema(manifest, manifest_schema_path, "G3 run manifest")
        if manifest_version == "loslab.g3_joint_evidence_run_manifest.v3":
            runtime_identity = manifest["runtime_identity"]
            report.update(
                {
                    "runtime_identity_status": "recorded_self_reported_unattested",
                    "runtime_identity_sha256": hashlib.sha256(
                        _canonical_json_bytes(runtime_identity)
                    ).hexdigest(),
                }
            )
        qualification_contract = manifest.get("qualification_contract")
        if require_signed_reviewer_admission and qualification_contract is None:
            raise ValueError("strict signed reviewer admission requires a frozen G3 contract")
        if require_signed_reviewer_admission and qualification_contract.get("schema") != (
            "loslab.g3_qualification_run_contract.v4"
        ):
            raise ValueError(
                "strict signed reviewer admission requires qualification contract v4 "
                "with raw path criteria and a complete powered design"
            )
        qualification_contract_sha256: str | None = None
        if qualification_contract is not None:
            if qualification_contract["run_id"] != manifest["run_id"]:
                raise ValueError("qualification contract run_id differs from run receipt")
            qualification_schema_path = (
                QUALIFICATION_CONTRACT_SCHEMA_V4_RELATIVE_PATH
                if qualification_contract.get("schema")
                == "loslab.g3_qualification_run_contract.v4"
                else QUALIFICATION_CONTRACT_SCHEMA_RELATIVE_PATH
                if qualification_contract.get("schema")
                == "loslab.g3_qualification_run_contract.v3"
                else QUALIFICATION_CONTRACT_SCHEMA_V2_RELATIVE_PATH
            )
            validate_schema(
                qualification_contract,
                qualification_schema_path,
                "G3 qualification contract",
            )
            qualification_contract_sha256 = verify_g3_qualification_run_contract(
                qualification_contract
            )["contract_sha256"]
        if not manifest_raw.endswith(b"\n") or manifest_raw != (
            json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8"):
            raise ValueError("G3 run manifest must use canonical compact JSON and one newline")

        root = Path(package_root) if package_root is not None else manifest_file.parent
        root = root.resolve(strict=True)
        if (
            not root.is_dir()
            or not hasattr(os, "O_DIRECTORY")
            or os.open not in os.supports_dir_fd
        ):
            raise ValueError("package root must be a directory with safe dir_fd support")
        root_fd = os.open(
            root, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
        )
        captured: dict[str, bytes] = {}
        total = 0
        seen_paths: set[tuple[str, ...]] = set()
        try:
            for field, expected_role, limit in (
                (
                    "input",
                    "joint_evaluator_input",
                    min(MAX_INPUT_BYTES, _MAX_CAPTURE_BYTES),
                ),
                (
                    "artifact",
                    "joint_evaluator_artifact",
                    min(MAX_ARTIFACT_BYTES, _MAX_CAPTURE_BYTES),
                ),
            ):
                entry = manifest[field]
                parts = _safe_relative_path(entry["path"])
                if parts in seen_paths:
                    raise ValueError("input and artifact paths must be distinct")
                seen_paths.add(parts)
                remaining = _MAX_MANIFEST_TOTAL_BYTES - total
                data, size, digest = _read_inventory_file(
                    root_fd, parts, min(limit, remaining), capture=True
                )
                if data is None:
                    raise ValueError(f"{field} bytes were not captured")
                total += size
                if size != entry["size_bytes"] or digest != entry["sha256"]:
                    raise ValueError(f"{field} size or SHA-256 differs from the manifest")
                if entry["role"] != expected_role:
                    raise ValueError(f"{field} role must be {expected_role!r}")
                captured[field] = data
        finally:
            os.close(root_fd)

        sources = manifest["evaluator_sources"]
        expected_source_paths = sorted(expected_source_paths)
        if [entry["path"] for entry in sources] != expected_source_paths:
            raise ValueError(
                "evaluator source roster differs from the selected manifest-version roster"
            )
        if manifest["evaluator_source_bundle_sha256"] != _source_bundle_sha256(sources):
            raise ValueError("evaluator source bundle digest does not match its file roster")
        if qualification_contract is not None and qualification_contract["source_identity"][
            "source_bundle_sha256"
        ] != manifest["evaluator_source_bundle_sha256"]:
            raise ValueError("qualification contract source bundle differs from run receipt sources")
        source_fd = os.open(
            source_dir,
            os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW,
        )
        source_bytes_total = 0
        try:
            for entry in sources:
                parts = _safe_relative_path(entry["path"])
                remaining_source_budget = _MAX_SOURCE_BUNDLE_BYTES - source_bytes_total
                if remaining_source_budget <= 0:
                    raise ValueError("evaluator source roster exceeds its total byte limit")
                _, size, digest = _read_inventory_file(
                    source_fd,
                    parts,
                    min(MAX_FILE_BYTES, remaining_source_budget),
                    capture=False,
                )
                source_bytes_total += size
                if size != entry["size_bytes"] or digest != entry["sha256"]:
                    raise ValueError(f"evaluator source bytes differ from manifest: {entry['path']}")
        finally:
            os.close(source_fd)

        input_payload = _json_bytes(captured["input"], "G3 joint evaluator input")
        if not isinstance(input_payload, dict):
            raise ValueError("G3 joint evaluator input must be an object")
        _validate_joint_evaluator_input_schema(input_payload, source_root=source_dir)
        artifact_schema_path = (
            ARTIFACT_SCHEMA_RELATIVE_PATH
            if manifest_version == "loslab.g3_joint_evidence_run_manifest.v3"
            else ARTIFACT_SCHEMA_V1_RELATIVE_PATH
        )
        validate_schema(
            _json_bytes(captured["artifact"], "G3 joint evaluator artifact"),
            artifact_schema_path,
            "G3 joint evaluator artifact",
        )
        if qualification_contract is not None:
            qualification_contract_sha256 = _verify_qualification_contract_bindings(
                qualification_contract,
                captured["input"],
                require_powered_design=require_signed_reviewer_admission,
                require_raw_path_criteria=require_signed_reviewer_admission,
            )
        if require_signed_reviewer_admission:
            report["signed_reviewer_admission_status"] = (
                "signature_role_policy_and_contract_preflight_passed"
            )
        coefficient_tolerance = (
            qualification_contract["path_matching"]["coefficient_relative_tolerance"]
            if qualification_contract is not None
            and qualification_contract.get("schema")
            == "loslab.g3_qualification_run_contract.v4"
            else None
        )
        replay = verify_g3_joint_evidence_bytes(
            captured["input"],
            captured["artifact"],
            coefficient_relative_tolerance=coefficient_tolerance,
        )
        report.update(
            {
                "status": "PASS_INPUT_SCOPED_RECOMPUTATION",
                "input_sha256": hashlib.sha256(captured["input"]).hexdigest(),
                "artifact_sha256": hashlib.sha256(captured["artifact"]).hexdigest(),
                "bundle_sha256": replay["bundle_sha256"],
                "branch_count": replay["branch_count"],
                "evaluator_source_bundle_sha256": manifest[
                    "evaluator_source_bundle_sha256"
                ],
                "qualification_contract_sha256": qualification_contract_sha256,
                "qualification_contract_bound": qualification_contract is not None,
            }
        )
    except (ImportError, OSError, TypeError, ValueError, RuntimeError, KeyError) as error:
        errors.append(str(error))
    _validate_run_verification_report(report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        type=Path,
        nargs="?",
        help="existing versioned G3 joint run manifest JSON to verify",
    )
    parser.add_argument("--create-run-id", help="create a manifest instead of verifying one")
    parser.add_argument(
        "--input", type=Path, help="joint evaluator input file for manifest creation"
    )
    parser.add_argument(
        "--artifact", type=Path, help="joint evaluator artifact for manifest creation"
    )
    parser.add_argument(
        "--output", type=Path, help="new manifest destination (must not exist)"
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        help="optional durable verification-report destination (must not exist)",
    )
    parser.add_argument(
        "--package-root",
        type=Path,
        help="directory against which the manifest's relative file paths resolve",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        help="repository/source directory containing the fixed evaluator source roster",
    )
    parser.add_argument(
        "--qualification-contract",
        type=Path,
        help="optional frozen G3 qualification contract to bind into the run receipt",
    )
    parser.add_argument("--signature", type=Path, help="detached signature for the run manifest")
    parser.add_argument(
        "--trusted-allowed-signers",
        type=Path,
        help="externally governed OpenSSH allowed_signers file",
    )
    parser.add_argument(
        "--trusted-signer-policy",
        type=Path,
        help="optional external policy mapping the signing principal to its approved role and purpose",
    )
    parser.add_argument("--signer-id", help="OpenSSH principal expected to sign the manifest")
    parser.add_argument(
        "--trusted-signer-policy-signature",
        type=Path,
        help="optional detached signature over the exact external signer-policy bytes",
    )
    parser.add_argument(
        "--trusted-policy-allowed-signers",
        type=Path,
        help="externally governed OpenSSH allowed_signers file for the signer-policy custodian",
    )
    parser.add_argument(
        "--policy-signer-id", help="OpenSSH principal expected to sign the signer policy"
    )
    parser.add_argument(
        "--require-signed-reviewer-admission",
        action="store_true",
        help=(
            "require a signed v3 run receipt, externally signed reviewer-role policy, "
            "and frozen G3 contract before reporting input-scoped replay"
        ),
    )
    parser.add_argument(
        "--print-source-bundle",
        action="store_true",
        help="print the fixed evaluator source inventory for preregistering a run",
    )
    args = parser.parse_args(argv)
    if args.print_source_bundle:
        if any(
            value is not None
            for value in (
                args.manifest,
                args.create_run_id,
                args.input,
                args.artifact,
                args.output,
                args.report_output,
                args.qualification_contract,
                args.signature,
                args.trusted_allowed_signers,
                args.trusted_signer_policy,
                args.signer_id,
                args.trusted_signer_policy_signature,
                args.trusted_policy_allowed_signers,
                args.policy_signer_id,
            )
        ) or args.require_signed_reviewer_admission:
            parser.error("--print-source-bundle cannot be combined with run-manifest arguments")
        try:
            print(json.dumps(g3_evaluator_source_bundle_identity(source_root=args.source_root)))
        except (OSError, TypeError, ValueError, RuntimeError) as error:
            print(f"G3 source inventory error: {error}", file=sys.stderr)
            return 2
        return 0
    if args.qualification_contract is not None and args.create_run_id is None:
        parser.error("--qualification-contract requires --create-run-id")
    if args.create_run_id is not None and any(
        value is not None
        for value in (
            args.signature,
            args.trusted_allowed_signers,
            args.trusted_signer_policy,
            args.signer_id,
            args.trusted_signer_policy_signature,
            args.trusted_policy_allowed_signers,
            args.policy_signer_id,
        )
    ):
        parser.error("manifest signature options are only valid when verifying an existing manifest")
    if args.create_run_id is not None and args.require_signed_reviewer_admission:
        parser.error("--require-signed-reviewer-admission is only valid when verifying a manifest")
    if args.create_run_id is not None and args.report_output is not None:
        parser.error("--report-output is only valid when verifying an existing manifest")
    if args.create_run_id is not None:
        if (
            args.manifest is not None
            or args.input is None
            or args.artifact is None
            or args.output is None
        ):
            parser.error("manifest creation requires --input, --artifact, and --output only")
        try:
            value = build_g3_joint_evidence_run_manifest(
                args.create_run_id,
                args.input,
                args.artifact,
                package_root=args.package_root,
                source_root=args.source_root,
                qualification_contract=(
                    _json_bytes(
                        _read_bounded_regular_file(
                            args.qualification_contract,
                            limit=MAX_JSON_ARTIFACT_BYTES,
                            description="G3 qualification contract",
                        ),
                        "G3 qualification contract",
                    )
                    if args.qualification_contract is not None
                    else None
                ),
            )
            with args.output.open("xb") as stream:
                stream.write(_canonical_json_bytes(value))
                stream.flush()
                os.fsync(stream.fileno())
        except (OSError, TypeError, ValueError, RuntimeError) as error:
            print(f"G3 run manifest creation error: {error}", file=sys.stderr)
            return 2
        print(json.dumps({"schema": value["schema"], "run_id": value["run_id"]}))
        return 0
    if args.manifest is None:
        parser.error("provide a manifest to verify or use --create-run-id")
    report = verify_g3_joint_evidence_run_manifest(
        args.manifest,
        package_root=args.package_root,
        source_root=args.source_root,
        signature_path=args.signature,
        trusted_allowed_signers_path=args.trusted_allowed_signers,
        trusted_signer_policy_path=args.trusted_signer_policy,
        signer_id=args.signer_id,
        trusted_signer_policy_signature_path=args.trusted_signer_policy_signature,
        trusted_policy_allowed_signers_path=args.trusted_policy_allowed_signers,
        policy_signer_id=args.policy_signer_id,
        require_signed_reviewer_admission=args.require_signed_reviewer_admission,
    )
    if args.report_output is not None:
        try:
            _write_new_artifact(args.report_output, report)
        except OSError as error:
            print(f"G3 verification report write error: {error}", file=sys.stderr)
            return 2
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0 if report["status"] == "PASS_INPUT_SCOPED_RECOMPUTATION" else 1


if __name__ == "__main__":
    sys.exit(main())
