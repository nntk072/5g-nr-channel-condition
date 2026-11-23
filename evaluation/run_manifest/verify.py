"""Verification of joint-evidence run manifests against their recorded inputs and sources."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from artifacts.rebinding import recorded_digest_resolves

from evaluation.acquisition_package import (
    MAX_FILE_BYTES,
    MAX_JSON_ARTIFACT_BYTES,
    MAX_PACKAGE_BYTES,
    MAX_SCHEMA_BYTES,
    _json_bytes,
    _read_bounded_regular_file,
    _read_inventory_file,
    _safe_relative_path,
)
from evaluation.joint_evidence_artifact import (
    MAX_ARTIFACT_BYTES,
    MAX_INPUT_BYTES,
    verify_joint_evidence_bytes,
)
from evaluation.qualification_contract import verify_qualification_run_contract
from evaluation.run_manifest.common import (
    _canonical_json_bytes,
)
from evaluation.run_manifest.constants import (
    _MAX_CAPTURE_BYTES,
    _MAX_MANIFEST_TOTAL_BYTES,
    _MAX_SOURCE_BUNDLE_BYTES,
    _MAX_V4_MANIFEST_TOTAL_BYTES,
    ARTIFACT_SCHEMA_RELATIVE_PATH,
    ARTIFACT_SCHEMA_V1_RELATIVE_PATH,
    EVALUATOR_SOURCE_PATHS,
    QUALIFICATION_CONTRACT_SCHEMA_RELATIVE_PATH,
    QUALIFICATION_CONTRACT_SCHEMA_V2_RELATIVE_PATH,
    QUALIFICATION_CONTRACT_SCHEMA_V4_RELATIVE_PATH,
    RUN_MANIFEST_SIGNATURE_NAMESPACE,
    RUN_VERIFICATION_REPORT_SIGNATURE_NAMESPACE,
    RUNTIME_INVENTORY_SCHEMA_V3_RELATIVE_PATH,
    RUNTIME_INVENTORY_SIGNATURE_NAMESPACE,
    SCHEMA_RELATIVE_PATH,
    SCHEMA_V3_RELATIVE_PATH,
    SCHEMA_V4_RELATIVE_PATH,
    SIGNER_POLICY_SIGNATURE_NAMESPACE,
    V1_EVALUATOR_SOURCE_PATHS,
    V1_SCHEMA_RELATIVE_PATH,
    V2_EVALUATOR_SOURCE_PATHS,
    V4_EVALUATOR_SOURCE_PATHS,
)
from evaluation.run_manifest.contracts import (
    _validate_joint_evaluator_input_schema,
    _verify_qualification_contract_bindings,
)
from evaluation.run_manifest.signatures import (
    _authorize_run_manifest_signer,
    _require_independent_signer_policy_custody,
    _validate_run_verification_report,
    _verify_signed_runtime_inventory_bindings,
)
from evaluation.run_manifest.sources import (
    _read_rooted_source_file,
    _source_bundle_sha256,
)
from evaluation.signed_attestation import verify_openssh_signature_for_bytes

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


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
    runtime_inventory_signature_path: str | Path | None = None,
    trusted_runtime_inventory_signers_path: str | Path | None = None,
    runtime_inventory_signer_id: str | None = None,
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
        "runtime_inventory_signature_status": "not_provided",
        "runtime_inventory_binding_status": "not_provided",
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
        runtime_signature_options = (
            runtime_inventory_signature_path,
            trusted_runtime_inventory_signers_path,
            runtime_inventory_signer_id,
        )
        if any(value is not None for value in runtime_signature_options) and not all(
            value is not None for value in runtime_signature_options
        ):
            raise ValueError(
                "runtime inventory signature, trusted keyring, and signer ID must be supplied together"
            )
        if require_signed_reviewer_admission and (
            not all(value is not None for value in signature_options)
            or trusted_signer_policy_path is None
            or not all(value is not None for value in policy_signature_options)
            or runtime_inventory_signature_path is None
            or trusted_runtime_inventory_signers_path is None
            or runtime_inventory_signer_id is None
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
                namespace=RUN_MANIFEST_SIGNATURE_NAMESPACE,
            )
            report.update(
                {
                    "manifest_signature_status": "verified",
                    "manifest_signature_namespace": signature_identity["namespace"],
                    "manifest_signer_id": signature_identity["signer_id"],
                    "manifest_signature_sha256": signature_identity["signature_sha256"],
                    "trusted_allowed_signers_sha256": signature_identity["allowed_signers_sha256"],
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
                        namespace=SIGNER_POLICY_SIGNATURE_NAMESPACE,
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
                            "signer_policy_signer_id": policy_signature_identity["signer_id"],
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
                raise ValueError(
                    "strict signed reviewer admission requires a verified run signature"
                )
            if report["signer_policy_signature_status"] != "verified":
                raise ValueError(
                    "strict signed reviewer admission requires a verified policy signature"
                )
        manifest = _json_bytes(manifest_raw, "G3 run manifest")
        if not isinstance(manifest, dict):
            raise ValueError("G3 run manifest root must be an object")
        report["run_id"] = manifest.get("run_id")
        import jsonschema

        source_dir = (
            Path(source_root).resolve(strict=True)
            if source_root is not None
            else Path(__file__).resolve().parents[2]
        )
        if not source_dir.is_dir():
            raise ValueError("source_root must be a directory")

        def validate_schema(document: Any, schema_relative_path: str, label: str) -> None:
            schema = _json_bytes(
                _read_rooted_source_file(source_dir, schema_relative_path, limit=MAX_SCHEMA_BYTES),
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
        elif manifest_version == "loslab.g3_joint_evidence_run_manifest.v4":
            manifest_schema_path = SCHEMA_V4_RELATIVE_PATH
            expected_source_paths = V4_EVALUATOR_SOURCE_PATHS
        else:
            raise ValueError("unsupported G3 run manifest schema")
        if require_signed_reviewer_admission and manifest_version != (
            "loslab.g3_joint_evidence_run_manifest.v4"
        ):
            raise ValueError(
                "strict signed reviewer admission requires v4 signed runtime inventory provenance"
            )
        validate_schema(manifest, manifest_schema_path, "G3 run manifest")
        if manifest_version in (
            "loslab.g3_joint_evidence_run_manifest.v3",
            "loslab.g3_joint_evidence_run_manifest.v4",
        ):
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
        if require_signed_reviewer_admission:
            if not isinstance(qualification_contract, dict):
                raise ValueError("strict signed reviewer admission requires a frozen G3 contract")
            if qualification_contract.get("schema") != ("loslab.g3_qualification_run_contract.v4"):
                raise ValueError(
                    "strict signed reviewer admission requires qualification contract v4 "
                    "with raw path criteria and a complete powered design"
                )
        qualification_contract_sha256: str | None = None
        if qualification_contract is not None:
            if not isinstance(qualification_contract, dict):
                raise ValueError("qualification contract must be an object")
            if qualification_contract["run_id"] != manifest["run_id"]:
                raise ValueError("qualification contract run_id differs from run receipt")
            qualification_schema_path = (
                QUALIFICATION_CONTRACT_SCHEMA_V4_RELATIVE_PATH
                if qualification_contract.get("schema") == "loslab.g3_qualification_run_contract.v4"
                else QUALIFICATION_CONTRACT_SCHEMA_RELATIVE_PATH
                if qualification_contract.get("schema") == "loslab.g3_qualification_run_contract.v3"
                else QUALIFICATION_CONTRACT_SCHEMA_V2_RELATIVE_PATH
            )
            validate_schema(
                qualification_contract,
                qualification_schema_path,
                "G3 qualification contract",
            )
            qualification_contract_sha256 = verify_qualification_run_contract(
                qualification_contract
            )["contract_sha256"]
        if not manifest_raw.endswith(b"\n") or manifest_raw != (
            json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode("utf-8"):
            raise ValueError("G3 run manifest must use canonical compact JSON and one newline")

        root = Path(package_root) if package_root is not None else manifest_file.parent
        root = root.resolve(strict=True)
        if not root.is_dir() or not hasattr(os, "O_DIRECTORY") or os.open not in os.supports_dir_fd:
            raise ValueError("package root must be a directory with safe dir_fd support")
        root_fd = os.open(
            root, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
        )
        captured: dict[str, bytes] = {}
        total = 0
        seen_paths: set[tuple[str, ...]] = set()
        try:
            capture_fields = [
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
            ]
            if manifest_version == "loslab.g3_joint_evidence_run_manifest.v4":
                capture_fields.append(
                    (
                        "runtime_inventory",
                        "runtime_inventory",
                        min(MAX_JSON_ARTIFACT_BYTES, _MAX_CAPTURE_BYTES),
                    )
                )
            for field, expected_role, limit in capture_fields:
                entry = manifest[field]
                parts = _safe_relative_path(entry["path"])
                if parts in seen_paths:
                    raise ValueError("manifest file paths must be distinct")
                seen_paths.add(parts)
                total_limit = (
                    _MAX_V4_MANIFEST_TOTAL_BYTES
                    if manifest_version == "loslab.g3_joint_evidence_run_manifest.v4"
                    else _MAX_MANIFEST_TOTAL_BYTES
                )
                remaining = total_limit - total
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
        sorted_expected_source_paths = sorted(expected_source_paths)
        if [entry["path"] for entry in sources] != sorted_expected_source_paths:
            raise ValueError(
                "evaluator source roster differs from the selected manifest-version roster"
            )
        if manifest["evaluator_source_bundle_sha256"] != _source_bundle_sha256(sources):
            raise ValueError("evaluator source bundle digest does not match its file roster")
        if qualification_contract is not None and qualification_contract["source_identity"][
            "source_bundle_sha256"
        ] != _source_bundle_sha256(
            [item for item in sources if item["path"] in EVALUATOR_SOURCE_PATHS]
        ):
            raise ValueError(
                "qualification contract source bundle differs from run receipt sources"
            )
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
                if digest != entry["sha256"]:
                    # A rebound digest is accepted; the recorded size belongs to the
                    # superseded bytes and is not compared.
                    if not recorded_digest_resolves(entry["sha256"], digest, _REPOSITORY_ROOT):
                        raise ValueError(
                            f"evaluator source bytes differ from manifest: {entry['path']}"
                        )
                elif size != entry["size_bytes"]:
                    raise ValueError(
                        f"evaluator source bytes differ from manifest: {entry['path']}"
                    )
        finally:
            os.close(source_fd)

        input_payload = _json_bytes(captured["input"], "G3 joint evaluator input")
        if not isinstance(input_payload, dict):
            raise ValueError("G3 joint evaluator input must be an object")
        _validate_joint_evaluator_input_schema(input_payload, source_root=source_dir)
        artifact_schema_path = (
            ARTIFACT_SCHEMA_RELATIVE_PATH
            if manifest_version
            in (
                "loslab.g3_joint_evidence_run_manifest.v3",
                "loslab.g3_joint_evidence_run_manifest.v4",
            )
            else ARTIFACT_SCHEMA_V1_RELATIVE_PATH
        )
        validate_schema(
            _json_bytes(captured["artifact"], "G3 joint evaluator artifact"),
            artifact_schema_path,
            "G3 joint evaluator artifact",
        )
        if manifest_version == "loslab.g3_joint_evidence_run_manifest.v4":
            runtime_inventory_payload = _json_bytes(
                captured["runtime_inventory"], "G3 runtime inventory"
            )
            validate_schema(
                runtime_inventory_payload,
                RUNTIME_INVENTORY_SCHEMA_V3_RELATIVE_PATH,
                "G3 runtime inventory",
            )
        if manifest_version == "loslab.g3_joint_evidence_run_manifest.v4":
            if runtime_inventory_signature_path is not None:
                assert trusted_runtime_inventory_signers_path is not None
                assert runtime_inventory_signer_id is not None
                runtime_signature_identity = verify_openssh_signature_for_bytes(
                    captured["runtime_inventory"],
                    runtime_inventory_signature_path,
                    trusted_runtime_inventory_signers_path,
                    signer_id=runtime_inventory_signer_id,
                    namespace=RUNTIME_INVENTORY_SIGNATURE_NAMESPACE,
                )
                report.update(
                    {
                        "runtime_inventory_signature_status": "verified",
                        "runtime_inventory_signer_id": runtime_signature_identity["signer_id"],
                        "runtime_inventory_signature_sha256": runtime_signature_identity[
                            "signature_sha256"
                        ],
                        "trusted_runtime_inventory_signers_sha256": runtime_signature_identity[
                            "allowed_signers_sha256"
                        ],
                    }
                )
                if signature_identity is not None:
                    if runtime_signature_identity["signer_id"] == signature_identity["signer_id"]:
                        raise ValueError(
                            "runtime inventory signer must differ from run-manifest signer"
                        )
                    if (
                        runtime_signature_identity["allowed_signers_sha256"]
                        == signature_identity["allowed_signers_sha256"]
                    ):
                        raise ValueError(
                            "runtime inventory trust keyring must differ from run-manifest keyring"
                        )
                if runtime_signature_identity["signer_id"] == report.get("signer_policy_signer_id"):
                    raise ValueError(
                        "runtime inventory signer must differ from signer-policy custodian"
                    )
                if runtime_signature_identity["allowed_signers_sha256"] == report.get(
                    "signer_policy_allowed_signers_sha256"
                ):
                    raise ValueError(
                        "runtime inventory trust keyring must differ from signer-policy keyring"
                    )
                _verify_signed_runtime_inventory_bindings(
                    runtime_inventory_payload,
                    manifest["runtime_identity"],
                    sources,
                )
                report.update(
                    {
                        "runtime_inventory_sha256": hashlib.sha256(
                            captured["runtime_inventory"]
                        ).hexdigest(),
                        "runtime_inventory_binding_status": "verified",
                    }
                )
            elif require_signed_reviewer_admission:
                raise ValueError(
                    "strict signed reviewer admission requires signed runtime inventory"
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
            and qualification_contract.get("schema") == "loslab.g3_qualification_run_contract.v4"
            else None
        )
        replay = verify_joint_evidence_bytes(
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
                "evaluator_source_bundle_sha256": manifest["evaluator_source_bundle_sha256"],
                "qualification_contract_sha256": qualification_contract_sha256,
                "qualification_contract_bound": qualification_contract is not None,
            }
        )
    except (ImportError, OSError, TypeError, ValueError, RuntimeError, KeyError) as error:
        errors.append(str(error))
    _validate_run_verification_report(report)
    return report


def verify_joint_evidence_run_report_signature(
    report_path: str | Path,
    signature_path: str | Path,
    trusted_allowed_signers_path: str | Path,
    *,
    signer_id: str,
) -> dict[str, Any]:
    """Verify an external detached signature over exact bytes of a saved run report.

    The signature authenticates the report bytes and principal under a dedicated
    namespace. It does not establish signer-key custody, truth of the report, or
    scientific qualification.
    """
    report_raw = _read_bounded_regular_file(
        Path(report_path), limit=MAX_PACKAGE_BYTES, description="G3 run verification report"
    )
    report = _json_bytes(report_raw, "G3 run verification report")
    if not isinstance(report, dict):
        raise ValueError("G3 run verification report root must be an object")
    _validate_run_verification_report(report)
    identity = verify_openssh_signature_for_bytes(
        report_raw,
        signature_path,
        trusted_allowed_signers_path,
        signer_id=signer_id,
        namespace=RUN_VERIFICATION_REPORT_SIGNATURE_NAMESPACE,
    )
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "datasets/schema/run-verification-signature-receipt.v1.schema.json"
    )
    schema_bytes = _read_bounded_regular_file(
        schema_path, limit=MAX_SCHEMA_BYTES, description="G3 report signature receipt schema"
    )
    receipt: dict[str, Any] = {
        "schema": "loslab.g3_run_verification_signature_receipt.v1",
        "status": "SIGNATURE_VERIFIED",
        "qualification_status": "NOT_QUALIFIED",
        "run_id": report["run_id"],
        "signed_report_sha256": hashlib.sha256(report_raw).hexdigest(),
        "report_schema_sha256": report["report_schema_sha256"],
        "signature_namespace": identity["namespace"],
        "signer_id": identity["signer_id"],
        "signature_sha256": identity["signature_sha256"],
        "trusted_allowed_signers_sha256": identity["allowed_signers_sha256"],
        "receipt_schema_sha256": hashlib.sha256(schema_bytes).hexdigest(),
        "limitations": [
            "The detached signature authenticates the exact report bytes and signer principal, not the truth of the report.",
            "Allowed-signers key custody, reviewer independence, runtime truth, measurement provenance, and scientific qualification remain external.",
        ],
    }
    import jsonschema

    schema = _json_bytes(schema_bytes, "G3 report signature receipt schema")
    jsonschema.Draft202012Validator.check_schema(schema)
    errors = sorted(
        jsonschema.Draft202012Validator(schema).iter_errors(receipt),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        raise ValueError("G3 report signature receipt violates schema: " + errors[0].message)
    return receipt
