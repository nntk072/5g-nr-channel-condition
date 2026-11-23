"""Verify file-backed, replayable G3 numerical evidence runs."""

from __future__ import annotations

import sys

from evaluation.acquisition_package import MAX_SCHEMA_BYTES  # noqa: F401
from evaluation.qualification_contract import verify_qualification_run_contract  # noqa: F401
from evaluation.run_manifest.build import (
    build_joint_evidence_run_manifest,
)
from evaluation.run_manifest.cli import (
    main,
)
from evaluation.run_manifest.common import (
    _canonical_json_bytes,
    _runtime_identity,
)
from evaluation.run_manifest.constants import (  # noqa: F401
    _MAX_CAPTURE_BYTES,
    _MAX_MANIFEST_TOTAL_BYTES,
    _MAX_SOURCE_BUNDLE_BYTES,
    _MAX_V4_MANIFEST_TOTAL_BYTES,
    _RUN_ID_PATTERN,
    ARTIFACT_SCHEMA_RELATIVE_PATH,
    ARTIFACT_SCHEMA_V1_RELATIVE_PATH,
    EVALUATOR_SOURCE_PATHS,
    INPUT_SCHEMA_RELATIVE_PATH,
    INPUT_SCHEMA_V2_RELATIVE_PATH,
    INPUT_SCHEMA_V3_RELATIVE_PATH,
    INPUT_SCHEMA_V4_RELATIVE_PATH,
    INPUT_SCHEMA_V5_RELATIVE_PATH,
    INPUT_SCHEMA_V6_RELATIVE_PATH,
    NEW_V5_SOURCE_PATHS,
    PATH_DIAGNOSTIC_SCHEMA_V2_RELATIVE_PATH,
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
    SCHEMA_V5_RELATIVE_PATH,
    SIGNER_POLICY_PURPOSE,
    SIGNER_POLICY_RELATIVE_PATH,
    SIGNER_POLICY_SCHEMA,
    SIGNER_POLICY_SIGNATURE_NAMESPACE,
    V1_EVALUATOR_SOURCE_PATHS,
    V1_SCHEMA_RELATIVE_PATH,
    V2_EVALUATOR_SOURCE_PATHS,
    V4_EVALUATOR_SOURCE_PATHS,
    V5_EVALUATOR_SOURCE_PATHS,
    VERIFICATION_REPORT_SCHEMA_RELATIVE_PATH,
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
    _rooted_file_identity,
    _source_bundle_sha256,
    evaluator_source_bundle_identity,
)
from evaluation.run_manifest.verify import (
    verify_g3_joint_evidence_run_manifest,
    verify_joint_evidence_run_report_signature,
)

__all__ = [
    "NEW_V5_SOURCE_PATHS",
    "SCHEMA_V5_RELATIVE_PATH",
    "V5_EVALUATOR_SOURCE_PATHS",
    "MAX_SCHEMA_BYTES",
    "verify_qualification_run_contract",
    "ARTIFACT_SCHEMA_RELATIVE_PATH",
    "ARTIFACT_SCHEMA_V1_RELATIVE_PATH",
    "EVALUATOR_SOURCE_PATHS",
    "INPUT_SCHEMA_RELATIVE_PATH",
    "INPUT_SCHEMA_V2_RELATIVE_PATH",
    "INPUT_SCHEMA_V3_RELATIVE_PATH",
    "INPUT_SCHEMA_V4_RELATIVE_PATH",
    "INPUT_SCHEMA_V5_RELATIVE_PATH",
    "INPUT_SCHEMA_V6_RELATIVE_PATH",
    "PATH_DIAGNOSTIC_SCHEMA_V2_RELATIVE_PATH",
    "QUALIFICATION_CONTRACT_SCHEMA_RELATIVE_PATH",
    "QUALIFICATION_CONTRACT_SCHEMA_V2_RELATIVE_PATH",
    "QUALIFICATION_CONTRACT_SCHEMA_V4_RELATIVE_PATH",
    "RUNTIME_INVENTORY_SCHEMA_V3_RELATIVE_PATH",
    "RUNTIME_INVENTORY_SIGNATURE_NAMESPACE",
    "RUN_MANIFEST_SIGNATURE_NAMESPACE",
    "RUN_VERIFICATION_REPORT_SIGNATURE_NAMESPACE",
    "SCHEMA_RELATIVE_PATH",
    "SCHEMA_V3_RELATIVE_PATH",
    "SCHEMA_V4_RELATIVE_PATH",
    "SIGNER_POLICY_PURPOSE",
    "SIGNER_POLICY_RELATIVE_PATH",
    "SIGNER_POLICY_SCHEMA",
    "SIGNER_POLICY_SIGNATURE_NAMESPACE",
    "V1_EVALUATOR_SOURCE_PATHS",
    "V1_SCHEMA_RELATIVE_PATH",
    "V2_EVALUATOR_SOURCE_PATHS",
    "V4_EVALUATOR_SOURCE_PATHS",
    "VERIFICATION_REPORT_SCHEMA_RELATIVE_PATH",
    "_MAX_CAPTURE_BYTES",
    "_MAX_MANIFEST_TOTAL_BYTES",
    "_MAX_SOURCE_BUNDLE_BYTES",
    "_MAX_V4_MANIFEST_TOTAL_BYTES",
    "_RUN_ID_PATTERN",
    "_authorize_run_manifest_signer",
    "_canonical_json_bytes",
    "_read_rooted_source_file",
    "_require_independent_signer_policy_custody",
    "_rooted_file_identity",
    "_runtime_identity",
    "_source_bundle_sha256",
    "_validate_joint_evaluator_input_schema",
    "_validate_run_verification_report",
    "_verify_qualification_contract_bindings",
    "_verify_signed_runtime_inventory_bindings",
    "build_joint_evidence_run_manifest",
    "evaluator_source_bundle_identity",
    "main",
    "verify_g3_joint_evidence_run_manifest",
    "verify_joint_evidence_run_report_signature",
]

if __name__ == "__main__":
    sys.exit(main())
