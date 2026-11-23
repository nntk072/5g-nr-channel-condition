"""Schema paths, signature namespaces, byte limits and evaluator source rosters for joint-evidence run manifests."""

from __future__ import annotations

import re

from evaluation.acquisition_package import (
    MAX_JSON_ARTIFACT_BYTES,
)

V1_SCHEMA_RELATIVE_PATH = "datasets/schema/joint-evidence-run-manifest.v1.schema.json"


SCHEMA_RELATIVE_PATH = "datasets/schema/joint-evidence-run-manifest.v2.schema.json"


SCHEMA_V3_RELATIVE_PATH = "datasets/schema/joint-evidence-run-manifest.v3.schema.json"


SCHEMA_V4_RELATIVE_PATH = "datasets/schema/joint-evidence-run-manifest.v4.schema.json"


RUNTIME_INVENTORY_SCHEMA_V3_RELATIVE_PATH = "datasets/schema/runtime-inventory.v3.schema.json"


RUNTIME_INVENTORY_SIGNATURE_NAMESPACE = "loslab-g3-runtime-inventory-v1@github.com"


SIGNER_POLICY_SCHEMA = "loslab.g3_run_manifest_signer_policy.v1"


SIGNER_POLICY_PURPOSE = "g3_joint_run_manifest_verification"


SIGNER_POLICY_RELATIVE_PATH = "datasets/schema/run-manifest-signer-policy.v1.schema.json"


VERIFICATION_REPORT_SCHEMA_RELATIVE_PATH = (
    "datasets/schema/joint-evidence-run-verification.v1.schema.json"
)


INPUT_SCHEMA_RELATIVE_PATH = "datasets/schema/joint-evaluator-input.v1.schema.json"


INPUT_SCHEMA_V2_RELATIVE_PATH = "datasets/schema/joint-evaluator-input.v2.schema.json"


INPUT_SCHEMA_V3_RELATIVE_PATH = "datasets/schema/joint-evaluator-input.v3.schema.json"


INPUT_SCHEMA_V4_RELATIVE_PATH = "datasets/schema/joint-evaluator-input.v4.schema.json"


INPUT_SCHEMA_V5_RELATIVE_PATH = "datasets/schema/joint-evaluator-input.v5.schema.json"


INPUT_SCHEMA_V6_RELATIVE_PATH = "datasets/schema/joint-evaluator-input.v6.schema.json"


PATH_DIAGNOSTIC_SCHEMA_V2_RELATIVE_PATH = (
    "datasets/schema/path-equivalence-roster-diagnostic.v2.schema.json"
)


RUN_MANIFEST_SIGNATURE_NAMESPACE = "loslab-g3-joint-run-manifest-v1@github.com"


SIGNER_POLICY_SIGNATURE_NAMESPACE = "loslab-g3-run-manifest-policy-v1@github.com"


RUN_VERIFICATION_REPORT_SIGNATURE_NAMESPACE = (
    "loslab-g3-joint-run-verification-report-v1@github.com"
)


ARTIFACT_SCHEMA_V1_RELATIVE_PATH = "datasets/schema/joint-evaluator-artifact.v1.schema.json"


ARTIFACT_SCHEMA_RELATIVE_PATH = "datasets/schema/joint-evaluator-artifact.v2.schema.json"


QUALIFICATION_CONTRACT_SCHEMA_RELATIVE_PATH = (
    "datasets/schema/qualification-run-contract.v3.schema.json"
)


QUALIFICATION_CONTRACT_SCHEMA_V2_RELATIVE_PATH = (
    "datasets/schema/qualification-run-contract.v2.schema.json"
)


QUALIFICATION_CONTRACT_SCHEMA_V4_RELATIVE_PATH = (
    "datasets/schema/qualification-run-contract.v4.schema.json"
)


_MAX_CAPTURE_BYTES = MAX_JSON_ARTIFACT_BYTES


_MAX_MANIFEST_TOTAL_BYTES = _MAX_CAPTURE_BYTES * 2


_MAX_V4_MANIFEST_TOTAL_BYTES = _MAX_CAPTURE_BYTES * 3


_MAX_SOURCE_BUNDLE_BYTES = 64 * 1024 * 1024


V1_EVALUATOR_SOURCE_PATHS = tuple(
    sorted(
        (
            V1_SCHEMA_RELATIVE_PATH,
            ARTIFACT_SCHEMA_V1_RELATIVE_PATH,
            "datasets/schema/joint-evaluator-input.v1.schema.json",
            "datasets/schema/joint-evaluator-input.v2.schema.json",
            "datasets/schema/qualification-run-contract.v2.schema.json",
            SIGNER_POLICY_RELATIVE_PATH,
            "evaluation/acquisition_package.py",
            "evaluation/complex_csi.py",
            "evaluation/joint_acceptance.py",
            "evaluation/joint_evaluation.py",
            "evaluation/joint_evidence_artifact.py",
            "evaluation/joint_evidence_run_manifest.py",
            "evaluation/qualification_contract.py",
            "evaluation/material_calibration.py",
            "evaluation/numeric.py",
            "evaluation/path_equivalence_assurance.py",
            "evaluation/propagation_metrics.py",
            "evaluation/receiver_impact.py",
            "evaluation/signed_attestation.py",
        )
    )
)


V2_EVALUATOR_SOURCE_PATHS = tuple(
    sorted(
        (
            "pyproject.toml",
            "uv.lock",
            ARTIFACT_SCHEMA_V1_RELATIVE_PATH,
            "datasets/schema/joint-evaluator-input.v1.schema.json",
            "datasets/schema/joint-evaluator-input.v2.schema.json",
            "datasets/schema/joint-sample-size-design-composition.v1.schema.json",
            INPUT_SCHEMA_V3_RELATIVE_PATH,
            INPUT_SCHEMA_V4_RELATIVE_PATH,
            INPUT_SCHEMA_V5_RELATIVE_PATH,
            INPUT_SCHEMA_V6_RELATIVE_PATH,
            PATH_DIAGNOSTIC_SCHEMA_V2_RELATIVE_PATH,
            SCHEMA_RELATIVE_PATH,
            "datasets/schema/qualification-run-contract.v2.schema.json",
            QUALIFICATION_CONTRACT_SCHEMA_RELATIVE_PATH,
            QUALIFICATION_CONTRACT_SCHEMA_V4_RELATIVE_PATH,
            SIGNER_POLICY_RELATIVE_PATH,
            "evaluation/acquisition_package.py",
            "evaluation/complex_csi.py",
            "evaluation/joint_acceptance.py",
            "evaluation/joint_evaluation.py",
            "evaluation/joint_evidence_artifact.py",
            "evaluation/joint_evidence_run_manifest.py",
            "evaluation/qualification_contract.py",
            "evaluation/material_calibration.py",
            "evaluation/numeric.py",
            "evaluation/path_equivalence.py",
            "evaluation/path_equivalence_assurance.py",
            "evaluation/propagation_assurance.py",
            "evaluation/propagation_metrics.py",
            "evaluation/receiver_impact.py",
            "evaluation/receiver_assurance.py",
            "evaluation/sample_size_planning.py",
            "evaluation/signed_attestation.py",
        )
    )
)


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


V4_EVALUATOR_SOURCE_PATHS = tuple(
    sorted(
        (
            *EVALUATOR_SOURCE_PATHS,
            SCHEMA_V4_RELATIVE_PATH,
            RUNTIME_INVENTORY_SCHEMA_V3_RELATIVE_PATH,
            "integrations/runtime_inventory.py",
            "integrations/runtime_compatibility.py",
            "scripts/report_runtime_inventory.py",
        )
    )
)


_RUN_ID_PATTERN = re.compile(r"^[a-z][a-z0-9._-]{2,63}$")
