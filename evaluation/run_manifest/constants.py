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
SCHEMA_V5_RELATIVE_PATH = "datasets/schema/joint-evidence-run-manifest.v5.schema.json"


RUNTIME_INVENTORY_SCHEMA_V3_RELATIVE_PATH = "datasets/schema/runtime-inventory.v3.schema.json"


RUNTIME_INVENTORY_SIGNATURE_NAMESPACE = "loslab-runtime-inventory-v1@github.com"


SIGNER_POLICY_SCHEMA = "loslab.run_manifest_signer_policy.v1"


SIGNER_POLICY_PURPOSE = "joint_run_manifest_verification"


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


RUN_MANIFEST_SIGNATURE_NAMESPACE = "loslab-joint-run-manifest-v1@github.com"


SIGNER_POLICY_SIGNATURE_NAMESPACE = "loslab-run-manifest-policy-v1@github.com"


RUN_VERIFICATION_REPORT_SIGNATURE_NAMESPACE = "loslab-joint-run-verification-report-v1@github.com"


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
            "evaluation/acceptance/complex_csi.py",
            "evaluation/acceptance/joint_acceptance.py",
            "evaluation/acceptance/joint_evaluation.py",
            "evaluation/joint_evidence_artifact.py",
            "evaluation/joint_evidence_run_manifest.py",
            "evaluation/acceptance/qualification_contract.py",
            "evaluation/acceptance/material_calibration.py",
            "evaluation/scoring/numeric.py",
            "evaluation/acceptance/path_equivalence_assurance.py",
            "evaluation/acceptance/propagation_metrics.py",
            "evaluation/acceptance/receiver_impact.py",
            "evaluation/acceptance/signed_attestation.py",
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
            "evaluation/acceptance/complex_csi.py",
            "evaluation/acceptance/joint_acceptance.py",
            "evaluation/acceptance/joint_evaluation.py",
            "evaluation/joint_evidence_artifact.py",
            "evaluation/joint_evidence_run_manifest.py",
            "evaluation/acceptance/qualification_contract.py",
            "evaluation/acceptance/material_calibration.py",
            "evaluation/scoring/numeric.py",
            "evaluation/acceptance/path_equivalence.py",
            "evaluation/acceptance/path_equivalence_assurance.py",
            "evaluation/acceptance/propagation_assurance.py",
            "evaluation/acceptance/propagation_metrics.py",
            "evaluation/acceptance/receiver_impact.py",
            "evaluation/acceptance/receiver_assurance.py",
            "evaluation/sample_size_planning.py",
            "evaluation/acceptance/signed_attestation.py",
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


# Protocol v5 (owner-approved, 2026-11-23): the split submodules of the evaluator
# modules join the signed roster. The V4 roster and schema are unchanged.
NEW_V5_SOURCE_PATHS = (
    "evaluation/acquisition/__init__.py",
    "evaluation/acquisition/canonical.py",
    "evaluation/acquisition/constants.py",
    "evaluation/acquisition/files.py",
    "evaluation/acquisition/index.py",
    "evaluation/acquisition/noise.py",
    "evaluation/acquisition/verify.py",
    "evaluation/run_manifest/__init__.py",
    "evaluation/run_manifest/build.py",
    "evaluation/run_manifest/cli.py",
    "evaluation/run_manifest/common.py",
    "evaluation/run_manifest/constants.py",
    "evaluation/run_manifest/contracts.py",
    "evaluation/run_manifest/signatures.py",
    "evaluation/run_manifest/sources.py",
    "evaluation/run_manifest/verify.py",
    "evaluation/sample_size/__init__.py",
    "evaluation/sample_size/binomial.py",
    "evaluation/sample_size/bootstrap.py",
    "evaluation/sample_size/constants.py",
    "evaluation/sample_size/independent_units.py",
    "evaluation/sample_size/joint_design.py",
)
V5_EVALUATOR_SOURCE_PATHS = tuple(sorted((*V4_EVALUATOR_SOURCE_PATHS, *NEW_V5_SOURCE_PATHS)))


_RUN_ID_PATTERN = re.compile(r"^[a-z][a-z0-9._-]{2,63}$")
