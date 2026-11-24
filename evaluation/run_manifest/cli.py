"""Command-line entry point for joint-evidence run manifests."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from evaluation.acquisition_package import (
    MAX_JSON_ARTIFACT_BYTES,
    _json_bytes,
    _read_bounded_regular_file,
)
from evaluation.joint_evidence_artifact import (
    _write_new_artifact,
)
from evaluation.run_manifest.build import (
    build_joint_evidence_run_manifest,
)
from evaluation.run_manifest.common import (
    _canonical_json_bytes,
)
from evaluation.run_manifest.sources import (
    evaluator_source_bundle_identity,
)
from evaluation.run_manifest.verify import (
    verify_joint_evidence_run_manifest,
)


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
    parser.add_argument("--output", type=Path, help="new manifest destination (must not exist)")
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
    parser.add_argument(
        "--runtime-inventory",
        type=Path,
        help="optional canonical runtime-inventory v3 JSON to bind into a v4 run receipt",
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
        "--runtime-inventory-signature",
        type=Path,
        help="detached signature over exact runtime-inventory bytes",
    )
    parser.add_argument(
        "--trusted-runtime-inventory-allowed-signers",
        type=Path,
        help="independent OpenSSH allowed_signers file for runtime inventory",
    )
    parser.add_argument(
        "--runtime-inventory-signer-id",
        help="OpenSSH principal expected to sign the runtime inventory",
    )
    parser.add_argument(
        "--require-signed-reviewer-admission",
        action="store_true",
        help=(
            "require a signed v4 run receipt, independent signed runtime inventory, "
            "externally signed reviewer-role policy, and frozen G3 contract before replay"
        ),
    )
    parser.add_argument(
        "--print-source-bundle",
        action="store_true",
        help="print the fixed evaluator source inventory for preregistering a run",
    )
    parser.add_argument(
        "--include-runtime-inventory-sources",
        action="store_true",
        help="include the v4 runtime-inventory verifier and runtime capture sources",
    )
    args = parser.parse_args(argv)
    if args.print_source_bundle:
        if (
            any(
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
                    args.runtime_inventory,
                    args.runtime_inventory_signature,
                    args.trusted_runtime_inventory_allowed_signers,
                    args.runtime_inventory_signer_id,
                )
            )
            or args.require_signed_reviewer_admission
        ):
            parser.error("--print-source-bundle cannot be combined with run-manifest arguments")
        try:
            print(
                json.dumps(
                    evaluator_source_bundle_identity(
                        source_root=args.source_root,
                        include_runtime_inventory=args.include_runtime_inventory_sources,
                    )
                )
            )
        except (OSError, TypeError, ValueError, RuntimeError) as error:
            print(f"G3 source inventory error: {error}", file=sys.stderr)
            return 2
        return 0
    if args.include_runtime_inventory_sources:
        parser.error("--include-runtime-inventory-sources requires --print-source-bundle")
    if args.runtime_inventory is not None and args.create_run_id is None:
        parser.error("--runtime-inventory requires --create-run-id")
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
            args.runtime_inventory_signature,
            args.trusted_runtime_inventory_allowed_signers,
            args.runtime_inventory_signer_id,
        )
    ):
        parser.error(
            "manifest signature options are only valid when verifying an existing manifest"
        )
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
            value = build_joint_evidence_run_manifest(
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
                runtime_inventory_path=args.runtime_inventory,
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
    report = verify_joint_evidence_run_manifest(
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
        runtime_inventory_signature_path=args.runtime_inventory_signature,
        trusted_runtime_inventory_signers_path=(args.trusted_runtime_inventory_allowed_signers),
        runtime_inventory_signer_id=args.runtime_inventory_signer_id,
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
