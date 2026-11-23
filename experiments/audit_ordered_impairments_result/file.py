"""Command-line audit of one ordered-impairment result file."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from artifacts.rebinding import recorded_digest_matches, resolve_recorded_path

from .constants import (
    _MAX_RESULT_BYTES,
    _MAX_WAVEFORM_EVIDENCE_BYTES,
    REPOSITORY_ROOT,
)
from .evidence_files import (
    _read_bounded_evidence_file,
)
from .hashing import (
    _reject_constant,
    _unique_object,
)
from .independent import (
    _audit_result,
)
from .replay import (
    _replay_experiment,
)
from .snr_curve import (
    _audit_independent_snr_curve,
)


def audit_result_file(
    path: Path,
    *,
    committed_revision: str | None = None,
    deterministic_replay: bool = False,
    acceptance_signature_path: Path | None = None,
    trusted_allowed_signers_path: Path | None = None,
    reviewer_policy_path: Path | None = None,
    reviewer_policy_signature_path: Path | None = None,
    trusted_reviewer_allowed_signers_path: Path | None = None,
    reviewer_policy_signer_id: str | None = None,
) -> dict[str, Any]:
    if (acceptance_signature_path is None) != (trusted_allowed_signers_path is None):
        raise ValueError(
            "acceptance signature and trusted allowed-signers file must be supplied together"
        )
    reviewer_policy_options = (
        reviewer_policy_path,
        reviewer_policy_signature_path,
        trusted_reviewer_allowed_signers_path,
        reviewer_policy_signer_id,
    )
    if any(option is not None for option in reviewer_policy_options) and not all(
        option is not None for option in reviewer_policy_options
    ):
        raise ValueError(
            "G5 reviewer policy, signature, keyring, and custodian must be supplied together"
        )
    if (
        any(option is not None for option in reviewer_policy_options)
        and acceptance_signature_path is None
    ):
        raise ValueError("G5 reviewer-policy recheck requires acceptance signature recheck inputs")
    if path.is_symlink() or not path.is_file():
        raise ValueError("result must be a regular non-symlink file")
    raw = path.read_bytes()
    if len(raw) > _MAX_RESULT_BYTES:
        raise ValueError(f"result exceeds {_MAX_RESULT_BYTES} bytes")
    result = json.loads(
        raw.decode("utf-8"), parse_constant=_reject_constant, object_pairs_hook=_unique_object
    )
    if not isinstance(result, dict):
        raise ValueError("result root must be an object")
    source_name = result.get("provenance", {}).get("source_path")
    if (
        not isinstance(source_name, str)
        or Path(source_name).is_absolute()
        or ".." in Path(source_name).parts
    ):
        raise ValueError("provenance source_path must be a repository-relative path")
    source_path = resolve_recorded_path(source_name, REPOSITORY_ROOT)
    source_bytes = source_path.read_bytes()
    experiment = result.get("experiment")
    if isinstance(experiment, dict) and experiment.get("schema") in {
        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v1",
        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v2",
        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v3",
        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v4",
        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v5",
        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v6",
        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v7",
        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v8",
        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v9",
        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v10",
        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v11",
        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v12",
        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v13",
        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v14",
        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v15",
        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v16",
        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v17",
        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v25",
    }:
        if deterministic_replay:
            raise ValueError("deterministic replay is not available for diagnostic SNR curves")
        committed_source: bytes | None = None
        if committed_revision is not None:
            completed = subprocess.run(
                ["git", "-C", str(REPOSITORY_ROOT), "show", f"{committed_revision}:{source_name}"],
                check=True,
                capture_output=True,
            )
            committed_source = completed.stdout
        curve_config = experiment.get("config")
        evidence_manifest = (
            curve_config.get("waveform_evidence") if isinstance(curve_config, dict) else None
        )
        curve_waveform_evidence_bytes: bytes | None = None
        if evidence_manifest is not None:
            if not isinstance(evidence_manifest, dict):
                raise ValueError("SNR curve waveform_evidence manifest must be an object")
            evidence_name = evidence_manifest.get("file_name")
            if (
                not isinstance(evidence_name, str)
                or not evidence_name
                or Path(evidence_name).name != evidence_name
                or "\\" in evidence_name
                or not evidence_name.endswith(".npz")
                or evidence_name in {".", ".."}
            ):
                raise ValueError("SNR curve waveform evidence must be a sibling file name")
            curve_waveform_evidence_bytes = _read_bounded_evidence_file(
                path.parent / evidence_name,
                maximum_bytes=_MAX_WAVEFORM_EVIDENCE_BYTES,
            )
        return _audit_independent_snr_curve(
            result,
            source_bytes,
            committed_source,
            waveform_evidence_bytes=curve_waveform_evidence_bytes,
            acceptance_signature_path=acceptance_signature_path,
            trusted_allowed_signers_path=trusted_allowed_signers_path,
            reviewer_policy_path=reviewer_policy_path,
            reviewer_policy_signature_path=reviewer_policy_signature_path,
            trusted_reviewer_allowed_signers_path=trusted_reviewer_allowed_signers_path,
            reviewer_policy_signer_id=reviewer_policy_signer_id,
        )
    config = experiment.get("config") if isinstance(experiment, dict) else None
    evidence_manifest = config.get("waveform_evidence") if isinstance(config, dict) else None
    waveform_evidence_bytes: bytes | None = None
    if evidence_manifest is not None:
        if not isinstance(evidence_manifest, dict):
            raise ValueError("waveform_evidence manifest must be an object")
        evidence_name = evidence_manifest.get("file_name")
        if (
            not isinstance(evidence_name, str)
            or not evidence_name
            or Path(evidence_name).name != evidence_name
            or "\\" in evidence_name
            or not evidence_name.endswith(".npz")
            or evidence_name in {".", ".."}
        ):
            raise ValueError("waveform evidence must be a sibling file name")
        evidence_path = path.parent / evidence_name
        waveform_evidence_bytes = _read_bounded_evidence_file(
            evidence_path, maximum_bytes=_MAX_WAVEFORM_EVIDENCE_BYTES
        )
    estimator_source_bytes: bytes | None = None
    estimator_config = config.get("channel_estimator") if isinstance(config, dict) else None
    if isinstance(estimator_config, dict) and estimator_config.get("name") == "finite_delay_dft":
        estimator_relative_path = estimator_config.get("implementation_path")
        if estimator_relative_path != "experiments/cp_bounded_dft_estimator.py":
            raise ValueError("unsupported estimator implementation path")
        estimator_source_bytes = (REPOSITORY_ROOT / estimator_relative_path).read_bytes()
    replayed_experiment: dict[str, Any] | None = None
    if deterministic_replay:
        if not recorded_digest_matches(
            result.get("provenance", {}).get("source_sha256"), source_bytes, REPOSITORY_ROOT
        ):
            raise ValueError("deterministic replay requires a matching source-to-worktree digest")
        experiment = result.get("experiment")
        if not isinstance(experiment, dict):
            raise ValueError("result experiment must be an object for deterministic replay")
        replayed_experiment = _replay_experiment(experiment)
    committed_bytes: bytes | None = None
    committed_estimator_bytes: bytes | None = None
    if committed_revision is not None:
        completed = subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), "show", f"{committed_revision}:{source_name}"],
            check=True,
            capture_output=True,
        )
        committed_bytes = completed.stdout
        if estimator_source_bytes is not None:
            completed_estimator = subprocess.run(
                [
                    "git",
                    "-C",
                    str(REPOSITORY_ROOT),
                    "show",
                    f"{committed_revision}:experiments/cp_bounded_dft_estimator.py",
                ],
                check=True,
                capture_output=True,
            )
            committed_estimator_bytes = completed_estimator.stdout
    return _audit_result(
        result,
        source_bytes,
        committed_bytes,
        replayed_experiment=replayed_experiment,
        waveform_evidence_bytes=waveform_evidence_bytes,
        estimator_source_bytes=estimator_source_bytes,
        committed_estimator_source=committed_estimator_bytes,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument(
        "--committed-revision",
        help="optional Git revision whose source blob should match the artifact source digest",
    )
    parser.add_argument(
        "--deterministic-replay",
        action="store_true",
        help="rerun the exact bounded CPU seed roster as a reproducibility audit, not a new confirmation",
    )
    parser.add_argument(
        "--acceptance-signature",
        type=Path,
        help="detached signature for rechecking a signed acceptance contract (schema-v6 or newer)",
    )
    parser.add_argument(
        "--acceptance-allowed-signers",
        type=Path,
        help="trusted OpenSSH allowed-signers file for rechecking signed acceptance",
    )
    parser.add_argument("--reviewer-policy", type=Path)
    parser.add_argument("--reviewer-policy-signature", type=Path)
    parser.add_argument("--reviewer-policy-allowed-signers", type=Path)
    parser.add_argument("--reviewer-policy-principal")
    args = parser.parse_args(argv)
    if (args.acceptance_signature is None) != (args.acceptance_allowed_signers is None):
        parser.error("--acceptance-signature and --acceptance-allowed-signers require one another")
    reviewer_options = (
        args.reviewer_policy,
        args.reviewer_policy_signature,
        args.reviewer_policy_allowed_signers,
        args.reviewer_policy_principal,
    )
    if any(option is not None for option in reviewer_options) and not all(
        option is not None for option in reviewer_options
    ):
        parser.error(
            "reviewer-policy file, signature, allowed-signers, and principal require one another"
        )
    try:
        report = audit_result_file(
            args.result,
            committed_revision=args.committed_revision,
            deterministic_replay=args.deterministic_replay,
            acceptance_signature_path=args.acceptance_signature,
            trusted_allowed_signers_path=args.acceptance_allowed_signers,
            reviewer_policy_path=args.reviewer_policy,
            reviewer_policy_signature_path=args.reviewer_policy_signature,
            trusted_reviewer_allowed_signers_path=args.reviewer_policy_allowed_signers,
            reviewer_policy_signer_id=args.reviewer_policy_principal,
        )
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        RuntimeError,
        ValueError,
        TypeError,
    ) as error:
        parser.exit(2, f"error: {error}\n")
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return 0 if report["status"] != "AUDIT_FAILED" else 1
