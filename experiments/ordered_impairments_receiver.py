"""Public entry point for the ordered-impairments receiver checks."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from experiments.acceptance import (
    evaluate_snr_curve_acceptance as _evaluate_independent_snr_curve_acceptance,  # noqa: F401
)
from experiments.acceptance import (
    freeze_snr_acceptance_criteria as _freeze_independent_snr_acceptance_criteria,  # noqa: F401
)
from experiments.acceptance import (
    load_snr_acceptance_contract as _load_snr_acceptance_contract,  # noqa: F401
)
from experiments.acceptance import (
    snr_execution_context_from_config as _snr_execution_context_from_config,  # noqa: F401
)

__all__ = [
    "BLOCKS_PER_SEED",
    "CFO_HZ",
    "CP_GUARD_SAMPLES",
    "DIRECT_COEFFICIENT",
    "ECHO_COEFFICIENT",
    "ECHO_DELAY_SAMPLES",
    "INDEPENDENT_CHANNEL_BOOTSTRAP_DRAWS",
    "INDEPENDENT_CHANNEL_BOOTSTRAP_SEED",
    "INDEPENDENT_CHANNEL_SEEDS",
    "INDEPENDENT_SNR_CURVE_BOOTSTRAP_DRAWS",
    "INDEPENDENT_SNR_CURVE_BOOTSTRAP_SEED",
    "MAX_CFO_ERROR_HZ",
    "MAX_CFO_ERROR_P95_HZ",
    "MAX_IMPAIRED_BLER",
    "MAX_REFERENCE_BLER",
    "MAX_TIMING_OFFSET_SAMPLES",
    "NOISE_SEED_OFFSET",
    "NOISE_VARIANCE",
    "PAYLOAD_SEEDS",
    "RESULT_SCHEMA",
    "RX_IQ_GAIN_IMBALANCE_DB",
    "RX_IQ_PHASE_IMBALANCE_RAD",
    "RX_LINEWIDTH_HZ",
    "RX_PHASE_SEED_OFFSET",
    "SCHEMA",
    "TIMING_OFFSET_SAMPLES",
    "TX_IQ_GAIN_IMBALANCE_DB",
    "TX_IQ_PHASE_IMBALANCE_RAD",
    "TX_LINEWIDTH_HZ",
    "TX_PHASE_SEED_OFFSET",
    "_BoundedBytesIO",
    "_LEGACY_RESULT_SCHEMA",
    "_MAX_INDEPENDENT_SNR_POINTS",
    "_MAX_INDEPENDENT_TRANSPORT_BLOCKS",
    "_MAX_WAVEFORM_EVIDENCE_BYTES",
    "_MAX_WAVEFORM_EVIDENCE_UNCOMPRESSED_BYTES",
    "_MAX_WAVEFORM_NPZ_MEMBERS",
    "_MAX_WAVEFORM_NPZ_MEMBER_NAME_BYTES",
    "_canonical_json_bytes",
    "_channel_any_failure_upper_bound",
    "_cluster_bootstrap_interval",
    "_combine_snr_curve_waveform_evidence",
    "_configure_probe_decoder",
    "_git_revision",
    "_independent_report_schema",
    "_independent_snr_curve_cluster_inference",
    "_known_chain",
    "_phase_noise",
    "_pool_repeated_waveforms_for_shared_sync",
    "_proper_complex_noise",
    "_run_independent_ordered_impairment_probe",
    "_run_ordered_impairment_probe",
    "_sha256",
    "_snr_execution_context",
    "_source_matches_git_revision",
    "_standard_complex_noise_draws",
    "_validate_execution_device",
    "_write_bounded_npz",
    "build_hash_bound_result",
    "main",
    "run_independent_ordered_impairment_probe",
    "run_independent_ordered_impairment_snr_curve",
    "run_ordered_impairment_probe",
    "verify_hash_bound_result",
    "write_hash_bound_result",
]

if TYPE_CHECKING:
    import torch
else:

    class _LazyTorch:
        def __getattr__(self, name: str) -> Any:
            module = importlib.import_module("torch")
            globals()["torch"] = module
            return getattr(module, name)

    torch = _LazyTorch()

from experiments.ordered_impairments.constants import (
    _LEGACY_RESULT_SCHEMA,
    _MAX_INDEPENDENT_SNR_POINTS,
    _MAX_INDEPENDENT_TRANSPORT_BLOCKS,
    _MAX_WAVEFORM_EVIDENCE_BYTES,
    _MAX_WAVEFORM_EVIDENCE_UNCOMPRESSED_BYTES,
    _MAX_WAVEFORM_NPZ_MEMBER_NAME_BYTES,
    _MAX_WAVEFORM_NPZ_MEMBERS,
    BLOCKS_PER_SEED,
    CFO_HZ,
    CP_GUARD_SAMPLES,
    DIRECT_COEFFICIENT,
    ECHO_COEFFICIENT,
    ECHO_DELAY_SAMPLES,
    INDEPENDENT_CHANNEL_BOOTSTRAP_DRAWS,
    INDEPENDENT_CHANNEL_BOOTSTRAP_SEED,
    INDEPENDENT_CHANNEL_SEEDS,
    INDEPENDENT_SNR_CURVE_BOOTSTRAP_DRAWS,
    INDEPENDENT_SNR_CURVE_BOOTSTRAP_SEED,
    MAX_CFO_ERROR_HZ,
    MAX_CFO_ERROR_P95_HZ,
    MAX_IMPAIRED_BLER,
    MAX_REFERENCE_BLER,
    MAX_TIMING_OFFSET_SAMPLES,
    NOISE_SEED_OFFSET,
    NOISE_VARIANCE,
    PAYLOAD_SEEDS,
    RESULT_SCHEMA,
    RX_IQ_GAIN_IMBALANCE_DB,
    RX_IQ_PHASE_IMBALANCE_RAD,
    RX_LINEWIDTH_HZ,
    RX_PHASE_SEED_OFFSET,
    SCHEMA,
    TIMING_OFFSET_SAMPLES,
    TX_IQ_GAIN_IMBALANCE_DB,
    TX_IQ_PHASE_IMBALANCE_RAD,
    TX_LINEWIDTH_HZ,
    TX_PHASE_SEED_OFFSET,
)
from experiments.ordered_impairments.hashing import (
    _canonical_json_bytes,
    _sha256,
)
from experiments.ordered_impairments.independent_probe import (
    _independent_report_schema,
    _run_independent_ordered_impairment_probe,
    _validate_execution_device,
    run_independent_ordered_impairment_probe,
)
from experiments.ordered_impairments.independent_snr_curve import (
    _independent_snr_curve_cluster_inference,
    _snr_execution_context,
    run_independent_ordered_impairment_snr_curve,
)
from experiments.ordered_impairments.paired_probe import (
    _configure_probe_decoder,
    _run_ordered_impairment_probe,
    run_ordered_impairment_probe,
)
from experiments.ordered_impairments.signal_models import (
    _known_chain,
    _phase_noise,
    _proper_complex_noise,
    _standard_complex_noise_draws,
)
from experiments.ordered_impairments.statistics import (
    _channel_any_failure_upper_bound,
    _cluster_bootstrap_interval,
    _pool_repeated_waveforms_for_shared_sync,
)
from experiments.ordered_impairments.waveform_evidence import (
    _BoundedBytesIO,
    _combine_snr_curve_waveform_evidence,
    _write_bounded_npz,
)


def _git_revision(repo_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _source_matches_git_revision(repo_root: Path, source_name: str, source: bytes) -> bool | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "show", f"HEAD:{source_name}"],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return source == completed.stdout


def build_hash_bound_result(
    report: dict[str, Any], *, created_utc: str | None = None
) -> dict[str, Any]:
    """Bind a probe report to its exact source, protocol, and runtime provenance."""
    source_path = Path(__file__).resolve()
    repo_root = source_path.parent.parent
    protocol = {
        "experiment_schema": report["schema"],
        "config": report["config"],
        "acceptance_criteria": report["acceptance"]["criteria_frozen_before_run"],
    }
    source_bytes = source_path.read_bytes()
    created = created_utc or datetime.now(UTC).isoformat()
    command = [sys.executable, *sys.argv]
    source_digest = _sha256(source_bytes)
    configuration_digest = _sha256(_canonical_json_bytes(report["config"]))
    try:
        source_name = source_path.relative_to(repo_root).as_posix()
    except ValueError:
        source_name = source_path.name
    packages: dict[str, str | None] = {}
    for distribution in ("torch", "sionna-no-rt", "sionna", "sionna-phy", "sionna-rt"):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None

    bound: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "command": command,
        "created_utc": created,
        "source_sha256": source_digest,
        "configuration_sha256": configuration_digest,
        "interpretation": {
            "criterion_scope": (
                "Descriptive multi-point curve; the single-point G5.09 criteria are not "
                "applied across this SNR family."
                if report.get("acceptance", {}).get("criteria_frozen_before_run", {}).get("status")
                == "descriptive_only"
                else "This verdict applies only to the declared deterministic synthetic CPU fixture."
                if report.get("config", {}).get("execution_device", "cpu") == "cpu"
                else "This verdict applies only to the declared deterministic synthetic fixture and execution device."
            ),
            "deployment_qualification": False,
            "g5_09_closed": False,
            "status": (
                "diagnostic_only"
                if report.get("acceptance", {}).get("criteria_frozen_before_run", {}).get("status")
                == "descriptive_only"
                else "conditional_fixture_not_evaluable"
                if report.get("acceptance", {}).get("status") == "NOT_EVALUABLE"
                else "conditional_fixture_pass"
                if report["acceptance"]["pass"]
                else "conditional_fixture_fail"
            ),
        },
        "experiment": report,
        "provenance": {
            "created_utc": created,
            "command": command,
            "git_revision": _git_revision(repo_root),
            "source_path": source_name,
            "source_sha256": source_digest,
            "source_matches_git_revision": _source_matches_git_revision(
                repo_root, source_name, source_bytes
            ),
            "protocol": protocol,
            "protocol_sha256": _sha256(_canonical_json_bytes(protocol)),
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "packages": packages,
                "torch_device": report.get("config", {}).get("execution_device", "cpu"),
            },
        },
    }
    bound["result_sha256"] = _sha256(_canonical_json_bytes(bound))
    return bound


def verify_hash_bound_result(result: dict[str, Any]) -> bool:
    """Check internal result/config digests without asserting scientific validity."""
    if result.get("schema") not in (RESULT_SCHEMA, _LEGACY_RESULT_SCHEMA):
        return False
    provenance = result.get("provenance")
    experiment = result.get("experiment")
    if not isinstance(provenance, dict) or not isinstance(experiment, dict):
        return False
    if result.get("schema") == RESULT_SCHEMA and (
        not isinstance(provenance.get("command"), list)
        or not provenance["command"]
        or any(not isinstance(item, str) or not item for item in provenance["command"])
        or result.get("command") != provenance.get("command")
        or result.get("created_utc") != provenance.get("created_utc")
        or result.get("source_sha256") != provenance.get("source_sha256")
        or result.get("configuration_sha256")
        != _sha256(_canonical_json_bytes(experiment.get("config")))
    ):
        return False
    if experiment.get("schema") in {
        f"umcd.g5_ordered_impairments_independent_channels.v{version}" for version in range(21, 25)
    }:
        config = experiment.get("config")
        receiver = config.get("receiver_configuration") if isinstance(config, dict) else None
        runtime = provenance.get("runtime")
        device = config.get("execution_device") if isinstance(config, dict) else None
        if (
            not isinstance(device, str)
            or not device.startswith("cuda:")
            or not isinstance(receiver, dict)
            or receiver.get("device") != device
            or not isinstance(runtime, dict)
            or runtime.get("torch_device") != device
        ):
            return False
    protocol = provenance.get("protocol")
    if not isinstance(protocol, dict):
        return False
    if experiment.get("schema") in {
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
        curve_schema_version = experiment["schema"].rsplit("v", maxsplit=1)[1]
        schema_path = (
            Path(__file__).resolve().parent.parent
            / "datasets"
            / "schema"
            / f"ordered-impairments-snr-curve.v{curve_schema_version}.schema.json"
        )
        try:
            schema_bytes = schema_path.read_bytes()
        except OSError:
            return False
        expected_schema_digest = _sha256(schema_bytes)
        if experiment.get("config", {}).get("schema_sha256") != expected_schema_digest:
            return False
        try:
            import jsonschema
        except ImportError:
            return False
        try:
            schema_document = json.loads(schema_bytes.decode("utf-8"))
            jsonschema.Draft202012Validator(schema_document).validate(experiment)
        except (
            OSError,
            json.JSONDecodeError,
            jsonschema.exceptions.SchemaError,
            jsonschema.exceptions.ValidationError,
        ):
            return False
    expected_protocol = {
        "experiment_schema": experiment.get("schema"),
        "config": experiment.get("config"),
        "acceptance_criteria": experiment.get("acceptance", {}).get("criteria_frozen_before_run"),
    }
    if protocol != expected_protocol:
        return False
    try:
        if provenance.get("protocol_sha256") != _sha256(_canonical_json_bytes(protocol)):
            return False
        unhashed = dict(result)
        supplied_digest = unhashed.pop("result_sha256", None)
        return supplied_digest == _sha256(_canonical_json_bytes(unhashed))
    except (TypeError, ValueError):
        return False


def write_hash_bound_result(path: str | Path, result: dict[str, Any]) -> None:
    """Write a verified result atomically as canonical, human-readable JSON."""
    if not verify_hash_bound_result(result):
        raise ValueError("refusing to write a result with invalid hash bindings")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as temporary_file:
            temporary_file.write(payload)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        Path(temporary_name).replace(destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-count", type=int, default=len(PAYLOAD_SEEDS))
    parser.add_argument("--blocks-per-seed", type=int, default=BLOCKS_PER_SEED)
    parser.add_argument(
        "--device",
        default="cpu",
        help="PUSCH receiver device for bounded independent-channel probes and SNR curves",
    )
    parser.add_argument(
        "--independent-channel-count",
        type=int,
        help="run the independent synthetic-channel probe for this many channel seeds",
    )
    parser.add_argument(
        "--channel-seed-start",
        type=int,
        help="use this explicit first seed for the independent-channel probe",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write a hash-bound local result JSON to this path",
    )
    parser.add_argument(
        "--waveform-evidence-output",
        type=Path,
        help="retain TX/oscillator/capture/AWGN samples for independent waveform auditing",
    )
    parser.add_argument(
        "--channel-estimator",
        choices=("sionna_default_ls", "finite_delay_dft"),
        default="sionna_default_ls",
        help="select the PUSCH DM-RS channel estimator for independent-channel probes",
    )
    parser.add_argument(
        "--snr-db",
        nargs="+",
        type=float,
        help="run an independent-channel SNR curve at these strictly increasing points",
    )
    parser.add_argument(
        "--acceptance-contract",
        type=Path,
        help=(
            "bounded JSON G5 SNR acceptance contract supplied before the curve run; "
            "only applies with --snr-db and independent-channel mode"
        ),
    )
    parser.add_argument(
        "--acceptance-signature",
        type=Path,
        help="detached OpenSSH signature over the exact canonical acceptance-contract bytes",
    )
    parser.add_argument(
        "--acceptance-allowed-signers",
        type=Path,
        help="externally trusted OpenSSH allowed-signers file for the acceptance contract",
    )
    parser.add_argument(
        "--acceptance-principal",
        help="expected signer principal in the allowed-signers file",
    )
    parser.add_argument(
        "--acceptance-reviewer-policy",
        type=Path,
        help="separate externally signed policy authorizing the acceptance signer as reviewer",
    )
    parser.add_argument(
        "--acceptance-reviewer-policy-signature",
        type=Path,
        help="detached OpenSSH signature over the exact canonical reviewer-policy bytes",
    )
    parser.add_argument(
        "--acceptance-reviewer-allowed-signers",
        type=Path,
        help="separate externally governed trust file for the reviewer-policy custodian",
    )
    parser.add_argument(
        "--acceptance-reviewer-policy-principal",
        help="expected custodian principal signing the reviewer policy",
    )
    args = parser.parse_args()
    if args.acceptance_contract is not None and args.snr_db is None:
        parser.error("--acceptance-contract requires --snr-db")
    signature_options = (
        args.acceptance_signature,
        args.acceptance_allowed_signers,
        args.acceptance_principal,
    )
    if any(value is not None for value in signature_options) and (
        args.acceptance_contract is None or any(value is None for value in signature_options)
    ):
        parser.error(
            "acceptance signature, allowed-signers, and principal require one another "
            "and --acceptance-contract"
        )
    reviewer_policy_options = (
        args.acceptance_reviewer_policy,
        args.acceptance_reviewer_policy_signature,
        args.acceptance_reviewer_allowed_signers,
        args.acceptance_reviewer_policy_principal,
    )
    if any(value is not None for value in reviewer_policy_options) and (
        args.acceptance_contract is None
        or any(value is None for value in reviewer_policy_options)
        or any(value is None for value in signature_options)
    ):
        parser.error(
            "reviewer policy, signature, trust keyring, and principal require a signed "
            "--acceptance-contract"
        )
    if args.independent_channel_count is not None:
        if args.independent_channel_count < 1:
            parser.error("--independent-channel-count must be positive")
        if args.channel_seed_start is None:
            if args.independent_channel_count > len(INDEPENDENT_CHANNEL_SEEDS):
                parser.error(
                    "--independent-channel-count exceeds the frozen default seed set; "
                    "provide --channel-seed-start to declare a separate seed range"
                )
            channel_seeds = INDEPENDENT_CHANNEL_SEEDS[: args.independent_channel_count]
        else:
            if args.channel_seed_start < 0:
                parser.error("--channel-seed-start must be nonnegative")
            channel_seeds = tuple(
                range(
                    args.channel_seed_start,
                    args.channel_seed_start + args.independent_channel_count,
                )
            )
        evidence_path = (
            args.waveform_evidence_output
            if args.snr_db is not None
            else args.waveform_evidence_output
            if args.waveform_evidence_output is not None
            else args.output.with_name(f"{args.output.stem}.waveforms.npz")
            if args.output is not None
            else None
        )
        if args.snr_db is not None:
            if args.output is not None and evidence_path is not None:
                if evidence_path.parent.resolve() != args.output.parent.resolve():
                    parser.error("waveform evidence must be a sibling of the result JSON")
                if evidence_path.name in {args.output.name, ".", ".."}:
                    parser.error("waveform evidence must use a distinct sibling file name")
            try:
                acceptance_criteria, acceptance_signature = (
                    _load_snr_acceptance_contract(
                        args.acceptance_contract,
                        signature_path=args.acceptance_signature,
                        trusted_allowed_signers_path=args.acceptance_allowed_signers,
                        signer_id=args.acceptance_principal,
                        reviewer_policy_path=args.acceptance_reviewer_policy,
                        reviewer_policy_signature_path=args.acceptance_reviewer_policy_signature,
                        trusted_reviewer_allowed_signers_path=(
                            args.acceptance_reviewer_allowed_signers
                        ),
                        reviewer_policy_signer_id=args.acceptance_reviewer_policy_principal,
                    )
                    if args.acceptance_contract is not None
                    else (None, None)
                )
                report = run_independent_ordered_impairment_snr_curve(
                    snr_db=args.snr_db,
                    channel_seeds=channel_seeds,
                    blocks_per_channel=args.blocks_per_seed,
                    channel_estimator=args.channel_estimator,
                    waveform_evidence_output=evidence_path,
                    acceptance_criteria=acceptance_criteria,
                    preregistration_signature=acceptance_signature,
                    device=args.device,
                )
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                parser.error(str(error))
            if args.output is not None:
                result = build_hash_bound_result(report)
                write_hash_bound_result(args.output, result)
                print(json.dumps(result, indent=2, sort_keys=True))
            else:
                print(json.dumps(report, indent=2, sort_keys=True))
            return
        if args.output is not None and evidence_path is not None:
            if evidence_path.parent.resolve() != args.output.parent.resolve():
                parser.error("waveform evidence must be a sibling of the result JSON")
            if evidence_path.name in {args.output.name, ".", ".."}:
                parser.error("waveform evidence must use a distinct sibling file name")
        if evidence_path is not None and (evidence_path.exists() or evidence_path.is_symlink()):
            parser.error(f"refusing to overwrite waveform evidence: {evidence_path}")
        report = run_independent_ordered_impairment_probe(
            channel_seeds=channel_seeds,
            blocks_per_channel=args.blocks_per_seed,
            waveform_evidence_output=evidence_path,
            channel_estimator=args.channel_estimator,
            device=args.device,
        )
        if args.output is not None:
            result = build_hash_bound_result(report)
            write_hash_bound_result(args.output, result)
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.channel_estimator != "sionna_default_ls":
        parser.error("--channel-estimator requires --independent-channel-count")
    if args.device != "cpu":
        parser.error("--device requires --independent-channel-count")
    if args.snr_db is not None:
        parser.error("--snr-db requires --independent-channel-count")
    if args.channel_seed_start is not None:
        parser.error("--channel-seed-start requires --independent-channel-count")
    if args.waveform_evidence_output is not None:
        parser.error("--waveform-evidence-output requires --independent-channel-count")
    if not 1 <= args.seed_count <= len(PAYLOAD_SEEDS):
        parser.error(f"--seed-count must be in [1, {len(PAYLOAD_SEEDS)}]")
    report = run_ordered_impairment_probe(
        payload_seeds=PAYLOAD_SEEDS[: args.seed_count], blocks_per_seed=args.blocks_per_seed
    )
    if args.output is not None:
        result = build_hash_bound_result(report)
        write_hash_bound_result(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
