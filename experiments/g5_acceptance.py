"""Torch-free validation and scoring for bounded G5 acceptance contracts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evaluation.g3_signed_attestation import verify_openssh_signature_for_bytes

# Structural minimum only; task-specific power/precision must be planned separately.
MIN_G5_INDEPENDENT_ACCEPTANCE_CHANNELS = 20
MAX_G5_INDEPENDENT_ACCEPTANCE_TRANSPORT_BLOCKS = 16_384
MAX_G5_INDEPENDENT_ACCEPTANCE_SNR_POINTS = 64
G5_SNR_ACCEPTANCE_SCHEMA = "umcd.g5_ordered_impairments_snr_acceptance.v1"
G5_SNR_TIMESTAMPED_ACCEPTANCE_SCHEMA = "umcd.g5_ordered_impairments_snr_acceptance.v2"
G5_SNR_ACCEPTANCE_SIGNATURE_NAMESPACE = "umcd-g5-snr-acceptance-v1"
G5_SNR_REVIEWER_POLICY_SCHEMA = "umcd.g5_snr_reviewer_policy.v1"
G5_SNR_REVIEWER_POLICY_PURPOSE = "g5_ordered_impairments_snr_preregistration"
G5_SNR_REVIEWER_POLICY_SIGNATURE_NAMESPACE = "umcd-g5-snr-reviewer-policy-v1"
G5_SNR_CURVE_BOOTSTRAP_SEED = 75_013
G5_SNR_CURVE_BOOTSTRAP_DRAWS = 4_096
G5_SNR_CURVE_BOOTSTRAP_BATCH_SIZE = 64
G5_SNR_CURVE_INFERENCE_UNIT = "independent synthetic channel; transport blocks are nested"
G5_SNR_CURVE_INFERENCE_METHOD = (
    "channel-cluster percentile intervals with simultaneous max-deviation band "
    "over conditions and paired correction effects"
)
G5_SNR_CURVE_PAIRED_CONTRAST = (
    "uncorrected macro-BLER minus corrected macro-BLER; positive values favor correction"
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def load_g5_snr_acceptance_contract(
    path: str | Path,
    *,
    signature_path: str | Path | None = None,
    trusted_allowed_signers_path: str | Path | None = None,
    signer_id: str | None = None,
    reviewer_policy_path: str | Path | None = None,
    reviewer_policy_signature_path: str | Path | None = None,
    trusted_reviewer_allowed_signers_path: str | Path | None = None,
    reviewer_policy_signer_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, str] | None]:
    """Read and optionally verify a canonical, exact-byte signed JSON contract.

    This admission path intentionally has no Torch or Sionna dependency, so
    preregistration and signature handling can be validated before receiver
    runtime setup.
    """
    supplied_signature_options = (
        signature_path is not None,
        trusted_allowed_signers_path is not None,
        signer_id is not None,
    )
    if any(supplied_signature_options) and not all(supplied_signature_options):
        raise ValueError(
            "acceptance signature, trusted allowed-signers file, and signer principal "
            "must be supplied together"
        )
    policy_options = (
        reviewer_policy_path,
        reviewer_policy_signature_path,
        trusted_reviewer_allowed_signers_path,
        reviewer_policy_signer_id,
    )
    if any(value is not None for value in policy_options) and not all(
        value is not None for value in policy_options
    ):
        raise ValueError(
            "G5 reviewer policy, signature, trust keyring, and custodian must be supplied together"
        )
    if any(value is not None for value in policy_options) and not all(supplied_signature_options):
        raise ValueError("G5 reviewer policy admission requires a signed acceptance contract")
    payload = _read_g5_bounded_single_link_bytes(path, label="acceptance contract")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"acceptance contract contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("acceptance contract must be valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError("acceptance contract root must be an object")
    if value.get("schema") == G5_SNR_TIMESTAMPED_ACCEPTANCE_SCHEMA and signature_path is None:
        raise ValueError("timestamped acceptance contract v2 requires a verified signature")
    signature_identity: dict[str, str] | None = None
    if signature_path is not None:
        canonical_payload = _canonical_json_bytes(value)
        if payload != canonical_payload:
            raise ValueError(
                "signed acceptance contract must use canonical sorted compact JSON bytes"
            )
        assert trusted_allowed_signers_path is not None
        assert signer_id is not None
        verified = verify_openssh_signature_for_bytes(
            payload,
            signature_path,
            trusted_allowed_signers_path,
            signer_id=signer_id,
            namespace=G5_SNR_ACCEPTANCE_SIGNATURE_NAMESPACE,
        )
        signature_identity = {
            "status": "verified_before_receiver_run",
            "namespace": verified["namespace"],
            "signer_id": verified["signer_id"],
            "contract_payload_sha256": hashlib.sha256(payload).hexdigest(),
            "signature_sha256": verified["signature_sha256"],
            "trusted_allowed_signers_sha256": verified["allowed_signers_sha256"],
            "trust_file_custody": "not_established_by_this_verifier",
        }
        if value.get("schema") == G5_SNR_TIMESTAMPED_ACCEPTANCE_SCHEMA:
            signature_identity["preregistered_utc"] = value.get("preregistered_utc")
        if reviewer_policy_path is not None:
            if value.get("schema") != G5_SNR_TIMESTAMPED_ACCEPTANCE_SCHEMA:
                raise ValueError(
                    "G5 reviewer-role policy admission requires timestamped contract v2"
                )
            assert reviewer_policy_signature_path is not None
            assert trusted_reviewer_allowed_signers_path is not None
            assert reviewer_policy_signer_id is not None
            signature_identity["reviewer_policy"] = _verify_g5_reviewer_policy(
                reviewer_policy_path,
                reviewer_policy_signature_path,
                trusted_reviewer_allowed_signers_path,
                reviewer_policy_signer_id,
                acceptance_signer_id=signer_id,
                acceptance_allowed_signers_sha256=verified["allowed_signers_sha256"],
            )
    return value, signature_identity


def _read_g5_bounded_single_link_bytes(path: str | Path, *, label: str) -> bytes:
    input_path = Path(path)
    if not hasattr(os, "O_NOFOLLOW"):
        raise ValueError("platform lacks safe no-symlink G5 policy opening")
    before_path = os.lstat(input_path)
    if not stat.S_ISREG(before_path.st_mode) or before_path.st_nlink != 1:
        raise ValueError(f"{label} must be a single-link regular file")
    descriptor = os.open(input_path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > 65_536:
            raise ValueError(f"{label} must be a single-link regular file no larger than 64 KiB")
        payload = os.read(descriptor, 65_537)
        after = os.fstat(descriptor)
        after_path = os.lstat(input_path)
    finally:
        os.close(descriptor)

    def file_signature(item: os.stat_result) -> tuple[int, ...]:
        return (
            item.st_dev,
            item.st_ino,
            item.st_mode,
            item.st_nlink,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )

    if (
        len(payload) > 65_536
        or len(payload) != before.st_size
        or file_signature(before) != file_signature(after)
        or file_signature(before) != file_signature(after_path)
    ):
        raise ValueError(f"{label} changed during bounded read")
    return payload


def _verify_g5_reviewer_policy(
    policy_path: str | Path,
    policy_signature_path: str | Path,
    trusted_allowed_signers_path: str | Path,
    policy_signer_id: str,
    *,
    acceptance_signer_id: str | None,
    acceptance_allowed_signers_sha256: str,
) -> dict[str, str]:
    """Authenticate a separate policy authorizing the acceptance signer as reviewer."""
    raw = _read_g5_bounded_single_link_bytes(policy_path, label="G5 reviewer policy")

    def reject_duplicate_policy_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"G5 reviewer policy contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        policy = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicate_policy_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("G5 reviewer policy must be valid UTF-8 JSON") from error
    if not isinstance(policy, dict) or _canonical_json_bytes(policy) != raw:
        raise ValueError("G5 reviewer policy must use canonical sorted compact JSON bytes")
    import jsonschema

    schema_path = (
        Path(__file__).resolve().parents[1]
        / "datasets/schema/g5-snr-reviewer-policy.v1.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(policy))
    if errors:
        raise ValueError(
            "G5 reviewer policy violates schema: "
            + "; ".join(error.message for error in errors[:10])
        )
    if acceptance_signer_id is None or policy_signer_id == acceptance_signer_id:
        raise ValueError("G5 reviewer-policy custodian must differ from the acceptance signer")
    policy_signature = verify_openssh_signature_for_bytes(
        raw,
        policy_signature_path,
        trusted_allowed_signers_path,
        signer_id=policy_signer_id,
        namespace=G5_SNR_REVIEWER_POLICY_SIGNATURE_NAMESPACE,
    )
    if policy_signature["allowed_signers_sha256"] == acceptance_allowed_signers_sha256:
        raise ValueError("G5 reviewer-policy keyring must differ from the acceptance keyring")
    authorized = [item for item in policy["signers"] if item["principal"] == acceptance_signer_id]
    if len(authorized) != 1:
        raise ValueError("acceptance signer must occur exactly once in the G5 reviewer policy")
    reviewer = authorized[0]
    if (
        "independent_reviewer" not in reviewer["roles"]
        or G5_SNR_REVIEWER_POLICY_PURPOSE not in reviewer["allowed_purposes"]
    ):
        raise ValueError("G5 acceptance signer is not authorized as an independent reviewer")
    return {
        "status": "verified",
        "policy_id": policy["policy_id"],
        "policy_sha256": hashlib.sha256(raw).hexdigest(),
        "reviewer_principal": acceptance_signer_id,
        "reviewer_organization": reviewer["organization"],
        "reviewer_roles": ",".join(sorted(reviewer["roles"])),
        "purpose": G5_SNR_REVIEWER_POLICY_PURPOSE,
        "custodian_principal": policy_signature["signer_id"],
        "signature_namespace": policy_signature["namespace"],
        "signature_sha256": policy_signature["signature_sha256"],
        "trusted_allowed_signers_sha256": policy_signature["allowed_signers_sha256"],
        "trust_file_custody": "not_established_by_this_verifier",
    }


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_finite_real(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _parse_utc_timestamp(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{field} must be an ISO-8601 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(f"{field} must be a valid ISO-8601 UTC timestamp") from error
    if parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError(f"{field} must use UTC")
    return parsed


def _validate_g5_snr_acceptance_criteria(criteria: Any) -> dict[str, Any]:
    """Validate the complete task contract independently of a receiver run."""
    if not isinstance(criteria, dict):
        raise TypeError("acceptance_criteria must be an object")
    base_fields = {
        "schema",
        "snr_db",
        "channel_seeds",
        "channel_estimator",
        "blocks_per_channel",
        "minimum_independent_channels",
        "maximum_corrected_macro_bler",
        "minimum_uncorrected_minus_corrected_macro_bler",
    }
    schema = criteria.get("schema")
    if schema == G5_SNR_ACCEPTANCE_SCHEMA:
        required = base_fields
    elif schema == G5_SNR_TIMESTAMPED_ACCEPTANCE_SCHEMA:
        required = base_fields | {"preregistered_utc"}
    else:
        raise ValueError("acceptance_criteria has an unsupported schema")
    if set(criteria) != required:
        raise ValueError("acceptance_criteria has missing or unknown fields")

    # Bound in-memory callers before canonicalization duplicates their object
    # into a serialized snapshot. File-based callers also have a 64 KiB cap,
    # but the public scoring API accepts Python objects directly.
    snr_points = criteria.get("snr_db")
    channel_seeds = criteria.get("channel_seeds")
    blocks_per_channel = criteria.get("blocks_per_channel")
    if isinstance(snr_points, list) and len(snr_points) > MAX_G5_INDEPENDENT_ACCEPTANCE_SNR_POINTS:
        raise ValueError(
            "acceptance contract SNR grid exceeds the bounded limit of "
            f"{MAX_G5_INDEPENDENT_ACCEPTANCE_SNR_POINTS} points"
        )
    if (
        isinstance(snr_points, list)
        and isinstance(channel_seeds, list)
        and isinstance(blocks_per_channel, int)
        and (
            not isinstance(blocks_per_channel, bool)
            and blocks_per_channel > 0
            and len(channel_seeds)
            > MAX_G5_INDEPENDENT_ACCEPTANCE_TRANSPORT_BLOCKS
            // max(1, len(snr_points))
            // blocks_per_channel
        )
    ):
        raise ValueError(
            "acceptance contract requests more than the bounded limit of "
            f"{MAX_G5_INDEPENDENT_ACCEPTANCE_TRANSPORT_BLOCKS} transport blocks"
        )
    try:
        frozen = json.loads(_canonical_json_bytes(criteria).decode("utf-8"))
    except (TypeError, ValueError, UnicodeDecodeError) as error:
        raise ValueError("acceptance_criteria must contain canonical JSON values") from error
    if schema == G5_SNR_TIMESTAMPED_ACCEPTANCE_SCHEMA:
        _parse_utc_timestamp(frozen["preregistered_utc"], field="preregistered_utc")

    contract_snr = frozen["snr_db"]
    if (
        not isinstance(contract_snr, list)
        or not contract_snr
        or any(not _is_finite_real(value) for value in contract_snr)
    ):
        raise ValueError("acceptance contract SNR grid must contain finite real points")
    normalized_snr = tuple(float(value) for value in contract_snr)
    if any(right <= left for left, right in zip(normalized_snr, normalized_snr[1:])):
        raise ValueError("acceptance contract SNR grid must be strictly increasing")
    estimator = frozen["channel_estimator"]
    if not isinstance(estimator, str) or estimator not in {
        "sionna_default_ls",
        "finite_delay_dft",
    }:
        raise ValueError("acceptance contract channel estimator is unsupported")
    contract_seeds = frozen["channel_seeds"]
    if (
        not isinstance(contract_seeds, list)
        or not contract_seeds
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in contract_seeds
        )
        or len(set(contract_seeds)) != len(contract_seeds)
    ):
        raise ValueError("acceptance contract channel seeds must be unique nonnegative integers")
    blocks = frozen["blocks_per_channel"]
    if isinstance(blocks, bool) or not isinstance(blocks, int) or blocks <= 0:
        raise ValueError("acceptance contract block count must be a positive integer")
    minimum_channels = frozen["minimum_independent_channels"]
    if (
        isinstance(minimum_channels, bool)
        or not isinstance(minimum_channels, int)
        or minimum_channels < MIN_G5_INDEPENDENT_ACCEPTANCE_CHANNELS
    ):
        raise ValueError(
            "minimum_independent_channels must be an integer of at least "
            f"{MIN_G5_INDEPENDENT_ACCEPTANCE_CHANNELS}"
        )
    if minimum_channels > len(contract_seeds):
        raise ValueError("minimum_independent_channels exceeds the frozen channel-seed roster")
    for name in (
        "maximum_corrected_macro_bler",
        "minimum_uncorrected_minus_corrected_macro_bler",
    ):
        values = frozen[name]
        if not isinstance(values, list) or len(values) != len(contract_snr):
            raise ValueError(f"acceptance contract {name} must cover every SNR point")
        if any(not _is_finite_real(value) for value in values):
            raise ValueError(f"acceptance contract {name} must contain finite real values")
    if any(value < 0 or value > 1 for value in frozen["maximum_corrected_macro_bler"]):
        raise ValueError("maximum corrected macro-BLER thresholds must be in [0, 1]")
    if any(
        value < 0 or value > 1 for value in frozen["minimum_uncorrected_minus_corrected_macro_bler"]
    ):
        raise ValueError("minimum paired BLER improvements must be in [0, 1]")
    return frozen


def validate_g5_preregistration_order(
    criteria: dict[str, Any], *, receiver_run_started_utc: str
) -> dict[str, str]:
    """Check declared UTC order for a timestamped acceptance contract.

    The returned status is a consistency check only. Neither this function nor
    an OpenSSH signature establishes trusted timestamp authority or signing time.
    """
    frozen = _validate_g5_snr_acceptance_criteria(criteria)
    if frozen["schema"] != G5_SNR_TIMESTAMPED_ACCEPTANCE_SCHEMA:
        raise ValueError("signed G5 preregistration requires acceptance contract schema v2")
    preregistered_at = _parse_utc_timestamp(frozen["preregistered_utc"], field="preregistered_utc")
    run_started_at = _parse_utc_timestamp(
        receiver_run_started_utc, field="receiver_run_started_utc"
    )
    if preregistered_at >= run_started_at:
        raise ValueError("preregistered_utc must precede receiver_run_started_utc")
    return {
        "declared_temporal_order_status": "preregistration_precedes_receiver_run",
        "preregistered_utc": frozen["preregistered_utc"],
        "receiver_run_started_utc": receiver_run_started_utc,
        "timestamp_authority": "not_established_by_this_verifier",
    }


def freeze_g5_snr_acceptance_criteria(
    criteria: dict[str, Any] | None,
    *,
    snr_values: tuple[float, ...],
    channel_seeds: tuple[int, ...],
    channel_estimator: str,
    blocks_per_channel: int,
) -> dict[str, Any] | None:
    """Validate and snapshot the caller's pre-run receiver acceptance contract."""
    if criteria is None:
        return None
    if isinstance(criteria, dict):
        declared_snr = criteria.get("snr_db")
        if (
            isinstance(declared_snr, list)
            and len(declared_snr) > MAX_G5_INDEPENDENT_ACCEPTANCE_SNR_POINTS
        ):
            raise ValueError(
                "acceptance contract SNR grid exceeds the bounded limit of "
                f"{MAX_G5_INDEPENDENT_ACCEPTANCE_SNR_POINTS} points"
            )
        if (
            isinstance(declared_snr, list)
            and all(_is_finite_real(value) for value in declared_snr)
            and tuple(float(value) for value in declared_snr) != snr_values
        ):
            raise ValueError("acceptance contract SNR grid must exactly match the requested curve")
    frozen = _validate_g5_snr_acceptance_criteria(criteria)
    contract_snr = tuple(float(value) for value in frozen["snr_db"])
    if contract_snr != snr_values:
        raise ValueError("acceptance contract SNR grid must exactly match the requested curve")
    if frozen["channel_estimator"] != channel_estimator:
        raise ValueError("acceptance contract channel estimator does not match the requested curve")
    if tuple(frozen["channel_seeds"]) != channel_seeds:
        raise ValueError("acceptance contract channel seeds must exactly match the requested curve")
    if frozen["blocks_per_channel"] != blocks_per_channel:
        raise ValueError("acceptance contract block count does not match the requested curve")
    return frozen


def evaluate_g5_snr_curve_acceptance(
    criteria: dict[str, Any],
    *,
    curve_inference: dict[str, Any],
    preregistration_signature: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Apply frozen thresholds to simultaneous independent-channel intervals."""
    criteria = _validate_g5_snr_acceptance_criteria(criteria)
    if not isinstance(curve_inference, dict):
        raise TypeError("curve_inference must be an object")

    def interval(
        row: dict[str, Any],
        *,
        field: str,
        lower_bound: float,
        upper_bound: float,
        label: str,
    ) -> tuple[float, float]:
        if not isinstance(row, dict):
            raise ValueError(f"curve inference {label} row must be an object")
        values = row.get(field)
        if not isinstance(values, list) or len(values) != 2:
            raise ValueError(f"curve inference is missing a two-endpoint {label} interval")
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"curve inference {label} interval endpoints must be finite numbers"
                )
            try:
                finite = math.isfinite(float(value))
            except (OverflowError, TypeError, ValueError):
                finite = False
            if not finite:
                raise ValueError(
                    f"curve inference {label} interval endpoints must be finite numbers"
                )
        low, high = (float(value) for value in values)
        if low > high or low < lower_bound or high > upper_bound:
            raise ValueError(f"curve inference {label} interval is reversed or out of domain")
        return low, high

    def matches_snr(row: dict[str, Any], expected_snr: float) -> bool:
        value = row.get("snr_db")
        return _is_finite_real(value) and float(value) == expected_snr

    digest = _sha256(_canonical_json_bytes(criteria))
    frozen_record: dict[str, Any] = {
        "status": "contract_supplied_before_run",
        "criteria": criteria,
        "criteria_sha256": digest,
        "confidence_level": 0.95,
        "interval_scope": "simultaneous across all three conditions and paired improvements",
        "decision_unit": G5_SNR_CURVE_INFERENCE_UNIT,
    }
    if preregistration_signature is not None:
        frozen_record["preregistration_signature"] = preregistration_signature
    cluster_count = curve_inference.get("cluster_count")
    if (
        curve_inference.get("status") != "available"
        or isinstance(cluster_count, bool)
        or not isinstance(cluster_count, int)
        or cluster_count < criteria["minimum_independent_channels"]
    ):
        return {
            "criteria_frozen_before_run": frozen_record,
            "status": "NOT_EVALUABLE",
            "reason": "insufficient independent-channel clusters for the frozen contract",
            "cluster_count": cluster_count,
            "pass": False,
            "point_decisions": [],
        }

    if cluster_count != len(criteria["channel_seeds"]):
        raise ValueError("curve inference cluster count does not match the frozen seed roster")
    inference_seeds = curve_inference.get("channel_seeds")
    if (
        not isinstance(inference_seeds, list)
        or any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            for seed in inference_seeds
        )
        or len(set(inference_seeds)) != len(inference_seeds)
        or inference_seeds != criteria["channel_seeds"]
    ):
        raise ValueError("curve inference channel seeds do not match the frozen seed roster")
    expected_family_size = len(criteria["snr_db"]) * 4
    if (
        curve_inference.get("unit") != G5_SNR_CURVE_INFERENCE_UNIT
        or curve_inference.get("bootstrap_seed") != G5_SNR_CURVE_BOOTSTRAP_SEED
        or curve_inference.get("bootstrap_draws") != G5_SNR_CURVE_BOOTSTRAP_DRAWS
        or curve_inference.get("family_size") != expected_family_size
        or curve_inference.get("method") != G5_SNR_CURVE_INFERENCE_METHOD
        or curve_inference.get("paired_contrast_definition") != G5_SNR_CURVE_PAIRED_CONTRAST
    ):
        raise ValueError("curve inference metadata does not match the frozen G5 method")
    simultaneous_radius = curve_inference.get("simultaneous_radius")
    if (
        not _is_finite_real(simultaneous_radius)
        or simultaneous_radius < 0
        or simultaneous_radius > 1
    ):
        raise ValueError("curve inference simultaneous radius must be finite and in [0, 1]")

    conditions = curve_inference.get("by_condition")
    if not isinstance(conditions, dict) or set(conditions) != {"clean", "uncorrected", "corrected"}:
        raise ValueError("curve inference must cover the complete frozen condition family")
    condition_rows: dict[str, list[dict[str, Any]]] = {}
    for condition, rows in conditions.items():
        if not isinstance(rows, list) or len(rows) != len(criteria["snr_db"]):
            raise ValueError(f"curve inference {condition} rows do not match the frozen SNR grid")
        normalized_rows: list[dict[str, Any]] = []
        for index, (row, snr) in enumerate(zip(rows, criteria["snr_db"], strict=True)):
            if not isinstance(row, dict) or not matches_snr(row, snr):
                raise ValueError(
                    f"curve inference {condition} SNR ordering differs from the frozen contract"
                )
            interval(
                row,
                field="simultaneous_95_percent_interval",
                lower_bound=0.0,
                upper_bound=1.0,
                label=f"{condition} macro-BLER at SNR index {index}",
            )
            normalized_rows.append(row)
        condition_rows[condition] = normalized_rows

    improvements = curve_inference.get("by_paired_contrast")
    if not isinstance(improvements, list) or len(improvements) != len(criteria["snr_db"]):
        raise ValueError("paired curve inference rows do not match the frozen SNR grid")
    for index, (row, snr) in enumerate(zip(improvements, criteria["snr_db"], strict=True)):
        if not isinstance(row, dict) or not matches_snr(row, snr):
            raise ValueError("curve inference paired SNR ordering differs from the frozen contract")
        interval(
            row,
            field="simultaneous_95_percent_interval",
            lower_bound=-1.0,
            upper_bound=1.0,
            label=f"paired BLER contrast at SNR index {index}",
        )

    corrected = condition_rows["corrected"]
    point_decisions = []
    all_pass = True
    for index, snr in enumerate(criteria["snr_db"]):
        corrected_row = corrected[index]
        improvement_row = improvements[index]
        if (
            not isinstance(corrected_row, dict)
            or not isinstance(improvement_row, dict)
            or not matches_snr(corrected_row, snr)
            or not matches_snr(improvement_row, snr)
        ):
            raise ValueError("curve inference SNR ordering differs from the frozen contract")
        _, corrected_upper = interval(
            corrected_row,
            field="simultaneous_95_percent_interval",
            lower_bound=0.0,
            upper_bound=1.0,
            label="corrected macro-BLER",
        )
        improvement_lower, _ = interval(
            improvement_row,
            field="simultaneous_95_percent_interval",
            lower_bound=-1.0,
            upper_bound=1.0,
            label="paired BLER contrast",
        )
        corrected_pass = corrected_upper <= criteria["maximum_corrected_macro_bler"][index]
        improvement_pass = (
            improvement_lower >= criteria["minimum_uncorrected_minus_corrected_macro_bler"][index]
        )
        point_pass = corrected_pass and improvement_pass
        all_pass = all_pass and point_pass
        point_decisions.append(
            {
                "snr_db": snr,
                "corrected_macro_bler_simultaneous_upper": corrected_upper,
                "maximum_corrected_macro_bler": criteria["maximum_corrected_macro_bler"][index],
                "corrected_bler_criterion_pass": corrected_pass,
                "uncorrected_minus_corrected_simultaneous_lower": improvement_lower,
                "minimum_uncorrected_minus_corrected_macro_bler": criteria[
                    "minimum_uncorrected_minus_corrected_macro_bler"
                ][index],
                "minimum_improvement_criterion_pass": improvement_pass,
                "pass": point_pass,
            }
        )
    return {
        "criteria_frozen_before_run": frozen_record,
        "status": "PASS" if all_pass else "FAIL",
        "cluster_count": cluster_count,
        "pass": all_pass,
        "point_decisions": point_decisions,
    }
