"""Preregistration order, reviewer-signature and execution-context checks for SNR-curve results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evaluation.signed_attestation import verify_openssh_signature_for_bytes

from experiments.acceptance import (
    validate_preregistration_order,
)

from .constants import (
    _EXACT_SEED_ROSTER_CURVE_SCHEMAS,
    _SNR_ACCEPTANCE_SIGNATURE_NAMESPACE,
)
from .hashing import (
    _canonical_json_bytes,
    _sha256,
)


def _inference_seed_roster_consistent(
    curve_schema: str,
    curve_inference: dict[str, Any],
    curve_seeds: list[int],
) -> bool:
    """Require exact inference IDs only in curve versions that declare them."""
    if curve_schema not in _EXACT_SEED_ROSTER_CURVE_SCHEMAS:
        return "channel_seeds" not in curve_inference
    inference_seeds = curve_inference.get("channel_seeds")
    return (
        isinstance(inference_seeds, list)
        and all(
            isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0
            for seed in inference_seeds
        )
        and len(inference_seeds) == len(set(inference_seeds))
        and inference_seeds == curve_seeds
    )


def _declared_preregistration_order_matches(criteria: Any, signature: Any) -> bool:
    """Check v2 declared UTC chronology without claiming trusted timestamp proof."""
    if not isinstance(criteria, dict) or not isinstance(signature, dict):
        return False
    receiver_run_started_utc = signature.get("receiver_run_started_utc")
    if not isinstance(receiver_run_started_utc, str):
        return False
    try:
        expected = validate_preregistration_order(
            criteria, receiver_run_started_utc=receiver_run_started_utc
        )
    except (TypeError, ValueError):
        return False
    return bool(expected) and all(
        signature.get(field) == value for field, value in expected.items()
    )


def _execution_context_matches(contract: Any, points: Any) -> bool:
    """Check a v3 context digest and exact receiver/waveform identity at every point."""
    if not isinstance(contract, dict) or not isinstance(points, list):
        return False
    context = contract.get("execution_context")
    if not isinstance(context, dict) or set(context) != {
        "receiver_configuration",
        "decoder_configuration",
        "waveform_configuration",
    }:
        return False
    try:
        if contract.get("execution_context_sha256") != _sha256(_canonical_json_bytes(context)):
            return False
    except (TypeError, ValueError):
        return False

    waveform_keys = (
        "sample_rate_hz",
        "fft_size",
        "cyclic_prefix_samples",
        "echo_delay_samples_range",
        "echo_amplitude_range",
        "direct_coefficient_real_imag",
        "cfo_hz",
        "timing_offset_samples",
        "tx_phase_noise_linewidth_hz",
        "rx_phase_noise_linewidth_hz",
        "tx_iq_gain_imbalance_db",
        "tx_iq_phase_imbalance_rad",
        "rx_iq_gain_imbalance_db",
        "rx_iq_phase_imbalance_rad",
    )
    for point in points:
        report = point.get("experiment") if isinstance(point, dict) else None
        config = report.get("config") if isinstance(report, dict) else None
        if not isinstance(config, dict) or any(key not in config for key in waveform_keys):
            return False
        observed = {
            "receiver_configuration": config.get("receiver_configuration"),
            "decoder_configuration": config.get("decoder_configuration"),
            "waveform_configuration": {
                "channel_family": "independently_seeded_static_two_path",
                **{key: config[key] for key in waveform_keys},
                "noise_variance_per_complex_sample": "10**(-snr_db/10)",
                "noise_reference": "unit_complex_waveform_power",
            },
        }
        if observed != context:
            return False
    return True


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _verify_preregistration_signature(
    contract: dict[str, Any],
    signature_record: dict[str, Any],
    *,
    signature_path: Path,
    trusted_allowed_signers_path: Path,
) -> bool:
    payload = _canonical_json_bytes(contract)
    if (
        signature_record.get("status") != "verified_before_receiver_run"
        or signature_record.get("namespace") != _SNR_ACCEPTANCE_SIGNATURE_NAMESPACE
        or not isinstance(signature_record.get("signer_id"), str)
        or not signature_record["signer_id"].strip()
        or signature_record.get("contract_payload_sha256") != _sha256(payload)
        or not _is_sha256(signature_record.get("signature_sha256"))
        or not _is_sha256(signature_record.get("trusted_allowed_signers_sha256"))
        or signature_record.get("trust_file_custody") != "not_established_by_this_verifier"
    ):
        return False
    try:
        identity = verify_openssh_signature_for_bytes(
            payload,
            signature_path,
            trusted_allowed_signers_path,
            signer_id=signature_record["signer_id"],
            namespace=_SNR_ACCEPTANCE_SIGNATURE_NAMESPACE,
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        return False
    return (
        identity["signature_sha256"] == signature_record["signature_sha256"]
        and identity["allowed_signers_sha256"] == signature_record["trusted_allowed_signers_sha256"]
    )
