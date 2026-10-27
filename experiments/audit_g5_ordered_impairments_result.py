"""Independently audit retained summaries in a hash-bound G5.09 result."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from evaluation.g3_signed_attestation import verify_openssh_signature_for_bytes
from scipy.stats import beta

from experiments.g5_acceptance import (
    G5_SNR_CURVE_BOOTSTRAP_BATCH_SIZE,
    G5_SNR_CURVE_BOOTSTRAP_DRAWS,
    G5_SNR_CURVE_BOOTSTRAP_SEED,
    G5_SNR_REVIEWER_POLICY_PURPOSE,
    G5_SNR_REVIEWER_POLICY_SIGNATURE_NAMESPACE,
    G5_SNR_TIMESTAMPED_ACCEPTANCE_SCHEMA,
    MIN_G5_INDEPENDENT_ACCEPTANCE_CHANNELS,
    _verify_g5_reviewer_policy,
    validate_g5_preregistration_order,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_RESULT_SCHEMA = "umcd.g5_ordered_impairments_result.v2"
_G5_SNR_ACCEPTANCE_SIGNATURE_NAMESPACE = "umcd-g5-snr-acceptance-v1"
_BLOCK_BOUND_REPORT_SCHEMAS = {
    f"umcd.g5_ordered_impairments_independent_channels.v{version}" for version in range(15, 25)
}
_PER_CHANNEL_WAVEFORM_DIAGNOSTIC_SCHEMAS = {
    f"umcd.g5_ordered_impairments_independent_channels.v{version}" for version in range(7, 25)
}
_CFO_CHANNEL_TAIL_SCHEMAS = {
    f"umcd.g5_ordered_impairments_independent_channels.v{version}" for version in range(6, 25)
}
_CUDA_DEVICE_REPORT_SCHEMAS = {
    f"umcd.g5_ordered_impairments_independent_channels.v{version}" for version in range(21, 25)
}
_MAX_RESULT_BYTES = 32 * 1024 * 1024
_HEX = set("0123456789abcdef")
_MAX_WAVEFORM_EVIDENCE_BYTES = 512 * 1024 * 1024
_MAX_WAVEFORM_EVIDENCE_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024
_EXACT_SEED_ROSTER_CURVE_SCHEMAS = {
    "umcd.g5_ordered_impairments_independent_channels_snr_curve.v7",
    "umcd.g5_ordered_impairments_independent_channels_snr_curve.v8",
    "umcd.g5_ordered_impairments_independent_channels_snr_curve.v10",
    "umcd.g5_ordered_impairments_independent_channels_snr_curve.v11",
}


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def _g5_declared_preregistration_order_matches(criteria: Any, signature: Any) -> bool:
    """Check v2 declared UTC chronology without claiming trusted timestamp proof."""
    if not isinstance(criteria, dict) or not isinstance(signature, dict):
        return False
    try:
        expected = validate_g5_preregistration_order(
            criteria, receiver_run_started_utc=signature.get("receiver_run_started_utc")
        )
    except (TypeError, ValueError):
        return False
    return bool(expected) and all(
        signature.get(field) == value for field, value in expected.items()
    )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _verify_g5_preregistration_signature(
    contract: dict[str, Any],
    signature_record: dict[str, Any],
    *,
    signature_path: Path,
    trusted_allowed_signers_path: Path,
) -> bool:
    payload = _canonical_json_bytes(contract)
    if (
        signature_record.get("status") != "verified_before_receiver_run"
        or signature_record.get("namespace") != _G5_SNR_ACCEPTANCE_SIGNATURE_NAMESPACE
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
            namespace=_G5_SNR_ACCEPTANCE_SIGNATURE_NAMESPACE,
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        return False
    return (
        identity["signature_sha256"] == signature_record["signature_sha256"]
        and identity["allowed_signers_sha256"] == signature_record["trusted_allowed_signers_sha256"]
    )


def _curve_report_schema_status(
    report: dict[str, Any], schema_path: Path
) -> tuple[str | None, bool]:
    """Return the exact schema digest and validate the curve against those bytes."""
    try:
        schema_bytes = schema_path.read_bytes()
    except OSError:
        return None, False
    digest = _sha256(schema_bytes)
    try:
        import jsonschema
    except ImportError:
        return digest, False
    try:
        schema_document = json.loads(schema_bytes.decode("utf-8"))
        jsonschema.Draft202012Validator(schema_document).validate(report)
    except (
        json.JSONDecodeError,
        UnicodeDecodeError,
        jsonschema.exceptions.SchemaError,
        jsonschema.exceptions.ValidationError,
    ):
        return digest, False
    return digest, True


def _estimator_source_checks(
    estimator: dict[str, Any],
    *,
    source_bytes: bytes | None,
    committed_bytes: bytes | None,
) -> dict[str, bool | None]:
    """Verify the fixed estimator identity against disk and an optional Git blob."""
    if estimator.get("name") == "sionna_default_ls":
        valid = estimator == {"name": "sionna_default_ls"}
        return {
            "estimator_source_worktree_digest": valid,
            "estimator_committed_source_digest": True if valid else False,
        }
    if estimator.get("name") != "finite_delay_dft":
        raise ValueError("unsupported channel estimator identity")

    digest = estimator.get("implementation_sha256")
    valid = (
        estimator.get("implementation_path") == "experiments/g5_cp_bounded_dft_estimator.py"
        and estimator.get("min_delay_samples") == 0
        and estimator.get("max_delay_samples") == 7
        and isinstance(digest, str)
        and len(digest) == 64
        and set(digest) <= _HEX
    )
    return {
        "estimator_source_worktree_digest": (
            valid and source_bytes is not None and digest == _sha256(source_bytes)
        ),
        "estimator_committed_source_digest": (
            None if committed_bytes is None else valid and digest == _sha256(committed_bytes)
        ),
    }


def _read_bounded_evidence_file(path: Path, *, maximum_bytes: int) -> bytes:
    """Read one immutable, single-link regular file without following its final symlink."""
    if isinstance(maximum_bytes, bool) or not isinstance(maximum_bytes, int) or maximum_bytes < 0:
        raise ValueError("maximum_bytes must be a nonnegative integer")
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_NONBLOCK"):
        raise RuntimeError("platform lacks no-follow file opening required for waveform evidence")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError("waveform evidence must be a single-link regular file")
        if before.st_size > maximum_bytes:
            raise ValueError("waveform evidence exceeds the configured size limit")

        contents = bytearray()
        while len(contents) <= maximum_bytes:
            remaining = maximum_bytes + 1 - len(contents)
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                break
            contents.extend(block)
        if len(contents) > maximum_bytes:
            raise ValueError("waveform evidence exceeds the configured size limit")

        after = os.fstat(descriptor)
        path_after = os.stat(path, follow_symlinks=False)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        identity_at_path = (
            path_after.st_dev,
            path_after.st_ino,
            path_after.st_mode,
            path_after.st_nlink,
            path_after.st_size,
            path_after.st_mtime_ns,
            path_after.st_ctime_ns,
        )
        if (
            identity_before != identity_after
            or identity_after != identity_at_path
            or not stat.S_ISREG(path_after.st_mode)
            or path_after.st_nlink != 1
            or len(contents) != before.st_size
        ):
            raise ValueError("waveform evidence changed while being read")
        return bytes(contents)
    finally:
        os.close(descriptor)


def _same_float(actual: Any, expected: float) -> bool:
    return (
        not isinstance(actual, bool)
        and isinstance(actual, (int, float))
        and math.isfinite(float(actual))
        and math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
    )


def _is_finite_nonnegative(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _iq_imbalance_reference(
    values: np.ndarray, *, gain_imbalance_db: float, phase_imbalance_rad: float
) -> np.ndarray:
    i_gain = 10.0 ** (gain_imbalance_db / 40.0)
    q_gain = 10.0 ** (-gain_imbalance_db / 40.0)
    return (
        i_gain * values.real
        - q_gain * math.sin(phase_imbalance_rad) * values.imag
        + 1j * q_gain * math.cos(phase_imbalance_rad) * values.imag
    )


def _path_sum_reference(
    values: np.ndarray, coefficients: list[list[float]], delays: list[int]
) -> np.ndarray:
    result = np.zeros(values.shape, dtype=np.complex128)
    for coefficient_pair, delay in zip(coefficients, delays, strict=True):
        coefficient = complex(float(coefficient_pair[0]), float(coefficient_pair[1]))
        if delay == 0:
            result += coefficient * values
        else:
            result[..., delay:] += coefficient * values[..., :-delay]
    return result


def _positive_integer_delay_reference(values: np.ndarray, delay: int) -> np.ndarray:
    result = np.zeros_like(values)
    result[..., delay:] = values[..., :-delay]
    return result


def _preflight_waveform_archive(
    evidence_bytes: bytes, *, required_arrays: set[str]
) -> dict[str, tuple[tuple[int, ...], np.dtype[Any]]]:
    """Check ZIP and NPY metadata before NumPy materializes decompressed arrays."""
    if len(evidence_bytes) > _MAX_WAVEFORM_EVIDENCE_BYTES:
        raise ValueError("waveform evidence exceeds the configured size limit")
    expected_members = {f"{name}.npy" for name in required_arrays}
    with zipfile.ZipFile(io.BytesIO(evidence_bytes)) as compressed:
        members = compressed.infolist()
        if (
            len(members) != len(expected_members)
            or {member.filename for member in members} != expected_members
            or sum(member.file_size for member in members)
            > _MAX_WAVEFORM_EVIDENCE_UNCOMPRESSED_BYTES
        ):
            raise ValueError("waveform evidence archive exceeds its member or size contract")
        metadata: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {}
        for info in members:
            with compressed.open(info) as member:
                version = np.lib.format.read_magic(member)
                if version == (1, 0):
                    shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
                        member, max_header_size=64 * 1024
                    )
                elif version == (2, 0):
                    shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(
                        member, max_header_size=64 * 1024
                    )
                else:
                    raise ValueError("waveform evidence uses an unsupported NPY format version")
                if (
                    fortran_order
                    or not shape
                    or len(shape) > 8
                    or any(
                        isinstance(size, bool) or not isinstance(size, int) or size < 1
                        for size in shape
                    )
                ):
                    raise ValueError("waveform evidence NPY shape or memory order is invalid")
                if dtype.hasobject or dtype.kind not in "iufc":
                    raise ValueError("waveform evidence contains an unsupported NPY dtype")
                payload_bytes = math.prod(shape) * dtype.itemsize
                if payload_bytes != info.file_size - member.tell():
                    raise ValueError("waveform evidence NPY shape disagrees with its member size")
                metadata[Path(info.filename).stem] = (tuple(shape), dtype)
    return metadata


def _recompute_waveform_diagnostics(
    config: dict[str, Any],
    records: list[dict[str, Any]],
    evidence_bytes: bytes,
    *,
    snr_index: int | None = None,
) -> tuple[float, float]:
    """Rebuild the declared chain with NumPy from retained raw input/output samples."""
    if len(evidence_bytes) > _MAX_WAVEFORM_EVIDENCE_BYTES:
        raise ValueError("waveform evidence exceeds the configured size limit")
    manifest = config.get("waveform_evidence")
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema")
        not in {
            "umcd.g5_ordered_impairments_waveform_evidence.v1",
            "umcd.g5_ordered_impairments_waveform_evidence.v2",
            "umcd.g5_ordered_impairments_waveform_evidence.v3",
            "umcd.g5_ordered_impairments_waveform_evidence.v4",
        }
        or not isinstance(manifest.get("file_name"), str)
        or Path(manifest.get("file_name", "")).name != manifest.get("file_name")
        or "\\" in manifest.get("file_name", "")
        or not manifest.get("file_name", "").endswith(".npz")
        or manifest.get("sha256") != hashlib.sha256(evidence_bytes).hexdigest()
        or manifest.get("byte_count") != len(evidence_bytes)
        or manifest.get("channel_count") != len(records)
    ):
        raise ValueError("waveform evidence does not match its hash-bound manifest")
    required_arrays = {
        "channel_seeds",
        "tx_waveform",
        "tx_phase_noise_rad",
        "rx_phase_noise_rad",
        "actual_capture",
    }
    if manifest["schema"] == "umcd.g5_ordered_impairments_waveform_evidence.v2":
        required_arrays.update({"awgn_standard_normal_real", "awgn_standard_normal_imag"})
    if manifest["schema"] == "umcd.g5_ordered_impairments_waveform_evidence.v3":
        required_arrays.update(
            {
                "awgn_standard_normal_real",
                "awgn_standard_normal_imag",
                "clean_receiver_input",
                "uncorrected_receiver_input",
                "corrected_receiver_input",
            }
        )
    if manifest["schema"] == "umcd.g5_ordered_impairments_waveform_evidence.v4":
        required_arrays.update(
            {
                "awgn_standard_normal_real",
                "awgn_standard_normal_imag",
                "clean_receiver_input_by_snr",
                "uncorrected_receiver_input_by_snr",
                "corrected_receiver_input_by_snr",
            }
        )
    if manifest.get("arrays") != sorted(required_arrays):
        raise ValueError("waveform evidence array roster is invalid")
    _preflight_waveform_archive(evidence_bytes, required_arrays=required_arrays)
    with np.load(io.BytesIO(evidence_bytes), allow_pickle=False) as archive:
        if set(archive.files) != required_arrays:
            raise ValueError("waveform evidence archive has an unexpected array roster")
        # NpzFile materializes independent arrays as members are accessed; an
        # additional copy here could double the accepted 1-GiB decompressed
        # payload peak without adding isolation from the closed ZIP reader.
        arrays = {name: archive[name] for name in required_arrays}
    seeds = arrays["channel_seeds"]
    if seeds.dtype.kind not in "iu" or seeds.shape != (len(records),):
        raise ValueError("waveform evidence channel seed array is invalid")
    if [int(value) for value in seeds] != [record.get("channel_seed") for record in records]:
        raise ValueError("waveform evidence channel roster disagrees with the result")
    tx = arrays["tx_waveform"]
    tx_phase = arrays["tx_phase_noise_rad"]
    rx_phase = arrays["rx_phase_noise_rad"]
    actual = arrays["actual_capture"]
    if (
        tx.ndim != 4
        or tx.shape[0] != len(records)
        or not np.iscomplexobj(tx)
        or tx_phase.shape != tx.shape
        or tx_phase.dtype.kind != "f"
        or rx_phase.shape != tx.shape
        or rx_phase.dtype.kind != "f"
        or actual.shape != tx.shape
        or not np.iscomplexobj(actual)
        or any(not np.isfinite(value).all() for value in (tx, tx_phase, rx_phase, actual))
    ):
        raise ValueError("waveform evidence arrays have invalid dtype, shape, or values")
    condition_inputs: dict[str, np.ndarray] | None = None
    if manifest["schema"] in {
        "umcd.g5_ordered_impairments_waveform_evidence.v2",
        "umcd.g5_ordered_impairments_waveform_evidence.v3",
        "umcd.g5_ordered_impairments_waveform_evidence.v4",
    }:
        awgn_real = arrays["awgn_standard_normal_real"]
        awgn_imag = arrays["awgn_standard_normal_imag"]
        variance = config.get("waveform_awgn_complex_variance")
        if (
            awgn_real.shape != tx.shape
            or awgn_imag.shape != tx.shape
            or awgn_real.dtype.kind != "f"
            or awgn_imag.dtype.kind != "f"
            or not np.isfinite(awgn_real).all()
            or not np.isfinite(awgn_imag).all()
            or not _is_finite_number(variance)
            or float(variance) <= 0
        ):
            raise ValueError("retained AWGN draws or variance are invalid")
        scale = np.float32(math.sqrt(float(variance) / 2.0))
        awgn = np.empty(tx.shape, dtype=np.complex64)
        awgn.real = awgn_real * scale
        awgn.imag = awgn_imag * scale
        receiver_input = actual + awgn
        computed_hashes = [
            hashlib.sha256(np.ascontiguousarray(channel).tobytes()).hexdigest()
            for channel in receiver_input
        ]
        if config.get("noisy_receiver_input_sha256_by_channel") != computed_hashes:
            raise ValueError("reconstructed noisy receiver inputs differ from retained digests")
        if manifest["schema"] == "umcd.g5_ordered_impairments_waveform_evidence.v3":
            condition_inputs = {
                "clean": arrays["clean_receiver_input"],
                "uncorrected": arrays["uncorrected_receiver_input"],
                "corrected": arrays["corrected_receiver_input"],
            }
        elif manifest["schema"] == "umcd.g5_ordered_impairments_waveform_evidence.v4":
            snr_values = manifest.get("snr_db")
            if (
                not isinstance(snr_values, list)
                or not snr_values
                or isinstance(manifest.get("snr_point_count"), bool)
                or not isinstance(manifest.get("snr_point_count"), int)
                or manifest.get("snr_point_count") != len(snr_values)
                or snr_index is None
                or isinstance(snr_index, bool)
                or not isinstance(snr_index, int)
                or not 0 <= snr_index < len(snr_values)
                or isinstance(snr_values[snr_index], bool)
                or not _is_finite_number(snr_values[snr_index])
                or not _is_finite_number(config.get("nominal_snr_db"))
                or float(config["nominal_snr_db"]) != snr_values[snr_index]
            ):
                raise ValueError("multi-SNR waveform evidence point binding is invalid")
            condition_arrays = {
                "clean": arrays["clean_receiver_input_by_snr"],
                "uncorrected": arrays["uncorrected_receiver_input_by_snr"],
                "corrected": arrays["corrected_receiver_input_by_snr"],
            }
            if any(
                values.shape != (len(snr_values), *tx.shape) for values in condition_arrays.values()
            ):
                raise ValueError("multi-SNR condition input has an invalid grid shape")
            condition_inputs = {
                condition: values[snr_index] for condition, values in condition_arrays.items()
            }
        if condition_inputs is not None:
            input_hashes = config.get("receiver_input_sha256_by_condition")
            if not isinstance(input_hashes, dict) or set(input_hashes) != set(condition_inputs):
                raise ValueError("condition receiver-input digest roster is invalid")
            for condition, values in condition_inputs.items():
                if (
                    values.shape != tx.shape
                    or not np.iscomplexobj(values)
                    or not np.isfinite(values).all()
                ):
                    raise ValueError("condition receiver-input waveform is invalid")
                hashes = [
                    hashlib.sha256(np.ascontiguousarray(channel).tobytes()).hexdigest()
                    for channel in values
                ]
                if input_hashes.get(condition) != hashes:
                    raise ValueError(f"{condition} receiver inputs differ from retained digests")
            if computed_hashes != input_hashes.get("uncorrected"):
                raise ValueError("uncorrected input digest disagrees with reconstructed waveform")
    sample_rate = config.get("sample_rate_hz")
    cfo = config.get("cfo_hz")
    timing = config.get("timing_offset_samples")
    if (
        isinstance(sample_rate, bool)
        or not isinstance(sample_rate, (int, float))
        or not math.isfinite(float(sample_rate))
        or sample_rate <= 0
        or isinstance(cfo, bool)
        or not isinstance(cfo, (int, float))
        or not math.isfinite(float(cfo))
        or isinstance(timing, bool)
        or not isinstance(timing, int)
        or not 0 < timing < tx.shape[-1]
    ):
        raise ValueError("waveform evidence configuration is invalid")

    sample_axis = np.arange(tx.shape[-1], dtype=np.float64) / float(sample_rate)
    cfo_rotation = np.exp(2j * np.pi * float(cfo) * sample_axis)
    reference_maxima: list[float] = []
    wrong_order_rmses: list[float] = []
    for index, record in enumerate(records):
        coefficients = record.get("path_coefficients_real_imag")
        delays = record.get("path_delays_samples")
        if (
            not isinstance(coefficients, list)
            or not isinstance(delays, list)
            or len(coefficients) != 2
            or len(delays) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in delays
            )
        ):
            raise ValueError("per-channel path declaration is invalid")
        tx_iq = _iq_imbalance_reference(
            tx[index],
            gain_imbalance_db=float(config["tx_iq_gain_imbalance_db"]),
            phase_imbalance_rad=float(config["tx_iq_phase_imbalance_rad"]),
        )
        tx_oscillator = tx_iq * np.exp(1j * tx_phase[index])
        propagated = _path_sum_reference(tx_oscillator, coefficients, delays)
        expected = _iq_imbalance_reference(
            propagated * cfo_rotation * np.exp(1j * rx_phase[index]),
            gain_imbalance_db=float(config["rx_iq_gain_imbalance_db"]),
            phase_imbalance_rad=float(config["rx_iq_phase_imbalance_rad"]),
        )
        expected = _positive_integer_delay_reference(expected, timing)

        if manifest["schema"] in {
            "umcd.g5_ordered_impairments_waveform_evidence.v3",
            "umcd.g5_ordered_impairments_waveform_evidence.v4",
        }:
            if condition_inputs is None:
                raise ValueError("condition receiver-input evidence is unavailable")
            clean_channel = _path_sum_reference(tx[index], coefficients, delays)
            clean_capture = _positive_integer_delay_reference(clean_channel, timing) + awgn[index]
            expected_clean = np.zeros_like(clean_capture)
            expected_clean[..., :-timing] = clean_capture[..., timing:]
            if not np.allclose(
                condition_inputs["clean"][index],
                expected_clean,
                rtol=2e-5,
                atol=2e-5,
            ):
                raise ValueError("retained clean receiver input disagrees with the declared chain")

            timing_errors = record.get("timing_error_samples")
            cfo_errors = record.get("cfo_error_hz")
            if (
                not isinstance(timing_errors, list)
                or not timing_errors
                or not isinstance(cfo_errors, list)
                or not cfo_errors
                or any(value != timing_errors[0] for value in timing_errors)
                or any(value != cfo_errors[0] for value in cfo_errors)
            ):
                raise ValueError("receiver synchronization estimates are not channel-shared")
            acquired_timing = timing + int(timing_errors[0])
            if abs(acquired_timing) >= tx.shape[-1]:
                raise ValueError("acquired timing correction exceeds the waveform length")
            aligned = np.zeros_like(receiver_input[index])
            if acquired_timing >= 0:
                aligned[..., : tx.shape[-1] - acquired_timing] = receiver_input[
                    index, ..., acquired_timing:
                ]
            else:
                advance = -acquired_timing
                aligned[..., advance:] = receiver_input[index, ..., : tx.shape[-1] - advance]
            estimated_cfo = float(cfo) + float(cfo_errors[0])
            time_s = np.arange(tx.shape[-1], dtype=np.float32) / np.float32(sample_rate)
            phase = (2.0 * math.pi * np.float32(-estimated_cfo)) * time_s
            corrected_reference = aligned * np.exp(1j * phase)
            if not np.allclose(
                condition_inputs["corrected"][index],
                corrected_reference,
                rtol=2e-5,
                atol=2e-5,
            ):
                raise ValueError(
                    "retained corrected receiver input disagrees with timing/CFO correction"
                )

        wrong_order = tx_oscillator * np.exp(1j * rx_phase[index])
        wrong_order = _path_sum_reference(wrong_order, coefficients, delays)
        wrong_order = _iq_imbalance_reference(
            wrong_order * cfo_rotation,
            gain_imbalance_db=float(config["rx_iq_gain_imbalance_db"]),
            phase_imbalance_rad=float(config["rx_iq_phase_imbalance_rad"]),
        )
        wrong_order = _positive_integer_delay_reference(wrong_order, timing)
        reference_maxima.append(float(np.max(np.abs(actual[index] - expected))))
        wrong_order_rmses.append(float(np.sqrt(np.mean(np.abs(actual[index] - wrong_order) ** 2))))
    return max(reference_maxima), min(wrong_order_rmses)


def _cluster_bootstrap_interval(
    values: list[float], *, seed: int, draws: int, alpha: float = 0.05
) -> tuple[float, float]:
    rng = random.Random(seed)
    count = len(values)
    samples = sorted(
        sum(values[rng.randrange(count)] for _ in range(count)) / count for _ in range(draws)
    )
    lower = math.floor((alpha / 2.0) * draws)
    upper = min(draws - 1, math.ceil((1.0 - alpha / 2.0) * draws) - 1)
    return samples[lower], samples[upper]


def _audit_result(
    result: dict[str, Any],
    source_bytes: bytes,
    committed_source: bytes | None,
    replayed_experiment: dict[str, Any] | None = None,
    waveform_evidence_bytes: bytes | None = None,
    estimator_source_bytes: bytes | None = None,
    committed_estimator_source: bytes | None = None,
) -> dict[str, Any]:
    checks: dict[str, bool | None] = {}
    checks["schema"] = result.get("schema") == _RESULT_SCHEMA
    provenance = result.get("provenance")
    experiment = result.get("experiment")
    if not isinstance(provenance, dict) or not isinstance(experiment, dict):
        raise ValueError("result must contain provenance and experiment objects")
    config = experiment.get("config")
    acceptance = experiment.get("acceptance")
    if not isinstance(config, dict) or not isinstance(acceptance, dict):
        raise ValueError("experiment config and acceptance must be objects")

    unhashed = dict(result)
    supplied_result_digest = unhashed.pop("result_sha256", None)
    checks["result_digest"] = supplied_result_digest == _sha256(_canonical_json_bytes(unhashed))
    checks["source_digest_bindings"] = result.get("source_sha256") == provenance.get(
        "source_sha256"
    )
    checks["source_worktree_digest"] = provenance.get("source_sha256") == _sha256(source_bytes)
    estimator = config.get("channel_estimator", {"name": "sionna_default_ls"})
    if not isinstance(estimator, dict):
        raise ValueError("channel_estimator must be a provenance object")
    checks.update(
        _estimator_source_checks(
            estimator,
            source_bytes=estimator_source_bytes,
            committed_bytes=committed_estimator_source,
        )
    )
    checks["configuration_digest"] = result.get("configuration_sha256") == _sha256(
        _canonical_json_bytes(config)
    )
    protocol = provenance.get("protocol")
    expected_protocol = {
        "experiment_schema": experiment.get("schema"),
        "config": config,
        "acceptance_criteria": acceptance.get("criteria_frozen_before_run"),
    }
    checks["protocol_digest"] = protocol == expected_protocol and provenance.get(
        "protocol_sha256"
    ) == _sha256(_canonical_json_bytes(expected_protocol))
    checks["committed_source_digest"] = (
        None
        if committed_source is None
        else provenance.get("source_sha256") == _sha256(committed_source)
    )
    checks["deterministic_replay"] = (
        None if replayed_experiment is None else replayed_experiment == experiment
    )

    blocks_per_channel = config.get("blocks_per_channel")
    seeds = config.get("channel_seeds")
    records = config.get("per_channel_draws")
    if (
        isinstance(blocks_per_channel, bool)
        or not isinstance(blocks_per_channel, int)
        or blocks_per_channel < 1
        or not isinstance(seeds, list)
        or not seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or len(seeds) != len(set(seeds))
        or not isinstance(records, list)
        or len(records) != len(seeds)
    ):
        raise ValueError("invalid independent-channel and block dimensions")
    if any(not isinstance(record, dict) for record in records):
        raise ValueError("per-channel records must be objects")
    if [record.get("channel_seed") for record in records] != seeds:
        raise ValueError("per-channel records do not follow the declared unique seed roster")
    per_channel_diagnostic_fields = (
        "samplewise_reference_max_absolute_error",
        "wrong_order_waveform_rmse",
    )
    diagnostic_field_presence = [
        field in record for record in records for field in per_channel_diagnostic_fields
    ]
    if any(diagnostic_field_presence) and not all(diagnostic_field_presence):
        raise ValueError("per-channel waveform diagnostics must be present as a complete roster")
    has_per_channel_waveform_diagnostics = all(diagnostic_field_presence)
    report_schema = experiment.get("schema")
    if (
        report_schema in _PER_CHANNEL_WAVEFORM_DIAGNOSTIC_SCHEMAS
        and not has_per_channel_waveform_diagnostics
    ):
        raise ValueError("report schema requires per-channel waveform diagnostics")

    if any(
        not isinstance(record.get("crc_pass_by_condition"), dict)
        or any(
            not isinstance(record["crc_pass_by_condition"].get(condition), list)
            or len(record["crc_pass_by_condition"][condition]) != blocks_per_channel
            or any(
                not isinstance(value, bool) for value in record["crc_pass_by_condition"][condition]
            )
            for condition in ("clean", "corrected", "uncorrected")
        )
        or not isinstance(record.get("cfo_error_hz"), list)
        or len(record["cfo_error_hz"]) != blocks_per_channel
        or any(not _is_finite_number(value) for value in record.get("cfo_error_hz", []))
        or not isinstance(record.get("timing_error_samples"), list)
        or len(record["timing_error_samples"]) != blocks_per_channel
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in record.get("timing_error_samples", [])
        )
        or (
            has_per_channel_waveform_diagnostics
            and (
                not _is_finite_nonnegative(record.get("samplewise_reference_max_absolute_error"))
                or not _is_finite_nonnegative(record.get("wrong_order_waveform_rmse"))
            )
        )
        for record in records
    ):
        raise ValueError("per-channel raw records do not match blocks_per_channel")

    if experiment.get("schema") in _BLOCK_BOUND_REPORT_SCHEMAS:
        decoder = config.get("decoder_configuration")
        if decoder != {
            "api": "SionnaPUSCHLSReceiver.decode_pusch",
            "channel_estimator": estimator,
            "crc_status_semantics": "true_means_transport_block_crc_passed",
        }:
            raise ValueError("decoder_configuration disagrees with the selected estimator contract")
        receiver = config.get("receiver_configuration")
        if not isinstance(receiver, dict):
            raise ValueError("new receiver reports require a complete receiver_configuration")
        execution_device = (
            config.get("execution_device", "cpu")
            if experiment.get("schema") in _CUDA_DEVICE_REPORT_SCHEMAS
            else "cpu"
        )
        expected_receiver_values = {
            "n_size_bwp": 6,
            "subcarrier_spacing_khz": 30,
            "num_antenna_ports": 1,
            "num_layers": 1,
            "n_cell_id": 1,
            "dmrs_n_id": None,
            "dmrs_n_scid": 0,
            "dmrs_config_type": 1,
            "dmrs_length": 1,
            "dmrs_additional_position": 0,
            "num_cdm_groups_without_data": 1,
            "mapping_type": "A",
            "symbol_allocation": [0, 14],
            "precoding": "non-codebook",
            "interpolation_type": "lin",
            "n_size_grid": 6,
            "n_start_grid": 0,
            "n_start_bwp": 0,
            "receiver": "PUSCHLSChannelEstimator",
            "device": execution_device,
            "error_variance_semantics": (
                "sionna_linear_interpolated_marginal_variance_not_exact_covariance"
            ),
            "resource_grid": {
                "num_ofdm_symbols": 14,
                "fft_size": 72,
                "subcarrier_spacing_hz": 30_000,
            },
            "carrier_grid": {
                "n_size_grid": 6,
                "n_start_grid": 0,
                "n_start_bwp": 0,
                "bwp_start_rb_in_grid": 0,
                "fft_size": 72,
            },
        }
        if any(receiver.get(key) != value for key, value in expected_receiver_values.items()):
            raise ValueError("receiver_configuration disagrees with the frozen CPU PUSCH fixture")
        runtime = provenance.get("runtime")
        runtime_packages = runtime.get("packages") if isinstance(runtime, dict) else None
        if (
            not isinstance(runtime_packages, dict)
            or not isinstance(receiver.get("sionna_version"), str)
            or not receiver.get("sionna_version")
            or receiver.get("sionna_version") != runtime_packages.get("sionna")
        ):
            raise ValueError("receiver Sionna version disagrees with runtime provenance")
        if experiment.get("schema") in _CUDA_DEVICE_REPORT_SCHEMAS:
            runtime = provenance.get("runtime")
            if (
                not isinstance(execution_device, str)
                or not execution_device.startswith("cuda:")
                or not isinstance(runtime, dict)
                or runtime.get("torch_device") != execution_device
            ):
                raise ValueError("CUDA execution device disagrees with receiver/runtime provenance")

        for seed, record in zip(seeds, records, strict=True):
            blocks = record.get("transport_blocks")
            payload_seed = record.get("payload_seed")
            if (
                isinstance(payload_seed, bool)
                or not isinstance(payload_seed, int)
                or payload_seed != seed + 1_000_000
                or not isinstance(blocks, list)
                or len(blocks) != blocks_per_channel
            ):
                raise ValueError("transport-block roster does not match channel and payload seeds")
            for index, block in enumerate(blocks):
                expected_id = f"channel-{seed}:payload-{payload_seed}:block-{index}"
                if (
                    not isinstance(block, dict)
                    or isinstance(block.get("block_index"), bool)
                    or not isinstance(block.get("block_index"), int)
                    or block.get("transport_block_id") != expected_id
                    or block.get("block_index") != index
                    or isinstance(block.get("payload_seed"), bool)
                    or not isinstance(block.get("payload_seed"), int)
                    or block.get("payload_seed") != payload_seed
                    or not isinstance(block.get("crc_pass_by_condition"), dict)
                    or any(
                        not isinstance(block["crc_pass_by_condition"].get(condition), bool)
                        for condition in ("clean", "corrected", "uncorrected")
                    )
                    or block.get("crc_pass_by_condition")
                    != {
                        condition: record["crc_pass_by_condition"][condition][index]
                        for condition in ("clean", "corrected", "uncorrected")
                    }
                    or block.get("timing_error_samples") != record["timing_error_samples"][index]
                    or isinstance(block.get("timing_error_samples"), bool)
                    or not isinstance(block.get("timing_error_samples"), int)
                    or isinstance(block.get("cfo_error_hz"), bool)
                    or not _is_finite_number(block.get("cfo_error_hz"))
                    or not _same_float(block.get("cfo_error_hz"), record["cfo_error_hz"][index])
                ):
                    raise ValueError(
                        "transport-block identity or outcome disagrees with channel record"
                    )
        checks["receiver_configuration_binding"] = True
        checks["transport_block_roster_binding"] = True
    else:
        checks["receiver_configuration_binding"] = None
        checks["transport_block_roster_binding"] = None

    flat_cfo = [float(value) for record in records for value in record["cfo_error_hz"]]
    flat_timing = [int(value) for record in records for value in record["timing_error_samples"]]
    checks["flattened_block_arrays"] = (
        experiment.get("cfo_error_hz_by_block") == flat_cfo
        and experiment.get("timing_error_samples_by_block") == flat_timing
    )
    outcomes = experiment.get("paired_outcomes")
    if not isinstance(outcomes, dict):
        raise ValueError("paired_outcomes must be an object")
    condition_rates: dict[str, list[float]] = {}
    total_blocks = len(seeds) * blocks_per_channel
    outcomes_consistent = True
    for condition in ("clean", "corrected", "uncorrected"):
        pass_vectors = [record["crc_pass_by_condition"][condition] for record in records]
        failures_by_channel = [blocks_per_channel - sum(values) for values in pass_vectors]
        rates = [failures / blocks_per_channel for failures in failures_by_channel]
        condition_rates[condition] = rates
        failed = sum(failures_by_channel)
        summary = outcomes.get(condition)
        if not isinstance(summary, dict):
            outcomes_consistent = False
            continue
        outcomes_consistent &= (
            summary.get("blocks") == total_blocks
            and summary.get("crc_failures") == failed
            and _same_float(summary.get("bler"), failed / total_blocks)
            and _same_float(summary.get("channel_macro_bler"), sum(rates) / len(rates))
        )
    checks["crc_and_bler_summaries"] = outcomes_consistent

    timing_limit = acceptance.get("criteria_frozen_before_run", {})
    observed = acceptance.get("observed", {})
    absolute_cfo = sorted(abs(value) for value in flat_cfo)
    p95_index = math.ceil(0.95 * len(absolute_cfo)) - 1
    recomputed = {
        "max_abs_timing_error_samples": max(abs(value) for value in flat_timing),
        "p95_abs_cfo_error_hz": absolute_cfo[p95_index],
        "max_abs_cfo_error_hz": absolute_cfo[-1],
        "clean_bler": outcomes["clean"]["bler"],
        "impaired_corrected_bler": outcomes["corrected"]["bler"],
    }
    if has_per_channel_waveform_diagnostics:
        recomputed["max_samplewise_reference_absolute_error"] = max(
            float(record["samplewise_reference_max_absolute_error"]) for record in records
        )
        recomputed["min_wrong_order_rmse"] = min(
            float(record["wrong_order_waveform_rmse"]) for record in records
        )
        checks["waveform_diagnostic_extrema"] = all(
            observed.get(key) == recomputed[key]
            for key in (
                "max_samplewise_reference_absolute_error",
                "min_wrong_order_rmse",
            )
        )
    else:
        for key in (
            "max_samplewise_reference_absolute_error",
            "min_wrong_order_rmse",
        ):
            if not _is_finite_nonnegative(observed.get(key)):
                raise ValueError(
                    "legacy reported waveform diagnostics must be finite and nonnegative"
                )
            recomputed[key] = observed[key]
        checks["waveform_diagnostic_extrema"] = None
    checks["retained_observed_statistics"] = all(
        observed.get(key) == value
        for key, value in recomputed.items()
        if has_per_channel_waveform_diagnostics
        or key not in {"max_samplewise_reference_absolute_error", "min_wrong_order_rmse"}
    )
    if config.get("waveform_evidence") is None:
        checks["waveform_equation_recomputation"] = (
            None
            if experiment.get("schema")
            not in {
                "umcd.g5_ordered_impairments_independent_channels.v8",
                "umcd.g5_ordered_impairments_independent_channels.v10",
                "umcd.g5_ordered_impairments_independent_channels.v12",
                "umcd.g5_ordered_impairments_independent_channels.v14",
                "umcd.g5_ordered_impairments_independent_channels.v16",
                "umcd.g5_ordered_impairments_independent_channels.v18",
                "umcd.g5_ordered_impairments_independent_channels.v19",
                "umcd.g5_ordered_impairments_independent_channels.v20",
                "umcd.g5_ordered_impairments_independent_channels.v21",
                "umcd.g5_ordered_impairments_independent_channels.v22",
                "umcd.g5_ordered_impairments_independent_channels.v23",
                "umcd.g5_ordered_impairments_independent_channels.v24",
            }
            else False
        )
    elif waveform_evidence_bytes is None:
        checks["waveform_equation_recomputation"] = False
    else:
        try:
            waveform_maximum, wrong_order_minimum = _recompute_waveform_diagnostics(
                config, records, waveform_evidence_bytes
            )
            checks["waveform_equation_recomputation"] = (
                abs(
                    waveform_maximum
                    - float(observed.get("max_samplewise_reference_absolute_error", math.inf))
                )
                <= 5e-6
                and abs(
                    wrong_order_minimum - float(observed.get("min_wrong_order_rmse", -math.inf))
                )
                <= 5e-6
                and waveform_maximum
                < float(timing_limit.get("max_samplewise_reference_absolute_error", -math.inf))
                and wrong_order_minimum > float(timing_limit.get("min_wrong_order_rmse", math.inf))
            )
        except (
            OSError,
            EOFError,
            ValueError,
            TypeError,
            KeyError,
            OverflowError,
            zipfile.BadZipFile,
        ):
            checks["waveform_equation_recomputation"] = False

    inference = acceptance.get("inference", {})
    paired_deltas = [
        corrected - clean
        for clean, corrected in zip(
            condition_rates["clean"], condition_rates["corrected"], strict=True
        )
    ]
    bootstrap_interval = _cluster_bootstrap_interval(
        paired_deltas,
        seed=inference.get("bootstrap_seed"),
        draws=inference.get("bootstrap_draws"),
    )
    expected_mean_delta = sum(paired_deltas) / len(paired_deltas)
    interval = inference.get("paired_corrected_minus_clean_channel_bootstrap_95_percent_interval")
    checks["paired_cluster_inference"] = (
        _same_float(inference.get("paired_corrected_minus_clean_mean_bler"), expected_mean_delta)
        and isinstance(interval, list)
        and len(interval) == 2
        and _same_float(interval[0], bootstrap_interval[0])
        and _same_float(interval[1], bootstrap_interval[1])
    )

    if experiment.get("schema") in _CFO_CHANNEL_TAIL_SCHEMAS:
        tails = inference.get("cfo_channel_tail_diagnostics")
        expected_tails = []
        for threshold in (
            timing_limit["p95_abs_cfo_error_hz"],
            timing_limit["max_abs_cfo_error_hz"],
        ):
            exceedances = sum(
                any(abs(float(value)) >= threshold for value in record["cfo_error_hz"])
                for record in records
            )
            upper = (
                1.0
                if exceedances == len(records)
                else float(beta.ppf(0.95, exceedances + 1, len(records) - exceedances))
            )
            expected_tails.append((threshold, exceedances, upper))
        checks["cfo_cluster_tail_bounds"] = (
            isinstance(tails, list)
            and len(tails) == len(expected_tails)
            and all(
                row.get("absolute_error_threshold_hz") == threshold
                and row.get("channels_with_any_block_exceedance") == exceedances
                and row.get("independent_channels") == len(records)
                and _same_float(
                    row.get("exact_one_sided_95_percent_channel_prevalence_upper_bound"), upper
                )
                for row, (threshold, exceedances, upper) in zip(tails, expected_tails, strict=True)
            )
        )
    else:
        checks["cfo_cluster_tail_bounds"] = None

    crc_channel_tail_diagnostics = []
    for condition in ("clean", "corrected", "uncorrected"):
        failed_channels = sum(
            not all(record["crc_pass_by_condition"][condition]) for record in records
        )
        upper = (
            1.0
            if failed_channels == len(records)
            else float(beta.ppf(0.95, failed_channels + 1, len(records) - failed_channels))
        )
        crc_channel_tail_diagnostics.append(
            {
                "condition": condition,
                "channels_with_any_block_failure": failed_channels,
                "independent_channels": len(records),
                "blocks_per_channel": blocks_per_channel,
                "exact_one_sided_95_percent_channel_prevalence_upper_bound": upper,
            }
        )
    if experiment.get("schema") in {
        "umcd.g5_ordered_impairments_independent_channels.v11",
        "umcd.g5_ordered_impairments_independent_channels.v12",
        "umcd.g5_ordered_impairments_independent_channels.v13",
        "umcd.g5_ordered_impairments_independent_channels.v14",
        "umcd.g5_ordered_impairments_independent_channels.v15",
        "umcd.g5_ordered_impairments_independent_channels.v16",
        "umcd.g5_ordered_impairments_independent_channels.v17",
        "umcd.g5_ordered_impairments_independent_channels.v18",
        "umcd.g5_ordered_impairments_independent_channels.v19",
        "umcd.g5_ordered_impairments_independent_channels.v20",
        "umcd.g5_ordered_impairments_independent_channels.v21",
        "umcd.g5_ordered_impairments_independent_channels.v22",
        "umcd.g5_ordered_impairments_independent_channels.v23",
        "umcd.g5_ordered_impairments_independent_channels.v24",
    }:
        retained_crc_tails = inference.get("crc_channel_failure_diagnostics")
        checks["crc_channel_cluster_tail_bounds"] = (
            isinstance(retained_crc_tails, list)
            and len(retained_crc_tails) == len(crc_channel_tail_diagnostics)
            and all(
                isinstance(row, dict)
                and row.get("condition") == expected["condition"]
                and row.get("channels_with_any_block_failure")
                == expected["channels_with_any_block_failure"]
                and row.get("independent_channels") == expected["independent_channels"]
                and row.get("blocks_per_channel") == expected["blocks_per_channel"]
                and _same_float(
                    row.get("exact_one_sided_95_percent_channel_prevalence_upper_bound"),
                    expected["exact_one_sided_95_percent_channel_prevalence_upper_bound"],
                )
                for row, expected in zip(
                    retained_crc_tails, crc_channel_tail_diagnostics, strict=True
                )
            )
        )
    else:
        checks["crc_channel_cluster_tail_bounds"] = None

    checks["acceptance_verdict"] = acceptance.get("pass") == (
        recomputed["max_abs_timing_error_samples"] <= timing_limit["max_abs_timing_error_samples"]
        and recomputed["p95_abs_cfo_error_hz"] < timing_limit["p95_abs_cfo_error_hz"]
        and recomputed["max_abs_cfo_error_hz"] < timing_limit["max_abs_cfo_error_hz"]
        and recomputed["max_samplewise_reference_absolute_error"]
        < timing_limit["max_samplewise_reference_absolute_error"]
        and recomputed["min_wrong_order_rmse"] > timing_limit["min_wrong_order_rmse"]
        and recomputed["clean_bler"] <= timing_limit["max_clean_reference_bler"]
        and recomputed["impaired_corrected_bler"] <= timing_limit["max_impaired_corrected_bler"]
    )

    optional_checks = {
        "committed_source_digest",
        "estimator_committed_source_digest",
        "source_worktree_digest",
        "deterministic_replay",
        "waveform_equation_recomputation",
        "crc_channel_cluster_tail_bounds",
        "receiver_configuration_binding",
        "transport_block_roster_binding",
        "waveform_diagnostic_extrema",
    }
    all_pass = all(value is True for key, value in checks.items() if key not in optional_checks)
    all_pass &= checks["deterministic_replay"] is not False
    if experiment.get("schema") in {
        "umcd.g5_ordered_impairments_independent_channels.v8",
        "umcd.g5_ordered_impairments_independent_channels.v10",
        "umcd.g5_ordered_impairments_independent_channels.v12",
        "umcd.g5_ordered_impairments_independent_channels.v14",
        "umcd.g5_ordered_impairments_independent_channels.v16",
        "umcd.g5_ordered_impairments_independent_channels.v18",
        "umcd.g5_ordered_impairments_independent_channels.v19",
        "umcd.g5_ordered_impairments_independent_channels.v20",
        "umcd.g5_ordered_impairments_independent_channels.v21",
        "umcd.g5_ordered_impairments_independent_channels.v22",
        "umcd.g5_ordered_impairments_independent_channels.v23",
        "umcd.g5_ordered_impairments_independent_channels.v24",
    }:
        all_pass &= checks["waveform_equation_recomputation"] is True
    if experiment.get("schema") in {
        "umcd.g5_ordered_impairments_independent_channels.v11",
        "umcd.g5_ordered_impairments_independent_channels.v12",
        "umcd.g5_ordered_impairments_independent_channels.v13",
        "umcd.g5_ordered_impairments_independent_channels.v14",
        "umcd.g5_ordered_impairments_independent_channels.v15",
        "umcd.g5_ordered_impairments_independent_channels.v16",
        "umcd.g5_ordered_impairments_independent_channels.v17",
        "umcd.g5_ordered_impairments_independent_channels.v18",
        "umcd.g5_ordered_impairments_independent_channels.v19",
        "umcd.g5_ordered_impairments_independent_channels.v20",
    }:
        all_pass &= checks["crc_channel_cluster_tail_bounds"] is True
    source_worktree_matches = checks["source_worktree_digest"]
    committed_matches = checks["committed_source_digest"]
    status = (
        "AUDIT_FAILED"
        if committed_matches is False
        else "VERIFIED_AGAINST_COMMITTED_SOURCE"
        if all_pass and committed_matches is True
        else "VERIFIED_RESULT_AND_WORKTREE_SOURCE_PENDING_COMMIT"
        if all_pass and source_worktree_matches
        else "AUDIT_FAILED"
        if not source_worktree_matches
        else "AUDIT_FAILED"
    )
    limitations = [
        "A passing audit verifies this declared synthetic fixture only; it does not establish a field channel population, G5 closure, or production RT authorization."
    ]
    if replayed_experiment is None:
        if checks["waveform_equation_recomputation"] is not True:
            limitations.insert(
                0,
                "The artifact does not provide hash-verified raw waveform evidence for an independent recomputation of the impairment equations.",
            )
        else:
            limitations.insert(
                0,
                "The auditor independently recomputes samplewise-reference and wrong-order waveform diagnostics from the hash-verified raw samples using NumPy equations.",
            )
    if checks["waveform_diagnostic_extrema"] is None:
        limitations.insert(
            0,
            "This legacy report schema predates per-channel waveform-diagnostic retention; those two frozen scalar criteria are compared with the result's reported acceptance values but are not independently recomputed.",
        )
    else:
        limitations.insert(
            0,
            "Deterministic replay checks reproducibility with the source-bound implementation; it is not an independently implemented validation of the reference mathematics.",
        )
    return {
        "schema": "umcd.g5_ordered_impairments_result_audit.v1",
        "status": status,
        "checks": checks,
        "independent_channel_count": len(records),
        "blocks_per_channel": blocks_per_channel,
        "limitations": limitations,
    }


def _audit_independent_snr_curve(
    result: dict[str, Any],
    source_bytes: bytes,
    committed_source: bytes | None,
    waveform_evidence_bytes: bytes | None = None,
    acceptance_signature_path: Path | None = None,
    trusted_allowed_signers_path: Path | None = None,
    reviewer_policy_path: Path | None = None,
    reviewer_policy_signature_path: Path | None = None,
    trusted_reviewer_allowed_signers_path: Path | None = None,
    reviewer_policy_signer_id: str | None = None,
) -> dict[str, Any]:
    """Audit curve identity, shared roster, per-point noise, and retained CRC counts."""
    experiment = result.get("experiment")
    provenance = result.get("provenance")
    if not isinstance(experiment, dict) or not isinstance(provenance, dict):
        raise ValueError("SNR curve result must contain experiment and provenance objects")
    config = experiment.get("config")
    acceptance = experiment.get("acceptance")
    points = experiment.get("points")
    if (
        not isinstance(config, dict)
        or not isinstance(acceptance, dict)
        or not isinstance(points, list)
    ):
        raise ValueError("SNR curve config, acceptance, and points have invalid types")
    curve_schema = experiment.get("schema")
    if curve_schema not in {
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
    }:
        raise ValueError("unsupported independent-channel SNR curve schema")

    checks: dict[str, bool | None] = {}
    unhashed = dict(result)
    supplied_digest = unhashed.pop("result_sha256", None)
    checks["result_digest"] = supplied_digest == _sha256(_canonical_json_bytes(unhashed))
    checks["source_worktree_digest"] = provenance.get("source_sha256") == _sha256(source_bytes)
    checks["source_digest_bindings"] = result.get("source_sha256") == provenance.get(
        "source_sha256"
    )
    checks["configuration_digest"] = result.get("configuration_sha256") == _sha256(
        _canonical_json_bytes(config)
    )
    expected_protocol = {
        "experiment_schema": experiment.get("schema"),
        "config": config,
        "acceptance_criteria": acceptance.get("criteria_frozen_before_run"),
    }
    checks["protocol_digest"] = provenance.get("protocol") == expected_protocol and provenance.get(
        "protocol_sha256"
    ) == _sha256(_canonical_json_bytes(expected_protocol))
    curve_schema_version = curve_schema.rsplit("v", maxsplit=1)[1]
    schema_path = (
        REPOSITORY_ROOT
        / "datasets"
        / "schema"
        / f"g5-ordered-impairments-snr-curve.v{curve_schema_version}.schema.json"
    )
    schema_digest, schema_valid = _curve_report_schema_status(experiment, schema_path)
    checks["schema_digest"] = config.get("schema_sha256") == schema_digest
    checks["curve_report_schema"] = schema_valid
    checks["committed_source_digest"] = (
        None
        if committed_source is None
        else provenance.get("source_sha256") == _sha256(committed_source)
    )

    seeds = config.get("channel_seeds")
    blocks = config.get("blocks_per_channel")
    snr_grid = config.get("snr_db")
    frozen_acceptance = acceptance.get("criteria_frozen_before_run")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)
        or len(seeds) != len(set(seeds))
        or isinstance(blocks, bool)
        or not isinstance(blocks, int)
        or blocks < 1
        or not isinstance(snr_grid, list)
        or len(snr_grid) != len(points)
        or (
            curve_schema
            in {
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v2",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v3",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v4",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v5",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v6",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v7",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v8",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v9",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v10",
            }
            and (len(snr_grid) > 64 or config.get("maximum_snr_points") != 64)
        )
        or any(not _is_finite_number(value) for value in snr_grid)
        or any(right <= left for left, right in zip(snr_grid, snr_grid[1:]))
        or not isinstance(frozen_acceptance, dict)
        or (
            curve_schema
            not in {
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v5",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v6",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v7",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v8",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v10",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v11",
            }
            and (
                acceptance.get("pass") is not False
                or frozen_acceptance.get("status") != "descriptive_only"
            )
        )
        or (
            curve_schema
            in {
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v5",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v6",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v7",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v8",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v10",
            }
            and frozen_acceptance.get("status") != "contract_supplied_before_run"
        )
    ):
        raise ValueError("invalid SNR curve seed, grid, or diagnostic-only acceptance contract")
    expected_total = len(snr_grid) * len(seeds) * blocks
    checks["bounded_snr_grid"] = (
        None
        if curve_schema == "umcd.g5_ordered_impairments_independent_channels_snr_curve.v1"
        else len(snr_grid) <= 64 and config.get("maximum_snr_points") == 64
    )
    checks["bounded_total_work"] = (
        config.get("transport_blocks_per_snr_point") == len(seeds) * blocks
        and config.get("transport_blocks_across_curve") == expected_total
        and expected_total <= config.get("maximum_transport_blocks_across_curve", 0)
        and config.get("maximum_transport_blocks_across_curve") == 16_384
    )

    points_consistent = True
    shared_path_signatures: list[tuple[Any, Any]] | None = None
    for snr_index, (snr, point) in enumerate(zip(snr_grid, points, strict=True)):
        if not isinstance(point, dict) or point.get("snr_db") != snr:
            points_consistent = False
            continue
        report = point.get("experiment")
        point_config = report.get("config") if isinstance(report, dict) else None
        estimator = (
            point_config.get("channel_estimator") if isinstance(point_config, dict) else None
        )
        point_evidence = (
            point_config.get("waveform_evidence") if isinstance(point_config, dict) else None
        )
        expected_point_schema = (
            "umcd.g5_ordered_impairments_independent_channels.v20"
            if config.get("channel_estimator") == "finite_delay_dft" and point_evidence is not None
            else "umcd.g5_ordered_impairments_independent_channels.v17"
            if config.get("channel_estimator") == "finite_delay_dft"
            else "umcd.g5_ordered_impairments_independent_channels.v19"
            if point_evidence is not None
            else "umcd.g5_ordered_impairments_independent_channels.v15"
        )
        if (
            not isinstance(report, dict)
            or report.get("schema") != expected_point_schema
            or not isinstance(point_config, dict)
            or point_config.get("channel_seeds") != seeds
            or point_config.get("blocks_per_channel") != blocks
            or not isinstance(estimator, dict)
            or estimator.get("name") != config.get("channel_estimator")
        ):
            points_consistent = False
            continue
        try:
            variance = math.pow(10.0, -float(snr) / 10.0)
        except OverflowError:
            points_consistent = False
            continue
        if not _same_float(
            point_config.get("waveform_awgn_complex_variance"), variance
        ) or not _same_float(point_config.get("nominal_snr_db"), float(snr)):
            points_consistent = False
            continue
        powers = point_config.get("transmit_waveform_power_by_channel")
        receiver_input_hashes = point_config.get("noisy_receiver_input_sha256_by_channel")
        if receiver_input_hashes is not None and (
            not isinstance(receiver_input_hashes, list)
            or len(receiver_input_hashes) != len(seeds)
            or any(
                not isinstance(digest, str) or len(digest) != 64 or not set(digest) <= _HEX
                for digest in receiver_input_hashes
            )
        ):
            points_consistent = False
        if (
            not isinstance(powers, list)
            or len(powers) != len(seeds)
            or any(not _is_finite_number(power) or power <= 0 for power in powers)
            or not _same_float(
                point_config.get("mean_transmit_waveform_power"), sum(powers) / len(powers)
            )
        ):
            points_consistent = False
            continue
        records = point_config.get("per_channel_draws")
        outcomes = report.get("paired_outcomes")
        if (
            not isinstance(records, list)
            or len(records) != len(seeds)
            or not isinstance(outcomes, dict)
        ):
            points_consistent = False
            continue
        if [row.get("channel_seed") if isinstance(row, dict) else None for row in records] != seeds:
            points_consistent = False
            continue
        if (
            point_config.get("payload_seed_offset") != 1_000_000
            or point_config.get("noise_seed_offset") != 1_200_000
            or point_config.get("tx_phase_noise_seed_offset") != 1_300_000
            or point_config.get("rx_phase_noise_seed_offset") != 1_400_000
        ):
            points_consistent = False
            continue
        path_signatures: list[tuple[Any, Any]] = []
        for seed, row in zip(seeds, records, strict=True):
            expected_seeds = {
                "payload_seed": seed + 1_000_000,
                "noise_seed": seed + 1_200_000,
                "tx_phase_noise_seed": seed + 1_300_000,
                "rx_phase_noise_seed": seed + 1_400_000,
            }
            if any(row.get(key) != value for key, value in expected_seeds.items()):
                points_consistent = False
            delays = row.get("path_delays_samples")
            coefficients = row.get("path_coefficients_real_imag")
            if not isinstance(delays, list) or not isinstance(coefficients, list):
                points_consistent = False
            path_signatures.append((delays, coefficients))
            transport_blocks = row.get("transport_blocks")
            if (
                not isinstance(transport_blocks, list)
                or len(transport_blocks) != blocks
                or any(not isinstance(block, dict) for block in transport_blocks)
            ):
                points_consistent = False
            else:
                for block_index, block in enumerate(transport_blocks):
                    if (
                        not isinstance(block, dict)
                        or block.get("transport_block_id")
                        != f"channel-{seed}:payload-{seed + 1_000_000}:block-{block_index}"
                        or block.get("block_index") != block_index
                        or block.get("payload_seed") != seed + 1_000_000
                    ):
                        points_consistent = False
        if shared_path_signatures is None:
            shared_path_signatures = path_signatures
        elif path_signatures != shared_path_signatures:
            points_consistent = False
        point_blocks = len(seeds) * blocks
        for condition in ("clean", "uncorrected", "corrected"):
            vectors = [
                crc.get(condition)
                if isinstance(crc := row.get("crc_pass_by_condition"), dict)
                else None
                for row in records
            ]
            if any(
                not isinstance(vector, list)
                or len(vector) != blocks
                or any(not isinstance(value, bool) for value in vector)
                for vector in vectors
            ):
                points_consistent = False
                continue
            for channel_index, vector in enumerate(vectors):
                blocks_for_channel = records[channel_index]["transport_blocks"]
                if any(
                    block.get("crc_pass_by_condition", {}).get(condition) != vector[block_index]
                    for block_index, block in enumerate(blocks_for_channel)
                ):
                    points_consistent = False
            failures = sum(sum(not value for value in vector) for vector in vectors)
            channel_rates = [sum(not value for value in vector) / blocks for vector in vectors]
            summary = outcomes.get(condition)
            if not isinstance(summary, dict) or not (
                summary.get("blocks") == point_blocks
                and summary.get("crc_failures") == failures
                and _same_float(summary.get("bler"), failures / point_blocks)
                and _same_float(summary.get("channel_macro_bler"), sum(channel_rates) / len(seeds))
            ):
                points_consistent = False
    checks["paired_roster_noise_and_crc_summaries"] = points_consistent
    curve_inference = experiment.get("curve_inference")
    inference_consistent = isinstance(curve_inference, dict)
    if inference_consistent:
        conditions = ("clean", "uncorrected", "corrected")
        by_condition = curve_inference.get("by_condition")
        seed_roster_consistent = _inference_seed_roster_consistent(
            curve_schema,
            curve_inference,
            seeds,
        )
        has_paired_contrast = curve_schema in {
            "umcd.g5_ordered_impairments_independent_channels_snr_curve.v4",
            "umcd.g5_ordered_impairments_independent_channels_snr_curve.v5",
            "umcd.g5_ordered_impairments_independent_channels_snr_curve.v6",
            "umcd.g5_ordered_impairments_independent_channels_snr_curve.v7",
            "umcd.g5_ordered_impairments_independent_channels_snr_curve.v8",
            "umcd.g5_ordered_impairments_independent_channels_snr_curve.v9",
            "umcd.g5_ordered_impairments_independent_channels_snr_curve.v10",
            "umcd.g5_ordered_impairments_independent_channels_snr_curve.v11",
        }
        expected_inference_method = (
            "channel-cluster percentile intervals with simultaneous max-deviation band over "
            "conditions and paired correction effects"
            if has_paired_contrast
            else "channel-cluster percentile intervals with simultaneous max-deviation band"
        )
        by_paired_contrast = curve_inference.get("by_paired_contrast")
        inference_consistent = (
            curve_inference.get("cluster_count") == len(seeds)
            and seed_roster_consistent
            and curve_inference.get("family_size")
            == len(snr_grid) * (len(conditions) + int(has_paired_contrast))
            and curve_inference.get("bootstrap_seed") == G5_SNR_CURVE_BOOTSTRAP_SEED
            and curve_inference.get("bootstrap_draws") == G5_SNR_CURVE_BOOTSTRAP_DRAWS
            and curve_inference.get("unit")
            == "independent synthetic channel; transport blocks are nested"
            and curve_inference.get("method") == expected_inference_method
            and (
                curve_inference.get("paired_contrast_definition")
                == "uncorrected macro-BLER minus corrected macro-BLER; positive values favor correction"
                if has_paired_contrast
                else curve_inference.get("paired_contrast_definition") is None
            )
            and isinstance(by_condition, dict)
            and set(by_condition) == set(conditions)
            and (
                isinstance(by_paired_contrast, list)
                if has_paired_contrast
                else by_paired_contrast is None
            )
        )
        if inference_consistent and len(seeds) < 2:
            inference_consistent = (
                curve_inference.get("status") == "unavailable"
                and isinstance(curve_inference.get("reason"), str)
                and all(by_condition[condition] == [] for condition in conditions)
                and (not has_paired_contrast or by_paired_contrast == [])
            )
        elif inference_consistent:
            inference_consistent = (
                points_consistent and curve_inference.get("status") == "available"
            )
            if inference_consistent:
                reports_for_rates = [point["experiment"] for point in points]
                rates = np.empty((len(points), len(conditions), len(seeds)), dtype=np.float64)
                for snr_index, report in enumerate(reports_for_rates):
                    for condition_index, condition in enumerate(conditions):
                        for channel_index, row in enumerate(report["config"]["per_channel_draws"]):
                            vector = row["crc_pass_by_condition"][condition]
                            rates[snr_index, condition_index, channel_index] = 1.0 - (
                                sum(vector) / blocks
                            )
                observed = rates.mean(axis=2)
                rng = np.random.default_rng(G5_SNR_CURVE_BOOTSTRAP_SEED)
                bootstrap_batches: list[np.ndarray] = []
                for start in range(
                    0, G5_SNR_CURVE_BOOTSTRAP_DRAWS, G5_SNR_CURVE_BOOTSTRAP_BATCH_SIZE
                ):
                    batch_size = min(
                        G5_SNR_CURVE_BOOTSTRAP_BATCH_SIZE,
                        G5_SNR_CURVE_BOOTSTRAP_DRAWS - start,
                    )
                    sampled = rng.integers(0, len(seeds), size=(batch_size, len(seeds)))
                    sampled_means = rates[:, :, sampled].mean(axis=-1)
                    bootstrap_batches.append(np.moveaxis(sampled_means, -1, 0))
                bootstrap = np.concatenate(bootstrap_batches, axis=0)
                deviations = np.max(np.abs(bootstrap - observed[None, :, :]), axis=(1, 2))
                if has_paired_contrast:
                    paired_improvement_rates = rates[:, 1, :] - rates[:, 2, :]
                    observed_paired_improvement = paired_improvement_rates.mean(axis=1)
                    paired_bootstrap_batches: list[np.ndarray] = []
                    rng = np.random.default_rng(G5_SNR_CURVE_BOOTSTRAP_SEED)
                    for start in range(
                        0, G5_SNR_CURVE_BOOTSTRAP_DRAWS, G5_SNR_CURVE_BOOTSTRAP_BATCH_SIZE
                    ):
                        batch_size = min(
                            G5_SNR_CURVE_BOOTSTRAP_BATCH_SIZE,
                            G5_SNR_CURVE_BOOTSTRAP_DRAWS - start,
                        )
                        sampled = rng.integers(0, len(seeds), size=(batch_size, len(seeds)))
                        sampled_improvements = paired_improvement_rates[:, sampled].mean(axis=-1)
                        paired_bootstrap_batches.append(np.moveaxis(sampled_improvements, -1, 0))
                    paired_bootstrap = np.concatenate(paired_bootstrap_batches, axis=0)
                    paired_deviations = np.max(
                        np.abs(paired_bootstrap - observed_paired_improvement[None, :]),
                        axis=1,
                    )
                    deviations = np.maximum(deviations, paired_deviations)
                simultaneous_radius = float(np.quantile(deviations, 0.95, method="higher"))
                inference_consistent &= _same_float(
                    curve_inference.get("simultaneous_radius"), simultaneous_radius
                )
                for condition_index, condition in enumerate(conditions):
                    rows = by_condition.get(condition)
                    if not isinstance(rows, list) or len(rows) != len(snr_grid):
                        inference_consistent = False
                        continue
                    for snr_index, row in enumerate(rows):
                        if not isinstance(row, dict):
                            inference_consistent = False
                            continue
                        ordered = np.sort(bootstrap[:, snr_index, condition_index])
                        expected_intervals = {
                            "pointwise_95_percent_interval": [
                                float(ordered[math.floor(0.025 * G5_SNR_CURVE_BOOTSTRAP_DRAWS)]),
                                float(ordered[math.ceil(0.975 * G5_SNR_CURVE_BOOTSTRAP_DRAWS) - 1]),
                            ],
                            "simultaneous_95_percent_interval": [
                                max(
                                    0.0,
                                    float(observed[snr_index, condition_index])
                                    - simultaneous_radius,
                                ),
                                min(
                                    1.0,
                                    float(observed[snr_index, condition_index])
                                    + simultaneous_radius,
                                ),
                            ],
                        }
                        inference_consistent &= (
                            row.get("snr_db") == snr_grid[snr_index]
                            and _same_float(
                                row.get("mean_channel_macro_bler"),
                                float(observed[snr_index, condition_index]),
                            )
                            and all(
                                isinstance(row.get(key), list)
                                and len(row[key]) == 2
                                and all(
                                    _same_float(actual, expected)
                                    for actual, expected in zip(
                                        row[key], expected_values, strict=True
                                    )
                                )
                                for key, expected_values in expected_intervals.items()
                            )
                        )
                if inference_consistent and has_paired_contrast:
                    if not isinstance(by_paired_contrast, list) or len(by_paired_contrast) != len(
                        snr_grid
                    ):
                        inference_consistent = False
                    else:
                        for snr_index, row in enumerate(by_paired_contrast):
                            if not isinstance(row, dict):
                                inference_consistent = False
                                continue
                            ordered = np.sort(paired_bootstrap[:, snr_index])
                            estimate = float(observed_paired_improvement[snr_index])
                            expected_intervals = {
                                "pointwise_95_percent_interval": [
                                    float(
                                        ordered[math.floor(0.025 * G5_SNR_CURVE_BOOTSTRAP_DRAWS)]
                                    ),
                                    float(
                                        ordered[math.ceil(0.975 * G5_SNR_CURVE_BOOTSTRAP_DRAWS) - 1]
                                    ),
                                ],
                                "simultaneous_95_percent_interval": [
                                    max(-1.0, estimate - simultaneous_radius),
                                    min(1.0, estimate + simultaneous_radius),
                                ],
                            }
                            inference_consistent &= (
                                row.get("snr_db") == snr_grid[snr_index]
                                and _same_float(
                                    row.get("mean_channel_macro_bler_improvement"), estimate
                                )
                                and all(
                                    isinstance(row.get(key), list)
                                    and len(row[key]) == 2
                                    and all(
                                        _same_float(actual, expected)
                                        for actual, expected in zip(
                                            row[key], expected_values, strict=True
                                        )
                                    )
                                    for key, expected_values in expected_intervals.items()
                                )
                            )
    checks["channel_cluster_curve_inference"] = inference_consistent
    if curve_schema in {
        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v5",
        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v6",
        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v7",
        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v8",
        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v10",
        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v11",
    }:
        frozen_contract = frozen_acceptance.get("criteria")
        contract_consistent = isinstance(frozen_contract, dict)
        if contract_consistent:
            base_contract_keys = {
                "schema",
                "snr_db",
                "channel_seeds",
                "channel_estimator",
                "blocks_per_channel",
                "minimum_independent_channels",
                "maximum_corrected_macro_bler",
                "minimum_uncorrected_minus_corrected_macro_bler",
            }
            timestamped_contract = (
                frozen_contract.get("schema") == G5_SNR_TIMESTAMPED_ACCEPTANCE_SCHEMA
            )
            expected_contract_keys = base_contract_keys | (
                {"preregistered_utc"} if timestamped_contract else set()
            )
            contract_consistent = (
                set(frozen_contract) == expected_contract_keys
                and frozen_contract.get("schema")
                in {
                    "umcd.g5_ordered_impairments_snr_acceptance.v1",
                    G5_SNR_TIMESTAMPED_ACCEPTANCE_SCHEMA,
                }
                and timestamped_contract
                == (
                    curve_schema
                    in {
                        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v10",
                        "umcd.g5_ordered_impairments_independent_channels_snr_curve.v11",
                    }
                )
                and frozen_contract.get("snr_db") == snr_grid
                and frozen_contract.get("channel_seeds") == seeds
                and frozen_contract.get("channel_estimator") == config.get("channel_estimator")
                and frozen_contract.get("blocks_per_channel") == blocks
                and isinstance(frozen_contract.get("minimum_independent_channels"), int)
                and not isinstance(frozen_contract.get("minimum_independent_channels"), bool)
                and MIN_G5_INDEPENDENT_ACCEPTANCE_CHANNELS
                <= frozen_contract["minimum_independent_channels"]
                <= len(seeds)
                and isinstance(frozen_contract.get("maximum_corrected_macro_bler"), list)
                and len(frozen_contract["maximum_corrected_macro_bler"]) == len(snr_grid)
                and isinstance(
                    frozen_contract.get("minimum_uncorrected_minus_corrected_macro_bler"),
                    list,
                )
                and len(frozen_contract["minimum_uncorrected_minus_corrected_macro_bler"])
                == len(snr_grid)
                and frozen_acceptance.get("criteria_sha256")
                == _sha256(_canonical_json_bytes(frozen_contract))
                and frozen_acceptance.get("confidence_level") == 0.95
                and frozen_acceptance.get("interval_scope")
                == "simultaneous across all three conditions and paired improvements"
                and frozen_acceptance.get("decision_unit")
                == "independent synthetic channel; transport blocks are nested"
            )
            if curve_schema in {
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v6",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v8",
            }:
                signature = frozen_acceptance.get("preregistration_signature")
                contract_consistent &= (
                    isinstance(signature, dict)
                    and signature.get("status") == "verified_before_receiver_run"
                    and signature.get("namespace") == "umcd-g5-snr-acceptance-v1"
                    and isinstance(signature.get("signer_id"), str)
                    and bool(signature["signer_id"].strip())
                    and signature.get("contract_payload_sha256")
                    == _sha256(_canonical_json_bytes(frozen_contract))
                    and _is_sha256(signature.get("signature_sha256"))
                    and _is_sha256(signature.get("trusted_allowed_signers_sha256"))
                    and signature.get("trust_file_custody") == "not_established_by_this_verifier"
                )
                signature_reverified = False
                if (
                    acceptance_signature_path is not None
                    and trusted_allowed_signers_path is not None
                    and isinstance(signature, dict)
                    and isinstance(frozen_contract, dict)
                ):
                    signature_reverified = _verify_g5_preregistration_signature(
                        frozen_contract,
                        signature,
                        signature_path=acceptance_signature_path,
                        trusted_allowed_signers_path=trusted_allowed_signers_path,
                    )
                checks["preregistration_signature_reverified"] = signature_reverified
            elif curve_schema in {
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v10",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v11",
            }:
                signature = frozen_acceptance.get("preregistration_signature")
                temporal_order_consistent = _g5_declared_preregistration_order_matches(
                    frozen_contract, signature
                )
                contract_consistent &= (
                    isinstance(signature, dict)
                    and signature.get("status") == "verified_before_receiver_run"
                    and signature.get("namespace") == _G5_SNR_ACCEPTANCE_SIGNATURE_NAMESPACE
                    and isinstance(signature.get("signer_id"), str)
                    and bool(signature["signer_id"].strip())
                    and signature.get("contract_payload_sha256")
                    == _sha256(_canonical_json_bytes(frozen_contract))
                    and _is_sha256(signature.get("signature_sha256"))
                    and _is_sha256(signature.get("trusted_allowed_signers_sha256"))
                    and signature.get("trust_file_custody") == "not_established_by_this_verifier"
                    and temporal_order_consistent
                )
                signature_reverified = False
                if (
                    acceptance_signature_path is not None
                    and trusted_allowed_signers_path is not None
                    and isinstance(signature, dict)
                    and isinstance(frozen_contract, dict)
                ):
                    signature_reverified = _verify_g5_preregistration_signature(
                        frozen_contract,
                        signature,
                        signature_path=acceptance_signature_path,
                        trusted_allowed_signers_path=trusted_allowed_signers_path,
                    )
                checks["preregistration_signature_reverified"] = signature_reverified
                policy_identity = (
                    signature.get("reviewer_policy") if isinstance(signature, dict) else None
                )
                policy_reverified = False
                policy_consistent = (
                    curve_schema != "umcd.g5_ordered_impairments_independent_channels_snr_curve.v11"
                )
                if curve_schema == "umcd.g5_ordered_impairments_independent_channels_snr_curve.v11":
                    policy_consistent = (
                        isinstance(policy_identity, dict)
                        and policy_identity.get("status") == "verified"
                        and policy_identity.get("reviewer_principal") == signature.get("signer_id")
                        and "independent_reviewer"
                        in str(policy_identity.get("reviewer_roles", "")).split(",")
                        and policy_identity.get("purpose") == G5_SNR_REVIEWER_POLICY_PURPOSE
                        and policy_identity.get("signature_namespace")
                        == G5_SNR_REVIEWER_POLICY_SIGNATURE_NAMESPACE
                        and policy_identity.get("custodian_principal") != signature.get("signer_id")
                        and policy_identity.get("trust_file_custody")
                        == "not_established_by_this_verifier"
                    )
                    if (
                        policy_consistent
                        and reviewer_policy_path is not None
                        and reviewer_policy_signature_path is not None
                        and trusted_reviewer_allowed_signers_path is not None
                        and reviewer_policy_signer_id is not None
                        and isinstance(signature, dict)
                    ):
                        try:
                            verified_policy = _verify_g5_reviewer_policy(
                                reviewer_policy_path,
                                reviewer_policy_signature_path,
                                trusted_reviewer_allowed_signers_path,
                                reviewer_policy_signer_id,
                                acceptance_signer_id=signature.get("signer_id"),
                                acceptance_allowed_signers_sha256=signature.get(
                                    "trusted_allowed_signers_sha256", ""
                                ),
                            )
                        except (OSError, RuntimeError, TypeError, ValueError):
                            verified_policy = None
                        policy_reverified = verified_policy == policy_identity
                    policy_consistent &= bool(policy_reverified)
                    contract_consistent &= policy_consistent
                checks["reviewer_policy_reverified"] = (
                    policy_reverified
                    if curve_schema
                    == "umcd.g5_ordered_impairments_independent_channels_snr_curve.v11"
                    else None
                )
            else:
                contract_consistent &= "preregistration_signature" not in frozen_acceptance
                checks["preregistration_signature_reverified"] = None
            for key, lower, upper in (
                ("maximum_corrected_macro_bler", 0.0, 1.0),
                ("minimum_uncorrected_minus_corrected_macro_bler", 0.0, 1.0),
            ):
                values = frozen_contract.get(key)
                contract_consistent &= isinstance(values, list) and all(
                    _is_finite_number(value) and lower <= value <= upper for value in values
                )
        decision_consistent = contract_consistent and inference_consistent
        expected_points: list[dict[str, Any]] = []
        if decision_consistent:
            assert isinstance(frozen_contract, dict)
            corrected_rows = curve_inference["by_condition"]["corrected"]
            improvement_rows = curve_inference["by_paired_contrast"]
            all_pass = True
            for index, snr in enumerate(snr_grid):
                corrected_upper = corrected_rows[index]["simultaneous_95_percent_interval"][1]
                improvement_lower = improvement_rows[index]["simultaneous_95_percent_interval"][0]
                corrected_pass = (
                    corrected_upper <= frozen_contract["maximum_corrected_macro_bler"][index]
                )
                improvement_pass = (
                    improvement_lower
                    >= frozen_contract["minimum_uncorrected_minus_corrected_macro_bler"][index]
                )
                point_pass = corrected_pass and improvement_pass
                all_pass &= point_pass
                expected_points.append(
                    {
                        "snr_db": snr,
                        "corrected_macro_bler_simultaneous_upper": corrected_upper,
                        "maximum_corrected_macro_bler": frozen_contract[
                            "maximum_corrected_macro_bler"
                        ][index],
                        "corrected_bler_criterion_pass": corrected_pass,
                        "uncorrected_minus_corrected_simultaneous_lower": improvement_lower,
                        "minimum_uncorrected_minus_corrected_macro_bler": frozen_contract[
                            "minimum_uncorrected_minus_corrected_macro_bler"
                        ][index],
                        "minimum_improvement_criterion_pass": improvement_pass,
                        "pass": point_pass,
                    }
                )
            decision_consistent &= (
                acceptance.get("status") == ("PASS" if all_pass else "FAIL")
                and acceptance.get("pass") is all_pass
                and acceptance.get("cluster_count") == len(seeds)
                and acceptance.get("point_decisions") == expected_points
            )
        checks["acceptance_contract_decision"] = decision_consistent
        checks["acceptance_is_diagnostic_only"] = None
    else:
        checks["acceptance_contract_decision"] = None
        checks["acceptance_is_diagnostic_only"] = (
            isinstance(frozen_acceptance, dict)
            and frozen_acceptance.get("status") == "descriptive_only"
            and acceptance.get("pass") is False
        )
    evidence_manifest = config.get("waveform_evidence")
    if evidence_manifest is None:
        point_manifests = [
            (
                point.get("experiment", {}).get("config", {}).get("waveform_evidence")
                if isinstance(point, dict)
                and isinstance(point.get("experiment"), dict)
                and isinstance(point["experiment"].get("config"), dict)
                else None
            )
            for point in points
        ]
        checks["waveform_equation_recomputation"] = (
            None if all(manifest is None for manifest in point_manifests) else False
        )
    elif waveform_evidence_bytes is None or not isinstance(evidence_manifest, dict):
        checks["waveform_equation_recomputation"] = False
    else:
        evidence_checks: list[bool] = []
        manifest_grid_matches = not (
            curve_schema
            in {
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v3",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v4",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v5",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v6",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v7",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v8",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v9",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v10",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v11",
            }
            and (
                evidence_manifest.get("schema")
                != "umcd.g5_ordered_impairments_waveform_evidence.v4"
                or evidence_manifest.get("snr_point_count") != len(snr_grid)
                or evidence_manifest.get("snr_db") != snr_grid
            )
        )
        for snr_index, point in enumerate(points):
            if not manifest_grid_matches:
                evidence_checks.append(False)
                continue
            report = point.get("experiment") if isinstance(point, dict) else None
            point_config = report.get("config") if isinstance(report, dict) else None
            if not isinstance(point_config, dict):
                evidence_checks.append(False)
                continue
            point_manifest = point_config.get("waveform_evidence")
            records = point_config.get("per_channel_draws")
            acceptance = report.get("acceptance") if isinstance(report, dict) else None
            observed = acceptance.get("observed") if isinstance(acceptance, dict) else None
            criteria = (
                acceptance.get("criteria_frozen_before_run")
                if isinstance(acceptance, dict)
                else None
            )
            if (
                point_manifest != evidence_manifest
                or not isinstance(records, list)
                or not isinstance(observed, dict)
                or not isinstance(criteria, dict)
            ):
                evidence_checks.append(False)
                continue
            try:
                maximum_error, minimum_wrong_order = _recompute_waveform_diagnostics(
                    point_config,
                    records,
                    waveform_evidence_bytes,
                    snr_index=snr_index,
                )
                evidence_checks.append(
                    abs(
                        maximum_error
                        - float(observed.get("max_samplewise_reference_absolute_error", math.inf))
                    )
                    <= 5e-6
                    and abs(
                        minimum_wrong_order - float(observed.get("min_wrong_order_rmse", -math.inf))
                    )
                    <= 5e-6
                    and maximum_error
                    < float(criteria.get("max_samplewise_reference_absolute_error", -math.inf))
                    and minimum_wrong_order > float(criteria.get("min_wrong_order_rmse", math.inf))
                )
            except (OSError, EOFError, ValueError, TypeError, KeyError, OverflowError):
                evidence_checks.append(False)
        checks["waveform_equation_recomputation"] = bool(evidence_checks) and all(evidence_checks)
    required_checks = [value for value in checks.values() if value is not None]
    return {
        "schema": "umcd.g5_ordered_impairments_result_audit.v1",
        "status": "AUDIT_PARTIAL" if required_checks and all(required_checks) else "AUDIT_FAILED",
        "scope": (
            "integrity, retained-count, and bounded fixture acceptance replay; no deployment qualification"
            if curve_schema
            in {
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v5",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v6",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v7",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v8",
                "umcd.g5_ordered_impairments_independent_channels_snr_curve.v10",
            }
            else "integrity and retained-count audit; no familywise scientific acceptance"
        ),
        "checks": checks,
    }


def _run_receiver_probe(**kwargs: Any) -> dict[str, Any]:
    """Load the optional Torch-backed receiver only for requested replay."""
    from experiments.g5_ordered_impairments_receiver import (
        run_independent_ordered_impairment_probe,
    )

    return run_independent_ordered_impairment_probe(**kwargs)


def _replay_experiment(experiment: dict[str, Any]) -> dict[str, Any]:
    config = experiment.get("config")
    if not isinstance(config, dict):
        raise ValueError("experiment config must be an object for deterministic replay")
    seeds = config.get("channel_seeds")
    blocks_per_channel = config.get("blocks_per_channel")
    if (
        not isinstance(seeds, list)
        or not seeds
        or len(seeds) > 64
        or isinstance(blocks_per_channel, bool)
        or not isinstance(blocks_per_channel, int)
        or not 1 <= blocks_per_channel <= 8
    ):
        raise ValueError("deterministic replay is bounded to at most 64 channels × 8 blocks")
    estimator_identity = config.get("channel_estimator", {"name": "sionna_default_ls"})
    if not isinstance(estimator_identity, dict) or estimator_identity.get("name") not in {
        "sionna_default_ls",
        "finite_delay_dft",
    }:
        raise ValueError("deterministic replay requires a supported estimator identity")
    estimator_name = estimator_identity["name"]
    evidence = config.get("waveform_evidence")
    if evidence is None:
        return _run_receiver_probe(
            channel_seeds=tuple(seeds),
            blocks_per_channel=blocks_per_channel,
            channel_estimator=estimator_name,
        )
    if not isinstance(evidence, dict) or not isinstance(evidence.get("file_name"), str):
        raise ValueError("waveform evidence manifest is invalid for deterministic replay")
    with tempfile.TemporaryDirectory(prefix="g5-waveform-replay-") as directory:
        evidence_path = Path(directory) / evidence["file_name"]
        return _run_receiver_probe(
            channel_seeds=tuple(seeds),
            blocks_per_channel=blocks_per_channel,
            waveform_evidence_output=evidence_path,
            channel_estimator=estimator_name,
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
    source_path = REPOSITORY_ROOT / source_name
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
        waveform_evidence_bytes = None
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
            waveform_evidence_bytes = _read_bounded_evidence_file(
                path.parent / evidence_name,
                maximum_bytes=_MAX_WAVEFORM_EVIDENCE_BYTES,
            )
        return _audit_independent_snr_curve(
            result,
            source_bytes,
            committed_source,
            waveform_evidence_bytes=waveform_evidence_bytes,
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
        if estimator_relative_path != "experiments/g5_cp_bounded_dft_estimator.py":
            raise ValueError("unsupported estimator implementation path")
        estimator_source_bytes = (REPOSITORY_ROOT / estimator_relative_path).read_bytes()
    replayed_experiment: dict[str, Any] | None = None
    if deterministic_replay:
        if result.get("provenance", {}).get("source_sha256") != _sha256(source_bytes):
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
                    f"{committed_revision}:experiments/g5_cp_bounded_dft_estimator.py",
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


if __name__ == "__main__":
    raise SystemExit(main())
