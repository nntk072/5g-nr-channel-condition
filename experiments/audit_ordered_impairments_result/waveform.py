"""Band waveform of audit_head."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import stat
import tempfile
import zipfile
from pathlib import Path
from typing import Any, TypeGuard

import numpy as np
from evaluation.signed_attestation import verify_openssh_signature_for_bytes

from experiments.acceptance import (
    validate_preregistration_order,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


_RESULT_SCHEMA = "umcd.g5_ordered_impairments_result.v2"


_SNR_ACCEPTANCE_SIGNATURE_NAMESPACE = "umcd-g5-snr-acceptance-v1"


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
    "umcd.g5_ordered_impairments_independent_channels_snr_curve.v13",
    "umcd.g5_ordered_impairments_independent_channels_snr_curve.v14",
    "umcd.g5_ordered_impairments_independent_channels_snr_curve.v15",
    "umcd.g5_ordered_impairments_independent_channels_snr_curve.v16",
    "umcd.g5_ordered_impairments_independent_channels_snr_curve.v17",
    "umcd.g5_ordered_impairments_independent_channels_snr_curve.v25",
}


_CUDA_CURVE_BASE_VERSIONS = {12: 9, 13: 7, 14: 8, 15: 10, 16: 11, 25: 17}


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
        estimator.get("implementation_path") == "experiments/cp_bounded_dft_estimator.py"
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


def _same_float(actual: Any, expected: Any) -> bool:
    return (
        not isinstance(actual, bool)
        and isinstance(actual, (int, float))
        and not isinstance(expected, bool)
        and isinstance(expected, (int, float))
        and math.isfinite(float(actual))
        and math.isfinite(float(expected))
        and math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12)
    )


def _is_finite_nonnegative(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _is_finite_number(value: Any) -> TypeGuard[int | float]:
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
        tx.ndim not in {4, 5}
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
                or not _same_float(config.get("nominal_snr_db"), snr_values[snr_index])
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


def _run_receiver_probe(**kwargs: Any) -> dict[str, Any]:
    """Load the optional Torch-backed receiver only for requested replay."""
    from experiments.ordered_impairments_receiver import (
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
