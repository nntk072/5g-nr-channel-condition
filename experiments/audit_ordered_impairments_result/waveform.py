"""Recomputation of waveform diagnostics and cluster bootstrap intervals from retained evidence."""

from __future__ import annotations

import hashlib
import io
import math
import random
from pathlib import Path
from typing import Any

import numpy as np

from .constants import (
    _MAX_WAVEFORM_EVIDENCE_BYTES,
)
from .evidence_files import (
    _preflight_waveform_archive,
)
from .references import (
    _iq_imbalance_reference,
    _is_finite_number,
    _path_sum_reference,
    _positive_integer_delay_reference,
    _same_float,
)


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
