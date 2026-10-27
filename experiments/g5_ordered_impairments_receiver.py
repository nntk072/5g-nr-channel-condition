"""Paired CPU PUSCH check for ordered RF impairments over known multipath.

This is a controlled receiver fixture, not a field BLER result. The known chain
is TX IQ imbalance/phase noise -> static two-tap propagation -> RX CFO/phase
noise/IQ imbalance -> integer capture delay -> AWGN. Timing and CFO are
acquired and corrected before DMRS channel estimation and PUSCH decoding.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import random
import subprocess
import sys
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from integrations.sionna.nr_pusch import SionnaPUSCHLSReceiver
from integrations.sionna.ofdm_impairments import (
    apply_integer_timing_offset,
    apply_iq_imbalance,
    apply_phase_noise,
    apply_residual_cfo,
    apply_time_varying_multipath,
    estimate_integer_timing_offset_from_cp,
    estimate_residual_cfo_from_cp,
    generate_wiener_phase_noise_rad,
)
from receivers.nr import NRPUSCHConfig
from scipy.stats import beta as beta_distribution

from experiments.g5_acceptance import (
    G5_SNR_ACCEPTANCE_SIGNATURE_NAMESPACE,
    G5_SNR_CURVE_BOOTSTRAP_BATCH_SIZE,
    G5_SNR_CURVE_BOOTSTRAP_DRAWS,
    G5_SNR_CURVE_BOOTSTRAP_SEED,
    G5_SNR_REVIEWER_POLICY_PURPOSE,
    G5_SNR_REVIEWER_POLICY_SIGNATURE_NAMESPACE,
    G5_SNR_TIMESTAMPED_ACCEPTANCE_SCHEMA,
    validate_g5_preregistration_order,
)
from experiments.g5_acceptance import (
    evaluate_g5_snr_curve_acceptance as _evaluate_independent_snr_curve_acceptance,
)
from experiments.g5_acceptance import (
    freeze_g5_snr_acceptance_criteria as _freeze_independent_snr_acceptance_criteria,
)
from experiments.g5_acceptance import (
    load_g5_snr_acceptance_contract as _load_snr_acceptance_contract,
)

SCHEMA = "umcd.g5_ordered_impairments_receiver.v2"
RESULT_SCHEMA = "umcd.g5_ordered_impairments_result.v2"
_LEGACY_RESULT_SCHEMA = "umcd.g5_ordered_impairments_result.v1"
PAYLOAD_SEEDS = tuple(range(73_501, 73_509))
BLOCKS_PER_SEED = 16
NOISE_SEED_OFFSET = 200_000
TX_PHASE_SEED_OFFSET = 300_000
RX_PHASE_SEED_OFFSET = 400_000
NOISE_VARIANCE = 1e-3
CFO_HZ = 600.0
TIMING_OFFSET_SAMPLES = 2
TX_LINEWIDTH_HZ = 10.0
RX_LINEWIDTH_HZ = 10.0
TX_IQ_GAIN_IMBALANCE_DB = 0.5
TX_IQ_PHASE_IMBALANCE_RAD = 0.02
RX_IQ_GAIN_IMBALANCE_DB = 0.5
RX_IQ_PHASE_IMBALANCE_RAD = -0.02
DIRECT_COEFFICIENT = 0.9 - 0.1j
ECHO_COEFFICIENT = 0.35 + 0.2j
ECHO_DELAY_SAMPLES = 2
MAX_TIMING_OFFSET_SAMPLES = 3
CP_GUARD_SAMPLES = ECHO_DELAY_SAMPLES

# Frozen before running the confirmatory seeds: timing acquisition must be
# exact; absolute CFO error must be below 250 Hz at P95 and 350 Hz maximum;
# clean reference BLER must be <=1%; corrected CRC retention must be >=90%.
MAX_CFO_ERROR_P95_HZ = 250.0
MAX_CFO_ERROR_HZ = 350.0
MAX_REFERENCE_BLER = 0.01
MAX_IMPAIRED_BLER = 0.10
INDEPENDENT_CHANNEL_SEEDS = tuple(range(74_001, 74_065))
INDEPENDENT_CHANNEL_BOOTSTRAP_SEED = 74_999
INDEPENDENT_SNR_CURVE_BOOTSTRAP_SEED = G5_SNR_CURVE_BOOTSTRAP_SEED
INDEPENDENT_SNR_CURVE_BOOTSTRAP_DRAWS = G5_SNR_CURVE_BOOTSTRAP_DRAWS
INDEPENDENT_CHANNEL_BOOTSTRAP_DRAWS = 20_000
_MAX_G5_INDEPENDENT_TRANSPORT_BLOCKS = 16_384
_MAX_G5_INDEPENDENT_SNR_POINTS = 64
_MAX_WAVEFORM_EVIDENCE_BYTES = 512 * 1024 * 1024
_MAX_WAVEFORM_EVIDENCE_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024
_MAX_WAVEFORM_NPZ_MEMBERS = 32
_MAX_WAVEFORM_NPZ_MEMBER_NAME_BYTES = 128


class _BoundedBytesIO(io.BytesIO):
    """BytesIO that rejects writes whose high-water mark exceeds a fixed cap."""

    def __init__(self, maximum_bytes: int) -> None:
        super().__init__()
        self._maximum_bytes = maximum_bytes
        self._high_water_mark = 0

    def write(self, content: Any) -> int:
        high_water_mark = max(self._high_water_mark, self.tell() + len(content))
        if high_water_mark > self._maximum_bytes:
            raise ValueError("compressed waveform evidence exceeds the configured size limit")
        written = super().write(content)
        self._high_water_mark = max(self._high_water_mark, self.tell())
        return written


def _write_bounded_npz(
    destination: _BoundedBytesIO,
    *,
    channel_count: int,
    channel_arrays: dict[str, list[np.ndarray]],
    scalar_arrays: dict[str, np.ndarray],
) -> None:
    """Stream channel-major arrays into compressed NPY members without stacking."""
    if isinstance(channel_count, bool) or not isinstance(channel_count, int) or channel_count < 1:
        raise ValueError("channel_count must be a positive integer")
    entries: dict[str, tuple[list[np.ndarray], bool]] = {
        **{name: (chunks, True) for name, chunks in channel_arrays.items()},
        **{name: ([array], False) for name, array in scalar_arrays.items()},
    }
    if set(channel_arrays).intersection(scalar_arrays):
        raise ValueError("NPZ member names must be unique")
    if not entries:
        raise ValueError("at least one NPZ array is required")
    if len(entries) > _MAX_WAVEFORM_NPZ_MEMBERS:
        raise ValueError("waveform evidence has too many NPZ members")

    prepared: dict[str, tuple[list[np.ndarray], tuple[int, ...], np.dtype[Any]]] = {}
    uncompressed_bytes = 0
    for name, (chunks, channel_major) in entries.items():
        if (
            not isinstance(name, str)
            or not name
            or not name.isascii()
            or len(name) > _MAX_WAVEFORM_NPZ_MEMBER_NAME_BYTES
            or not name[0].isalnum()
            or any(not (character.isalnum() or character in "_.-") for character in name)
            or not chunks
        ):
            raise ValueError("NPZ array names must be short safe identifiers with nonempty chunks")
        if channel_major and len(chunks) != channel_count:
            raise ValueError(f"NPZ channel count disagrees for member {name}")
        first = np.asarray(chunks[0])
        if first.dtype.hasobject or first.dtype.kind not in "iufc":
            raise ValueError(f"NPZ member {name} has an unsupported dtype")
        for chunk in chunks:
            value = np.asarray(chunk)
            if value.shape != first.shape or value.dtype != first.dtype:
                raise ValueError(f"NPZ member {name} has inconsistent channel shape or dtype")
            uncompressed_bytes += value.nbytes
        shape = (channel_count, *first.shape) if channel_major else first.shape
        prepared[name] = (chunks, shape, first.dtype)
    if uncompressed_bytes + 64 * 1024 > _MAX_WAVEFORM_EVIDENCE_UNCOMPRESSED_BYTES:
        raise ValueError("uncompressed waveform evidence exceeds the configured size limit")

    with zipfile.ZipFile(destination, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, (chunks, shape, dtype) in sorted(prepared.items()):
            with archive.open(f"{name}.npy", mode="w", force_zip64=True) as member:
                np.lib.format.write_array_header_2_0(
                    member,
                    {
                        "descr": np.lib.format.dtype_to_descr(dtype),
                        "fortran_order": False,
                        "shape": shape,
                    },
                )
                for chunk in chunks:
                    contiguous = np.ascontiguousarray(chunk)
                    member.write(memoryview(contiguous).cast("B"))


def _combine_snr_curve_waveform_evidence(
    point_paths: tuple[Path, ...],
    *,
    output_path: Path,
    snr_values: tuple[float, ...],
) -> dict[str, Any]:
    """Combine per-point evidence while storing shared raw draws only once."""
    if len(point_paths) != len(snr_values) or not point_paths:
        raise ValueError("SNR waveform evidence paths must match the nonempty SNR grid")
    raw_names = (
        "channel_seeds",
        "tx_waveform",
        "tx_phase_noise_rad",
        "rx_phase_noise_rad",
        "actual_capture",
        "awgn_standard_normal_real",
        "awgn_standard_normal_imag",
    )
    condition_names = {
        "clean": "clean_receiver_input",
        "uncorrected": "uncorrected_receiver_input",
        "corrected": "corrected_receiver_input",
    }
    expected_point_names = {*raw_names, *condition_names.values()}
    planned_uncompressed_bytes = 0
    staging_compressed_bytes = 0
    for point_index, point_path in enumerate(point_paths):
        staging_compressed_bytes += point_path.stat().st_size
        with zipfile.ZipFile(point_path) as point_archive:
            members = point_archive.infolist()
            if len(members) != len(expected_point_names) or {
                member.filename for member in members
            } != {f"{name}.npy" for name in expected_point_names}:
                raise ValueError("per-point waveform evidence has an unexpected array roster")
            member_bytes = {Path(member.filename).stem: member.file_size for member in members}
            planned_uncompressed_bytes += sum(
                member_bytes[name] for name in (raw_names if point_index == 0 else ())
            )
            planned_uncompressed_bytes += sum(
                member_bytes[name] for name in condition_names.values()
            )
    if planned_uncompressed_bytes + 64 * 1024 > _MAX_WAVEFORM_EVIDENCE_UNCOMPRESSED_BYTES:
        raise ValueError(
            "multi-SNR waveform evidence exceeds the configured uncompressed size limit"
        )
    if staging_compressed_bytes > 2 * _MAX_WAVEFORM_EVIDENCE_BYTES:
        raise ValueError("multi-SNR waveform evidence staging exceeds its disk size limit")
    raw_arrays: dict[str, np.ndarray] | None = None
    condition_chunks: dict[str, list[np.ndarray]] = {
        f"{name}_receiver_input_by_snr": [] for name in condition_names
    }
    uncompressed_bytes = 0
    for point_index, point_path in enumerate(point_paths):
        with np.load(point_path, allow_pickle=False) as archive:
            if set(archive.files) != expected_point_names:
                raise ValueError("per-point waveform evidence has an unexpected array roster")
            if point_index == 0:
                raw_arrays = {name: archive[name] for name in raw_names}
                uncompressed_bytes += sum(array.nbytes for array in raw_arrays.values())
            else:
                if raw_arrays is None:
                    raise RuntimeError("shared raw waveform arrays were not initialized")
                for name in raw_names:
                    if not np.array_equal(raw_arrays[name], archive[name]):
                        raise ValueError("raw waveform draws differ across paired SNR points")
            for condition, source_name in condition_names.items():
                values = archive[source_name]
                if not np.iscomplexobj(values) or not np.isfinite(values).all():
                    raise ValueError(f"{condition} waveform evidence is invalid")
                uncompressed_bytes += values.nbytes
                if uncompressed_bytes + 64 * 1024 > _MAX_WAVEFORM_EVIDENCE_UNCOMPRESSED_BYTES:
                    raise ValueError(
                        "multi-SNR waveform evidence exceeds the configured uncompressed size limit"
                    )
                condition_chunks[f"{condition}_receiver_input_by_snr"].append(values)
    if raw_arrays is None:
        raise ValueError("SNR waveform evidence did not contain shared raw draws")
    evidence_buffer = _BoundedBytesIO(_MAX_WAVEFORM_EVIDENCE_BYTES)
    _write_bounded_npz(
        evidence_buffer,
        channel_count=len(snr_values),
        channel_arrays=condition_chunks,
        scalar_arrays=raw_arrays,
    )
    evidence_bytes = evidence_buffer.getvalue()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as temporary_file:
            temporary_file.write(evidence_bytes)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.link(temporary_name, output_path)
        Path(temporary_name).unlink()
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return {
        "schema": "umcd.g5_ordered_impairments_waveform_evidence.v4",
        "file_name": output_path.name,
        "sha256": _sha256(evidence_bytes),
        "byte_count": len(evidence_bytes),
        "channel_count": int(raw_arrays["channel_seeds"].shape[0]),
        "snr_point_count": len(snr_values),
        "snr_db": list(snr_values),
        "arrays": sorted([*raw_names, *condition_chunks]),
    }


def _proper_complex_noise(shape: Any, seed: int, variance: float) -> torch.Tensor:
    real, imag = _standard_complex_noise_draws(shape, seed)
    return torch.complex(real, imag) * math.sqrt(variance / 2.0)


def _standard_complex_noise_draws(
    shape: Any, seed: int, *, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return CPU-seeded unscaled Gaussian draws on the requested device."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    real = torch.randn(shape, generator=generator, dtype=torch.float32)
    imag = torch.randn(shape, generator=generator, dtype=torch.float32)
    return real.to(device=device), imag.to(device=device)


def _phase_noise(
    shape: Any,
    seed: int,
    sample_rate_hz: float,
    linewidth_hz: float,
    *,
    device: str = "cpu",
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    phase = generate_wiener_phase_noise_rad(
        shape[-1],
        sampling_frequency_hz=sample_rate_hz,
        linewidth_hz=linewidth_hz,
        generator=generator,
        leading_shape=tuple(shape[:-1]),
    )
    return phase.to(device=device)


def _known_chain(
    waveform: torch.Tensor,
    *,
    sample_rate_hz: float,
    tx_phase: torch.Tensor,
    rx_phase: torch.Tensor,
    capture_delay: int,
    path_coefficients: tuple[complex, ...] = (DIRECT_COEFFICIENT, ECHO_COEFFICIENT),
    path_delays_samples: tuple[int, ...] = (0, ECHO_DELAY_SAMPLES),
) -> torch.Tensor:
    """Apply the physical impairment sequence declared by this experiment."""
    tx_iq = apply_iq_imbalance(
        waveform,
        gain_imbalance_db=TX_IQ_GAIN_IMBALANCE_DB,
        phase_imbalance_rad=TX_IQ_PHASE_IMBALANCE_RAD,
    )
    tx_noisy = apply_phase_noise(tx_iq, phase_noise_rad=tx_phase)
    propagated = apply_time_varying_multipath(
        tx_noisy,
        path_coefficients=path_coefficients,
        path_delays_samples=path_delays_samples,
        path_dopplers_hz=(0.0,) * len(path_coefficients),
        sampling_frequency_hz=sample_rate_hz,
    )
    cfo = apply_residual_cfo(propagated, cfo_hz=CFO_HZ, sampling_frequency_hz=sample_rate_hz)
    rx_noisy = apply_phase_noise(cfo, phase_noise_rad=rx_phase)
    rx_iq = apply_iq_imbalance(
        rx_noisy,
        gain_imbalance_db=RX_IQ_GAIN_IMBALANCE_DB,
        phase_imbalance_rad=RX_IQ_PHASE_IMBALANCE_RAD,
    )
    return apply_integer_timing_offset(rx_iq, offset_samples=capture_delay)


def _cluster_bootstrap_interval(
    values: list[float], *, seed: int, alpha: float = 0.05, draws: int = 20_000
) -> tuple[float, float]:
    """Percentile interval over independent channel clusters."""
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("values must be a nonempty finite sequence")
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 100:
        raise ValueError("draws must be an integer >= 100")
    rng = random.Random(seed)
    count = len(values)
    samples = sorted(
        sum(values[rng.randrange(count)] for _ in range(count)) / count for _ in range(draws)
    )
    lower_index = math.floor((alpha / 2.0) * draws)
    upper_index = min(draws - 1, math.ceil((1.0 - alpha / 2.0) * draws) - 1)
    return samples[lower_index], samples[upper_index]


def _channel_any_failure_upper_bound(failed_channels: int, channels: int, *, alpha: float) -> float:
    """Exact one-sided bound for a conservative channel-level failure prevalence."""
    if isinstance(failed_channels, bool) or isinstance(channels, bool):
        raise TypeError("channel counts must be integers")
    if not isinstance(failed_channels, int) or not isinstance(channels, int):
        raise TypeError("channel counts must be integers")
    if channels < 1 or not 0 <= failed_channels <= channels:
        raise ValueError("require 0 <= failed_channels <= channels")
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if failed_channels == channels:
        return 1.0
    return float(
        beta_distribution.ppf(1.0 - alpha, failed_channels + 1, channels - failed_channels)
    )


def _pool_repeated_waveforms_for_shared_sync(waveform: torch.Tensor) -> torch.Tensor:
    """Concatenate repeated blocks within one channel for shared sync acquisition."""
    if waveform.ndim < 2 or waveform.shape[0] < 1 or waveform.shape[-1] < 1:
        raise ValueError("waveform must have a nonempty block axis and sample axis")
    blocks_first = waveform.movedim(0, -2)
    return blocks_first.reshape(1, *waveform.shape[1:-1], -1)


def _run_ordered_impairment_probe(
    *,
    payload_seeds: tuple[int, ...] = PAYLOAD_SEEDS,
    blocks_per_seed: int = BLOCKS_PER_SEED,
) -> dict[str, Any]:
    """Run paired clean/impaired CPU PUSCH with independent payload/noise seeds."""
    if not payload_seeds or len(set(payload_seeds)) != len(payload_seeds):
        raise ValueError("payload_seeds must be nonempty and unique")
    if any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in payload_seeds
    ):
        raise ValueError("payload_seeds must contain nonnegative integers")
    if (
        isinstance(blocks_per_seed, bool)
        or not isinstance(blocks_per_seed, int)
        or blocks_per_seed < 1
    ):
        raise ValueError("blocks_per_seed must be a positive integer")

    config = NRPUSCHConfig(n_size_bwp=6, subcarrier_spacing_khz=30)
    receiver = SionnaPUSCHLSReceiver(config, device="cpu")
    grid = receiver.resource_grid
    sample_rate_hz = float(grid.fft_size * config.subcarrier_spacing_khz * 1_000)
    cp_length = int(grid.cyclic_prefix_length)
    if MAX_TIMING_OFFSET_SAMPLES >= cp_length:
        raise RuntimeError("configured capture timing search must fit within the CP")

    outcomes: dict[str, list[bool]] = {
        "clean": [],
        "impaired_uncorrected": [],
        "impaired_corrected": [],
    }
    payload_bit_errors = {name: 0 for name in outcomes}
    cfo_errors: list[float] = []
    timing_errors: list[int] = []
    max_reference_error = 0.0
    min_wrong_order_difference = math.inf

    for payload_seed in payload_seeds:
        tx_grid, tx_bits = receiver.transmit(blocks_per_seed, seed=payload_seed)
        tx_waveform = receiver.modulate_slot_grid(tx_grid)
        shape = tx_waveform.shape
        tx_phase = _phase_noise(
            shape, payload_seed + TX_PHASE_SEED_OFFSET, sample_rate_hz, TX_LINEWIDTH_HZ
        )
        rx_phase = _phase_noise(
            shape, payload_seed + RX_PHASE_SEED_OFFSET, sample_rate_hz, RX_LINEWIDTH_HZ
        )

        # Independently spell out the declared sample-domain reference rather
        # than using _known_chain to generate its expected values.
        tx_iq_reference = apply_iq_imbalance(
            tx_waveform,
            gain_imbalance_db=TX_IQ_GAIN_IMBALANCE_DB,
            phase_imbalance_rad=TX_IQ_PHASE_IMBALANCE_RAD,
        )
        tx_reference = tx_iq_reference * torch.polar(torch.ones_like(tx_phase), tx_phase)
        sample_axis = torch.arange(tx_waveform.shape[-1], dtype=torch.float32)
        delayed_tx = torch.nn.functional.pad(tx_reference, (ECHO_DELAY_SAMPLES, 0))[
            ..., : tx_reference.shape[-1]
        ]
        channel_reference = DIRECT_COEFFICIENT * tx_reference + ECHO_COEFFICIENT * delayed_tx
        cfo_phase = 2.0 * math.pi * CFO_HZ * sample_axis / sample_rate_hz
        cfo_reference = channel_reference * torch.polar(torch.ones_like(cfo_phase), cfo_phase)
        rx_reference = cfo_reference * torch.polar(torch.ones_like(rx_phase), rx_phase)
        rx_reference = apply_iq_imbalance(
            rx_reference,
            gain_imbalance_db=RX_IQ_GAIN_IMBALANCE_DB,
            phase_imbalance_rad=RX_IQ_PHASE_IMBALANCE_RAD,
        )
        reference_capture = torch.cat(
            (
                torch.zeros_like(rx_reference[..., :TIMING_OFFSET_SAMPLES]),
                rx_reference[..., :-TIMING_OFFSET_SAMPLES],
            ),
            dim=-1,
        )
        actual_capture = _known_chain(
            tx_waveform,
            sample_rate_hz=sample_rate_hz,
            tx_phase=tx_phase,
            rx_phase=rx_phase,
            capture_delay=TIMING_OFFSET_SAMPLES,
        )
        max_reference_error = max(
            max_reference_error,
            float(torch.max(torch.abs(actual_capture - reference_capture))),
        )

        # An order-negative control moves receiver phase noise ahead of the
        # multipath operator. It must differ because delayed samples carry the
        # oscillator phase from their transmit time, not their receive time.
        rx_phase_before_channel = apply_phase_noise(tx_reference, phase_noise_rad=rx_phase)
        wrong_order = apply_time_varying_multipath(
            rx_phase_before_channel,
            path_coefficients=(DIRECT_COEFFICIENT, ECHO_COEFFICIENT),
            path_delays_samples=(0, ECHO_DELAY_SAMPLES),
            path_dopplers_hz=(0.0, 0.0),
            sampling_frequency_hz=sample_rate_hz,
        )
        wrong_order = apply_residual_cfo(
            wrong_order, cfo_hz=CFO_HZ, sampling_frequency_hz=sample_rate_hz
        )
        wrong_order = apply_iq_imbalance(
            wrong_order,
            gain_imbalance_db=RX_IQ_GAIN_IMBALANCE_DB,
            phase_imbalance_rad=RX_IQ_PHASE_IMBALANCE_RAD,
        )
        wrong_order = apply_integer_timing_offset(wrong_order, offset_samples=TIMING_OFFSET_SAMPLES)
        min_wrong_order_difference = min(
            min_wrong_order_difference,
            float(torch.sqrt(torch.mean(torch.abs(actual_capture - wrong_order) ** 2))),
        )

        noise_seed = payload_seed + NOISE_SEED_OFFSET
        noise = _proper_complex_noise(shape, noise_seed, NOISE_VARIANCE)
        clean_channel = apply_time_varying_multipath(
            tx_waveform,
            path_coefficients=(DIRECT_COEFFICIENT, ECHO_COEFFICIENT),
            path_delays_samples=(0, ECHO_DELAY_SAMPLES),
            path_dopplers_hz=(0.0, 0.0),
            sampling_frequency_hz=sample_rate_hz,
        )
        clean_capture = (
            apply_integer_timing_offset(clean_channel, offset_samples=TIMING_OFFSET_SAMPLES) + noise
        )
        clean_grid = receiver.demodulate_slot_waveform(
            apply_integer_timing_offset(clean_capture, offset_samples=-TIMING_OFFSET_SAMPLES)
        )
        clean_bits, clean_crc = receiver.decode_pusch(clean_grid, torch.tensor(NOISE_VARIANCE))
        outcomes["clean"].extend(bool(x) for x in clean_crc.reshape(-1).tolist())
        payload_bit_errors["clean"] += int((clean_bits != tx_bits).sum().item())

        noisy_capture = actual_capture + noise
        uncorrected_grid = receiver.demodulate_slot_waveform(noisy_capture)
        uncorrected_bits, uncorrected_crc = receiver.decode_pusch(
            uncorrected_grid, torch.tensor(NOISE_VARIANCE)
        )
        outcomes["impaired_uncorrected"].extend(
            bool(x) for x in uncorrected_crc.reshape(-1).tolist()
        )
        payload_bit_errors["impaired_uncorrected"] += int(
            (uncorrected_bits != tx_bits).sum().item()
        )
        pooled_capture = _pool_repeated_waveforms_for_shared_sync(noisy_capture)
        acquired_timing = estimate_integer_timing_offset_from_cp(
            pooled_capture,
            fft_size=int(grid.fft_size),
            cyclic_prefix_length=cp_length,
            max_offset_samples=MAX_TIMING_OFFSET_SAMPLES,
            guard_samples=CP_GUARD_SAMPLES,
        )
        timing_error = acquired_timing - TIMING_OFFSET_SAMPLES
        timing_errors.extend(int(x) for x in timing_error.reshape(-1).tolist())
        aligned = apply_integer_timing_offset(
            noisy_capture, offset_samples=-int(acquired_timing[0].item())
        )[..., : tx_waveform.shape[-1]]
        estimated_cfo = estimate_residual_cfo_from_cp(
            aligned,
            fft_size=int(grid.fft_size),
            cyclic_prefix_length=cp_length,
            sampling_frequency_hz=sample_rate_hz,
            guard_samples=CP_GUARD_SAMPLES,
        )
        cfo_errors.extend(float(x) for x in (estimated_cfo - CFO_HZ).reshape(-1).tolist())
        corrected = apply_residual_cfo(
            aligned,
            cfo_hz=-estimated_cfo,
            sampling_frequency_hz=sample_rate_hz,
        )
        received_grid = receiver.demodulate_slot_waveform(corrected)
        decoded, crc = receiver.decode_pusch(received_grid, torch.tensor(NOISE_VARIANCE))
        outcomes["impaired_corrected"].extend(bool(x) for x in crc.reshape(-1).tolist())
        if decoded.shape != tx_bits.shape:
            raise RuntimeError("PUSCH decoded payload shape changed unexpectedly")
        payload_bit_errors["impaired_corrected"] += int((decoded != tx_bits).sum().item())

    total = len(payload_seeds) * blocks_per_seed
    failures = {name: total - sum(values) for name, values in outcomes.items()}
    sorted_cfo_errors = sorted(abs(value) for value in cfo_errors)
    cfo_p95 = sorted_cfo_errors[math.ceil(0.95 * len(sorted_cfo_errors)) - 1]
    summary = {
        name: {
            "blocks": total,
            "crc_failures": failures[name],
            "bler": failures[name] / total,
            "payload_bit_errors": payload_bit_errors[name],
        }
        for name in outcomes
    }
    acceptance: dict[str, Any] = {
        "criteria_frozen_before_run": {
            "max_abs_timing_error_samples": 0,
            "p95_abs_cfo_error_hz": MAX_CFO_ERROR_P95_HZ,
            "max_abs_cfo_error_hz": MAX_CFO_ERROR_HZ,
            "max_clean_reference_bler": MAX_REFERENCE_BLER,
            "max_impaired_corrected_bler": MAX_IMPAIRED_BLER,
            "max_samplewise_reference_absolute_error": 2e-5,
            "min_wrong_order_rmse": 1e-4,
        },
        "observed": {
            "max_abs_timing_error_samples": max((abs(x) for x in timing_errors), default=0),
            "p95_abs_cfo_error_hz": cfo_p95,
            "max_abs_cfo_error_hz": max((abs(x) for x in cfo_errors), default=0.0),
            "max_samplewise_reference_absolute_error": max_reference_error,
            "min_wrong_order_rmse": min_wrong_order_difference,
            "clean_bler": summary["clean"]["bler"],
            "impaired_corrected_bler": summary["impaired_corrected"]["bler"],
        },
    }
    observed = acceptance["observed"]
    acceptance["pass"] = (
        observed["max_abs_timing_error_samples"] == 0
        and observed["p95_abs_cfo_error_hz"] < MAX_CFO_ERROR_P95_HZ
        and observed["max_abs_cfo_error_hz"] < MAX_CFO_ERROR_HZ
        and observed["max_samplewise_reference_absolute_error"] < 2e-5
        and observed["min_wrong_order_rmse"] > 1e-4
        and observed["clean_bler"] <= MAX_REFERENCE_BLER
        and observed["impaired_corrected_bler"] <= MAX_IMPAIRED_BLER
    )
    return {
        "schema": SCHEMA,
        "scope": "one static synthetic two-tap channel, 6-RB/30-kHz CPU PUSCH, known Wiener oscillator and memoryless TX/RX IQ-imbalance models; conditional receiver check, not deployment BLER",
        "config": {
            "sample_rate_hz": sample_rate_hz,
            "fft_size": int(grid.fft_size),
            "cyclic_prefix_samples": cp_length,
            "path_coefficients": [
                [DIRECT_COEFFICIENT.real, DIRECT_COEFFICIENT.imag],
                [ECHO_COEFFICIENT.real, ECHO_COEFFICIENT.imag],
            ],
            "path_delays_samples": [0, ECHO_DELAY_SAMPLES],
            "cfo_hz": CFO_HZ,
            "integer_capture_delay_samples": TIMING_OFFSET_SAMPLES,
            "tx_phase_noise_linewidth_hz": TX_LINEWIDTH_HZ,
            "rx_phase_noise_linewidth_hz": RX_LINEWIDTH_HZ,
            "tx_iq_gain_imbalance_db": TX_IQ_GAIN_IMBALANCE_DB,
            "tx_iq_phase_imbalance_rad": TX_IQ_PHASE_IMBALANCE_RAD,
            "rx_iq_gain_imbalance_db": RX_IQ_GAIN_IMBALANCE_DB,
            "rx_iq_phase_imbalance_rad": RX_IQ_PHASE_IMBALANCE_RAD,
            "waveform_awgn_complex_variance": NOISE_VARIANCE,
            "payload_seeds": list(payload_seeds),
            "noise_seeds": [seed + NOISE_SEED_OFFSET for seed in payload_seeds],
            "tx_phase_noise_seeds": [seed + TX_PHASE_SEED_OFFSET for seed in payload_seeds],
            "rx_phase_noise_seeds": [seed + RX_PHASE_SEED_OFFSET for seed in payload_seeds],
            "blocks_per_seed": blocks_per_seed,
        },
        "paired_outcomes": summary,
        "timing_error_samples_by_block": timing_errors,
        "cfo_error_hz_by_block": cfo_errors,
        "acceptance": acceptance,
    }


def run_ordered_impairment_probe(
    *,
    payload_seeds: tuple[int, ...] = PAYLOAD_SEEDS,
    blocks_per_seed: int = BLOCKS_PER_SEED,
) -> dict[str, Any]:
    """Run with bounded CPU threads and always restore the caller's setting."""
    original_torch_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return _run_ordered_impairment_probe(
            payload_seeds=payload_seeds, blocks_per_seed=blocks_per_seed
        )
    finally:
        torch.set_num_threads(original_torch_threads)


def _configure_probe_decoder(receiver: Any, channel_estimator: str) -> tuple[Any, dict[str, Any]]:
    """Select one frozen decoder path and capture the estimator implementation identity."""
    if channel_estimator == "sionna_default_ls":
        return receiver.decode_pusch, {"name": channel_estimator}
    if channel_estimator != "finite_delay_dft":
        raise ValueError("channel_estimator must be sionna_default_ls or finite_delay_dft")

    from experiments.g5_cp_bounded_dft_estimator import make_finite_delay_dft_decoder

    estimator_path = Path(__file__).resolve().with_name("g5_cp_bounded_dft_estimator.py")
    estimator_bytes = estimator_path.read_bytes()
    estimator_config = {
        "name": channel_estimator,
        "min_delay_samples": 0,
        "max_delay_samples": 7,
        "implementation_path": "experiments/g5_cp_bounded_dft_estimator.py",
        "implementation_sha256": _sha256(estimator_bytes),
    }
    decoder = make_finite_delay_dft_decoder(receiver, min_delay_samples=0, max_delay_samples=7)
    return decoder, estimator_config


def _independent_report_schema(
    channel_estimator: str,
    has_waveform_evidence: bool,
    has_condition_input_evidence: bool = True,
    device: str = "cpu",
) -> str:
    if device != "cpu":
        if channel_estimator == "finite_delay_dft":
            return (
                "umcd.g5_ordered_impairments_independent_channels.v24"
                if has_waveform_evidence and has_condition_input_evidence
                else "umcd.g5_ordered_impairments_independent_channels.v23"
            )
        if channel_estimator == "sionna_default_ls":
            return (
                "umcd.g5_ordered_impairments_independent_channels.v22"
                if has_waveform_evidence and has_condition_input_evidence
                else "umcd.g5_ordered_impairments_independent_channels.v21"
            )
        raise ValueError("channel_estimator must be sionna_default_ls or finite_delay_dft")
    if channel_estimator == "finite_delay_dft":
        return (
            "umcd.g5_ordered_impairments_independent_channels.v20"
            if has_waveform_evidence and has_condition_input_evidence
            else "umcd.g5_ordered_impairments_independent_channels.v18"
            if has_waveform_evidence
            else "umcd.g5_ordered_impairments_independent_channels.v17"
        )
    if channel_estimator == "sionna_default_ls":
        return (
            "umcd.g5_ordered_impairments_independent_channels.v19"
            if has_waveform_evidence and has_condition_input_evidence
            else "umcd.g5_ordered_impairments_independent_channels.v16"
            if has_waveform_evidence
            else "umcd.g5_ordered_impairments_independent_channels.v15"
        )
    raise ValueError("channel_estimator must be sionna_default_ls or finite_delay_dft")


def _validate_execution_device(device: str) -> str:
    if not isinstance(device, str) or not (device == "cpu" or device.startswith("cuda:")):
        raise ValueError("device must be cpu or an explicit CUDA device such as cuda:0")
    if device == "cpu":
        return device
    try:
        parsed_device = torch.device(device)
    except (RuntimeError, ValueError) as exc:
        raise ValueError("device must be cpu or an explicit CUDA device such as cuda:0") from exc
    if parsed_device.type != "cuda" or parsed_device.index is None or str(parsed_device) != device:
        raise ValueError("device must be cpu or an explicit CUDA device such as cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but this PyTorch runtime has no available CUDA device"
        )
    if parsed_device.index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device index {parsed_device.index} is not available")
    return device


def _run_independent_ordered_impairment_probe(
    *,
    channel_seeds: tuple[int, ...],
    blocks_per_channel: int,
    waveform_evidence_output: Path | None = None,
    channel_estimator: str = "sionna_default_ls",
    noise_variance: float = NOISE_VARIANCE,
    include_condition_inputs: bool = True,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run paired ordered-impairment PUSCH trials over independent two-path channels."""
    evidence_path = Path(waveform_evidence_output) if waveform_evidence_output is not None else None
    if not isinstance(include_condition_inputs, bool):
        raise TypeError("include_condition_inputs must be a Boolean")
    device = _validate_execution_device(device)
    if evidence_path is not None:
        if "\\" in evidence_path.name or not evidence_path.name.endswith(".npz"):
            raise ValueError("waveform evidence output must be a sibling-safe .npz filename")
        if evidence_path.exists() or evidence_path.is_symlink():
            raise FileExistsError(f"refusing to overwrite waveform evidence: {evidence_path}")
    if not channel_seeds or len(set(channel_seeds)) != len(channel_seeds):
        raise ValueError("channel_seeds must be nonempty and unique")
    if any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in channel_seeds
    ):
        raise ValueError("channel_seeds must contain nonnegative integers")
    if evidence_path is not None and any(seed > np.iinfo(np.int64).max for seed in channel_seeds):
        raise ValueError(
            "channel_seeds must fit signed 64-bit storage when waveform evidence is requested"
        )
    if isinstance(blocks_per_channel, bool) or not isinstance(blocks_per_channel, int):
        raise TypeError("blocks_per_channel must be an integer")
    if blocks_per_channel < 1:
        raise ValueError("blocks_per_channel must be positive")
    if isinstance(noise_variance, bool) or not isinstance(noise_variance, (int, float)):
        raise TypeError("noise_variance must be a real scalar")
    if not math.isfinite(noise_variance) or noise_variance <= 0:
        raise ValueError("noise_variance must be finite and positive")
    noise_variance = float(noise_variance)
    noise_variance_float32 = np.float32(noise_variance)
    if not np.isfinite(noise_variance_float32) or noise_variance_float32 <= 0:
        raise ValueError("noise_variance must be representable as finite positive float32")
    requested_transport_blocks = len(channel_seeds) * blocks_per_channel
    if requested_transport_blocks > _MAX_G5_INDEPENDENT_TRANSPORT_BLOCKS:
        raise ValueError(
            "independent-channel probe requests "
            f"{requested_transport_blocks} transport blocks, exceeding the bounded limit of "
            f"{_MAX_G5_INDEPENDENT_TRANSPORT_BLOCKS}"
        )

    config = NRPUSCHConfig(n_size_bwp=6, subcarrier_spacing_khz=30)
    receiver = SionnaPUSCHLSReceiver(config, device=device)
    decode_pusch, estimator_config = _configure_probe_decoder(receiver, channel_estimator)
    grid = receiver.resource_grid
    sample_rate_hz = float(grid.fft_size * config.subcarrier_spacing_khz * 1_000)
    cp_length = int(grid.cyclic_prefix_length)
    max_echo_delay = min(6, cp_length - 1)
    if max_echo_delay < 2:
        raise RuntimeError("cyclic prefix is too short for the declared channel family")
    timing_guard_samples = min(CP_GUARD_SAMPLES, max_echo_delay)
    cfo_guard_samples = max_echo_delay

    records: list[dict[str, Any]] = []
    evidence_tx_waveforms: list[np.ndarray] = []
    evidence_tx_phase: list[np.ndarray] = []
    evidence_rx_phase: list[np.ndarray] = []
    evidence_actual_capture: list[np.ndarray] = []
    evidence_awgn_real: list[np.ndarray] = []
    evidence_awgn_imag: list[np.ndarray] = []
    evidence_receiver_inputs: dict[str, list[np.ndarray]] = {
        "clean": [],
        "uncorrected": [],
        "corrected": [],
    }
    evidence_uncompressed_bytes = 0
    channel_rates: dict[str, list[float]] = {"clean": [], "uncorrected": [], "corrected": []}
    channel_failure_counts = {"clean": 0, "uncorrected": 0, "corrected": 0}
    timing_errors: list[int] = []
    cfo_errors: list[float] = []
    transmit_mean_powers: list[float] = []
    noisy_input_hashes: list[str] = []
    receiver_input_hashes: dict[str, list[str]] = {
        "clean": [],
        "uncorrected": [],
        "corrected": [],
    }
    max_reference_error = 0.0
    min_wrong_order_difference = math.inf

    for channel_seed in channel_seeds:
        channel_rng = random.Random(channel_seed)
        echo_delay = channel_rng.randint(2, max_echo_delay)
        echo_amplitude = channel_rng.uniform(0.20, 0.50)
        echo_phase = channel_rng.uniform(-math.pi, math.pi)
        echo_coefficient = echo_amplitude * complex(math.cos(echo_phase), math.sin(echo_phase))
        path_coefficients = (DIRECT_COEFFICIENT, echo_coefficient)
        path_delays = (0, echo_delay)

        payload_seed = channel_seed + 1_000_000
        tx_grid, tx_bits = receiver.transmit(blocks_per_channel, seed=payload_seed)
        tx_waveform = receiver.modulate_slot_grid(tx_grid)
        transmit_mean_powers.append(float(torch.mean(torch.abs(tx_waveform) ** 2).item()))
        shape = tx_waveform.shape
        tx_phase_seed = channel_seed + TX_PHASE_SEED_OFFSET + 1_000_000
        rx_phase_seed = channel_seed + RX_PHASE_SEED_OFFSET + 1_000_000
        noise_seed = channel_seed + NOISE_SEED_OFFSET + 1_000_000
        tx_phase = _phase_noise(
            shape, tx_phase_seed, sample_rate_hz, TX_LINEWIDTH_HZ, device=device
        )
        rx_phase = _phase_noise(
            shape, rx_phase_seed, sample_rate_hz, RX_LINEWIDTH_HZ, device=device
        )

        tx_iq_reference = apply_iq_imbalance(
            tx_waveform,
            gain_imbalance_db=TX_IQ_GAIN_IMBALANCE_DB,
            phase_imbalance_rad=TX_IQ_PHASE_IMBALANCE_RAD,
        )
        tx_reference = tx_iq_reference * torch.polar(torch.ones_like(tx_phase), tx_phase)
        channel_reference = torch.zeros_like(tx_reference)
        for coefficient, delay in zip(path_coefficients, path_delays, strict=True):
            delayed = torch.nn.functional.pad(tx_reference, (delay, 0))[
                ..., : tx_reference.shape[-1]
            ]
            channel_reference = channel_reference + coefficient * delayed
        sample_axis = torch.arange(tx_waveform.shape[-1], dtype=torch.float32, device=device)
        cfo_phase = 2.0 * math.pi * CFO_HZ * sample_axis / sample_rate_hz
        rx_reference = channel_reference * torch.polar(torch.ones_like(cfo_phase), cfo_phase)
        rx_reference = rx_reference * torch.polar(torch.ones_like(rx_phase), rx_phase)
        rx_reference = apply_iq_imbalance(
            rx_reference,
            gain_imbalance_db=RX_IQ_GAIN_IMBALANCE_DB,
            phase_imbalance_rad=RX_IQ_PHASE_IMBALANCE_RAD,
        )
        expected_capture = torch.cat(
            (
                torch.zeros_like(rx_reference[..., :TIMING_OFFSET_SAMPLES]),
                rx_reference[..., :-TIMING_OFFSET_SAMPLES],
            ),
            dim=-1,
        )
        actual_capture = _known_chain(
            tx_waveform,
            sample_rate_hz=sample_rate_hz,
            tx_phase=tx_phase,
            rx_phase=rx_phase,
            capture_delay=TIMING_OFFSET_SAMPLES,
            path_coefficients=path_coefficients,
            path_delays_samples=path_delays,
        )
        awgn_real, awgn_imag = _standard_complex_noise_draws(shape, noise_seed, device=device)
        noise = torch.complex(awgn_real, awgn_imag) * math.sqrt(noise_variance / 2.0)
        noisy_capture = actual_capture + noise
        noisy_input_hashes.append(
            _sha256(np.ascontiguousarray(noisy_capture.detach().cpu().numpy()).tobytes())
        )
        channel_reference_error = float(torch.max(torch.abs(actual_capture - expected_capture)))
        max_reference_error = max(max_reference_error, channel_reference_error)

        # The order-negative control applies RX oscillator noise before the
        # channel, which assigns delayed echoes the wrong oscillator phase.
        wrong_order = apply_phase_noise(tx_iq_reference, phase_noise_rad=tx_phase)
        wrong_order = apply_phase_noise(wrong_order, phase_noise_rad=rx_phase)
        wrong_order = apply_time_varying_multipath(
            wrong_order,
            path_coefficients=path_coefficients,
            path_delays_samples=path_delays,
            path_dopplers_hz=(0.0, 0.0),
            sampling_frequency_hz=sample_rate_hz,
        )
        wrong_order = apply_residual_cfo(
            wrong_order, cfo_hz=CFO_HZ, sampling_frequency_hz=sample_rate_hz
        )
        wrong_order = apply_iq_imbalance(
            wrong_order,
            gain_imbalance_db=RX_IQ_GAIN_IMBALANCE_DB,
            phase_imbalance_rad=RX_IQ_PHASE_IMBALANCE_RAD,
        )
        wrong_order = apply_integer_timing_offset(wrong_order, offset_samples=TIMING_OFFSET_SAMPLES)
        channel_wrong_order_rmse = float(
            torch.sqrt(torch.mean(torch.abs(actual_capture - wrong_order) ** 2))
        )
        min_wrong_order_difference = min(min_wrong_order_difference, channel_wrong_order_rmse)
        if waveform_evidence_output is not None:
            channel_evidence_bytes = sum(
                tensor.numel() * tensor.element_size()
                for tensor in (
                    tx_waveform,
                    tx_phase,
                    rx_phase,
                    actual_capture,
                    awgn_real,
                    awgn_imag,
                )
            ) + (
                3 * tx_waveform.numel() * tx_waveform.element_size()
                if include_condition_inputs
                else 0
            )
            evidence_uncompressed_bytes += channel_evidence_bytes
            if evidence_uncompressed_bytes + 64 * 1024 > _MAX_WAVEFORM_EVIDENCE_UNCOMPRESSED_BYTES:
                raise ValueError("uncompressed waveform evidence exceeds the configured size limit")
            evidence_tx_waveforms.append(tx_waveform.detach().cpu().numpy().copy())
            evidence_tx_phase.append(tx_phase.detach().cpu().numpy().copy())
            evidence_rx_phase.append(rx_phase.detach().cpu().numpy().copy())
            evidence_actual_capture.append(actual_capture.detach().cpu().numpy().copy())
            evidence_awgn_real.append(awgn_real.detach().cpu().numpy().copy())
            evidence_awgn_imag.append(awgn_imag.detach().cpu().numpy().copy())

        clean_channel = apply_time_varying_multipath(
            tx_waveform,
            path_coefficients=path_coefficients,
            path_delays_samples=path_delays,
            path_dopplers_hz=(0.0, 0.0),
            sampling_frequency_hz=sample_rate_hz,
        )
        clean_capture = (
            apply_integer_timing_offset(clean_channel, offset_samples=TIMING_OFFSET_SAMPLES) + noise
        )
        clean_aligned = apply_integer_timing_offset(
            clean_capture, offset_samples=-TIMING_OFFSET_SAMPLES
        )[..., : tx_waveform.shape[-1]]
        clean_grid = receiver.demodulate_slot_waveform(clean_aligned)
        receiver_noise_variance = torch.tensor(noise_variance, device=device)
        _, clean_crc = decode_pusch(clean_grid, receiver_noise_variance)

        uncorrected_grid = receiver.demodulate_slot_waveform(noisy_capture)
        _, uncorrected_crc = decode_pusch(uncorrected_grid, receiver_noise_variance)
        acquired_timing = estimate_integer_timing_offset_from_cp(
            noisy_capture,
            fft_size=int(grid.fft_size),
            cyclic_prefix_length=cp_length,
            max_offset_samples=MAX_TIMING_OFFSET_SAMPLES,
            guard_samples=timing_guard_samples,
        )
        shared_timing_error = int(acquired_timing.reshape(-1)[0].item()) - TIMING_OFFSET_SAMPLES
        channel_timing_errors = [shared_timing_error] * blocks_per_channel
        timing_errors.extend(channel_timing_errors)
        aligned = apply_integer_timing_offset(
            noisy_capture, offset_samples=-int(acquired_timing[0].item())
        )[..., : tx_waveform.shape[-1]]
        pooled_aligned = _pool_repeated_waveforms_for_shared_sync(aligned)
        estimated_cfo = estimate_residual_cfo_from_cp(
            pooled_aligned,
            fft_size=int(grid.fft_size),
            cyclic_prefix_length=cp_length,
            sampling_frequency_hz=sample_rate_hz,
            guard_samples=cfo_guard_samples,
        )
        shared_cfo_hz = float(estimated_cfo.reshape(-1)[0].item())
        channel_cfo_errors = [shared_cfo_hz - CFO_HZ] * blocks_per_channel
        cfo_errors.extend(channel_cfo_errors)
        corrected = apply_residual_cfo(
            aligned, cfo_hz=-shared_cfo_hz, sampling_frequency_hz=sample_rate_hz
        )
        if waveform_evidence_output is not None and include_condition_inputs:
            receiver_inputs = {
                "clean": clean_aligned,
                "uncorrected": noisy_capture,
                "corrected": corrected,
            }
            for condition, waveform in receiver_inputs.items():
                receiver_array = np.ascontiguousarray(waveform.detach().cpu().numpy())
                receiver_input_hashes[condition].append(_sha256(receiver_array.tobytes()))
                evidence_receiver_inputs[condition].append(receiver_array.copy())
        corrected_grid = receiver.demodulate_slot_waveform(corrected)
        _, corrected_crc = decode_pusch(corrected_grid, receiver_noise_variance)

        outcomes = {
            "clean": [bool(value) for value in clean_crc.reshape(-1).tolist()],
            "uncorrected": [bool(value) for value in uncorrected_crc.reshape(-1).tolist()],
            "corrected": [bool(value) for value in corrected_crc.reshape(-1).tolist()],
        }
        if len(outcomes["clean"]) != blocks_per_channel or any(
            len(values) != blocks_per_channel for values in outcomes.values()
        ):
            raise RuntimeError(
                "receiver CRC output count disagrees with the transport-block roster"
            )
        transport_blocks = [
            {
                "transport_block_id": f"channel-{channel_seed}:payload-{payload_seed}:block-{index}",
                "block_index": index,
                "payload_seed": payload_seed,
                "crc_pass_by_condition": {
                    condition: values[index] for condition, values in outcomes.items()
                },
                "timing_error_samples": channel_timing_errors[index],
                "cfo_error_hz": channel_cfo_errors[index],
            }
            for index in range(blocks_per_channel)
        ]
        for condition, passed in outcomes.items():
            channel_failure_counts[condition] += sum(not value for value in passed)
            channel_rates[condition].append(1.0 - sum(passed) / len(passed))
        records.append(
            {
                "channel_seed": channel_seed,
                "payload_seed": payload_seed,
                "noise_seed": noise_seed,
                "tx_phase_noise_seed": tx_phase_seed,
                "rx_phase_noise_seed": rx_phase_seed,
                "path_delays_samples": list(path_delays),
                "path_coefficients_real_imag": [
                    [value.real, value.imag] for value in path_coefficients
                ],
                "transport_blocks": transport_blocks,
                "timing_error_samples": channel_timing_errors,
                "cfo_error_hz": channel_cfo_errors,
                "samplewise_reference_max_absolute_error": channel_reference_error,
                "wrong_order_waveform_rmse": channel_wrong_order_rmse,
                "crc_pass_by_condition": outcomes,
            }
        )

    total = len(channel_seeds) * blocks_per_channel
    summary = {
        condition: {
            "blocks": total,
            "crc_failures": channel_failure_counts[condition],
            "bler": channel_failure_counts[condition] / total,
            "channel_macro_bler": sum(channel_rates[condition]) / len(channel_rates[condition]),
        }
        for condition in channel_rates
    }
    paired_deltas = [
        corrected - clean
        for clean, corrected in zip(channel_rates["clean"], channel_rates["corrected"], strict=True)
    ]
    paired_interval = _cluster_bootstrap_interval(
        paired_deltas,
        seed=INDEPENDENT_CHANNEL_BOOTSTRAP_SEED,
        draws=INDEPENDENT_CHANNEL_BOOTSTRAP_DRAWS,
    )
    absolute_cfo = sorted(abs(value) for value in cfo_errors)
    observed = {
        "max_abs_timing_error_samples": max((abs(value) for value in timing_errors), default=0),
        "p95_abs_cfo_error_hz": absolute_cfo[math.ceil(0.95 * len(absolute_cfo)) - 1],
        "max_abs_cfo_error_hz": max(absolute_cfo),
        "max_samplewise_reference_absolute_error": max_reference_error,
        "min_wrong_order_rmse": min_wrong_order_difference,
        "clean_bler": summary["clean"]["bler"],
        "impaired_corrected_bler": summary["corrected"]["bler"],
    }
    criteria = {
        "max_abs_timing_error_samples": 0,
        "p95_abs_cfo_error_hz": MAX_CFO_ERROR_P95_HZ,
        "max_abs_cfo_error_hz": MAX_CFO_ERROR_HZ,
        "max_clean_reference_bler": MAX_REFERENCE_BLER,
        "max_impaired_corrected_bler": MAX_IMPAIRED_BLER,
        "max_samplewise_reference_absolute_error": 2e-5,
        "min_wrong_order_rmse": 1e-4,
    }
    cfo_channel_tail_diagnostics = []
    for threshold_hz in (MAX_CFO_ERROR_P95_HZ, MAX_CFO_ERROR_HZ):
        channels_with_exceedance = sum(
            any(abs(error_hz) >= threshold_hz for error_hz in record["cfo_error_hz"])
            for record in records
        )
        cfo_channel_tail_diagnostics.append(
            {
                "absolute_error_threshold_hz": threshold_hz,
                "channels_with_any_block_exceedance": channels_with_exceedance,
                "independent_channels": len(records),
                "exact_one_sided_95_percent_channel_prevalence_upper_bound": (
                    _channel_any_failure_upper_bound(
                        channels_with_exceedance, len(records), alpha=0.05
                    )
                ),
            }
        )
    crc_channel_failure_diagnostics = []
    for condition in ("clean", "corrected", "uncorrected"):
        channels_with_failure = sum(
            not all(record["crc_pass_by_condition"][condition]) for record in records
        )
        crc_channel_failure_diagnostics.append(
            {
                "condition": condition,
                "channels_with_any_block_failure": channels_with_failure,
                "independent_channels": len(records),
                "blocks_per_channel": blocks_per_channel,
                "exact_one_sided_95_percent_channel_prevalence_upper_bound": (
                    _channel_any_failure_upper_bound(
                        channels_with_failure, len(records), alpha=0.05
                    )
                ),
            }
        )
    acceptance = {
        "criteria_frozen_before_run": criteria,
        "observed": observed,
        "pass": (
            observed["max_abs_timing_error_samples"] == 0
            and observed["p95_abs_cfo_error_hz"] < MAX_CFO_ERROR_P95_HZ
            and observed["max_abs_cfo_error_hz"] < MAX_CFO_ERROR_HZ
            and observed["max_samplewise_reference_absolute_error"] < 2e-5
            and observed["min_wrong_order_rmse"] > 1e-4
            and observed["clean_bler"] <= MAX_REFERENCE_BLER
            and observed["impaired_corrected_bler"] <= MAX_IMPAIRED_BLER
        ),
        "inference": {
            "unit": "independently seeded synthetic channel; blocks are nested within channel",
            "paired_corrected_minus_clean_mean_bler": sum(paired_deltas) / len(paired_deltas),
            "paired_corrected_minus_clean_channel_bootstrap_95_percent_interval": list(
                paired_interval
            ),
            "bootstrap_seed": INDEPENDENT_CHANNEL_BOOTSTRAP_SEED,
            "bootstrap_draws": INDEPENDENT_CHANNEL_BOOTSTRAP_DRAWS,
            "cfo_channel_tail_diagnostics": cfo_channel_tail_diagnostics,
            "cfo_channel_tail_interpretation": (
                "Each event is at least one block in the independent channel exceeding the "
                "listed absolute-error threshold; the exact bound is conservative and "
                "conditional on blocks_per_channel. It supplements pooled block P95/max "
                "criteria and is not a separate acceptance threshold."
            ),
            "crc_channel_failure_diagnostics": crc_channel_failure_diagnostics,
            "crc_channel_failure_interpretation": (
                "Each event is an independent channel with at least one CRC failure in the "
                "fixed repeated-block window. The exact bound describes channel-level event "
                "prevalence, not mean BLER; it is diagnostic only and is not an additional "
                "acceptance threshold."
            ),
        },
    }
    waveform_evidence = None
    if waveform_evidence_output is not None:
        if not evidence_tx_waveforms:
            raise RuntimeError(
                "waveform evidence was requested but no channel records were retained"
            )
        evidence_buffer = _BoundedBytesIO(_MAX_WAVEFORM_EVIDENCE_BYTES)
        _write_bounded_npz(
            evidence_buffer,
            channel_count=len(channel_seeds),
            scalar_arrays={
                "channel_seeds": np.asarray(channel_seeds, dtype=np.int64),
            },
            channel_arrays={
                "tx_waveform": evidence_tx_waveforms,
                "tx_phase_noise_rad": evidence_tx_phase,
                "rx_phase_noise_rad": evidence_rx_phase,
                "actual_capture": evidence_actual_capture,
                "awgn_standard_normal_real": evidence_awgn_real,
                "awgn_standard_normal_imag": evidence_awgn_imag,
                **(
                    {
                        "clean_receiver_input": evidence_receiver_inputs["clean"],
                        "uncorrected_receiver_input": evidence_receiver_inputs["uncorrected"],
                        "corrected_receiver_input": evidence_receiver_inputs["corrected"],
                    }
                    if include_condition_inputs
                    else {}
                ),
            },
        )
        evidence_bytes = evidence_buffer.getvalue()
        assert evidence_path is not None
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{evidence_path.name}.", suffix=".tmp", dir=evidence_path.parent
        )
        try:
            with os.fdopen(descriptor, "wb") as temporary_file:
                temporary_file.write(evidence_bytes)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.link(temporary_name, evidence_path)
            Path(temporary_name).unlink()
        except BaseException:
            Path(temporary_name).unlink(missing_ok=True)
            raise
        waveform_evidence = {
            "schema": (
                "umcd.g5_ordered_impairments_waveform_evidence.v3"
                if include_condition_inputs
                else "umcd.g5_ordered_impairments_waveform_evidence.v2"
            ),
            "file_name": evidence_path.name,
            "sha256": _sha256(evidence_bytes),
            "byte_count": len(evidence_bytes),
            "channel_count": len(channel_seeds),
            "arrays": sorted(
                [
                    "channel_seeds",
                    "tx_waveform",
                    "tx_phase_noise_rad",
                    "rx_phase_noise_rad",
                    "actual_capture",
                    "awgn_standard_normal_real",
                    "awgn_standard_normal_imag",
                    *(
                        [
                            "clean_receiver_input",
                            "uncorrected_receiver_input",
                            "corrected_receiver_input",
                        ]
                        if include_condition_inputs
                        else []
                    ),
                ]
            ),
        }

    return {
        "schema": _independent_report_schema(
            channel_estimator,
            waveform_evidence is not None,
            has_condition_input_evidence=include_condition_inputs,
            device=device,
        ),
        "scope": (
            f"synthetic 6-RB/30-kHz {device} PUSCH over independently seeded static two-path channels; "
            "combined IQ imbalance, Wiener phase noise, CFO, integer timing and AWGN receiver "
            "fixture with one CP-derived timing/CFO estimate shared over repeated blocks in each "
            "channel cluster, using fixed timing and family-maximum CFO guards without per-channel "
            "path truth; "
            "not deployment qualification or a field BLER estimate"
        ),
        "inference_scope": "the declared seeded synthetic channel family only",
        "config": {
            "sample_rate_hz": sample_rate_hz,
            "fft_size": int(grid.fft_size),
            "cyclic_prefix_samples": cp_length,
            "channel_seeds": list(channel_seeds),
            **({"execution_device": device} if device != "cpu" else {}),
            "blocks_per_channel": blocks_per_channel,
            "payload_seed_offset": 1_000_000,
            "noise_seed_offset": NOISE_SEED_OFFSET + 1_000_000,
            "tx_phase_noise_seed_offset": TX_PHASE_SEED_OFFSET + 1_000_000,
            "rx_phase_noise_seed_offset": RX_PHASE_SEED_OFFSET + 1_000_000,
            "echo_delay_samples_range": [2, max_echo_delay],
            "timing_cp_guard_samples": timing_guard_samples,
            "cfo_cp_guard_samples": cfo_guard_samples,
            "sync_estimation_scope": "shared_per_channel_over_repeated_blocks",
            "channel_estimator": estimator_config,
            "decoder_configuration": {
                "api": "SionnaPUSCHLSReceiver.decode_pusch",
                "channel_estimator": estimator_config,
                "crc_status_semantics": "true_means_transport_block_crc_passed",
            },
            "receiver_configuration": receiver.metadata,
            "waveform_evidence": waveform_evidence,
            "echo_amplitude_range": [0.20, 0.50],
            "direct_coefficient_real_imag": [DIRECT_COEFFICIENT.real, DIRECT_COEFFICIENT.imag],
            "cfo_hz": CFO_HZ,
            "timing_offset_samples": TIMING_OFFSET_SAMPLES,
            "tx_phase_noise_linewidth_hz": TX_LINEWIDTH_HZ,
            "rx_phase_noise_linewidth_hz": RX_LINEWIDTH_HZ,
            "tx_iq_gain_imbalance_db": TX_IQ_GAIN_IMBALANCE_DB,
            "tx_iq_phase_imbalance_rad": TX_IQ_PHASE_IMBALANCE_RAD,
            "rx_iq_gain_imbalance_db": RX_IQ_GAIN_IMBALANCE_DB,
            "rx_iq_phase_imbalance_rad": RX_IQ_PHASE_IMBALANCE_RAD,
            "waveform_awgn_complex_variance": noise_variance,
            "nominal_snr_db": -10.0 * math.log10(noise_variance),
            "nominal_snr_definition": (
                "10*log10(1/noise_variance); unit-reference complex waveform power, "
                "with measured transmit waveform power reported separately"
            ),
            "mean_transmit_waveform_power": sum(transmit_mean_powers) / len(transmit_mean_powers),
            "transmit_waveform_power_by_channel": transmit_mean_powers,
            "noisy_receiver_input_sha256_by_channel": noisy_input_hashes,
            **(
                {"receiver_input_sha256_by_condition": receiver_input_hashes}
                if waveform_evidence is not None and include_condition_inputs
                else {}
            ),
            "per_channel_draws": records,
        },
        "paired_outcomes": summary,
        "timing_error_samples_by_block": timing_errors,
        "cfo_error_hz_by_block": cfo_errors,
        "acceptance": acceptance,
    }


def run_independent_ordered_impairment_probe(
    *,
    channel_seeds: tuple[int, ...] = INDEPENDENT_CHANNEL_SEEDS,
    blocks_per_channel: int = 8,
    waveform_evidence_output: Path | None = None,
    channel_estimator: str = "sionna_default_ls",
    noise_variance: float = NOISE_VARIANCE,
    include_condition_inputs: bool = True,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run the independent-channel probe with bounded CPU threads."""
    original_torch_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return _run_independent_ordered_impairment_probe(
            channel_seeds=channel_seeds,
            blocks_per_channel=blocks_per_channel,
            waveform_evidence_output=waveform_evidence_output,
            channel_estimator=channel_estimator,
            noise_variance=noise_variance,
            include_condition_inputs=include_condition_inputs,
            device=device,
        )
    finally:
        torch.set_num_threads(original_torch_threads)


def run_independent_ordered_impairment_snr_curve(
    *,
    snr_db: Sequence[float],
    channel_seeds: tuple[int, ...] = INDEPENDENT_CHANNEL_SEEDS,
    blocks_per_channel: int = 8,
    channel_estimator: str = "sionna_default_ls",
    waveform_evidence_output: Path | None = None,
    acceptance_criteria: dict[str, Any] | None = None,
    preregistration_signature: dict[str, str] | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run a bounded paired diagnostic curve over common channel/payload seeds.

    Each SNR point uses the same channel, payload, oscillator, and standardized
    AWGN random-number streams. Only AWGN amplitude changes. The convention is
    ``noise_variance = 10**(-snr_db/10)`` for unit-reference complex waveform
    power; the realized transmit waveform power is retained at every point.
    Optional evidence retains shared TX, oscillator, pre-AWGN capture, and
    standardized AWGN draws once, then stores the exact clean, uncorrected,
    and corrected receiver inputs for every SNR point. The auditor binds each
    point to its evidence index and noise realization. This curve is
    descriptive unless an exact, bounded acceptance contract is supplied before
    the run. A contract decision applies only to this synthetic fixture; it is
    not field or deployment qualification.
    """
    device = _validate_execution_device(device)
    if isinstance(snr_db, (str, bytes)) or not (
        isinstance(snr_db, Sequence) or isinstance(snr_db, np.ndarray)
    ):
        raise TypeError("snr_db must be a bounded sequence of real scalars")
    if isinstance(snr_db, np.ndarray) and snr_db.ndim != 1:
        raise ValueError("snr_db array must be one-dimensional")
    if len(snr_db) > _MAX_G5_INDEPENDENT_SNR_POINTS:
        raise ValueError(
            f"independent-channel SNR curve has {len(snr_db)} points, exceeding the "
            f"bounded limit of {_MAX_G5_INDEPENDENT_SNR_POINTS}"
        )
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in snr_db):
        raise TypeError("snr_db values must be real scalars")
    snr_values = tuple(float(value) for value in snr_db)
    if not snr_values or any(not math.isfinite(value) for value in snr_values):
        raise ValueError("snr_db must contain finite values")
    if any(right <= left for left, right in zip(snr_values, snr_values[1:])):
        raise ValueError("snr_db must be strictly increasing without duplicates")
    if isinstance(blocks_per_channel, bool) or not isinstance(blocks_per_channel, int):
        raise TypeError("blocks_per_channel must be an integer")
    if blocks_per_channel < 1:
        raise ValueError("blocks_per_channel must be positive")
    if isinstance(channel_seeds, (str, bytes)) or not isinstance(channel_seeds, Sequence):
        raise TypeError("channel_seeds must be a bounded sequence of integers")
    if not channel_seeds:
        raise ValueError("channel_seeds must be nonempty and unique")
    if len(channel_seeds) > _MAX_G5_INDEPENDENT_TRANSPORT_BLOCKS // (
        len(snr_values) * blocks_per_channel
    ):
        requested = len(snr_values) * len(channel_seeds) * blocks_per_channel
        raise ValueError(
            "independent-channel SNR curve requests "
            f"{requested} transport blocks, exceeding the bounded limit of "
            f"{_MAX_G5_INDEPENDENT_TRANSPORT_BLOCKS}"
        )
    if any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in channel_seeds
    ):
        raise ValueError("channel_seeds must contain nonnegative integers")
    if len(set(channel_seeds)) != len(channel_seeds):
        raise ValueError("channel_seeds must be nonempty and unique")
    if not isinstance(channel_estimator, str):
        raise TypeError("channel_estimator must be a string")
    if channel_estimator not in {"sionna_default_ls", "finite_delay_dft"}:
        raise ValueError("channel_estimator must be sionna_default_ls or finite_delay_dft")
    frozen_acceptance = _freeze_independent_snr_acceptance_criteria(
        acceptance_criteria,
        snr_values=snr_values,
        channel_seeds=channel_seeds,
        channel_estimator=channel_estimator,
        blocks_per_channel=blocks_per_channel,
    )
    if preregistration_signature is not None and frozen_acceptance is None:
        raise ValueError("a preregistration signature requires an acceptance contract")
    if (
        frozen_acceptance is not None
        and frozen_acceptance.get("schema") == G5_SNR_TIMESTAMPED_ACCEPTANCE_SCHEMA
        and preregistration_signature is None
    ):
        raise ValueError("timestamped acceptance contract v2 requires a verified signature")
    receiver_run_started_utc = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    preregistration_order: dict[str, str] | None = None
    if preregistration_signature is not None:
        required_signature_fields = {
            "status",
            "namespace",
            "signer_id",
            "contract_payload_sha256",
            "signature_sha256",
            "trusted_allowed_signers_sha256",
            "trust_file_custody",
        }
        if frozen_acceptance is not None and frozen_acceptance.get("schema") == (
            G5_SNR_TIMESTAMPED_ACCEPTANCE_SCHEMA
        ):
            required_signature_fields.add("preregistered_utc")
        reviewer_policy_identity = (
            preregistration_signature.get("reviewer_policy")
            if isinstance(preregistration_signature, dict)
            else None
        )
        if reviewer_policy_identity is not None:
            required_signature_fields.add("reviewer_policy")
        if (
            not isinstance(preregistration_signature, dict)
            or set(preregistration_signature) != required_signature_fields
            or preregistration_signature.get("status") != "verified_before_receiver_run"
            or preregistration_signature.get("namespace") != G5_SNR_ACCEPTANCE_SIGNATURE_NAMESPACE
            or not isinstance(preregistration_signature.get("signer_id"), str)
            or not preregistration_signature["signer_id"].strip()
            or preregistration_signature.get("contract_payload_sha256")
            != _sha256(_canonical_json_bytes(frozen_acceptance))
            or preregistration_signature.get("trust_file_custody")
            != "not_established_by_this_verifier"
            or any(
                not isinstance(preregistration_signature.get(field), str)
                or len(preregistration_signature[field]) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in preregistration_signature[field]
                )
                for field in ("signature_sha256", "trusted_allowed_signers_sha256")
            )
        ):
            raise ValueError("preregistration signature identity does not bind the frozen contract")
        if reviewer_policy_identity is not None:
            expected_policy_fields = {
                "status",
                "policy_id",
                "policy_sha256",
                "reviewer_principal",
                "reviewer_organization",
                "reviewer_roles",
                "purpose",
                "custodian_principal",
                "signature_namespace",
                "signature_sha256",
                "trusted_allowed_signers_sha256",
                "trust_file_custody",
            }
            if (
                frozen_acceptance.get("schema") != G5_SNR_TIMESTAMPED_ACCEPTANCE_SCHEMA
                or not isinstance(reviewer_policy_identity, dict)
                or set(reviewer_policy_identity) != expected_policy_fields
                or reviewer_policy_identity.get("status") != "verified"
                or reviewer_policy_identity.get("reviewer_principal")
                != preregistration_signature.get("signer_id")
                or "independent_reviewer"
                not in str(reviewer_policy_identity.get("reviewer_roles", "")).split(",")
                or reviewer_policy_identity.get("purpose") != G5_SNR_REVIEWER_POLICY_PURPOSE
                or reviewer_policy_identity.get("signature_namespace")
                != G5_SNR_REVIEWER_POLICY_SIGNATURE_NAMESPACE
                or reviewer_policy_identity.get("custodian_principal")
                == preregistration_signature.get("signer_id")
                or reviewer_policy_identity.get("trust_file_custody")
                != "not_established_by_this_verifier"
                or any(
                    not isinstance(reviewer_policy_identity.get(field), str)
                    or len(reviewer_policy_identity[field]) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in reviewer_policy_identity[field]
                    )
                    for field in (
                        "policy_sha256",
                        "signature_sha256",
                        "trusted_allowed_signers_sha256",
                    )
                )
            ):
                raise ValueError("G5 independent reviewer policy identity is invalid")
        if frozen_acceptance.get("schema") == G5_SNR_TIMESTAMPED_ACCEPTANCE_SCHEMA:
            if preregistration_signature.get("preregistered_utc") != frozen_acceptance.get(
                "preregistered_utc"
            ):
                raise ValueError(
                    "preregistration signature timestamp differs from the frozen contract"
                )
            preregistration_order = validate_g5_preregistration_order(
                frozen_acceptance, receiver_run_started_utc=receiver_run_started_utc
            )
            preregistration_signature = {
                **preregistration_signature,
                **preregistration_order,
            }
    cpu_curve_schema_version = (
        11
        if isinstance(preregistration_signature, dict)
        and "reviewer_policy" in preregistration_signature
        else 10
        if preregistration_order is not None
        else 8
        if preregistration_signature is not None
        else 7
        if frozen_acceptance is not None
        else 9
    )
    cuda_curve_schema_versions = {9: 12, 7: 13, 8: 14, 10: 15, 11: 16}
    curve_schema_version = (
        cuda_curve_schema_versions[cpu_curve_schema_version]
        if device != "cpu"
        else cpu_curve_schema_version
    )
    evidence_path = Path(waveform_evidence_output) if waveform_evidence_output is not None else None
    if evidence_path is not None:
        if "\\" in evidence_path.name or not evidence_path.name.endswith(".npz"):
            raise ValueError("waveform evidence output must be a sibling-safe .npz filename")
        if evidence_path.exists() or evidence_path.is_symlink():
            raise FileExistsError(f"refusing to overwrite waveform evidence: {evidence_path}")
    requested_transport_blocks = len(snr_values) * len(channel_seeds) * blocks_per_channel
    if requested_transport_blocks > _MAX_G5_INDEPENDENT_TRANSPORT_BLOCKS:
        raise ValueError(
            "independent-channel SNR curve requests "
            f"{requested_transport_blocks} transport blocks, exceeding the bounded limit of "
            f"{_MAX_G5_INDEPENDENT_TRANSPORT_BLOCKS}"
        )

    variances: list[float] = []
    for snr in snr_values:
        try:
            variance = math.pow(10.0, -snr / 10.0)
        except OverflowError as error:
            raise ValueError(
                f"SNR {snr} dB does not yield a finite positive noise variance"
            ) from error
        if not math.isfinite(variance) or variance <= 0:
            raise ValueError(f"SNR {snr} dB does not yield a finite positive noise variance")
        if not np.isfinite(np.float32(variance)) or np.float32(variance) <= 0:
            raise ValueError(f"SNR {snr} dB noise variance is not representable as float32")
        variances.append(variance)

    points: list[dict[str, Any]] = []
    evidence_manifest: dict[str, Any] | None = None
    staging_directory: str | None = None
    staging_evidence_path: Path | None = None
    staging_point_evidence_paths: list[Path] = []
    staging_point_evidence_bytes = 0
    try:
        if evidence_path is not None:
            evidence_path.parent.mkdir(parents=True, exist_ok=True)
            staging_directory = tempfile.mkdtemp(
                prefix=f".{evidence_path.stem}-", dir=evidence_path.parent
            )
            staging_evidence_path = Path(staging_directory) / evidence_path.name
        for index, (snr, variance) in enumerate(zip(snr_values, variances, strict=True)):
            point_evidence_path = None
            if staging_directory is not None:
                assert evidence_path is not None
                point_evidence_path = (
                    Path(staging_directory) / f"{evidence_path.stem}.point-{index:04d}.npz"
                )
                staging_point_evidence_paths.append(point_evidence_path)
            report = run_independent_ordered_impairment_probe(
                channel_seeds=channel_seeds,
                blocks_per_channel=blocks_per_channel,
                channel_estimator=channel_estimator,
                noise_variance=variance,
                waveform_evidence_output=point_evidence_path,
                include_condition_inputs=point_evidence_path is not None,
                device=device,
            )
            config = report.get("config")
            if not isinstance(config, dict):
                raise ValueError("SNR point report has no configuration object")
            if point_evidence_path is not None:
                manifest = config.get("waveform_evidence")
                if not isinstance(manifest, dict):
                    raise ValueError("SNR point did not retain requested waveform evidence")
                byte_count = manifest.get("byte_count")
                if (
                    isinstance(byte_count, bool)
                    or not isinstance(byte_count, int)
                    or byte_count < 1
                ):
                    raise ValueError("SNR point waveform evidence has an invalid byte count")
                staging_point_evidence_bytes += byte_count
                if staging_point_evidence_bytes > 2 * _MAX_WAVEFORM_EVIDENCE_BYTES:
                    raise ValueError(
                        "multi-SNR waveform evidence staging exceeds its disk size limit"
                    )
            points.append({"snr_db": snr, "experiment": report})

        if staging_evidence_path is not None:
            evidence_manifest = _combine_snr_curve_waveform_evidence(
                tuple(staging_point_evidence_paths),
                output_path=staging_evidence_path,
                snr_values=snr_values,
            )
            suffix = (
                "24"
                if device != "cpu" and channel_estimator == "finite_delay_dft"
                else "23"
                if device != "cpu"
                else "20"
                if channel_estimator == "finite_delay_dft"
                else "19"
            )
            for point in points:
                point_report = point["experiment"]
                point_config = point_report["config"]
                point_config["waveform_evidence"] = dict(evidence_manifest)
                point_report["schema"] = (
                    f"umcd.g5_ordered_impairments_independent_channels.v{suffix}"
                )

        curve_inference = _independent_snr_curve_cluster_inference(
            [point["experiment"] for point in points],
            snr_values,
            seed=INDEPENDENT_SNR_CURVE_BOOTSTRAP_SEED,
            draws=INDEPENDENT_SNR_CURVE_BOOTSTRAP_DRAWS,
            include_exact_seed_roster=curve_schema_version in {7, 8, 10, 13, 14, 15},
        )
        if staging_evidence_path is not None and evidence_path is not None:
            os.link(staging_evidence_path, evidence_path)
            staging_evidence_path.unlink()
    finally:
        if staging_directory is not None:
            try:
                if staging_evidence_path is not None:
                    staging_evidence_path.unlink(missing_ok=True)
                for point_path in staging_point_evidence_paths:
                    point_path.unlink(missing_ok=True)
                Path(staging_directory).rmdir()
            except OSError:
                # Do not mask the receiver or evidence-publication failure.
                pass
    schema_version = curve_schema_version
    acceptance = (
        _evaluate_independent_snr_curve_acceptance(
            frozen_acceptance,
            curve_inference=curve_inference,
            preregistration_signature=preregistration_signature,
        )
        if frozen_acceptance is not None
        else {
            "criteria_frozen_before_run": {
                "status": "descriptive_only",
                "reason": ("no acceptance contract was supplied before the SNR curve run"),
            },
            "pass": False,
        }
    )
    return {
        "schema": (f"umcd.g5_ordered_impairments_independent_channels_snr_curve.v{schema_version}"),
        "scope": (
            "paired diagnostic SNR curve over a seeded synthetic two-path PUSCH fixture; "
            "same channel, payload, oscillator, and standardized AWGN draws across points; "
            + (
                "uses a caller-supplied pre-run acceptance contract; "
                "not field or deployment qualification"
                if frozen_acceptance is not None
                else "no acceptance contract; not a field BLER estimate"
            )
        ),
        "inference_scope": (
            "simultaneous channel-cluster inference against a caller-supplied pre-run contract"
            if frozen_acceptance is not None
            else (
                "descriptive pointwise and simultaneous channel-cluster uncertainty; "
                "no acceptance claim"
            )
        ),
        "config": {
            "schema_sha256": _sha256(
                (
                    Path(__file__).resolve().parent.parent
                    / "datasets"
                    / "schema"
                    / f"g5-ordered-impairments-snr-curve.v{curve_schema_version}.schema.json"
                ).read_bytes()
            ),
            "snr_db": list(snr_values),
            "channel_seeds": list(channel_seeds),
            "blocks_per_channel": blocks_per_channel,
            "channel_estimator": channel_estimator,
            "shared_random_streams_across_snr": [
                "channel",
                "payload",
                "TX phase noise",
                "RX phase noise",
                "standardized AWGN",
            ],
            "snr_definition": (
                "noise variance per complex sample = 10**(-SNR_dB/10), "
                "referenced to unit complex waveform power; measured TX waveform power is "
                "reported at each point"
            ),
            "transport_blocks_per_snr_point": len(channel_seeds) * blocks_per_channel,
            "transport_blocks_across_curve": requested_transport_blocks,
            "maximum_transport_blocks_across_curve": _MAX_G5_INDEPENDENT_TRANSPORT_BLOCKS,
            "maximum_snr_points": _MAX_G5_INDEPENDENT_SNR_POINTS,
            **({"execution_device": device} if device != "cpu" else {}),
            "waveform_evidence": evidence_manifest,
        },
        "acceptance": acceptance,
        "curve_inference": curve_inference,
        "points": points,
    }


def _independent_snr_curve_cluster_inference(
    reports: list[dict[str, Any]],
    snr_values: tuple[float, ...],
    *,
    seed: int,
    draws: int,
    include_exact_seed_roster: bool = False,
) -> dict[str, Any]:
    """Compute pointwise and simultaneous BLER bands by resampling channels."""
    conditions = ("clean", "uncorrected", "corrected")
    first_config = reports[0].get("config")
    channel_seeds = first_config.get("channel_seeds") if isinstance(first_config, dict) else None
    if (
        not isinstance(channel_seeds, list)
        or not channel_seeds
        or any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            for seed in channel_seeds
        )
        or len(set(channel_seeds)) != len(channel_seeds)
    ):
        raise ValueError("SNR-curve inference requires a canonical channel-seed roster")
    for report in reports:
        config = report.get("config") if isinstance(report, dict) else None
        records = config.get("per_channel_draws") if isinstance(config, dict) else None
        if (
            not isinstance(config, dict)
            or config.get("channel_seeds") != channel_seeds
            or not isinstance(records, list)
            or [row.get("channel_seed") for row in records if isinstance(row, dict)]
            != channel_seeds
        ):
            raise ValueError("SNR-curve points do not share the exact channel-seed roster")
    channel_count = len(channel_seeds)
    blocks_per_channel = reports[0]["config"]["blocks_per_channel"]
    cells: dict[str, list[dict[str, float | None]]] = {condition: [] for condition in conditions}
    if channel_count < 2:
        return {
            "status": "unavailable",
            "unit": "independent synthetic channel; transport blocks are nested",
            "cluster_count": channel_count,
            "bootstrap_seed": seed,
            "bootstrap_draws": draws,
            "family_size": len(snr_values) * (len(conditions) + 1),
            "method": (
                "channel-cluster percentile intervals with simultaneous max-deviation band "
                "over conditions and paired correction effects"
            ),
            "paired_contrast_definition": (
                "uncorrected macro-BLER minus corrected macro-BLER; positive values favor correction"
            ),
            "reason": "at least two independent channel clusters are required for resampling",
            "by_condition": cells,
            "by_paired_contrast": [],
            **({"channel_seeds": list(channel_seeds)} if include_exact_seed_roster else {}),
        }

    rates = np.empty((len(reports), len(conditions), channel_count), dtype=np.float64)
    for snr_index, report in enumerate(reports):
        records = report["config"]["per_channel_draws"]
        for condition_index, condition in enumerate(conditions):
            for channel_index, record in enumerate(records):
                outcomes = record["crc_pass_by_condition"][condition]
                rates[snr_index, condition_index, channel_index] = 1.0 - (
                    sum(outcomes) / blocks_per_channel
                )
    observed = rates.mean(axis=2)
    paired_improvement_rates = rates[:, 1, :] - rates[:, 2, :]
    observed_paired_improvement = paired_improvement_rates.mean(axis=1)
    rng = np.random.default_rng(seed)
    batches: list[np.ndarray] = []
    paired_improvement_batches: list[np.ndarray] = []
    batch_size = max(1, min(G5_SNR_CURVE_BOOTSTRAP_BATCH_SIZE, draws))
    for start in range(0, draws, batch_size):
        size = min(batch_size, draws - start)
        sampled_channels = rng.integers(0, channel_count, size=(size, channel_count))
        sampled_means = rates[:, :, sampled_channels].mean(axis=-1)
        batches.append(np.moveaxis(sampled_means, -1, 0))
        sampled_improvements = paired_improvement_rates[:, sampled_channels].mean(axis=-1)
        paired_improvement_batches.append(np.moveaxis(sampled_improvements, -1, 0))
    bootstrap = np.concatenate(batches, axis=0)
    paired_improvement_bootstrap = np.concatenate(paired_improvement_batches, axis=0)
    lower_index = math.floor(0.025 * draws)
    upper_index = min(draws - 1, math.ceil(0.975 * draws) - 1)
    condition_deviations = np.max(np.abs(bootstrap - observed[None, :, :]), axis=(1, 2))
    paired_deviations = np.max(
        np.abs(paired_improvement_bootstrap - observed_paired_improvement[None, :]), axis=1
    )
    deviations = np.maximum(condition_deviations, paired_deviations)
    simultaneous_radius = float(np.quantile(deviations, 0.95, method="higher"))
    for condition_index, condition in enumerate(conditions):
        for snr_index, snr in enumerate(snr_values):
            ordered = np.sort(bootstrap[:, snr_index, condition_index])
            point_lower = float(ordered[lower_index])
            point_upper = float(ordered[upper_index])
            estimate = float(observed[snr_index, condition_index])
            cells[condition].append(
                {
                    "snr_db": snr,
                    "mean_channel_macro_bler": estimate,
                    "pointwise_95_percent_interval": [point_lower, point_upper],
                    "simultaneous_95_percent_interval": [
                        max(0.0, estimate - simultaneous_radius),
                        min(1.0, estimate + simultaneous_radius),
                    ],
                }
            )
    paired_contrast_rows = []
    for snr_index, snr in enumerate(snr_values):
        ordered = np.sort(paired_improvement_bootstrap[:, snr_index])
        estimate = float(observed_paired_improvement[snr_index])
        paired_contrast_rows.append(
            {
                "snr_db": snr,
                "mean_channel_macro_bler_improvement": estimate,
                "pointwise_95_percent_interval": [
                    float(ordered[lower_index]),
                    float(ordered[upper_index]),
                ],
                "simultaneous_95_percent_interval": [
                    max(-1.0, estimate - simultaneous_radius),
                    min(1.0, estimate + simultaneous_radius),
                ],
            }
        )
    return {
        "status": "available",
        "unit": "independent synthetic channel; transport blocks are nested",
        "cluster_count": channel_count,
        "bootstrap_seed": seed,
        "bootstrap_draws": draws,
        "family_size": len(snr_values) * (len(conditions) + 1),
        "method": (
            "channel-cluster percentile intervals with simultaneous max-deviation band "
            "over conditions and paired correction effects"
        ),
        "paired_contrast_definition": (
            "uncorrected macro-BLER minus corrected macro-BLER; positive values favor correction"
        ),
        "simultaneous_radius": simultaneous_radius,
        "by_condition": cells,
        "by_paired_contrast": paired_contrast_rows,
        **({"channel_seeds": list(channel_seeds)} if include_exact_seed_roster else {}),
    }


def _canonical_json_bytes(value: Any) -> bytes:
    """Return the single canonical byte representation used for result hashes."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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
    }:
        curve_schema_version = experiment["schema"].rsplit("v", maxsplit=1)[1]
        schema_path = (
            Path(__file__).resolve().parent.parent
            / "datasets"
            / "schema"
            / f"g5-ordered-impairments-snr-curve.v{curve_schema_version}.schema.json"
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
