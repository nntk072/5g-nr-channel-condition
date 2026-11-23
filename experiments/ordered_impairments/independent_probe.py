"""Independent probe for ordered impairments receiver."""

from __future__ import annotations

import importlib
import importlib.metadata
import math
import os
import random
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from integrations.sionna.nr_pusch import SionnaPUSCHLSReceiver
from integrations.sionna.ofdm_impairments import (
    apply_integer_timing_offset,
    apply_iq_imbalance,
    apply_phase_noise,
    apply_residual_cfo,
    apply_time_varying_multipath,
    estimate_integer_timing_offset_from_cp,
    estimate_residual_cfo_from_cp,
)
from receivers.nr import NRPUSCHConfig

if TYPE_CHECKING:
    import torch
else:

    class _LazyTorch:
        def __getattr__(self, name: str) -> Any:
            module = importlib.import_module("torch")
            globals()["torch"] = module
            return getattr(module, name)

    torch = _LazyTorch()

from .constants import (
    _MAX_INDEPENDENT_TRANSPORT_BLOCKS,
    _MAX_WAVEFORM_EVIDENCE_BYTES,
    _MAX_WAVEFORM_EVIDENCE_UNCOMPRESSED_BYTES,
    CFO_HZ,
    CP_GUARD_SAMPLES,
    DIRECT_COEFFICIENT,
    INDEPENDENT_CHANNEL_BOOTSTRAP_DRAWS,
    INDEPENDENT_CHANNEL_BOOTSTRAP_SEED,
    INDEPENDENT_CHANNEL_SEEDS,
    MAX_CFO_ERROR_HZ,
    MAX_CFO_ERROR_P95_HZ,
    MAX_IMPAIRED_BLER,
    MAX_REFERENCE_BLER,
    MAX_TIMING_OFFSET_SAMPLES,
    NOISE_SEED_OFFSET,
    NOISE_VARIANCE,
    RX_IQ_GAIN_IMBALANCE_DB,
    RX_IQ_PHASE_IMBALANCE_RAD,
    RX_LINEWIDTH_HZ,
    RX_PHASE_SEED_OFFSET,
    TIMING_OFFSET_SAMPLES,
    TX_IQ_GAIN_IMBALANCE_DB,
    TX_IQ_PHASE_IMBALANCE_RAD,
    TX_LINEWIDTH_HZ,
    TX_PHASE_SEED_OFFSET,
)
from .hashing import (
    _sha256,
)
from .paired_probe import (
    _configure_probe_decoder,
)
from .signal_models import (
    _known_chain,
    _phase_noise,
    _standard_complex_noise_draws,
)
from .statistics import (
    _channel_any_failure_upper_bound,
    _cluster_bootstrap_interval,
    _pool_repeated_waveforms_for_shared_sync,
)
from .waveform_evidence import (
    _BoundedBytesIO,
    _write_bounded_npz,
)


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
    if requested_transport_blocks > _MAX_INDEPENDENT_TRANSPORT_BLOCKS:
        raise ValueError(
            "independent-channel probe requests "
            f"{requested_transport_blocks} transport blocks, exceeding the bounded limit of "
            f"{_MAX_INDEPENDENT_TRANSPORT_BLOCKS}"
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
