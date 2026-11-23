"""Paired probe for ordered impairments receiver."""

from __future__ import annotations

import importlib
import importlib.metadata
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
    BLOCKS_PER_SEED,
    CFO_HZ,
    CP_GUARD_SAMPLES,
    DIRECT_COEFFICIENT,
    ECHO_COEFFICIENT,
    ECHO_DELAY_SAMPLES,
    MAX_CFO_ERROR_HZ,
    MAX_CFO_ERROR_P95_HZ,
    MAX_IMPAIRED_BLER,
    MAX_REFERENCE_BLER,
    MAX_TIMING_OFFSET_SAMPLES,
    NOISE_SEED_OFFSET,
    NOISE_VARIANCE,
    PAYLOAD_SEEDS,
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
from .hashing import (
    _sha256,
)
from .signal_models import (
    _known_chain,
    _phase_noise,
    _proper_complex_noise,
)
from .statistics import (
    _pool_repeated_waveforms_for_shared_sync,
)


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

    from experiments.cp_bounded_dft_estimator import make_finite_delay_dft_decoder

    estimator_path = Path(__file__).resolve().parents[1] / "cp_bounded_dft_estimator.py"
    estimator_bytes = estimator_path.read_bytes()
    estimator_config = {
        "name": channel_estimator,
        "min_delay_samples": 0,
        "max_delay_samples": 7,
        "implementation_path": "experiments/cp_bounded_dft_estimator.py",
        "implementation_sha256": _sha256(estimator_bytes),
    }
    decoder = make_finite_delay_dft_decoder(receiver, min_delay_samples=0, max_delay_samples=7)
    return decoder, estimator_config
