"""Reproducible Sionna TDL-PUSCH link-level BLER curves."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Sequence

from artifacts.provenance import package_source_sha256, versions


def wilson_score_interval_95(errors: int, total: int) -> tuple[float, float]:
    """Return a two-sided 95% Wilson score interval for a binomial rate."""
    if isinstance(errors, bool) or not isinstance(errors, int):
        raise TypeError("errors must be an integer")
    if isinstance(total, bool) or not isinstance(total, int):
        raise TypeError("total must be an integer")
    if total <= 0 or not 0 <= errors <= total:
        raise ValueError("require total > 0 and 0 <= errors <= total")

    z = 1.959963984540054
    if errors == 0:
        return 0.0, z**2 / (total + z**2)
    if errors == total:
        return total / (total + z**2), 1.0
    rate = errors / total
    denominator = 1.0 + z**2 / total
    center = (rate + z**2 / (2.0 * total)) / denominator
    radius = z * math.sqrt(rate * (1.0 - rate) / total + z**2 / (4.0 * total**2)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def run_tdl_pusch_bler_curve(
    *,
    snr_db: Sequence[float],
    frames_per_snr: int,
    batch_size: int,
    seed: int,
    profile: str = "A",
    delay_spread_s: float = 100e-9,
    carrier_frequency_hz: float = 3.5e9,
    speed_mps: float = 3.0,
    n_size_bwp: int = 2,
    subcarrier_spacing_khz: int = 30,
) -> dict[str, Any]:
    """Measure PUSCH BLER through a seeded, time-varying Sionna TDL channel.

    The Sionna ``OFDMChannel`` path provides TDL fading across OFDM symbols
    with perfect synchronization and no ICI/ISI. Noise variance is mapped from
    the nominal unit-energy resource-grid SNR, ``10**(-SNR_dB/10)``. Identical
    data, channel, and standardized noise seeds are reused at every SNR point
    (common random numbers), enabling paired comparisons across the curve.
    """
    snr_values = tuple(float(value) for value in snr_db)
    if not snr_values or any(not math.isfinite(value) for value in snr_values):
        raise ValueError("snr_db must contain finite values")
    if isinstance(frames_per_snr, bool) or not isinstance(frames_per_snr, int):
        raise TypeError("frames_per_snr must be an integer")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer")
    if frames_per_snr <= 0 or batch_size <= 0:
        raise ValueError("frames_per_snr and batch_size must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not profile or profile.upper() not in {"A", "B", "C", "D", "E"}:
        raise ValueError("profile must be one of TDL-A through TDL-E")
    if not math.isfinite(delay_spread_s) or delay_spread_s <= 0:
        raise ValueError("delay_spread_s must be finite and positive")
    if not math.isfinite(carrier_frequency_hz) or carrier_frequency_hz <= 0:
        raise ValueError("carrier_frequency_hz must be finite and positive")
    if not math.isfinite(speed_mps) or speed_mps < 0:
        raise ValueError("speed_mps must be finite and nonnegative")

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install PyTorch to run the TDL-PUSCH link curve") from exc

    from integrations.sionna.nr_pusch import SionnaPUSCHLSReceiver
    from integrations.sionna.tdl_pusch import (
        SionnaPUSCHLinkDecoder,
        generate_tdl_pusch_batch,
    )
    from receivers.nr import NRPUSCHConfig

    if not torch.cuda.is_available():
        raise RuntimeError("The TDL-PUSCH curve requires the available CUDA environment")
    device = "cuda:0"
    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(
            n_size_bwp=n_size_bwp,
            subcarrier_spacing_khz=subcarrier_spacing_khz,
        ),
        device=device,
    )
    decoder = SionnaPUSCHLinkDecoder(receiver)

    points: list[dict[str, Any]] = []
    batch_seeds = [seed + index for index in range(math.ceil(frames_per_snr / batch_size))]
    matched_reference: dict[int, tuple[Any, Any]] = {}
    matched_checks: dict[int, dict[str, Any]] = {}
    for snr in snr_values:
        noise_variance = 10.0 ** (-snr / 10.0)
        frame_errors = 0
        bit_errors = 0
        frames = 0
        bits = 0
        for batch_seed in batch_seeds:
            count = min(batch_size, frames_per_snr - frames)
            batch = generate_tdl_pusch_batch(
                receiver,
                model=profile.upper(),
                batch_size=count,
                seed=batch_seed,
                carrier_frequency_hz=carrier_frequency_hz,
                delay_spread_s=delay_spread_s,
                noise_variance=noise_variance,
                min_speed_mps=speed_mps,
                max_speed_mps=speed_mps,
            )
            reference_bits = batch.transmit_bits.detach().cpu().contiguous()
            reference_channel = batch.channel_truth.detach().cpu().contiguous()
            if batch_seed not in matched_reference:
                matched_reference[batch_seed] = (reference_bits, reference_channel)
                matched_checks[batch_seed] = {
                    "transmit_bits_sha256": hashlib.sha256(
                        reference_bits.numpy().tobytes()
                    ).hexdigest(),
                    "channel_truth_sha256": hashlib.sha256(
                        reference_channel.numpy().tobytes()
                    ).hexdigest(),
                    "transmit_bits_match_across_snr": True,
                    "channel_truth_max_abs_delta": 0.0,
                    "channel_truth_allclose_across_snr": True,
                }
            else:
                expected_bits, expected_channel = matched_reference[batch_seed]
                bits_match = torch.equal(reference_bits, expected_bits)
                channel_match = torch.allclose(
                    reference_channel, expected_channel, rtol=1e-5, atol=1e-6
                )
                channel_delta = float(torch.max(torch.abs(reference_channel - expected_channel)))
                check = matched_checks[batch_seed]
                check["channel_truth_max_abs_delta"] = max(
                    float(check["channel_truth_max_abs_delta"]), channel_delta
                )
                check["transmit_bits_match_across_snr"] = bool(
                    check.get("transmit_bits_match_across_snr", True) and bits_match
                )
                check["channel_truth_allclose_across_snr"] = bool(
                    check["channel_truth_allclose_across_snr"] and channel_match
                )
                if not bits_match or not channel_match:
                    raise RuntimeError("matched SNR points did not reproduce bits/channel truth")
            decoded_bits, crc_ok = decoder(batch.received_grid, batch.noise_variance)
            valid = crc_ok.reshape(-1).to(dtype=torch.bool)
            target_bits = batch.transmit_bits
            bit_errors += int(torch.count_nonzero(decoded_bits != target_bits).item())
            bits += int(target_bits.numel())
            frame_errors += int(torch.count_nonzero(~valid).item())
            frames += int(valid.numel())

        lower, upper = wilson_score_interval_95(frame_errors, frames)
        points.append(
            {
                "snr_db": snr,
                "noise_variance_per_complex_dimension": noise_variance,
                "frames": frames,
                "bler_errors": frame_errors,
                "bler": frame_errors / frames,
                "bler_wilson_95_ci": [lower, upper],
                "bit_errors": bit_errors,
                "bits": bits,
                "ber": bit_errors / bits,
            }
        )

    return {
        "schema": "loslab.tdl_pusch_bler_curve.v1",
        "evidence_grade": "SIMULATION_ONLY_UNVALIDATED",
        "configuration": {
            "profile": f"TDL-{profile.upper()}",
            "spec_version": "19.2",
            "frames_per_snr": frames_per_snr,
            "batch_size": batch_size,
            "batch_seeds_shared_across_snr": batch_seeds,
            "matched_realization_checks": [matched_checks[key] for key in batch_seeds],
            "seed": seed,
            "carrier_frequency_hz": carrier_frequency_hz,
            "delay_spread_s": delay_spread_s,
            "speed_mps": speed_mps,
            "n_size_bwp_resource_blocks": n_size_bwp,
            "subcarrier_spacing_khz": subcarrier_spacing_khz,
            "snr_definition": "unit-energy resource-grid Es/N0; noise variance=10**(-SNR_dB/10)",
            "channel_path": "Sionna OFDMChannel over standardized TDL; perfect synchronization; no ICI/ISI",
            "decoder": "Sionna PUSCHReceiver with DMRS-based channel estimation and CRC status",
            "confidence_interval": "two-sided 95% Wilson score interval across codewords",
            "versions": versions(),
            "package_source_sha256": package_source_sha256(),
        },
        "points": points,
        "limitations": [
            "OFDMChannel does not model sample-domain ISI or ICI.",
            "Synchronization is perfect; this does not evaluate the timing detector.",
            "The sweep is simulation evidence for this configuration, not standards certification or external validation.",
            "Wilson intervals count codewords and do not account for independent-scene uncertainty.",
        ],
    }
