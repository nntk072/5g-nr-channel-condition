"""Reproducible Sionna TDL-PUSCH link-level BLER curves."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Sequence

from artifacts.provenance import package_source_sha256, versions
from evaluation.statistics import paired_binary_error_comparison


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


def run_tdl_pusch_time_bler_curve(
    *,
    snr_db: Sequence[float],
    frames_per_snr: int,
    batch_size: int,
    seed: int,
    profile: str = "A",
    delay_spread_s: float = 100e-9,
    carrier_frequency_hz: float = 3.5e9,
    speed_mps: float = 3.0,
    n_size_bwp: int = 12,
    dmrs_additional_position: int = 0,
    subcarrier_spacing_khz: int = 30,
    num_rx_ant: int = 1,
    allow_cp_overrun: bool = False,
) -> dict[str, Any]:
    """Measure paired PUSCH BLER over the sample-domain, time-varying TDL link."""
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
    if isinstance(dmrs_additional_position, bool) or not isinstance(dmrs_additional_position, int):
        raise TypeError("dmrs_additional_position must be an integer")
    if not 0 <= dmrs_additional_position <= 3:
        raise ValueError("dmrs_additional_position must be in [0, 3]")
    if isinstance(num_rx_ant, bool) or not isinstance(num_rx_ant, int):
        raise TypeError("num_rx_ant must be an integer")
    if num_rx_ant <= 0:
        raise ValueError("num_rx_ant must be positive")
    if not torch_cuda_available():
        raise RuntimeError("The sample-domain TDL-PUSCH curve requires CUDA")

    import torch
    from integrations.sionna.nr_pusch import SionnaPUSCHLSReceiver
    from integrations.sionna.tdl_pusch import SionnaTDLTimePUSCHLink, seeded_complex_awgn
    from receivers.nr import NRPUSCHConfig

    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(
            n_size_bwp=n_size_bwp,
            subcarrier_spacing_khz=subcarrier_spacing_khz,
            dmrs_additional_position=dmrs_additional_position,
        ),
        device="cuda:0",
    )
    link = SionnaTDLTimePUSCHLink(
        receiver,
        model=profile.upper(),
        delay_spread_s=delay_spread_s,
        carrier_frequency_hz=carrier_frequency_hz,
        speed_mps=speed_mps,
        num_rx_ant=num_rx_ant,
        allow_cp_overrun=allow_cp_overrun,
    )
    if seed > 2**63 - 3:
        raise ValueError("seed must leave room for derived channel and noise seeds")

    batches: list[dict[str, Any]] = []
    matched_checks: list[dict[str, Any]] = []
    frames_left = frames_per_snr
    batch_index = 0
    while frames_left:
        count = min(batch_size, frames_left)
        batch_seed = seed + batch_index
        waveform, bits, clean, h_time = link.transmit_channel(count, seed=batch_seed)
        bits_cpu = bits.detach().cpu().contiguous()
        h_cpu = h_time.detach().cpu().contiguous()
        batches.append(
            {
                "seed": batch_seed,
                "count": count,
                "waveform": waveform,
                "waveform_mean_power": float(waveform.abs().square().mean().item()),
                "bits": bits,
                "clean": clean,
                "h_time": h_time,
                "noise_seed": batch_seed + 2,
            }
        )
        matched_checks.append(
            {
                "seed": batch_seed,
                "noise_seed": batch_seed + 2,
                "transmit_bits_sha256": hashlib.sha256(bits_cpu.numpy().tobytes()).hexdigest(),
                "channel_truth_sha256": hashlib.sha256(h_cpu.numpy().tobytes()).hexdigest(),
                "transmit_bits_match_across_snr": True,
                "channel_truth_match_across_snr": True,
                "noise_real_seed": batch_seed + 2,
                "noise_imag_seed": batch_seed + 3,
            }
        )
        frames_left -= count
        batch_index += 1

    points: list[dict[str, Any]] = []
    for snr in snr_values:
        noise_variance = 10.0 ** (-snr / 10.0)
        frame_errors = bit_errors = frames = bits_total = 0
        frame_error_mask: list[bool] = []
        for batch in batches:
            clean = batch["clean"]
            noise = seeded_complex_awgn(
                tuple(clean.shape),
                device="cuda:0",
                noise_variance=noise_variance,
                real_seed=batch["noise_seed"],
                imag_seed=batch["noise_seed"] + 1,
            )
            received = clean + noise
            decoded, crc_ok = link.decode(received, noise_variance)
            target = batch["bits"]
            valid = crc_ok.reshape(-1).to(dtype=torch.bool)
            frame_error_mask.extend((~valid).detach().cpu().tolist())
            bit_errors += int(torch.count_nonzero(decoded != target).item())
            bits_total += int(target.numel())
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
                "frame_error_mask": frame_error_mask,
                "bit_errors": bit_errors,
                "bits": bits_total,
                "ber": bit_errors / bits_total,
                "mean_transmitted_tb_bits_per_codeword": bits_total / frames,
                "successfully_decoded_tb_bits_per_codeword": (
                    (bits_total / frames) * (1.0 - frame_errors / frames)
                ),
            }
        )

    return {
        "schema": "loslab.tdl_pusch_time_bler_curve.v1",
        "evidence_grade": "SIMULATION_ONLY_UNVALIDATED",
        "configuration": {
            **link.metadata,
            "frames_per_snr": frames_per_snr,
            "batch_size": batch_size,
            "batch_seeds_shared_across_snr": [batch["seed"] for batch in batches],
            "transmit_waveform_mean_power_by_batch": [
                batch["waveform_mean_power"] for batch in batches
            ],
            "matched_realization_checks": matched_checks,
            "seed": seed,
            "n_size_bwp_resource_blocks": n_size_bwp,
            "subcarrier_spacing_khz": subcarrier_spacing_khz,
            "dmrs_additional_position": dmrs_additional_position,
            "snr_definition": (
                "sampled waveform mean complex power / complex AWGN variance; "
                "noise variance=10**(-SNR_dB/10) and measured transmit mean power is recorded"
            ),
            "decoder": "Sionna time-domain PUSCHReceiver with DMRS estimation and CRC status",
            "confidence_interval": "two-sided 95% Wilson score interval across codewords",
            "versions": versions(),
            "package_source_sha256": package_source_sha256(),
        },
        "points": points,
        "limitations": [
            "Sionna sampled TDL and its finite sinc-interpolation tap support are simulation models.",
            "The receiver has no injected timing or carrier-frequency synchronization error.",
            (
                "The cyclic prefix covers the configured discrete channel memory."
                if link.metadata["cp_covers_channel_memory"]
                else "The discrete channel memory exceeds the cyclic prefix; this is an explicit "
                "ISI/ICI stress condition and not a normal CP-contained PUSCH configuration."
            ),
            "Wilson intervals count codewords and do not account for independent-scene uncertainty.",
            "This curve is not standards certification or validation against external measurements.",
        ],
    }


def run_tdl_pusch_rx_diversity_curve(
    *,
    snr_db: Sequence[float],
    frames_per_snr: int,
    batch_size: int,
    seed: int,
    profile: str = "A",
    delay_spread_s: float = 100e-9,
    carrier_frequency_hz: float = 3.5e9,
    speed_mps: float = 3.0,
    n_size_bwp: int = 12,
    dmrs_additional_position: int = 0,
    subcarrier_spacing_khz: int = 30,
) -> dict[str, Any]:
    """Compare one versus two RX antennas on the exact same two-RX channel draw."""
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
    if not torch_cuda_available():
        raise RuntimeError("The receive-antenna diversity curve requires CUDA")

    import torch
    from integrations.sionna.nr_pusch import SionnaPUSCHLSReceiver
    from integrations.sionna.tdl_pusch import SionnaTDLTimePUSCHLink, seeded_complex_awgn
    from receivers.nr import NRPUSCHConfig

    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(
            n_size_bwp=n_size_bwp,
            subcarrier_spacing_khz=subcarrier_spacing_khz,
            dmrs_additional_position=dmrs_additional_position,
        ),
        device="cuda:0",
    )
    link = SionnaTDLTimePUSCHLink(
        receiver,
        model=profile.upper(),
        delay_spread_s=delay_spread_s,
        carrier_frequency_hz=carrier_frequency_hz,
        speed_mps=speed_mps,
        num_rx_ant=2,
    )
    if seed > 2**63 - 3:
        raise ValueError("seed must leave room for derived channel and noise seeds")

    batches: list[dict[str, Any]] = []
    matched_checks: list[dict[str, Any]] = []
    frames_left = frames_per_snr
    batch_index = 0
    while frames_left:
        count = min(batch_size, frames_left)
        batch_seed = seed + batch_index
        waveform, bits, clean, channel_truth = link.transmit_channel(count, seed=batch_seed)
        bits_cpu = bits.detach().cpu().contiguous()
        channel_truth_cpu = channel_truth.detach().cpu().contiguous()
        first_rx_truth = channel_truth_cpu[:, :, :1].contiguous()
        batches.append(
            {
                "seed": batch_seed,
                "bits": bits,
                "clean": clean,
                "noise_seed": batch_seed + 2,
                "waveform_mean_power": float(waveform.abs().square().mean().item()),
            }
        )
        matched_checks.append(
            {
                "seed": batch_seed,
                "transmit_bits_sha256": hashlib.sha256(bits_cpu.numpy().tobytes()).hexdigest(),
                "first_rx_channel_truth_sha256": hashlib.sha256(
                    first_rx_truth.numpy().tobytes()
                ).hexdigest(),
                "all_rx_channel_truth_sha256": hashlib.sha256(
                    channel_truth_cpu.numpy().tobytes()
                ).hexdigest(),
                "noise_real_seed": batch_seed + 2,
                "noise_imag_seed": batch_seed + 3,
                "single_rx_is_first_antenna_of_two_rx_draw": True,
            }
        )
        frames_left -= count
        batch_index += 1

    points: list[dict[str, Any]] = []
    for snr in snr_values:
        noise_variance = 10.0 ** (-snr / 10.0)
        one_error_mask: list[bool] = []
        two_error_mask: list[bool] = []
        one_bit_errors = two_bit_errors = bits_total = 0
        for batch in batches:
            clean = batch["clean"]
            noise = seeded_complex_awgn(
                tuple(clean.shape),
                device="cuda:0",
                noise_variance=noise_variance,
                real_seed=batch["noise_seed"],
                imag_seed=batch["noise_seed"] + 1,
            )
            received_two = clean + noise
            received_one = received_two[:, :, :1, :]
            decoded_one, crc_one = link.decode(received_one, noise_variance)
            decoded_two, crc_two = link.decode(received_two, noise_variance)
            bits = batch["bits"]
            one_error_mask.extend((~crc_one.reshape(-1).bool()).cpu().tolist())
            two_error_mask.extend((~crc_two.reshape(-1).bool()).cpu().tolist())
            one_bit_errors += int(torch.count_nonzero(decoded_one != bits).item())
            two_bit_errors += int(torch.count_nonzero(decoded_two != bits).item())
            bits_total += int(bits.numel())

        single_errors = sum(one_error_mask)
        diversity_errors = sum(two_error_mask)
        single_ci = wilson_score_interval_95(single_errors, frames_per_snr)
        diversity_ci = wilson_score_interval_95(diversity_errors, frames_per_snr)
        paired = paired_binary_error_comparison(
            one_error_mask,
            two_error_mask,
            seed=1,
            draws=20_000,
            control_bits_per_success=(bits_total / frames_per_snr),
            treatment_bits_per_success=(bits_total / frames_per_snr),
        )
        points.append(
            {
                "snr_db": snr,
                "frames": frames_per_snr,
                "bits": bits_total,
                "single_rx": {
                    "bler_errors": single_errors,
                    "bler": single_errors / frames_per_snr,
                    "bler_wilson_95_ci": list(single_ci),
                    "bit_errors": one_bit_errors,
                    "ber": one_bit_errors / bits_total,
                    "frame_error_mask": one_error_mask,
                },
                "two_rx": {
                    "bler_errors": diversity_errors,
                    "bler": diversity_errors / frames_per_snr,
                    "bler_wilson_95_ci": list(diversity_ci),
                    "bit_errors": two_bit_errors,
                    "ber": two_bit_errors / bits_total,
                    "frame_error_mask": two_error_mask,
                },
                "paired_comparison": paired,
            }
        )

    ordered = sorted(
        range(len(points)),
        key=lambda index: points[index]["paired_comparison"]["mcnemar_exact_two_sided_p_value"],
    )
    adjusted_max = 0.0
    for rank, index in enumerate(ordered):
        raw_p = points[index]["paired_comparison"]["mcnemar_exact_two_sided_p_value"]
        adjusted_max = max(adjusted_max, (len(points) - rank) * raw_p)
        points[index]["paired_comparison"]["mcnemar_exact_two_sided_p_value_holm"] = min(
            1.0, adjusted_max
        )

    return {
        "schema": "loslab.tdl_pusch_rx_diversity.v1",
        "evidence_grade": "SIMULATION_ONLY_UNVALIDATED",
        "configuration": {
            **link.metadata,
            "frames_per_snr": frames_per_snr,
            "batch_size": batch_size,
            "seed": seed,
            "batch_seeds_shared_across_snr": [item["seed"] for item in matched_checks],
            "matched_realization_checks": matched_checks,
            "transmit_waveform_mean_power_by_batch": [
                item["waveform_mean_power"] for item in batches
            ],
            "snr_db": list(snr_values),
            "snr_definition": "unit-average-power sampled waveform with complex AWGN variance 10**(-SNR_dB/10)",
            "single_rx_policy": "decode receive antenna index 0 from each generated two-antenna waveform",
            "two_rx_policy": "decode receive antenna indices 0 and 1 from the same waveform and channel draw",
            "versions": versions(),
            "package_source_sha256": package_source_sha256(),
        },
        "points": points,
        "limitations": [
            "Only Sionna TDL-A, one numerology, one bandwidth, one DMRS pattern, speed, and delay spread are evaluated.",
            "The two receive antennas follow Sionna's default spatial-correlation configuration; no measured array calibration is used.",
            "Synchronization is ideal and the 100-ns channel memory is CP-contained for this link.",
            "Inference is conditional on the simulated codewords and does not represent independent scenes or measured channels.",
        ],
    }


def torch_cuda_available() -> bool:
    """Check CUDA availability without making PyTorch a module-level dependency."""
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()
