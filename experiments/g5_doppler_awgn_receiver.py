"""Paired CPU PUSCH diagnostic for path Doppler with waveform AWGN.

This extends the fixed two-path Doppler receiver probe with sample-domain
proper-complex AWGN shared across channel conditions. Results are conditional
on this synthetic channel, waveform SNR, NR configuration, and payloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from integrations.sionna_runtime import sionna_phy_version

SCHEMA = "umcd.g5_doppler_awgn_receiver.v1"
BATCH_SEEDS = tuple(range(29_061, 29_073))
BLOCKS_PER_BATCH = 16
NOISE_VARIANCE = 0.01
CONFIRMATORY_DOPPLER_HZ = 750.0
CHANNEL_CONDITIONS = (
    (0.0, "continuous"),
    (350.0, "continuous"),
    (750.0, "continuous"),
    (1500.0, "continuous"),
    (CONFIRMATORY_DOPPLER_HZ, "symbol_static"),
)
MINIMUM_TOTAL_DOPPLER_BLER_INCREASE = 0.20
MAXIMUM_REFERENCE_BLER = 0.05
MAXIMUM_PAIRED_P_VALUE = 0.01
MINIMUM_INTRA_SYMBOL_BLER_INCREASE = 0.10


def condition_key(doppler_hz: float, channel_model: str) -> str:
    return f"{int(doppler_hz)}hz_{channel_model}"


def exact_paired_binomial_p_value(reference_pass: list[bool], comparison_pass: list[bool]) -> float:
    """Return a two-sided exact McNemar/binomial p-value for paired blocks."""
    if not reference_pass or len(reference_pass) != len(comparison_pass):
        raise ValueError("paired outcomes must have the same nonzero length")
    reference_only_failures = sum(a and not b for a, b in zip(reference_pass, comparison_pass))
    comparison_only_failures = sum(not a and b for a, b in zip(reference_pass, comparison_pass))
    discordant = reference_only_failures + comparison_only_failures
    if discordant == 0:
        return 1.0
    tail = min(reference_only_failures, comparison_only_failures)
    one_tail = sum(math.comb(discordant, k) for k in range(tail + 1)) / (2**discordant)
    return min(1.0, 2.0 * one_tail)


def wilson_interval(failures: int, trials: int) -> tuple[float, float]:
    """Return the two-sided 95% Wilson score interval for a binomial rate."""
    if isinstance(failures, bool) or isinstance(trials, bool):
        raise TypeError("failures and trials must be integers")
    if not isinstance(failures, int) or not isinstance(trials, int):
        raise TypeError("failures and trials must be integers")
    if trials < 1 or not 0 <= failures <= trials:
        raise ValueError("require 0 <= failures <= trials and trials >= 1")
    z = 1.959963984540054
    p = failures / trials
    z2 = z * z
    denominator = 1 + z2 / trials
    center = (p + z2 / (2 * trials)) / denominator
    radius = z * math.sqrt(p * (1 - p) / trials + z2 / (4 * trials**2)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _installed_version(distribution: str) -> str:
    if distribution == "sionna":
        try:
            return sionna_phy_version()
        except PackageNotFoundError:
            return "not-installed"
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not-installed"


def _revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _source_fingerprints() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    sources = (
        Path(__file__).resolve(),
        root / "integrations/sionna/ofdm_impairments.py",
        root / "integrations/sionna/nr_pusch.py",
        root / "receivers/nr/pusch.py",
    )
    return {
        source.relative_to(root).as_posix(): hashlib.sha256(source.read_bytes()).hexdigest()
        for source in sources
    }


def run_doppler_awgn_probe(*, batch_seeds: tuple[int, ...] = BATCH_SEEDS) -> dict[str, Any]:
    """Run same-payload, same-waveform-noise CPU PUSCH Doppler conditions."""
    if not batch_seeds or len(set(batch_seeds)) != len(batch_seeds):
        raise ValueError("batch_seeds must be nonempty and unique")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in batch_seeds):
        raise ValueError("batch_seeds must contain nonnegative integers")

    import torch
    from integrations.sionna.nr_pusch import SionnaPUSCHLSReceiver
    from integrations.sionna.ofdm_impairments import apply_time_varying_multipath
    from receivers.nr import NRPUSCHConfig

    config = NRPUSCHConfig(n_size_bwp=6, subcarrier_spacing_khz=30)
    receiver = SionnaPUSCHLSReceiver(config, device="cpu")
    grid = receiver.resource_grid
    sample_rate_hz = float(grid.fft_size * config.subcarrier_spacing_khz * 1_000)
    symbol_length_samples = int(grid.fft_size + grid.cyclic_prefix_length)
    block_passes: dict[str, list[bool]] = {
        condition_key(*condition): [] for condition in CHANNEL_CONDITIONS
    }
    block_errors: dict[str, list[int]] = {
        condition_key(*condition): [] for condition in CHANNEL_CONDITIONS
    }
    records: list[dict[str, Any]] = []

    for seed in batch_seeds:
        tx_grid, tx_bits = receiver.transmit(BLOCKS_PER_BATCH, seed=seed)
        tx_waveform = receiver.modulate_slot_grid(tx_grid)
        noise_generator = torch.Generator(device="cpu").manual_seed(seed + 100_000)
        waveform_noise = torch.complex(
            torch.randn(tx_waveform.shape, generator=noise_generator),
            torch.randn(tx_waveform.shape, generator=noise_generator),
        ) / math.sqrt(2.0)
        waveform_noise = waveform_noise * math.sqrt(NOISE_VARIANCE)
        batch_record: dict[str, Any] = {
            "batch_seed": seed,
            "waveform_noise_seed": seed + 100_000,
            "conditions": {},
        }

        for doppler_hz, channel_model in CHANNEL_CONDITIONS:
            condition = condition_key(doppler_hz, channel_model)
            channel_waveform = apply_time_varying_multipath(
                tx_waveform,
                path_coefficients=(0.85 + 0.0j, 0.28 + 0.16j),
                path_delays_samples=(0, 3),
                path_dopplers_hz=(0.0, doppler_hz),
                sampling_frequency_hz=sample_rate_hz,
                symbol_length_samples=(
                    symbol_length_samples if channel_model == "symbol_static" else None
                ),
            )
            received_waveform = channel_waveform + waveform_noise
            received_grid = receiver.demodulate_slot_waveform(received_waveform)
            decoded_bits, crc_ok = receiver.decode_pusch(
                received_grid, torch.tensor(NOISE_VARIANCE)
            )
            passed = [bool(value) for value in crc_ok.reshape(-1).tolist()]
            errors = [
                int(value)
                for value in (decoded_bits != tx_bits)
                .reshape(BLOCKS_PER_BATCH, -1)
                .sum(dim=-1)
                .tolist()
            ]
            block_passes[condition].extend(passed)
            block_errors[condition].extend(errors)
            batch_record["conditions"][condition] = {
                "crc_pass_by_block": passed,
                "bit_errors_by_block": errors,
                "mean_received_signal_power": float(torch.mean(torch.abs(channel_waveform) ** 2)),
                "mean_realized_awgn_power": float(torch.mean(torch.abs(waveform_noise) ** 2)),
            }
        records.append(batch_record)

    summaries: dict[str, dict[str, Any]] = {}
    for condition, passed in block_passes.items():
        failures = len(passed) - sum(passed)
        lower, upper = wilson_interval(failures, len(passed))
        errors = block_errors[condition]
        summaries[condition] = {
            "blocks": len(passed),
            "crc_failures": failures,
            "bler": failures / len(passed),
            "bler_wilson_95_percent_interval": [lower, upper],
            "bit_errors": sum(errors),
            "total_bits": len(passed) * int(tx_bits.shape[-1]),
            "bit_error_rate": sum(errors) / (len(passed) * int(tx_bits.shape[-1])),
            "crc_pass_by_block": passed,
        }

    key_zero = condition_key(0.0, "continuous")
    key_mobile = condition_key(CONFIRMATORY_DOPPLER_HZ, "continuous")
    baseline_bler = summaries[key_zero]["bler"]
    mobile_bler = summaries[key_mobile]["bler"]
    mobile_increase = mobile_bler - baseline_bler
    mobile_p = exact_paired_binomial_p_value(block_passes[key_zero], block_passes[key_mobile])
    acceptance = {
        "protocol": "same seeded payloads and sample-domain proper-complex AWGN; fixed two-path integer-delay channel; CPU PUSCH transmit, OFDM, and DMRS-estimating receiver",
        "primary_conditions": [key_zero, key_mobile],
        "waveform_awgn_complex_variance": NOISE_VARIANCE,
        "minimum_absolute_bler_increase": MINIMUM_TOTAL_DOPPLER_BLER_INCREASE,
        "maximum_static_reference_bler": MAXIMUM_REFERENCE_BLER,
        "maximum_two_sided_exact_p_value": MAXIMUM_PAIRED_P_VALUE,
        "observed_absolute_bler_increase": mobile_increase,
        "observed_paired_exact_p_value": mobile_p,
        "pass": baseline_bler <= MAXIMUM_REFERENCE_BLER
        and mobile_increase >= MINIMUM_TOTAL_DOPPLER_BLER_INCREASE
        and mobile_p <= MAXIMUM_PAIRED_P_VALUE,
        "inference_scope": "conditional on the fixed synthetic channel, declared waveform noise, NR configuration, and sampled payloads; not a field BLER claim",
    }
    static_key = condition_key(CONFIRMATORY_DOPPLER_HZ, "symbol_static")
    static_bler = summaries[static_key]["bler"]
    intra_symbol_increase = mobile_bler - static_bler
    intra_symbol_p = exact_paired_binomial_p_value(
        block_passes[static_key], block_passes[key_mobile]
    )
    intra_symbol_acceptance = {
        "interpretation": "continuous sample-time Doppler versus equal Doppler frozen at each OFDM-symbol start; both include inter-symbol aging and identical waveform AWGN",
        "minimum_absolute_bler_increase": MINIMUM_INTRA_SYMBOL_BLER_INCREASE,
        "maximum_two_sided_exact_p_value": MAXIMUM_PAIRED_P_VALUE,
        "observed_absolute_bler_increase": intra_symbol_increase,
        "observed_paired_exact_p_value": intra_symbol_p,
        "pass": intra_symbol_increase >= MINIMUM_INTRA_SYMBOL_BLER_INCREASE
        and intra_symbol_p <= MAXIMUM_PAIRED_P_VALUE,
        "inference_scope": "conditional on this fixed synthetic channel and sampled payloads; the symbol-static model is a diagnostic control, not a deployment channel",
    }
    configuration = {
        "n_size_bwp_rb": config.n_size_bwp,
        "subcarrier_spacing_hz": config.subcarrier_spacing_khz * 1_000,
        "fft_size": int(grid.fft_size),
        "cyclic_prefix_samples": int(grid.cyclic_prefix_length),
        "sample_rate_hz": sample_rate_hz,
        "path_coefficients_real_imag": [[0.85, 0.0], [0.28, 0.16]],
        "path_delays_samples": [0, 3],
        "echo_dopplers_hz": [
            doppler for doppler, model in CHANNEL_CONDITIONS if model == "continuous"
        ],
        "symbol_static_control": "echo phase evaluated at OFDM-symbol start and held through symbol; inter-symbol aging retained",
        "symbol_length_samples": symbol_length_samples,
        "waveform_awgn_complex_variance": NOISE_VARIANCE,
        "awgn_model": "independent proper-complex Gaussian sample noise; same realization reused across all paired channel conditions",
        "decoder_assumed_grid_noise_variance": NOISE_VARIANCE,
        "blocks_per_seed": BLOCKS_PER_BATCH,
        "bits_per_block": int(tx_bits.shape[-1]),
        "batch_seeds": list(batch_seeds),
        "waveform_noise_seeds": [seed + 100_000 for seed in batch_seeds],
        "independent_channel_draws": 0,
    }
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "command": list(sys.orig_argv),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": _revision(),
        "source_sha256": _source_fingerprints(),
        "config_sha256": hashlib.sha256(
            json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": _installed_version("torch"),
            "sionna": _installed_version("sionna"),
            "device": "cpu",
        },
        "configuration": configuration,
        "acceptance": acceptance,
        "intra_symbol_acceptance": intra_symbol_acceptance,
        "summary_by_condition": summaries,
        "paired_batch_outcomes": records,
    }
    result["result_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/research/results/g5-time-varying-multipath-pusch-awgn-20261023.json"),
    )
    args = parser.parse_args()
    report = run_doppler_awgn_probe()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"wrote {args.output}: 0-Hz BLER="
        f"{report['summary_by_condition']['0hz_continuous']['bler']:.3f}, "
        f"750-Hz BLER={report['summary_by_condition']['750hz_continuous']['bler']:.3f}, "
        f"750-Hz symbol-static BLER="
        f"{report['summary_by_condition']['750hz_symbol_static']['bler']:.3f}, "
        f"combined={'PASS' if report['acceptance']['pass'] else 'NOT MET'}, "
        f"intra-symbol={'PASS' if report['intra_symbol_acceptance']['pass'] else 'NOT MET'}"
    )


if __name__ == "__main__":
    main()
