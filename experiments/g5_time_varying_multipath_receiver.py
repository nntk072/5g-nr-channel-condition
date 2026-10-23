"""Paired CPU PUSCH diagnostic for intra-symbol per-path Doppler.

This fixed two-path fixture measures receiver sensitivity to sample-domain
channel variation. It is not a propagation model, deployment BLER curve, or
qualification of the RT channel generator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

SCHEMA = "umcd.g5_time_varying_multipath_receiver.v2"
DOPPLERS_HZ = (0.0, 350.0, 750.0, 1500.0)
CONFIRMATORY_DOPPLER_HZ = 750.0
CHANNEL_CONDITIONS = tuple((doppler, "continuous") for doppler in DOPPLERS_HZ) + (
    (CONFIRMATORY_DOPPLER_HZ, "symbol_static"),
)
BATCH_SEEDS = tuple(range(29_061, 29_073))
BLOCKS_PER_BATCH = 16
DETECTOR_NOISE_VARIANCE = 1e-5
MINIMUM_BLER_INCREASE = 0.20
MAXIMUM_BASELINE_BLER = 0.05
MAXIMUM_PAIRED_P_VALUE = 0.01
MINIMUM_INTRA_SYMBOL_BLER_INCREASE = 0.10
MAXIMUM_INTRA_SYMBOL_PAIRED_P_VALUE = 0.01


def _condition_key(doppler_hz: float, channel_model: str) -> str:
    return f"{int(doppler_hz)}hz_{channel_model}"


def wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    """Return the two-sided 95% Wilson interval for a binomial proportion."""
    if isinstance(successes, bool) or isinstance(trials, bool):
        raise TypeError("successes and trials must be integers")
    if not isinstance(successes, int) or not isinstance(trials, int):
        raise TypeError("successes and trials must be integers")
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("require 0 <= successes <= trials and trials >= 1")
    z = 1.959963984540054
    p = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    center = (p + z2 / (2.0 * trials)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / trials + z2 / (4.0 * trials**2)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def exact_paired_binomial_p_value(reference_pass: list[bool], comparison_pass: list[bool]) -> float:
    """Two-sided exact McNemar/binomial p-value for paired binary outcomes."""
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


def _installed_version(distribution: str) -> str:
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
    paths = (
        Path(__file__).resolve(),
        root / "integrations/sionna/ofdm_impairments.py",
        root / "integrations/sionna/nr_pusch.py",
        root / "receivers/nr/pusch.py",
    )
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def run_receiver_impact_probe(*, batch_seeds: tuple[int, ...] = BATCH_SEEDS) -> dict[str, Any]:
    """Run paired static-versus-mobile PUSCH views at frozen fixture settings."""
    if not batch_seeds or len(set(batch_seeds)) != len(batch_seeds):
        raise ValueError("batch_seeds must be nonempty and unique")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in batch_seeds):
        raise ValueError("batch_seeds must contain nonnegative integers")

    import torch
    from integrations.sionna.nr_pusch import SionnaPUSCHLSReceiver
    from integrations.sionna.ofdm_impairments import apply_time_varying_multipath
    from receivers.nr import NRPUSCHConfig

    config = NRPUSCHConfig(n_size_bwp=6, subcarrier_spacing_khz=30)
    transmitter = SionnaPUSCHLSReceiver(config, device="cpu")
    grid = transmitter.resource_grid
    sample_rate_hz = float(grid.fft_size * config.subcarrier_spacing_khz * 1_000)
    outcomes: dict[str, list[bool]] = {
        _condition_key(doppler, model): [] for doppler, model in CHANNEL_CONDITIONS
    }
    records: list[dict[str, Any]] = []

    for seed in batch_seeds:
        tx_grid, tx_bits = transmitter.transmit(BLOCKS_PER_BATCH, seed=seed)
        tx_waveform = transmitter.modulate_slot_grid(tx_grid)
        for doppler_hz, channel_model in CHANNEL_CONDITIONS:
            rx_waveform = apply_time_varying_multipath(
                tx_waveform,
                path_coefficients=(0.85 + 0.0j, 0.28 + 0.16j),
                path_delays_samples=(0, 3),
                path_dopplers_hz=(0.0, doppler_hz),
                sampling_frequency_hz=sample_rate_hz,
                symbol_length_samples=(
                    int(grid.fft_size + grid.cyclic_prefix_length)
                    if channel_model == "symbol_static"
                    else None
                ),
            )
            rx_grid = transmitter.demodulate_slot_waveform(rx_waveform)
            decoded_bits, crc_ok = transmitter.decode_pusch(
                rx_grid, torch.tensor(DETECTOR_NOISE_VARIANCE)
            )
            bit_errors = (decoded_bits != tx_bits).reshape(BLOCKS_PER_BATCH, -1).sum(dim=-1)
            block_pass = [bool(value) for value in crc_ok.reshape(-1).tolist()]
            outcomes[_condition_key(doppler_hz, channel_model)].extend(block_pass)
            records.append(
                {
                    "batch_seed": seed,
                    "echo_doppler_hz": doppler_hz,
                    "channel_model": channel_model,
                    "crc_pass_by_block": block_pass,
                    "bit_errors_by_block": [int(value) for value in bit_errors.tolist()],
                }
            )

    summary: dict[str, Any] = {}
    for condition, block_pass in outcomes.items():
        failures = len(block_pass) - sum(block_pass)
        errors = [
            error
            for record in records
            if _condition_key(record["echo_doppler_hz"], record["channel_model"]) == condition
            for error in record["bit_errors_by_block"]
        ]
        low, high = wilson_interval(failures, len(block_pass))
        doppler_hz, channel_model = next(
            condition_spec
            for condition_spec in CHANNEL_CONDITIONS
            if _condition_key(*condition_spec) == condition
        )
        summary[condition] = {
            "echo_doppler_hz": doppler_hz,
            "channel_model": channel_model,
            "blocks": len(block_pass),
            "crc_failures": failures,
            "bler": failures / len(block_pass),
            "bler_wilson_95_percent_interval": [low, high],
            "bit_error_rate": sum(errors) / (len(errors) * int(tx_bits.shape[-1])),
            "crc_pass_by_block": block_pass,
        }

    reference = outcomes[_condition_key(0.0, "continuous")]
    comparison = outcomes[_condition_key(CONFIRMATORY_DOPPLER_HZ, "continuous")]
    paired_p = exact_paired_binomial_p_value(reference, comparison)
    baseline_bler = summary[_condition_key(0.0, "continuous")]["bler"]
    confirmatory_bler = summary[_condition_key(CONFIRMATORY_DOPPLER_HZ, "continuous")]["bler"]
    effect = confirmatory_bler - baseline_bler
    acceptance = {
        "protocol": "same payloads; fixed two-path integer-delay channel; CPU PUSCH TX/DMRS-estimating RX; continuous sample-domain echo Doppler",
        "primary_comparison_hz": [0.0, CONFIRMATORY_DOPPLER_HZ],
        "minimum_absolute_bler_increase": MINIMUM_BLER_INCREASE,
        "maximum_reference_bler": MAXIMUM_BASELINE_BLER,
        "maximum_two_sided_exact_p_value": MAXIMUM_PAIRED_P_VALUE,
        "observed_absolute_bler_increase": effect,
        "observed_paired_exact_p_value": paired_p,
        "pass": baseline_bler <= MAXIMUM_BASELINE_BLER
        and effect >= MINIMUM_BLER_INCREASE
        and paired_p <= MAXIMUM_PAIRED_P_VALUE,
        "inference_scope": "conditional on this fixed synthetic channel, NR configuration, SNR, and tested payload population; not a field BLER claim",
    }
    symbol_static = outcomes[_condition_key(CONFIRMATORY_DOPPLER_HZ, "symbol_static")]
    symbol_static_bler = summary[_condition_key(CONFIRMATORY_DOPPLER_HZ, "symbol_static")]["bler"]
    intra_symbol_p = exact_paired_binomial_p_value(symbol_static, comparison)
    intra_symbol_increase = confirmatory_bler - symbol_static_bler
    intra_symbol_acceptance = {
        "interpretation": "continuous sample-time Doppler versus the same Doppler frozen at each OFDM-symbol start; both retain inter-symbol channel aging",
        "minimum_absolute_bler_increase": MINIMUM_INTRA_SYMBOL_BLER_INCREASE,
        "maximum_two_sided_exact_p_value": MAXIMUM_INTRA_SYMBOL_PAIRED_P_VALUE,
        "observed_absolute_bler_increase": intra_symbol_increase,
        "observed_paired_exact_p_value": intra_symbol_p,
        "pass": intra_symbol_increase >= MINIMUM_INTRA_SYMBOL_BLER_INCREASE
        and intra_symbol_p <= MAXIMUM_INTRA_SYMBOL_PAIRED_P_VALUE,
        "inference_scope": "conditional on this fixed synthetic channel and tested payload population; the symbol-static model is a diagnostic control, not a deployment channel",
    }
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": _revision(),
        "source_sha256": _source_fingerprints(),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": _installed_version("torch"),
            "sionna": _installed_version("sionna"),
            "device": "cpu",
        },
        "configuration": {
            "n_size_bwp_rb": 6,
            "subcarrier_spacing_hz": 30_000,
            "fft_size": int(grid.fft_size),
            "cyclic_prefix_samples": int(grid.cyclic_prefix_length),
            "sample_rate_hz": sample_rate_hz,
            "path_coefficients_real_imag": [[0.85, 0.0], [0.28, 0.16]],
            "path_delays_samples": [0, 3],
            "direct_path_doppler_hz": 0.0,
            "echo_dopplers_hz": list(DOPPLERS_HZ),
            "waveform_awgn_injected": False,
            "symbol_static_control": "echo coefficient phase evaluated at each OFDM-symbol start and held over that symbol",
            "symbol_length_samples": int(grid.fft_size + grid.cyclic_prefix_length),
            "decoder_assumed_noise_variance": DETECTOR_NOISE_VARIANCE,
            "blocks_per_seed": BLOCKS_PER_BATCH,
            "bits_per_block": int(tx_bits.shape[-1]),
            "batch_seeds": list(batch_seeds),
            "independent_channel_draws": 0,
        },
        "acceptance": acceptance,
        "intra_symbol_acceptance": intra_symbol_acceptance,
        "summary_by_condition": summary,
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
        default=Path(
            "docs/research/results/g5-time-varying-multipath-pusch-symbol-control-20261023.json"
        ),
    )
    args = parser.parse_args()
    report = run_receiver_impact_probe()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"wrote {args.output}: 0-Hz BLER="
        f"{report['summary_by_condition']['0hz_continuous']['bler']:.3f}, "
        f"750-Hz continuous BLER="
        f"{report['summary_by_condition']['750hz_continuous']['bler']:.3f}, "
        f"750-Hz symbol-static BLER="
        f"{report['summary_by_condition']['750hz_symbol_static']['bler']:.3f}, "
        f"combined={'PASS' if report['acceptance']['pass'] else 'FAIL'}, "
        f"intra-symbol={'PASS' if report['intra_symbol_acceptance']['pass'] else 'FAIL'}"
    )


if __name__ == "__main__":
    main()
