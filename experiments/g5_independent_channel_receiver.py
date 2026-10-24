"""Paired PUSCH receiver sensitivity study across independent synthetic channels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

SCHEMA = "umcd.g5_independent_channel_receiver.v3"
CHANNEL_SEEDS = tuple(range(73_001, 73_065))
BLOCKS_PER_CHANNEL = 8
NOISE_VARIANCE = 0.01
MOBILE_DOPPLER_HZ = 750.0
DOPPLER_STRESS_HZ = 1500.0
BOOTSTRAP_SEED = 73_999
BOOTSTRAP_REPLICATES = 20_000
FAMILYWISE_ALPHA = 0.01
MAXIMUM_REFERENCE_BLER = 0.05
MAXIMUM_REFERENCE_UPPER_BOUND = 0.10
MINIMUM_MOBILE_BLER_INCREASE = 0.10
MINIMUM_INTRA_SYMBOL_BLER_INCREASE = 0.10


def cluster_bootstrap_interval(
    paired_values: list[float],
    *,
    seed: int,
    alpha: float,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> tuple[float, float]:
    """Return a percentile interval by resampling independent channel clusters."""
    if not paired_values:
        raise ValueError("paired_values must be nonempty")
    if any(not math.isfinite(value) for value in paired_values):
        raise ValueError("paired_values must be finite")
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and strictly between zero and one")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 100:
        raise ValueError("replicates must be an integer of at least 100")

    rng = random.Random(seed)
    count = len(paired_values)
    estimates = sorted(
        sum(paired_values[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(replicates)
    )
    tail = alpha / 2.0
    lower = estimates[math.floor(tail * replicates)]
    upper = estimates[min(replicates - 1, math.ceil((1.0 - tail) * replicates) - 1)]
    return lower, upper


def clopper_pearson_upper_bound(successes: int, trials: int, *, alpha: float) -> float:
    """Return an exact one-sided binomial upper confidence bound."""
    if isinstance(successes, bool) or isinstance(trials, bool):
        raise TypeError("successes and trials must be integers")
    if not isinstance(successes, int) or not isinstance(trials, int):
        raise TypeError("successes and trials must be integers")
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("require 0 <= successes <= trials and trials >= 1")
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and strictly between zero and one")
    if successes == trials:
        return 1.0

    lower, upper = 0.0, 1.0
    for _ in range(80):
        probability = (lower + upper) / 2.0
        cdf = sum(
            math.comb(trials, count) * probability**count * (1.0 - probability) ** (trials - count)
            for count in range(successes + 1)
        )
        if cdf > alpha:
            lower = probability
        else:
            upper = probability
    return (lower + upper) / 2.0


def _revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _installed_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not-installed"


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


def run_independent_channel_probe(
    *, channel_seeds: tuple[int, ...] = CHANNEL_SEEDS, blocks_per_channel: int = BLOCKS_PER_CHANNEL
) -> dict[str, Any]:
    """Measure paired static/mobile PUSCH outcomes over seeded channel draws."""
    if not channel_seeds or len(set(channel_seeds)) != len(channel_seeds):
        raise ValueError("channel_seeds must be nonempty and unique")
    if any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in channel_seeds
    ):
        raise ValueError("channel_seeds must contain nonnegative integers")
    if isinstance(blocks_per_channel, bool) or not isinstance(blocks_per_channel, int):
        raise TypeError("blocks_per_channel must be an integer")
    if blocks_per_channel < 1:
        raise ValueError("blocks_per_channel must be positive")

    import torch
    from integrations.sionna.nr_pusch import SionnaPUSCHLSReceiver
    from integrations.sionna.ofdm_impairments import apply_time_varying_multipath
    from receivers.nr import NRPUSCHConfig

    config = NRPUSCHConfig(n_size_bwp=6, subcarrier_spacing_khz=30)
    receiver = SionnaPUSCHLSReceiver(config, device="cpu")
    grid = receiver.resource_grid
    sample_rate_hz = float(grid.fft_size * config.subcarrier_spacing_khz * 1_000)
    conditions = (
        "static",
        "750hz_symbol_static",
        "750hz_continuous",
        "1500hz_symbol_static",
        "1500hz_continuous",
    )
    records: list[dict[str, Any]] = []
    cluster_rates: dict[str, list[float]] = {condition: [] for condition in conditions}

    for channel_seed in channel_seeds:
        channel_rng = random.Random(channel_seed)
        delay_samples = channel_rng.randint(2, 8)
        echo_amplitude = channel_rng.uniform(0.20, 0.50)
        echo_phase = channel_rng.uniform(-math.pi, math.pi)
        coefficients = (
            0.85 + 0.0j,
            echo_amplitude * complex(math.cos(echo_phase), math.sin(echo_phase)),
        )

        tx_grid, tx_bits = receiver.transmit(blocks_per_channel, seed=channel_seed + 1_000_000)
        tx_waveform = receiver.modulate_slot_grid(tx_grid)
        noise_rng = torch.Generator(device="cpu").manual_seed(channel_seed + 2_000_000)
        waveform_noise = torch.complex(
            torch.randn(tx_waveform.shape, generator=noise_rng),
            torch.randn(tx_waveform.shape, generator=noise_rng),
        ) * math.sqrt(NOISE_VARIANCE / 2.0)
        by_condition: dict[str, list[bool]] = {}

        for condition in conditions:
            doppler = 0.0 if condition == "static" else float(condition.split("hz_", maxsplit=1)[0])
            symbol_static = condition.endswith("_symbol_static")
            received = (
                apply_time_varying_multipath(
                    tx_waveform,
                    path_coefficients=coefficients,
                    path_delays_samples=(0, delay_samples),
                    path_dopplers_hz=(0.0, doppler),
                    sampling_frequency_hz=sample_rate_hz,
                    symbol_length_samples=(
                        int(grid.fft_size + grid.cyclic_prefix_length) if symbol_static else None
                    ),
                )
                + waveform_noise
            )
            received_grid = receiver.demodulate_slot_waveform(received)
            _, crc_ok = receiver.decode_pusch(received_grid, torch.tensor(NOISE_VARIANCE))
            passed = [bool(value) for value in crc_ok.reshape(-1).tolist()]
            by_condition[condition] = passed
            cluster_rates[condition].append(1.0 - sum(passed) / len(passed))

        records.append(
            {
                "channel_seed": channel_seed,
                "payload_seed": channel_seed + 1_000_000,
                "waveform_noise_seed": channel_seed + 2_000_000,
                "echo_delay_samples": delay_samples,
                "echo_coefficient_real_imag": [coefficients[1].real, coefficients[1].imag],
                "crc_pass_by_condition": by_condition,
            }
        )

    reference_rates = cluster_rates["static"]
    per_interval_alpha = FAMILYWISE_ALPHA / 5.0
    reference_channels_with_failures = sum(rate > 0.0 for rate in reference_rates)
    reference_upper_bound = clopper_pearson_upper_bound(
        reference_channels_with_failures,
        len(reference_rates),
        alpha=per_interval_alpha,
    )
    reference_mean = sum(reference_rates) / len(reference_rates)
    mobile_effects: dict[str, dict[str, Any]] = {}
    intra_symbol_effects: dict[str, dict[str, Any]] = {}
    for index, doppler_hz in enumerate((MOBILE_DOPPLER_HZ, DOPPLER_STRESS_HZ)):
        doppler_key = str(int(doppler_hz))
        mobile_rates = cluster_rates[f"{doppler_key}hz_continuous"]
        symbol_static_rates = cluster_rates[f"{doppler_key}hz_symbol_static"]
        paired_differences = [
            mobile - static for static, mobile in zip(reference_rates, mobile_rates, strict=True)
        ]
        difference_interval = cluster_bootstrap_interval(
            paired_differences,
            seed=BOOTSTRAP_SEED + 1 + 2 * index,
            alpha=per_interval_alpha,
            replicates=BOOTSTRAP_REPLICATES,
        )
        difference_mean = sum(paired_differences) / len(paired_differences)
        mobile_effects[doppler_key] = {
            "bler": sum(mobile_rates) / len(mobile_rates),
            "paired_mean_bler_increase": difference_mean,
            "paired_increase_cluster_bootstrap_interval": list(difference_interval),
            "pass": difference_mean >= MINIMUM_MOBILE_BLER_INCREASE
            and difference_interval[0] > 0.0,
        }

        intra_symbol_differences = [
            continuous - symbol_static
            for symbol_static, continuous in zip(symbol_static_rates, mobile_rates, strict=True)
        ]
        intra_symbol_interval = cluster_bootstrap_interval(
            intra_symbol_differences,
            seed=BOOTSTRAP_SEED + 2 + 2 * index,
            alpha=per_interval_alpha,
            replicates=BOOTSTRAP_REPLICATES,
        )
        intra_symbol_mean = sum(intra_symbol_differences) / len(intra_symbol_differences)
        intra_symbol_effects[doppler_key] = {
            "symbol_static_bler": sum(symbol_static_rates) / len(symbol_static_rates),
            "continuous_bler": sum(mobile_rates) / len(mobile_rates),
            "bler_increase": intra_symbol_mean,
            "increase_cluster_bootstrap_interval": list(intra_symbol_interval),
            "pass": intra_symbol_mean >= MINIMUM_INTRA_SYMBOL_BLER_INCREASE
            and intra_symbol_interval[0] > 0.0,
        }
    mobile_effect_pass = (
        reference_mean <= MAXIMUM_REFERENCE_BLER
        and reference_upper_bound <= MAXIMUM_REFERENCE_UPPER_BOUND
        and all(result["pass"] for result in mobile_effects.values())
    )
    intra_symbol_effect_pass = all(result["pass"] for result in intra_symbol_effects.values())
    acceptance = {
        "protocol": "paired static, symbol-static, and continuous echo Doppler at 750 and 1,500 Hz over independent seeded two-path channel draws; channel draw is the statistical unit; equal channel weighting",
        "familywise_confidence": 1.0 - FAMILYWISE_ALPHA,
        "interval_count": 5,
        "per_interval_confidence": 1.0 - per_interval_alpha,
        "maximum_reference_bler_point_estimate": MAXIMUM_REFERENCE_BLER,
        "maximum_reference_bler_upper_bound": MAXIMUM_REFERENCE_UPPER_BOUND,
        "minimum_mobile_bler_increase": MINIMUM_MOBILE_BLER_INCREASE,
        "reference_bler": reference_mean,
        "reference_channels_with_any_crc_failure": reference_channels_with_failures,
        "reference_bler_upper_bound": reference_upper_bound,
        "reference_bler_upper_bound_method": "one-sided Clopper-Pearson bound on channel-level any-failure prevalence; conservative upper bound for mean BLER",
        "mobile_effect_by_doppler_hz": mobile_effects,
        "minimum_intra_symbol_bler_increase": MINIMUM_INTRA_SYMBOL_BLER_INCREASE,
        "intra_symbol_effect_by_doppler_hz": intra_symbol_effects,
        "intra_symbol_control": "echo Doppler frozen at OFDM-symbol starts; retains inter-symbol aging and removes within-symbol phase evolution",
        "mobile_effect_pass": mobile_effect_pass,
        "intra_symbol_effect_pass": intra_symbol_effect_pass,
        "pass": mobile_effect_pass and intra_symbol_effect_pass,
        "inference_scope": "conditional on the preregistered synthetic two-path channel distribution, 6-RB/30-kHz PUSCH, injected AWGN, and tested payload generation; not field or deployment qualification",
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
            "sample_rate_hz": sample_rate_hz,
            "channel_draws": len(channel_seeds),
            "channel_seeds": list(channel_seeds),
            "blocks_per_channel": blocks_per_channel,
            "channel_distribution": {
                "direct_path_coefficient": [0.85, 0.0],
                "echo_delay_samples_inclusive": [2, 8],
                "echo_amplitude_uniform": [0.20, 0.50],
                "echo_phase_uniform_rad": [-math.pi, math.pi],
                "echo_doppler_hz_conditions": [MOBILE_DOPPLER_HZ, DOPPLER_STRESS_HZ],
                "waveform_awgn_variance": NOISE_VARIANCE,
            },
            "symbol_static_control": {
                "model": "freeze Doppler phase at each OFDM-symbol start, preserving inter-symbol aging",
                "symbol_length_samples": int(grid.fft_size + grid.cyclic_prefix_length),
            },
            "bootstrap": {
                "resampling_unit": "independent_channel_draw",
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": BOOTSTRAP_SEED,
                "familywise_alpha": FAMILYWISE_ALPHA,
                "simultaneous_claims": [
                    "static-reference BLER upper bound",
                    "continuous-Doppler paired BLER increase at 750 Hz",
                    "continuous versus symbol-static intra-symbol BLER increase at 750 Hz",
                    "continuous-Doppler paired BLER increase at 1,500 Hz",
                    "continuous versus symbol-static intra-symbol BLER increase at 1,500 Hz",
                ],
            },
        },
        "acceptance": acceptance,
        "channel_draw_records": records,
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
            "docs/research/results/g5-independent-channel-pusch-doppler-stress-20261024.json"
        ),
    )
    args = parser.parse_args()
    report = run_independent_channel_probe()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = report["acceptance"]
    print(
        f"wrote {args.output}: channels={report['configuration']['channel_draws']}, "
        f"baseline BLER={result['reference_bler']:.3f}, "
        f"mobile={result['mobile_effect_by_doppler_hz']}, "
        f"intra-symbol={result['intra_symbol_effect_by_doppler_hz']}, "
        f"decision={'PASS' if result['pass'] else 'NOT MET'}"
    )


if __name__ == "__main__":
    main()
