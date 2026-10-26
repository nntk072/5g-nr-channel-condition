"""Development-only AWGN operating-point scan for the G5 Doppler fixture.

This scan helps select a receiver-noise operating point for a later frozen
experiment. It produces descriptive paired outcomes only and never qualifies
G5.09 or changes any acceptance criterion.
"""

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

SCHEMA = "umcd.g5_doppler_operating_point_scan.v1"
CHANNEL_SEEDS = tuple(range(83_001, 83_017))
BLOCKS_PER_CHANNEL = 8
NOISE_VARIANCES = (0.001, 0.003, 0.01, 0.03)
DOPPLER_CONDITIONS = (
    (0.0, "continuous"),
    (750.0, "symbol_static"),
    (750.0, "continuous"),
    (1_500.0, "symbol_static"),
    (1_500.0, "continuous"),
)


def _condition_key(doppler_hz: float, model: str) -> str:
    return f"{int(doppler_hz)}hz_{model}"


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
        root / "experiments/g5_independent_channel_receiver.py",
        root / "integrations/sionna/ofdm_impairments.py",
        root / "integrations/sionna/nr_pusch.py",
        root / "receivers/nr/pusch.py",
    )
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def _validate_design(channel_seeds: tuple[int, ...], noise_variances: tuple[float, ...]) -> None:
    if not channel_seeds or len(set(channel_seeds)) != len(channel_seeds):
        raise ValueError("channel_seeds must be nonempty and unique")
    if any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in channel_seeds
    ):
        raise ValueError("channel_seeds must contain nonnegative integers")
    if not noise_variances or len(set(noise_variances)) != len(noise_variances):
        raise ValueError("noise_variances must be nonempty and unique")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
        for value in noise_variances
    ):
        raise ValueError("noise_variances must contain finite positive real values")


def run_operating_point_scan(
    *,
    channel_seeds: tuple[int, ...] = CHANNEL_SEEDS,
    blocks_per_channel: int = BLOCKS_PER_CHANNEL,
    noise_variances: tuple[float, ...] = NOISE_VARIANCES,
) -> dict[str, Any]:
    """Return paired descriptive outcomes across noise levels and Doppler models."""
    _validate_design(channel_seeds, noise_variances)
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
    records: list[dict[str, Any]] = []
    observations: dict[float, dict[str, list[list[bool]]]] = {
        float(variance): {_condition_key(*condition): [] for condition in DOPPLER_CONDITIONS}
        for variance in noise_variances
    }

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
        unit_noise = torch.complex(
            torch.randn(tx_waveform.shape, generator=noise_rng),
            torch.randn(tx_waveform.shape, generator=noise_rng),
        ) / math.sqrt(2.0)
        channel_record: dict[str, Any] = {
            "channel_seed": channel_seed,
            "payload_seed": channel_seed + 1_000_000,
            "noise_seed": channel_seed + 2_000_000,
            "echo_delay_samples": delay_samples,
            "echo_coefficient_real_imag": [coefficients[1].real, coefficients[1].imag],
            "outcomes_by_noise_variance": {},
        }

        for variance in noise_variances:
            noise_variance = float(variance)
            waveform_noise = unit_noise * math.sqrt(noise_variance)
            outcomes_for_variance: dict[str, list[bool]] = {}
            for doppler_hz, channel_model in DOPPLER_CONDITIONS:
                condition = _condition_key(doppler_hz, channel_model)
                received = (
                    apply_time_varying_multipath(
                        tx_waveform,
                        path_coefficients=coefficients,
                        path_delays_samples=(0, delay_samples),
                        path_dopplers_hz=(0.0, doppler_hz),
                        sampling_frequency_hz=sample_rate_hz,
                        symbol_length_samples=(
                            int(grid.fft_size + grid.cyclic_prefix_length)
                            if channel_model == "symbol_static"
                            else None
                        ),
                    )
                    + waveform_noise
                )
                received_grid = receiver.demodulate_slot_waveform(received)
                _, crc_ok = receiver.decode_pusch(received_grid, torch.tensor(noise_variance))
                block_pass = [bool(value) for value in crc_ok.reshape(-1).tolist()]
                outcomes_for_variance[condition] = block_pass
                observations[noise_variance][condition].append(block_pass)
            channel_record["outcomes_by_noise_variance"][str(noise_variance)] = (
                outcomes_for_variance
            )
        records.append(channel_record)

    summaries: dict[str, Any] = {}
    for variance, conditions in observations.items():
        summary_by_condition: dict[str, Any] = {}
        for condition, per_channel in conditions.items():
            channel_rates = [1.0 - sum(passed) / len(passed) for passed in per_channel]
            failures = sum(sum(not passed for passed in blocks) for blocks in per_channel)
            block_count = sum(len(blocks) for blocks in per_channel)
            summary_by_condition[condition] = {
                "blocks": block_count,
                "crc_failures": failures,
                "pooled_bler_descriptive": failures / block_count,
                "equal_channel_mean_bler_descriptive": sum(channel_rates) / len(channel_rates),
                "channel_bler_min_max": [min(channel_rates), max(channel_rates)],
            }
        reference = [1.0 - sum(passed) / len(passed) for passed in conditions["0hz_continuous"]]
        comparisons: dict[str, Any] = {}
        for doppler_hz in (750, 1_500):
            static_key = f"{doppler_hz}hz_symbol_static"
            continuous_key = f"{doppler_hz}hz_continuous"
            mobile = [1.0 - sum(passed) / len(passed) for passed in conditions[continuous_key]]
            symbol_static = [1.0 - sum(passed) / len(passed) for passed in conditions[static_key]]
            comparisons[str(doppler_hz)] = {
                "continuous_minus_static_mean_bler_descriptive": sum(
                    right - left for left, right in zip(reference, mobile, strict=True)
                )
                / len(reference),
                "continuous_minus_symbol_static_mean_bler_descriptive": sum(
                    right - left for left, right in zip(symbol_static, mobile, strict=True)
                )
                / len(reference),
            }
        summaries[str(variance)] = {
            "summary_by_condition": summary_by_condition,
            "paired_equal_channel_effects_descriptive": comparisons,
        }

    configuration = {
        "channel_seeds": list(channel_seeds),
        "payload_seed_offset": 1_000_000,
        "noise_seed_offset": 2_000_000,
        "channel_draws": len(channel_seeds),
        "blocks_per_channel": blocks_per_channel,
        "noise_variances": [float(value) for value in noise_variances],
        "noise_pairing": "one standard proper-complex Gaussian waveform per channel, scaled identically across Doppler conditions and noise levels",
        "channel_distribution": {
            "direct_coefficient": [0.85, 0.0],
            "echo_delay_samples_inclusive": [2, 8],
            "echo_amplitude_uniform": [0.20, 0.50],
            "echo_phase_uniform_rad": [-math.pi, math.pi],
            "doppler_conditions": [list(item) for item in DOPPLER_CONDITIONS],
        },
        "nr_configuration": {
            "n_size_bwp_rb": 6,
            "subcarrier_spacing_hz": 30_000,
            "fft_size": int(grid.fft_size),
            "cyclic_prefix_samples": int(grid.cyclic_prefix_length),
            "sample_rate_hz": sample_rate_hz,
        },
        "interpretation": "development-only descriptive scan; all reported BLER/effect values are selection diagnostics and cannot qualify G5 or act as confirmatory inference",
    }
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "DEVELOPMENT_ONLY_NOT_QUALIFIED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": _revision(),
        "command": list(__import__("sys").orig_argv),
        "source_sha256": _source_fingerprints(),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": _installed_version("torch"),
            "sionna": _installed_version("sionna"),
            "device": "cpu",
        },
        "configuration": configuration,
        "summary_by_noise_variance": summaries,
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
        default=Path("docs/research/results/g5-doppler-operating-point-scan-20261024.json"),
    )
    args = parser.parse_args()
    report = run_operating_point_scan()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"wrote {args.output}: channels={report['configuration']['channel_draws']}, "
        f"noise_variances={report['configuration']['noise_variances']}; "
        "development-only, no acceptance decision"
    )


if __name__ == "__main__":
    main()
