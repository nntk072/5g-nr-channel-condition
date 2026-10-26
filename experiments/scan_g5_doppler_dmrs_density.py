"""Development-only paired scan of PUSCH DM-RS density under channel aging.

The scan compares additional-position 0 and 3 on matched synthetic channels.
It reports both BLER and useful transport-block bits per slot because extra
DM-RS consumes data resources. It is exploratory and does not qualify G5.09.
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

SCHEMA = "umcd.g5_doppler_dmrs_density_scan.v1"
CHANNEL_SEEDS = tuple(range(84_001, 84_065))
BLOCKS_PER_CHANNEL = 8
WAVEFORM_NOISE_VARIANCE = 0.001
DMRS_ADDITIONAL_POSITIONS = (0, 3)
CHANNEL_CONDITIONS = ("static", "750hz_continuous")


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
        root / "integrations/sionna/nr_pusch.py",
        root / "integrations/sionna/ofdm_impairments.py",
        root / "receivers/nr/pusch.py",
    )
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def _validate_design(channel_seeds: tuple[int, ...], blocks_per_channel: int) -> None:
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


def run_dmrs_density_scan(
    *,
    channel_seeds: tuple[int, ...] = CHANNEL_SEEDS,
    blocks_per_channel: int = BLOCKS_PER_CHANNEL,
) -> dict[str, Any]:
    """Run paired PUSCH receive conditions with two standard DM-RS densities."""
    _validate_design(channel_seeds, blocks_per_channel)

    import torch
    from integrations.sionna.nr_pusch import SionnaPUSCHLSReceiver
    from integrations.sionna.ofdm_impairments import apply_time_varying_multipath
    from receivers.nr import NRPUSCHConfig

    receivers = {
        position: SionnaPUSCHLSReceiver(
            NRPUSCHConfig(
                n_size_bwp=6,
                subcarrier_spacing_khz=30,
                dmrs_additional_position=position,
            ),
            device="cpu",
        )
        for position in DMRS_ADDITIONAL_POSITIONS
    }
    reference_grid = receivers[0].resource_grid
    sample_rate_hz = float(reference_grid.fft_size * 30_000)
    per_configuration: dict[int, dict[str, list[list[bool]]]] = {
        position: {condition: [] for condition in CHANNEL_CONDITIONS}
        for position in DMRS_ADDITIONAL_POSITIONS
    }
    records: list[dict[str, Any]] = []

    for channel_seed in channel_seeds:
        channel_rng = random.Random(channel_seed)
        delay_samples = channel_rng.randint(2, 8)
        echo_amplitude = channel_rng.uniform(0.20, 0.50)
        echo_phase = channel_rng.uniform(-math.pi, math.pi)
        coefficients = (
            0.85 + 0.0j,
            echo_amplitude * complex(math.cos(echo_phase), math.sin(echo_phase)),
        )
        noise_seed = channel_seed + 2_000_000
        config_records: dict[str, Any] = {}

        for position, receiver in receivers.items():
            tx_seed = channel_seed + 1_000_000
            tx_grid, tx_bits = receiver.transmit(blocks_per_channel, seed=tx_seed)
            tx_waveform = receiver.modulate_slot_grid(tx_grid)
            noise_generator = torch.Generator(device="cpu").manual_seed(noise_seed)
            noise = torch.complex(
                torch.randn(tx_waveform.shape, generator=noise_generator),
                torch.randn(tx_waveform.shape, generator=noise_generator),
            ) * math.sqrt(WAVEFORM_NOISE_VARIANCE / 2.0)
            noise_sha256 = hashlib.sha256(noise.numpy().tobytes()).hexdigest()
            outcomes: dict[str, list[bool]] = {}

            for condition in CHANNEL_CONDITIONS:
                echo_doppler_hz = 0.0 if condition == "static" else 750.0
                received_waveform = (
                    apply_time_varying_multipath(
                        tx_waveform,
                        path_coefficients=coefficients,
                        path_delays_samples=(0, delay_samples),
                        path_dopplers_hz=(0.0, echo_doppler_hz),
                        sampling_frequency_hz=sample_rate_hz,
                    )
                    + noise
                )
                received_grid = receiver.demodulate_slot_waveform(received_waveform)
                _, crc_ok = receiver.decode_pusch(
                    received_grid, torch.tensor(WAVEFORM_NOISE_VARIANCE)
                )
                passed = [bool(value) for value in crc_ok.reshape(-1).tolist()]
                outcomes[condition] = passed
                per_configuration[position][condition].append(passed)

            config_records[str(position)] = {
                "payload_seed": tx_seed,
                "waveform_noise_sha256": noise_sha256,
                "transport_block_bits": int(tx_bits.shape[-1]),
                "dmrs_resource_elements_per_slot": int(receiver.pilot_mask.sum().item()),
                "crc_pass_by_condition": outcomes,
            }
        records.append(
            {
                "channel_seed": channel_seed,
                "noise_seed": noise_seed,
                "echo_delay_samples": delay_samples,
                "echo_coefficient_real_imag": [coefficients[1].real, coefficients[1].imag],
                "configuration_results": config_records,
            }
        )

    summaries: dict[str, Any] = {}
    for position, conditions in per_configuration.items():
        summaries[str(position)] = {}
        transport_block_bits = records[0]["configuration_results"][str(position)][
            "transport_block_bits"
        ]
        for condition, channel_outcomes in conditions.items():
            channel_blers = [1.0 - sum(passed) / len(passed) for passed in channel_outcomes]
            failures = sum(sum(not passed for passed in blocks) for blocks in channel_outcomes)
            block_count = sum(len(blocks) for blocks in channel_outcomes)
            successful_blocks = block_count - failures
            summaries[str(position)][condition] = {
                "transport_block_bits_per_payload": transport_block_bits,
                "blocks": block_count,
                "crc_failures": failures,
                "pooled_bler_descriptive": failures / block_count,
                "equal_channel_mean_bler_descriptive": sum(channel_blers) / len(channel_blers),
                "successful_transport_block_bits_per_slot_descriptive": (
                    successful_blocks * transport_block_bits / block_count
                ),
                "channel_bler_min_max": [min(channel_blers), max(channel_blers)],
            }

    goodput_delta_by_condition: dict[str, Any] = {}
    for condition in CHANNEL_CONDITIONS:
        per_channel_delta = []
        for record in records:
            low = record["configuration_results"]["0"]["crc_pass_by_condition"][condition]
            high = record["configuration_results"]["3"]["crc_pass_by_condition"][condition]
            low_tbs = record["configuration_results"]["0"]["transport_block_bits"]
            high_tbs = record["configuration_results"]["3"]["transport_block_bits"]
            low_goodput = sum(low) * low_tbs / blocks_per_channel
            high_goodput = sum(high) * high_tbs / blocks_per_channel
            per_channel_delta.append(high_goodput - low_goodput)
        goodput_delta_by_condition[condition] = {
            "mean_additional_position_3_minus_0_bits_per_slot_descriptive": sum(per_channel_delta)
            / len(per_channel_delta),
            "per_channel_delta_min_max_bits_per_slot": [
                min(per_channel_delta),
                max(per_channel_delta),
            ],
        }

    configuration = {
        "channel_seeds": list(channel_seeds),
        "channel_draws": len(channel_seeds),
        "blocks_per_channel": blocks_per_channel,
        "payload_seed_offset": 1_000_000,
        "noise_seed_offset": 2_000_000,
        "waveform_awgn_complex_variance": WAVEFORM_NOISE_VARIANCE,
        "channel_distribution": {
            "direct_coefficient": [0.85, 0.0],
            "echo_delay_samples_inclusive": [2, 8],
            "echo_amplitude_uniform": [0.20, 0.50],
            "echo_phase_uniform_rad": [-math.pi, math.pi],
            "echo_doppler_hz_conditions": {"static": 0.0, "750hz_continuous": 750.0},
        },
        "nr_configuration": {
            "n_size_bwp_rb": 6,
            "subcarrier_spacing_hz": 30_000,
            "fft_size": int(reference_grid.fft_size),
            "cyclic_prefix_samples": int(reference_grid.cyclic_prefix_length),
            "sample_rate_hz": sample_rate_hz,
            "dmrs_additional_positions": list(DMRS_ADDITIONAL_POSITIONS),
            "decoder_channel_estimator": "Sionna PUSCH receiver default DM-RS LS with linear interpolation",
        },
        "pairing": "same channel realization, waveform AWGN seed, and transmitter seed across DM-RS configurations; transport-block sizes differ with pilot overhead",
        "interpretation": "development-only descriptive scan; no confirmatory inference or G5 qualification",
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
        "summary_by_dmrs_additional_position": summaries,
        "goodput_delta_by_condition": goodput_delta_by_condition,
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
        default=Path("docs/research/results/g5-doppler-dmrs-density-scan-20261024.json"),
    )
    args = parser.parse_args()
    report = run_dmrs_density_scan()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"wrote {args.output}: channels={report['configuration']['channel_draws']}, "
        f"goodput delta={report['goodput_delta_by_condition']}; "
        "development-only, no acceptance decision"
    )


if __name__ == "__main__":
    main()
