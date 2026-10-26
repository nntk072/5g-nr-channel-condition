"""Run the frozen finite delay-domain DFT G5 receiver confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.g5_cp_bounded_dft_estimator import make_finite_delay_dft_decoder
from experiments.g5_independent_channel_receiver import (
    clopper_pearson_upper_bound,
    cluster_bootstrap_interval,
)

SCHEMA = "umcd.g5_delay_domain_dft_confirmatory.v1"
CHANNEL_SEEDS = tuple(range(88_001, 88_065))
BLOCKS_PER_CHANNEL = 8
NOISE_VARIANCE = 0.001
POSITIONS = (0, 3)
CONDITIONS = ("static", "750hz_continuous")
ALPHA = 0.01
CLAIM_COUNT = 3
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED_BLER = 88_099
BOOTSTRAP_SEED_GOODPUT = 88_100


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )


def _source_fingerprints() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    paths = (
        Path(__file__).resolve(),
        root / "experiments/g5_cp_bounded_dft_estimator.py",
        root / "experiments/g5_independent_channel_receiver.py",
        root / "integrations/sionna/nr_pusch.py",
        root / "integrations/sionna/ofdm_impairments.py",
        root / "receivers/nr/pusch.py",
    )
    return {p.relative_to(root).as_posix(): _sha256(p.read_bytes()) for p in paths}


def _bler(records: list[dict[str, Any]], position: str, condition: str) -> list[float]:
    return [
        1.0
        - sum(r["configuration_results"][position]["crc_pass_by_condition"][condition])
        / BLOCKS_PER_CHANNEL
        for r in records
    ]


def _goodput(records: list[dict[str, Any]], position: str, condition: str) -> list[float]:
    return [
        sum(r["configuration_results"][position]["crc_pass_by_condition"][condition])
        * r["configuration_results"][position]["transport_block_bits"]
        / BLOCKS_PER_CHANNEL
        for r in records
    ]


def _run_channel_experiment() -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
        for position in POSITIONS
    }
    decoders = {
        p: make_finite_delay_dft_decoder(r, max_delay_samples=8) for p, r in receivers.items()
    }
    grid = receivers[0].resource_grid
    sample_rate_hz = float(grid.fft_size * 30_000)
    records: list[dict[str, Any]] = []

    for channel_seed in CHANNEL_SEEDS:
        rng = random.Random(channel_seed)
        delay = rng.randint(2, 8)
        amplitude = rng.uniform(0.20, 0.50)
        phase = rng.uniform(-math.pi, math.pi)
        echo = amplitude * complex(math.cos(phase), math.sin(phase))
        tx_seed = channel_seed + 1_000_000
        noise_seed = channel_seed + 2_000_000
        by_position: dict[str, Any] = {}
        noise_digests: set[str] = set()

        for position, receiver in receivers.items():
            tx_grid, tx_bits = receiver.transmit(BLOCKS_PER_CHANNEL, seed=tx_seed)
            tx_waveform = receiver.modulate_slot_grid(tx_grid)
            generator = torch.Generator(device="cpu").manual_seed(noise_seed)
            noise = torch.complex(
                torch.randn(tx_waveform.shape, generator=generator),
                torch.randn(tx_waveform.shape, generator=generator),
            ) * math.sqrt(NOISE_VARIANCE / 2.0)
            noise_digests.add(_sha256(noise.numpy().tobytes()))
            outcomes: dict[str, list[bool]] = {}
            for condition in CONDITIONS:
                doppler = 0.0 if condition == "static" else 750.0
                waveform = (
                    apply_time_varying_multipath(
                        tx_waveform,
                        path_coefficients=(0.85 + 0j, echo),
                        path_delays_samples=(0, delay),
                        path_dopplers_hz=(0.0, doppler),
                        sampling_frequency_hz=sample_rate_hz,
                    )
                    + noise
                )
                received_grid = receiver.demodulate_slot_waveform(waveform)
                _, crc = decoders[position](received_grid, torch.tensor(NOISE_VARIANCE))
                outcomes[condition] = [bool(v) for v in crc.reshape(-1).tolist()]
            by_position[str(position)] = {
                "payload_seed": tx_seed,
                "waveform_noise_sha256": _sha256(noise.numpy().tobytes()),
                "transport_block_bits": int(tx_bits.shape[-1]),
                "dmrs_resource_elements_per_slot": int(receiver.pilot_mask.sum().item()),
                "crc_pass_by_condition": outcomes,
            }
        if len(noise_digests) != 1:
            raise RuntimeError("paired DM-RS configurations did not produce identical noise")
        records.append(
            {
                "channel_seed": channel_seed,
                "payload_seed": tx_seed,
                "waveform_noise_seed": noise_seed,
                "echo_delay_samples": delay,
                "echo_amplitude": amplitude,
                "echo_phase_rad": phase,
                "echo_coefficient_real_imag": [echo.real, echo.imag],
                "configuration_results": by_position,
            }
        )
    configuration = {
        "channel_seeds": list(CHANNEL_SEEDS),
        "channel_draws": len(CHANNEL_SEEDS),
        "blocks_per_channel": BLOCKS_PER_CHANNEL,
        "waveform_awgn_complex_variance": NOISE_VARIANCE,
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
            "fft_size": int(grid.fft_size),
            "cyclic_prefix_samples": int(grid.cyclic_prefix_length),
            "sample_rate_hz": sample_rate_hz,
            "dmrs_additional_positions": list(POSITIONS),
            "channel_estimator": "raw-DMRS finite delay-domain complex LS, integer delays 0..8, linear temporal interpolation",
        },
        "pairing": "same channel, payload seed, and waveform noise across DM-RS settings; same settings across static/mobile conditions",
    }
    return configuration, records


def run_frozen_confirmation() -> dict[str, Any]:
    """Execute the protocol-registered 64-channel receiver comparison."""
    root = Path(__file__).resolve().parents[1]
    protocol_path = root / "docs/research/G5_DELAY_DOMAIN_DFT_CONFIRMATORY_PROTOCOL_2026-10-24.md"
    protocol_bytes = protocol_path.read_bytes()
    configuration, records = _run_channel_experiment()
    per_claim_alpha = ALPHA / CLAIM_COUNT
    static_rates = _bler(records, "0", "static")
    static_failures = sum(rate > 0 for rate in static_rates)
    static_upper = clopper_pearson_upper_bound(static_failures, len(records), alpha=per_claim_alpha)
    mobile_reduction = [
        a - b
        for a, b in zip(
            _bler(records, "0", "750hz_continuous"),
            _bler(records, "3", "750hz_continuous"),
            strict=True,
        )
    ]
    mobile_goodput = [
        b - a
        for a, b in zip(
            _goodput(records, "0", "750hz_continuous"),
            _goodput(records, "3", "750hz_continuous"),
            strict=True,
        )
    ]
    bler_ci = cluster_bootstrap_interval(
        mobile_reduction,
        seed=BOOTSTRAP_SEED_BLER,
        alpha=per_claim_alpha,
        replicates=BOOTSTRAP_REPLICATES,
    )
    goodput_ci = cluster_bootstrap_interval(
        mobile_goodput,
        seed=BOOTSTRAP_SEED_GOODPUT,
        alpha=per_claim_alpha,
        replicates=BOOTSTRAP_REPLICATES,
    )
    bler_mean = sum(mobile_reduction) / len(mobile_reduction)
    goodput_mean = sum(mobile_goodput) / len(mobile_goodput)
    acceptance: dict[str, Any] = {
        "familywise_alpha": ALPHA,
        "family_claim_count": CLAIM_COUNT,
        "per_claim_alpha": per_claim_alpha,
        "bootstrap_resampling_unit": "independent channel seed; eight transport blocks nested",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed_bler": BOOTSTRAP_SEED_BLER,
        "bootstrap_seed_goodput": BOOTSTRAP_SEED_GOODPUT,
        "static_reference": {
            "position_0_mean_bler": sum(static_rates) / len(static_rates),
            "channels_with_any_failure": static_failures,
            "independent_channels": len(records),
            "one_sided_exact_family_adjusted_channel_failure_upper_bound": static_upper,
            "pass": sum(static_rates) / len(static_rates) <= 0.05 and static_upper <= 0.10,
        },
        "mobile_bler_reduction_position_0_minus_3": {
            "equal_channel_mean": bler_mean,
            "minimum_required_mean_reduction": 0.10,
            "family_adjusted_cluster_bootstrap_interval": list(bler_ci),
            "pass": bler_mean >= 0.10 and bler_ci[0] > 0,
        },
        "mobile_goodput_increase_position_3_minus_0_bits_per_slot": {
            "equal_channel_mean": goodput_mean,
            "family_adjusted_cluster_bootstrap_interval": list(goodput_ci),
            "pass": goodput_ci[0] > 0,
        },
        "descriptive_static_bler_by_position": {
            str(p): sum(_bler(records, str(p), "static")) / len(records) for p in POSITIONS
        },
    }
    passed = all(
        acceptance[k]["pass"]
        for k in (
            "static_reference",
            "mobile_bler_reduction_position_0_minus_3",
            "mobile_goodput_increase_position_3_minus_0_bits_per_slot",
        )
    )
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "PASS_NAMED_SYNTHETIC_CONFIGURATION_ONLY" if passed else "NOT_MET",
        "g5_09_closed": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_path": "docs/research/G5_DELAY_DOMAIN_DFT_CONFIRMATORY_PROTOCOL_2026-10-24.md",
        "protocol_sha256": _sha256(protocol_bytes),
        "source_sha256": _source_fingerprints(),
        "configuration": configuration,
        "configuration_sha256": _canonical_sha256(configuration),
        "acceptance": acceptance,
        "interpretation": "Evidence is limited to this synthetic two-path PUSCH configuration and estimator; it does not qualify field channels, general mobility, G5.09, or production RT generation.",
        "confirmatory_channel_draw_records": records,
    }
    report["result_sha256"] = _canonical_sha256(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/research/results/g5-delay-domain-dft-confirm-20261024.json"),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen result: {args.output}")
    report = run_frozen_confirmation()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}: {report['status']}; acceptance={report['acceptance']}")


if __name__ == "__main__":
    main()
