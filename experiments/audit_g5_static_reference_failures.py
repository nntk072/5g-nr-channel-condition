"""Post-hoc receiver-path diagnostic for failed G5 static-reference draws.

This script intentionally replays only channel draws already identified as
failures in the frozen confirmation. Its outputs are diagnostic, never
confirmatory evidence or a qualification result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from integrations.sionna.nr_pusch import SionnaPUSCHLSReceiver
from integrations.sionna.ofdm_impairments import apply_time_varying_multipath
from receivers.nr import NRPUSCHConfig

CONFIRMATION_PATH = Path("docs/research/results/g5-doppler-dmrs-density-confirm-20261024.json")
CONFIRMATION_SOURCE_COMMIT = "8ba3482539ccc4774175f86b4e9938081424a268"
DIAGNOSTIC_SOURCES = (
    "experiments/audit_g5_static_reference_failures.py",
    "experiments/qualify_g5_doppler_dmrs_density.py",
    "experiments/scan_g5_doppler_dmrs_density.py",
    "experiments/g5_independent_channel_receiver.py",
    "integrations/sionna/nr_pusch.py",
    "integrations/sionna/ofdm_impairments.py",
    "receivers/nr/pusch.py",
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )


def _source_hashes(root: Path) -> dict[str, str]:
    return {name: _sha256((root / name).read_bytes()) for name in DIAGNOSTIC_SOURCES}


def _validate_confirmation_sources(root: Path, declared_sources: dict[str, Any]) -> dict[str, str]:
    """Check current bytes or the frozen pre-change source commit for each dependency."""
    validation: dict[str, str] = {}
    for name, expected in declared_sources.items():
        if _sha256((root / name).read_bytes()) == expected:
            validation[name] = "current_worktree"
            continue
        snapshot = subprocess.run(
            ["git", "show", f"{CONFIRMATION_SOURCE_COMMIT}:{name}"],
            cwd=root,
            check=False,
            capture_output=True,
        )
        if snapshot.returncode == 0 and _sha256(snapshot.stdout) == expected:
            validation[name] = f"git:{CONFIRMATION_SOURCE_COMMIT}"
            continue
        raise ValueError(f"confirmation source fingerprint is unverifiable: {name}")
    return validation


def replay_static_reference_failures(
    confirmation_path: Path = CONFIRMATION_PATH,
) -> dict[str, Any]:
    """Compare retained DM-RS outcomes with an oracle-CSI decoder replay."""
    root = Path(__file__).resolve().parents[1]
    confirmation_bytes = confirmation_path.read_bytes()
    confirmation = json.loads(confirmation_bytes.decode("utf-8"))
    if confirmation.get("schema") != "umcd.g5_doppler_dmrs_density_confirmatory.v1":
        raise ValueError("unsupported G5 confirmation artifact")
    declared_sources = confirmation.get("source_sha256")
    if not isinstance(declared_sources, dict):
        raise ValueError("confirmation artifact lacks source fingerprints")
    source_validation = _validate_confirmation_sources(root, declared_sources)

    failures = [
        item
        for item in confirmation["confirmatory_channel_draw_records"]
        if not all(item["configuration_results"]["0"]["crc_pass_by_condition"]["static"])
    ]
    if not failures:
        raise ValueError("confirmation contains no static-reference failures to diagnose")

    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(
            n_size_bwp=6,
            subcarrier_spacing_khz=30,
            dmrs_additional_position=0,
        ),
        device="cpu",
    )
    linear_estimator = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(
            n_size_bwp=6,
            subcarrier_spacing_khz=30,
            dmrs_additional_position=0,
            interpolation_type="lin",
        ),
        device="cpu",
    )
    grid = receiver.resource_grid
    sample_rate_hz = float(grid.fft_size * 30_000)
    observations: list[dict[str, Any]] = []

    for item in failures:
        seed = int(item["channel_seed"])
        delay = int(item["echo_delay_samples"])
        echo_real, echo_imag = item["echo_coefficient_real_imag"]
        echo = complex(float(echo_real), float(echo_imag))
        tx_grid, _ = receiver.transmit(8, seed=seed + 1_000_000)
        tx_waveform = receiver.modulate_slot_grid(tx_grid)
        generator = torch.Generator(device="cpu").manual_seed(int(item["noise_seed"]))
        noise = torch.complex(
            torch.randn(tx_waveform.shape, generator=generator),
            torch.randn(tx_waveform.shape, generator=generator),
        ) * math.sqrt(0.001 / 2.0)
        channel_waveform = apply_time_varying_multipath(
            tx_waveform,
            path_coefficients=(0.85 + 0.0j, echo),
            path_delays_samples=(0, delay),
            path_dopplers_hz=(0.0, 0.0),
            sampling_frequency_hz=sample_rate_hz,
        )
        channel_only_grid = receiver.demodulate_slot_waveform(channel_waveform)
        received_grid = receiver.demodulate_slot_waveform(channel_waveform + noise)
        _, estimated_crc = receiver.decode_pusch(received_grid, torch.tensor(0.001))

        frequency_bins = torch.arange(grid.fft_size, dtype=torch.float32) - grid.fft_size // 2
        response = 0.85 + echo * torch.exp(-2j * math.pi * frequency_bins * delay / grid.fft_size)
        truth = response.reshape(1, 1, 1, 1, 1, 1, -1).expand(
            8, 1, 1, 1, 1, grid.num_ofdm_symbols, grid.fft_size
        )
        _, oracle_crc = receiver.decode_pusch_with_channel_truth(
            received_grid, torch.tensor(0.001), truth
        )
        estimate = linear_estimator.estimate(received_grid, torch.tensor(0.001)).h_hat
        noiseless_estimate = linear_estimator.estimate(channel_only_grid, torch.tensor(0.0)).h_hat
        active_response = response[grid.effective_subcarrier_ind].reshape(1, 1, 1, 1, 1, 1, -1)
        estimator_nmse = float(
            torch.mean(torch.abs(estimate - active_response) ** 2)
            / torch.mean(torch.abs(active_response) ** 2)
        )
        noiseless_estimator_nmse = float(
            torch.mean(torch.abs(noiseless_estimate - active_response) ** 2)
            / torch.mean(torch.abs(active_response) ** 2)
        )
        estimated_passes = [bool(value) for value in estimated_crc.reshape(-1).tolist()]
        oracle_passes = [bool(value) for value in oracle_crc.reshape(-1).tolist()]
        retained_passes = item["configuration_results"]["0"]["crc_pass_by_condition"]["static"]
        if estimated_passes != retained_passes:
            raise ValueError(f"DM-RS replay differs from retained outcomes for seed {seed}")
        observations.append(
            {
                "channel_seed": seed,
                "echo_delay_samples": delay,
                "echo_delay_us": delay / sample_rate_hz * 1e6,
                "echo_amplitude": abs(echo),
                "dmrs_receiver_crc_passes": sum(estimated_passes),
                "oracle_frequency_csi_crc_passes": sum(oracle_passes),
                "blocks": len(estimated_passes),
                "estimated_channel_nmse": estimator_nmse,
                "noiseless_estimated_channel_nmse": noiseless_estimator_nmse,
            }
        )

    return {
        "schema": "umcd.g5_static_reference_failure_path_diagnostic.v1",
        "status": "POST_HOC_DIAGNOSTIC_NOT_QUALIFICATION",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "confirmation_artifact": str(confirmation_path),
        "confirmation_artifact_sha256": _sha256(confirmation_bytes),
        "confirmation_source_validation": source_validation,
        "source_sha256": _source_hashes(root),
        "selection_rule": (
            "All channel draws with any retained position-0 static CRC failure; selected "
            "from the already-inspected frozen confirmation outcomes."
        ),
        "oracle_condition": (
            "The Sionna PUSCH receiver receives the analytically derived full-band frequency "
            "response for the two static integer-delay paths. This oracle omits ISI when a "
            "delay exceeds the cyclic prefix and is a diagnostic comparator only."
        ),
        "observations": observations,
        "interpretation": (
            "A DM-RS versus oracle-CSI outcome gap localizes these failures to the estimated-"
            "channel receiver path under the frozen synthetic draws. Post-hoc selected cases "
            "do not estimate population performance, validate a channel law, or qualify G5.09."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation", type=Path, default=CONFIRMATION_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/research/results/g5-static-reference-failure-path-diagnostic-20261024.json"
        ),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    report = replay_static_reference_failures(args.confirmation)
    report["result_sha256"] = _canonical_sha256(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(
        f"wrote {args.output}: draws={len(report['observations'])}, "
        "POST_HOC_DIAGNOSTIC_NOT_QUALIFICATION"
    )


if __name__ == "__main__":
    main()
