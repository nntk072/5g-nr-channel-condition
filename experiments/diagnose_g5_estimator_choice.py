"""Post-hoc comparison of DM-RS interpolation on failed G5.09 reference draws.

The input channels were selected because the frozen confirmation had static
position-0 failures. Results are diagnostic and cannot qualify G5 or alter its
frozen acceptance criteria.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from integrations.sionna.nr_pusch import SionnaPUSCHLSReceiver
from integrations.sionna.ofdm_impairments import apply_time_varying_multipath
from receivers.nr import NRPUSCHConfig

CONFIRMATION = Path("docs/research/results/g5-doppler-dmrs-density-confirm-20261024.json")
INTERPOLATORS = ("nn", "lin", "lin_time_avg")


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _confirmation_bytes(path: Path) -> bytes:
    payload = path.read_bytes()
    report = json.loads(payload)
    unhashed = dict(report)
    digest = unhashed.pop("result_sha256", None)
    if report.get("schema") != "umcd.g5_doppler_dmrs_density_confirmatory.v1":
        raise ValueError("unsupported G5.09 confirmation artifact")
    if digest != _canonical_sha256(unhashed):
        raise ValueError("confirmation result digest does not match its contents")
    return payload


def diagnose_estimator_choice(confirmation_path: Path = CONFIRMATION) -> dict[str, Any]:
    """Replay selected failures through three configured LS interpolators."""
    confirmation_bytes = _confirmation_bytes(confirmation_path)
    confirmation = json.loads(confirmation_bytes)
    selected = [
        record
        for record in confirmation["confirmatory_channel_draw_records"]
        if not all(record["configuration_results"]["0"]["crc_pass_by_condition"]["static"])
    ]
    if not selected:
        raise ValueError("confirmation has no static position-0 failures to diagnose")

    observations: list[dict[str, Any]] = []
    for record in selected:
        seed = int(record["channel_seed"])
        delay = int(record["echo_delay_samples"])
        echo_real, echo_imag = record["echo_coefficient_real_imag"]
        echo = complex(float(echo_real), float(echo_imag))
        receivers = {
            interpolation: SionnaPUSCHLSReceiver(
                NRPUSCHConfig(
                    n_size_bwp=6,
                    subcarrier_spacing_khz=30,
                    dmrs_additional_position=0,
                    interpolation_type=interpolation,
                ),
                device="cpu",
            )
            for interpolation in INTERPOLATORS
        }
        reference_receiver = receivers["lin"]
        tx_grid, _ = reference_receiver.transmit(8, seed=seed + 1_000_000)
        tx_waveform = reference_receiver.modulate_slot_grid(tx_grid)
        noise_generator = torch.Generator(device="cpu").manual_seed(int(record["noise_seed"]))
        noise = torch.complex(
            torch.randn(tx_waveform.shape, generator=noise_generator),
            torch.randn(tx_waveform.shape, generator=noise_generator),
        ) * math.sqrt(0.001 / 2.0)
        sample_rate_hz = reference_receiver.resource_grid.fft_size * 30_000
        channel_waveform = apply_time_varying_multipath(
            tx_waveform,
            path_coefficients=(0.85 + 0.0j, echo),
            path_delays_samples=(0, delay),
            path_dopplers_hz=(0.0, 0.0),
            sampling_frequency_hz=sample_rate_hz,
        )
        received_grid = reference_receiver.demodulate_slot_waveform(channel_waveform + noise)
        noiseless_grid = reference_receiver.demodulate_slot_waveform(channel_waveform)
        grid = reference_receiver.resource_grid
        frequency_bins = torch.arange(grid.fft_size, dtype=torch.float32) - grid.fft_size // 2
        response = 0.85 + echo * torch.exp(-2j * math.pi * frequency_bins * delay / grid.fft_size)
        expected = response[grid.effective_subcarrier_ind].reshape(1, 1, 1, 1, 1, 1, -1)

        by_interpolator: dict[str, Any] = {}
        for interpolation, receiver in receivers.items():
            _, crc = receiver.decode_pusch(received_grid, torch.tensor(0.001))
            estimate = receiver.estimate(noiseless_grid, torch.tensor(0.0)).h_hat
            nmse = float(
                (estimate - expected).abs().square().mean() / expected.abs().square().mean()
            )
            by_interpolator[interpolation] = {
                "crc_passes": [bool(value) for value in crc.reshape(-1).tolist()],
                "noiseless_estimate_nmse": nmse,
            }
        recorded = record["configuration_results"]["0"]["crc_pass_by_condition"]["static"]
        if by_interpolator["lin"]["crc_passes"] != recorded:
            raise ValueError(f"linear interpolation replay differs for channel seed {seed}")
        observations.append(
            {
                "channel_seed": seed,
                "echo_delay_samples": delay,
                "interpolators": by_interpolator,
            }
        )

    totals = {
        interpolation: {
            "crc_passes": sum(
                sum(row["interpolators"][interpolation]["crc_passes"]) for row in observations
            ),
            "blocks": sum(
                len(row["interpolators"][interpolation]["crc_passes"]) for row in observations
            ),
            "mean_noiseless_estimate_nmse": sum(
                row["interpolators"][interpolation]["noiseless_estimate_nmse"]
                for row in observations
            )
            / len(observations),
        }
        for interpolation in INTERPOLATORS
    }
    root = Path(__file__).resolve().parents[1]
    sources = (
        "experiments/diagnose_g5_estimator_choice.py",
        "integrations/sionna/nr_pusch.py",
        "integrations/sionna/ofdm_impairments.py",
        "receivers/nr/pusch.py",
    )
    result = {
        "schema": "umcd.g5_estimator_choice_posthoc_diagnostic.v1",
        "status": "POST_HOC_DIAGNOSTIC_NOT_QUALIFICATION",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "confirmation_artifact": confirmation_path.as_posix(),
        "confirmation_artifact_sha256": hashlib.sha256(confirmation_bytes).hexdigest(),
        "selection_rule": "all channel draws with any frozen position-0 static CRC failure",
        "selection_is_outcome_dependent": True,
        "interpolators": list(INTERPOLATORS),
        "independent_channel_count": len(observations),
        "blocks_per_channel": 8,
        "totals": totals,
        "observations": observations,
        "source_sha256": {
            path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in sources
        },
        "interpretation": (
            "This selected-failure replay compares estimator behavior only on channels chosen "
            "after the frozen outcome was observed. It is useful for receiver diagnosis, not "
            "population inference, estimator selection, protocol changes, or G5.09 closure. "
            "Any revised receiver requires a new preregistered protocol and disjoint channel seeds."
        ),
    }
    result["result_sha256"] = _canonical_sha256(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation", type=Path, default=CONFIRMATION)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/research/results/g5-estimator-choice-posthoc-20261024.json"),
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing diagnostic: {args.output}")
    result = diagnose_estimator_choice(args.confirmation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "result_sha256": result["result_sha256"]}))


if __name__ == "__main__":
    main()
