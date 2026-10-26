"""Post-hoc G5 diagnostic of CP-bounded DFT estimation on frozen failures."""

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

from experiments.g5_cp_bounded_dft_estimator import (
    FiniteDelayDFTChannelEstimator,
    make_finite_delay_dft_decoder,
)

CONFIRMATION_PATH = Path("docs/research/results/g5-doppler-dmrs-density-confirm-20261024.json")
DEFAULT_OUTPUT = Path("docs/research/results/g5-cp-bounded-dft-selected-failures-20261024.json")
CYCLIC_PREFIX_SAMPLES = 7
SOURCE_PATHS = (
    "experiments/audit_g5_cp_bounded_dft_diagnostic.py",
    "experiments/g5_cp_bounded_dft_estimator.py",
    "integrations/sionna/nr_pusch.py",
    "integrations/sionna/ofdm_impairments.py",
    "receivers/nr/pusch.py",
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return _sha256(encoded)


def _validate_confirmation(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    report = json.loads(raw.decode("utf-8"))
    if report.get("schema") != "umcd.g5_doppler_dmrs_density_confirmatory.v1":
        raise ValueError("unsupported G5 confirmation schema")
    expected_digest = report.get("result_sha256")
    unhashed = dict(report)
    unhashed.pop("result_sha256", None)
    if expected_digest != _canonical_sha256(unhashed):
        raise ValueError("frozen G5 confirmation digest does not match its contents")
    return raw, report


def diagnose_selected_failures(confirmation_path: Path = CONFIRMATION_PATH) -> dict[str, Any]:
    """Compare linear LS and raw-pilot CP-bounded DFT estimates on selected failures."""
    confirmation_bytes, confirmation = _validate_confirmation(confirmation_path)
    configuration = confirmation["configuration"]
    nr = configuration["nr_configuration"]
    if int(nr["cyclic_prefix_samples"]) != CYCLIC_PREFIX_SAMPLES:
        raise ValueError("diagnostic tap span must match the frozen confirmation CP")
    if configuration["channel_distribution"]["echo_doppler_hz_conditions"]["static"] != 0.0:
        raise ValueError("diagnostic requires the frozen static reference condition")
    failures = [
        item
        for item in confirmation["confirmatory_channel_draw_records"]
        if not all(item["configuration_results"]["0"]["crc_pass_by_condition"]["static"])
    ]
    if not failures:
        raise ValueError("frozen confirmation contains no static-reference failures")

    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(
            n_size_bwp=int(nr["n_size_bwp_rb"]),
            subcarrier_spacing_khz=int(nr["subcarrier_spacing_hz"] // 1000),
            dmrs_additional_position=0,
            interpolation_type="lin",
        ),
        device="cpu",
    )
    estimator = FiniteDelayDFTChannelEstimator(receiver, max_delay_samples=CYCLIC_PREFIX_SAMPLES)
    decoder = make_finite_delay_dft_decoder(receiver, max_delay_samples=CYCLIC_PREFIX_SAMPLES)
    noise_variance = float(configuration["waveform_awgn_complex_variance"])
    sample_rate_hz = float(nr["sample_rate_hz"])
    fft_size = int(nr["fft_size"])
    frequency_bins = torch.arange(fft_size, dtype=torch.float32) - fft_size // 2
    rows: list[dict[str, Any]] = []

    for item in failures:
        channel_seed = int(item["channel_seed"])
        delay = int(item["echo_delay_samples"])
        echo_real, echo_imag = item["echo_coefficient_real_imag"]
        echo = complex(float(echo_real), float(echo_imag))
        tx_grid, _ = receiver.transmit(8, seed=channel_seed + 1_000_000)
        waveform = receiver.modulate_slot_grid(tx_grid)
        generator = torch.Generator(device="cpu").manual_seed(int(item["noise_seed"]))
        noise = torch.complex(
            torch.randn(waveform.shape, generator=generator),
            torch.randn(waveform.shape, generator=generator),
        ) * math.sqrt(noise_variance / 2.0)
        channel_waveform = apply_time_varying_multipath(
            waveform,
            path_coefficients=(0.85 + 0.0j, echo),
            path_delays_samples=(0, delay),
            path_dopplers_hz=(0.0, 0.0),
            sampling_frequency_hz=sample_rate_hz,
        )
        received_grid = receiver.demodulate_slot_waveform(channel_waveform + noise)
        linear_estimate = receiver.estimate(received_grid, torch.tensor(noise_variance)).h_hat
        dft_estimate, dft_error_variance = estimator(received_grid, torch.tensor(noise_variance))
        response = 0.85 + echo * torch.exp(-2j * math.pi * frequency_bins * delay / fft_size)
        truth = response.reshape(1, 1, 1, 1, 1, 1, -1).expand_as(dft_estimate)
        truth_power = truth.abs().square().mean()
        linear_nmse = float((linear_estimate - truth).abs().square().mean() / truth_power)
        dft_nmse = float((dft_estimate - truth).abs().square().mean() / truth_power)
        _, linear_crc = receiver.decode_pusch(received_grid, torch.tensor(noise_variance))
        _, dft_crc = decoder(received_grid, torch.tensor(noise_variance))
        retained = item["configuration_results"]["0"]["crc_pass_by_condition"]["static"]
        if [bool(value) for value in linear_crc.reshape(-1).tolist()] != retained:
            raise ValueError(
                f"linear estimator replay differs from frozen result for {channel_seed}"
            )
        rows.append(
            {
                "channel_seed": channel_seed,
                "echo_delay_samples": delay,
                "echo_amplitude": abs(echo),
                "exceeds_cyclic_prefix": delay > CYCLIC_PREFIX_SAMPLES,
                "linear_crc_passes": int(linear_crc.sum().item()),
                "cp_bounded_dft_crc_passes": int(dft_crc.sum().item()),
                "linear_noiseless_channel_nmse": linear_nmse,
                "cp_bounded_dft_noiseless_channel_nmse": dft_nmse,
                "mean_propagated_error_variance": float(dft_error_variance.mean().item()),
            }
        )

    strata: dict[str, dict[str, Any]] = {}
    for label, selected_rows in (
        ("within_cyclic_prefix", [row for row in rows if not row["exceeds_cyclic_prefix"]]),
        ("exceeds_cyclic_prefix", [row for row in rows if row["exceeds_cyclic_prefix"]]),
    ):
        if not selected_rows:
            continue
        blocks = 8 * len(selected_rows)
        strata[label] = {
            "selected_channels": len(selected_rows),
            "blocks": blocks,
            "linear_crc_passes": sum(row["linear_crc_passes"] for row in selected_rows),
            "cp_bounded_dft_crc_passes": sum(
                row["cp_bounded_dft_crc_passes"] for row in selected_rows
            ),
            "mean_linear_channel_nmse": sum(
                row["linear_noiseless_channel_nmse"] for row in selected_rows
            )
            / len(selected_rows),
            "mean_cp_bounded_dft_channel_nmse": sum(
                row["cp_bounded_dft_noiseless_channel_nmse"] for row in selected_rows
            )
            / len(selected_rows),
        }

    report = {
        "schema": "umcd.g5_cp_bounded_dft_selected_failure_diagnostic.v1",
        "status": "POST_HOC_DIAGNOSTIC_NOT_QUALIFICATION",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "confirmation_artifact": confirmation_path.as_posix(),
        "confirmation_artifact_sha256": _sha256(confirmation_bytes),
        "selection_rule": (
            "All frozen channel draws with any position-0 static CRC failure; selection is "
            "outcome-dependent and occurred before this estimator replay."
        ),
        "configuration": {
            "cyclic_prefix_samples": CYCLIC_PREFIX_SAMPLES,
            "fitted_delay_samples_inclusive": [0, CYCLIC_PREFIX_SAMPLES],
            "fit_input": "raw DM-RS observations before Sionna adjacent-frequency CDM averaging",
            "error_variance": "LS pilot variance propagated through the pseudoinverse frequency basis",
        },
        "independent_channel_count": len(rows),
        "blocks_per_channel": 8,
        "observations": rows,
        "posthoc_descriptive_strata": strata,
        "source_sha256": {
            path: _sha256((Path(__file__).resolve().parents[1] / path).read_bytes())
            for path in SOURCE_PATHS
        },
        "interpretation": (
            "The selected failures are not a population sample. The CP-bounded DFT estimate "
            "is an experimental receiver candidate only; these post-hoc outcomes cannot select, "
            "qualify, or confirm it. The delay-8 cases exceed the fitted CP support and their "
            "oracle comparison omits ISI. Any candidate qualification requires a frozen protocol "
            "and disjoint channel seeds."
        ),
    }
    report["result_sha256"] = _canonical_sha256(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation", type=Path, default=CONFIRMATION_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing diagnostic: {args.output}")
    report = diagnose_selected_failures(args.confirmation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "channels": report["independent_channel_count"],
                "result_sha256": report["result_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
