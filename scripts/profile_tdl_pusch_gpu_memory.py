#!/usr/bin/env python3
"""Measure sample-domain TDL/PUSCH CUDA runtime and peak tensor memory."""

from __future__ import annotations

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from artifacts.provenance import atomic_json, package_source_sha256, versions  # noqa: E402


def measure_cuda_phase(
    torch: Any, name: str, operation: Callable[[], Any]
) -> tuple[Any, dict[str, Any]]:
    """Measure synchronized wall time and CUDA allocator peaks for one phase."""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    baseline_allocated = int(torch.cuda.memory_allocated())
    baseline_reserved = int(torch.cuda.memory_reserved())
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = operation()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    return result, {
        "phase": name,
        "elapsed_seconds": elapsed,
        "baseline_allocated_bytes": baseline_allocated,
        "peak_allocated_bytes": peak_allocated,
        "peak_allocated_increment_bytes": max(0, peak_allocated - baseline_allocated),
        "baseline_reserved_bytes": baseline_reserved,
        "peak_reserved_bytes": peak_reserved,
        "peak_reserved_increment_bytes": max(0, peak_reserved - baseline_reserved),
    }


def _decode(link: Any, received: Any, noise_variance: float, first_rx_only: bool) -> Any:
    if first_rx_only:
        received = received[:, :, :1, :]
    return link.decode(received, noise_variance)


def _generate_noise(
    generator: Callable[..., Any],
    shape: tuple[int, ...],
    seed: int,
    noise_variance: float,
) -> Any:
    return generator(
        shape,
        device="cuda:0",
        noise_variance=noise_variance,
        real_seed=seed + 2,
        imag_seed=seed + 3,
    )


def _add_noise(clean: Any, noise: Any) -> Any:
    return clean + noise


def profile_tdl_pusch(
    *,
    batch_sizes: tuple[int, ...],
    seed: int,
    profile: str,
    delay_spread_s: float,
    carrier_frequency_hz: float,
    speed_mps: float,
    n_size_bwp: int,
    subcarrier_spacing_khz: int,
    dmrs_additional_position: int,
) -> dict[str, Any]:
    """Profile a representative two-RX CUDA waveform link at each batch size."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install the pinned Sionna/CUDA environment") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the PUSCH memory profile")
    if not batch_sizes or any(
        isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in batch_sizes
    ):
        raise ValueError("batch_sizes must contain positive integers")
    if len(set(batch_sizes)) != len(batch_sizes):
        raise ValueError("batch_sizes must be unique")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    from integrations.sionna.nr_pusch import SionnaPUSCHLSReceiver
    from integrations.sionna.tdl_pusch import SionnaTDLTimePUSCHLink, seeded_complex_awgn
    from receivers.nr import NRPUSCHConfig

    receiver_config = NRPUSCHConfig(
        n_size_bwp=n_size_bwp,
        subcarrier_spacing_khz=subcarrier_spacing_khz,
        dmrs_additional_position=dmrs_additional_position,
    )
    receiver, receiver_memory = measure_cuda_phase(
        torch,
        "receiver_construction",
        lambda: SionnaPUSCHLSReceiver(receiver_config, device="cuda:0"),
    )
    link, link_memory = measure_cuda_phase(
        torch,
        "tdl_link_construction",
        lambda: SionnaTDLTimePUSCHLink(
            receiver,
            model=profile,
            delay_spread_s=delay_spread_s,
            carrier_frequency_hz=carrier_frequency_hz,
            speed_mps=speed_mps,
            num_rx_ant=2,
        ),
    )

    runs: list[dict[str, Any]] = []
    for batch_index, batch_size in enumerate(batch_sizes):
        batch_seed = seed + batch_index * 1000
        (waveform, bits, clean, channel_truth), channel_stats = measure_cuda_phase(
            torch,
            "waveform_and_tdl_channel",
            lambda: link.transmit_channel(batch_size, seed=batch_seed),
        )
        noise_variance = 10.0 ** (-10.0 / 10.0)
        noise, noise_stats = measure_cuda_phase(
            torch,
            "two_rx_awgn",
            partial(
                _generate_noise,
                seeded_complex_awgn,
                tuple(clean.shape),
                batch_seed,
                noise_variance,
            ),
        )
        received, addition_stats = measure_cuda_phase(
            torch,
            "add_awgn_to_two_rx",
            partial(_add_noise, clean, noise),
        )
        (decoded_one, crc_one), decode_one_stats = measure_cuda_phase(
            torch,
            "decode_one_rx",
            partial(_decode, link, received, noise_variance, True),
        )
        (decoded_two, crc_two), decode_two_stats = measure_cuda_phase(
            torch,
            "decode_two_rx",
            partial(_decode, link, received, noise_variance, False),
        )
        runs.append(
            {
                "batch_size_codewords": batch_size,
                "seed": batch_seed,
                "waveform_shape": list(waveform.shape),
                "channel_truth_shape": list(channel_truth.shape),
                "mean_received_power_per_rx": clean.abs()
                .square()
                .mean(dim=(0, 1, 3))
                .detach()
                .cpu()
                .tolist(),
                "crc_success_one_rx": int(crc_one.sum().item()),
                "crc_success_two_rx": int(crc_two.sum().item()),
                "decoded_bit_count_one_rx": int(decoded_one.numel()),
                "decoded_bit_count_two_rx": int(decoded_two.numel()),
                "phases": [
                    channel_stats,
                    noise_stats,
                    addition_stats,
                    decode_one_stats,
                    decode_two_stats,
                ],
            }
        )
        del waveform, bits, clean, channel_truth, noise, received
        del decoded_one, crc_one, decoded_two, crc_two

    device = torch.cuda.get_device_properties("cuda:0")
    return {
        "schema": "loslab.tdl_pusch_cuda_memory_profile.v1",
        "evidence_grade": "ENGINEERING_PROFILE_NOT_LINK_PERFORMANCE_EVIDENCE",
        "configuration": {
            **link.metadata,
            "n_size_bwp_resource_blocks": n_size_bwp,
            "subcarrier_spacing_khz": subcarrier_spacing_khz,
            "dmrs_additional_position": dmrs_additional_position,
            "snr_db": 10.0,
            "batch_sizes_codewords": list(batch_sizes),
            "seed": seed,
            "receive_antennas": 2,
            "rx_diversity_decode_policy": "same generated noisy waveform decoded with RX0 and RX0+RX1",
        },
        "hardware": {
            "device_name": device.name,
            "compute_capability": list(torch.cuda.get_device_capability("cuda:0")),
            "total_memory_bytes": int(device.total_memory),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
        },
        "versions": versions(),
        "package_source_sha256": package_source_sha256(),
        "construction_phases": [receiver_memory, link_memory],
        "runs": runs,
        "measurement_method": (
            "CUDA-synchronized perf_counter wall time; PyTorch peak allocated and reserved "
            "allocator bytes reset around each phase; values are process-local and exclude "
            "driver/display memory not owned by PyTorch."
        ),
        "limitations": [
            "One Sionna version, GPU, numerology, TDL profile, bandwidth, decoder and synchronization setting are profiled.",
            "Memory figures are allocator observations, not total-device utilization or throughput guarantees.",
            "The profile does not qualify scientific link performance or calibrated hardware behavior.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new JSON result path; existing files refused")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 16, 32, 64, 128])
    parser.add_argument("--seed", type=int, default=73000)
    parser.add_argument("--profile", choices=list("ABCDE"), default="A")
    parser.add_argument("--delay-spread-ns", type=float, default=100.0)
    parser.add_argument("--carrier-ghz", type=float, default=3.5)
    parser.add_argument("--speed-mps", type=float, default=3.0)
    parser.add_argument("--n-size-bwp", type=int, default=12)
    parser.add_argument("--subcarrier-spacing-khz", type=int, default=30)
    parser.add_argument("--dmrs-additional-position", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    try:
        result = profile_tdl_pusch(
            batch_sizes=tuple(args.batch_sizes),
            seed=args.seed,
            profile=args.profile,
            delay_spread_s=args.delay_spread_ns * 1e-9,
            carrier_frequency_hz=args.carrier_ghz * 1e9,
            speed_mps=args.speed_mps,
            n_size_bwp=args.n_size_bwp,
            subcarrier_spacing_khz=args.subcarrier_spacing_khz,
            dmrs_additional_position=args.dmrs_additional_position,
        )
    except (TypeError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    atomic_json(args.output, result)
    print(
        json.dumps(
            [
                {
                    "batch_size_codewords": run["batch_size_codewords"],
                    "decode_2rx_peak_allocated_mib": round(
                        next(
                            phase["peak_allocated_bytes"]
                            for phase in run["phases"]
                            if phase["phase"] == "decode_two_rx"
                        )
                        / 2**20,
                        2,
                    ),
                    "channel_seconds": round(run["phases"][0]["elapsed_seconds"], 4),
                    "decode_2rx_seconds": round(run["phases"][4]["elapsed_seconds"], 4),
                }
                for run in result["runs"]
            ],
            indent=2,
        )
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
