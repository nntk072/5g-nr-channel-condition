"""Monte Carlo implementation check for the pinned Sionna CDL A-E profiles."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
from typing import Any

import torch
from artifacts.provenance import atomic_json, package_source_sha256, sha256_file, versions
from integrations.sionna.random import call_with_sionna_seed


def _expected_profile(profile: str, delay_spread_s: float, device: str) -> tuple[Any, Any, str]:
    from sionna.phy.channel.tr38901 import models

    resource = models.parameter_file(f"CDL-{profile}.json", "19.2")
    parameters = models.load_json(resource)
    delays = torch.as_tensor(parameters["delays"], dtype=torch.float32, device=device)
    delays = delays * delay_spread_s
    powers = torch.pow(
        torch.tensor(10.0, dtype=torch.float32, device=device),
        torch.as_tensor(parameters["powers"], dtype=torch.float32, device=device) / 10.0,
    )
    powers = powers / powers.sum()
    if bool(parameters["los"]):
        specular_power = powers[0]
        nlos_powers = powers[1:]
        k_factor = specular_power / nlos_powers.sum()
        delays = delays[1:]
        nlos_powers = nlos_powers / nlos_powers.sum()
        powers = torch.cat(
            (
                (k_factor + nlos_powers[:1]) / (k_factor + 1.0),
                nlos_powers[1:] / (k_factor + 1.0),
            )
        )
    order = torch.argsort(delays)
    resource_hash = hashlib.sha256(resource.read_bytes()).hexdigest()
    return delays[order], powers[order], resource_hash


def validate_profiles(
    *,
    output: Path,
    batch_size: int,
    seed: int,
    delay_spread_ns: float,
    device: str,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite result: {output}")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 32:
        raise ValueError("batch_size must be at least 32 for a Monte Carlo check")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be nonnegative")
    if not math.isfinite(delay_spread_ns) or delay_spread_ns <= 0:
        raise ValueError("delay_spread_ns must be finite and positive")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    if device != "cpu" and not device.startswith("cuda"):
        raise ValueError("device must be cpu or a CUDA device such as cuda:0")

    from sionna.phy.channel.tr38901 import CDL, PanelArray

    delay_spread_s = delay_spread_ns * 1e-9
    carrier_hz = 3.5e9
    array = PanelArray(
        num_rows_per_panel=1,
        num_cols_per_panel=1,
        polarization="single",
        polarization_type="V",
        antenna_pattern="omni",
        carrier_frequency=carrier_hz,
        device=device,
    )
    profiles: list[dict[str, Any]] = []
    resource_hashes: dict[str, str] = {}
    with torch.no_grad():
        for profile_index, profile in enumerate("ABCDE"):
            channel = CDL(
                model=profile,
                delay_spread=delay_spread_s,
                carrier_frequency=carrier_hz,
                ut_array=array,
                bs_array=array,
                direction="downlink",
                min_speed=0.0,
                max_speed=0.0,
                device=device,
                spec_version="19.2",
            )
            expected_delays, expected_powers, resource_hash = _expected_profile(
                profile, delay_spread_s, device
            )
            coefficients, delays = call_with_sionna_seed(
                device,
                seed + profile_index,
                lambda: channel(
                    batch_size=batch_size, num_time_steps=1, sampling_frequency=30_000.0
                ),
            )
            empirical_powers = torch.mean(torch.abs(coefficients[:, 0, 0, 0, 0, :, 0]) ** 2, dim=0)
            if not torch.allclose(delays[0, 0, 0], expected_delays, rtol=0.0, atol=1e-14):
                raise AssertionError(
                    f"{profile}: generated delays differ from sorted profile table"
                )
            if not torch.isclose(
                empirical_powers.sum(),
                torch.tensor(1.0, dtype=empirical_powers.dtype, device=device),
                rtol=0.02,
                atol=0.01,
            ):
                raise AssertionError(f"{profile}: empirical total path power is not normalized")
            torch.testing.assert_close(empirical_powers, expected_powers, rtol=0.1, atol=0.005)
            absolute_errors = torch.abs(empirical_powers - expected_powers)
            reliable = expected_powers >= 0.02
            relative_errors = absolute_errors[reliable] / expected_powers[reliable]
            resource_hashes[profile] = resource_hash
            profiles.append(
                {
                    "profile": profile,
                    "los_profile": bool(channel._has_los),
                    "k_factor": float(channel._k_factor.cpu()) if channel._has_los else None,
                    "num_profile_clusters": int(channel._num_clusters),
                    "num_generated_paths": int(coefficients.shape[-2]),
                    "empirical_total_path_power": float(empirical_powers.sum().cpu()),
                    "max_absolute_tap_power_error": float(absolute_errors.max().cpu()),
                    "max_relative_error_for_expected_power_at_least_0_02": (
                        float(relative_errors.max().cpu()) if relative_errors.numel() else None
                    ),
                    "taps": [
                        {
                            "delay_s": float(delay.cpu()),
                            "expected_power": float(expected.cpu()),
                            "empirical_power": float(empirical.cpu()),
                        }
                        for delay, expected, empirical in zip(
                            expected_delays, expected_powers, empirical_powers, strict=True
                        )
                    ],
                }
            )

    dependencies = versions()
    hardware: dict[str, Any] = {"device": device, "cuda_available": torch.cuda.is_available()}
    if device.startswith("cuda"):
        hardware["cuda_device_name"] = torch.cuda.get_device_name(torch.device(device))
    report = {
        "schema": "loslab.cdl_profile_validation.v1",
        "evidence_grade": "PINNED_SIMULATOR_IMPLEMENTATION_CHECK_NOT_FULL_3GPP_CERTIFICATION",
        "source_revision": dependencies.get("git_sha"),
        "package_source_sha256": package_source_sha256(),
        "runner_sha256": sha256_file(__file__),
        "versions": dependencies,
        "hardware": hardware,
        "configuration": {
            "spec_version": "19.2",
            "profiles": list("ABCDE"),
            "batch_size_per_profile": batch_size,
            "seed_base": seed,
            "delay_spread_ns": delay_spread_ns,
            "carrier_hz": carrier_hz,
            "speed_mps": 0.0,
            "rx_tx_antennas": "1x1 single-polarized omni",
        },
        "profile_resource_sha256": resource_hashes,
        "profile_results": profiles,
        "limitations": [
            "Expected PDP is computed from the same pinned Sionna versioned CDL resource tables.",
            "This checks table loading, delay ordering/scaling, LOS K-factor split, normalization, and Monte Carlo generation only.",
            "One carrier, delay spread, speed, antenna, seed per profile, and static snapshot do not establish full TR 38.901 conformance.",
            "The profile labels are not geometric LOS/NLOS truth or measured propagation evidence.",
        ],
    }
    atomic_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=31900)
    parser.add_argument("--delay-spread-ns", type=float, default=100.0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    try:
        report = validate_profiles(
            output=args.output,
            batch_size=args.batch_size,
            seed=args.seed,
            delay_spread_ns=args.delay_spread_ns,
            device=args.device,
        )
    except (AssertionError, FileExistsError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(f"Validated CDL A-E against pinned profile tables: {args.output}")
    for profile in report["profile_results"]:
        print(
            f"{profile['profile']}: paths={profile['num_generated_paths']} "
            f"max_abs_power_error={profile['max_absolute_tap_power_error']:.4g}"
        )


if __name__ == "__main__":
    main()
