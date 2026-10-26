"""Exploratory CPU study of TDL Doppler and PUSCH DM-RS density.

This is a development diagnostic only. It does not modify or satisfy the frozen
G5 confirmatory protocol, establish physical validity, or qualify a receiver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from evaluation.statistics import paired_cluster_binary_error_comparison

from experiments.link_curves import run_tdl_pusch_time_bler_curve

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("docs/research/results/g5-tdl-cpu-dmrs-density-development-20261024.json")
SOURCE_FILES = (
    "experiments/audit_g5_tdl_dmrs_density_development.py",
    "experiments/link_curves.py",
    "integrations/sionna/tdl_pusch.py",
    "integrations/sionna/nr_pusch.py",
    "receivers/nr/pusch.py",
    "evaluation/statistics.py",
)
SNR_DB = 18.0
PROFILE = "C"
DELAY_SPREAD_S = 100e-9
CARRIER_FREQUENCY_HZ = 3.5e9
SPEED_MPS = 120.0 / 3.6
N_SIZE_BWP = 12
SCS_KHZ = 30


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )


def _point(result: dict[str, Any]) -> dict[str, Any]:
    points = result.get("points")
    if not isinstance(points, list) or len(points) != 1:
        raise ValueError("each run must contain exactly one SNR point")
    point = points[0]
    if point.get("snr_db") != SNR_DB:
        raise ValueError("run returned an unexpected SNR")
    errors = point.get("frame_error_mask")
    if (
        not isinstance(errors, list)
        or not errors
        or not all(type(value) is bool for value in errors)
    ):
        raise ValueError("run lacks a valid per-codeword CRC error mask")
    return point


def _checks(result: dict[str, Any]) -> list[dict[str, Any]]:
    checks = result.get("configuration", {}).get("matched_realization_checks")
    if not isinstance(checks, list) or not checks:
        raise ValueError("run lacks matched channel/noise realization records")
    return checks


def run_development_audit(
    *, frames: int = 64, seed: int = 932001, draws: int = 5000
) -> dict[str, Any]:
    """Run the three preregistered exploratory conditions on CPU."""
    if isinstance(frames, bool) or not isinstance(frames, int) or frames < 2:
        raise ValueError("frames must be an integer of at least two")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(draws, bool) or not isinstance(draws, int) or draws <= 0:
        raise ValueError("draws must be a positive integer")

    conditions = {
        "static_position_0": {"speed_mps": 0.0, "dmrs_additional_position": 0},
        "mobile_position_0": {"speed_mps": SPEED_MPS, "dmrs_additional_position": 0},
        "mobile_position_3": {"speed_mps": SPEED_MPS, "dmrs_additional_position": 3},
    }
    runs: dict[str, dict[str, Any]] = {}
    for name, values in conditions.items():
        runs[name] = run_tdl_pusch_time_bler_curve(
            snr_db=(SNR_DB,),
            frames_per_snr=frames,
            batch_size=1,
            seed=seed,
            profile=PROFILE,
            delay_spread_s=DELAY_SPREAD_S,
            carrier_frequency_hz=CARRIER_FREQUENCY_HZ,
            speed_mps=values["speed_mps"],
            n_size_bwp=N_SIZE_BWP,
            dmrs_additional_position=int(values["dmrs_additional_position"]),
            subcarrier_spacing_khz=SCS_KHZ,
            device="cpu",
        )

    points = {name: _point(result) for name, result in runs.items()}
    checks = {name: _checks(result) for name, result in runs.items()}
    expected_seeds = list(range(seed, seed + frames))
    for name, records in checks.items():
        if [item.get("seed") for item in records] != expected_seeds:
            raise ValueError(f"{name} channel seeds do not match the frozen sequence")
    for left, right in (("mobile_position_0", "mobile_position_3"),):
        for first, second in zip(checks[left], checks[right], strict=True):
            if first.get("channel_truth_sha256") != second.get("channel_truth_sha256"):
                raise ValueError("DM-RS positions did not share exact channel truth")
            if first.get("noise_real_seed") != second.get("noise_real_seed") or first.get(
                "noise_imag_seed"
            ) != second.get("noise_imag_seed"):
                raise ValueError("DM-RS positions did not share their noise seeds")

    groups = [f"channel-draw-{value}" for value in expected_seeds]
    comparisons = {
        "mobile_position_0_minus_static_position_0": paired_cluster_binary_error_comparison(
            points["static_position_0"]["frame_error_mask"],
            points["mobile_position_0"]["frame_error_mask"],
            groups,
            seed=seed + 100_000,
            draws=draws,
        ),
        "mobile_position_3_minus_mobile_position_0": paired_cluster_binary_error_comparison(
            points["mobile_position_0"]["frame_error_mask"],
            points["mobile_position_3"]["frame_error_mask"],
            groups,
            seed=seed + 100_001,
            draws=draws,
        ),
    }
    sources = {path: _sha256((ROOT / path).read_bytes()) for path in SOURCE_FILES}
    protocol = {
        "snr_db": SNR_DB,
        "profile": f"TDL-{PROFILE}",
        "delay_spread_s": DELAY_SPREAD_S,
        "carrier_frequency_hz": CARRIER_FREQUENCY_HZ,
        "speed_kmh_mobile": 120.0,
        "resource_blocks": N_SIZE_BWP,
        "subcarrier_spacing_khz": SCS_KHZ,
        "frames_per_condition": frames,
        "batch_size": 1,
        "device": "cpu",
        "channel_seeds": expected_seeds,
        "conditions": conditions,
        "cluster_unit": "paired per-seed channel draw; descriptive simulation inference only",
    }
    return {
        "schema": "umcd.g5_tdl_dmrs_density_development.v1",
        "status": "DEVELOPMENT_ONLY_NOT_QUALIFIED",
        "purpose": "Explore whether additional DM-RS density changes CPU sample-domain TDL-C PUSCH CRC outcomes under mobility, with a static reference.",
        "protocol": protocol,
        "protocol_sha256": _canonical_sha256(protocol),
        "source_sha256": sources,
        "source_inventory_sha256": _canonical_sha256(sources),
        "conditions": {
            name: {
                "configuration": result["configuration"],
                "point": points[name],
                "matched_realization_checks": checks[name],
                "raw_run_sha256": _canonical_sha256(result),
            }
            for name, result in runs.items()
        },
        "paired_comparisons": comparisons,
        "interpretation_limits": [
            "Exploratory development evidence, not the frozen G5 confirmatory test.",
            "Sionna TDL is a statistical simulation channel and is not external physical validation.",
            "The static and mobile conditions are paired by seed index, not by identical channel truth.",
            "The paired bootstrap is conditional on these channel draws and cannot establish cross-site generalization.",
            "A favorable outcome does not close the G5 static-reference failure or authorize production RT generation.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--seed", type=int, default=932001)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing audit: {output}")
    report = run_development_audit(frames=args.frames, seed=args.seed, draws=args.draws)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "status": report["status"],
                "protocol_sha256": report["protocol_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
