"""Run the frozen TDL-C DM-RS density confirmation on its disjoint seed set."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from experiments.g5_independent_channel_receiver import (
    clopper_pearson_upper_bound,
    cluster_bootstrap_interval,
)
from experiments.link_curves import run_tdl_pusch_time_bler_curve

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("docs/research/results/g5-tdl-c-dmrs-density-confirm-20261024.json")
PROTOCOL = Path("docs/research/G5_TDL_C_DMRS_DENSITY_CONFIRMATORY_PROTOCOL_2026-10-24.md")
CHANNEL_SEEDS = tuple(range(933_001, 933_065))
FRAME_COUNT = len(CHANNEL_SEEDS)
SNR_DB = 18.0
PROFILE = "C"
DELAY_SPREAD_S = 100e-9
CARRIER_FREQUENCY_HZ = 3.5e9
SPEED_MPS = 120.0 / 3.6
RESOURCE_BLOCKS = 12
SCS_KHZ = 30
FAMILYWISE_ALPHA = 0.01
FAMILY_CLAIM_COUNT = 3
PER_CLAIM_ALPHA = FAMILYWISE_ALPHA / FAMILY_CLAIM_COUNT
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 933_099
MAXIMUM_STATIC_BLER = 0.05
MAXIMUM_STATIC_CHANNEL_FAILURE_UPPER = 0.10
MINIMUM_MOBILE_BLER_REDUCTION = 0.10
SOURCE_FILES = (
    "experiments/qualify_g5_tdl_c_dmrs_density.py",
    "experiments/g5_independent_channel_receiver.py",
    "experiments/link_curves.py",
    "integrations/sionna/nr_pusch.py",
    "integrations/sionna/tdl_pusch.py",
    "receivers/nr/pusch.py",
    "evaluation/statistics.py",
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )


def _git_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _installed_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not-installed"


def _source_hashes(
    *, protocol_path: Path = PROTOCOL, extra_source_paths: tuple[str, ...] = ()
) -> dict[str, str]:
    return {
        path: _sha256((ROOT / path).read_bytes())
        for path in (*SOURCE_FILES, *extra_source_paths, protocol_path.as_posix())
    }


def _point(
    run: dict[str, Any], condition: str, *, expected_count: int = FRAME_COUNT
) -> dict[str, Any]:
    points = run.get("points")
    if not isinstance(points, list) or len(points) != 1:
        raise ValueError(f"{condition} must contain exactly one SNR point")
    point = points[0]
    if point.get("snr_db") != SNR_DB:
        raise ValueError(f"{condition} returned an unexpected SNR")
    errors = point.get("frame_error_mask")
    if (
        not isinstance(errors, list)
        or len(errors) != expected_count
        or not all(type(value) is bool for value in errors)
    ):
        raise ValueError(f"{condition} lacks the expected per-channel CRC outcomes")
    return point


def _checks(
    run: dict[str, Any], condition: str, *, expected_count: int = FRAME_COUNT
) -> list[dict[str, Any]]:
    configuration = run.get("configuration")
    checks = (
        configuration.get("matched_realization_checks") if isinstance(configuration, dict) else None
    )
    if not isinstance(checks, list) or len(checks) != expected_count:
        raise ValueError(f"{condition} lacks per-channel realization checks")
    return checks


def _validate_run_set(
    runs: dict[str, dict[str, Any]],
    points: dict[str, dict[str, Any]],
    *,
    profile: str = PROFILE,
    channel_seeds: tuple[int, ...] = CHANNEL_SEEDS,
) -> dict[str, list[dict[str, Any]]]:
    checks = {
        name: _checks(run, name, expected_count=len(channel_seeds)) for name, run in runs.items()
    }
    expected_seeds = list(channel_seeds)
    for name, records in checks.items():
        if [item.get("seed") for item in records] != expected_seeds:
            raise ValueError(f"{name} channel seeds differ from the frozen sequence")
        configuration = runs[name].get("configuration", {})
        if configuration.get("allow_cp_overrun") is not False:
            raise ValueError(f"{name} did not disable CP overrun")
        if configuration.get("cp_covers_channel_memory") is not True:
            raise ValueError(f"{name} channel memory is not covered by the CP")
        if configuration.get("profile") != f"TDL-{profile}":
            raise ValueError(f"{name} returned an unexpected channel profile")
        if configuration.get("dmrs_additional_position") not in {0, 3}:
            raise ValueError(f"{name} returned an unexpected DM-RS position")

    for low, high in zip(checks["mobile_position_0"], checks["mobile_position_3"], strict=True):
        if low.get("channel_truth_sha256") != high.get("channel_truth_sha256"):
            raise ValueError("mobile DM-RS conditions do not share channel truth")
        if low.get("seed") != high.get("seed"):
            raise ValueError("mobile DM-RS conditions do not share transmitter/channel seed")
        for key in ("noise_real_seed", "noise_imag_seed"):
            if low.get(key) != high.get(key):
                raise ValueError(f"mobile DM-RS conditions do not share {key}")

    for static, mobile in zip(
        checks["static_position_0"], checks["mobile_position_0"], strict=True
    ):
        if static.get("transmit_bits_sha256") != mobile.get("transmit_bits_sha256"):
            raise ValueError("static and mobile position-0 conditions do not share payload bits")
        for key in ("noise_real_seed", "noise_imag_seed"):
            if static.get(key) != mobile.get(key):
                raise ValueError(f"static and mobile position-0 conditions do not share {key}")

    for name, point in points.items():
        if len(point["frame_error_mask"]) != len(checks[name]):
            raise ValueError(f"{name} CRC outcomes do not align with channel checks")
    return checks


def _goodput_by_channel(point: dict[str, Any]) -> list[float]:
    errors = point["frame_error_mask"]
    tbs = point.get("mean_transmitted_tb_bits_per_codeword")
    if isinstance(tbs, bool) or not isinstance(tbs, (int, float)) or tbs <= 0:
        raise ValueError("run lacks a positive transport-block size")
    if not all(type(error) is bool for error in errors):
        raise ValueError("CRC outcomes must be booleans")
    return [0.0 if error else float(tbs) for error in errors]


def run_confirmation(
    *,
    profile: str = PROFILE,
    channel_seeds: tuple[int, ...] = CHANNEL_SEEDS,
    per_claim_alpha: float = PER_CLAIM_ALPHA,
    family_claim_count: int = FAMILY_CLAIM_COUNT,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    protocol_path: Path = PROTOCOL,
    extra_source_paths: tuple[str, ...] = (),
    channel_estimator_kind: str = "sionna_ls",
    estimator_min_delay_samples: int | None = None,
    estimator_max_delay_samples: int | None = None,
) -> dict[str, Any]:
    """Execute the frozen CPU conditions for one named TDL profile."""
    if profile not in {"A", "B", "C", "D", "E"}:
        raise ValueError("profile must be one of TDL-A through TDL-E")
    if (
        not channel_seeds
        or any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            for seed in channel_seeds
        )
        or len(set(channel_seeds)) != len(channel_seeds)
        or tuple(sorted(channel_seeds)) != channel_seeds
        or channel_seeds != tuple(range(channel_seeds[0], channel_seeds[0] + len(channel_seeds)))
    ):
        raise ValueError("channel_seeds must be nonempty, unique, ascending, and consecutive")
    if not 0.0 < per_claim_alpha < FAMILYWISE_ALPHA:
        raise ValueError("per_claim_alpha must be positive and below family-wise alpha")
    if isinstance(family_claim_count, bool) or family_claim_count < 3:
        raise ValueError("family_claim_count must include the three per-profile claims")
    if per_claim_alpha != FAMILYWISE_ALPHA / family_claim_count:
        raise ValueError("per_claim_alpha must equal family-wise alpha divided by claim count")
    if isinstance(bootstrap_seed, bool) or bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be a nonnegative integer")
    frame_count = len(channel_seeds)
    protocol_bytes = (ROOT / protocol_path).read_bytes()
    condition_specs = {
        "static_position_0": {"speed_mps": 0.0, "dmrs_additional_position": 0},
        "mobile_position_0": {"speed_mps": SPEED_MPS, "dmrs_additional_position": 0},
        "mobile_position_3": {"speed_mps": SPEED_MPS, "dmrs_additional_position": 3},
    }
    runs: dict[str, dict[str, Any]] = {}
    for name, spec in condition_specs.items():
        runs[name] = run_tdl_pusch_time_bler_curve(
            snr_db=(SNR_DB,),
            frames_per_snr=frame_count,
            batch_size=1,
            seed=channel_seeds[0],
            profile=profile,
            delay_spread_s=DELAY_SPREAD_S,
            carrier_frequency_hz=CARRIER_FREQUENCY_HZ,
            speed_mps=spec["speed_mps"],
            n_size_bwp=RESOURCE_BLOCKS,
            dmrs_additional_position=int(spec["dmrs_additional_position"]),
            subcarrier_spacing_khz=SCS_KHZ,
            channel_estimator_kind=channel_estimator_kind,
            estimator_min_delay_samples=estimator_min_delay_samples,
            estimator_max_delay_samples=estimator_max_delay_samples,
            device="cpu",
        )
    expected_estimator = {
        "kind": channel_estimator_kind,
        "min_delay_samples": estimator_min_delay_samples,
        "max_delay_samples": estimator_max_delay_samples,
    }
    for name, run in runs.items():
        if run["configuration"].get("channel_estimator") != expected_estimator:
            raise ValueError(f"{name} did not use the preregistered channel estimator")
    points = {name: _point(run, name, expected_count=frame_count) for name, run in runs.items()}
    checks = _validate_run_set(runs, points, profile=profile, channel_seeds=channel_seeds)

    static_errors = points["static_position_0"]["frame_error_mask"]
    static_failures = sum(static_errors)
    static_bler = static_failures / frame_count
    static_failure_upper = clopper_pearson_upper_bound(
        static_failures, frame_count, alpha=per_claim_alpha
    )
    mobile_zero = points["mobile_position_0"]["frame_error_mask"]
    mobile_three = points["mobile_position_3"]["frame_error_mask"]
    paired_bler_reduction = [
        float(error_zero) - float(error_three)
        for error_zero, error_three in zip(mobile_zero, mobile_three, strict=True)
    ]
    bler_reduction_mean = sum(paired_bler_reduction) / frame_count
    bler_interval = cluster_bootstrap_interval(
        paired_bler_reduction,
        seed=bootstrap_seed,
        alpha=per_claim_alpha,
        replicates=BOOTSTRAP_REPLICATES,
    )
    goodput_zero = _goodput_by_channel(points["mobile_position_0"])
    goodput_three = _goodput_by_channel(points["mobile_position_3"])
    paired_goodput_increase = [
        dense - sparse for sparse, dense in zip(goodput_zero, goodput_three, strict=True)
    ]
    goodput_mean = sum(paired_goodput_increase) / frame_count
    goodput_interval = cluster_bootstrap_interval(
        paired_goodput_increase,
        seed=bootstrap_seed + 1,
        alpha=per_claim_alpha,
        replicates=BOOTSTRAP_REPLICATES,
    )

    acceptance: dict[str, Any] = {
        "familywise_alpha": FAMILYWISE_ALPHA,
        "family_claim_count": family_claim_count,
        "per_claim_alpha": per_claim_alpha,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "paired_bler_bootstrap_seed": bootstrap_seed,
        "paired_goodput_bootstrap_seed": bootstrap_seed + 1,
        "bootstrap_resampling_unit": "one independent TDL channel seed; one transport block per condition",
        "static_reference": {
            "position_0_bler": static_bler,
            "crc_failures": static_failures,
            "independent_channel_seeds": frame_count,
            "family_adjusted_one_sided_exact_failure_prevalence_upper_bound": static_failure_upper,
            "maximum_bler": MAXIMUM_STATIC_BLER,
            "maximum_failure_prevalence_upper_bound": MAXIMUM_STATIC_CHANNEL_FAILURE_UPPER,
            "pass": static_bler <= MAXIMUM_STATIC_BLER
            and static_failure_upper <= MAXIMUM_STATIC_CHANNEL_FAILURE_UPPER,
        },
        "mobile_bler_reduction_position_0_minus_3": {
            "equal_channel_mean": bler_reduction_mean,
            "minimum_required_mean_reduction": MINIMUM_MOBILE_BLER_REDUCTION,
            "family_adjusted_paired_cluster_bootstrap_interval": list(bler_interval),
            "pass": bler_reduction_mean >= MINIMUM_MOBILE_BLER_REDUCTION and bler_interval[0] > 0,
        },
        "mobile_goodput_increase_position_3_minus_0_bits_per_slot": {
            "equal_channel_mean": goodput_mean,
            "family_adjusted_paired_cluster_bootstrap_interval": list(goodput_interval),
            "pass": goodput_interval[0] > 0,
        },
    }
    passed = all(
        acceptance[key]["pass"]
        for key in (
            "static_reference",
            "mobile_bler_reduction_position_0_minus_3",
            "mobile_goodput_increase_position_3_minus_0_bits_per_slot",
        )
    )
    source_hashes = _source_hashes(
        protocol_path=protocol_path,
        extra_source_paths=extra_source_paths,
    )
    condition_records: dict[str, Any] = {}
    for name, run in runs.items():
        condition_records[name] = {
            "configuration": run["configuration"],
            "runtime": run.get("runtime", {}),
            "point": points[name],
            "matched_realization_checks": checks[name],
            "raw_run_sha256": _canonical_sha256(run),
        }
    report: dict[str, Any] = {
        "schema": "umcd.g5_tdl_c_dmrs_density_confirmatory.v1"
        if profile == "C" and protocol_path == PROTOCOL
        else "umcd.g5_tdl_dmrs_density_profile_confirmatory.v1",
        "status": f"PASS_BOUNDED_TDL_{profile}_FIXTURE" if passed else "NOT_MET",
        "g5_09_closed": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": _git_revision(),
        "command": __import__("sys").orig_argv,
        "runtime": {
            "python": platform.python_version(),
            "torch": _installed_version("torch"),
            "sionna": _installed_version("sionna"),
            "device": "cpu",
        },
        "protocol_path": protocol_path.as_posix(),
        "protocol_sha256": _sha256(protocol_bytes),
        "source_sha256": source_hashes,
        "source_inventory_sha256": _canonical_sha256(source_hashes),
        "configuration": {
            "profile": f"TDL-{profile}",
            "channel_estimator": {
                "kind": channel_estimator_kind,
                "min_delay_samples": estimator_min_delay_samples,
                "max_delay_samples": estimator_max_delay_samples,
            },
            "delay_spread_s": DELAY_SPREAD_S,
            "carrier_frequency_hz": CARRIER_FREQUENCY_HZ,
            "speed_mps_mobile": SPEED_MPS,
            "resource_blocks": RESOURCE_BLOCKS,
            "subcarrier_spacing_khz": SCS_KHZ,
            "configured_receive_grid_snr_db": SNR_DB,
            "independent_channel_seeds": list(channel_seeds),
            "frames_per_condition": frame_count,
            "batch_size": 1,
            "device": "cpu",
            "conditions": condition_specs,
            "cp_coverage_by_condition": {
                name: {
                    "cp_covers_channel_memory": run["configuration"]["cp_covers_channel_memory"],
                    "channel_memory_samples": run["configuration"]["channel_memory_samples"],
                    "cyclic_prefix_min_samples": run["configuration"]["cyclic_prefix_min_samples"],
                }
                for name, run in runs.items()
            },
            "mobile_positions_share_channel_payload_and_noise_bindings": True,
            "static_mobile_channel_truth_byte_equality_required": False,
        },
        "acceptance": acceptance,
        "interpretation": (
            "A pass supports only this CPU TDL-C 100-ns, 3.5-GHz, 120-km/h, "
            "18-dB, 12-RB/30-kHz PUSCH configuration. It is not field-channel "
            "validation, general G5.09 closure, G3 qualification, or production RT authorization."
        ),
        "conditions": condition_records,
        "paired_channel_outcomes": [
            {
                "seed": seed,
                "static_position_0_crc_error": static_errors[index],
                "mobile_position_0_crc_error": mobile_zero[index],
                "mobile_position_3_crc_error": mobile_three[index],
                "mobile_position_0_goodput_bits_per_slot": goodput_zero[index],
                "mobile_position_3_goodput_bits_per_slot": goodput_three[index],
                "paired_bler_reduction": paired_bler_reduction[index],
                "paired_goodput_increase_bits_per_slot": paired_goodput_increase[index],
                "mobile_position_0_channel_truth_sha256": checks["mobile_position_0"][index][
                    "channel_truth_sha256"
                ],
                "mobile_position_3_channel_truth_sha256": checks["mobile_position_3"][index][
                    "channel_truth_sha256"
                ],
                "mobile_position_0_transmit_bits_sha256": checks["mobile_position_0"][index][
                    "transmit_bits_sha256"
                ],
                "mobile_position_3_transmit_bits_sha256": checks["mobile_position_3"][index][
                    "transmit_bits_sha256"
                ],
                "noise_real_seed": checks["mobile_position_0"][index]["noise_real_seed"],
                "noise_imag_seed": checks["mobile_position_0"][index]["noise_imag_seed"],
            }
            for index, seed in enumerate(channel_seeds)
        ],
    }
    report["configuration_sha256"] = _canonical_sha256(report["configuration"])
    report["result_sha256"] = _canonical_sha256(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing confirmation: {output}")
    report = run_confirmation()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "status": report["status"],
                "acceptance": report["acceptance"],
                "result_sha256": report["result_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
