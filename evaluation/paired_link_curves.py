"""Matched comparisons for reproducible sample-domain link curves."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from artifacts.provenance import package_source_sha256, sha256_file, versions

from evaluation.statistics import paired_binary_error_comparison


def load_curve(path: str | Path) -> dict[str, Any]:
    """Load a supported, explicitly simulation-only sample-domain curve."""
    path = Path(path)
    result = json.loads(path.read_text(encoding="utf8"))
    if result.get("schema") != "loslab.tdl_pusch_time_bler_curve.v1":
        raise ValueError(f"unsupported sample-domain curve schema in {path}")
    if result.get("evidence_grade") != "SIMULATION_ONLY_UNVALIDATED":
        raise ValueError(f"unexpected evidence grade in {path}")
    return result


def compare_curves(
    control_path: str | Path,
    treatment_path: str | Path,
    *,
    varying_config_field: str = "speed_mps",
) -> dict[str, Any]:
    """Validate a matched pair and compute BLER and decoded-throughput deltas."""
    control_path = Path(control_path)
    treatment_path = Path(treatment_path)
    control = load_curve(control_path)
    treatment = load_curve(treatment_path)
    control_config = control["configuration"]
    treatment_config = treatment["configuration"]
    if varying_config_field not in {
        "speed_mps",
        "dmrs_additional_position",
        "delay_spread_s",
        "profile",
    }:
        raise ValueError(
            "varying_config_field must be speed_mps, dmrs_additional_position, "
            "delay_spread_s, or profile"
        )
    varying = {
        varying_config_field,
        "matched_realization_checks",
        "transmit_waveform_mean_power_by_batch",
    }
    if varying_config_field == "delay_spread_s":
        varying.update(
            {
                "allow_cp_overrun",
                "channel_memory_samples",
                "cp_covers_channel_memory",
                "l_max",
            }
        )
    if varying_config_field == "profile":
        varying.update({"channel_model"})
    for key in set(control_config) | set(treatment_config):
        if key not in varying and control_config.get(key) != treatment_config.get(key):
            raise ValueError(f"curve configuration differs unexpectedly at {key}")
    default_varying_value = 0 if varying_config_field == "dmrs_additional_position" else None
    control_value = control_config.get(varying_config_field, default_varying_value)
    treatment_value = treatment_config.get(varying_config_field, default_varying_value)
    if control_value == treatment_value:
        raise ValueError(f"control and treatment must differ at {varying_config_field}")
    control_checks = control_config["matched_realization_checks"]
    treatment_checks = treatment_config["matched_realization_checks"]
    if len(control_checks) != len(treatment_checks):
        raise ValueError("matched realization batch counts differ")
    same_bits = True
    same_channel = True
    for left, right in zip(control_checks, treatment_checks, strict=True):
        if left["seed"] != right["seed"]:
            raise ValueError("batch seeds differ across conditions")
        if left.get("noise_seed") != right.get("noise_seed"):
            raise ValueError(f"noise seeds differ for matched batch {left['seed']}")
        for component in ("noise_real_seed", "noise_imag_seed"):
            if left.get(component) != right.get(component):
                raise ValueError(f"{component} differs for matched batch {left['seed']}")
        same_bits = same_bits and left["transmit_bits_sha256"] == right["transmit_bits_sha256"]
        same_channel = (
            same_channel and left["channel_truth_sha256"] == right["channel_truth_sha256"]
        )
    if varying_config_field == "speed_mps" and not same_bits:
        raise ValueError("transmitted bits differ for the matched mobility experiment")
    if varying_config_field == "profile":
        if not same_bits:
            raise ValueError("transmitted bits differ for the matched channel-profile experiment")
        if same_channel:
            raise ValueError("channel truth is identical across channel profiles")
    if varying_config_field == "dmrs_additional_position" and not same_channel:
        raise ValueError("channel truth differs for the matched DMRS experiment")
    if varying_config_field == "delay_spread_s":
        if not same_bits:
            raise ValueError("transmitted bits differ for the matched delay-spread experiment")
        if any(
            check.get(component) is None
            for check in control_checks + treatment_checks
            for component in ("noise_real_seed", "noise_imag_seed")
        ):
            raise ValueError("delay-spread curves must record separate real/imaginary noise seeds")
        if any(
            check["noise_real_seed"] == check["noise_imag_seed"]
            for check in control_checks + treatment_checks
        ):
            raise ValueError("real and imaginary AWGN streams must use distinct seeds")
        if same_channel:
            raise ValueError("channel truth is identical across delay-spread conditions")
        if control_config.get("allow_cp_overrun") is not False or not control_config.get(
            "cp_covers_channel_memory"
        ):
            raise ValueError("control curve must have a CP-contained channel")
        if treatment_config.get("allow_cp_overrun") is not True or treatment_config.get(
            "cp_covers_channel_memory"
        ):
            raise ValueError("delay-spread treatment must explicitly record CP overrun")
        if treatment_config["delay_spread_s"] <= control_config["delay_spread_s"]:
            raise ValueError("treatment delay spread must exceed control delay spread")

    control_points = control["points"]
    treatment_points = treatment["points"]
    if len(control_points) != len(treatment_points):
        raise ValueError("SNR point counts differ")
    comparisons: list[dict[str, Any]] = []
    for left, right in zip(control_points, treatment_points, strict=True):
        if left["snr_db"] != right["snr_db"] or left["frames"] != right["frames"]:
            raise ValueError("SNR values or codeword counts differ")
        comparisons.append(
            {
                "snr_db": left["snr_db"],
                **paired_binary_error_comparison(
                    left["frame_error_mask"],
                    right["frame_error_mask"],
                    seed=1,
                    draws=20_000,
                    control_bits_per_success=left["mean_transmitted_tb_bits_per_codeword"]
                    if "mean_transmitted_tb_bits_per_codeword" in left
                    else left["bits"] / left["frames"],
                    treatment_bits_per_success=right["mean_transmitted_tb_bits_per_codeword"]
                    if "mean_transmitted_tb_bits_per_codeword" in right
                    else right["bits"] / right["frames"],
                ),
            }
        )
    ordered = sorted(
        range(len(comparisons)),
        key=lambda index: comparisons[index]["mcnemar_exact_two_sided_p_value"],
    )
    adjusted_max = 0.0
    point_count = len(comparisons)
    for rank, index in enumerate(ordered):
        raw_p = comparisons[index]["mcnemar_exact_two_sided_p_value"]
        adjusted_max = max(adjusted_max, (point_count - rank) * raw_p)
        comparisons[index]["mcnemar_exact_two_sided_p_value_holm"] = min(1.0, adjusted_max)

    return {
        "schema": "loslab.paired_tdl_pusch_time_bler.v1",
        "evidence_grade": "SIMULATION_ONLY_UNVALIDATED",
        "control": {
            "path": str(control_path),
            "sha256": sha256_file(control_path),
            "varying_config_value": control_value,
            "speed_mps": control_config["speed_mps"],
            "dmrs_additional_position": control_config.get("dmrs_additional_position", 0),
            "delay_spread_s": control_config.get("delay_spread_s"),
            "profile": control_config.get("profile"),
            "channel_memory_samples": control_config.get("channel_memory_samples"),
            "cp_covers_channel_memory": control_config.get("cp_covers_channel_memory"),
            "transmit_waveform_mean_power_by_batch": control_config.get(
                "transmit_waveform_mean_power_by_batch"
            ),
        },
        "treatment": {
            "path": str(treatment_path),
            "sha256": sha256_file(treatment_path),
            "varying_config_value": treatment_value,
            "speed_mps": treatment_config["speed_mps"],
            "dmrs_additional_position": treatment_config.get("dmrs_additional_position", 0),
            "delay_spread_s": treatment_config.get("delay_spread_s"),
            "profile": treatment_config.get("profile"),
            "channel_memory_samples": treatment_config.get("channel_memory_samples"),
            "cp_covers_channel_memory": treatment_config.get("cp_covers_channel_memory"),
            "transmit_waveform_mean_power_by_batch": treatment_config.get(
                "transmit_waveform_mean_power_by_batch"
            ),
        },
        "paired_design": {
            "same_configuration_except": varying_config_field,
            "additional_control_policy_change": (
                "allow_cp_overrun: false to true"
                if varying_config_field == "delay_spread_s"
                else None
            ),
            "derived_configuration_changes": (
                ["channel_memory_samples", "cp_covers_channel_memory", "l_max"]
                if varying_config_field == "delay_spread_s"
                else []
            ),
            "same_transmitted_bits_by_seed": same_bits,
            "same_channel_truth_by_seed": same_channel,
            "same_batch_seeds_and_noise_seeds": True,
            "statistic": "paired codeword bootstrap CI and Holm-adjusted exact McNemar tests",
            "channel_truth_is_intentionally_changed": varying_config_field
            != "dmrs_additional_position",
            "input_environments": {
                "control": control_config.get("versions"),
                "treatment": treatment_config.get("versions"),
            },
            "input_package_source_sha256": {
                "control": control_config.get("package_source_sha256"),
                "treatment": treatment_config.get("package_source_sha256"),
            },
            "analysis_environment": versions(),
            "analysis_package_source_sha256": package_source_sha256(),
        },
        "points": comparisons,
        "limitations": [
            (
                "Inference is conditional on the listed standardized TDL profile pair and link configuration."
                if varying_config_field == "profile"
                else "Inference is conditional on one standardized TDL simulation configuration."
            ),
            "Codeword-paired uncertainty does not represent independent-scene uncertainty.",
            "The decoder and synchronization configuration are fixed; the result is not standards certification.",
        ],
    }


def compare_rx_correlation_curves(
    control_path: str | Path,
    treatment_path: str | Path,
) -> dict[str, Any]:
    """Compare paired 2-RX BLER curves with uncorrelated and correlated branches."""
    control_path = Path(control_path)
    treatment_path = Path(treatment_path)
    control = json.loads(control_path.read_text(encoding="utf8"))
    treatment = json.loads(treatment_path.read_text(encoding="utf8"))
    for path, curve in ((control_path, control), (treatment_path, treatment)):
        if curve.get("schema") != "loslab.tdl_pusch_rx_diversity.v1":
            raise ValueError(f"unsupported receive-diversity schema in {path}")
        if curve.get("evidence_grade") != "SIMULATION_ONLY_UNVALIDATED":
            raise ValueError(f"unexpected evidence grade in {path}")

    control_config = control["configuration"]
    treatment_config = treatment["configuration"]
    control_rho = control_config.get("rx_correlation_coefficient", 0.0)
    treatment_rho = treatment_config.get("rx_correlation_coefficient")
    if control_rho != 0.0 or treatment_rho is None or treatment_rho == 0.0:
        raise ValueError("control must be uncorrelated and treatment must use nonzero correlation")
    varying = {
        "rx_correlation_coefficient",
        "rx_correlation_matrix_real",
        "rx_correlation_matrix_imag",
        "receive_spatial_correlation_assumption",
        "matched_realization_checks",
        "clean_receive_power_by_antenna_per_batch",
        "package_source_sha256",
        "versions",
    }
    for key in set(control_config) | set(treatment_config):
        if key not in varying and control_config.get(key) != treatment_config.get(key):
            raise ValueError(f"receive-diversity configuration differs unexpectedly at {key}")

    control_checks = control_config["matched_realization_checks"]
    treatment_checks = treatment_config["matched_realization_checks"]
    if len(control_checks) != len(treatment_checks):
        raise ValueError("matched realization batch counts differ")
    for left, right in zip(control_checks, treatment_checks, strict=True):
        for key in ("seed", "noise_real_seed", "noise_imag_seed", "transmit_bits_sha256"):
            if left.get(key) != right.get(key):
                raise ValueError(f"{key} differs for matched batch {left.get('seed')}")
        if left.get("first_rx_channel_truth_sha256") != right.get("first_rx_channel_truth_sha256"):
            raise ValueError("first receive-antenna channel truth differs across correlation")
        treatment_latent_hash = right.get("latent_uncorrelated_channel_truth_sha256")
        if treatment_latent_hash != left.get("all_rx_channel_truth_sha256"):
            raise ValueError("correlated treatment does not reference the control's latent channel")

    control_points = control["points"]
    treatment_points = treatment["points"]
    if len(control_points) != len(treatment_points):
        raise ValueError("SNR point counts differ")
    comparisons: list[dict[str, Any]] = []
    for left, right in zip(control_points, treatment_points, strict=True):
        if left["snr_db"] != right["snr_db"] or left["frames"] != right["frames"]:
            raise ValueError("SNR values or codeword counts differ")
        if left["single_rx"]["frame_error_mask"] != right["single_rx"]["frame_error_mask"]:
            raise ValueError("single-RX outcomes differ despite identical first-antenna inputs")
        comparisons.append(
            {
                "snr_db": left["snr_db"],
                **paired_binary_error_comparison(
                    left["two_rx"]["frame_error_mask"],
                    right["two_rx"]["frame_error_mask"],
                    seed=1,
                    draws=20_000,
                ),
            }
        )
    ordered = sorted(
        range(len(comparisons)),
        key=lambda index: comparisons[index]["mcnemar_exact_two_sided_p_value"],
    )
    adjusted_max = 0.0
    for rank, index in enumerate(ordered):
        raw_p = comparisons[index]["mcnemar_exact_two_sided_p_value"]
        adjusted_max = max(adjusted_max, (len(comparisons) - rank) * raw_p)
        comparisons[index]["mcnemar_exact_two_sided_p_value_holm"] = min(1.0, adjusted_max)

    return {
        "schema": "loslab.paired_tdl_pusch_rx_correlation.v1",
        "evidence_grade": "SIMULATION_ONLY_UNVALIDATED",
        "control": {
            "path": str(control_path),
            "sha256": sha256_file(control_path),
            "rx_correlation_coefficient": control_rho,
            "versions": control_config.get("versions"),
            "package_source_sha256": control_config.get("package_source_sha256"),
        },
        "treatment": {
            "path": str(treatment_path),
            "sha256": sha256_file(treatment_path),
            "rx_correlation_coefficient": treatment_rho,
            "versions": treatment_config.get("versions"),
            "package_source_sha256": treatment_config.get("package_source_sha256"),
        },
        "paired_design": {
            "same_transmitted_bits": True,
            "same_latent_channel_draw": True,
            "same_first_receive_antenna_channel": True,
            "same_single_rx_outcomes": True,
            "statistic": "paired codeword bootstrap CI and Holm-adjusted exact McNemar tests",
            "analysis_environment": versions(),
            "analysis_package_source_sha256": package_source_sha256(),
        },
        "points": comparisons,
        "limitations": [
            "The Cholesky transform is applied to simulated TDL receive branches, not a measured array response.",
            "Codeword-paired uncertainty is conditional on one TDL-A, numerology, speed and link configuration; it is not scene-level inference.",
            "The decoder and synchronization configuration are fixed; the result is not standards certification.",
        ],
    }


def summarize_tdl_profile_sweep(curve_paths: list[str | Path]) -> dict[str, Any]:
    """Audit and compare matched sample-domain TDL-A–E curves against TDL-A."""
    curves_by_profile: dict[str, Path] = {}
    loaded_by_profile: dict[str, dict[str, Any]] = {}
    for curve_path in curve_paths:
        path = Path(curve_path)
        curve = load_curve(path)
        profile = str(curve["configuration"].get("profile", ""))
        if profile not in {f"TDL-{letter}" for letter in "ABCDE"}:
            raise ValueError(f"unsupported profile {profile!r} in {path}")
        if profile in curves_by_profile:
            raise ValueError(f"duplicate {profile} curve")
        curves_by_profile[profile] = path
        loaded_by_profile[profile] = curve
    required = {f"TDL-{letter}" for letter in "ABCDE"}
    if set(curves_by_profile) != required:
        raise ValueError("profile sweep requires exactly one curve for every TDL-A through TDL-E")

    control = loaded_by_profile["TDL-A"]
    control_config = control["configuration"]
    control_checks = control_config["matched_realization_checks"]
    control_points = control["points"]
    for profile in sorted(required - {"TDL-A"}):
        treatment = loaded_by_profile[profile]
        treatment_config = treatment["configuration"]
        varying = {
            "profile",
            "matched_realization_checks",
            "transmit_waveform_mean_power_by_batch",
        }
        for key in set(control_config) | set(treatment_config):
            if key not in varying and control_config.get(key) != treatment_config.get(key):
                raise ValueError(f"{profile} configuration differs unexpectedly at {key}")
        treatment_checks = treatment_config["matched_realization_checks"]
        if len(control_checks) != len(treatment_checks):
            raise ValueError(f"{profile} matched batch count differs from TDL-A")
        for left, right in zip(control_checks, treatment_checks, strict=True):
            if left["seed"] != right["seed"]:
                raise ValueError(f"{profile} batch seeds differ from TDL-A")
            for key in ("noise_seed", "noise_real_seed", "noise_imag_seed"):
                if left.get(key) != right.get(key):
                    raise ValueError(f"{profile} {key} differs from TDL-A")
            if left["transmit_bits_sha256"] != right["transmit_bits_sha256"]:
                raise ValueError(f"{profile} transmitted bits differ from TDL-A")
            if left["channel_truth_sha256"] == right["channel_truth_sha256"]:
                raise ValueError(f"{profile} channel truth unexpectedly matches TDL-A")
        if len(control_points) != len(treatment["points"]):
            raise ValueError(f"{profile} SNR grid differs from TDL-A")
        for left, right in zip(control_points, treatment["points"], strict=True):
            if left["snr_db"] != right["snr_db"] or left["frames"] != right["frames"]:
                raise ValueError(f"{profile} SNR values or frame counts differ from TDL-A")

    comparisons: dict[str, dict[str, Any]] = {}
    family: list[tuple[str, int, float]] = []
    for profile in sorted(required - {"TDL-A"}):
        comparison = compare_curves(
            curves_by_profile["TDL-A"],
            curves_by_profile[profile],
            varying_config_field="profile",
        )
        comparisons[profile] = comparison
        for index, point in enumerate(comparison["points"]):
            family.append((profile, index, point["mcnemar_exact_two_sided_p_value"]))

    ordered = sorted(range(len(family)), key=lambda index: family[index][2])
    adjusted_max = 0.0
    family_size = len(family)
    for rank, family_index in enumerate(ordered):
        profile, point_index, raw_p = family[family_index]
        adjusted_max = max(adjusted_max, (family_size - rank) * raw_p)
        comparisons[profile]["points"][point_index][
            "mcnemar_exact_two_sided_p_value_holm_family20"
        ] = min(1.0, adjusted_max)

    return {
        "schema": "loslab.tdl_pusch_tdl_profile_sweep.v1",
        "evidence_grade": "SIMULATION_ONLY_UNVALIDATED",
        "control_profile": "TDL-A",
        "paired_design": {
            "same_transmitted_bits_by_batch": True,
            "same_batch_and_awgn_seeds": True,
            "different_channel_truth_by_profile": True,
            "multiplicity_family": "four TDL contrasts versus TDL-A across all five SNR points (20 tests)",
        },
        "curves": {
            profile: {
                "path": str(curves_by_profile[profile]),
                "sha256": sha256_file(curves_by_profile[profile]),
                "configuration": {
                    key: value
                    for key, value in loaded_by_profile[profile]["configuration"].items()
                    if key
                    not in {
                        "matched_realization_checks",
                        "transmit_waveform_mean_power_by_batch",
                    }
                },
                "points": [
                    {
                        key: point[key]
                        for key in (
                            "snr_db",
                            "frames",
                            "bler_errors",
                            "bler",
                            "bler_wilson_95_ci",
                            "bit_errors",
                            "bits",
                            "ber",
                        )
                    }
                    for point in loaded_by_profile[profile]["points"]
                ],
            }
            for profile in sorted(required)
        },
        "comparisons_vs_tdl_a": comparisons,
        "analysis": {
            "source_revision": versions().get("git_sha"),
            "package_source_sha256": package_source_sha256(),
        },
        "limitations": [
            "Results are conditional on the configured standardized Sionna TDL models and one PUSCH receiver configuration.",
            "All compared profiles use ideal synchronization; no sample-domain residual CFO or timing error is injected.",
            "The curves are simulation evidence, not standards certification or validation against measured channels.",
            "Codeword-level intervals do not quantify uncertainty across independent channel environments.",
        ],
    }
