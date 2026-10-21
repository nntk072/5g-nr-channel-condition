"""Experiment entry point for the native PUSCH cyclic-prefix diagnostic."""

from __future__ import annotations

import math
from typing import Any

from artifacts.provenance import package_source_sha256, versions
from evaluation.statistics import paired_binary_error_comparison
from integrations.sionna.cp_margin import run_static_cp_margin_experiment as _run_native

from experiments.link_curves import wilson_score_interval_95


def run_static_cp_margin_experiment(**kwargs: Any) -> dict[str, Any]:
    """Run the Sionna-backed controlled CP-margin experiment."""
    result = _run_native(**kwargs)
    points = result["points"]
    for name, point in points.items():
        errors = point["bler_errors"]
        frames = point["frames"]
        lower, upper = wilson_score_interval_95(errors, frames)
        point["bler"] = errors / frames
        point["bler_wilson_95_ci"] = [lower, upper]
        point["ber"] = point["bit_errors"] / point["bits"]
        point["mean_transmitted_tb_bits_per_codeword"] = point["bits"] // frames
        point["successfully_decoded_tb_bits_per_codeword"] = point[
            "mean_transmitted_tb_bits_per_codeword"
        ] * (1.0 - point["bler"])
        powers = result["receive_waveform_mean_power_by_batch"][name]
        point["received_waveform_mean_power_by_batch"] = powers
        point["measured_receive_snr_db_by_batch"] = [
            10.0 * math.log10(power / result["configuration"]["noise_variance_per_complex_sample"])
            for power in powers
        ]
    bits_per_frame = points["inside_cp"]["bits"] / points["inside_cp"]["frames"]
    result["paired_comparisons_vs_inside_cp"] = {
        condition: paired_binary_error_comparison(
            points["inside_cp"]["frame_error_mask"],
            point["frame_error_mask"],
            seed=result["configuration"]["seed"],
            draws=20_000,
            control_bits_per_success=bits_per_frame,
            treatment_bits_per_success=bits_per_frame,
        )
        for condition, point in points.items()
        if condition != "inside_cp"
    }
    ordered_comparisons = sorted(
        result["paired_comparisons_vs_inside_cp"],
        key=lambda name: result["paired_comparisons_vs_inside_cp"][name][
            "mcnemar_exact_two_sided_p_value"
        ],
    )
    adjusted_max = 0.0
    comparison_count = len(ordered_comparisons)
    for rank, condition in enumerate(ordered_comparisons):
        comparison = result["paired_comparisons_vs_inside_cp"][condition]
        raw_p = comparison["mcnemar_exact_two_sided_p_value"]
        adjusted_max = max(adjusted_max, (comparison_count - rank) * raw_p)
        comparison["mcnemar_exact_two_sided_p_value_holm"] = min(1.0, adjusted_max)
    result["paired_comparison"] = result["paired_comparisons_vs_inside_cp"]["beyond_cp"]
    result.update(
        {
            "schema": "loslab.pusch_static_cp_margin.v1",
            "evidence_grade": "SIMULATION_ONLY_UNVALIDATED",
        }
    )
    result["configuration"].update(
        {"versions": versions(), "package_source_sha256": package_source_sha256()}
    )
    return result
