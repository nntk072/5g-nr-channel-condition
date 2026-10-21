#!/usr/bin/env python3
"""Repeat independent calibration/evaluation splits for delay-set stress conditions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from artifacts.provenance import atomic_json, sha256_file
from experiments.identifiability_repeated_calibration import (
    build_repeated_calibration_study,
    summarize_repeated_calibration_study,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_output", type=Path)
    parser.add_argument("summary_output", type=Path)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--trials", type=int, default=120)
    parser.add_argument("--calibration-trials", type=int, default=120)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--tone-counts", type=int, nargs="+", default=[64])
    parser.add_argument("--tone-spacing-hz", type=float, default=30_000.0)
    parser.add_argument("--snr-db", type=float, nargs="+", default=[20.0])
    parser.add_argument("--separation-fractions", type=float, nargs="+", default=[0.5])
    parser.add_argument("--gain-phase-difference-deg", type=float, nargs="+", default=[90.0])
    parser.add_argument("--delay-grid-step-fraction", type=float, default=1.0 / 32.0)
    parser.add_argument("--maximum-delay-fraction", type=float, default=1.0)
    parser.add_argument(
        "--noise-variance-scale-factors", type=float, nargs="+", default=[0.5, 1.0, 2.0]
    )
    parser.add_argument("--third-path-amplitude-ratios", type=float, nargs="+", default=[0.0, 0.25])
    args = parser.parse_args()
    if args.raw_output.exists() or args.summary_output.exists():
        parser.error("refusing to overwrite an existing output")
    try:
        report = build_repeated_calibration_study(
            {
                "seed": args.seed,
                "trials": args.trials,
                "calibration_trials": args.calibration_trials,
                "tone_counts": args.tone_counts,
                "tone_spacing_hz": args.tone_spacing_hz,
                "snr_db_values": args.snr_db,
                "separation_fractions": args.separation_fractions,
                "gain_phase_differences_deg": args.gain_phase_difference_deg,
                "delay_grid_step_fraction": args.delay_grid_step_fraction,
                "maximum_delay_fraction": args.maximum_delay_fraction,
                "noise_variance_scale_factors": args.noise_variance_scale_factors,
                "third_path_amplitude_ratios": args.third_path_amplitude_ratios,
                "randomize_path_order": True,
            },
            repetitions=args.repetitions,
            bootstrap_draws=args.bootstrap_draws,
        )
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    report["runner_sha256"] = sha256_file(__file__)
    atomic_json(args.raw_output, report)
    summary = summarize_repeated_calibration_study(
        report, raw_report_sha256=sha256_file(args.raw_output)
    )
    atomic_json(args.summary_output, summary)
    print(
        json.dumps(
            [
                {
                    "noise_variance_ratio": case["assumed_to_true_noise_variance_ratio"],
                    "third_path_ratio": case["third_path_amplitude_ratio_to_strongest"],
                    "mean_split_coverage": case["mean_split_coverage"],
                    "split_bootstrap_interval_95": case["split_bootstrap_percentile_interval_95"],
                }
                for case in summary["cases"]
            ],
            indent=2,
        )
    )
    print(f"Saved raw report and summary to {args.raw_output} and {args.summary_output}")


if __name__ == "__main__":
    main()
