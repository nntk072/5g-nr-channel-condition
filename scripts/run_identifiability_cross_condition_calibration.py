"""Run leave-one-condition-out uncertainty calibration stress tests."""

from __future__ import annotations

import argparse
from pathlib import Path

from artifacts.provenance import atomic_json, sha256_file
from experiments.identifiability_cross_condition import build_cross_condition_calibration_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--trials", type=int, default=512)
    parser.add_argument("--calibration-trials-per-condition", type=int, default=512)
    parser.add_argument("--tone-counts", type=int, nargs="+", default=[256, 1024])
    parser.add_argument("--tone-spacing-hz", type=float, default=30_000.0)
    parser.add_argument("--snr-db", type=float, nargs="+", default=[20.0, 30.0])
    parser.add_argument("--separation-fractions", type=float, nargs="+", default=[0.1, 1.0])
    parser.add_argument("--gain-phase-difference-deg", type=float, nargs="+", default=[0.0, 90.0])
    parser.add_argument("--delay-grid-step-fraction", type=float, default=1.0 / 128.0)
    parser.add_argument("--maximum-delay-fraction", type=float, default=2.0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    try:
        result = build_cross_condition_calibration_sweep(
            seed=args.seed,
            trials=args.trials,
            calibration_trials_per_condition=args.calibration_trials_per_condition,
            tone_counts=args.tone_counts,
            tone_spacing_hz=args.tone_spacing_hz,
            snr_db_values=args.snr_db,
            separation_fractions=args.separation_fractions,
            gain_phase_differences_deg=args.gain_phase_difference_deg,
            delay_grid_step_fraction=args.delay_grid_step_fraction,
            maximum_delay_fraction=args.maximum_delay_fraction,
        )
    except ValueError as error:
        parser.error(str(error))
    result["runner_sha256"] = sha256_file(__file__)
    atomic_json(args.output, result)
    print(f"Saved {len(result['cases'])} cases to {args.output}")


if __name__ == "__main__":
    main()
