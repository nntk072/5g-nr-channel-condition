"""Run matched profile-likelihood and classical OMP uncertainty experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from artifacts.provenance import (
    atomic_json,
    package_source_sha256,
    sha256_file,
    versions,
)
from experiments.identifiability_matched_estimators import (
    run_matched_uncertainty_estimators,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--trials", type=int, default=256)
    parser.add_argument("--calibration-trials", type=int, default=256)
    parser.add_argument("--tone-counts", type=int, nargs="+", default=[256, 1024])
    parser.add_argument("--tone-spacing-hz", type=float, default=30_000.0)
    parser.add_argument("--snr-db", type=float, nargs="+", default=[20.0, 30.0])
    parser.add_argument("--separation-fractions", type=float, nargs="+", default=[0.1, 1.0])
    parser.add_argument("--gain-phase-differences-deg", type=float, nargs="+", default=[0.0, 90.0])
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    try:
        result = run_matched_uncertainty_estimators(
            seed=args.seed,
            trials=args.trials,
            calibration_trials=args.calibration_trials,
            tone_counts=args.tone_counts,
            tone_spacing_hz=args.tone_spacing_hz,
            snr_db_values=args.snr_db,
            separation_fractions=args.separation_fractions,
            gain_phase_differences_deg=args.gain_phase_differences_deg,
        )
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    runtime = versions()
    result.update(
        {
            "source_revision": runtime.get("git_sha"),
            "package_source_sha256": package_source_sha256(),
            "experiment_module_sha256": sha256_file(
                Path(__file__).resolve().parents[1]
                / "experiments"
                / "identifiability_matched_estimators.py"
            ),
            "runner_sha256": sha256_file(__file__),
            "versions": runtime,
        }
    )
    atomic_json(args.output, result)
    print(f"Saved matched estimator study to {args.output}")


if __name__ == "__main__":
    main()
