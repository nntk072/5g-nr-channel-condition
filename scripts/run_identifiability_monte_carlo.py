"""Run a truth-centered local MLE check of nuisance-aware two-path CRLBs."""

from __future__ import annotations

import argparse
from pathlib import Path

from artifacts.provenance import atomic_json, sha256_file
from experiments.identifiability_monte_carlo import build_monte_carlo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--trials", type=int, default=128)
    parser.add_argument("--tone-counts", type=int, nargs="+", default=[256, 1024])
    parser.add_argument("--tone-spacing-hz", type=float, default=30_000.0)
    parser.add_argument("--snr-db", type=float, nargs="+", default=[30.0])
    parser.add_argument("--separation-fractions", type=float, nargs="+", default=[0.1, 1.0])
    parser.add_argument("--gain-phase-difference-deg", type=float, nargs="+", default=[0.0, 90.0])
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    try:
        result = build_monte_carlo(
            seed=args.seed,
            trials=args.trials,
            tone_counts=args.tone_counts,
            tone_spacing_hz=args.tone_spacing_hz,
            snr_db_values=args.snr_db,
            separation_fractions=args.separation_fractions,
            gain_phase_differences_deg=args.gain_phase_difference_deg,
        )
    except ValueError as error:
        parser.error(str(error))
    result["runner_sha256"] = sha256_file(__file__)
    atomic_json(args.output, result)
    cases = result.get("cases")
    if not isinstance(cases, list):
        raise RuntimeError("experiment result is missing its cases list")
    print(f"Saved {len(cases)} cases to {args.output}")


if __name__ == "__main__":
    main()
