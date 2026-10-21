"""Run a fixed-split small/medium/large neural baseline capacity sweep."""

from __future__ import annotations

import argparse
import json

from experiments.capacity_sweep import run_capacity_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, help="channel dataset NPZ files")
    parser.add_argument("--output", required=True, help="new immutable sweep directory")
    parser.add_argument(
        "--methods", nargs="+", required=True, help="e.g. mlp raw_mlp cnn cvnn coral coral0"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 47, 59, 71, 83])
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--holdout")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--phase", action="store_true")
    parser.add_argument(
        "--observation", choices=("ideal_cfr", "comb_ls", "precomputed"), default="ideal_cfr"
    )
    parser.add_argument("--pilot-spacing", type=int, default=4)
    parser.add_argument("--pilot-snr-db", type=float, default=12.0)
    parser.add_argument("--observation-seed", type=int, default=2026)
    args = parser.parse_args()
    report = run_capacity_sweep(
        args.input,
        args.output,
        methods=args.methods,
        seeds=args.seeds,
        bootstrap=args.bootstrap,
        holdout=args.holdout,
        include_phase=args.phase,
        observation=args.observation,
        pilot_spacing=args.pilot_spacing,
        pilot_snr_db=args.pilot_snr_db,
        observation_seed=args.observation_seed,
        device=args.device,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
