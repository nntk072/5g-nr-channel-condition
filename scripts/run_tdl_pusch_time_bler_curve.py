#!/usr/bin/env python3
"""Run and persist a reproducible sample-domain Sionna TDL-PUSCH BLER curve."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from artifacts.provenance import atomic_json  # noqa: E402
from experiments.link_curves import run_tdl_pusch_time_bler_curve  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output", type=Path, help="new JSON output path; existing files are refused"
    )
    parser.add_argument("--snr-db", type=float, nargs="+", default=[0, 6, 12, 18, 24])
    parser.add_argument("--frames", type=int, default=1024, help="codewords per SNR point")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=41000)
    parser.add_argument("--profile", choices=list("ABCDE"), default="A")
    parser.add_argument("--delay-spread-ns", type=float, default=100.0)
    parser.add_argument("--carrier-ghz", type=float, default=3.5)
    parser.add_argument("--speed-mps", type=float, default=3.0)
    parser.add_argument("--n-size-bwp", type=int, default=12)
    parser.add_argument("--dmrs-additional-position", type=int, choices=range(4), default=0)
    parser.add_argument("--subcarrier-spacing-khz", type=int, default=30)
    parser.add_argument("--num-rx-ant", type=int, default=1)
    parser.add_argument(
        "--allow-cp-overrun",
        action="store_true",
        help="allow an explicit channel-memory-over-CP stress run",
    )
    parser.add_argument(
        "--channel-estimator",
        choices=("sionna_ls", "finite_delay_dft"),
        default="sionna_ls",
    )
    parser.add_argument("--estimator-min-delay-samples", type=int)
    parser.add_argument("--estimator-max-delay-samples", type=int)
    args = parser.parse_args()

    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    result = run_tdl_pusch_time_bler_curve(
        snr_db=args.snr_db,
        frames_per_snr=args.frames,
        batch_size=args.batch_size,
        seed=args.seed,
        profile=args.profile,
        delay_spread_s=args.delay_spread_ns * 1e-9,
        carrier_frequency_hz=args.carrier_ghz * 1e9,
        speed_mps=args.speed_mps,
        n_size_bwp=args.n_size_bwp,
        dmrs_additional_position=args.dmrs_additional_position,
        subcarrier_spacing_khz=args.subcarrier_spacing_khz,
        num_rx_ant=args.num_rx_ant,
        allow_cp_overrun=args.allow_cp_overrun,
        channel_estimator_kind=args.channel_estimator,
        estimator_min_delay_samples=args.estimator_min_delay_samples,
        estimator_max_delay_samples=args.estimator_max_delay_samples,
    )
    atomic_json(args.output, result)
    print(json.dumps(result["points"], indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
