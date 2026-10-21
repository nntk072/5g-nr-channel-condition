#!/usr/bin/env python3
"""Run and save the fixed-payload native PUSCH cyclic-prefix margin check."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from artifacts.provenance import atomic_json  # noqa: E402
from experiments.cp_margin import run_static_cp_margin_experiment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output", type=Path, help="new JSON output path; existing files are refused"
    )
    parser.add_argument("--frames", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=50_000)
    parser.add_argument("--snr-db", type=float, default=30.0)
    parser.add_argument("--n-size-bwp", type=int, default=12)
    parser.add_argument("--subcarrier-spacing-khz", type=int, default=30)
    parser.add_argument("--in-cp-guard-samples", type=int, default=2)
    parser.add_argument("--beyond-cp-samples", type=int, default=3)
    parser.add_argument(
        "--additional-echo-delays-samples",
        type=int,
        nargs="*",
        default=[],
        help="additional positive discrete echo delays, in the same sample-domain setup",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    result = run_static_cp_margin_experiment(
        frames=args.frames,
        batch_size=args.batch_size,
        seed=args.seed,
        snr_db=args.snr_db,
        n_size_bwp=args.n_size_bwp,
        subcarrier_spacing_khz=args.subcarrier_spacing_khz,
        in_cp_guard_samples=args.in_cp_guard_samples,
        beyond_cp_samples=args.beyond_cp_samples,
        additional_echo_delays_samples=args.additional_echo_delays_samples,
    )
    atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "points": {
                    name: {
                        key: point[key]
                        for key in ("frames", "bler_errors", "bler", "bler_wilson_95_ci", "ber")
                    }
                    for name, point in result["points"].items()
                },
                "paired_comparison": {
                    key: result["paired_comparison"][key]
                    for key in (
                        "treatment_minus_control_bler",
                        "paired_bootstrap_ci95",
                        "treatment_minus_control_goodput_bits_per_codeword",
                        "paired_goodput_bootstrap_ci95",
                        "mcnemar_exact_two_sided_p_value",
                    )
                },
                "paired_comparisons_vs_inside_cp": {
                    condition: {
                        key: comparison[key]
                        for key in (
                            "treatment_minus_control_bler",
                            "paired_bootstrap_ci95",
                            "mcnemar_exact_two_sided_p_value",
                            "mcnemar_exact_two_sided_p_value_holm",
                        )
                    }
                    for condition, comparison in result["paired_comparisons_vs_inside_cp"].items()
                },
            },
            indent=2,
        )
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
