#!/usr/bin/env python3
"""Run the synthetic unknown-path-order and noise-estimation stress test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.multipath_order_stress import run_multipath_order_stress


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--trials-per-cell", type=int, default=40)
    parser.add_argument("--tone-count", type=int, default=64)
    parser.add_argument("--delay-grid-size", type=int, default=257)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {args.output}")
    result = run_multipath_order_stress(
        seed=args.seed,
        trials_per_cell=args.trials_per_cell,
        tone_count=args.tone_count,
        delay_grid_size=args.delay_grid_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "result_sha256": result["result_sha256"]}))


if __name__ == "__main__":
    main()
