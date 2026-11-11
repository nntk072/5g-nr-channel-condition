"""Evaluate grouped split-conformal intervals and selective scalar regression risk."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from artifacts.provenance import sha256_file
from evaluation.uncertainty_io import (
    evaluate_prediction_file,
    evaluate_prediction_file_build_node,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, help="frozen input JSON")
    parser.add_argument("output", type=Path, help="new report JSON (will not overwrite)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument(
        "--build-cache",
        type=Path,
        help="optional immutable build-DAG store for verified report reuse",
    )
    args = parser.parse_args()
    if not args.predictions.is_file():
        parser.error(f"prediction input does not exist: {args.predictions}")
    if args.output.exists():
        parser.error(f"refusing to overwrite existing report: {args.output}")
    node = None
    if args.build_cache is None:
        report = evaluate_prediction_file(
            args.predictions, seed=args.seed, bootstrap_draws=args.bootstrap_draws
        )
    else:
        report, node = evaluate_prediction_file_build_node(
            args.predictions,
            args.build_cache,
            seed=args.seed,
            bootstrap_draws=args.bootstrap_draws,
        )
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    response = {"report": str(args.output), "sha256": sha256_file(args.output)}
    if node is not None:
        response["build_node"] = node
    print(json.dumps(response))


if __name__ == "__main__":
    main()
