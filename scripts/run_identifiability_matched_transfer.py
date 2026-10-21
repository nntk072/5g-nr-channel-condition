"""Analyze cross-condition calibration transfer on paired estimator evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from artifacts.provenance import atomic_json, package_source_sha256, sha256_file, versions
from experiments.identifiability_matched_transfer import (
    analyze_matched_cross_condition_transfer,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--bootstrap-draws", type=int, default=5_000)
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"input report does not exist: {args.input}")
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    try:
        with args.input.open(encoding="utf-8") as source:
            matched_result: dict[str, Any] = json.load(source)
        result = analyze_matched_cross_condition_transfer(
            matched_result,
            seed=args.seed,
            bootstrap_draws=args.bootstrap_draws,
        )
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    result.update(
        {
            "input_report": str(args.input),
            "input_report_sha256": sha256_file(args.input),
            "source_revision": versions().get("git_sha"),
            "package_source_sha256": package_source_sha256(),
            "analysis_module_sha256": sha256_file(
                Path(__file__).resolve().parents[1]
                / "experiments"
                / "identifiability_matched_transfer.py"
            ),
            "runner_sha256": sha256_file(__file__),
            "versions": versions(),
        }
    )
    atomic_json(args.output, result)
    print(f"Saved matched cross-condition transfer analysis to {args.output}")


if __name__ == "__main__":
    main()
