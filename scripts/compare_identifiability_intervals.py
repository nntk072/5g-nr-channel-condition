"""Compare nominal profile intervals with calibrated profile-likelihood sets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from artifacts.provenance import atomic_json, package_source_sha256, sha256_file, versions
from experiments.identifiability_interval_comparison import compare_profile_intervals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    try:
        study: Any = json.loads(args.study.read_text(encoding="utf8"))
        if not isinstance(study, dict):
            raise ValueError("study root must be a JSON object")
        result = compare_profile_intervals(
            study, seed=args.seed, bootstrap_draws=args.bootstrap_draws
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        parser.error(str(error))
    result["input"] = {
        "path": str(args.study),
        "sha256": sha256_file(args.study),
    }
    result["versions"] = versions()
    result["package_source_sha256"] = package_source_sha256()
    result["comparison_module_sha256"] = sha256_file(
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "identifiability_interval_comparison.py"
    )
    result["runner_sha256"] = sha256_file(__file__)
    atomic_json(args.output, result)
    print(f"Saved paired interval comparison to {args.output}")


if __name__ == "__main__":
    main()
