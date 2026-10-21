#!/usr/bin/env python3
"""Validate and compact a raw calibrated profile-likelihood stress report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from artifacts.provenance import atomic_json, sha256_file  # noqa: E402
from experiments.identifiability_stress_summary import (  # noqa: E402
    summarize_profile_likelihood_stress,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="raw experiment report")
    parser.add_argument("output", type=Path, help="new compact summary path")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        summary = summarize_profile_likelihood_stress(
            report, raw_report_sha256=sha256_file(args.report)
        )
        atomic_json(args.output, summary)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(f"Saved {len(summary['cases'])} stress cases to {args.output}")


if __name__ == "__main__":
    main()
