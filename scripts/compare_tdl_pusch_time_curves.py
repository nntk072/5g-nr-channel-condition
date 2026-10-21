#!/usr/bin/env python3
"""Compare two paired sample-domain Sionna TDL-PUSCH BLER curves."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from artifacts.provenance import atomic_json  # noqa: E402
from evaluation.paired_link_curves import compare_curves  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path, help="control-condition curve JSON")
    parser.add_argument("treatment", type=Path, help="treatment-condition curve JSON")
    parser.add_argument("output", type=Path, help="new comparison JSON output")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    result = compare_curves(args.control, args.treatment)
    atomic_json(args.output, result)
    print(json.dumps(result["points"], indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
