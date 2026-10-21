#!/usr/bin/env python3
"""Audit matched sample-domain TDL-A–E curves and summarize paired results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from artifacts.provenance import atomic_json  # noqa: E402
from evaluation.paired_link_curves import summarize_tdl_profile_sweep  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new compact JSON output; existing files refused")
    parser.add_argument(
        "curves", type=Path, nargs="+", help="raw per-profile curve JSONs for TDL-A through TDL-E"
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    missing = [str(path) for path in args.curves if not path.is_file()]
    if missing:
        parser.error(f"curve input does not exist: {missing[0]}")
    try:
        result = summarize_tdl_profile_sweep(args.curves)
    except (KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    atomic_json(args.output, result)
    print(
        json.dumps(
            {profile: curve["points"] for profile, curve in result["curves"].items()}, indent=2
        )
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
