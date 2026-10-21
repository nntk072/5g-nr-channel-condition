#!/usr/bin/env python3
"""Compare paired one- and two-RX TDL curves across receive correlation settings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from artifacts.provenance import atomic_json  # noqa: E402
from evaluation.paired_link_curves import compare_rx_correlation_curves  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path, help="uncorrelated RX diversity curve JSON")
    parser.add_argument("treatment", type=Path, help="correlated RX diversity curve JSON")
    parser.add_argument("output", type=Path, help="new paired comparison JSON path")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    try:
        comparison = compare_rx_correlation_curves(args.control, args.treatment)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    atomic_json(args.output, comparison)
    print(
        json.dumps(
            [
                {
                    "snr_db": point["snr_db"],
                    "control_bler_errors": point["control_errors"],
                    "treatment_bler_errors": point["treatment_errors"],
                    "treatment_minus_control_bler": point["treatment_minus_control_bler"],
                    "holm_p": point["mcnemar_exact_two_sided_p_value_holm"],
                }
                for point in comparison["points"]
            ],
            indent=2,
        )
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
