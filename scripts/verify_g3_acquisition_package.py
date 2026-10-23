"""Verify byte-level and semantic consistency of a proposal-only G3.03 package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluation.g3_acquisition_package import (  # noqa: E402
    verify_g3_acquisition_package,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path, help="package JSON file")
    parser.add_argument(
        "--package-root",
        type=Path,
        help="directory against which package inventory paths resolve (default: package directory)",
    )
    parser.add_argument("--output", type=Path, help="write report JSON to this file")
    args = parser.parse_args(argv)

    report = verify_g3_acquisition_package(
        args.package,
        package_root=args.package_root,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(encoded)
    else:
        args.output.write_text(encoded, encoding="utf-8")
    return 0 if report["status"] == "consistent_proposal_only" else 1


if __name__ == "__main__":
    raise SystemExit(main())
