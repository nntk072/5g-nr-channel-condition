"""Compare paired outputs from two seeded NR sync operating-characteristic studies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from artifacts.provenance import atomic_json, sha256_file
from experiments.nr_sync_operating_characteristics import (
    compare_nr_sync_operating_characteristics,
)


def _read_study(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read study JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"study root must be a JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path)
    parser.add_argument("treatment", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bootstrap-seed", type=int, default=20261027)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    try:
        result = compare_nr_sync_operating_characteristics(
            _read_study(args.control),
            _read_study(args.treatment),
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_draws=args.bootstrap_draws,
        )
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    result["inputs"] = {
        "control": {"path": str(args.control), "sha256": sha256_file(args.control)},
        "treatment": {"path": str(args.treatment), "sha256": sha256_file(args.treatment)},
    }
    atomic_json(args.output, result)
    print(f"Saved paired NR sync comparison to {args.output}")


if __name__ == "__main__":
    main()
