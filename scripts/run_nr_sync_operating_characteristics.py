"""Run a seeded, noise-calibrated NR PSS/SSS acquisition study."""

from __future__ import annotations

import argparse
from pathlib import Path

from artifacts.provenance import atomic_json, package_source_sha256, sha256_file, versions
from experiments.nr_sync_operating_characteristics import (
    run_nr_sync_operating_characteristics,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--calibration-trials", type=int, default=1024)
    parser.add_argument("--validation-trials", type=int, default=2048)
    parser.add_argument("--signal-trials-per-point", type=int, default=256)
    parser.add_argument("--target-false-alarm-probability", type=float, default=0.05)
    parser.add_argument("--snr-db", type=float, nargs="+", default=[0.0, 6.0, 12.0, 18.0])
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    try:
        result = run_nr_sync_operating_characteristics(
            seed=args.seed,
            calibration_trials=args.calibration_trials,
            validation_trials=args.validation_trials,
            signal_trials_per_point=args.signal_trials_per_point,
            target_false_alarm_probability=args.target_false_alarm_probability,
            snr_db_values=args.snr_db,
        )
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    runtime = versions()
    result.update(
        {
            "source_revision": runtime.get("git_sha"),
            "package_source_sha256": package_source_sha256(),
            "runner_sha256": sha256_file(__file__),
            "versions": runtime,
        }
    )
    atomic_json(args.output, result)
    print(f"Saved NR acquisition study to {args.output}")
    print(
        "Held-out noise false alarms: "
        f"{result['false_alarm_validation']['false_alarms']}/"
        f"{result['false_alarm_validation']['trials']}"
    )


if __name__ == "__main__":
    main()
