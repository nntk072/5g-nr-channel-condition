"""Generate one timestamped, hash-addressed Sionna RT-to-NR trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from integrations.sionna.rt_trajectory import RTTrajectoryConfig, generate_rt_nr_trajectory


def _triple(value: Any, *, field: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{field} must be a JSON array with three numbers")
    return tuple(float(item) for item in value)  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path, help="JSON trajectory configuration")
    parser.add_argument("store", type=Path, help="content-addressed trajectory store")
    args = parser.parse_args()
    if not args.spec.is_file():
        parser.error(f"trajectory spec does not exist: {args.spec}")
    try:
        values = json.loads(args.spec.read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("trajectory spec root must be a JSON object")
        config = RTTrajectoryConfig(
            **{
                **values,
                "transmitter_position_m": _triple(
                    values.get("transmitter_position_m"), field="transmitter_position_m"
                ),
                "receiver_start_m": _triple(
                    values.get("receiver_start_m"), field="receiver_start_m"
                ),
                "receiver_velocity_mps": _triple(
                    values.get("receiver_velocity_mps"), field="receiver_velocity_mps"
                ),
            }
        )
        stored = generate_rt_nr_trajectory(config, args.store)
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "trajectory_id": stored.trajectory_id,
                "content_sha256": stored.content_sha256,
                "manifest": stored.manifest_relative_path,
                "sample_ids": list(stored.sample_ids),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
