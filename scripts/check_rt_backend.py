"""Probe installed Mitsuba variants with a minimal native Sionna RT trace."""

from __future__ import annotations

import argparse
import json

from integrations.sionna.backend import activate_rt_backend, select_rt_backend


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        default="auto",
        help="auto, cuda, llvm, or an exact installed Mitsuba variant",
    )
    args = parser.parse_args()
    try:
        selection = select_rt_backend(args.backend)
        activate_rt_backend(selection)
    except RuntimeError as exc:
        parser.error(str(exc))
    print(json.dumps(selection.as_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
