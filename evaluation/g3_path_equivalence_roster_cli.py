"""Audit completeness and directional coverage of a preregistered G3 path roster."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

from evaluation.g3_joint_acceptance import (
    FAMILY_SCHEMA_V4,
    validate_g3_joint_acceptance_family,
)
from evaluation.g3_path_equivalence_assurance import (
    evaluate_path_equivalence_roster,
)

_INPUT_SCHEMA = "loslab.g3_path_equivalence_roster_input.v1"
_MAX_INPUT_BYTES = 64 * 1024 * 1024


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_input(path: Path) -> dict[str, Any]:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_NONBLOCK"):
        raise RuntimeError("platform lacks no-follow input opening required for the audit")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError("input must be a single-link regular file")
        if before.st_size > _MAX_INPUT_BYTES:
            raise ValueError("input exceeds the configured size limit")
        chunks: list[bytes] = []
        remaining = _MAX_INPUT_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
        if len(raw) > _MAX_INPUT_BYTES:
            raise ValueError("input exceeds the configured size limit")

        def identity(item: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
            return (
                item.st_dev,
                item.st_ino,
                item.st_mode,
                item.st_nlink,
                item.st_size,
                item.st_mtime_ns,
                item.st_ctime_ns,
            )

        if identity(before) != identity(after) or identity(after) != identity(current):
            raise ValueError("input changed while it was being read")
    finally:
        os.close(descriptor)
    value = json.loads(
        raw.decode("utf-8"),
        parse_constant=_reject_constant,
        object_pairs_hook=_unique_object,
    )
    if not isinstance(value, dict):
        raise ValueError("input root must be an object")
    if set(value) != {"schema", "joint_acceptance_family", "records"}:
        raise ValueError("input must contain exactly schema, joint_acceptance_family, and records")
    if value.get("schema") != _INPUT_SCHEMA:
        raise ValueError(f"input schema must be {_INPUT_SCHEMA}")
    family = validate_g3_joint_acceptance_family(value.get("joint_acceptance_family"))
    if family["schema"] != FAMILY_SCHEMA_V4:
        raise ValueError("path-roster CLI requires a joint-acceptance family v4")
    records = value.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("records must be a nonempty list")
    value["joint_acceptance_family"] = family
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="frozen JSON path-roster diagnostic input")
    args = parser.parse_args(argv)
    try:
        value = _read_input(args.input)
        family = value["joint_acceptance_family"]
        report = evaluate_path_equivalence_roster(
            value["records"],
            expected_scenarios=[
                tuple(item) for item in family["path_equivalence_expected_scenarios"]
            ],
            joint_acceptance_family=family,
        )
    except (OSError, UnicodeDecodeError, ValueError, TypeError, RuntimeError, RecursionError) as error:
        parser.exit(2, f"error: {error}\n")
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
