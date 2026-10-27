"""Command-line interface for deterministic pilot sampling and result audits."""

from __future__ import annotations

import argparse
import json
import os
import stat
from pathlib import Path
from typing import Any

from artifacts.provenance import atomic_json

from .pilot_sampling import audit_pilot_outcomes, build_pilot_sample

_MAX_INPUT_BYTES = 64 * 1024 * 1024


def _identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_json(path: Path, *, label: str) -> Any:
    before_path = os.lstat(path)
    if not stat.S_ISREG(before_path.st_mode) or before_path.st_nlink != 1:
        raise ValueError(f"{label} must be a single-link regular file")
    if before_path.st_size > _MAX_INPUT_BYTES:
        raise ValueError(f"{label} exceeds the {_MAX_INPUT_BYTES}-byte limit")
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        before = os.fstat(descriptor)
        if _identity(before) != _identity(before_path):
            raise ValueError(f"{label} changed while opening")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, _MAX_INPUT_BYTES + 1 - size))
            if not chunk:
                break
            size += len(chunk)
            if size > _MAX_INPUT_BYTES:
                raise ValueError(f"{label} exceeds the {_MAX_INPUT_BYTES}-byte limit")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if size != before.st_size or _identity(before) != _identity(after):
            raise ValueError(f"{label} changed while being read")
    finally:
        os.close(descriptor)
    try:
        return json.loads(
            b"".join(chunks).decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError(f"{label} is not bounded valid UTF-8 JSON") from error


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON field {key!r}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value {value!r} is forbidden")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sample_parser = commands.add_parser("sample", help="build a deterministic pilot roster")
    sample_parser.add_argument("--design", type=Path, required=True)
    sample_parser.add_argument("--candidates", type=Path, required=True)
    sample_parser.add_argument("--output", type=Path, required=True)
    audit_parser = commands.add_parser("audit", help="summarize observed pilot outcomes and costs")
    audit_parser.add_argument("--sample", type=Path, required=True)
    audit_parser.add_argument("--outcomes", type=Path, required=True)
    audit_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    try:
        if args.command == "sample":
            design = _read_json(args.design, label="pilot design")
            candidates = _read_json(args.candidates, label="pilot candidate roster")
            result = build_pilot_sample(design, candidates)
            complete_status = "SAMPLE_ROSTER_COMPLETE"
        else:
            sample = _read_json(args.sample, label="pilot sample report")
            outcomes = _read_json(args.outcomes, label="pilot outcomes")
            result = audit_pilot_outcomes(sample, outcomes)
            complete_status = "COMPLETE_DIAGNOSTIC"
        atomic_json(args.output, result, overwrite=False)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(f"{result['status']}: {args.output}")
    if result["status"] != complete_status:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
