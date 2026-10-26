"""Find preregistered integer seeds in saved JSON result corpora before a run."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any

_MAX_FILE_BYTES = 64 * 1024 * 1024
# The largest retained research JSON currently expands to about 304 MiB. Keep
# parsing bounded while admitting that artifact to the pre-run seed audit.
_MAX_JSON_BYTES = 384 * 1024 * 1024
_SEED_KEYS = {"seed", "seeds", "seed_roster", "seed_base"}


class ScanLimitError(ValueError):
    """A corpus file exceeded a declared read/parse size ceiling."""


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_file(path: Path) -> bytes:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_NONBLOCK"):
        raise RuntimeError("seed audit requires no-follow file opening support")
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"corpus input must be a single-link regular file: {path}")
        if before.st_size > _MAX_FILE_BYTES:
            raise ScanLimitError(f"corpus file exceeds compressed/input byte limit: {path}")
        chunks: list[bytes] = []
        total = 0
        while total <= _MAX_FILE_BYTES:
            chunk = os.read(descriptor, min(1024 * 1024, _MAX_FILE_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        after = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
        if total > _MAX_FILE_BYTES:
            raise ScanLimitError(f"corpus file exceeds compressed/input byte limit: {path}")
        if _identity(before) != _identity(after) or _identity(after) != _identity(current):
            raise ValueError(f"corpus file changed while being read: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _read_json(path: Path, raw: bytes) -> Any:
    if path.name.endswith(".json.gz"):
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
            payload = stream.read(_MAX_JSON_BYTES + 1)
    else:
        payload = raw
    if len(payload) > _MAX_JSON_BYTES:
        raise ScanLimitError(f"expanded JSON exceeds configured size limit: {path}")
    return json.loads(
        payload.decode("utf-8"),
        parse_constant=_reject_constant,
        object_pairs_hook=_unique_object,
    )


def _is_seed_key(key: str) -> bool:
    normalized = key.casefold()
    if normalized in {
        "per_seed",
        "split_seed",
        "transport_blocks_are_nested_within_channel_seed",
    } or normalized.startswith("same_"):
        return False
    if normalized.endswith("_across_seeds") or re.search(r"_over_\d+_seeds$", normalized):
        return False
    return (
        normalized in _SEED_KEYS
        or normalized.endswith("_seed")
        or normalized.endswith("_seeds")
        or normalized.endswith("_seed_base")
    )


def _is_seed_range_key(key: str) -> bool:
    normalized = key.casefold()
    return "seed" in normalized and normalized.endswith("_range")


def _matches(
    value: Any,
    *,
    path: str,
    targets: set[int],
    output: list[dict[str, Any]],
    malformed: list[dict[str, str]],
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else key
            if _is_seed_range_key(key):
                _matches_seed_range(
                    child,
                    path=child_path,
                    targets=targets,
                    output=output,
                    malformed=malformed,
                )
            elif _is_seed_key(key):
                _matches_seed_value(
                    child,
                    path=child_path,
                    targets=targets,
                    output=output,
                    malformed=malformed,
                    allow_identifier=key.casefold() == "seed",
                )
            if "seed" in key.casefold():
                for target in targets:
                    if re.search(rf"(?<![0-9]){target}(?![0-9])", key):
                        output.append(
                            {
                                "json_path": f"{child_path}::<key>",
                                "seed": target,
                                "encoded_type": "key",
                            }
                        )
            _matches(
                child,
                path=child_path,
                targets=targets,
                output=output,
                malformed=malformed,
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _matches(
                child,
                path=f"{path}[{index}]",
                targets=targets,
                output=output,
                malformed=malformed,
            )


def _matches_seed_value(
    value: Any,
    *,
    path: str,
    targets: set[int],
    output: list[dict[str, Any]],
    malformed: list[dict[str, str]],
    allow_identifier: bool = False,
) -> None:
    if isinstance(value, bool):
        malformed.append({"json_path": path, "observed_type": "boolean"})
    elif isinstance(value, int):
        if value in targets:
            output.append({"json_path": path, "seed": value, "encoded_type": "integer"})
    elif isinstance(value, float):
        if math.isfinite(value) and value.is_integer() and int(value) in targets:
            output.append({"json_path": path, "seed": int(value), "encoded_type": "number"})
        else:
            malformed.append({"json_path": path, "observed_type": "number"})
    elif isinstance(value, str):
        normalized = value.strip()
        if normalized.isdecimal() and int(normalized) in targets:
            output.append({"json_path": path, "seed": int(normalized), "encoded_type": "string"})
        elif not (allow_identifier and normalized and not normalized.isdecimal()):
            malformed.append({"json_path": path, "observed_type": "string"})
    elif isinstance(value, list):
        if not value:
            malformed.append({"json_path": path, "observed_type": "empty_list"})
            return
        for index, child in enumerate(value):
            _matches_seed_value(
                child,
                path=f"{path}[{index}]",
                targets=targets,
                output=output,
                malformed=malformed,
                allow_identifier=allow_identifier,
            )
    elif value is None:
        malformed.append({"json_path": path, "observed_type": "null"})
    elif isinstance(value, dict):
        if not value:
            malformed.append({"json_path": path, "observed_type": "object"})
        for key in value:
            if key.isdecimal() and int(key) in targets:
                output.append(
                    {"json_path": f"{path}::{key}", "seed": int(key), "encoded_type": "key"}
                )


def _matches_seed_range(
    value: Any,
    *,
    path: str,
    targets: set[int],
    output: list[dict[str, Any]],
    malformed: list[dict[str, str]],
) -> None:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value)
        or value[0] > value[1]
    ):
        malformed.append({"json_path": path, "observed_type": "range"})
        return
    lower, upper = value
    for target in sorted(targets):
        if lower <= target <= upper:
            output.append({"json_path": path, "seed": target, "encoded_type": "range"})


def _corpus_files(root: Path) -> list[Path]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"corpus root must be a real directory: {root}")
    result: list[Path] = []
    for current, directories, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        for directory in directories:
            if (current_path / directory).is_symlink():
                raise ValueError(f"corpus contains a symlink directory: {current_path / directory}")
        for filename in filenames:
            candidate = current_path / filename
            if candidate.suffix == ".json" or filename.endswith(".json.gz"):
                if candidate.is_symlink():
                    raise ValueError(f"corpus contains a symlink result file: {candidate}")
                result.append(candidate)
    return sorted(result)


def audit_seed_roster_collisions(roots: list[Path], reserved_seeds: list[int]) -> dict[str, Any]:
    """Scan JSON/JSON.GZ result artifacts for exact integer values under seed keys."""
    if not roots:
        raise ValueError("at least one corpus root is required")
    if not reserved_seeds:
        raise ValueError("reserved seed roster must be nonempty")
    if any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in reserved_seeds
    ):
        raise ValueError("reserved seeds must be nonnegative integers")
    if len(set(reserved_seeds)) != len(reserved_seeds):
        raise ValueError("reserved seed roster must not contain duplicates")

    target_set = set(reserved_seeds)
    scanned: list[dict[str, Any]] = []
    collisions: list[dict[str, Any]] = []
    malformed_fields: list[dict[str, str]] = []
    incomplete_files: list[dict[str, str]] = []
    seen_files: set[Path] = set()
    root_summaries: list[dict[str, Any]] = []
    for root in roots:
        root_files = _corpus_files(root)
        root_summaries.append(
            {
                "path": str(root.absolute()),
                "eligible_json_file_count": len(root_files),
            }
        )
        for path in root_files:
            absolute_path = path.absolute()
            if absolute_path in seen_files:
                continue
            seen_files.add(absolute_path)
            try:
                raw = _read_file(path)
                document = _read_json(path, raw)
            except ScanLimitError as error:
                incomplete_files.append({"path": str(path), "reason": str(error).split(": ", 1)[0]})
                continue
            matches: list[dict[str, Any]] = []
            malformed: list[dict[str, str]] = []
            _matches(
                document,
                path="",
                targets=target_set,
                output=matches,
                malformed=malformed,
            )
            file_record = {
                "path": str(path),
                "byte_count": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "collision_count": len(matches),
            }
            scanned.append(file_record)
            if matches:
                collisions.append({**file_record, "matches": matches})
            malformed_fields.extend({"file_path": str(path), **item} for item in malformed)

    if not scanned and not incomplete_files:
        raise ValueError("no .json or .json.gz result files were found under the supplied roots")

    return {
        "schema": "loslab.seed_roster_collision_audit.v2",
        "status": (
            "INCOMPLETE_SCAN"
            if incomplete_files
            else "COLLISION_FOUND"
            if collisions
            else "MALFORMED_SEED_FIELDS"
            if malformed_fields
            else "CLEAR_WITHIN_SCANNED_JSON_CORPUS"
        ),
        "reserved_seeds": sorted(reserved_seeds),
        "seed_field_rule": (
            "integer values under seed-named fields, inclusive two-integer seed ranges, "
            "or exact reserved integers embedded in seed-named keys"
        ),
        "scan_scope": "regular single-link .json and .json.gz files under supplied roots",
        "roots": root_summaries,
        "scanned_file_count": len(scanned),
        "scanned_files": scanned,
        "incomplete_files": incomplete_files,
        "collisions": collisions,
        "malformed_seed_fields": malformed_fields,
        "global_exclusivity_established": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--seed", type=int, action="append", default=[])
    parser.add_argument(
        "--seed-range",
        type=int,
        nargs=2,
        action="append",
        metavar=("START", "STOP"),
        default=[],
        help="add every integer in the half-open interval [START, STOP)",
    )
    args = parser.parse_args(argv)
    seeds = set(args.seed)
    for start, stop in args.seed_range:
        if start < 0 or stop <= start:
            parser.error("--seed-range requires 0 <= START < STOP")
        seeds.update(range(start, stop))
    if not seeds:
        parser.error("provide at least one --seed or --seed-range")
    try:
        report = audit_seed_roster_collisions(args.root, sorted(seeds))
    except (
        OSError,
        UnicodeDecodeError,
        ValueError,
        TypeError,
        RuntimeError,
        RecursionError,
    ) as error:
        parser.exit(2, f"error: {error}\n")
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    if report["incomplete_files"]:
        return 2
    return 1 if report["collisions"] or report["malformed_seed_fields"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
