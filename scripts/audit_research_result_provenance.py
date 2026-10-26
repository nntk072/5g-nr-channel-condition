"""Read-only audit of provenance fields in research JSON result files.

Legacy records are reported as incomplete rather than rewritten or rejected by
default. --strict is intended for CI and exits nonzero if any issue exists.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import stat
import sys
import zlib
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse
from urllib.request import url2pathname

_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_COMMAND_KEYS = ("command", "command_line", "invocation", "argv")
_TIMESTAMP_KEYS = (
    "generated_at_utc",
    "started_utc",
    "started_at_utc",
    "created_utc",
    "created_at_utc",
    "timestamp_utc",
    "generated_at",
    "started_at",
    "created_at",
    "timestamp",
)
_SOURCE_HASH_KEYS = (
    "source_sha256",
    "source_hash",
    "source_code_sha256",
    "package_source_sha256",
    "runner_sha256",
    "script_sha256",
)
_CONFIG_HASH_KEYS = ("config_sha256", "config_hash", "configuration_sha256")
_INPUT_KEYS = ("inputs", "input_files", "input_artifacts")
_OUTPUT_KEYS = ("outputs", "output_files", "output_artifacts")
_PATH_KEYS = ("path", "file", "filename", "uri", "location")
_PATH_SUFFIXES = ("_path", "_file", "_filename", "_uri", "_location")


class DuplicateJSONKeyError(ValueError):
    """Raised when a JSON object contains ambiguous duplicate keys."""


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJSONKeyError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _first_present(record: dict[str, Any], keys: Iterable[str]) -> tuple[str, Any] | None:
    for key in keys:
        if key in record:
            return key, record[key]
    return None


def _valid_command(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(part, str) and bool(part.strip()) for part in value)
    )


def _valid_hash(value: Any) -> bool:
    if isinstance(value, str):
        return bool(_SHA256.fullmatch(value))
    return (
        isinstance(value, dict)
        and bool(value)
        and all(
            isinstance(name, str)
            and bool(name.strip())
            and isinstance(digest, str)
            and bool(_SHA256.fullmatch(digest))
            for name, digest in value.items()
        )
    )


def _is_utc_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    candidate = value.strip()
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return False
    offset = parsed.utcoffset()
    return parsed.tzinfo is not None and offset is not None and offset.total_seconds() == 0


def _looks_like_path(value: str) -> bool:
    parsed = urlparse(value)
    return bool(parsed.scheme or "/" in value or "\\" in value or Path(value).suffix)


def _reference_values(
    value: Any, label: str, *, strings_are_paths: bool = False
) -> list[tuple[str, str]]:
    """Extract (display name, path) references from declared input/output fields."""
    if isinstance(value, str):
        return [(label, value)] if strings_are_paths or _looks_like_path(value) else []
    if isinstance(value, list):
        references: list[tuple[str, str]] = []
        for index, item in enumerate(value):
            references.extend(
                _reference_values(item, f"{label}[{index}]", strings_are_paths=strings_are_paths)
            )
        return references
    if isinstance(value, dict):
        found: list[tuple[str, str]] = []
        for key, candidate in value.items():
            if key in _PATH_KEYS or key.endswith(_PATH_SUFFIXES):
                if isinstance(candidate, str):
                    found.append((f"{label}.{key}", candidate))
        for key, item in value.items():
            if (
                key not in _PATH_KEYS
                and not key.endswith(_PATH_SUFFIXES)
                and isinstance(item, (list, dict))
            ):
                found.extend(
                    _reference_values(item, f"{label}.{key}", strings_are_paths=strings_are_paths)
                )
        return found
    return []


def _reference_hashes(value: Any, label: str) -> dict[str, Any]:
    """Return declared digests paired with their path fields, when present."""
    found: dict[str, Any] = {}
    if isinstance(value, list):
        for index, item in enumerate(value):
            found.update(_reference_hashes(item, f"{label}[{index}]"))
    elif isinstance(value, dict):
        for key in value:
            if key not in _PATH_KEYS and not key.endswith(_PATH_SUFFIXES):
                continue
            stem = key
            for suffix in _PATH_SUFFIXES:
                if stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
                    break
            candidates = (
                (f"{stem}_sha256", f"{stem}_hash", f"{stem}_checksum_sha256")
                if stem
                else ("sha256", "hash", "checksum_sha256")
            )
            for digest_key in candidates:
                if digest_key in value:
                    found[f"{label}.{key}"] = value[digest_key]
                    break
        for key, item in value.items():
            if (
                key not in _PATH_KEYS
                and not key.endswith(_PATH_SUFFIXES)
                and isinstance(item, (list, dict))
            ):
                found.update(_reference_hashes(item, f"{label}.{key}"))
    return found


def _file_signature(
    metadata: os.stat_result,
) -> tuple[int, int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _sha256_file(path: Path) -> str:
    """Hash one stable regular file without following a leaf symlink."""
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode):
        raise OSError("referenced artifact is a symlink")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        current = path.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_ISLNK(current.st_mode)
            or _file_signature(opened) != _file_signature(before)
            or _file_signature(opened) != _file_signature(current)
        ):
            raise OSError("referenced artifact changed while opening or is not a regular file")
        byte_count = 0
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                byte_count += len(chunk)
            opened_after = os.fstat(handle.fileno())
            current_after = path.lstat()
            if (
                stat.S_ISLNK(current_after.st_mode)
                or _file_signature(opened_after) != _file_signature(opened)
                or _file_signature(opened_after) != _file_signature(current_after)
                or byte_count != opened.st_size
            ):
                raise OSError("referenced artifact changed while hashing")
            return digest.hexdigest()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _has_invalid_reference(value: Any) -> bool:
    if isinstance(value, str):
        return False
    if isinstance(value, list):
        return any(_has_invalid_reference(item) for item in value)
    if isinstance(value, dict):
        for key, item in value.items():
            if key in _PATH_KEYS or key.endswith(_PATH_SUFFIXES):
                if not isinstance(item, str) or not item.strip():
                    return True
        return any(
            _has_invalid_reference(item)
            for key, item in value.items()
            if key not in _PATH_KEYS
            and not key.endswith(_PATH_SUFFIXES)
            and isinstance(item, (list, dict))
        )
    return value is not None and not isinstance(value, str)


def _resolve_reference(
    reference: str, result_file: Path, root: Path, repository_root: Path
) -> Path | None:
    parsed = urlparse(reference)
    if parsed.scheme and parsed.scheme != "file":
        return None
    candidate = Path(
        url2pathname(parsed.path) if parsed.scheme == "file" else reference
    ).expanduser()
    if candidate.is_absolute():
        return candidate
    for base in (result_file.parent, root, repository_root):
        local = base / candidate
        if local.exists():
            return local
    return repository_root / candidate


def audit_record(
    record: Any, result_file: Path, root: Path, repository_root: Path | None = None
) -> list[dict[str, str]]:
    """Return provenance issues for one decoded result without changing it."""
    issues: list[dict[str, str]] = []
    repository_root = repository_root or root
    if not isinstance(record, dict):
        return [
            {
                "code": "invalid_record",
                "field": "$",
                "message": "top-level JSON value must be an object",
            }
        ]

    command = _first_present(record, _COMMAND_KEYS)
    if command is None:
        issues.append(
            {
                "code": "missing_command",
                "field": "command",
                "message": "no command/invocation field",
            }
        )
    elif not _valid_command(command[1]):
        issues.append(
            {
                "code": "invalid_command",
                "field": command[0],
                "message": "expected a nonempty command string or argv list",
            }
        )

    timestamp = _first_present(record, _TIMESTAMP_KEYS)
    if timestamp is None:
        issues.append(
            {
                "code": "missing_utc_timestamp",
                "field": "timestamp",
                "message": "no recognized UTC timestamp field",
            }
        )
    elif not _is_utc_timestamp(timestamp[1]):
        issues.append(
            {
                "code": "invalid_utc_timestamp",
                "field": timestamp[0],
                "message": "expected an ISO-8601 timestamp with UTC timezone",
            }
        )

    for label, keys in (("source", _SOURCE_HASH_KEYS), ("config", _CONFIG_HASH_KEYS)):
        item = _first_present(record, keys)
        if item is None:
            issues.append(
                {
                    "code": f"missing_{label}_hash",
                    "field": label,
                    "message": f"no recognized {label} SHA-256 field",
                }
            )
        elif not _valid_hash(item[1]):
            issues.append(
                {
                    "code": f"invalid_{label}_hash",
                    "field": item[0],
                    "message": "expected a SHA-256 string or a nonempty name-to-SHA-256 map",
                }
            )

    for kind, keys in (("input", _INPUT_KEYS), ("output", _OUTPUT_KEYS)):
        declared = _first_present(record, keys)
        if declared is None:
            continue
        references = _reference_values(
            declared[1],
            declared[0],
            strings_are_paths=declared[0]
            in {"input_files", "input_artifacts", "output_files", "output_artifacts"},
        )
        reference_hashes = _reference_hashes(declared[1], declared[0])
        if _has_invalid_reference(declared[1]):
            issues.append(
                {
                    "code": "invalid_reference",
                    "field": declared[0],
                    "message": "declared references contain no usable paths",
                }
            )
        for field, reference in references:
            if not reference.strip():
                issues.append(
                    {
                        "code": "invalid_reference",
                        "field": field,
                        "message": "reference path is empty",
                    }
                )
                continue
            resolved = _resolve_reference(reference, result_file, root, repository_root)
            if resolved is not None and not resolved.exists():
                issues.append(
                    {
                        "code": f"unresolved_{kind}_reference",
                        "field": field,
                        "message": f"referenced local artifact does not exist: {reference}",
                    }
                )
                continue
            if field not in reference_hashes:
                continue
            expected_hash = reference_hashes[field]
            if not isinstance(expected_hash, str) or not _SHA256.fullmatch(expected_hash):
                issues.append(
                    {
                        "code": "invalid_reference_hash",
                        "field": field,
                        "message": "associated artifact digest must be a SHA-256 hex string",
                    }
                )
                continue
            if resolved is None:
                continue
            try:
                actual_hash = _sha256_file(resolved)
            except OSError as error:
                issues.append(
                    {
                        "code": "unreadable_reference",
                        "field": field,
                        "message": f"cannot hash referenced artifact: {error}",
                    }
                )
                continue
            if actual_hash.lower() != expected_hash.lower():
                issues.append(
                    {
                        "code": "reference_hash_mismatch",
                        "field": field,
                        "message": "referenced artifact bytes do not match the declared SHA-256",
                    }
                )
    return issues


def _read_json(path: Path) -> Any:
    if path.name.endswith(".json.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle, object_pairs_hook=_reject_duplicate_json_keys)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=_reject_duplicate_json_keys)


def audit_tree(root: Path) -> dict[str, Any]:
    """Audit all .json and .json.gz files under root, read-only."""
    root = root.resolve()
    repository_root = next(
        (parent for parent in (root, *root.parents) if (parent / ".git").exists()), root
    )
    candidates = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and (path.name.endswith(".json") or path.name.endswith(".json.gz"))
    )
    files: list[dict[str, Any]] = []
    issue_counts: Counter[str] = Counter()
    malformed_count = 0
    valid_count = 0
    for path in candidates:
        try:
            record = _read_json(path)
        except (
            OSError,
            EOFError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            DuplicateJSONKeyError,
            gzip.BadGzipFile,
            zlib.error,
        ) as error:
            malformed_count += 1
            if isinstance(error, DuplicateJSONKeyError):
                code = "duplicate_json_key"
            else:
                code = "malformed_gzip" if path.name.endswith(".json.gz") else "malformed_json"
            issues = [{"code": code, "field": "$", "message": str(error)}]
        else:
            valid_count += 1
            issues = audit_record(record, path, root, repository_root)
        issue_counts.update(issue["code"] for issue in issues)
        files.append({"path": path.relative_to(root).as_posix(), "issues": issues})
    return {
        "root": str(root),
        "scanned_files": len(candidates),
        "valid_json_files": valid_count,
        "malformed_files": malformed_count,
        "files_with_issues": sum(bool(item["issues"]) for item in files),
        "issue_counts": dict(sorted(issue_counts.items())),
        "files": files,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path("docs/research/results"),
        help="directory to scan recursively (default: docs/research/results)",
    )
    parser.add_argument(
        "--strict", action="store_true", help="exit 1 if any file has a provenance issue"
    )
    args = parser.parse_args(argv)
    if not args.root.is_dir():
        parser.error(f"not a directory: {args.root}")
    report = audit_tree(args.root)
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return int(args.strict and report["files_with_issues"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
