"""Read-only audit of provenance fields in research JSON result files.

Legacy records are reported as incomplete rather than rewritten or rejected by
default. --strict is intended for CI and exits nonzero if any issue exists.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import zlib
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

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


def _reference_values(value: Any, label: str) -> list[tuple[str, str]]:
    """Extract (display name, path) references from declared input/output fields."""
    if isinstance(value, str):
        return [(label, value)] if _looks_like_path(value) else []
    if isinstance(value, list):
        references: list[tuple[str, str]] = []
        for index, item in enumerate(value):
            references.extend(_reference_values(item, f"{label}[{index}]"))
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
                found.extend(_reference_values(item, f"{label}.{key}"))
        return found
    return []


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
    candidate = Path(parsed.path if parsed.scheme == "file" else reference).expanduser()
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
        elif not isinstance(item[1], str) or not _SHA256.fullmatch(item[1]):
            issues.append(
                {
                    "code": f"invalid_{label}_hash",
                    "field": item[0],
                    "message": "expected 64 hexadecimal SHA-256 characters",
                }
            )

    for kind, keys in (("input", _INPUT_KEYS), ("output", _OUTPUT_KEYS)):
        declared = _first_present(record, keys)
        if declared is None:
            continue
        references = _reference_values(declared[1], declared[0])
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
    return issues


def _read_json(path: Path) -> Any:
    if path.name.endswith(".json.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
            gzip.BadGzipFile,
            zlib.error,
        ) as error:
            malformed_count += 1
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
