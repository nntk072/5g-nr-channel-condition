"""Verify exact bytes for externally downloaded G4 normative reference files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

_MANIFEST_SCHEMA = "loslab.g4_standard_reference_manifest.v1"
_AUDIT_SCHEMA = "loslab.g4_standard_reference_audit.v1"
_MAX_MANIFEST_BYTES = 1 * 1024 * 1024
_MAX_REFERENCE_BYTES = 32 * 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
_SOURCE_FIELDS = {
    "source_id",
    "standard",
    "version",
    "filename",
    "source_url",
    "sha256",
    "size_bytes",
    "reference_scope",
    "limitations",
}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


def _load_manifest(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = _read_stable_regular_file(path, _MAX_MANIFEST_BYTES, "G4 reference manifest")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("G4 reference manifest is not valid bounded UTF-8 JSON") from error
    _validate_manifest(value)
    return value, raw


def _read_stable_regular_file(path: Path, maximum_bytes: int, label: str) -> bytes:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_NONBLOCK"):
        raise RuntimeError("platform lacks no-follow regular-file opening required for audit")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"{label} must be a single-link regular file")
        if before.st_size > maximum_bytes:
            raise ValueError(f"{label} exceeds the {maximum_bytes}-byte limit")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - size))
            if not chunk:
                break
            size += len(chunk)
            if size > maximum_bytes:
                raise ValueError(f"{label} exceeds the {maximum_bytes}-byte limit")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
        if _file_identity(before) != _file_identity(after) or _file_identity(
            after
        ) != _file_identity(current):
            raise ValueError(f"{label} changed while it was being read")
        if size != after.st_size:
            raise ValueError(f"{label} byte count changed while it was being read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _validate_manifest(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != {
        "schema",
        "snapshot_date",
        "redistribution_policy",
        "sources",
        "status",
    }:
        raise ValueError("G4 reference manifest has an unsupported field roster")
    if value["schema"] != _MANIFEST_SCHEMA:
        raise ValueError("unsupported G4 reference manifest schema")
    if not isinstance(value["snapshot_date"], str):
        raise ValueError("snapshot_date must use YYYY-MM-DD")
    try:
        date.fromisoformat(value["snapshot_date"])
    except ValueError as error:
        raise ValueError(
            "snapshot_date must be a real calendar date in YYYY-MM-DD form"
        ) from error
    if value["redistribution_policy"] != "reference_identity_only; do_not_redistribute_etsi_pdf":
        raise ValueError("reference manifest must prohibit unlicensed PDF redistribution")
    if value["status"] != "SOURCE_IDENTITIES_PINNED_NOT_QUALIFIED":
        raise ValueError("reference manifest must remain explicitly nonqualifying")
    sources = value["sources"]
    if not isinstance(sources, list) or not sources:
        raise ValueError("sources must be a nonempty array")
    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    for index, source in enumerate(sources):
        if not isinstance(source, dict) or set(source) != _SOURCE_FIELDS:
            raise ValueError(f"source {index} has an unsupported field roster")
        source_id, filename = source["source_id"], source["filename"]
        if not isinstance(source_id, str) or not re.fullmatch(
            r"[a-z0-9][a-z0-9.-]+", source_id
        ):
            raise ValueError(f"source {index} has an invalid source_id")
        if source_id in seen_ids:
            raise ValueError(f"duplicate G4 source_id: {source_id}")
        seen_ids.add(source_id)
        if not isinstance(filename, str) or not re.fullmatch(
            r"[A-Za-z0-9._-]+\.pdf", filename
        ):
            raise ValueError(f"source {source_id} filename must be a PDF basename")
        if filename in seen_names:
            raise ValueError(f"duplicate G4 reference filename: {filename}")
        seen_names.add(filename)
        for field in ("standard", "limitations"):
            if not isinstance(source[field], str) or not source[field].strip():
                raise ValueError(f"source {source_id} {field} must be nonempty text")
        if not isinstance(source["version"], str) or not _VERSION_RE.fullmatch(
            source["version"]
        ):
            raise ValueError(f"source {source_id} version is invalid")
        url = source["source_url"]
        parsed_url = urlsplit(url) if isinstance(url, str) else None
        if (
            parsed_url is None
            or parsed_url.scheme != "https"
            or parsed_url.hostname != "www.etsi.org"
            or parsed_url.username is not None
            or parsed_url.password is not None
            or parsed_url.port is not None
        ):
            raise ValueError(f"source {source_id} must use an official ETSI HTTPS URL")
        if not isinstance(source["sha256"], str) or not _SHA256_RE.fullmatch(
            source["sha256"]
        ):
            raise ValueError(f"source {source_id} sha256 is invalid")
        size = source["size_bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= _MAX_REFERENCE_BYTES:
            raise ValueError(f"source {source_id} size_bytes exceeds the supported range")
        scope = source["reference_scope"]
        if not isinstance(scope, list) or not scope or any(
            not isinstance(item, str) or not item.strip() for item in scope
        ):
            raise ValueError(f"source {source_id} reference_scope must contain nonempty text")


def _file_identity(info: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _hash_reference(path: Path, expected_size: int) -> tuple[str, int]:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_NONBLOCK"):
        raise RuntimeError("platform lacks no-follow regular-file opening required for audit")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError("reference must be a single-link regular file")
        if before.st_size != expected_size or before.st_size > _MAX_REFERENCE_BYTES:
            raise ValueError("reference byte count differs from the manifest")
        digest_builder = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > _MAX_REFERENCE_BYTES:
                raise ValueError("reference exceeds the 32 MiB limit")
            digest_builder.update(chunk)
        after = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
        if _file_identity(before) != _file_identity(after) or _file_identity(
            after
        ) != _file_identity(current):
            raise ValueError("reference changed while it was being read")
    finally:
        os.close(descriptor)
    if size != expected_size:
        raise ValueError("reference byte count differs from the manifest")
    return digest_builder.hexdigest(), size


def audit_g4_standard_references(
    manifest_path: str | Path, reference_root: str | Path
) -> dict[str, Any]:
    """Verify local bytes against the external-reference identities in a manifest."""
    manifest, manifest_raw = _load_manifest(Path(manifest_path))
    root = Path(reference_root)
    if not root.is_dir():
        raise ValueError("reference_root must be an existing directory")
    results: list[dict[str, Any]] = []
    for source in manifest["sources"]:
        path = root / source["filename"]
        digest, size = _hash_reference(path, source["size_bytes"])
        if digest != source["sha256"]:
            raise ValueError(f"reference bytes do not match the manifest: {source['source_id']}")
        results.append(
            {
                "source_id": source["source_id"],
                "filename": source["filename"],
                "sha256": digest,
                "size_bytes": size,
                "status": "PASS_EXACT_REFERENCE_BYTES",
            }
        )
    return {
        "schema": _AUDIT_SCHEMA,
        "status": "PASS_REFERENCE_SOURCE_BYTES_ONLY",
        "qualification_status": "NOT_QUALIFIED",
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "source_count": len(results),
        "sources": results,
        "interpretation": (
            "The local files match the pinned source bytes. This does not establish that extracted values, "
            "implemented models, statistical tests, or standardized conformance are correct."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="versioned G4 reference manifest JSON")
    parser.add_argument(
        "reference_root",
        type=Path,
        help="directory containing the externally downloaded PDF files named in the manifest",
    )
    args = parser.parse_args(argv)
    try:
        report = audit_g4_standard_references(args.manifest, args.reference_root)
    except (OSError, UnicodeDecodeError, ValueError, TypeError, RuntimeError, RecursionError) as error:
        parser.exit(2, f"error: {error}\n")
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
