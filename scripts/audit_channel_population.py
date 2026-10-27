"""Compare ChannelDataset rows to a frozen expected-population outcome roster."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any

from datasets.channel import ChannelDataset
from evaluation.population_audit import (
    audit_expected_population,
    validate_expected_population_roster,
)

MAX_ROSTER_BYTES = 64 * 1024 * 1024
MAX_DATASET_BYTES = 8 * 1024 * 1024 * 1024


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_stable_regular_file(path: Path, *, maximum_bytes: int) -> bytes:
    before_path = os.lstat(path)
    if not stat.S_ISREG(before_path.st_mode) or before_path.st_nlink != 1:
        raise ValueError(f"input must be a single-link regular file: {path}")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"opened input is not a single-link regular file: {path}")
        if before.st_size > maximum_bytes:
            raise ValueError(f"input exceeds its byte limit: {path}")
        chunks = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum_bytes:
                raise ValueError(f"input exceeds its byte limit: {path}")
        after = os.fstat(descriptor)
        after_path = os.lstat(path)
    finally:
        os.close(descriptor)
    if _identity(before) != _identity(after) or _identity(before) != _identity(after_path):
        raise ValueError(f"input changed while it was being read: {path}")
    return b"".join(chunks)


def _validate_json(instance: Any, schema_path: Path, label: str) -> None:
    import jsonschema

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(instance))
    if errors:
        raise ValueError(
            f"{label} violates schema: " + "; ".join(error.message for error in errors[:10])
        )


def _publish_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def audit_files(roster_path: Path, dataset_paths: list[Path]) -> dict[str, Any]:
    roster_bytes = _read_stable_regular_file(roster_path, maximum_bytes=MAX_ROSTER_BYTES)
    try:
        roster_raw = json.loads(roster_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("expected-population roster must be UTF-8 JSON") from error
    _validate_json(
        roster_raw,
        Path(__file__).resolve().parents[1]
        / "datasets/schema/expected-population-roster.v1.schema.json",
        "expected-population roster",
    )
    roster = validate_expected_population_roster(roster_raw)

    datasets = []
    source_artifacts = []
    total_bytes = 0
    for source_path in dataset_paths:
        payload = _read_stable_regular_file(source_path, maximum_bytes=MAX_DATASET_BYTES)
        total_bytes += len(payload)
        if total_bytes > MAX_DATASET_BYTES:
            raise ValueError("combined ChannelDataset inputs exceed the byte limit")
        datasets.append(ChannelDataset.load_bytes(payload))
        source_artifacts.append(
            {
                "path": str(source_path.resolve()),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "byte_count": len(payload),
            }
        )
    data = ChannelDataset.concatenate(datasets)
    sample_ids = []
    for index, metadata in enumerate(data.metadata):
        sample_id = metadata.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id != sample_id.strip():
            raise ValueError(f"dataset row {index} lacks a canonical metadata sample_id")
        sample_ids.append(sample_id)
    report = audit_expected_population(
        roster,
        observed_sample_ids=sample_ids,
        observed_family_ids=data.source.astype(str),
        observed_domain_ids=data.domain.astype(str),
        observed_group_ids=data.group.astype(str),
    )
    report["roster_source"] = {
        "path": str(roster_path.resolve()),
        "sha256": hashlib.sha256(roster_bytes).hexdigest(),
        "byte_count": len(roster_bytes),
    }
    report["dataset_inputs"] = source_artifacts
    _validate_json(
        report,
        Path(__file__).resolve().parents[1]
        / "evaluation/schema/expected-population-audit.v1.schema.json",
        "expected-population audit report",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roster", type=Path, required=True, help="Frozen expected-population JSON"
    )
    parser.add_argument(
        "--input", type=Path, nargs="+", required=True, help="ChannelDataset NPZ files"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New audit JSON; never overwritten"
    )
    args = parser.parse_args()
    report = audit_files(args.roster, args.input)
    _publish_json_exclusive(args.output, report)
    print(json.dumps({"status": report["status"], "roster_id": report["roster_id"]}))
    if report["status"] != "ROSTER_MATCH":
        raise SystemExit(2)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, TypeError) as error:
        print(f"population audit failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
