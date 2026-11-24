"""Bind frozen child rosters to a candidate split registry without changing parents."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jsonschema
from artifacts.provenance import atomic_json, package_source_sha256, versions
from datasets.split_policy import (
    SplitChildBinding,
    bind_split_children,
    freeze_split_policy_registry,
    read_split_policy_registry,
)

_MAX_INPUT_BYTES = 256 * 1024 * 1024


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _read_regular_bytes(path: Path, *, label: str) -> bytes:
    before = os.lstat(path)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ValueError(f"{label} must be a single-link regular file")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0))
    try:
        opened = os.fstat(descriptor)
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != identity:
            raise ValueError(f"{label} changed while opening")
        if opened.st_size > _MAX_INPUT_BYTES:
            raise ValueError(f"{label} exceeds the {_MAX_INPUT_BYTES}-byte limit")
        chunks: list[bytes] = []
        size = 0
        while chunk := os.read(descriptor, min(1024 * 1024, _MAX_INPUT_BYTES + 1 - size)):
            size += len(chunk)
            if size > _MAX_INPUT_BYTES:
                raise ValueError(f"{label} exceeds the {_MAX_INPUT_BYTES}-byte limit")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        path_after = os.lstat(path)
        if (
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != identity
            or (path_after.st_dev, path_after.st_ino, path_after.st_size, path_after.st_mtime_ns)
            != identity
            or size != before.st_size
        ):
            raise ValueError(f"{label} changed while reading")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _load_roster(path: Path) -> tuple[tuple[SplitChildBinding, ...], dict[str, Any]]:
    path = path.expanduser().absolute()
    payload = _read_regular_bytes(path, label="split child roster")
    try:
        document = json.loads(payload, object_pairs_hook=_unique_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("split child roster is not valid bounded UTF-8 JSON") from error
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "datasets/schema/channel-dataset-child-roster.v1.schema.json"
    )
    jsonschema.validate(document, json.loads(schema_path.read_text(encoding="utf-8")))
    bindings = tuple(
        SplitChildBinding(
            child_id=item["child_id"],
            parent_subject_id=item["parent_subject_id"],
            child_kind=item["child_kind"],
        )
        for item in document["bindings"]
    )
    if len({item.child_id for item in bindings}) != len(bindings):
        raise ValueError("split child roster contains duplicate child IDs")
    if any(item.child_kind != document["child_kind"] for item in bindings):
        raise ValueError("split child roster entries disagree with its child_kind")
    source = {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "child_kind": document["child_kind"],
        "binding_count": len(bindings),
    }
    return bindings, source


def bind_registry_children(
    registry_path: Path,
    child_rosters: Sequence[Path],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Bind exact child roster snapshots and atomically publish a new candidate lock."""
    if not child_rosters:
        raise ValueError("at least one --child-roster input is required")
    registry_path = registry_path.expanduser().absolute()
    output_dir = output_dir.expanduser().absolute()
    if os.path.lexists(output_dir):
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")

    registry_bytes = _read_regular_bytes(registry_path, label="parent split registry")
    registry = read_split_policy_registry(registry_path)
    if _read_regular_bytes(registry_path, label="parent split registry") != registry_bytes:
        raise ValueError("parent split registry changed while loading")
    loaded = [_load_roster(path) for path in child_rosters]
    roster_paths = [item[1]["path"] for item in loaded]
    if len(set(roster_paths)) != len(roster_paths):
        raise ValueError("child roster paths must be unique")
    bindings = tuple(binding for roster, _ in loaded for binding in roster)
    if len({item.child_id for item in bindings}) != len(bindings):
        raise ValueError("child roster inputs contain duplicate child IDs")
    bound = bind_split_children(registry, bindings)
    registry_bytes_out = (json.dumps(bound.to_dict(), indent=2, sort_keys=True) + "\n").encode()
    runtime_versions = versions()
    report = {
        "schema": "umcd.split_registry_child_binding_report.v1",
        "status": "CANDIDATE_NOT_EXTERNALLY_REVIEWED",
        "qualification_status": "NOT_QUALIFIED",
        "created_utc": datetime.now(UTC).isoformat(),
        "parent_registry_path": str(registry_path),
        "parent_registry_sha256": hashlib.sha256(registry_bytes).hexdigest(),
        "policy_sha256": registry.policy.to_dict()["policy_sha256"],
        "child_rosters": [source for _, source in loaded],
        "child_binding_count": len(bindings),
        "bound_registry_sha256": bound.to_dict()["registry_sha256"],
        "bound_registry_file_sha256": hashlib.sha256(registry_bytes_out).hexdigest(),
        "source_identity": {
            "git_revision": runtime_versions.get("git_sha", "unavailable"),
            "package_source_sha256": package_source_sha256(),
            "source_module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "runtime_versions": runtime_versions,
        },
        "command_argv": list(sys.argv) or ["bind_split_registry_children"],
    }
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "datasets/schema/split-registry-child-binding-report.v1.schema.json"
    )
    jsonschema.validate(report, json.loads(schema_path.read_text(encoding="utf-8")))

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        output_registry = staging / "split-registry.json"
        digest = freeze_split_policy_registry(output_registry, bound)
        if digest != report["bound_registry_file_sha256"]:
            raise RuntimeError("bound registry bytes differ from the report identity")
        atomic_json(staging / "binding-report.json", report, overwrite=False)
        os.rename(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--child-roster", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = bind_registry_children(
            args.registry,
            args.child_roster,
            output_dir=args.output_dir,
        )
    except (OSError, RuntimeError, TypeError, ValueError, RecursionError) as error:
        parser.exit(2, f"error: {error}\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "qualification_status": report["qualification_status"],
                "bound_registry_sha256": report["bound_registry_sha256"],
                "child_binding_count": report["child_binding_count"],
                "output_dir": str(args.output_dir),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
