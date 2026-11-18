"""Build a candidate release split lock from immutable standardized-channel runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

import jsonschema
from artifacts.provenance import atomic_json, package_source_sha256, versions
from datasets.split_lineage import make_standardized_run_split_registry
from datasets.split_policy import SplitPolicy, freeze_split_policy_registry
from datasets.standardized import StoredStandardizedRun, read_standardized_run

_MAX_INPUT_BYTES = 256 * 1024 * 1024


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _read_regular_bytes(path: Path, *, label: str, max_bytes: int = _MAX_INPUT_BYTES) -> bytes:
    before = os.lstat(path)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ValueError(f"{label} must be a single-link regular file")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0))
    try:
        opened = os.fstat(descriptor)
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != identity:
            raise ValueError(f"{label} changed while opening")
        if opened.st_size > max_bytes:
            raise ValueError(f"{label} exceeds the {max_bytes}-byte limit")
        chunks: list[bytes] = []
        size = 0
        while chunk := os.read(descriptor, min(1024 * 1024, max_bytes + 1 - size)):
            chunks.append(chunk)
            size += len(chunk)
            if size > max_bytes:
                raise ValueError(f"{label} exceeds the {max_bytes}-byte limit")
        after = os.fstat(descriptor)
        path_after = os.lstat(path)
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        path_identity_after = (
            path_after.st_dev,
            path_after.st_ino,
            path_after.st_size,
            path_after.st_mtime_ns,
        )
        if identity != identity_after or identity != path_identity_after or size != before.st_size:
            raise ValueError(f"{label} changed while reading")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _load_policy(path: Path) -> tuple[SplitPolicy, str]:
    payload = _read_regular_bytes(path, label="split policy")
    try:
        document = json.loads(payload, object_pairs_hook=_unique_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("split policy is not bounded canonical JSON") from error
    if not isinstance(document, dict):
        raise ValueError("split policy root must be an object")
    return SplitPolicy.from_dict(document), hashlib.sha256(payload).hexdigest()


def _load_run(name: str, manifest_path: Path) -> tuple[Any, dict[str, Any]]:
    if not name or name != name.strip():
        raise ValueError("run names must be canonical nonempty strings")
    manifest_path = manifest_path.expanduser().absolute()
    raw = _read_regular_bytes(manifest_path, label=f"standardized manifest {name!r}")
    try:
        document = json.loads(raw, object_pairs_hook=_unique_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError(f"standardized manifest {name!r} is not valid JSON") from error
    if not isinstance(document, dict) or set(document) != {
        "schema",
        "schema_version",
        "run_id",
        "records",
        "content_sha256",
    }:
        raise ValueError(f"standardized manifest {name!r} has an invalid field roster")
    if document["schema"] != "standardized_channel_run.v1" or document["schema_version"] != 1:
        raise ValueError(f"standardized manifest {name!r} has an unsupported schema")
    if not isinstance(document["records"], list) or not document["records"]:
        raise ValueError(f"standardized manifest {name!r} has no records")
    record_ids: list[str] = []
    for record in document["records"]:
        if not isinstance(record, dict) or set(record) != {
            "sample_id",
            "content_sha256",
            "relative_path",
        }:
            raise ValueError(f"standardized manifest {name!r} has an invalid record reference")
        record_ids.append(record["sample_id"])
    if not all(isinstance(item, str) and item for item in record_ids):
        raise ValueError(f"standardized manifest {name!r} has an invalid sample roster")

    store_root = manifest_path.parent.parent.parent
    relative_manifest = manifest_path.relative_to(store_root).as_posix()
    relative = PurePosixPath(relative_manifest)
    if relative.parts[:2] != ("manifests", "runs") or relative.suffix != ".json":
        raise ValueError("standardized run manifest must be under manifests/runs/*.json")
    pointer = StoredStandardizedRun(
        run_id=document["run_id"],
        content_sha256=document["content_sha256"],
        relative_path=relative_manifest,
        sample_ids=tuple(record_ids),
        schema_version=document["schema_version"],
    )
    run = read_standardized_run(store_root, pointer)
    if _read_regular_bytes(manifest_path, label=f"standardized manifest {name!r}") != raw:
        raise ValueError(f"standardized manifest {name!r} changed while loading its records")
    source = {
        "name": name,
        "run_id": pointer.run_id,
        "manifest_path": relative_manifest,
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "run_content_sha256": pointer.content_sha256,
        "draw_count": len(run.records),
        "record_sha256": [item["content_sha256"] for item in document["records"]],
    }
    return run, source


def build_standardized_split_registry(
    runs: Sequence[tuple[str, Path]],
    *,
    policy_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Create a no-overwrite registry/report directory from stored run manifests."""
    if not runs:
        raise ValueError("at least one --run input is required")
    output_dir = output_dir.expanduser().absolute()
    if os.path.lexists(output_dir):
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    policy_path = policy_path.expanduser().absolute()
    policy, policy_sha256 = _load_policy(policy_path)
    loaded = [_load_run(name, path) for name, path in runs]
    names = [source["name"] for _, source in loaded]
    if len(set(names)) != len(names):
        raise ValueError("standardized run input names must be unique")
    standardized_runs = [run for run, _ in loaded]
    registry, lineage = make_standardized_run_split_registry(standardized_runs, policy=policy)
    registry_bytes = (json.dumps(registry.to_dict(), indent=2, sort_keys=True) + "\n").encode()
    created_utc = datetime.now(timezone.utc).isoformat()
    runtime_versions = versions()
    source_module_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = {
        "schema": "umcd.standardized_split_registry_build_report.v1",
        "status": "CANDIDATE_NOT_EXTERNALLY_REVIEWED",
        "qualification_status": "NOT_QUALIFIED",
        "created_utc": created_utc,
        "registry_sha256": registry.to_dict()["registry_sha256"],
        "registry_file_sha256": hashlib.sha256(registry_bytes).hexdigest(),
        "policy_sha256": policy.to_dict()["policy_sha256"],
        "policy_file_sha256": policy_sha256,
        "inputs": [source for _, source in sorted(loaded, key=lambda item: item[1]["name"])],
        "lineage": lineage,
        "source_identity": {
            "git_revision": runtime_versions.get("git_sha", "unavailable"),
            "package_source_sha256": package_source_sha256(),
            "source_module_sha256": source_module_sha256,
            "runtime_versions": runtime_versions,
        },
        "command_argv": ["build_standardized_split_registry", *sys.argv[1:]],
    }
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "datasets/schema/standardized-split-registry-build-report.v1.schema.json"
    )
    jsonschema.validate(report, json.loads(schema_path.read_text(encoding="utf-8")))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        lock_path = staging / "split-registry.json"
        lock_digest = freeze_split_policy_registry(lock_path, registry)
        if lock_digest != report["registry_file_sha256"]:
            raise RuntimeError("frozen registry bytes differ from the report identity")
        atomic_json(staging / "build-report.json", report, overwrite=False)
        os.rename(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report


def _parse_run(value: str) -> tuple[str, Path]:
    name, separator, manifest = value.partition("=")
    if not separator or not name or not manifest:
        raise argparse.ArgumentTypeError("run must use NAME=/STORE_ROOT/manifests/runs/RUN.json")
    return name, Path(manifest)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=_parse_run, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = build_standardized_split_registry(
            args.run,
            policy_path=args.policy,
            output_dir=args.output_dir,
        )
    except (OSError, RuntimeError, TypeError, ValueError, RecursionError) as error:
        parser.exit(2, f"error: {error}\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "registry_sha256": report["registry_sha256"],
                "output_dir": str(args.output_dir),
                "qualification_status": report["qualification_status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
