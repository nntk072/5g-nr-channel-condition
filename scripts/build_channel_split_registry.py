"""Build an immutable candidate split registry from saved ChannelDatasets."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import stat
import zipfile
from pathlib import Path
from typing import Any, Sequence

from artifacts.build_dag import BuildNodeSpec, resolve_build_artifact, run_build_node
from artifacts.provenance import package_source_sha256, versions
from datasets.channel import ChannelDataset
from datasets.legacy_rt_split import upgrade_legacy_rt_split_metadata
from datasets.split_lineage import make_channel_dataset_split_registry
from datasets.split_policy import (
    SplitPolicy,
    freeze_split_policy_registry,
)

_MAX_POLICY_BYTES = 1024 * 1024
_DEFAULT_MAX_DATASET_BYTES = 8 * 1024 * 1024 * 1024
_REGISTRY_BUILD_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": [
        "operation",
        "datasets",
        "policy_sha256",
        "spatial_buffer_m",
        "max_dataset_bytes",
        "runtime_versions",
    ],
    "properties": {
        "operation": {"const": "candidate_channel_split_registry_v1"},
        "datasets": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["name", "sha256"],
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                },
                "additionalProperties": False,
            },
        },
        "policy_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "spatial_buffer_m": {"type": "number", "minimum": 0},
        "max_dataset_bytes": {"type": "integer", "minimum": 1},
        "runtime_versions": {
            "type": "object",
            "additionalProperties": {"type": "string", "minLength": 1},
        },
    },
    "additionalProperties": False,
}
_REGISTRY_BUILD_OUTPUTS = ("build-report.json", "split-registry.json")
_RT_SPLIT_FIELDS = frozenset(
    {
        "split_subject_id",
        "split_parent_type",
        "split_source_family",
        "split_leakage_relations",
        "split_identity_basis",
        "split_identity_complete",
        "split_block_size_m",
        "split_coordinate_frame_id",
        "split_positions_m",
    }
)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _stable_file_bytes(path: Path, *, label: str, max_bytes: int) -> bytes:
    before_path = os.lstat(path)
    if not stat.S_ISREG(before_path.st_mode) or before_path.st_nlink != 1:
        raise ValueError(f"{label} must be a single-link regular file")
    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before_fd = os.fstat(descriptor)
        identity = _stat_identity(before_fd)
        if identity != _stat_identity(before_path):
            raise ValueError(f"{label} changed while opening")
        if before_fd.st_size > max_bytes:
            raise ValueError(f"{label} exceeds the {max_bytes}-byte limit")
        chunks: list[bytes] = []
        size = 0
        while True:
            request = min(1024 * 1024, max_bytes + 1 - size)
            if request <= 0:
                raise ValueError(f"{label} exceeds the {max_bytes}-byte limit")
            chunk = os.read(descriptor, request)
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
        after_fd = os.fstat(descriptor)
        after_path = os.lstat(path)
        if (
            size != before_fd.st_size
            or _stat_identity(after_fd) != identity
            or _stat_identity(after_path) != identity
        ):
            raise ValueError(f"{label} changed while reading")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _stable_file_sha256(path: Path, *, label: str, max_bytes: int) -> tuple[str, int]:
    """Hash an input with bounded memory and the same stable-file contract."""
    before_path = os.lstat(path)
    if not stat.S_ISREG(before_path.st_mode) or before_path.st_nlink != 1:
        raise ValueError(f"{label} must be a single-link regular file")
    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    size = 0
    try:
        before_fd = os.fstat(descriptor)
        if _stat_identity(before_fd) != _stat_identity(before_path):
            raise ValueError(f"{label} changed while opening")
        if before_fd.st_size > max_bytes:
            raise ValueError(f"{label} exceeds the {max_bytes}-byte limit")
        while chunk := os.read(descriptor, 1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                raise ValueError(f"{label} exceeds the {max_bytes}-byte limit")
            digest.update(chunk)
        after_fd = os.fstat(descriptor)
        after_path = os.lstat(path)
        identity = _stat_identity(before_fd)
        if (
            size != before_fd.st_size
            or _stat_identity(after_fd) != identity
            or _stat_identity(after_path) != identity
        ):
            raise ValueError(f"{label} changed while hashing")
    finally:
        os.close(descriptor)
    return digest.hexdigest(), size


def _stat_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _validate_npz_size(payload: bytes, *, label: str, max_bytes: int) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            entries = archive.infolist()
            if len(entries) > 32:
                raise ValueError(f"{label} contains too many NPZ entries")
            names = [entry.filename for entry in entries]
            if len(names) != len(set(names)):
                raise ValueError(f"{label} contains duplicate NPZ entries")
            if sum(entry.file_size for entry in entries) > max_bytes:
                raise ValueError(f"{label} exceeds the {max_bytes}-byte uncompressed limit")
    except zipfile.BadZipFile as error:
        raise ValueError(f"{label} is not a valid NPZ archive") from error


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _load_policy(path: Path) -> tuple[SplitPolicy, str]:
    payload = _stable_file_bytes(path, label="split policy", max_bytes=_MAX_POLICY_BYTES)
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise ValueError("split policy must be strict UTF-8 JSON without duplicate keys") from error
    if not isinstance(value, dict):
        raise ValueError("split policy must be a JSON object")
    return SplitPolicy.from_dict(value), hashlib.sha256(payload).hexdigest()


def build_candidate_split_registry(
    datasets: Sequence[tuple[str, Path]],
    *,
    policy_path: Path,
    spatial_buffer_m: float,
    registry_path: Path,
    report_path: Path,
    max_dataset_bytes: int = _DEFAULT_MAX_DATASET_BYTES,
    build_cache_root: Path | None = None,
) -> dict[str, Any]:
    """Build and freeze a candidate registry while binding exact input bytes.

    The registry is an immutable assignment artifact, not an externally
    approved release lock. The report binds its registry digest to the exact
    policy and ChannelDataset bytes used to derive the assignments.
    """
    if build_cache_root is not None:
        return _build_candidate_split_registry_cached(
            datasets,
            policy_path=policy_path,
            spatial_buffer_m=spatial_buffer_m,
            registry_path=registry_path,
            report_path=report_path,
            max_dataset_bytes=max_dataset_bytes,
            build_cache_root=Path(build_cache_root),
        )
    if not datasets:
        raise ValueError("at least one named ChannelDataset input is required")
    if isinstance(max_dataset_bytes, bool) or not isinstance(max_dataset_bytes, int):
        raise TypeError("max_dataset_bytes must be an integer")
    if max_dataset_bytes <= 0:
        raise ValueError("max_dataset_bytes must be positive")
    normalized: list[tuple[str, Path]] = []
    seen_names: set[str] = set()
    seen_paths: set[Path] = set()
    for item in datasets:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError("datasets must contain (name, path) pairs")
        name, path = item
        if not isinstance(name, str) or not name.strip() or name.strip() != name:
            raise ValueError("dataset names must be nonempty canonical strings")
        if name in seen_names:
            raise ValueError(f"duplicate dataset name: {name}")
        resolved_path = Path(path).expanduser().absolute()
        if resolved_path in seen_paths:
            raise ValueError(f"duplicate dataset path: {resolved_path}")
        seen_names.add(name)
        seen_paths.add(resolved_path)
        normalized.append((name, resolved_path))

    policy_path = Path(policy_path).expanduser().absolute()
    registry_path = Path(registry_path).expanduser().absolute()
    report_path = Path(report_path).expanduser().absolute()
    if registry_path == report_path:
        raise ValueError("registry and build report paths must be different")
    if os.path.lexists(registry_path) or os.path.lexists(report_path):
        raise FileExistsError("refusing to overwrite a registry or build report")
    policy, policy_file_sha256 = _load_policy(policy_path)

    loaded: list[tuple[str, ChannelDataset]] = []
    input_records: list[dict[str, Any]] = []
    for name, path in sorted(normalized, key=lambda item: item[0]):
        payload = _stable_file_bytes(
            path, label=f"ChannelDataset {name!r}", max_bytes=max_dataset_bytes
        )
        _validate_npz_size(payload, label=f"ChannelDataset {name!r}", max_bytes=max_dataset_bytes)
        dataset = ChannelDataset.load_bytes(payload)
        is_rt_only = all(str(source) == "sionna_rt" for source in dataset.source)
        has_split_fields = any(
            _RT_SPLIT_FIELDS.intersection(metadata) for metadata in dataset.metadata
        )
        migrated_legacy_rt = False
        if is_rt_only and not has_split_fields:
            dataset, migrated_legacy_rt = upgrade_legacy_rt_split_metadata(
                dataset, dataset_name=name
            )
        loaded.append((name, dataset))
        input_records.append(
            {
                "name": name,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "row_count": len(dataset.y),
                "split_metadata_migration": (
                    "legacy_rt_complete_scene_identity_v1" if migrated_legacy_rt else None
                ),
            }
        )

    registry, lineage_report = make_channel_dataset_split_registry(
        loaded, spatial_buffer_m=spatial_buffer_m, policy=policy
    )
    registry_dict = registry.to_dict()
    registry_file_bytes = (json.dumps(registry_dict, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    report: dict[str, Any] = {
        "schema": "umcd.channel_split_registry_build_report.v2",
        "status": "CANDIDATE_NOT_EXTERNALLY_REVIEWED",
        "qualification_status": "NOT_QUALIFIED",
        "registry_sha256": registry_dict["registry_sha256"],
        "registry_file_sha256": hashlib.sha256(registry_file_bytes).hexdigest(),
        "policy_sha256": policy.to_dict()["policy_sha256"],
        "policy_file_sha256": policy_file_sha256,
        "spatial_buffer_m": float(spatial_buffer_m),
        "inputs": input_records,
        "lineage": lineage_report,
    }
    report_bytes = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    _write_exclusive(report_path, report_bytes)
    try:
        lock_sha256 = freeze_split_policy_registry(registry_path, registry)
    except BaseException:
        report_path.unlink(missing_ok=True)
        raise
    if lock_sha256 != report["registry_file_sha256"]:
        raise RuntimeError("frozen split registry digest differs from its canonical bytes")
    return {**report, "registry_file_sha256": lock_sha256}


def _build_candidate_split_registry_cached(
    datasets: Sequence[tuple[str, Path]],
    *,
    policy_path: Path,
    spatial_buffer_m: float,
    registry_path: Path,
    report_path: Path,
    max_dataset_bytes: int,
    build_cache_root: Path,
) -> dict[str, Any]:
    """Build candidate split outputs through the immutable workflow DAG."""
    if not datasets:
        raise ValueError("at least one named ChannelDataset input is required")
    if (
        isinstance(spatial_buffer_m, bool)
        or not isinstance(spatial_buffer_m, (int, float))
        or not math.isfinite(spatial_buffer_m)
        or spatial_buffer_m < 0
    ):
        raise ValueError("spatial_buffer_m must be finite and nonnegative")
    if isinstance(max_dataset_bytes, bool) or not isinstance(max_dataset_bytes, int):
        raise TypeError("max_dataset_bytes must be an integer")
    if max_dataset_bytes <= 0:
        raise ValueError("max_dataset_bytes must be positive")

    normalized: list[tuple[str, Path]] = []
    seen_names: set[str] = set()
    seen_paths: set[Path] = set()
    for item in datasets:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError("datasets must contain (name, path) pairs")
        name, raw_path = item
        if not isinstance(name, str) or not name or name != name.strip():
            raise ValueError("dataset names must be nonempty canonical strings")
        path = Path(raw_path).expanduser().absolute()
        if name in seen_names:
            raise ValueError(f"duplicate dataset name: {name}")
        if path in seen_paths:
            raise ValueError(f"duplicate dataset path: {path}")
        seen_names.add(name)
        seen_paths.add(path)
        normalized.append((name, path))
    normalized.sort(key=lambda row: row[0])

    policy_path = Path(policy_path).expanduser().absolute()
    registry_path = Path(registry_path).expanduser().absolute()
    report_path = Path(report_path).expanduser().absolute()
    build_cache_root = Path(build_cache_root).expanduser().absolute()
    if registry_path == report_path:
        raise ValueError("registry and build report paths must be different")
    if os.path.lexists(registry_path) or os.path.lexists(report_path):
        raise FileExistsError("refusing to overwrite a registry or build report")
    if policy_path in seen_paths:
        raise ValueError("split policy path cannot also be a ChannelDataset input")

    policy_sha256, _policy_size = _stable_file_sha256(
        policy_path, label="split policy", max_bytes=_MAX_POLICY_BYTES
    )
    ordered_inputs = []
    input_files: dict[str, str] = {str(policy_path): policy_sha256}
    for name, path in normalized:
        digest, _size = _stable_file_sha256(
            path, label=f"ChannelDataset {name!r}", max_bytes=max_dataset_bytes
        )
        ordered_inputs.append({"name": name, "sha256": digest})
        input_files[str(path)] = digest

    runtime_versions = versions()
    configuration = {
        "operation": "candidate_channel_split_registry_v1",
        "datasets": ordered_inputs,
        "policy_sha256": policy_sha256,
        "spatial_buffer_m": float(spatial_buffer_m),
        "max_dataset_bytes": max_dataset_bytes,
        "runtime_versions": runtime_versions,
    }
    root = Path(__file__).resolve().parents[1]
    source_paths = (
        Path(__file__).resolve(),
        root / "artifacts/schema/build-node-manifest.v1.schema.json",
        root / "datasets/schema/channel-split-registry-build-report.v2.schema.json",
    )
    source_files = []
    for path in source_paths:
        digest, size = _stable_file_sha256(
            path, label="split-registry build source", max_bytes=16 * 1024 * 1024
        )
        source_files.append(
            {"path": path.relative_to(root).as_posix(), "sha256": digest, "size_bytes": size}
        )
    source_bundle_sha256 = hashlib.sha256(
        json.dumps(
            {
                "package_source_sha256": package_source_sha256(),
                "external_source_files": source_files,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    spec = BuildNodeSpec(
        node_id="split_policy.candidate_registry",
        node_version="1.0.0",
        configuration=configuration,
        configuration_schema=_REGISTRY_BUILD_SCHEMA,
        source_revision=runtime_versions.get("git_sha") or "unknown",
        source_bundle_sha256=source_bundle_sha256,
        input_files=input_files,
        dependencies={},
        outputs=_REGISTRY_BUILD_OUTPUTS,
    )

    def build(staging: Path) -> None:
        report = build_candidate_split_registry(
            normalized,
            policy_path=policy_path,
            spatial_buffer_m=spatial_buffer_m,
            registry_path=staging / "split-registry.json",
            report_path=staging / "build-report.json",
            max_dataset_bytes=max_dataset_bytes,
        )
        actual_inputs = [{"name": row["name"], "sha256": row["sha256"]} for row in report["inputs"]]
        if actual_inputs != ordered_inputs or report["policy_file_sha256"] != policy_sha256:
            raise ValueError("split-registry build inputs changed after DAG identity was frozen")

    result = run_build_node(build_cache_root, spec, build)

    def cached_output_bytes(name: str, *, maximum_bytes: int) -> bytes:
        artifact = next(row for row in result.artifacts if row["path"] == name)
        path = resolve_build_artifact(build_cache_root, result, name)
        payload = _stable_file_bytes(
            path, label=f"cached split-registry artifact {name!r}", max_bytes=maximum_bytes
        )
        if (
            len(payload) != artifact["size_bytes"]
            or hashlib.sha256(payload).hexdigest() != artifact["sha256"]
        ):
            raise ValueError("cached split-registry artifact changed after resolution")
        return payload

    report_bytes = cached_output_bytes("build-report.json", maximum_bytes=16 * 1024 * 1024)
    registry_bytes = cached_output_bytes("split-registry.json", maximum_bytes=256 * 1024 * 1024)
    report = json.loads(report_bytes.decode("utf-8"))
    if (
        not isinstance(report, dict)
        or report.get("schema") != "umcd.channel_split_registry_build_report.v2"
        or report.get("registry_file_sha256") != hashlib.sha256(registry_bytes).hexdigest()
        or report.get("inputs") is None
    ):
        raise ValueError("cached split-registry outputs do not match their report contract")
    import jsonschema

    report_schema = json.loads(
        (root / "datasets/schema/channel-split-registry-build-report.v2.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.validate(report, report_schema)

    report_published = False
    try:
        _write_exclusive(report_path, report_bytes)
        report_published = True
        _write_exclusive(registry_path, registry_bytes)
    except BaseException:
        if report_published:
            report_path.unlink(missing_ok=True)
        raise
    report["build_node"] = {
        "node_key": result.node_key,
        "manifest_sha256": hashlib.sha256(
            _stable_file_bytes(
                result.manifest_path,
                label="split-registry build-node manifest",
                max_bytes=16 * 1024 * 1024,
            )
        ).hexdigest(),
        "reused": result.reused,
        "cache_root": str(build_cache_root),
    }
    return report


def _parse_dataset(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("dataset must use NAME=PATH")
    return name, Path(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", type=_parse_dataset, required=True)
    parser.add_argument("--policy", type=Path, required=True, help="canonical split-policy v2 JSON")
    parser.add_argument("--spatial-buffer-m", type=float, required=True)
    parser.add_argument("--max-dataset-bytes", type=int, default=_DEFAULT_MAX_DATASET_BYTES)
    parser.add_argument("--registry-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument(
        "--build-cache-root",
        type=Path,
        help="optional immutable content-addressed build DAG root for reproducible reuse",
    )
    args = parser.parse_args(argv)
    try:
        report = build_candidate_split_registry(
            args.dataset,
            policy_path=args.policy,
            spatial_buffer_m=args.spatial_buffer_m,
            registry_path=args.registry_output,
            report_path=args.report_output,
            max_dataset_bytes=args.max_dataset_bytes,
            build_cache_root=args.build_cache_root,
        )
    except (OSError, RuntimeError, TypeError, ValueError, RecursionError) as error:
        parser.exit(2, f"error: {error}\n")
    summary = {
        "status": report["status"],
        "registry_sha256": report["registry_sha256"],
        "registry_file_sha256": report["registry_file_sha256"],
        "report": str(args.report_output),
    }
    if "build_node" in report:
        summary["build_node"] = report["build_node"]
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
