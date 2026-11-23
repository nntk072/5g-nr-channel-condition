"""Child artifact verification for the channel-split lineage audit."""

from __future__ import annotations

import argparse
import errno
import hashlib
import io
import json
import math
import os
import re
import stat
import zipfile
import zlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import jsonschema
import numpy as np
from datasets.channel import ChannelDataset
from datasets.split_policy import (
    SplitChildBinding,
    VerifiedOpenSSHArtifact,
)

_INVENTORY_SCHEMA = "umcd.channel_dataset_inventory.v1"


_LEGACY_ASSET_RECONSTRUCTION_SCHEMA = "umcd.legacy_rt_scene_asset_reconstruction.v1"


_INVENTORY_SIGNATURE_NAMESPACE = "umcd-channel-dataset-inventory-v1@github.com"


_CHILD_ARTIFACT_ROSTER_SIGNATURE_NAMESPACE = "umcd-channel-child-artifact-roster-v1@github.com"


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


_SPLIT_AUDIT_SCHEMA_PATH = (
    _REPOSITORY_ROOT / "datasets/schema/channel-split-lineage-audit-bundle.v1.schema.json"
)


_MAX_CHILD_ROSTER_BYTES = 16 * 1024 * 1024


_MAX_CHILD_METADATA_NPY_BYTES = 64 * 1024 * 1024


_MAX_CHILD_ARRAY_HEADER_BYTES = 64 * 1024


_MAX_CHILD_ARTIFACT_TREE_FILES = 100_000


_MAX_CHILD_ARTIFACT_TREE_DEPTH = 32


_V3_SEMANTIC_CHILD_ARTIFACT_TYPES = frozenset(
    {
        "channel_dataset_npz",
        "nr_trajectory_store_manifest_json",
        "nr_record_store_manifest_json",
        "nr_record_store_content_npz",
    }
)


_V4_SEMANTIC_CHILD_ARTIFACT_TYPES = _V3_SEMANTIC_CHILD_ARTIFACT_TYPES | {
    "umcd_episode_store_manifest_json",
    "umcd_episode_store_content_npz",
}


_V6_MULTI_OWNER_ARTIFACT_TYPES = frozenset(
    {
        "umcd_episode_store_content_npz",
        "scenario_gbsm_run_manifest_json",
        "scenario_gbsm_record_content_npz",
    }
)


_V6_SEMANTIC_CHILD_ARTIFACT_TYPES = _V4_SEMANTIC_CHILD_ARTIFACT_TYPES | {
    "scenario_gbsm_run_manifest_json",
    "scenario_gbsm_record_content_npz",
}


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _require_exact_child_object_owners(
    object_path: str,
    artifact_child_ids: Sequence[str],
    references: Sequence[tuple[str, str]],
    *,
    label: str,
) -> None:
    """Match every content-object manifest reference to the frozen owner roster."""
    owners = [
        child_id for child_id, referenced_path in references if referenced_path == object_path
    ]
    if len(owners) != len(set(owners)) or sorted(owners) != sorted(artifact_child_ids):
        raise ValueError(f"{label} manifest owners differ from the signed child roster")


def _require_shared_content_owners_share_split(
    artifact_type: str,
    owner_child_ids: Sequence[str],
    expected_child_bindings: Mapping[str, SplitChildBinding],
    split_registry: Any,
) -> None:
    """Reject content-addressed arrays whose frozen owners cross split boundaries."""
    if artifact_type not in {
        "umcd_episode_store_content_npz",
        "scenario_gbsm_record_content_npz",
    }:
        return
    try:
        owner_splits = {
            split_registry.split_for_parent(expected_child_bindings[owner].parent_subject_id)
            for owner in owner_child_ids
        }
    except KeyError as error:
        raise ValueError(
            f"shared {artifact_type} object owner is absent from the frozen split registry"
        ) from error
    if len(owner_splits) > 1:
        raise ValueError(
            f"shared {artifact_type} object has owners assigned to different frozen splits"
        )


def _stat_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_stable_regular_file(path: Path, *, label: str, max_bytes: int | None = None) -> bytes:
    """Read a stable regular file, optionally bounding memory before and during I/O."""
    if max_bytes is not None and (
        isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0
    ):
        raise ValueError("max_bytes must be a nonnegative integer or None")
    before = os.lstat(path)
    if stat.S_ISLNK(before.st_mode):
        raise ValueError(f"{label} must not be a symlink")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise ValueError(f"{label} must not be a symlink") from error
        raise
    try:
        opened = os.fstat(descriptor)
        current = os.lstat(path)
        if stat.S_ISLNK(current.st_mode):
            raise ValueError(f"{label} must not be a symlink")
        if (
            not stat.S_ISREG(opened.st_mode)
            or _stat_identity(opened) != _stat_identity(before)
            or _stat_identity(opened) != _stat_identity(current)
        ):
            raise ValueError(f"{label} changed while opening or is not a regular file")
        if max_bytes is not None and opened.st_size > max_bytes:
            raise ValueError(f"{label} exceeds {max_bytes} bytes")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            payload = stream.read() if max_bytes is None else stream.read(max_bytes + 1)
            if max_bytes is not None and len(payload) > max_bytes:
                raise ValueError(f"{label} exceeds {max_bytes} bytes")
            opened_after = os.fstat(stream.fileno())
            current_after = os.lstat(path)
            if (
                stat.S_ISLNK(current_after.st_mode)
                or _stat_identity(opened_after) != _stat_identity(opened)
                or _stat_identity(opened_after) != _stat_identity(current_after)
            ):
                raise ValueError(f"{label} changed while reading")
        return payload
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _load_split_child_roster(
    path: Path, *, expected_child_kind: str
) -> tuple[tuple[SplitChildBinding, ...], dict[str, Any]]:
    payload = _read_stable_regular_file(
        path, label="split child roster", max_bytes=_MAX_CHILD_ROSTER_BYTES
    )
    value = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_json_object)
    if (
        not isinstance(value, dict)
        or set(value) != {"schema", "child_kind", "bindings"}
        or value.get("schema") != "umcd.channel_dataset_child_roster.v1"
        or value.get("child_kind") != expected_child_kind
    ):
        raise ValueError("split child roster has an invalid schema or child kind")
    entries = value["bindings"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("split child roster bindings must be a nonempty array")
    bindings: list[SplitChildBinding] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != {
            "child_id",
            "parent_subject_id",
            "child_kind",
        }:
            raise ValueError(f"split child roster binding {index} has invalid fields")
        if entry["child_kind"] != expected_child_kind:
            raise ValueError(f"split child roster binding {index} has a different child kind")
        if any(
            not isinstance(entry[field], str)
            or not entry[field]
            or entry[field].strip() != entry[field]
            for field in ("child_id", "parent_subject_id", "child_kind")
        ):
            raise ValueError(f"split child roster binding {index} has noncanonical identifiers")
        bindings.append(
            SplitChildBinding(
                child_id=entry["child_id"],
                parent_subject_id=entry["parent_subject_id"],
                child_kind=entry["child_kind"],
            )
        )
    return tuple(bindings), {
        "path": str(Path(os.path.abspath(path))),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "child_kind": expected_child_kind,
    }


def _read_inventory_relative_file(root: Path, relative_path: str, *, label: str) -> bytes:
    """Read below an inventory directory using no-follow directory descriptors.

    Every path component is opened relative to an already-open directory. This
    makes the containment check survive directory replacement between path
    validation and file access, which pathname-only ``lstat`` checks cannot.
    """
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    if hasattr(os, "O_BINARY"):
        file_flags |= os.O_BINARY

    directory_descriptor = os.open(root, directory_flags)
    file_descriptor = -1
    try:
        components = relative_path.split("/")
        for component in components[:-1]:
            next_descriptor = os.open(component, directory_flags, dir_fd=directory_descriptor)
            if directory_descriptor != -1:
                os.close(directory_descriptor)
            directory_descriptor = next_descriptor

        file_descriptor = os.open(components[-1], file_flags, dir_fd=directory_descriptor)
        before = os.fstat(file_descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"{label} must be a single-link regular file")
        with os.fdopen(file_descriptor, "rb") as stream:
            file_descriptor = -1
            payload = stream.read()
            after = os.fstat(stream.fileno())
        if _stat_identity(before) != _stat_identity(after) or len(payload) != before.st_size:
            raise ValueError(f"{label} changed while reading")
        return payload
    except OSError as error:
        if error.errno in (errno.ELOOP, errno.ENOTDIR):
            raise ValueError(
                f"{label} path contains a symlink or non-directory component"
            ) from error
        raise
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)
        if directory_descriptor >= 0:
            os.close(directory_descriptor)


def _hash_inventory_relative_file(root: Path, relative_path: str, *, label: str) -> tuple[int, str]:
    """Hash a child artifact below root without loading its bytes into memory."""
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    if hasattr(os, "O_BINARY"):
        file_flags |= os.O_BINARY
    directory_descriptor = os.open(root, directory_flags)
    file_descriptor = -1
    try:
        components = relative_path.split("/")
        for component in components[:-1]:
            next_descriptor = os.open(component, directory_flags, dir_fd=directory_descriptor)
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor
        file_descriptor = os.open(components[-1], file_flags, dir_fd=directory_descriptor)
        before = os.fstat(file_descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"{label} must be a single-link regular file")
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(file_descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(file_descriptor)
        if _stat_identity(before) != _stat_identity(after) or size != before.st_size:
            raise ValueError(f"{label} changed while being hashed")
        return size, digest.hexdigest()
    except OSError as error:
        if error.errno in (errno.ELOOP, errno.ENOTDIR):
            raise ValueError(
                f"{label} path contains a symlink or non-directory component"
            ) from error
        raise
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)
        if directory_descriptor >= 0:
            os.close(directory_descriptor)


def _enumerate_regular_files_beneath(root: Path) -> set[str]:
    """Return every regular-file path below a no-symlink artifact directory."""
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    before = os.lstat(root)
    if not stat.S_ISDIR(before.st_mode):
        raise ValueError("child artifact inventory root must be a real directory")
    root_descriptor = os.open(root, directory_flags)
    files: set[str] = set()

    def walk(descriptor: int, prefix: str, depth: int) -> None:
        if depth > _MAX_CHILD_ARTIFACT_TREE_DEPTH:
            raise ValueError("child artifact inventory exceeds its directory-depth limit")
        for name in os.listdir(descriptor):
            if name in {".", ".."} or "/" in name or "\\" in name:
                raise ValueError("child artifact inventory contains an unsafe entry name")
            metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            relative = f"{prefix}/{name}" if prefix else name
            if stat.S_ISDIR(metadata.st_mode):
                nested_descriptor = os.open(name, directory_flags, dir_fd=descriptor)
                try:
                    opened = os.fstat(nested_descriptor)
                    if _stat_identity(opened) != _stat_identity(metadata):
                        raise ValueError("child artifact directory changed during enumeration")
                    walk(nested_descriptor, relative, depth + 1)
                    after = os.fstat(nested_descriptor)
                    path_after = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                    if _stat_identity(after) != _stat_identity(opened) or _stat_identity(
                        path_after
                    ) != _stat_identity(opened):
                        raise ValueError("child artifact directory changed during enumeration")
                finally:
                    os.close(nested_descriptor)
            elif stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
                files.add(relative)
                if len(files) > _MAX_CHILD_ARTIFACT_TREE_FILES:
                    raise ValueError("child artifact inventory exceeds its file-count limit")
            else:
                raise ValueError(
                    "child artifact inventory may contain only directories and single-link regular files"
                )

    try:
        opened_root = os.fstat(root_descriptor)
        if _stat_identity(opened_root) != _stat_identity(before):
            raise ValueError("child artifact inventory root changed while opening")
        walk(root_descriptor, "", 0)
        after = os.fstat(root_descriptor)
        path_after = os.lstat(root)
        if _stat_identity(after) != _stat_identity(opened_root) or _stat_identity(
            path_after
        ) != _stat_identity(opened_root):
            raise ValueError("child artifact inventory root changed during enumeration")
    finally:
        os.close(root_descriptor)
    return files


def _read_channel_dataset_child_rows(
    root: Path,
    relative_path: str,
    *,
    child_id: str,
    child_kind: str,
    parent_subject_id: str,
    expected_bytes: int,
    expected_sha256: str,
) -> dict[str, Any]:
    """Read bounded NPZ row metadata and bind every row to its frozen child."""
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    if hasattr(os, "O_BINARY"):
        file_flags |= os.O_BINARY
    directory_descriptor = os.open(root, directory_flags)
    file_descriptor = -1
    try:
        components = relative_path.split("/")
        for component in components[:-1]:
            next_descriptor = os.open(component, directory_flags, dir_fd=directory_descriptor)
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor
        file_descriptor = os.open(components[-1], file_flags, dir_fd=directory_descriptor)
        before = os.fstat(file_descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(
                f"split child artifact {child_id!r} must be a single-link regular file"
            )
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(file_descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        if size != expected_bytes or digest.hexdigest() != expected_sha256:
            raise ValueError(
                f"split child artifact {child_id!r} changed after its roster hash was checked"
            )
        os.lseek(file_descriptor, 0, os.SEEK_SET)
        with os.fdopen(file_descriptor, "rb") as stream:
            file_descriptor = -1
            observed_shapes: dict[str, tuple[int, ...]] = {}
            observed_dtypes: dict[str, np.dtype[Any]] = {}
            try:
                with zipfile.ZipFile(stream) as archive:
                    members: dict[str, zipfile.ZipInfo] = {}
                    for entry in archive.infolist():
                        if not entry.filename.endswith(".npy"):
                            raise ValueError(
                                f"split child artifact {child_id!r} has an unexpected NPZ member"
                            )
                        name = entry.filename[:-4]
                        if name in members:
                            raise ValueError(
                                f"split child artifact {child_id!r} repeats array {name!r}"
                            )
                        members[name] = entry
                    required_members = {"h", "y", "group", "domain", "source", "metadata_json"}
                    optional_members = {"generation_report_json", "label_semantics_json"}
                    if (
                        not required_members.issubset(members)
                        or set(members) - required_members - optional_members
                    ):
                        raise ValueError(
                            f"split child artifact {child_id!r} has an invalid ChannelDataset array roster"
                        )
                    metadata_member = members["metadata_json"]
                    if metadata_member.file_size > _MAX_CHILD_METADATA_NPY_BYTES:
                        raise ValueError(
                            f"split child artifact {child_id!r} metadata exceeds 64 MiB"
                        )
                    with archive.open(metadata_member) as metadata_stream:
                        metadata_payload = metadata_stream.read(_MAX_CHILD_METADATA_NPY_BYTES + 1)
                    if len(metadata_payload) > _MAX_CHILD_METADATA_NPY_BYTES:
                        raise ValueError(
                            f"split child artifact {child_id!r} metadata exceeds 64 MiB"
                        )
                    for name, member in members.items():
                        try:
                            with archive.open(member) as array_stream:
                                version = np.lib.format.read_magic(array_stream)
                                if version == (1, 0):
                                    shape, _fortran_order, dtype = (
                                        np.lib.format.read_array_header_1_0(
                                            array_stream,
                                            max_header_size=_MAX_CHILD_ARRAY_HEADER_BYTES,
                                        )
                                    )
                                elif version in {(2, 0), (3, 0)}:
                                    shape, _fortran_order, dtype = (
                                        np.lib.format.read_array_header_2_0(
                                            array_stream,
                                            max_header_size=_MAX_CHILD_ARRAY_HEADER_BYTES,
                                        )
                                    )
                                else:
                                    raise ValueError("unsupported NPY version")
                                expected_member_bytes = array_stream.tell() + (
                                    math.prod(shape) * np.dtype(dtype).itemsize
                                )
                                if expected_member_bytes != member.file_size:
                                    raise ValueError(
                                        "NPY header shape/dtype disagrees with member size"
                                    )
                                observed_shapes[name] = tuple(shape)
                                observed_dtypes[name] = np.dtype(dtype)
                        except (EOFError, OSError, ValueError, zipfile.BadZipFile) as error:
                            raise ValueError(
                                f"split child artifact {child_id!r} array {name!r} "
                                "has an invalid NPY header"
                            ) from error
            except (OSError, zipfile.BadZipFile, RuntimeError, zlib.error) as error:
                raise ValueError(
                    f"split child artifact {child_id!r} is not a readable ChannelDataset NPZ"
                ) from error
            after = os.fstat(stream.fileno())
        if _stat_identity(before) != _stat_identity(after):
            raise ValueError(f"split child artifact {child_id!r} changed while reading metadata")

        try:
            metadata_array = np.load(io.BytesIO(metadata_payload), allow_pickle=False)
        except (ValueError, EOFError, OSError) as error:
            raise ValueError(
                f"split child artifact {child_id!r} has invalid ChannelDataset metadata"
            ) from error
        if metadata_array.shape != () or metadata_array.dtype.kind != "U":
            raise ValueError(
                f"split child artifact {child_id!r} metadata_json.npy must be a Unicode scalar"
            )
        try:
            rows = json.loads(str(metadata_array.item()), object_pairs_hook=_unique_json_object)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(
                f"split child artifact {child_id!r} metadata JSON is invalid"
            ) from error
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"split child artifact {child_id!r} must contain at least one row")

        # Inspect every persisted array header without materializing the channel
        # tensor.  The signed byte digest authenticates the artifact bytes; this
        # check binds the row metadata to the actual storage shapes and dtypes.
        expected_rows = len(rows)
        expected_array_shapes: dict[str, tuple[int, ...]] = {
            "h": (expected_rows,),
            "y": (expected_rows,),
            "group": (expected_rows,),
            "domain": (expected_rows,),
            "source": (expected_rows,),
            "metadata_json": (),
        }
        for optional_scalar in ("generation_report_json", "label_semantics_json"):
            if optional_scalar in members:
                expected_array_shapes[optional_scalar] = ()
        if any(
            observed_shapes.get(name, ())[:1] != shape
            for name, shape in expected_array_shapes.items()
        ):
            raise ValueError(
                f"split child artifact {child_id!r} array row counts disagree with metadata"
            )
        if (
            len(observed_shapes["h"]) != 3
            or observed_shapes["h"][0] != expected_rows
            or observed_shapes["h"][1] < 1
            or observed_shapes["h"][2] < 8
            or observed_dtypes["h"].kind != "c"
            or observed_dtypes["y"].kind not in "biuf"
            or any(observed_dtypes[name].kind != "U" for name in ("group", "domain", "source"))
            or observed_dtypes["metadata_json"].kind != "U"
            or any(
                observed_dtypes[name].kind != "U"
                for name in ("generation_report_json", "label_semantics_json")
                if name in members
            )
        ):
            raise ValueError(
                f"split child artifact {child_id!r} arrays have invalid shapes or dtypes"
            )
        for row_index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(
                    f"split child artifact {child_id!r} row {row_index} is not an object"
                )
            if row.get("split_child_id") != child_id:
                raise ValueError(
                    f"split child artifact {child_id!r} row {row_index} has a different child ID"
                )
            if row.get("split_child_kind") != child_kind:
                raise ValueError(
                    f"split child artifact {child_id!r} row {row_index} has a different child kind"
                )
            if row.get("split_subject_id") != parent_subject_id:
                raise ValueError(
                    f"split child artifact {child_id!r} row {row_index} has a different parent"
                )
        return {
            "row_count": len(rows),
            "row_metadata_sha256": hashlib.sha256(metadata_payload).hexdigest(),
        }
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)
        if directory_descriptor >= 0:
            os.close(directory_descriptor)


def _load_frozen_inventory(
    path: Path, verified_artifact: VerifiedOpenSSHArtifact | None = None
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load a strict inventory that declares the complete ChannelDataset file set."""
    inventory_path = Path(os.path.abspath(path))
    metadata = os.lstat(inventory_path)
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError("inventory manifest must not be a symlink")
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ValueError("inventory must be a single-link regular file")
    payload = _read_stable_regular_file(
        inventory_path, label="inventory manifest", max_bytes=16 * 1024 * 1024
    )
    if verified_artifact is not None and payload != verified_artifact.payload:
        raise ValueError("inventory changed after signature verification")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate inventory JSON key: {key}")
            result[key] = value
        return result

    inventory = json.loads(payload, object_pairs_hook=unique_object)
    if not isinstance(inventory, dict) or set(inventory) != {
        "schema",
        "frozen",
        "inventory_id",
        "datasets",
    }:
        raise ValueError("inventory must contain exactly schema, frozen, inventory_id, datasets")
    if inventory["schema"] != _INVENTORY_SCHEMA or inventory["frozen"] is not True:
        raise ValueError(f"inventory must be frozen and use schema {_INVENTORY_SCHEMA}")
    if not isinstance(inventory["inventory_id"], str) or not inventory["inventory_id"].strip():
        raise ValueError("inventory_id must be a nonempty string")
    entries = inventory["datasets"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("inventory datasets must be a nonempty array")

    by_name: dict[str, dict[str, Any]] = {}
    by_path: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != {"name", "path", "sha256", "bytes", "rows"}:
            raise ValueError(f"inventory dataset {index} has invalid fields")
        name, relative_path, digest = entry["name"], entry["path"], entry["sha256"]
        if not isinstance(name, str) or not name.strip() or name.strip() != name:
            raise ValueError(f"inventory dataset {index} has an invalid name")
        if name in by_name:
            raise ValueError(f"inventory has duplicate dataset name: {name}")
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or relative_path.startswith("/")
            or re.match(r"^[A-Za-z]:", relative_path) is not None
            or "\\" in relative_path
            or any(part in {"", ".", ".."} for part in relative_path.split("/"))
        ):
            raise ValueError(f"inventory dataset {name!r} has an unsafe relative path")
        if relative_path in by_path:
            raise ValueError(f"inventory has duplicate dataset path: {relative_path}")
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise ValueError(f"inventory dataset {name!r} has an invalid SHA-256")
        for field in ("bytes", "rows"):
            value = entry[field]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"inventory dataset {name!r} has invalid {field}")
        by_name[name] = {
            **entry,
            "absolute_path": Path(
                os.path.abspath(inventory_path.parent.joinpath(*relative_path.split("/")))
            ),
        }
        by_path.add(relative_path)

    metadata_result: dict[str, Any] = {
        "path": str(inventory_path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "inventory_id": inventory["inventory_id"],
        "dataset_count": len(entries),
    }
    if verified_artifact is not None:
        metadata_result.update(
            {
                "signature_sha256": verified_artifact.signature_sha256,
                "allowed_signers_sha256": verified_artifact.allowed_signers_sha256,
                "signer_id": verified_artifact.signer_id,
                "signature_namespace": verified_artifact.namespace,
            }
        )
    return by_name, metadata_result


def _load_legacy_asset_reconstruction(
    path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    reconstruction_path = Path(os.path.abspath(path))
    payload = _read_stable_regular_file(
        reconstruction_path,
        label="legacy scene-asset reconstruction manifest",
        max_bytes=16 * 1024 * 1024,
    )

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate reconstruction JSON key: {key}")
            result[key] = value
        return result

    reconstruction = json.loads(payload, object_pairs_hook=unique_object)
    expected_fields = {
        "schema",
        "status",
        "reconstruction_method",
        "reconstruction_tool_source_sha256",
        "asset_manifest_builder_source_sha256",
        "source_revision",
        "package_source_sha256",
        "uv_lock_sha256",
        "sionna_rt_version",
        "sionna_rt_wheel_sha256",
        "datasets",
        "limitations",
    }
    if not isinstance(reconstruction, dict) or set(reconstruction) != expected_fields:
        raise ValueError("legacy scene-asset reconstruction manifest has invalid fields")
    if (
        reconstruction["schema"] != _LEGACY_ASSET_RECONSTRUCTION_SCHEMA
        or reconstruction["status"] != "NOT_QUALIFIED"
        or reconstruction["reconstruction_method"]
        != "historical_git_tree_and_lockfile_pinned_sionna_rt_wheel"
        or not isinstance(reconstruction["source_revision"], str)
        or _GIT_SHA_RE.fullmatch(reconstruction["source_revision"]) is None
        or not isinstance(reconstruction["sionna_rt_version"], str)
        or not reconstruction["sionna_rt_version"]
        or not isinstance(reconstruction["limitations"], list)
        or not all(isinstance(item, str) and item for item in reconstruction["limitations"])
    ):
        raise ValueError("legacy scene-asset reconstruction manifest has invalid provenance")
    for field in (
        "reconstruction_tool_source_sha256",
        "asset_manifest_builder_source_sha256",
        "package_source_sha256",
        "uv_lock_sha256",
        "sionna_rt_wheel_sha256",
    ):
        value = reconstruction[field]
        if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
            raise ValueError(f"legacy scene-asset reconstruction has invalid {field}")
    entries = reconstruction["datasets"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("legacy scene-asset reconstruction datasets must be a nonempty array")
    by_name: dict[str, dict[str, Any]] = {}
    entry_fields = {
        "name",
        "input_sha256",
        "input_bytes",
        "rows",
        "source_revision",
        "package_source_sha256",
        "scene",
        "scene_config",
        "scene_asset_manifest",
    }
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != entry_fields:
            raise ValueError(f"legacy scene reconstruction dataset {index} has invalid fields")
        name = entry["name"]
        if not isinstance(name, str) or not name.strip() or name.strip() != name or name in by_name:
            raise ValueError(f"legacy scene reconstruction dataset {index} has an invalid name")
        if (
            not isinstance(entry["input_sha256"], str)
            or _SHA256_RE.fullmatch(entry["input_sha256"]) is None
            or isinstance(entry["input_bytes"], bool)
            or not isinstance(entry["input_bytes"], int)
            or entry["input_bytes"] <= 0
            or isinstance(entry["rows"], bool)
            or not isinstance(entry["rows"], int)
            or entry["rows"] <= 0
            or entry["source_revision"] != reconstruction["source_revision"]
            or entry["package_source_sha256"] != reconstruction["package_source_sha256"]
            or not isinstance(entry["scene"], str)
            or not entry["scene"]
            or not isinstance(entry["scene_config"], dict)
            or not isinstance(entry["scene_asset_manifest"], dict)
        ):
            raise ValueError(f"legacy scene reconstruction dataset {name!r} has invalid values")
        by_name[name] = entry
    metadata = {
        "path": str(reconstruction_path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "reconstruction_id": reconstruction["source_revision"],
        "dataset_count": len(by_name),
        "method": reconstruction["reconstruction_method"],
        "reconstruction_tool_source_sha256": reconstruction["reconstruction_tool_source_sha256"],
        "asset_manifest_builder_source_sha256": reconstruction[
            "asset_manifest_builder_source_sha256"
        ],
        "source_revision": reconstruction["source_revision"],
        "package_source_sha256": reconstruction["package_source_sha256"],
        "uv_lock_sha256": reconstruction["uv_lock_sha256"],
        "sionna_rt_version": reconstruction["sionna_rt_version"],
        "sionna_rt_wheel_sha256": reconstruction["sionna_rt_wheel_sha256"],
        "authority": "unsigned retrospective reconstruction; not external run-time attestation",
        "limitations": reconstruction["limitations"],
    }
    return by_name, metadata


def _attach_reconstructed_scene_assets(
    data: ChannelDataset,
    *,
    dataset_name: str,
    entry: dict[str, Any],
    input_sha256: str,
    input_bytes: int,
    manifest_metadata: dict[str, Any],
) -> ChannelDataset:
    if (
        input_sha256 != entry["input_sha256"]
        or input_bytes != entry["input_bytes"]
        or len(data.y) != entry["rows"]
    ):
        raise ValueError(f"{dataset_name} bytes or row count disagree with reconstruction manifest")
    report = dict(data.generation_report)
    if "scene_asset_manifest" in report:
        raise ValueError(f"{dataset_name} already contains a saved scene-asset manifest")
    environment = report.get("environment")
    if (
        not isinstance(environment, dict)
        or environment.get("git_sha") != entry["source_revision"]
        or report.get("package_source_sha256") != entry["package_source_sha256"]
        or environment.get("sionna-rt") != manifest_metadata["sionna_rt_version"]
        or report.get("scene_config") != entry["scene_config"]
        or any(
            metadata.get("scene") != entry["scene"]
            or metadata.get("scene_config") != entry["scene_config"]
            for metadata in data.metadata
        )
    ):
        raise ValueError(
            f"{dataset_name} rows or run provenance disagree with reconstruction manifest"
        )
    report["scene_asset_manifest"] = entry["scene_asset_manifest"]
    metadata_rows = []
    asset_hash = entry["scene_asset_manifest"].get("asset_set_sha256")
    for original in data.metadata:
        row = dict(original)
        row["scene_asset_set_sha256"] = asset_hash
        row["scene_asset_manifest_complete"] = True
        metadata_rows.append(row)
    return ChannelDataset(
        h=data.h,
        y=data.y,
        group=data.group,
        domain=data.domain,
        source=data.source,
        metadata=tuple(metadata_rows),
        generation_report=report,
        label_semantics=data.label_semantics,
    )


def _validate_split_audit_report(report: dict[str, Any]) -> None:
    try:
        schema = json.loads(_SPLIT_AUDIT_SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot load channel split audit report schema: {error}") from error
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(report), key=lambda error: tuple(map(str, error.path)))
    if errors:
        schema_error = errors[0]
        location = ".".join(str(part) for part in schema_error.path) or "<root>"
        raise ValueError(
            f"channel split audit report violates schema at {location}: {schema_error.message}"
        )


def _parse_dataset(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("dataset must be NAME=PATH")
    if name != name.strip():
        raise argparse.ArgumentTypeError("dataset NAME must be canonical (no padding)")
    return name, Path(path)


def _parse_split_child_roster(value: str) -> tuple[str, Path]:
    child_kind, separator, path = value.partition("=")
    if not separator or not child_kind.strip() or not path.strip():
        raise argparse.ArgumentTypeError("split child roster must be KIND=PATH")
    if child_kind != child_kind.strip():
        raise argparse.ArgumentTypeError("split child roster KIND must be canonical (no padding)")
    return child_kind, Path(path)
