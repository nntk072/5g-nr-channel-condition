"""Build an input-hash-bound split-lineage report from saved ChannelDatasets."""

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
from typing import Any, Sequence

import numpy as np
from datasets.channel import ChannelDataset
from datasets.legacy_rt_split import upgrade_legacy_rt_split_metadata
from datasets.split_lineage import derive_channel_dataset_split_evidence
from datasets.split_policy import (
    SPLIT_POLICY_SIGNATURE_NAMESPACE,
    SplitChildBinding,
    VerifiedOpenSSHArtifact,
    audit_split_policy_child_roster,
    audit_split_policy_parent_roster,
    verify_openssh_signed_artifact,
    verify_registry_spatiotemporal_audit,
    verify_signed_split_policy_registry_attestation,
)

_INVENTORY_SCHEMA = "umcd.channel_dataset_inventory.v1"
_LEGACY_ASSET_RECONSTRUCTION_SCHEMA = "umcd.legacy_rt_scene_asset_reconstruction.v1"
_INVENTORY_SIGNATURE_NAMESPACE = "umcd-channel-dataset-inventory-v1@github.com"
_CHILD_ARTIFACT_ROSTER_SIGNATURE_NAMESPACE = "umcd-channel-child-artifact-roster-v1@github.com"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
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
    owners = [child_id for child_id, referenced_path in references if referenced_path == object_path]
    if len(owners) != len(set(owners)) or sorted(owners) != sorted(artifact_child_ids):
        raise ValueError(f"{label} manifest owners differ from the signed child roster")


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


def _read_stable_regular_file(
    path: Path, *, label: str, max_bytes: int | None = None
) -> bytes:
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
                    if (
                        _stat_identity(after) != _stat_identity(opened)
                        or _stat_identity(path_after) != _stat_identity(opened)
                    ):
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
        if (
            _stat_identity(after) != _stat_identity(opened_root)
            or _stat_identity(path_after) != _stat_identity(opened_root)
        ):
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
            raise ValueError(f"split child artifact {child_id!r} must be a single-link regular file")
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
                        metadata_payload = metadata_stream.read(
                            _MAX_CHILD_METADATA_NPY_BYTES + 1
                        )
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
                raise ValueError(f"split child artifact {child_id!r} row {row_index} is not an object")
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


def _load_split_child_artifact_roster(
    path: Path,
    *,
    expected_child_kind: str,
    expected_child_bindings: dict[str, SplitChildBinding],
    verified_artifact: VerifiedOpenSSHArtifact,
    split_registry: Any | None = None,
) -> dict[str, Any]:
    payload = verified_artifact.payload
    if len(payload) > _MAX_CHILD_ROSTER_BYTES:
        raise ValueError("split child artifact roster exceeds the 16 MiB limit")
    value = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_json_object)
    if not isinstance(value, dict) or value.get("child_kind") != expected_child_kind:
        raise ValueError("split child artifact roster has an invalid schema or child kind")
    schema = value.get("schema")
    if schema == "umcd.channel_dataset_child_artifact_roster.v1":
        if set(value) != {"schema", "child_kind", "artifacts"}:
            raise ValueError("split child artifact v1 roster has unsupported fields")
        roster_version = 1
    elif schema == "umcd.channel_dataset_child_artifact_roster.v2":
        if set(value) != {"schema", "child_kind", "artifacts"}:
            raise ValueError("split child artifact v2 roster has unsupported fields")
        roster_version = 2
    elif schema == "umcd.channel_dataset_child_artifact_roster.v3":
        if set(value) != {"schema", "child_kind", "artifact_root", "artifacts"}:
            raise ValueError("split child artifact v3 roster has unsupported fields")
        roster_version = 3
    elif schema == "umcd.channel_dataset_child_artifact_roster.v4":
        if set(value) != {"schema", "child_kind", "artifact_root", "artifacts"}:
            raise ValueError("split child artifact v4 roster has unsupported fields")
        roster_version = 4
    elif schema == "umcd.channel_dataset_child_artifact_roster.v5":
        if set(value) != {"schema", "child_kind", "artifact_root", "artifacts"}:
            raise ValueError("split child artifact v5 roster has unsupported fields")
        roster_version = 5
    elif schema == "umcd.channel_dataset_child_artifact_roster.v6":
        if set(value) != {"schema", "child_kind", "artifact_root", "artifacts"}:
            raise ValueError("split child artifact v6 roster has unsupported fields")
        roster_version = 6
    else:
        raise ValueError("split child artifact roster has an unsupported schema")
    entries = value["artifacts"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("split child artifact roster artifacts must be a nonempty array")
    observed_child_ids: set[str] = set()
    observed_artifact_ids: set[str] = set()
    observed_paths: set[str] = set()
    observed_trajectory_ids: set[str] = set()
    observed_episode_ids: set[str] = set()
    observed_scenario_run_ids: set[str] = set()
    declared: list[dict[str, Any]] = []
    roster_directory = Path(os.path.abspath(path)).parent
    artifact_root_name: str | None = None
    if roster_version >= 3:
        artifact_root_name = value["artifact_root"]
        if (
            not isinstance(artifact_root_name, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", artifact_root_name)
            is None
            or artifact_root_name in {".", ".."}
        ):
            raise ValueError("split child artifact v3 root must be one safe directory name")
    artifact_root = (
        roster_directory / artifact_root_name
        if artifact_root_name is not None
        else roster_directory
    )
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"split child artifact {index} has invalid fields")
        if roster_version == 1:
            expected_fields = {"child_id", "path", "bytes", "sha256"}
        elif roster_version in {5, 6}:
            artifact_type_hint = entry.get("artifact_type")
            owner_field = (
                "child_ids"
                if (
                    artifact_type_hint == "umcd_episode_store_content_npz"
                    if roster_version == 5
                    else artifact_type_hint in _V6_MULTI_OWNER_ARTIFACT_TYPES
                )
                else "child_id"
            )
            expected_fields = {
                "artifact_id",
                "artifact_type",
                owner_field,
                "path",
                "bytes",
                "sha256",
            }
        else:
            expected_fields = {
                "artifact_id",
                "artifact_type",
                "child_id",
                "path",
                "bytes",
                "sha256",
            }
        if set(entry) != expected_fields:
            raise ValueError(f"split child artifact {index} has invalid fields")
        relative_path = entry["path"]
        byte_count = entry["bytes"]
        digest = entry["sha256"]
        if roster_version == 1:
            child_id = entry["child_id"]
            if not isinstance(child_id, str) or not child_id or child_id.strip() != child_id:
                raise ValueError(f"split child artifact {index} has an invalid child ID")
            owner_child_ids = [child_id]
            observed_child_ids.add(child_id)
            artifact_id = child_id
            artifact_type = "channel_dataset_npz"
            if artifact_id in observed_artifact_ids:
                raise ValueError(f"split child artifact roster repeats child ID: {child_id}")
        else:
            artifact_id = entry["artifact_id"]
            artifact_type = entry["artifact_type"]
            if (
                roster_version == 5 and artifact_type == "umcd_episode_store_content_npz"
            ) or (
                roster_version == 6 and artifact_type in _V6_MULTI_OWNER_ARTIFACT_TYPES
            ):
                owner_child_ids = entry["child_ids"]
                if (
                    not isinstance(owner_child_ids, list)
                    or not owner_child_ids
                    or any(
                        not isinstance(owner, str) or not owner or owner.strip() != owner
                        for owner in owner_child_ids
                    )
                    or owner_child_ids != sorted(set(owner_child_ids))
                ):
                    raise ValueError(
                        f"split child artifact {index} has an invalid canonical child_ids roster"
                    )
                observed_child_ids.update(owner_child_ids)
            else:
                child_id = entry["child_id"]
                if not isinstance(child_id, str) or not child_id or child_id.strip() != child_id:
                    raise ValueError(f"split child artifact {index} has an invalid child ID")
                owner_child_ids = [child_id]
                observed_child_ids.add(child_id)
            if (
                not isinstance(artifact_id, str)
                or not artifact_id
                or artifact_id.strip() != artifact_id
            ):
                raise ValueError(f"split child artifact {index} has an invalid artifact ID")
            if (
                not isinstance(artifact_type, str)
                or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", artifact_type) is None
            ):
                raise ValueError(f"split child artifact {index} has an invalid artifact type")
            semantic_types = (
                _V6_SEMANTIC_CHILD_ARTIFACT_TYPES
                if roster_version >= 6
                else _V4_SEMANTIC_CHILD_ARTIFACT_TYPES
                if roster_version >= 4
                else _V3_SEMANTIC_CHILD_ARTIFACT_TYPES
            )
            if roster_version >= 3 and artifact_type not in semantic_types:
                raise ValueError(
                    "split child artifact v3 requires a semantic validator for artifact type "
                    f"{artifact_type!r}"
                )
            if artifact_id in observed_artifact_ids:
                raise ValueError(f"split child artifact roster repeats artifact ID: {artifact_id}")
        observed_artifact_ids.add(artifact_id)
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or relative_path.startswith("/")
            or "\\" in relative_path
            or re.match(r"^[A-Za-z]:", relative_path) is not None
            or any(part in {"", ".", ".."} for part in relative_path.split("/"))
            or re.fullmatch(r"[A-Za-z0-9._/-]+", relative_path) is None
        ):
            raise ValueError(f"split child artifact {index} has an unsafe relative path")
        if relative_path in observed_paths:
            raise ValueError(f"split child artifact roster repeats path: {relative_path}")
        observed_paths.add(relative_path)
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
            raise ValueError(f"split child artifact {index} has an invalid byte count")
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise ValueError(f"split child artifact {index} has an invalid SHA-256")
        if artifact_type == "channel_dataset_npz" and not relative_path.endswith(".npz"):
            raise ValueError("channel_dataset_npz artifacts must use an .npz path")
        if artifact_type == "umcd_episode_store_manifest_json" and not relative_path.endswith(
            ".json"
        ):
            raise ValueError("umcd_episode_store_manifest_json artifacts must use a .json path")
        if artifact_type == "umcd_episode_store_content_npz" and not relative_path.endswith(
            ".npz"
        ):
            raise ValueError("umcd_episode_store_content_npz artifacts must use a .npz path")
        declaration = {
            "artifact_id": artifact_id,
            "artifact_type": artifact_type,
            "path": relative_path,
            "bytes": byte_count,
            "sha256": digest,
        }
        if (
            roster_version == 5 and artifact_type == "umcd_episode_store_content_npz"
        ) or (
            roster_version == 6 and artifact_type in _V6_MULTI_OWNER_ARTIFACT_TYPES
        ):
            if any(owner not in expected_child_bindings for owner in owner_child_ids):
                raise ValueError(f"split child artifact {index} references an unknown child ID")
            declaration["child_ids"] = list(owner_child_ids)
        else:
            declaration["child_id"] = owner_child_ids[0]
        declared.append(declaration)
    if roster_version >= 3:
        declared_paths = {entry["path"] for entry in declared}
        observed_paths_under_root = _enumerate_regular_files_beneath(artifact_root)
        if observed_paths_under_root != declared_paths:
            raise ValueError(
                "split child artifact v3 roster does not exactly cover its artifact_root "
                f"(missing={sorted(observed_paths_under_root - declared_paths)}, "
                f"unlisted={sorted(declared_paths - observed_paths_under_root)})"
            )
    scenario_content_owner_references: dict[str, list[tuple[str, str]]] = {}
    if roster_version >= 6:
        for candidate in declared:
            if candidate["artifact_type"] != "scenario_gbsm_run_manifest_json":
                continue
            manifest_bytes = _read_stable_regular_file(
                artifact_root / candidate["path"],
                label="scenario GBSM run manifest for owner inventory",
                max_bytes=64 * 1024 * 1024,
            )
            try:
                manifest_value = json.loads(
                    manifest_bytes.decode("utf-8"),
                    object_pairs_hook=_unique_json_object,
                )
            except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
                raise ValueError(
                    "scenario GBSM run manifest is not valid bounded UTF-8 JSON"
                ) from error
            if not isinstance(manifest_value, dict) or not isinstance(
                manifest_value.get("records"), list
            ):
                raise ValueError("scenario GBSM run manifest record roster is invalid")
            owner_parent_children: dict[str, list[str]] = {}
            for owner in candidate["child_ids"]:
                owner_parent_children.setdefault(
                    expected_child_bindings[owner].parent_subject_id, []
                ).append(owner)
            for reference in manifest_value["records"]:
                if not isinstance(reference, dict):
                    raise ValueError("scenario GBSM record reference is malformed")
                parent_id = reference.get("sample_id")
                object_relative_path = reference.get("relative_path")
                if (
                    not isinstance(parent_id, str)
                    or not parent_id
                    or parent_id.strip() != parent_id
                    or not isinstance(object_relative_path, str)
                    or not object_relative_path.startswith("scenario_gbsm/records/")
                    or not object_relative_path.endswith(".npz")
                ):
                    raise ValueError("scenario GBSM record reference has invalid identity or path")
                owners = owner_parent_children.get(parent_id)
                if not owners:
                    raise ValueError("scenario GBSM run manifest omits a record parent owner")
                scenario_content_owner_references.setdefault(object_relative_path, []).extend(
                    (owner, object_relative_path) for owner in owners
                )
    expected_child_ids = set(expected_child_bindings)
    if observed_child_ids != expected_child_ids:
        raise ValueError(
            "split child artifact roster does not cover exactly the frozen children "
            f"(missing={sorted(expected_child_ids - observed_child_ids)}, "
            f"extra={sorted(observed_child_ids - expected_child_ids)})"
        )
    semantic_child_ids = {
        entry["child_id"]
        for entry in declared
        if entry["artifact_type"] == "channel_dataset_npz"
    }
    if semantic_child_ids != expected_child_ids:
        raise ValueError(
            "split child artifact roster must include a ChannelDataset NPZ for every frozen child "
            f"(missing={sorted(expected_child_ids - semantic_child_ids)})"
        )
    verified: list[dict[str, Any]] = []
    for entry in declared:
        artifact_id = entry["artifact_id"]
        artifact_type = entry["artifact_type"]
        child_ids = entry.get("child_ids")
        child_id = entry.get("child_id", child_ids[0] if child_ids else None)
        relative_path = entry["path"]
        byte_count = entry["bytes"]
        digest = entry["sha256"]
        actual_bytes, actual_digest = _hash_inventory_relative_file(
            artifact_root, relative_path, label=f"split child artifact {child_id!r}"
        )
        if actual_bytes != byte_count:
            raise ValueError(f"split child artifact byte count differs from roster: {child_id}")
        if actual_digest != digest:
            raise ValueError(f"split child artifact SHA-256 differs from roster: {artifact_id}")
        if artifact_type == "channel_dataset_npz":
            semantic_rows = _read_channel_dataset_child_rows(
                artifact_root,
                relative_path,
                child_id=child_id,
                child_kind=expected_child_kind,
                parent_subject_id=expected_child_bindings[child_id].parent_subject_id,
                expected_bytes=actual_bytes,
                expected_sha256=actual_digest,
            )
            semantic_rows["semantic_validator"] = "channel_dataset_child_parent_rows_v1"
        elif artifact_type == "nr_trajectory_store_manifest_json":
            if not relative_path.startswith("manifests/trajectories/"):
                raise ValueError(
                    "nr_trajectory_store_manifest_json must be under "
                    "manifests/trajectories/"
                )
            from datasets.nr_store import read_nr_trajectory

            trajectory = read_nr_trajectory(artifact_root, relative_path)
            if trajectory.trajectory_id in observed_trajectory_ids:
                raise ValueError(
                    f"split child artifact roster repeats NR trajectory ID: "
                    f"{trajectory.trajectory_id}"
                )
            observed_trajectory_ids.add(trajectory.trajectory_id)
            parent_subject_id = expected_child_bindings[child_id].parent_subject_id
            for sample_index, record in enumerate(trajectory.records):
                expected_lineage = {
                    "split_subject_id": parent_subject_id,
                    "split_child_id": child_id,
                    "split_child_kind": expected_child_kind,
                }
                for field, expected_value in expected_lineage.items():
                    if record.provenance.get(field) != expected_value:
                        raise ValueError(
                            f"NR trajectory child artifact {artifact_id!r} sample "
                            f"{sample_index} has a different {field}"
                        )
            sample_ids_payload = json.dumps(
                list(trajectory.sample_ids),
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            semantic_rows = {
                "semantic_validator": "nr_trajectory_store_v1",
                "trajectory_id": trajectory.trajectory_id,
                "trajectory_sample_count": len(trajectory.sample_ids),
                "trajectory_sample_ids_sha256": hashlib.sha256(sample_ids_payload).hexdigest(),
            }
        elif artifact_type in {
            "nr_record_store_manifest_json",
            "nr_record_store_content_npz",
        }:
            from datasets.nr_store import read_nr_record

            if artifact_type == "nr_record_store_manifest_json":
                if not relative_path.startswith("manifests/") or relative_path.startswith(
                    "manifests/trajectories/"
                ):
                    raise ValueError(
                        "nr_record_store_manifest_json must be under manifests/ outside "
                        "manifests/trajectories/"
                    )
                manifest_relative_path = relative_path
            else:
                if not relative_path.startswith("records/") or not relative_path.endswith(
                    ".npz"
                ):
                    raise ValueError(
                        "nr_record_store_content_npz must be beneath records/ with an .npz suffix"
                    )
                manifest_candidates: list[tuple[str, dict[str, Any], str]] = []
                for candidate in declared:
                    if (
                        candidate["artifact_type"] != "nr_record_store_manifest_json"
                    ):
                        continue
                    manifest_path = candidate["path"]
                    manifest_bytes = _read_stable_regular_file(
                        artifact_root / manifest_path,
                        label="NR record manifest for content artifact",
                        max_bytes=1 * 1024 * 1024,
                    )
                    try:
                        manifest_value = json.loads(
                            manifest_bytes.decode("utf-8"),
                            object_pairs_hook=_unique_json_object,
                        )
                    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
                        raise ValueError("NR record manifest is not valid bounded UTF-8 JSON") from error
                    if not isinstance(manifest_value, dict):
                        raise ValueError("NR record manifest must be a JSON object")
                    if manifest_value.get("relative_path") == relative_path:
                        manifest_candidates.append(
                            (manifest_path, manifest_value, candidate["child_id"])
                        )
                if len(manifest_candidates) != 1:
                    raise ValueError(
                        "nr_record_store_content_npz must be referenced by exactly one "
                        "rostered record manifest"
                    )
                manifest_relative_path, matched_manifest, manifest_child_id = (
                    manifest_candidates[0]
                )
                _require_exact_child_object_owners(
                    relative_path,
                    [child_id],
                    [(manifest_child_id, matched_manifest["relative_path"])],
                    label="NR record content object",
                )
                content_sha256 = matched_manifest.get("content_sha256")
                if (
                    not isinstance(content_sha256, str)
                    or _SHA256_RE.fullmatch(content_sha256) is None
                    or relative_path != f"records/{content_sha256[:2]}/{content_sha256}.npz"
                ):
                    raise ValueError(
                        "NR record content path does not match its canonical content hash"
                    )
            record = read_nr_record(artifact_root, manifest_relative_path)
            parent_subject_id = expected_child_bindings[child_id].parent_subject_id
            expected_lineage = {
                "split_subject_id": parent_subject_id,
                "split_child_id": child_id,
                "split_child_kind": expected_child_kind,
            }
            for field, expected_value in expected_lineage.items():
                if record.provenance.get(field) != expected_value:
                    raise ValueError(
                        f"NR record child artifact {artifact_id!r} has a different {field}"
                    )
            semantic_rows = {
                "semantic_validator": "nr_record_store_v1",
                "nr_sample_id": record.observation.sample_id,
                "nr_target_count": len(record.targets),
            }
        elif artifact_type == "scenario_gbsm_run_manifest_json":
            if roster_version < 6:
                raise ValueError("scenario GBSM run manifests require child artifact roster v6")
            if not relative_path.startswith("scenario_gbsm/manifests/"):
                raise ValueError(
                    "scenario_gbsm_run_manifest_json must be under scenario_gbsm/manifests/"
                )
            if split_registry is None:
                raise ValueError("scenario GBSM semantic admission requires the frozen registry")
            from datasets.scenario_gbsm import read_scenario_gbsm_run

            run = read_scenario_gbsm_run(artifact_root, relative_path)
            if run.run_id in observed_scenario_run_ids:
                raise ValueError(f"split child artifact roster repeats scenario run ID: {run.run_id}")
            observed_scenario_run_ids.add(run.run_id)
            parent_ids = {record.sample_id for record in run.records}
            parent_to_children: dict[str, list[str]] = {}
            for candidate_child_id, binding in expected_child_bindings.items():
                parent_to_children.setdefault(binding.parent_subject_id, []).append(
                    candidate_child_id
                )
            missing_parents = parent_ids - set(parent_to_children)
            if missing_parents:
                raise ValueError(
                    "scenario GBSM run has parents absent from the frozen child roster: "
                    f"{sorted(missing_parents)}"
                )
            expected_owners = sorted(
                child
                for parent in parent_ids
                for child in parent_to_children[parent]
            )
            if child_ids != expected_owners:
                raise ValueError(
                    "scenario GBSM run child_ids do not exactly cover its frozen parent rows"
                )
            split_audit = run.audit_split_assignment(split_registry)
            semantic_rows = {
                "semantic_validator": "scenario_gbsm_split_assignment_v1",
                "scenario_run_id": run.run_id,
                "scenario_parent_count": len(parent_ids),
                "scenario_record_count": len(run.records),
                "scenario_split_component_id": split_audit["component_id"],
                "scenario_split": split_audit["split"],
                "scenario_split_audit_sha256": split_audit["audit_sha256"],
            }
        elif artifact_type == "scenario_gbsm_record_content_npz":
            if roster_version < 6:
                raise ValueError("scenario GBSM record objects require child artifact roster v6")
            if (
                not relative_path.startswith("scenario_gbsm/records/")
                or not relative_path.endswith(".npz")
            ):
                raise ValueError(
                    "scenario_gbsm_record_content_npz must be beneath scenario_gbsm/records/"
                )
            scenario_references = scenario_content_owner_references.get(relative_path, [])
            if not scenario_references:
                raise ValueError(
                    "scenario GBSM record content must be referenced by a rostered run manifest"
                )
            _require_exact_child_object_owners(
                relative_path,
                child_ids if child_ids is not None else [child_id],
                scenario_references,
                label="scenario GBSM record content object",
            )
            semantic_rows = {"semantic_validator": "scenario_gbsm_record_content_v1"}
        elif artifact_type == "umcd_episode_store_manifest_json":
            if not relative_path.startswith("manifests/episodes/"):
                raise ValueError(
                    "umcd_episode_store_manifest_json must be under manifests/episodes/"
                )
            from datasets.episode_store import read_episode_bundle

            bundle = read_episode_bundle(
                artifact_root, relative_path, max_shard_bytes=512 * 1024 * 1024
            )
            episode_id = bundle.episode.episode_id
            if episode_id in observed_episode_ids:
                raise ValueError(
                    "split child artifact roster repeats UMCD episode ID: " f"{episode_id}"
                )
            observed_episode_ids.add(episode_id)
            parent_subject_id = expected_child_bindings[child_id].parent_subject_id
            if episode_id != parent_subject_id:
                raise ValueError(
                    f"UMCD episode child artifact {artifact_id!r} has a different "
                    "split parent episode ID"
                )
            semantic_rows = {
                "semantic_validator": "umcd_episode_store_v1",
                "episode_id": episode_id,
                "episode_link_count": len(bundle.episode.links),
                "episode_frame_count": len(bundle.episode.frame_times_s),
                "episode_target_count": len(bundle.targets),
            }
        elif artifact_type == "umcd_episode_store_content_npz":
            if not relative_path.startswith("objects/"):
                raise ValueError("umcd_episode_store_content_npz must be beneath objects/")
            manifest_candidates: list[tuple[str, str, dict[str, Any]]] = []
            for candidate in declared:
                if candidate["artifact_type"] != "umcd_episode_store_manifest_json":
                    continue
                manifest_path = candidate["path"]
                manifest_bytes = _read_stable_regular_file(
                    artifact_root / manifest_path,
                    label="UMCD episode bundle manifest for content artifact",
                    max_bytes=1024 * 1024,
                )
                try:
                    manifest_value = json.loads(
                        manifest_bytes.decode("utf-8"),
                        object_pairs_hook=_unique_json_object,
                    )
                except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
                    raise ValueError(
                        "UMCD episode bundle manifest is not valid bounded UTF-8 JSON"
                    ) from error
                if not isinstance(manifest_value, dict) or not isinstance(
                    manifest_value.get("layers"), list
                ):
                    raise ValueError("UMCD episode bundle manifest layers are invalid")
                for reference in manifest_value["layers"]:
                    if isinstance(reference, dict) and reference.get("path") == relative_path:
                        manifest_candidates.append(
                            (manifest_path, candidate["child_id"], reference)
                        )
            if not manifest_candidates:
                raise ValueError(
                    "umcd_episode_store_content_npz must be referenced by a rostered "
                    "episode manifest"
                )
            _require_exact_child_object_owners(
                relative_path,
                child_ids if child_ids is not None else [child_id],
                [(manifest_child_id, reference["path"]) for _, manifest_child_id, reference in manifest_candidates],
                label="UMCD episode content object",
            )
            manifest_paths = sorted({manifest_path for manifest_path, _, _ in manifest_candidates})
            _, _, reference = manifest_candidates[0]
            layer = reference.get("layer")
            content_sha256 = reference.get("sha256")
            if (
                not isinstance(layer, str)
                or not layer
                or not isinstance(content_sha256, str)
                or _SHA256_RE.fullmatch(content_sha256) is None
                or relative_path != f"objects/{layer}/{content_sha256[:2]}/{content_sha256}.npz"
            ):
                raise ValueError("UMCD episode content path does not match its layer digest")
            semantic_rows = {
                "semantic_validator": "umcd_episode_store_content_v1",
                "episode_layer": layer,
            }
            if roster_version == 4:
                semantic_rows["episode_manifest_path"] = manifest_paths[0]
            else:
                semantic_rows["episode_manifest_paths"] = manifest_paths
        else:
            semantic_rows = {}
        verified_entry = {
            "artifact_id": artifact_id,
            "artifact_type": artifact_type,
            "path": relative_path,
            "bytes": actual_bytes,
            "sha256": actual_digest,
            **semantic_rows,
        }
        if child_ids is not None:
            verified_entry["child_ids"] = child_ids
        else:
            verified_entry["child_id"] = child_id
        verified.append(verified_entry)
    verified.sort(
        key=lambda item: (
            item.get("child_id", item.get("child_ids", [""])[0]),
            item["artifact_id"],
        )
    )
    byte_only_count = sum("semantic_validator" not in item for item in verified)
    semantic_artifact_count = len(verified) - byte_only_count
    authority = (
        "detached OpenSSH signature authenticates the typed child-to-artifact roster; all "
        "artifact bytes are verified; ChannelDataset rows are checked against frozen "
        "child/kind/parent bindings; recognized NR trajectory stores are fully traversed "
        "through their content and record hashes"
    )
    if roster_version == 1:
        authority += "; roster v1 admits only semantically checked ChannelDataset artifacts"
    elif roster_version == 2:
        authority += "; unrecognized supplemental types receive byte-integrity checks only"
    else:
        authority += "; artifact types without semantic validators are rejected"
    if roster_version >= 3:
        authority += (
            "; roster v3 and later require exact regular-file inventory coverage beneath "
            "the declared artifact_root"
        )
    if roster_version >= 4:
        authority += (
            f"; roster v{roster_version} also validates UMCD episode-store manifests, parent "
            "episode IDs, and exact content-object child ownership"
        )
    if roster_version >= 6:
        authority += (
            "; roster v6 also traverses scenario GBSM runs and checks every parent against the "
            "frozen registry's shared trajectory-session component and split assignment"
        )
    audit: dict[str, Any] = {
        "schema": (
            "umcd.split_child_artifact_audit.v7"
            if roster_version == 6
            else "umcd.split_child_artifact_audit.v6"
            if roster_version == 5
            else "umcd.split_child_artifact_audit.v5"
            if roster_version == 4
            else "umcd.split_child_artifact_audit.v4"
        ),
        "status": (
            "PASS_EXACT_FROZEN_CHILD_ARTIFACT_BYTES_WITH_PARTIAL_SEMANTICS"
            if byte_only_count
            else "PASS_EXACT_FROZEN_CHILD_BYTES_AND_ROWS"
        ),
        "child_kind": expected_child_kind,
        "roster_schema": schema,
        "child_count": len(observed_child_ids),
        "artifact_count": len(verified),
        "artifact_root": artifact_root_name,
        "artifact_directory_inventory_status": (
            "EXACT" if roster_version >= 3 else "NOT_DECLARED"
        ),
        "semantic_artifact_count": semantic_artifact_count,
        "byte_only_artifact_count": byte_only_count,
        "artifact_types": sorted({item["artifact_type"] for item in verified}),
        "total_bytes": sum(item["bytes"] for item in verified),
        "authority": authority,
        "artifacts": verified,
        "row_count": sum(item.get("row_count", 0) for item in verified),
        "trajectory_sample_count": sum(
            item.get("trajectory_sample_count", 0) for item in verified
        ),
        "trajectory_count": len(observed_trajectory_ids),
        "scenario_gbsm_run_count": len(observed_scenario_run_ids),
        "scenario_gbsm_parent_count": sum(
            item.get("scenario_parent_count", 0) for item in verified
        ),
    }
    if roster_version >= 4:
        audit["episode_count"] = len(observed_episode_ids)
    audit["audit_sha256"] = hashlib.sha256(
        json.dumps(audit, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return {
        "path": str(Path(os.path.abspath(path))),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "child_kind": expected_child_kind,
        "signature_sha256": verified_artifact.signature_sha256,
        "allowed_signers_sha256": verified_artifact.allowed_signers_sha256,
        "signer_id": verified_artifact.signer_id,
        "signature_namespace": verified_artifact.namespace,
        "audit": audit,
    }


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


def audit_channel_datasets(
    inputs: Sequence[tuple[str, Path]],
    *,
    spatial_buffer_m: float,
    inventory_manifest: Path | None = None,
    inventory_signature: Path | None = None,
    trusted_allowed_signers: Path | None = None,
    inventory_signer_id: str | None = None,
    split_registry_path: Path | None = None,
    split_registry_signature: Path | None = None,
    trusted_split_allowed_signers: Path | None = None,
    split_registry_signer_id: str | None = None,
    split_child_rosters: Sequence[tuple[str, Path]] = (),
    split_child_artifact_rosters: Sequence[tuple[str, Path]] = (),
    require_complete_split_audit: bool = False,
    derive_legacy_rt_split_identities: bool = False,
    legacy_scene_assets_manifest: Path | None = None,
) -> dict[str, Any]:
    """Audit named NPZ datasets and bind the result to the exact bytes read.

    Args:
        inputs: Named saved ChannelDataset files to audit.
        spatial_buffer_m: Minimum separation used for spatial leakage links.
        inventory_manifest: Optional frozen inventory covering every input.
        inventory_signature: Optional detached signature for the inventory.
        trusted_allowed_signers: External keyring used to verify that signature.
        inventory_signer_id: Expected signer principal for the inventory.
        split_registry_path: Optional signed immutable split-registry v3 lock.
        split_registry_signature: Detached OpenSSH signature for that registry.
        trusted_split_allowed_signers: External trust file for the registry signer.
        split_registry_signer_id: Expected principal for the split-registry signature.
        split_child_rosters: Optional complete JSON binding roster per frozen child
            kind. This verifies registry-to-roster correspondence, not artifact bytes.
        split_child_artifact_rosters: Optional complete per-kind roster of child
            ChannelDataset NPZ paths, byte counts, and hashes, relative to each
            roster file. Each row must bind the roster child ID/kind and frozen
            parent subject in metadata. Each roster requires a detached `.sig`
            OpenSSH signature from the configured split-registry signer under a
            separate namespace.
        require_complete_split_audit: Require signed corpus inventory and signed
            split registry, plus complete binding and semantic artifact rosters
            for every registered child kind. This checks release integrity inputs;
            it does not qualify the dataset scientifically.
        derive_legacy_rt_split_identities: Explicitly derive missing split metadata
            in memory from complete RT scene provenance; never rewrites inputs.
        legacy_scene_assets_manifest: Optional retrospective asset reconstruction
            bound to exact legacy dataset bytes and historical source provenance.

    Returns:
        Hash-bound, explicitly unqualified lineage report for the supplied inputs.

    Raises:
        ValueError: An input, inventory, or requested legacy identity is invalid.
    """
    if not inputs:
        raise ValueError("at least one named ChannelDataset input is required")
    if type(require_complete_split_audit) is not bool:
        raise TypeError("require_complete_split_audit must be a boolean")
    if require_complete_split_audit:
        if inventory_manifest is None or inventory_signature is None:
            raise ValueError(
                "complete split audit requires a signed frozen corpus inventory"
            )
        if split_registry_path is None:
            raise ValueError("complete split audit requires a signed split registry")
    split_signature_arguments = (
        split_registry_path,
        split_registry_signature,
        trusted_split_allowed_signers,
        split_registry_signer_id,
    )
    if any(value is not None for value in split_signature_arguments) and any(
        value is None for value in split_signature_arguments
    ):
        raise ValueError(
            "split-registry verification requires a registry, signature, external trust file, "
            "and signer ID"
        )
    if split_child_rosters and split_registry_path is None:
        raise ValueError("split child rosters require a signed split registry")
    if split_child_artifact_rosters and split_registry_path is None:
        raise ValueError("split child artifact rosters require a signed split registry")
    if split_child_artifact_rosters and not split_child_rosters:
        raise ValueError("split child artifact rosters require complete child binding rosters")
    if any(
        not isinstance(entry, (tuple, list)) or len(entry) != 2
        for entry in split_child_rosters
    ):
        raise ValueError("split child rosters must be (child_kind, path) pairs")
    child_roster_kinds = [kind for kind, _ in split_child_rosters]
    if any(
        not isinstance(kind, str) or not kind or kind != kind.strip()
        for kind in child_roster_kinds
    ):
        raise ValueError("split child roster kinds must be canonical nonempty strings")
    if len(set(child_roster_kinds)) != len(child_roster_kinds):
        raise ValueError("split child roster kinds must be unique")
    if any(
        not isinstance(entry, (tuple, list)) or len(entry) != 2
        for entry in split_child_artifact_rosters
    ):
        raise ValueError("split child artifact rosters must be (child_kind, path) pairs")
    child_artifact_kinds = [kind for kind, _ in split_child_artifact_rosters]
    if any(
        not isinstance(kind, str) or not kind or kind != kind.strip()
        for kind in child_artifact_kinds
    ):
        raise ValueError("split child artifact roster kinds must be canonical nonempty strings")
    if len(set(child_artifact_kinds)) != len(child_artifact_kinds):
        raise ValueError("split child artifact roster kinds must be unique")
    if legacy_scene_assets_manifest is not None and not derive_legacy_rt_split_identities:
        raise ValueError(
            "legacy scene-asset reconstruction requires --derive-legacy-rt-split-identities"
        )

    declared_inventory: dict[str, dict[str, Any]] | None = None
    inventory_metadata: dict[str, Any] | None = None
    signature_arguments = (
        inventory_signature,
        trusted_allowed_signers,
        inventory_signer_id,
    )
    if any(value is not None for value in signature_arguments) and (
        inventory_manifest is None or any(value is None for value in signature_arguments)
    ):
        raise ValueError(
            "signed inventory verification requires a manifest, signature, allowed-signers file, and signer ID"
        )
    if inventory_manifest is not None:
        verified_artifact = None
        if inventory_signature is not None:
            assert trusted_allowed_signers is not None and inventory_signer_id is not None
            verified_artifact = verify_openssh_signed_artifact(
                inventory_manifest,
                inventory_signature,
                trusted_allowed_signers_path=trusted_allowed_signers,
                signer_id=inventory_signer_id,
                namespace=_INVENTORY_SIGNATURE_NAMESPACE,
            )
        declared_inventory, inventory_metadata = _load_frozen_inventory(
            inventory_manifest, verified_artifact
        )
        actual_names = {name for name, _ in inputs if isinstance(name, str)}
        if actual_names != set(declared_inventory):
            missing = sorted(set(declared_inventory) - actual_names)
            extra = sorted(actual_names - set(declared_inventory))
            raise ValueError(
                f"dataset inputs do not match frozen inventory (missing={missing}, extra={extra})"
            )

    reconstructed_assets: dict[str, dict[str, Any]] | None = None
    reconstruction_metadata: dict[str, Any] | None = None
    if legacy_scene_assets_manifest is not None:
        reconstructed_assets, reconstruction_metadata = _load_legacy_asset_reconstruction(
            legacy_scene_assets_manifest
        )
        actual_names = {name for name, _ in inputs if isinstance(name, str)}
        if actual_names != set(reconstructed_assets):
            missing = sorted(set(reconstructed_assets) - actual_names)
            extra = sorted(actual_names - set(reconstructed_assets))
            raise ValueError(
                "dataset inputs do not match legacy scene reconstruction "
                f"(missing={missing}, extra={extra})"
            )

    loaded: list[tuple[str, ChannelDataset]] = []
    inventory: list[dict[str, Any]] = []
    migrated_datasets: list[str] = []
    migrated_rows = 0
    seen_names: set[str] = set()
    for name, path in inputs:
        if not isinstance(name, str) or not name or name != name.strip():
            raise ValueError("dataset names must be canonical nonempty strings")
        normalized_name = name
        if normalized_name in seen_names:
            raise ValueError(f"duplicate channel dataset name: {normalized_name}")
        seen_names.add(normalized_name)
        dataset_path = Path(os.path.abspath(path))
        expected: dict[str, Any] | None = None
        if declared_inventory is not None:
            expected = declared_inventory[normalized_name]
            if dataset_path != expected["absolute_path"]:
                raise ValueError(f"dataset path differs from frozen inventory: {normalized_name}")
        if expected is None:
            payload = _read_stable_regular_file(
                dataset_path,
                label=f"ChannelDataset {normalized_name!r}",
            )
        else:
            assert inventory_manifest is not None
            payload = _read_inventory_relative_file(
                Path(os.path.abspath(inventory_manifest)).parent,
                expected["path"],
                label=f"ChannelDataset {normalized_name!r}",
            )
        if expected is not None:
            if hashlib.sha256(payload).hexdigest() != expected["sha256"]:
                raise ValueError(
                    f"dataset SHA-256 differs from frozen inventory: {normalized_name}"
                )
            if len(payload) != expected["bytes"]:
                raise ValueError(
                    f"dataset byte count differs from frozen inventory: {normalized_name}"
                )
        channel_dataset = ChannelDataset.load_bytes(payload)
        if expected is not None and len(channel_dataset.y) != expected["rows"]:
            raise ValueError(f"dataset row count differs from frozen inventory: {normalized_name}")
        if derive_legacy_rt_split_identities:
            if reconstructed_assets is not None:
                assert reconstruction_metadata is not None
                channel_dataset = _attach_reconstructed_scene_assets(
                    channel_dataset,
                    dataset_name=normalized_name,
                    entry=reconstructed_assets[normalized_name],
                    input_sha256=hashlib.sha256(payload).hexdigest(),
                    input_bytes=len(payload),
                    manifest_metadata=reconstruction_metadata,
                )
            channel_dataset, was_upgraded = upgrade_legacy_rt_split_metadata(
                channel_dataset, dataset_name=normalized_name
            )
            if was_upgraded:
                migrated_datasets.append(normalized_name)
                migrated_rows += len(channel_dataset.y)
        loaded.append((normalized_name, channel_dataset))
        try:
            display_path = dataset_path.relative_to(_REPOSITORY_ROOT).as_posix()
        except ValueError:
            display_path = str(dataset_path)
        inventory.append(
            {
                "name": normalized_name,
                "path": display_path,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
                "rows": len(loaded[-1][1].y),
            }
        )

    subjects, footprints, lineage_report = derive_channel_dataset_split_evidence(
        loaded, spatial_buffer_m=spatial_buffer_m
    )
    inventory.sort(key=lambda item: item["name"])
    identity_derivation: dict[str, Any] = {
        "mode": (
            "opt_in_historical_rt_scene_asset_reconstruction"
            if reconstructed_assets is not None
            else "opt_in_complete_rt_legacy_metadata"
            if derive_legacy_rt_split_identities
            else "disabled"
        ),
        "datasets_upgraded_in_memory": migrated_datasets,
        "rows_upgraded_in_memory": migrated_rows,
        "input_hashes_cover_original_bytes": True,
    }
    if reconstruction_metadata is not None:
        identity_derivation["scene_asset_reconstruction"] = reconstruction_metadata
    report = {
        "schema": "loslab.channel_split_lineage_audit_bundle.v1",
        "status": "NOT_QUALIFIED",
        "spatial_buffer_m": float(spatial_buffer_m),
        "identity_derivation": identity_derivation,
        "input_completeness": (
            {
                "status": (
                    "matched_signed_frozen_inventory"
                    if "signature_sha256" in inventory_metadata
                    else "matched_frozen_inventory"
                ),
                **inventory_metadata,
                "matched_dataset_count": len(inventory),
                "authority": (
                    "OpenSSH signature verified against the caller-supplied allowed-signers file"
                    if "signature_sha256" in inventory_metadata
                    else "self-declared inventory; signature and external trust are not verified"
                ),
            }
            if inventory_metadata is not None
            else {
                "status": "unverified_input_scope",
                "authority": "no frozen corpus inventory supplied",
            }
        ),
        "inputs": inventory,
        "lineage": lineage_report,
        "subjects": [
            {
                "subject_id": subject.subject_id,
                **subject.to_dict(),
            }
            for subject in subjects
        ],
        "interpretation": (
            "This report covers only the listed input bytes. It does not prove corpus "
            "completeness, acquisition provenance, missing relations, independent review, "
            "or release qualification."
        ),
    }
    if split_registry_path is not None:
        assert split_registry_signature is not None
        assert trusted_split_allowed_signers is not None
        assert split_registry_signer_id is not None
        registry, signed_artifact = verify_signed_split_policy_registry_attestation(
            split_registry_path,
            split_registry_signature,
            trusted_allowed_signers_path=trusted_split_allowed_signers,
            signer_id=split_registry_signer_id,
        )
        if require_complete_split_audit and not registry.child_bindings:
            raise ValueError(
                "complete split audit requires at least one frozen child binding"
            )
        if lineage_report.get("subjects_with_provisional_split_identity", 0) != 0:
            raise ValueError(
                "signed split-registry audit refuses provisional RT scene identities; "
                "resolve complete scene-asset lineage before release-lock comparison"
            )
        parent_roster_audit = audit_split_policy_parent_roster(registry, subjects)
        footprint_audit = verify_registry_spatiotemporal_audit(
            registry, footprints, spatial_buffer_m=spatial_buffer_m
        )
        registry_payload = registry.to_dict()
        child_roster_reports: list[dict[str, Any]] = []
        child_artifact_reports: list[dict[str, Any]] = []
        if split_child_rosters:
            expected_kinds = {binding.child_kind for binding in registry.child_bindings.values()}
            supplied_kinds = set(child_roster_kinds)
            if supplied_kinds != expected_kinds:
                raise ValueError(
                    "split child roster files must cover every frozen child kind "
                    f"(missing={sorted(expected_kinds - supplied_kinds)}, "
                    f"extra={sorted(supplied_kinds - expected_kinds)})"
                )
            for child_kind, roster_path in sorted(
                split_child_rosters, key=lambda item: item[0]
            ):
                bindings, roster_metadata = _load_split_child_roster(
                    roster_path, expected_child_kind=child_kind
                )
                child_roster_reports.append(
                    {
                        **roster_metadata,
                        "audit": audit_split_policy_child_roster(
                            registry, bindings, child_kind=child_kind
                        ),
                    }
                )
        if split_child_artifact_rosters:
            expected_kinds = {binding.child_kind for binding in registry.child_bindings.values()}
            supplied_kinds = set(child_artifact_kinds)
            if supplied_kinds != expected_kinds:
                raise ValueError(
                    "split child artifact roster files must cover every frozen child kind "
                    f"(missing={sorted(expected_kinds - supplied_kinds)}, "
                    f"extra={sorted(supplied_kinds - expected_kinds)})"
                )
            for child_kind, artifact_roster_path in sorted(
                split_child_artifact_rosters, key=lambda item: item[0]
            ):
                assert trusted_split_allowed_signers is not None
                assert split_registry_signer_id is not None
                artifact_roster_signature = Path(f"{artifact_roster_path}.sig")
                verified_roster = verify_openssh_signed_artifact(
                    artifact_roster_path,
                    artifact_roster_signature,
                    trusted_allowed_signers_path=trusted_split_allowed_signers,
                    signer_id=split_registry_signer_id,
                    namespace=_CHILD_ARTIFACT_ROSTER_SIGNATURE_NAMESPACE,
                )
                expected_child_bindings = {
                    child_id: binding
                    for child_id, binding in registry.child_bindings.items()
                    if binding.child_kind == child_kind
                }
                child_artifact_reports.append(
                    _load_split_child_artifact_roster(
                        artifact_roster_path,
                        expected_child_kind=child_kind,
                        expected_child_bindings=expected_child_bindings,
                        verified_artifact=verified_roster,
                        split_registry=registry,
                    )
                )
        byte_only_artifact_count = sum(
            roster["audit"]["byte_only_artifact_count"]
            for roster in child_artifact_reports
        )
        semantic_artifact_count = sum(
            roster["audit"]["semantic_artifact_count"]
            for roster in child_artifact_reports
        )
        trajectory_sample_count = sum(
            roster["audit"]["trajectory_sample_count"]
            for roster in child_artifact_reports
        )
        trajectory_count = sum(
            roster["audit"]["trajectory_count"] for roster in child_artifact_reports
        )
        episode_count = sum(
            roster["audit"].get("episode_count", 0) for roster in child_artifact_reports
        )
        scenario_gbsm_run_count = sum(
            roster["audit"].get("scenario_gbsm_run_count", 0)
            for roster in child_artifact_reports
        )
        scenario_gbsm_parent_count = sum(
            roster["audit"].get("scenario_gbsm_parent_count", 0)
            for roster in child_artifact_reports
        )
        expected_child_kinds = {
            binding.child_kind for binding in registry.child_bindings.values()
        }
        if require_complete_split_audit:
            supplied_binding_kinds = {kind for kind, _ in split_child_rosters}
            supplied_artifact_kinds = {
                kind for kind, _ in split_child_artifact_rosters
            }
            if supplied_binding_kinds != expected_child_kinds:
                raise ValueError(
                    "complete split audit requires binding rosters for every frozen child kind "
                    f"(missing={sorted(expected_child_kinds - supplied_binding_kinds)}, "
                    f"extra={sorted(supplied_binding_kinds - expected_child_kinds)})"
                )
            if supplied_artifact_kinds != expected_child_kinds:
                raise ValueError(
                    "complete split audit requires artifact rosters for every frozen child kind "
                    f"(missing={sorted(expected_child_kinds - supplied_artifact_kinds)}, "
                    f"extra={sorted(supplied_artifact_kinds - expected_child_kinds)})"
                )
            if byte_only_artifact_count:
                raise ValueError(
                    "complete split audit rejects child artifacts without semantic validators"
                )
        roster_schemas = {
            roster["audit"]["roster_schema"] for roster in child_artifact_reports
        }
        has_umcd_episode_artifacts = bool(
            roster_schemas
            & {
                "umcd.channel_dataset_child_artifact_roster.v4",
                "umcd.channel_dataset_child_artifact_roster.v5",
                "umcd.channel_dataset_child_artifact_roster.v6",
            }
        )
        report["schema"] = (
            "loslab.channel_split_lineage_audit_bundle.v10"
            if "umcd.channel_dataset_child_artifact_roster.v6" in roster_schemas
            else "loslab.channel_split_lineage_audit_bundle.v9"
            if "umcd.channel_dataset_child_artifact_roster.v5" in roster_schemas
            else "loslab.channel_split_lineage_audit_bundle.v8"
            if has_umcd_episode_artifacts
            else "loslab.channel_split_lineage_audit_bundle.v7"
            if split_child_artifact_rosters
            else "loslab.channel_split_lineage_audit_bundle.v3"
            if split_child_rosters
            else "loslab.channel_split_lineage_audit_bundle.v2"
        )
        report["split_registry_verification"] = {
            "status": (
                "PASS_SIGNED_COMPLETE_SPLIT_INTEGRITY"
                if require_complete_split_audit
                else (
                    "PASS_SIGNED_PARENT_FOOTPRINT_CHILD_BINDINGS_AND_ARTIFACT_BYTES"
                    if byte_only_artifact_count
                    else "PASS_SIGNED_PARENT_FOOTPRINT_CHILD_BINDINGS_AND_SEMANTIC_ARTIFACT_ROWS"
                )
                if split_child_artifact_rosters
                else
                "PASS_SIGNED_PARENT_FOOTPRINT_AND_CHILD_BINDING_ROSTERS"
                if split_child_rosters
                else "PASS_SIGNED_PARENT_AND_FOOTPRINT_AUDIT"
            ),
            "schema": registry_payload["schema"],
            "registry_sha256": registry_payload["registry_sha256"],
            "signer_id": split_registry_signer_id,
            "signature_namespace": SPLIT_POLICY_SIGNATURE_NAMESPACE,
            "registry_file_sha256": signed_artifact.artifact_sha256,
            "signature_sha256": signed_artifact.signature_sha256,
            "allowed_signers_sha256": signed_artifact.allowed_signers_sha256,
            "parent_roster_audit": parent_roster_audit,
            "spatiotemporal_audit": footprint_audit,
            "children_audited": bool(split_child_artifact_rosters),
            "child_bindings_verified": bool(split_child_rosters),
            "child_artifact_existence_verified": bool(split_child_artifact_rosters),
            "child_artifact_bytes_verified": bool(split_child_artifact_rosters),
            "child_artifact_semantic_rows_verified": (
                bool(split_child_artifact_rosters) and byte_only_artifact_count == 0
            ),
            "child_artifact_semantic_count": semantic_artifact_count,
            "child_artifact_byte_only_count": byte_only_artifact_count,
            "child_artifact_trajectory_sample_count": trajectory_sample_count,
            "child_artifact_trajectory_count": trajectory_count,
            "child_artifact_episode_count": episode_count,
            "child_artifact_scenario_gbsm_run_count": scenario_gbsm_run_count,
            "child_artifact_scenario_gbsm_parent_count": scenario_gbsm_parent_count,
            "complete_split_integrity_audit": require_complete_split_audit,
            "child_roster_audits": child_roster_reports,
            "child_artifact_roster_audits": child_artifact_reports,
        }
    return report


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        type=_parse_dataset,
        required=True,
        metavar="NAME=PATH",
        help="saved ChannelDataset NPZ; repeat once per dataset/run",
    )
    parser.add_argument("--spatial-buffer-m", type=float, required=True)
    parser.add_argument(
        "--derive-legacy-rt-split-identities",
        action="store_true",
        help=(
            "opt in to derive missing RT split IDs in memory from complete saved scene closure, "
            "receiver positions, and matching block-size metadata"
        ),
    )
    parser.add_argument(
        "--legacy-scene-assets-manifest",
        type=Path,
        help=(
            "explicitly use a retrospective asset reconstruction bound to legacy input bytes; "
            "the report remains NOT_QUALIFIED"
        ),
    )
    parser.add_argument(
        "--inventory-manifest",
        type=Path,
        help="frozen JSON inventory that declares the complete dataset file set",
    )
    parser.add_argument("--inventory-signature", type=Path)
    parser.add_argument("--allowed-signers", type=Path)
    parser.add_argument("--inventory-signer-id")
    parser.add_argument("--split-registry", type=Path)
    parser.add_argument("--split-registry-signature", type=Path)
    parser.add_argument("--split-registry-allowed-signers", type=Path)
    parser.add_argument("--split-registry-signer-id")
    parser.add_argument(
        "--require-complete-split-audit",
        action="store_true",
        help=(
            "require signed corpus inventory and split registry plus complete, "
            "semantically validated child artifact rosters; does not qualify scientific evidence"
        ),
    )
    parser.add_argument(
        "--split-child-roster",
        action="append",
        type=_parse_split_child_roster,
        default=[],
        metavar="KIND=PATH",
        help="complete JSON roster for one frozen split-child kind; requires a signed registry",
    )
    parser.add_argument(
        "--split-child-artifact-roster",
        action="append",
        type=_parse_split_child_roster,
        default=[],
        metavar="KIND=PATH",
        help=(
            "complete JSON path/size/hash roster for one frozen child kind; "
            "requires PATH.sig signed by the split-registry signer under the child-artifact-roster namespace; "
            "artifact paths are relative to the roster directory and must identify ChannelDataset NPZ files "
            "whose row metadata matches the frozen child ID, kind, and parent"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")

    try:
        report = audit_channel_datasets(
            args.dataset,
            spatial_buffer_m=args.spatial_buffer_m,
            inventory_manifest=args.inventory_manifest,
            inventory_signature=args.inventory_signature,
            trusted_allowed_signers=args.allowed_signers,
            inventory_signer_id=args.inventory_signer_id,
            split_registry_path=args.split_registry,
            split_registry_signature=args.split_registry_signature,
            trusted_split_allowed_signers=args.split_registry_allowed_signers,
            split_registry_signer_id=args.split_registry_signer_id,
            split_child_rosters=args.split_child_roster,
            split_child_artifact_rosters=args.split_child_artifact_roster,
            require_complete_split_audit=args.require_complete_split_audit,
            derive_legacy_rt_split_identities=args.derive_legacy_rt_split_identities,
            legacy_scene_assets_manifest=args.legacy_scene_assets_manifest,
        )
    except (OSError, RuntimeError, TypeError, ValueError, RecursionError) as error:
        parser.exit(2, f"error: {error}\n")
    payload = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                "status": report["status"],
                "dataset_count": report["lineage"]["dataset_count"],
                "unique_split_subjects": report["lineage"]["unique_split_subjects"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
