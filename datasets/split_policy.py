"""Sparse, versioned leakage relations and immutable UMCD release splits.

The split subjects in this module are physical/statistical parents, not model
views. Network episode descendants and standardized draw observations inherit
their parent's assignment through ``split_for_child``.
"""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import stat
import tempfile
from dataclasses import dataclass, field
from enum import StrEnum
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from evaluation.independent_splits import make_independent_split

from datasets.network import NetworkEpisode
from datasets.standardized import StandardizedChannelRecord

_NAMESPACE_ORDER = (
    "asset_family",
    "site",
    "scene_realization",
    "trajectory_session",
    "capture_session",
    "counterfactual_family",
    "material_realization",
    "equipment_family",
    "spatial_buffer",
    "overlapping_time_window",
)
_SPLIT_ORDER = ("train", "validation", "calibration", "test")
_DEFAULT_FOUR_WAY = {"train": 0.7, "validation": 0.1, "calibration": 0.1, "test": 0.1}
_SPLIT_REGISTRY_V2_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "policy",
        "subjects",
        "components",
        "children",
        "report",
        "registry_sha256",
    }
)
_SPLIT_REGISTRY_V3_FIELDS = _SPLIT_REGISTRY_V2_FIELDS | {"spatiotemporal_audit"}
SPLIT_POLICY_SIGNATURE_NAMESPACE = "umcd-split-policy-v2@github.com"
MAX_SPLIT_POLICY_LOCK_BYTES = 256 * 1024 * 1024
MAX_SPLIT_POLICY_SIGNATURE_BYTES = 1024 * 1024
MAX_SPLIT_POLICY_ALLOWED_SIGNERS_BYTES = 4 * 1024 * 1024
_READ_CHUNK_BYTES = 1024 * 1024


def _require_canonical_identity(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a canonical nonempty string")
    return value


@dataclass(frozen=True)
class VerifiedOpenSSHArtifact:
    """Immutable artifact bytes verified under an external OpenSSH trust file."""

    payload: bytes
    artifact_sha256: str
    signature_sha256: str
    allowed_signers_sha256: str
    signer_id: str
    namespace: str


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject ambiguous JSON objects instead of silently keeping the last key."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _lock_stat_signature(metadata: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    """Return metadata fields that must remain stable during a lock read."""
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _fsync_directory(path: Path) -> None:
    """Persist directory-entry changes on filesystems that expose directory fsync."""
    if os.name != "posix" or not hasattr(os, "O_DIRECTORY"):
        return
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_immutable_lock_bytes(
    path: Path,
    *,
    artifact_label: str = "split-policy lock",
    max_bytes: int = MAX_SPLIT_POLICY_LOCK_BYTES,
) -> bytes:
    """Read one regular, single-link policy artifact without following leaf symlinks."""
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    path_metadata_before = os.lstat(path)
    if stat.S_ISLNK(path_metadata_before.st_mode):
        raise ValueError(f"{artifact_label} must not be a symlink")
    if path_metadata_before.st_size > max_bytes:
        raise ValueError(f"{artifact_label} exceeds its {max_bytes}-byte limit")

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ValueError(f"{artifact_label} must not be a symlink") from exc
        raise
    try:
        metadata = os.fstat(descriptor)
        path_metadata_after = os.lstat(path)
        if stat.S_ISLNK(path_metadata_after.st_mode):
            raise ValueError(f"{artifact_label} must not be a symlink")
        same_open_file = (
            (metadata.st_dev, metadata.st_ino)
            == (
                path_metadata_after.st_dev,
                path_metadata_after.st_ino,
            )
            == (path_metadata_before.st_dev, path_metadata_before.st_ino)
        )
        if not same_open_file:
            raise ValueError(f"{artifact_label} path changed while opening")
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or not stat.S_ISREG(path_metadata_after.st_mode)
            or path_metadata_after.st_nlink != 1
        ):
            raise ValueError(f"{artifact_label} must be a single-link regular file")
        if metadata.st_size > max_bytes:
            raise ValueError(f"{artifact_label} exceeds its {max_bytes}-byte limit")
        if _lock_stat_signature(metadata) != _lock_stat_signature(
            path_metadata_before
        ) or _lock_stat_signature(metadata) != _lock_stat_signature(path_metadata_after):
            raise ValueError(f"{artifact_label} changed while opening")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            chunks: list[bytes] = []
            total_bytes = 0
            while True:
                chunk = stream.read(min(_READ_CHUNK_BYTES, max_bytes + 1 - total_bytes))
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise ValueError(f"{artifact_label} exceeds its {max_bytes}-byte limit")
                chunks.append(chunk)
            contents = b"".join(chunks)
            metadata_after_read = os.fstat(stream.fileno())
            path_metadata_after_read = os.lstat(path)
            if stat.S_ISLNK(path_metadata_after_read.st_mode):
                raise ValueError(f"{artifact_label} must not be a symlink")
            same_open_file_after_read = (
                (metadata_after_read.st_dev, metadata_after_read.st_ino)
                == (metadata.st_dev, metadata.st_ino)
                == (path_metadata_after_read.st_dev, path_metadata_after_read.st_ino)
            )
            if not same_open_file_after_read:
                raise ValueError(f"{artifact_label} path changed while reading")
            if (
                not stat.S_ISREG(metadata_after_read.st_mode)
                or metadata_after_read.st_nlink != 1
                or not stat.S_ISREG(path_metadata_after_read.st_mode)
                or path_metadata_after_read.st_nlink != 1
            ):
                raise ValueError(f"{artifact_label} must be a single-link regular file")
            if (
                _lock_stat_signature(metadata_after_read) != _lock_stat_signature(metadata)
                or _lock_stat_signature(path_metadata_after_read) != _lock_stat_signature(metadata)
                or len(contents) != metadata.st_size
            ):
                raise ValueError(f"{artifact_label} changed while reading")
            return contents
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _verify_openssh_signature_bytes(
    payload: bytes,
    signature: bytes,
    allowed_signers: bytes,
    *,
    signer_id: str,
    namespace: str,
) -> None:
    import shutil
    import subprocess

    executable = shutil.which("ssh-keygen")
    if executable is None:
        raise RuntimeError("OpenSSH ssh-keygen is required to verify signed artifacts")
    with tempfile.TemporaryDirectory(prefix="umcd-openssh-verify-") as temporary_directory:
        snapshot = Path(temporary_directory)
        artifact_path = snapshot / "artifact.bin"
        signature_path = snapshot / "artifact.bin.sig"
        allowed_signers_path = snapshot / "allowed_signers"
        artifact_path.write_bytes(payload)
        signature_path.write_bytes(signature)
        allowed_signers_path.write_bytes(allowed_signers)
        try:
            result = subprocess.run(
                [
                    executable,
                    "-Y",
                    "verify",
                    "-f",
                    str(allowed_signers_path),
                    "-I",
                    signer_id,
                    "-n",
                    namespace,
                    "-s",
                    str(signature_path),
                ],
                check=False,
                capture_output=True,
                input=payload,
                timeout=30,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("OpenSSH signature verification timed out") from error
        if result.returncode != 0:
            raise ValueError("artifact signature is invalid or signer is not trusted")


def verify_openssh_signed_artifact(
    path: str | Path,
    signature_path: str | Path,
    *,
    trusted_allowed_signers_path: str | Path,
    signer_id: str,
    namespace: str,
) -> VerifiedOpenSSHArtifact:
    """Verify arbitrary artifact bytes against an independently supplied OpenSSH trust file."""
    for field_name, value in (("signer_id", signer_id), ("namespace", namespace)):
        _require_canonical_identity(value, field_name)
        if any(character in value for character in ("\n", "\r", "\x00")):
            raise ValueError(f"{field_name} must be a single-line value")
    payload = _read_immutable_lock_bytes(
        Path(path), artifact_label="signed artifact", max_bytes=MAX_SPLIT_POLICY_LOCK_BYTES
    )
    signature = _read_immutable_lock_bytes(
        Path(signature_path),
        artifact_label="artifact signature",
        max_bytes=MAX_SPLIT_POLICY_SIGNATURE_BYTES,
    )
    allowed_signers = _read_immutable_lock_bytes(
        Path(trusted_allowed_signers_path),
        artifact_label="trusted artifact keyring",
        max_bytes=MAX_SPLIT_POLICY_ALLOWED_SIGNERS_BYTES,
    )
    _verify_openssh_signature_bytes(
        payload,
        signature,
        allowed_signers,
        signer_id=signer_id,
        namespace=namespace,
    )
    return VerifiedOpenSSHArtifact(
        payload=payload,
        artifact_sha256=hashlib.sha256(payload).hexdigest(),
        signature_sha256=hashlib.sha256(signature).hexdigest(),
        allowed_signers_sha256=hashlib.sha256(allowed_signers).hexdigest(),
        signer_id=signer_id,
        namespace=namespace,
    )


class LeakageNamespace(StrEnum):
    """Field-qualified identities that induce conservative component linkage."""

    ASSET_FAMILY = "asset_family"
    SITE = "site"
    SCENE_REALIZATION = "scene_realization"
    TRAJECTORY_SESSION = "trajectory_session"
    CAPTURE_SESSION = "capture_session"
    COUNTERFACTUAL_FAMILY = "counterfactual_family"
    MATERIAL_REALIZATION = "material_realization"
    EQUIPMENT_FAMILY = "equipment_family"
    SPATIAL_BUFFER = "spatial_buffer"
    OVERLAPPING_TIME_WINDOW = "overlapping_time_window"


class SplitParentType(StrEnum):
    """Supported statistical parent units in Release Split Policy v2."""

    NETWORK_EPISODE = "network_episode"
    STANDARDIZED_CHANNEL_DRAW = "standardized_channel_draw"


@dataclass(frozen=True, slots=True, order=True)
class LeakageRelation:
    """One sparse, namespace-qualified relation; equal IDs in different fields differ."""

    namespace: LeakageNamespace
    identity: str

    def __post_init__(self) -> None:
        if not isinstance(self.namespace, LeakageNamespace):
            try:
                object.__setattr__(self, "namespace", LeakageNamespace(self.namespace))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"unsupported leakage namespace: {self.namespace!r}") from exc
        _require_canonical_identity(self.identity, "leakage relation identity")

    @property
    def qualified_identity(self) -> str:
        """Human-readable field-qualified value for reports."""
        return f"{self.namespace.value}:{self.identity}"

    def to_dict(self) -> dict[str, str]:
        return {"namespace": self.namespace.value, "identity": self.identity}


@dataclass(frozen=True, slots=True)
class SplitSubject:
    """A split parent with only the leakage relations that actually apply."""

    subject_id: str
    parent_type: SplitParentType
    source_family: str
    relations: tuple[LeakageRelation, ...] = ()
    partition_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("subject_id", "source_family"):
            value = getattr(self, name)
            _require_canonical_identity(value, name)
        if not isinstance(self.parent_type, SplitParentType):
            try:
                object.__setattr__(self, "parent_type", SplitParentType(self.parent_type))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"unsupported split parent type: {self.parent_type!r}") from exc
        relations = tuple(self.relations)
        if any(not isinstance(item, LeakageRelation) for item in relations):
            raise TypeError("relations must contain LeakageRelation values")
        keys = [(item.namespace, item.identity) for item in relations]
        if len(set(keys)) != len(keys):
            raise ValueError("a split subject cannot repeat a leakage relation")
        object.__setattr__(self, "relations", tuple(sorted(relations)))
        partition_id = self.subject_id if self.partition_id is None else self.partition_id
        _require_canonical_identity(partition_id, "partition_id")
        object.__setattr__(self, "partition_id", partition_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_type": self.parent_type.value,
            "source_family": self.source_family,
            "partition_id": self.partition_id,
            "relations": [item.to_dict() for item in self.relations],
        }


@dataclass(frozen=True, slots=True)
class SpatiotemporalFootprint:
    """Comparable 3D sample positions and an optional half-open time window.

    Positions must use meters in a shared, named coordinate frame. Time
    windows must use seconds on a shared, named clock. These identifiers keep
    unrelated coordinate systems and clock domains from being joined.
    """

    subject_id: str
    positions_m: tuple[tuple[float, float, float], ...] = ()
    coordinate_frame_id: str | None = None
    time_scope_id: str | None = None
    start_time_s: float | None = None
    end_time_s: float | None = None

    def __post_init__(self) -> None:
        _require_canonical_identity(self.subject_id, "subject_id")
        if any(
            isinstance(value, bool) or not isinstance(value, Real)
            for point in self.positions_m
            for value in point
        ):
            raise TypeError("positions_m coordinates must be real numbers")
        positions = tuple(tuple(float(value) for value in point) for point in self.positions_m)
        if any(len(point) != 3 for point in positions):
            raise ValueError("positions_m must contain 3D points")
        if any(not math.isfinite(value) for point in positions for value in point):
            raise ValueError("positions_m coordinates must be finite")
        object.__setattr__(self, "positions_m", positions)
        if positions:
            if (
                not isinstance(self.coordinate_frame_id, str)
                or not self.coordinate_frame_id.strip()
            ):
                raise ValueError("coordinate_frame_id is required when positions_m are present")
            _require_canonical_identity(self.coordinate_frame_id, "coordinate_frame_id")
        elif self.coordinate_frame_id is not None:
            raise ValueError("coordinate_frame_id requires at least one position")

        time_values = (self.time_scope_id, self.start_time_s, self.end_time_s)
        if all(value is None for value in time_values):
            return
        if not isinstance(self.time_scope_id, str) or not self.time_scope_id.strip():
            raise ValueError("time_scope_id is required when a time window is present")
        _require_canonical_identity(self.time_scope_id, "time_scope_id")
        if self.start_time_s is None or self.end_time_s is None:
            raise ValueError("time windows require both start_time_s and end_time_s")
        if any(
            isinstance(value, bool) or not isinstance(value, Real)
            for value in (self.start_time_s, self.end_time_s)
        ):
            raise TypeError("time window endpoints must be real numbers")
        start, end = float(self.start_time_s), float(self.end_time_s)
        if not math.isfinite(start) or not math.isfinite(end) or start >= end:
            raise ValueError("time window must be finite and satisfy start_time_s < end_time_s")
        object.__setattr__(self, "start_time_s", start)
        object.__setattr__(self, "end_time_s", end)


def derive_spatiotemporal_leakage_relations(
    footprints: Sequence[SpatiotemporalFootprint], *, spatial_buffer_m: float
) -> Mapping[str, tuple[LeakageRelation, ...]]:
    """Derive connected leakage relations before assigning release splits.

    Spatially linked subjects have at least one pair of sampled positions at
    Euclidean distance ``<= spatial_buffer_m`` in the same coordinate frame.
    Time windows are half-open, so touching endpoints do not overlap. Spatial
    and temporal links are transitively grouped; this can conservatively merge
    a chain of pairwise-related subjects into one split component.
    """
    if isinstance(spatial_buffer_m, bool) or not isinstance(spatial_buffer_m, (float, int)):
        raise TypeError("spatial_buffer_m must be a real number")
    buffer_m = float(spatial_buffer_m)
    if not math.isfinite(buffer_m) or buffer_m < 0:
        raise ValueError("spatial_buffer_m must be finite and non-negative")
    by_id: dict[str, SpatiotemporalFootprint] = {}
    for footprint in footprints:
        if not isinstance(footprint, SpatiotemporalFootprint):
            raise TypeError("footprints must contain SpatiotemporalFootprint values")
        if footprint.subject_id in by_id:
            raise ValueError(f"duplicate spatiotemporal footprint: {footprint.subject_id}")
        by_id[footprint.subject_id] = footprint

    def cell(point: tuple[float, float, float]) -> tuple[int, int, int]:
        if buffer_m == 0:
            raise RuntimeError("zero-radius points use exact-coordinate lookup")

        def coordinate_cell(coordinate: float) -> int:
            quotient = coordinate / buffer_m
            if math.isfinite(quotient):
                return math.floor(quotient)
            numerator, denominator = coordinate.as_integer_ratio()
            buffer_numerator, buffer_denominator = buffer_m.as_integer_ratio()
            return (numerator * buffer_denominator) // (denominator * buffer_numerator)

        return (
            coordinate_cell(point[0]),
            coordinate_cell(point[1]),
            coordinate_cell(point[2]),
        )

    spatial_buckets: dict[
        tuple[str, tuple[int, int, int] | tuple[float, float, float]],
        list[tuple[str, tuple[float, float, float]]],
    ] = {}
    neighbor_offsets = tuple((x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1))
    spatial_component_parent = {subject_id: subject_id for subject_id in by_id}

    def find_spatial(subject_id: str) -> str:
        while spatial_component_parent[subject_id] != subject_id:
            spatial_component_parent[subject_id] = spatial_component_parent[
                spatial_component_parent[subject_id]
            ]
            subject_id = spatial_component_parent[subject_id]
        return subject_id

    def union_spatial(left: str, right: str) -> None:
        left_root, right_root = find_spatial(left), find_spatial(right)
        if left_root != right_root:
            spatial_component_parent[max(left_root, right_root)] = min(left_root, right_root)

    for subject_id in sorted(by_id):
        footprint = by_id[subject_id]
        if not footprint.positions_m:
            continue
        assert footprint.coordinate_frame_id is not None
        for point in footprint.positions_m:
            candidate_keys: tuple[tuple[int, int, int] | tuple[float, float, float], ...]
            if buffer_m == 0:
                key: tuple[int, int, int] | tuple[float, float, float] = point
                candidate_keys = (key,)
            else:
                current = cell(point)
                candidate_keys = tuple(
                    (
                        current[0] + offset[0],
                        current[1] + offset[1],
                        current[2] + offset[2],
                    )
                    for offset in neighbor_offsets
                )
            for candidate_key in candidate_keys:
                bucket_key = (footprint.coordinate_frame_id, candidate_key)
                for other_id, other_point in spatial_buckets.get(bucket_key, ()):
                    if other_id == subject_id:
                        continue
                    if math.dist(point, other_point) <= buffer_m:
                        union_spatial(subject_id, other_id)
            own_key: tuple[int, int, int] | tuple[float, float, float]
            own_key = point if buffer_m == 0 else cell(point)
            spatial_buckets.setdefault((footprint.coordinate_frame_id, own_key), []).append(
                (subject_id, point)
            )

    spatial_members: dict[str, list[str]] = {}
    for subject_id in sorted(by_id):
        if by_id[subject_id].positions_m:
            spatial_members.setdefault(find_spatial(subject_id), []).append(subject_id)

    time_scopes: dict[str, list[tuple[float, float, str]]] = {}
    for subject_id, footprint in by_id.items():
        if footprint.time_scope_id is not None:
            assert footprint.start_time_s is not None and footprint.end_time_s is not None
            time_scopes.setdefault(footprint.time_scope_id, []).append(
                (footprint.start_time_s, footprint.end_time_s, subject_id)
            )

    time_groups: list[tuple[str, list[str]]] = []
    for scope_id, intervals in sorted(time_scopes.items()):
        intervals.sort()
        group: list[str] = []
        group_end = -math.inf
        for start, end, subject_id in intervals:
            if group and start >= group_end:
                if len(group) > 1:
                    time_groups.append((scope_id, group))
                group = []
                group_end = -math.inf
            group.append(subject_id)
            group_end = max(group_end, end)
        if len(group) > 1:
            time_groups.append((scope_id, group))

    result: dict[str, list[LeakageRelation]] = {subject_id: [] for subject_id in by_id}

    def add_component_relations(
        namespace: LeakageNamespace,
        scope_id: str,
        members: Sequence[str],
        evidence: Mapping[str, Any],
    ) -> None:
        if len(members) < 2:
            return
        canonical = _canonical_json(
            {
                "algorithm": "umcd.spatiotemporal-leakage.v2",
                "namespace": namespace.value,
                "scope": scope_id,
                "subjects": sorted(members),
                "evidence": evidence,
            }
        )
        identity = hashlib.sha256(canonical).hexdigest()
        relation = LeakageRelation(namespace, f"v2:{scope_id}:{identity}")
        for subject_id in members:
            result[subject_id].append(relation)

    for component in spatial_members.values():
        if len(component) > 1:
            scope = by_id[component[0]].coordinate_frame_id
            assert scope is not None
            evidence = {
                "spatial_buffer_m": buffer_m,
                "positions_m": {
                    subject_id: sorted(by_id[subject_id].positions_m)
                    for subject_id in sorted(component)
                },
            }
            add_component_relations(
                LeakageNamespace.SPATIAL_BUFFER, scope, component, evidence
            )
    for scope, component in time_groups:
        evidence = {
            "interval_semantics": "half-open",
            "time_windows_s": {
                subject_id: [by_id[subject_id].start_time_s, by_id[subject_id].end_time_s]
                for subject_id in sorted(component)
            },
        }
        add_component_relations(
            LeakageNamespace.OVERLAPPING_TIME_WINDOW, scope, component, evidence
        )

    return MappingProxyType(
        {subject_id: tuple(sorted(relations)) for subject_id, relations in sorted(result.items())}
    )


def audit_spatiotemporal_leakage_relations(
    footprints: Sequence[SpatiotemporalFootprint],
    subjects: Sequence[SplitSubject],
    *,
    spatial_buffer_m: float,
) -> dict[str, Any]:
    """Re-derive and verify spatial/time relations against frozen parent subjects.

    The relation identities bind the exact geometry/time evidence, derivation
    version, and spatial buffer. This audit is still required to establish
    that registered relations were actually derived from the supplied
    authoritative footprint inventory.
    """
    derived = derive_spatiotemporal_leakage_relations(
        footprints, spatial_buffer_m=spatial_buffer_m
    )
    subject_by_id: dict[str, SplitSubject] = {}
    for subject in subjects:
        if not isinstance(subject, SplitSubject):
            raise TypeError("subjects must contain SplitSubject values")
        if subject.subject_id in subject_by_id:
            raise ValueError(f"duplicate split subject in footprint audit: {subject.subject_id}")
        subject_by_id[subject.subject_id] = subject
    if set(subject_by_id) != set(derived):
        raise ValueError("footprints and split subjects must cover exactly the same parents")

    namespaces = {
        LeakageNamespace.SPATIAL_BUFFER,
        LeakageNamespace.OVERLAPPING_TIME_WINDOW,
    }
    for subject_id, subject in subject_by_id.items():
        observed = {item for item in subject.relations if item.namespace in namespaces}
        expected = set(derived[subject_id])
        if observed != expected:
            raise ValueError(
                f"derived spatial/time relations do not match footprint evidence for {subject_id}"
            )

    footprint_payload = {
        item.subject_id: {
            "positions_m": sorted(item.positions_m),
            "coordinate_frame_id": item.coordinate_frame_id,
            "time_scope_id": item.time_scope_id,
            "start_time_s": item.start_time_s,
            "end_time_s": item.end_time_s,
        }
        for item in sorted(footprints, key=lambda item: item.subject_id)
    }
    relation_payload = {
        subject_id: [item.to_dict() for item in sorted(derived[subject_id])]
        for subject_id in sorted(derived)
    }
    report: dict[str, Any] = {
        "schema": "umcd.spatiotemporal_lineage_audit.v2",
        "status": "PASS_INPUT_SCOPED",
        "spatial_buffer_m": float(spatial_buffer_m),
        "derivation_algorithm": "umcd.spatiotemporal-leakage.v2",
        "parent_count": len(subject_by_id),
        "subject_ids_sha256": hashlib.sha256(
            _canonical_json(sorted(subject_by_id))
        ).hexdigest(),
        "spatially_linked_parent_count": sum(
            any(
                relation.namespace is LeakageNamespace.SPATIAL_BUFFER
                for relation in derived[item]
            )
            for item in derived
        ),
        "temporally_linked_parent_count": sum(
            any(
                relation.namespace is LeakageNamespace.OVERLAPPING_TIME_WINDOW
                for relation in derived[item]
            )
            for item in derived
        ),
        "footprints_sha256": hashlib.sha256(_canonical_json(footprint_payload)).hexdigest(),
        "derived_relations_sha256": hashlib.sha256(_canonical_json(relation_payload)).hexdigest(),
    }
    report["audit_sha256"] = hashlib.sha256(_canonical_json(report)).hexdigest()
    return report


def bind_spatiotemporal_audit_to_registry(
    registry: SplitPolicyRegistry,
    footprints: Sequence[SpatiotemporalFootprint],
    *,
    spatial_buffer_m: float,
) -> SplitPolicyRegistry:
    """Return a v3 registry lock that binds the complete footprint audit digest."""
    if not isinstance(registry, SplitPolicyRegistry):
        raise TypeError("registry must be a SplitPolicyRegistry")
    audit = audit_spatiotemporal_leakage_relations(
        footprints,
        tuple(registry.subjects.values()),
        spatial_buffer_m=spatial_buffer_m,
    )
    if registry.spatiotemporal_audit is not None and dict(registry.spatiotemporal_audit) != audit:
        raise ValueError("frozen registry spatiotemporal audit cannot be replaced")
    return SplitPolicyRegistry(
        policy=registry.policy,
        subject_to_component=registry.subject_to_component,
        component_to_split=registry.component_to_split,
        subjects=registry.subjects,
        child_bindings=registry.child_bindings,
        spatiotemporal_audit=audit,
    )


def verify_registry_spatiotemporal_audit(
    registry: SplitPolicyRegistry,
    footprints: Sequence[SpatiotemporalFootprint],
    *,
    spatial_buffer_m: float,
) -> dict[str, Any]:
    """Recompute and compare the footprint inventory committed by a v3 registry."""
    if registry.spatiotemporal_audit is None:
        raise ValueError("registry has no frozen spatiotemporal footprint audit")
    observed = audit_spatiotemporal_leakage_relations(
        footprints,
        tuple(registry.subjects.values()),
        spatial_buffer_m=spatial_buffer_m,
    )
    if dict(registry.spatiotemporal_audit) != observed:
        raise ValueError("spatiotemporal footprint inventory differs from frozen registry audit")
    return observed


def network_episode_subject(
    episode: NetworkEpisode,
    *,
    relations: Sequence[LeakageRelation] = (),
) -> SplitSubject:
    """Adapt a network episode; its legacy split group remains a family relation.

    Additional site, scene, trajectory, equipment, or material lineage must be
    supplied explicitly by the caller. ``family_id`` is reporting metadata and
    does not itself join episodes.
    """
    if not isinstance(episode, NetworkEpisode):
        raise TypeError("episode must be a NetworkEpisode")
    return SplitSubject(
        subject_id=episode.episode_id,
        parent_type=SplitParentType.NETWORK_EPISODE,
        source_family=episode.family_id,
        relations=(
            LeakageRelation(LeakageNamespace.ASSET_FAMILY, episode.split_group_id),
            *tuple(relations),
        ),
        partition_id=episode.episode_id,
    )


def standardized_draw_subject(
    record: StandardizedChannelRecord,
    *,
    relations: Sequence[LeakageRelation] = (),
    partition_id: str | None = None,
) -> SplitSubject:
    """Adapt one draw without ranking on profile/category/target metadata.

    The default opaque key is derived from the generation seed and sample
    index, never profile, target, or source-family labels. If independent
    generator streams reuse that seed/index pair, callers must provide a
    stable opaque ``partition_id`` for each draw; registry construction rejects
    collisions instead of silently joining unrelated parents.
    """
    if not isinstance(record, StandardizedChannelRecord):
        raise TypeError("record must be a StandardizedChannelRecord")
    if partition_id is None:
        if "partition_id" in record.metadata:
            partition_id = _require_canonical_identity(
                record.metadata["partition_id"], "standardized partition_id"
            )
        else:
            seed = record.metadata.get("seed")
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError(
                    "standardized draw needs generation seed metadata or an explicit partition_id"
                )
            opaque_payload = _canonical_json(
                {"draw_seed": seed, "sample_index": record.sample_index}
            )
            partition_id = "standardized-draw-" + hashlib.sha256(opaque_payload).hexdigest()
    return SplitSubject(
        subject_id=record.sample_id,
        parent_type=SplitParentType.STANDARDIZED_CHANNEL_DRAW,
        source_family=f"standardized.{record.family.value}",
        relations=tuple(relations),
        partition_id=partition_id,
    )


@dataclass(frozen=True, slots=True)
class SplitPolicy:
    """Immutable assignment contract; its digest changes with every policy choice."""

    seed: str
    split_fractions: Mapping[str, float]
    relation_namespaces: tuple[LeakageNamespace, ...] = tuple(LeakageNamespace)
    policy_version: str = "umcd.release_split.v2"
    component_algorithm: str = "sha256-connected-component-v2"

    def __post_init__(self) -> None:
        _require_canonical_identity(self.seed, "policy seed")
        _require_canonical_identity(self.policy_version, "policy_version")
        if self.component_algorithm != "sha256-connected-component-v2":
            raise ValueError("unsupported component assignment algorithm")
        fractions = dict(self.split_fractions)
        if set(fractions) not in (
            {"train", "validation", "test"},
            {"train", "validation", "calibration", "test"},
        ):
            raise ValueError("split fractions must define the three-way or four-way split set")
        ordered_fractions: dict[str, float] = {}
        for name in _SPLIT_ORDER:
            if name not in fractions:
                continue
            fraction = fractions[name]
            if isinstance(fraction, bool) or not isinstance(fraction, Real):
                raise TypeError("split fractions must be real numbers")
            try:
                ordered_fractions[name] = float(fraction)
            except OverflowError as error:
                raise ValueError("split fractions must be finite and positive") from error
        if any(not 0 < value < 1 for value in ordered_fractions.values()):
            raise ValueError("split fractions must be finite and positive")
        if any(not math.isfinite(value) for value in ordered_fractions.values()):
            raise ValueError("split fractions must be finite and positive")
        if abs(sum(ordered_fractions.values()) - 1.0) > 1e-9:
            raise ValueError("split fractions must sum to one")
        namespaces = tuple(
            item if isinstance(item, LeakageNamespace) else LeakageNamespace(item)
            for item in self.relation_namespaces
        )
        if not namespaces or len(set(namespaces)) != len(namespaces):
            raise ValueError("relation_namespaces must be non-empty and unique")
        namespaces = tuple(sorted(namespaces, key=lambda item: _NAMESPACE_ORDER.index(item.value)))
        object.__setattr__(self, "split_fractions", MappingProxyType(ordered_fractions))
        object.__setattr__(self, "relation_namespaces", namespaces)

    @classmethod
    def release_v2(cls, *, seed: str) -> SplitPolicy:
        """Construct the four-way default release policy."""
        return cls(seed=seed, split_fractions=_DEFAULT_FOUR_WAY)

    def to_dict(self) -> dict[str, Any]:
        core = {
            "schema": "umcd.split_policy.v2",
            "schema_version": 2,
            "policy_version": self.policy_version,
            "seed": self.seed,
            "split_fractions": dict(self.split_fractions),
            "relation_namespaces": [item.value for item in self.relation_namespaces],
            "component_algorithm": self.component_algorithm,
        }
        return {**core, "policy_sha256": hashlib.sha256(_canonical_json(core)).hexdigest()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SplitPolicy:
        if value.get("schema") != "umcd.split_policy.v2" or value.get("schema_version") != 2:
            raise ValueError("unsupported UMCD split policy schema")
        core = {key: item for key, item in value.items() if key != "policy_sha256"}
        if hashlib.sha256(_canonical_json(core)).hexdigest() != value.get("policy_sha256"):
            raise ValueError("UMCD split policy hash is invalid")
        policy = cls(
            seed=value["seed"],
            split_fractions=value["split_fractions"],
            relation_namespaces=tuple(value["relation_namespaces"]),
            policy_version=value["policy_version"],
            component_algorithm=value["component_algorithm"],
        )
        if policy.to_dict() != dict(value):
            raise ValueError("UMCD split policy representation is not canonical")
        return policy


@dataclass(frozen=True, slots=True)
class SplitChildBinding:
    """A persisted release child linked to exactly one frozen parent."""

    child_id: str
    parent_subject_id: str
    child_kind: str

    def __post_init__(self) -> None:
        for name in ("child_id", "parent_subject_id", "child_kind"):
            value = getattr(self, name)
            _require_canonical_identity(value, name)

    def to_dict(self) -> dict[str, str]:
        return {
            "parent_subject_id": self.parent_subject_id,
            "child_kind": self.child_kind,
        }


def derive_split_components(
    subjects: Sequence[SplitSubject],
) -> tuple[tuple[SplitSubject, ...], ...]:
    """Return deterministic connected components without assigning split partitions."""
    if not subjects:
        raise ValueError("at least one split subject is required")
    subject_by_id: dict[str, SplitSubject] = {}
    for subject in subjects:
        if not isinstance(subject, SplitSubject):
            raise TypeError("subjects must contain SplitSubject values")
        if subject.subject_id in subject_by_id:
            raise ValueError(f"duplicate split subject ID: {subject.subject_id}")
        subject_by_id[subject.subject_id] = subject
    parent = {subject_id: subject_id for subject_id in subject_by_id}

    def find(subject_id: str) -> str:
        root = subject_id
        while parent[root] != root:
            parent[root] = parent[parent[root]]
            root = parent[root]
        while parent[subject_id] != subject_id:
            next_id = parent[subject_id]
            parent[subject_id] = root
            subject_id = next_id
        return root

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    first_subject_for_relation: dict[tuple[LeakageNamespace, str], str] = {}
    for subject_id in sorted(subject_by_id):
        for relation in subject_by_id[subject_id].relations:
            key = (relation.namespace, relation.identity)
            previous = first_subject_for_relation.setdefault(key, subject_id)
            union(subject_id, previous)

    members: dict[str, list[str]] = {}
    for subject_id in sorted(subject_by_id):
        members.setdefault(find(subject_id), []).append(subject_id)
    return tuple(
        tuple(subject_by_id[subject_id] for subject_id in component)
        for component in sorted(members.values(), key=lambda items: items[0])
    )


def _derive_registry_assignments(
    subjects: Mapping[str, SplitSubject], policy: SplitPolicy
) -> tuple[dict[str, str], dict[str, str]]:
    """Rebuild canonical components and assignments from frozen inputs only."""
    if not isinstance(policy, SplitPolicy):
        raise TypeError("policy must be a SplitPolicy")
    if not subjects:
        raise ValueError("at least one split subject is required")
    subject_by_id = dict(subjects)
    if any(not isinstance(subject, SplitSubject) for subject in subject_by_id.values()):
        raise TypeError("subjects must contain SplitSubject values")
    if any(key != subject.subject_id for key, subject in subject_by_id.items()):
        raise ValueError("subject mapping keys must equal subject IDs")
    if len({subject.partition_id for subject in subject_by_id.values()}) != len(subject_by_id):
        raise ValueError("partition_id values must uniquely identify parent subjects")
    for subject in subject_by_id.values():
        unsupported = set(item.namespace for item in subject.relations) - set(
            policy.relation_namespaces
        )
        if unsupported:
            raise ValueError(f"subject has relations excluded by policy: {sorted(unsupported)}")

    subject_to_component: dict[str, str] = {}
    for component in derive_split_components(tuple(subject_by_id.values())):
        partition_ids: list[str] = []
        for subject in component:
            if subject.partition_id is None:
                raise ValueError("split subject is missing its canonical partition ID")
            partition_ids.append(subject.partition_id)
        partition_ids.sort()
        component_id = "component-" + hashlib.sha256(_canonical_json(partition_ids)).hexdigest()
        for subject in component:
            subject_to_component[subject.subject_id] = component_id

    component_ids = sorted(set(subject_to_component.values()))
    required_component_count = len(policy.split_fractions)
    if len(component_ids) < required_component_count:
        missing_component_count = required_component_count - len(component_ids)
        raise ValueError(
            f"split policy requires at least {required_component_count} independent leakage "
            f"components for {required_component_count}-way assignment; found "
            f"{len(component_ids)}, need {missing_component_count} more"
        )
    independent = make_independent_split(
        [
            {"sample_id": component_id, "component_id": component_id}
            for component_id in component_ids
        ],
        seed=policy.seed,
        group_field="component_id",
        fractions=policy.split_fractions,
    )
    if set(independent.group_to_split.values()) != set(policy.split_fractions):
        raise ValueError("deterministic assignment omitted a required split partition")
    return subject_to_component, dict(independent.group_to_split)


@dataclass(frozen=True, slots=True)
class SplitPolicyRegistry:
    """Frozen assignments and relation snapshots for parent subjects."""

    policy: SplitPolicy
    subject_to_component: Mapping[str, str]
    component_to_split: Mapping[str, str]
    subjects: Mapping[str, SplitSubject]
    child_bindings: Mapping[str, SplitChildBinding] = field(default_factory=dict)
    spatiotemporal_audit: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        subject_components = dict(self.subject_to_component)
        component_splits = dict(self.component_to_split)
        subjects = dict(self.subjects)
        children = dict(self.child_bindings)
        lineage_audit = (
            None if self.spatiotemporal_audit is None else dict(self.spatiotemporal_audit)
        )
        if not subject_components or set(subject_components) != set(subjects):
            raise ValueError("registry must assign every registered parent subject")
        expected_subject_components, expected_component_splits = _derive_registry_assignments(
            subjects, self.policy
        )
        if subject_components != expected_subject_components:
            raise ValueError("registry components do not match canonical relation reconstruction")
        if component_splits != expected_component_splits:
            raise ValueError(
                "registry assignments do not match deterministic policy reconstruction"
            )
        for key, binding in children.items():
            if not isinstance(binding, SplitChildBinding):
                raise TypeError("child_bindings must contain SplitChildBinding values")
            if key != binding.child_id:
                raise ValueError("child mapping keys must equal child IDs")
            if binding.child_id in subjects:
                raise ValueError(
                    f"child ID collides with a parent subject ID: {binding.child_id}"
                )
            if binding.parent_subject_id not in subjects:
                raise ValueError(
                    f"child references an unknown split parent: {binding.parent_subject_id}"
                )
        if lineage_audit is not None:
            expected_audit_keys = {
                "schema",
                "status",
                "spatial_buffer_m",
                "derivation_algorithm",
                "parent_count",
                "subject_ids_sha256",
                "spatially_linked_parent_count",
                "temporally_linked_parent_count",
                "footprints_sha256",
                "derived_relations_sha256",
                "audit_sha256",
            }
            audit_core = {
                key: value for key, value in lineage_audit.items() if key != "audit_sha256"
            }
            audit_digests = (
                lineage_audit.get("subject_ids_sha256"),
                lineage_audit.get("footprints_sha256"),
                lineage_audit.get("derived_relations_sha256"),
                lineage_audit.get("audit_sha256"),
            )
            digests_are_valid = all(
                isinstance(digest, str)
                and len(digest) == 64
                and all(character in "0123456789abcdef" for character in digest)
                for digest in audit_digests
            )
            buffer_m = lineage_audit.get("spatial_buffer_m")
            counts_are_valid = all(
                isinstance(lineage_audit.get(key), int)
                and not isinstance(lineage_audit.get(key), bool)
                and lineage_audit[key] >= 0
                for key in (
                    "parent_count",
                    "spatially_linked_parent_count",
                    "temporally_linked_parent_count",
                )
            )
            if (
                set(lineage_audit) != expected_audit_keys
                or lineage_audit.get("schema")
                != "umcd.spatiotemporal_lineage_audit.v2"
                or lineage_audit.get("status") != "PASS_INPUT_SCOPED"
                or lineage_audit.get("derivation_algorithm")
                != "umcd.spatiotemporal-leakage.v2"
                or isinstance(buffer_m, bool)
                or not isinstance(buffer_m, Real)
                or not math.isfinite(float(buffer_m))
                or buffer_m < 0
                or not counts_are_valid
                or not digests_are_valid
                or lineage_audit.get("parent_count") != len(subjects)
                or lineage_audit.get("subject_ids_sha256")
                != hashlib.sha256(_canonical_json(sorted(subjects))).hexdigest()
                or hashlib.sha256(_canonical_json(audit_core)).hexdigest()
                != lineage_audit.get("audit_sha256")
            ):
                raise ValueError("registry spatiotemporal audit is invalid or roster-mismatched")
        object.__setattr__(self, "subject_to_component", MappingProxyType(subject_components))
        object.__setattr__(self, "component_to_split", MappingProxyType(component_splits))
        object.__setattr__(self, "subjects", MappingProxyType(subjects))
        object.__setattr__(self, "child_bindings", MappingProxyType(children))
        if lineage_audit is not None:
            object.__setattr__(self, "spatiotemporal_audit", MappingProxyType(lineage_audit))

    def split_for_parent(self, subject_id: str) -> str:
        try:
            return self.component_to_split[self.subject_to_component[subject_id]]
        except KeyError as exc:
            raise KeyError(f"parent subject is absent from frozen split: {subject_id}") from exc

    def split_for_child(self, child_id: str) -> str:
        """Resolve a registered child's split through its frozen parent binding."""
        try:
            binding = self.child_bindings[child_id]
        except KeyError as exc:
            raise KeyError(f"child is absent from frozen split registry: {child_id}") from exc
        return self.split_for_parent(binding.parent_subject_id)

    @property
    def independent_component_counts(self) -> Mapping[str, int]:
        return MappingProxyType(
            {
                split: sum(value == split for value in self.component_to_split.values())
                for split in self.policy.split_fractions
            }
        )

    def report(self) -> dict[str, Any]:
        """Report parent and independent-component counts overall and per source family."""
        children_by_split = {
            split: sum(self.split_for_child(child_id) == split for child_id in self.child_bindings)
            for split in self.policy.split_fractions
        }
        children_by_kind: dict[str, dict[str, int]] = {}
        for child_id, binding in self.child_bindings.items():
            counts = children_by_kind.setdefault(
                binding.child_kind, dict.fromkeys(self.policy.split_fractions, 0)
            )
            counts[self.split_for_child(child_id)] += 1
        result: dict[str, Any] = {
            "policy_sha256": self.policy.to_dict()["policy_sha256"],
            "splits": {},
            "child_bindings": {
                "total": len(self.child_bindings),
                "by_split": children_by_split,
                "by_kind": {
                    kind: dict(sorted(counts.items()))
                    for kind, counts in sorted(children_by_kind.items())
                },
            },
        }
        for split in self.policy.split_fractions:
            subject_ids = [
                subject_id
                for subject_id in self.subjects
                if self.split_for_parent(subject_id) == split
            ]
            by_family: dict[str, set[str]] = {}
            for subject_id in subject_ids:
                subject = self.subjects[subject_id]
                by_family.setdefault(subject.source_family, set()).add(
                    self.subject_to_component[subject_id]
                )
            result["splits"][split] = {
                "independent_components": self.independent_component_counts[split],
                "parent_subjects": len(subject_ids),
                "by_source_family": {
                    family: {
                        "independent_components": len(components),
                        "parent_subjects": sum(
                            self.subjects[item].source_family == family for item in subject_ids
                        ),
                    }
                    for family, components in sorted(by_family.items())
                },
            }
        return result

    def to_dict(self) -> dict[str, Any]:
        core = {
            "schema": (
                "umcd.split_registry.v2"
                if self.spatiotemporal_audit is None
                else "umcd.split_registry.v3"
            ),
            "schema_version": 2 if self.spatiotemporal_audit is None else 3,
            "policy": self.policy.to_dict(),
            "subjects": {
                subject_id: {
                    **self.subjects[subject_id].to_dict(),
                    "component_id": self.subject_to_component[subject_id],
                    "split": self.split_for_parent(subject_id),
                }
                for subject_id in sorted(self.subjects)
            },
            "components": {
                component: {"split": split}
                for component, split in sorted(self.component_to_split.items())
            },
            "children": {
                child_id: {
                    **self.child_bindings[child_id].to_dict(),
                    "split": self.split_for_child(child_id),
                }
                for child_id in sorted(self.child_bindings)
            },
            "report": self.report(),
        }
        if self.spatiotemporal_audit is not None:
            core["spatiotemporal_audit"] = dict(self.spatiotemporal_audit)
        return {**core, "registry_sha256": hashlib.sha256(_canonical_json(core)).hexdigest()}


def make_split_policy_registry(
    subjects: Sequence[SplitSubject], *, policy: SplitPolicy
) -> SplitPolicyRegistry:
    """Union parents by sparse qualified relations and assign whole components."""
    if not isinstance(policy, SplitPolicy):
        raise TypeError("policy must be a SplitPolicy")
    subject_by_id: dict[str, SplitSubject] = {}
    for subject in subjects:
        if not isinstance(subject, SplitSubject):
            raise TypeError("subjects must contain SplitSubject values")
        if subject.subject_id in subject_by_id:
            raise ValueError(f"duplicate split subject ID: {subject.subject_id}")
        subject_by_id[subject.subject_id] = subject
    subject_to_component, component_to_split = _derive_registry_assignments(subject_by_id, policy)
    return SplitPolicyRegistry(
        policy=policy,
        subject_to_component=subject_to_component,
        component_to_split=component_to_split,
        subjects=subject_by_id,
    )


def bind_split_children(
    registry: SplitPolicyRegistry, bindings: Sequence[SplitChildBinding]
) -> SplitPolicyRegistry:
    """Return a registry with immutable child-to-parent bindings added.

    Repeating the exact same binding is idempotent. Reusing a child ID for a
    different parent or child kind is rejected.
    """
    children = dict(registry.child_bindings)
    observed: dict[str, SplitChildBinding] = {}
    for binding in bindings:
        if not isinstance(binding, SplitChildBinding):
            raise TypeError("bindings must contain SplitChildBinding values")
        previous_input = observed.setdefault(binding.child_id, binding)
        if previous_input != binding:
            raise ValueError(f"child has conflicting parent or kind: {binding.child_id}")
        previous = children.get(binding.child_id)
        if previous is not None and previous != binding:
            raise ValueError(f"frozen child binding cannot change: {binding.child_id}")
        children[binding.child_id] = binding
    return SplitPolicyRegistry(
        policy=registry.policy,
        subject_to_component=registry.subject_to_component,
        component_to_split=registry.component_to_split,
        subjects=registry.subjects,
        child_bindings=children,
        spatiotemporal_audit=registry.spatiotemporal_audit,
    )


def audit_split_policy_registry(
    registry: SplitPolicyRegistry,
    subjects: Sequence[SplitSubject],
    child_bindings: Sequence[SplitChildBinding] = (),
) -> None:
    """Require exact parent and child coverage and unchanged frozen lineage."""
    observed: dict[str, SplitSubject] = {}
    relation_components: dict[tuple[LeakageNamespace, str], str] = {}
    for subject in subjects:
        if subject.subject_id in observed:
            raise ValueError(f"duplicate split subject in audit: {subject.subject_id}")
        observed[subject.subject_id] = subject
        frozen = registry.subjects.get(subject.subject_id)
        if frozen is None:
            raise ValueError(f"subject is absent from frozen split registry: {subject.subject_id}")
        if subject != frozen:
            raise ValueError(f"frozen split lineage changed for subject {subject.subject_id}")
        component = registry.subject_to_component[subject.subject_id]
        for relation in subject.relations:
            key = (relation.namespace, relation.identity)
            previous = relation_components.setdefault(key, component)
            if previous != component:
                raise ValueError(
                    f"shared leakage relation crosses components: {relation.qualified_identity}"
                )
    if set(observed) != set(registry.subjects):
        raise ValueError("audit subjects do not cover exactly the frozen parent set")
    observed_children: dict[str, SplitChildBinding] = {}
    for binding in child_bindings:
        if binding.child_id in observed_children:
            raise ValueError(f"duplicate child binding in audit: {binding.child_id}")
        frozen_child = registry.child_bindings.get(binding.child_id)
        if frozen_child is None:
            raise ValueError(f"child is absent from frozen split registry: {binding.child_id}")
        if binding != frozen_child:
            raise ValueError(f"frozen parent binding changed for child {binding.child_id}")
        observed_children[binding.child_id] = binding
    if set(observed_children) != set(registry.child_bindings):
        raise ValueError("audit bindings do not cover exactly the frozen child set")


def audit_split_policy_parent_roster(
    registry: SplitPolicyRegistry, subjects: Sequence[SplitSubject]
) -> dict[str, Any]:
    """Verify exact frozen parent coverage without asserting child-data coverage."""
    if not isinstance(registry, SplitPolicyRegistry):
        raise TypeError("registry must be a SplitPolicyRegistry")
    observed: dict[str, SplitSubject] = {}
    for subject in subjects:
        if not isinstance(subject, SplitSubject):
            raise TypeError("subjects must contain SplitSubject values")
        if subject.subject_id in observed:
            raise ValueError(f"duplicate split subject in audit: {subject.subject_id}")
        frozen = registry.subjects.get(subject.subject_id)
        if frozen is None:
            raise ValueError(f"subject is absent from frozen split registry: {subject.subject_id}")
        if subject != frozen:
            raise ValueError(f"frozen split lineage changed for subject {subject.subject_id}")
        observed[subject.subject_id] = subject
    if set(observed) != set(registry.subjects):
        raise ValueError("audit subjects do not cover exactly the frozen parent set")
    report: dict[str, Any] = {
        "schema": "umcd.split_parent_roster_audit.v1",
        "status": "PASS_INPUT_SCOPED",
        "registry_sha256": registry.to_dict()["registry_sha256"],
        "parent_count": len(observed),
        "parent_ids_sha256": hashlib.sha256(_canonical_json(sorted(observed))).hexdigest(),
    }
    report["audit_sha256"] = hashlib.sha256(_canonical_json(report)).hexdigest()
    return report


def audit_split_policy_child_roster(
    registry: SplitPolicyRegistry,
    child_bindings: Sequence[SplitChildBinding],
    *,
    child_kind: str,
) -> dict[str, Any]:
    """Verify exact frozen child coverage for one artifact kind.

    This permits auditing a saved derived view against only the corresponding
    child class while the registry also contains other descendant kinds. The
    supplied bindings must exactly match every frozen child of ``child_kind``.
    """

    if not isinstance(registry, SplitPolicyRegistry):
        raise TypeError("registry must be a SplitPolicyRegistry")
    child_kind = _require_canonical_identity(child_kind, "child_kind")
    expected = {
        child_id: binding
        for child_id, binding in registry.child_bindings.items()
        if binding.child_kind == child_kind
    }
    if not expected:
        raise ValueError(f"registry has no frozen children of kind {child_kind!r}")

    observed: dict[str, SplitChildBinding] = {}
    for binding in child_bindings:
        if not isinstance(binding, SplitChildBinding):
            raise TypeError("child_bindings must contain SplitChildBinding values")
        if binding.child_kind != child_kind:
            raise ValueError(
                f"child roster contains kind {binding.child_kind!r}; expected {child_kind!r}"
            )
        if binding.child_id in observed:
            raise ValueError(f"duplicate child binding in audit: {binding.child_id}")
        frozen = expected.get(binding.child_id)
        if frozen is None:
            raise ValueError(f"child is absent from frozen {child_kind!r} roster: {binding.child_id}")
        if binding != frozen:
            raise ValueError(f"frozen parent binding changed for child {binding.child_id}")
        observed[binding.child_id] = binding
    if set(observed) != set(expected):
        raise ValueError(f"audit bindings do not cover exactly the frozen {child_kind!r} roster")

    report: dict[str, Any] = {
        "schema": "umcd.split_child_roster_audit.v1",
        "status": "PASS_INPUT_SCOPED",
        "registry_sha256": registry.to_dict()["registry_sha256"],
        "child_kind": child_kind,
        "child_count": len(observed),
        "child_ids_sha256": hashlib.sha256(_canonical_json(sorted(observed))).hexdigest(),
        "bindings_sha256": hashlib.sha256(
            _canonical_json(
                [observed[child_id].to_dict() for child_id in sorted(observed)]
            )
        ).hexdigest(),
    }
    report["audit_sha256"] = hashlib.sha256(_canonical_json(report)).hexdigest()
    return report


def audit_whole_scene_holdout(
    registry: SplitPolicyRegistry,
    scene_subjects: Mapping[str, Sequence[str]],
    *,
    expected_scene_ids: Sequence[str],
    expected_subject_ids: Sequence[str],
) -> dict[str, Any]:
    """Verify complete input-scoped scene membership stays within one split.

    The expected scene and parent rosters must come from a separately reviewed
    corpus inventory. This function verifies exact coverage and the frozen
    registry's assignments; it cannot establish that the supplied inventory
    represents the complete or legally/physically authoritative corpus.
    A parent may belong to multiple scenes, in which case the connected
    relation graph must place those scenes in the same split as well.
    """
    if not isinstance(registry, SplitPolicyRegistry):
        raise TypeError("registry must be a SplitPolicyRegistry")
    if not isinstance(scene_subjects, Mapping):
        raise TypeError("scene_subjects must be a mapping of scene IDs to parent IDs")

    def validate_roster(values: Sequence[str], label: str) -> tuple[str, ...]:
        if isinstance(values, (str, bytes)):
            raise TypeError(f"{label} must be a sequence of IDs")
        roster = tuple(values)
        for item in roster:
            _require_canonical_identity(item, f"{label} entry")
        if len(set(roster)) != len(roster):
            raise ValueError(f"{label} must not contain duplicate IDs")
        return roster

    expected_scenes = validate_roster(expected_scene_ids, "expected_scene_ids")
    expected_subjects = validate_roster(expected_subject_ids, "expected_subject_ids")
    if not expected_scenes or not expected_subjects:
        raise ValueError("expected scene and parent rosters must be non-empty")
    normalized_mapping: dict[str, tuple[str, ...]] = {}
    for scene_id, parent_ids in scene_subjects.items():
        normalized_scene_id = _require_canonical_identity(scene_id, "scene_subjects key")
        if normalized_scene_id in normalized_mapping:
            raise ValueError(f"duplicate scene ID: {normalized_scene_id}")
        normalized_mapping[normalized_scene_id] = validate_roster(
            parent_ids, f"scene_subjects[{normalized_scene_id!r}]"
        )
        if not normalized_mapping[normalized_scene_id]:
            raise ValueError(f"scene has no parent subjects: {normalized_scene_id}")
    if set(normalized_mapping) != set(expected_scenes):
        raise ValueError("scene membership does not match expected scene inventory")
    observed_subjects: set[str] = set()
    scene_component_ids: dict[str, str] = {}
    scene_splits: dict[str, str] = {}
    membership_count = 0
    for scene_id in sorted(expected_scenes):
        parent_ids = normalized_mapping[scene_id]
        membership_count += len(parent_ids)
        observed_subjects.update(parent_ids)
        component_ids: set[str] = set()
        splits: set[str] = set()
        required_relation = LeakageRelation(LeakageNamespace.SCENE_REALIZATION, scene_id)
        for parent_id in parent_ids:
            subject = registry.subjects.get(parent_id)
            if subject is None:
                raise ValueError(f"scene parent is absent from frozen registry: {parent_id}")
            if required_relation not in subject.relations:
                raise ValueError(f"scene parent lacks its scene-realization relation: {parent_id}")
            component_ids.add(registry.subject_to_component[parent_id])
            splits.add(registry.split_for_parent(parent_id))
        if len(component_ids) != 1 or len(splits) != 1:
            raise ValueError(f"whole-scene holdout is violated for scene: {scene_id}")
        scene_component_ids[scene_id] = next(iter(component_ids))
        scene_splits[scene_id] = next(iter(splits))

    if observed_subjects != set(expected_subjects):
        raise ValueError("scene memberships do not cover exactly the expected parent roster")
    registered_scene_subjects = {
        subject_id
        for subject_id, subject in registry.subjects.items()
        if any(
            relation.namespace is LeakageNamespace.SCENE_REALIZATION
            for relation in subject.relations
        )
    }
    omitted_scene_subjects = registered_scene_subjects - observed_subjects
    if omitted_scene_subjects:
        raise ValueError(
            "scene memberships omit frozen scene-lineage parents: "
            + ", ".join(sorted(omitted_scene_subjects))
        )
    registered_scene_ids = {
        relation.identity
        for subject in registry.subjects.values()
        for relation in subject.relations
        if relation.namespace is LeakageNamespace.SCENE_REALIZATION
    }
    if registered_scene_ids != set(expected_scenes):
        raise ValueError(
            "expected scene inventory does not cover exactly the frozen "
            "scene-realization identities"
        )
    report = {
        "status": "PASS_INPUT_SCOPED",
        "policy_sha256": registry.policy.to_dict()["policy_sha256"],
        "registry_sha256": registry.to_dict()["registry_sha256"],
        "expected_scene_ids_sha256": hashlib.sha256(
            _canonical_json(sorted(expected_scenes))
        ).hexdigest(),
        "expected_subject_ids_sha256": hashlib.sha256(
            _canonical_json(sorted(expected_subjects))
        ).hexdigest(),
        "scene_membership_sha256": hashlib.sha256(
            _canonical_json(
                {
                    scene_id: sorted(parent_ids)
                    for scene_id, parent_ids in sorted(normalized_mapping.items())
                }
            )
        ).hexdigest(),
        "scene_count": len(expected_scenes),
        "parent_subject_count": len(expected_subjects),
        "scene_membership_count": membership_count,
        "scene_component_ids": dict(sorted(scene_component_ids.items())),
        "scene_splits": dict(sorted(scene_splits.items())),
    }
    report["audit_sha256"] = hashlib.sha256(_canonical_json(report)).hexdigest()
    return report


def freeze_split_policy_registry(path: str | Path, registry: SplitPolicyRegistry) -> str:
    """Atomically create a canonical immutable registry lock."""
    if not isinstance(registry, SplitPolicyRegistry):
        raise TypeError("registry must be a SplitPolicyRegistry")
    has_spatiotemporal_relations = any(
        relation.namespace
        in {
            LeakageNamespace.SPATIAL_BUFFER,
            LeakageNamespace.OVERLAPPING_TIME_WINDOW,
        }
        for subject in registry.subjects.values()
        for relation in subject.relations
    )
    if has_spatiotemporal_relations and registry.spatiotemporal_audit is None:
        raise ValueError(
            "cannot freeze spatial/time leakage relations without a bound footprint audit"
        )
    destination = Path(path)
    encoded = (json.dumps(registry.to_dict(), indent=2, sort_keys=True) + "\n").encode()
    if len(encoded) > MAX_SPLIT_POLICY_LOCK_BYTES:
        raise ValueError(
            "split-policy lock exceeds its "
            f"{MAX_SPLIT_POLICY_LOCK_BYTES}-byte reader limit"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _read_immutable_lock_bytes(destination) != encoded:
            raise FileExistsError(f"refusing to modify existing split-policy lock: {destination}")
        return hashlib.sha256(encoded).hexdigest()
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
            temp_path = stream.name
        try:
            os.link(temp_path, destination)
            _fsync_directory(destination.parent)
        except FileExistsError:
            if _read_immutable_lock_bytes(destination) != encoded:
                raise FileExistsError(
                    f"refusing to modify existing split-policy lock: {destination}"
                )
    finally:
        if temp_path is not None:
            try:
                Path(temp_path).unlink(missing_ok=True)
            finally:
                _fsync_directory(destination.parent)
    return hashlib.sha256(encoded).hexdigest()


def read_split_policy_registry(path: str | Path) -> SplitPolicyRegistry:
    """Read a split lock and independently reconstruct its frozen assignment."""
    try:
        payload = json.loads(
            _read_immutable_lock_bytes(Path(path)).decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("UMCD split registry is not valid bounded UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("UMCD split registry root must be a JSON object")
    schema = payload.get("schema")
    schema_version = payload.get("schema_version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise ValueError("UMCD split registry schema_version must be an integer")
    if (schema, schema_version) == ("umcd.split_registry.v2", 2):
        expected_fields = _SPLIT_REGISTRY_V2_FIELDS
    elif (schema, schema_version) == ("umcd.split_registry.v3", 3):
        expected_fields = _SPLIT_REGISTRY_V3_FIELDS
    else:
        raise ValueError("unsupported UMCD split registry schema")
    if set(payload) != expected_fields:
        missing = sorted(expected_fields - set(payload))
        extra = sorted(set(payload) - expected_fields)
        raise ValueError(
            f"UMCD split registry field roster is invalid; missing={missing}, extra={extra}"
        )
    for field_name in ("policy", "subjects", "components", "children", "report"):
        if not isinstance(payload[field_name], dict):
            raise ValueError(f"UMCD split registry {field_name} must be a JSON object")
    if schema_version == 3 and not isinstance(payload["spatiotemporal_audit"], dict):
        raise ValueError("UMCD split registry spatiotemporal_audit must be a JSON object")
    core = {key: value for key, value in payload.items() if key != "registry_sha256"}
    if hashlib.sha256(_canonical_json(core)).hexdigest() != payload.get("registry_sha256"):
        raise ValueError("UMCD split registry hash is invalid")
    policy = SplitPolicy.from_dict(payload["policy"])
    subjects: dict[str, SplitSubject] = {}
    subject_to_component: dict[str, str] = {}
    for subject_id, item in payload["subjects"].items():
        subjects[subject_id] = SplitSubject(
            subject_id=subject_id,
            parent_type=SplitParentType(item["parent_type"]),
            source_family=item["source_family"],
            relations=tuple(
                LeakageRelation(LeakageNamespace(rel["namespace"]), rel["identity"])
                for rel in item["relations"]
            ),
            partition_id=item.get("partition_id", subject_id),
        )
        subject_to_component[subject_id] = item["component_id"]
    component_to_split = {
        component: item["split"] for component, item in payload["components"].items()
    }
    children = {
        child_id: SplitChildBinding(
            child_id=child_id,
            parent_subject_id=item["parent_subject_id"],
            child_kind=item["child_kind"],
        )
        for child_id, item in payload.get("children", {}).items()
    }
    registry = SplitPolicyRegistry(
        policy,
        subject_to_component,
        component_to_split,
        subjects,
        children,
        payload.get("spatiotemporal_audit"),
    )
    if registry.to_dict() != payload:
        raise ValueError("UMCD split registry representation is not canonical")
    return registry


def verify_signed_split_policy_registry_attestation(
    path: str | Path,
    signature_path: str | Path,
    *,
    trusted_allowed_signers_path: str | Path,
    signer_id: str,
) -> tuple[SplitPolicyRegistry, VerifiedOpenSSHArtifact]:
    """Verify a split lock and return its exact artifact/trust attestation.

    The lock, detached signature, and externally governed OpenSSH
    ``allowed_signers`` trust file are snapshotted as single-link regular files
    before verification. The trusted keyring must be supplied independently
    of the release bundle. Signatures use the ``umcd-split-policy-v2`` SSH
    namespace, separating them from signatures for other purposes.
    """
    signer_id = _require_canonical_identity(signer_id, "signer_id")
    if any(character in signer_id for character in ("\n", "\r", "\x00")):
        raise ValueError("signer_id must be a single-line principal")

    verified = verify_openssh_signed_artifact(
        path,
        signature_path,
        trusted_allowed_signers_path=trusted_allowed_signers_path,
        signer_id=signer_id,
        namespace=SPLIT_POLICY_SIGNATURE_NAMESPACE,
    )
    with tempfile.TemporaryDirectory(prefix="umcd-split-policy-verify-") as temporary_directory:
        lock_snapshot = Path(temporary_directory) / "split-registry.json"
        lock_snapshot.write_bytes(verified.payload)
        return read_split_policy_registry(lock_snapshot), verified


def verify_signed_split_policy_registry(
    path: str | Path,
    signature_path: str | Path,
    *,
    trusted_allowed_signers_path: str | Path,
    signer_id: str,
) -> SplitPolicyRegistry:
    """Verify the detached signature and return the frozen split registry."""
    registry, _ = verify_signed_split_policy_registry_attestation(
        path,
        signature_path,
        trusted_allowed_signers_path=trusted_allowed_signers_path,
        signer_id=signer_id,
    )
    return registry
