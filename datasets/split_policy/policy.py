"""Signed split-policy artifacts, leakage relations, and split derivation."""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import stat
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from datasets.independent_splits import make_independent_split
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
    if isinstance(spatial_buffer_m, bool) or not isinstance(spatial_buffer_m, Real):
        raise TypeError("spatial_buffer_m must be a real number")
    try:
        buffer_m = float(spatial_buffer_m)
    except OverflowError as error:
        raise ValueError("spatial_buffer_m must be finite and non-negative") from error
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
            add_component_relations(LeakageNamespace.SPATIAL_BUFFER, scope, component, evidence)
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
    derived = derive_spatiotemporal_leakage_relations(footprints, spatial_buffer_m=spatial_buffer_m)
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
        "subject_ids_sha256": hashlib.sha256(_canonical_json(sorted(subject_by_id))).hexdigest(),
        "spatially_linked_parent_count": sum(
            any(relation.namespace is LeakageNamespace.SPATIAL_BUFFER for relation in derived[item])
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
            *(
                LeakageRelation(LeakageNamespace(namespace), identity)
                for namespace, identity in episode.split_leakage_relations
            ),
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
