"""Content-addressed, pickle-free storage for NR observation records."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import sqlite3
import stat
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterator, Mapping, cast

import numpy as np
from receivers.nr import NRChannelEstimate, NRPUSCHConfig

from .nr import NRChannelRecord, NRObservation, NRTrajectory, TargetDefinition


@dataclass(frozen=True, slots=True)
class NRStoredRecord:
    """Immutable manifest pointer for one content-addressed NR record."""

    sample_id: str
    content_sha256: str
    relative_path: str
    schema_version: int = 1


@dataclass(frozen=True, slots=True)
class NRStoredTrajectory:
    """Immutable manifest pointer for a time-ordered NR trajectory."""

    trajectory_id: str
    content_sha256: str
    relative_path: str
    sample_ids: tuple[str, ...]
    schema_version: int = 1

    @property
    def manifest_relative_path(self) -> str:
        """Return the stable path of the trajectory-ID manifest pointer."""
        key = hashlib.sha256(self.trajectory_id.encode()).hexdigest()
        return f"manifests/trajectories/{key}.json"


def _as_numpy(value: Any, *, field: str) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError(f"{field} may not contain Python objects")
    if array.dtype.kind not in "biufc":
        raise TypeError(f"{field} must use a numeric, boolean, or complex dtype")
    return np.ascontiguousarray(array)


def _json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise TypeError("record metadata must be finite JSON-compatible values") from exc


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _open_store_record(base: Path, relative_path: str) -> BinaryIO:
    """Open a regular record file without following store-internal symlinks."""
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("manifest record path must remain inside the store")
    resolved_base = base.resolve()
    candidate = base.joinpath(*relative.parts)
    resolved_candidate = candidate.resolve(strict=True)
    if not resolved_candidate.is_relative_to(resolved_base):
        raise ValueError("manifest record path resolves outside the store")

    if os.name == "posix" and hasattr(os, "O_DIRECTORY") and hasattr(os, "O_NOFOLLOW"):
        directory_fd = os.open(resolved_base, os.O_RDONLY | os.O_DIRECTORY)
        try:
            for part in relative.parts[:-1]:
                next_fd = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=directory_fd,
                )
                os.close(directory_fd)
                directory_fd = next_fd
            descriptor = os.open(
                relative.parts[-1],
                os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0),
                dir_fd=directory_fd,
            )
        finally:
            os.close(directory_fd)
    else:
        descriptor = os.open(
            resolved_candidate,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
        )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("NR record object must be a regular file")
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise


def _nr_record_memory_estimate(
    base: Path,
    stored: NRStoredRecord,
    *,
    max_shard_bytes: int,
) -> int:
    """Estimate two compressed buffers plus two uncompressed-member equivalents."""
    try:
        with _open_store_record(base, stored.relative_path) as record_stream:
            compressed_size = os.fstat(record_stream.fileno()).st_size
            if compressed_size > max_shard_bytes:
                raise ValueError(
                    f"NR record {stored.sample_id!r} compressed size exceeds max_shard_bytes"
                )
            record_stream.seek(0)
            with zipfile.ZipFile(record_stream) as archive:
                members = archive.infolist()
                names = [member.filename for member in members]
                if not members or len(names) != len(set(names)):
                    raise ValueError("NR record archive must have unique nonempty members")
                uncompressed_size = sum(member.file_size for member in members)
                estimated_bytes = 2 * compressed_size + 2 * uncompressed_size
                if estimated_bytes > max_shard_bytes:
                    raise ValueError(
                        f"NR record {stored.sample_id!r} estimate ({estimated_bytes} bytes) "
                        f"exceeds max_shard_bytes ({max_shard_bytes})"
                    )
                for member in members:
                    if not member.filename.endswith(".npy"):
                        raise ValueError("NR record archive members must be NPY arrays")
                    with archive.open(member) as stream:
                        version = np.lib.format.read_magic(stream)
                        if version == (1, 0):
                            shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                        elif version == (2, 0):
                            shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                        else:
                            raise ValueError(f"unsupported NPY format version: {version}")
                        if dtype.hasobject or dtype.kind not in "biufc":
                            raise ValueError(
                                "NR record arrays must have numeric, non-object dtypes"
                            )
                        declared_size = math.prod(shape) * dtype.itemsize
                        available_size = member.file_size - stream.tell()
                        if declared_size != available_size:
                            raise ValueError("NR record NPY header size does not match its member")
    except zipfile.BadZipFile as exc:
        raise ValueError("NR record is not a readable NPZ archive") from exc
    # This conservative accounting charges two whole uncompressed-member
    # equivalents. It does not assert that domain objects retain a second copy
    # of each decoded array. Python/allocator overhead is excluded.
    return estimated_bytes


def _iter_manifest_paths(manifest_dir: Path) -> Iterator[Path]:
    """Yield JSON manifest paths without materializing the directory listing."""
    with os.scandir(manifest_dir) as entries:
        for entry in entries:
            if entry.name.endswith(".json"):
                yield Path(entry.path)


def iter_nr_record_shards(
    root: str | Path,
    *,
    max_shard_bytes: int,
) -> Iterator[tuple[NRChannelRecord, ...]]:
    """Load deterministic sample-ID-ordered batches within an estimated byte budget.

    Manifest metadata is staged in a temporary disk-backed SQLite index with
    a bounded page cache. NPZ payloads are opened one shard at a time. The byte
    estimate charges two compressed-buffer equivalents and two
    uncompressed-member equivalents, including NPY headers. This is
    conservative accounting rather than a measured allocation model; it
    excludes Python and allocator overhead and is not an RSS guarantee.
    An individual record above the budget fails before its arrays are loaded.
    """
    if isinstance(max_shard_bytes, bool) or not isinstance(max_shard_bytes, int):
        raise TypeError("max_shard_bytes must be an integer")
    if max_shard_bytes <= 0:
        raise ValueError("max_shard_bytes must be positive")

    base = Path(root).resolve()
    manifest_dir = (base / "manifests").resolve()
    if not manifest_dir.is_relative_to(base):
        raise ValueError("NR manifest directory resolves outside the store")
    if not manifest_dir.exists():
        return

    with tempfile.TemporaryDirectory(prefix="nr-manifest-inventory-") as temporary_dir:
        database_path = Path(temporary_dir) / "inventory.sqlite3"
        connection = sqlite3.connect(database_path)
        try:
            # SQLite interprets a negative cache size as a kibibyte limit.
            connection.execute("PRAGMA cache_size = -256")
            connection.execute("PRAGMA temp_store = FILE")
            connection.execute("PRAGMA journal_mode = OFF")
            connection.execute(
                "CREATE TABLE inventory ("
                "sample_id TEXT PRIMARY KEY COLLATE BINARY, "
                "content_sha256 TEXT NOT NULL, "
                "relative_path TEXT NOT NULL, "
                "schema_version INTEGER NOT NULL, "
                "estimated_bytes INTEGER NOT NULL"
                ") WITHOUT ROWID"
            )
            for manifest_path in _iter_manifest_paths(manifest_dir):
                resolved_manifest = manifest_path.resolve(strict=True)
                if not resolved_manifest.is_relative_to(base):
                    raise ValueError("NR manifest path resolves outside the store")
                payload = json.loads(
                    resolved_manifest.read_text(encoding="utf-8"),
                    object_pairs_hook=_reject_duplicate_json_keys,
                )
                if not isinstance(payload, dict):
                    raise ValueError("NR record manifest must be a JSON object")
                stored = NRStoredRecord(**payload)
                if not isinstance(stored.sample_id, str) or not stored.sample_id.strip():
                    raise ValueError("NR record manifest sample_id must be a nonempty string")
                expected_manifest_name = (
                    f"{hashlib.sha256(stored.sample_id.encode()).hexdigest()}.json"
                )
                if resolved_manifest.name != expected_manifest_name:
                    raise ValueError("NR record manifest filename does not match its sample ID")
                if stored.schema_version != 1:
                    raise ValueError(
                        f"unsupported NR record schema version: {stored.schema_version}"
                    )
                try:
                    connection.execute(
                        "INSERT INTO inventory VALUES (?, ?, ?, ?, 0)",
                        (
                            stored.sample_id,
                            stored.content_sha256,
                            stored.relative_path,
                            stored.schema_version,
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    if "UNIQUE constraint failed" not in str(exc):
                        raise
                    raise ValueError(
                        f"duplicate NR record manifest for sample_id {stored.sample_id!r}"
                    ) from exc
                estimated_bytes = _nr_record_memory_estimate(
                    base,
                    stored,
                    max_shard_bytes=max_shard_bytes,
                )
                connection.execute(
                    "UPDATE inventory SET estimated_bytes = ? WHERE sample_id = ?",
                    (estimated_bytes, stored.sample_id),
                )

            connection.commit()
            pending: list[NRStoredRecord] = []
            pending_bytes = 0
            cursor = connection.execute(
                "SELECT sample_id, content_sha256, relative_path, schema_version, "
                "estimated_bytes FROM inventory ORDER BY sample_id COLLATE BINARY"
            )
            for sample_id, content_sha256, relative_path, schema_version, estimated_bytes in cursor:
                if pending and pending_bytes + estimated_bytes > max_shard_bytes:
                    yield tuple(read_nr_record(base, item) for item in pending)
                    pending = []
                    pending_bytes = 0
                pending.append(
                    NRStoredRecord(
                        sample_id=sample_id,
                        content_sha256=content_sha256,
                        relative_path=relative_path,
                        schema_version=schema_version,
                    )
                )
                pending_bytes += estimated_bytes
            if pending:
                yield tuple(read_nr_record(base, item) for item in pending)
        finally:
            connection.close()


def _publish_immutable(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            temp_path = stream.name
        try:
            os.link(temp_path, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise
    finally:
        if temp_path is not None:
            Path(temp_path).unlink(missing_ok=True)


def write_nr_record(root: str | Path, record: NRChannelRecord) -> NRStoredRecord:
    """Write a record as a SHA256-addressed NPZ plus immutable JSON manifest.

    NPZ loading always uses ``allow_pickle=False``. A repeated write of the
    same sample and content is idempotent; changing an existing sample ID is
    rejected so published manifests cannot silently drift.
    """
    base = Path(root)
    arrays: dict[str, np.ndarray] = {
        "received_grid": _as_numpy(record.observation.received_grid, field="received_grid"),
        "received_grid_mask": _as_numpy(
            record.observation.received_grid_mask, field="received_grid_mask"
        ),
        "h_hat": _as_numpy(record.observation.channel_estimate.h_hat, field="h_hat"),
        "error_variance": _as_numpy(
            record.observation.channel_estimate.error_variance, field="error_variance"
        ),
        "pilot_mask": _as_numpy(record.observation.channel_estimate.pilot_mask, field="pilot_mask"),
    }
    target_keys: dict[str, str] = {}
    mask_keys: dict[str, str] = {}
    for index, name in enumerate(sorted(record.targets)):
        key = f"target_{index:04d}"
        target_keys[name] = key
        arrays[key] = _as_numpy(record.targets[name], field=f"target {name}")
    for index, name in enumerate(sorted(record.target_masks)):
        key = f"target_mask_{index:04d}"
        mask_keys[name] = key
        arrays[key] = _as_numpy(record.target_masks[name], field=f"target mask {name}")

    metadata = {
        "schema_version": 1,
        "sample_id": record.observation.sample_id,
        "receiver_config": asdict(record.observation.receiver_config),
        "fft_bin_indices": list(record.observation.fft_bin_indices),
        "estimate_metadata": dict(record.observation.channel_estimate.metadata),
        "targets": target_keys,
        "target_masks": mask_keys,
        "provenance": dict(record.provenance),
        "target_definitions": {
            name: asdict(definition) for name, definition in record.target_definitions.items()
        },
    }
    arrays["metadata_json"] = np.frombuffer(_json_bytes(metadata), dtype=np.uint8)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **cast(Any, arrays))
    payload = buffer.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    relative_path = f"records/{digest[:2]}/{digest}.npz"
    _publish_immutable(base / relative_path, payload)

    manifest = NRStoredRecord(record.observation.sample_id, digest, relative_path)
    sample_key = hashlib.sha256(manifest.sample_id.encode()).hexdigest()
    manifest_path = base / "manifests" / f"{sample_key}.json"
    manifest_data = _json_bytes(asdict(manifest)) + b"\n"
    _publish_immutable(manifest_path, manifest_data)
    return manifest


def read_nr_record(root: str | Path, manifest: NRStoredRecord | str | Path) -> NRChannelRecord:
    """Read and verify one stored record without enabling pickle deserialization."""
    base = Path(root)
    if isinstance(manifest, NRStoredRecord):
        stored = manifest
    else:
        manifest_path = Path(manifest)
        if not manifest_path.is_absolute():
            manifest_path = base / manifest_path
        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        stored = NRStoredRecord(**raw_manifest)
    if stored.schema_version != 1:
        raise ValueError(f"unsupported NR record schema version: {stored.schema_version}")
    if Path(stored.relative_path).is_absolute() or ".." in Path(stored.relative_path).parts:
        raise ValueError("manifest record path must remain inside the store")

    with _open_store_record(base, stored.relative_path) as stream:
        payload = stream.read()
    actual_hash = hashlib.sha256(payload).hexdigest()
    if actual_hash != stored.content_sha256:
        raise ValueError("NR record content hash does not match its manifest")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata_json"].tobytes().decode("utf-8"))
        if metadata["sample_id"] != stored.sample_id:
            raise ValueError("NR record sample ID does not match its manifest")
        estimate = NRChannelEstimate(
            h_hat=archive["h_hat"],
            error_variance=archive["error_variance"],
            pilot_mask=archive["pilot_mask"],
            metadata=metadata["estimate_metadata"],
        )
        observation = NRObservation(
            sample_id=metadata["sample_id"],
            received_grid=archive["received_grid"],
            channel_estimate=estimate,
            fft_bin_indices=tuple(metadata["fft_bin_indices"]),
            receiver_config=NRPUSCHConfig(**metadata["receiver_config"]),
            received_grid_mask=(
                archive["received_grid_mask"] if "received_grid_mask" in archive.files else None
            ),
        )
        targets: Mapping[str, np.ndarray] = {
            name: archive[key] for name, key in metadata["targets"].items()
        }
        masks: Mapping[str, np.ndarray] = {
            name: archive[key] for name, key in metadata["target_masks"].items()
        }
        definitions = {
            name: TargetDefinition(**value)
            for name, value in metadata.get("target_definitions", {}).items()
        }
        return NRChannelRecord(
            observation=observation,
            targets=targets,
            target_masks=masks,
            provenance=metadata["provenance"],
            target_definitions=definitions,
        )


def write_nr_trajectory(root: str | Path, trajectory: NRTrajectory) -> NRStoredTrajectory:
    """Store trajectory records individually and publish an immutable ordered index."""
    base = Path(root)
    record_manifests = [write_nr_record(base, record) for record in trajectory.records]
    trajectory_payload = {
        "schema_version": 1,
        "trajectory_id": trajectory.trajectory_id,
        "sample_times_s": list(trajectory.sample_times_s),
        "records": [asdict(manifest) for manifest in record_manifests],
    }
    payload = _json_bytes(trajectory_payload)
    digest = hashlib.sha256(payload).hexdigest()
    relative_path = f"trajectories/{digest[:2]}/{digest}.json"
    _publish_immutable(base / relative_path, payload)
    stored = NRStoredTrajectory(
        trajectory_id=trajectory.trajectory_id,
        content_sha256=digest,
        relative_path=relative_path,
        sample_ids=trajectory.sample_ids,
    )
    trajectory_key = hashlib.sha256(stored.trajectory_id.encode()).hexdigest()
    manifest_data = _json_bytes(asdict(stored)) + b"\n"
    _publish_immutable(
        base / "manifests" / "trajectories" / f"{trajectory_key}.json", manifest_data
    )
    return stored


def read_nr_trajectory(
    root: str | Path,
    manifest: NRStoredTrajectory | str | Path,
) -> NRTrajectory:
    """Load and validate a stored trajectory and every referenced NR record."""
    base = Path(root)
    if isinstance(manifest, NRStoredTrajectory):
        stored = manifest
    else:
        manifest_path = Path(manifest)
        if not manifest_path.is_absolute():
            manifest_path = base / manifest_path
        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw_manifest["sample_ids"] = tuple(raw_manifest["sample_ids"])
        stored = NRStoredTrajectory(**raw_manifest)
    if stored.schema_version != 1:
        raise ValueError(f"unsupported NR trajectory schema version: {stored.schema_version}")
    relative_path = Path(stored.relative_path)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("trajectory path must remain inside the store")
    payload = (base / relative_path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != stored.content_sha256:
        raise ValueError("NR trajectory content hash does not match its manifest")
    values = json.loads(payload.decode("utf-8"))
    if values.get("schema_version") != 1:
        raise ValueError("unsupported NR trajectory payload schema")
    if values.get("trajectory_id") != stored.trajectory_id:
        raise ValueError("NR trajectory ID does not match its manifest")
    record_manifests = [NRStoredRecord(**value) for value in values.get("records", [])]
    if tuple(record.sample_id for record in record_manifests) != stored.sample_ids:
        raise ValueError("NR trajectory sample IDs do not match its manifest")
    records = tuple(read_nr_record(base, record) for record in record_manifests)
    return NRTrajectory(
        trajectory_id=stored.trajectory_id,
        sample_times_s=tuple(values["sample_times_s"]),
        records=records,
    )
