"""Content-addressed, pickle-free storage for NR observation records."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, cast

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

    payload = (base / stored.relative_path).read_bytes()
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
