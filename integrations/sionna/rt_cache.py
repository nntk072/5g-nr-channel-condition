"""Immutable, integrity-checked cache for complete static Sionna RT datasets."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import stat
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Mapping

from artifacts.provenance import canonical_hash
from datasets.channel import ChannelDataset

_CACHE_SCHEMA = "loslab.rt_dataset_cache.v1"
_KEY_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_MAX_BYTES = 64 * 1024
DEFAULT_MAX_PAYLOAD_BYTES = 2 * 1024 * 1024 * 1024


class RTCacheConflictError(RuntimeError):
    """Raised when one complete run identity produces different immutable bytes."""


def _cache_key(identity: Mapping[str, Any]) -> str:
    if identity.get("schema") != "loslab.rt_run_identity.v1" or not identity.get("complete"):
        raise ValueError("only complete Sionna RT run identities can be cached")
    key = identity.get("cache_key_sha256")
    if not isinstance(key, str) or not _KEY_PATTERN.fullmatch(key):
        raise ValueError("RT run identity must contain a lowercase SHA256 cache key")
    inputs = identity.get("inputs")
    if not isinstance(inputs, Mapping) or canonical_hash(dict(inputs)) != key:
        raise ValueError("RT run identity inputs do not match its cache key")
    return key


def _paths(cache_root: str | Path, key: str) -> tuple[Path, Path]:
    directory = Path(cache_root).expanduser() / key[:2]
    return directory / f"{key}.npz", directory / f"{key}.json"


def _file_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _read_stable_file(path: Path, *, label: str, max_bytes: int) -> bytes:
    """Read one bounded, single-link regular file without following its leaf."""
    before_path = os.lstat(path)
    if not stat.S_ISREG(before_path.st_mode) or before_path.st_nlink != 1:
        raise ValueError(f"{label} must be a single-link regular file")
    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before_fd = os.fstat(descriptor)
        if _file_identity(before_fd) != _file_identity(before_path):
            raise ValueError(f"{label} changed while opening")
        if before_fd.st_size > max_bytes:
            raise ValueError(f"{label} exceeds the configured byte limit")
        chunks: list[bytes] = []
        size = 0
        while True:
            request_size = min(1024 * 1024, max_bytes + 1 - size)
            if request_size <= 0:
                raise ValueError(f"{label} exceeds the configured byte limit")
            chunk = os.read(descriptor, request_size)
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
        after_fd = os.fstat(descriptor)
        after_path = os.lstat(path)
        identity = _file_identity(before_fd)
        if (
            size != before_fd.st_size
            or _file_identity(after_fd) != identity
            or _file_identity(after_path) != identity
        ):
            raise ValueError(f"{label} changed while being read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate cache manifest key: {key}")
        result[key] = value
    return result


def _validate_npz_uncompressed_size(payload: bytes, *, max_bytes: int) -> None:
    """Bound ZIP expansion before NumPy materializes any cached arrays."""
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            entries = archive.infolist()
            if len(entries) > 32:
                raise ValueError("RT cache NPZ contains too many array entries")
            names = [entry.filename for entry in entries]
            if len(set(names)) != len(names):
                raise ValueError("RT cache NPZ contains duplicate array entries")
            total_uncompressed = sum(entry.file_size for entry in entries)
            if total_uncompressed > max_bytes:
                raise ValueError("RT cache NPZ exceeds the configured uncompressed byte limit")
    except zipfile.BadZipFile as error:
        raise ValueError("RT cache payload is not a valid NPZ archive") from error


def _atomic_immutable_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            temp_path = Path(stream.name)
        try:
            os.link(temp_path, path)
        except FileExistsError:
            try:
                existing = _read_stable_file(
                    path, label="immutable RT cache entry", max_bytes=len(payload)
                )
            except (OSError, ValueError) as error:
                raise RTCacheConflictError(
                    f"immutable RT cache entry conflicts at {path}"
                ) from error
            if existing != payload:
                raise RTCacheConflictError(f"immutable RT cache entry conflicts at {path}")
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def store_rt_dataset(
    cache_root: str | Path,
    dataset: ChannelDataset,
    identity: Mapping[str, Any],
) -> Path:
    """Store one dataset immutably under its verified complete run identity."""
    key = _cache_key(identity)
    recorded_identity = dataset.generation_report.get("rt_run_identity")
    if not isinstance(recorded_identity, Mapping) or _cache_key(recorded_identity) != key:
        raise ValueError("dataset generation report does not match the supplied RT identity")
    payload_path, manifest_path = _paths(cache_root, key)
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    temp_fd, temp_name = tempfile.mkstemp(dir=payload_path.parent, prefix=f".{key}.")
    os.close(temp_fd)
    temporary_payload = Path(temp_name)
    try:
        dataset.save(temporary_payload)
        payload = temporary_payload.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        _atomic_immutable_write(payload_path, payload)
        manifest = {
            "schema": _CACHE_SCHEMA,
            "cache_key_sha256": key,
            "payload_sha256": digest,
            "sample_count": len(dataset.y),
            "shape": list(dataset.h.shape),
            "generation_report_sha256": canonical_hash(dataset.generation_report),
        }
        manifest_bytes = (
            json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode("utf-8")
        _atomic_immutable_write(manifest_path, manifest_bytes)
    finally:
        temporary_payload.unlink(missing_ok=True)
    return payload_path


def load_rt_dataset(
    cache_root: str | Path,
    identity: Mapping[str, Any],
    *,
    max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
) -> ChannelDataset | None:
    """Load a verified cache entry from one bounded, stable byte snapshot.

    ``max_payload_bytes`` bounds both disk reads and the bytes passed to the
    pickle-free NPZ parser. Callers handling a deliberately larger cache must
    opt into a larger limit explicitly.
    """
    if isinstance(max_payload_bytes, bool) or not isinstance(max_payload_bytes, int):
        raise TypeError("max_payload_bytes must be an integer")
    if max_payload_bytes <= 0:
        raise ValueError("max_payload_bytes must be positive")
    key = _cache_key(identity)
    payload_path, manifest_path = _paths(cache_root, key)
    payload_exists = os.path.lexists(payload_path)
    manifest_exists = os.path.lexists(manifest_path)
    if not payload_exists and not manifest_exists:
        return None
    if payload_exists != manifest_exists:
        raise ValueError("incomplete RT cache entry: payload and manifest must both exist")
    try:
        manifest_bytes = _read_stable_file(
            manifest_path, label="RT cache manifest", max_bytes=_MANIFEST_MAX_BYTES
        )
        manifest = json.loads(
            manifest_bytes.decode("utf-8"), object_pairs_hook=_unique_json_object
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("invalid RT cache manifest") from exc
    if (
        not isinstance(manifest, dict)
        or set(manifest)
        != {
            "schema",
            "cache_key_sha256",
            "payload_sha256",
            "sample_count",
            "shape",
            "generation_report_sha256",
        }
        or manifest.get("schema") != _CACHE_SCHEMA
        or manifest.get("cache_key_sha256") != key
        or not isinstance(manifest.get("payload_sha256"), str)
        or _KEY_PATTERN.fullmatch(manifest["payload_sha256"]) is None
        or not isinstance(manifest.get("generation_report_sha256"), str)
        or _KEY_PATTERN.fullmatch(manifest["generation_report_sha256"]) is None
        or isinstance(manifest.get("sample_count"), bool)
        or not isinstance(manifest.get("sample_count"), int)
        or manifest["sample_count"] < 0
        or not isinstance(manifest.get("shape"), list)
        or any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension < 0
            for dimension in manifest["shape"]
        )
    ):
        raise ValueError("RT cache manifest schema or key mismatch")
    payload = _read_stable_file(
        payload_path, label="RT cache payload", max_bytes=max_payload_bytes
    )
    _validate_npz_uncompressed_size(payload, max_bytes=max_payload_bytes)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != manifest.get("payload_sha256"):
        raise ValueError("RT cache payload hash mismatch")
    dataset = ChannelDataset.load_bytes(payload)
    recorded_identity = dataset.generation_report.get("rt_run_identity")
    if not isinstance(recorded_identity, Mapping) or _cache_key(recorded_identity) != key:
        raise ValueError("cached dataset generation identity does not match its cache key")
    if canonical_hash(dataset.generation_report) != manifest.get("generation_report_sha256"):
        raise ValueError("RT cache generation report hash mismatch")
    if list(dataset.h.shape) != manifest.get("shape") or len(dataset.y) != manifest.get(
        "sample_count"
    ):
        raise ValueError("RT cache manifest shape/sample count mismatch")
    return dataset
