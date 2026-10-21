"""Immutable, integrity-checked cache for complete static Sionna RT datasets."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping

from artifacts.provenance import canonical_hash
from datasets.channel import ChannelDataset

_CACHE_SCHEMA = "loslab.rt_dataset_cache.v1"
_KEY_PATTERN = re.compile(r"^[0-9a-f]{64}$")


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
            if path.read_bytes() != payload:
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


def load_rt_dataset(cache_root: str | Path, identity: Mapping[str, Any]) -> ChannelDataset | None:
    """Load a verified cache entry, return ``None`` on a genuine cache miss."""
    key = _cache_key(identity)
    payload_path, manifest_path = _paths(cache_root, key)
    payload_exists = payload_path.is_file()
    manifest_exists = manifest_path.is_file()
    if not payload_exists and not manifest_exists:
        return None
    if payload_exists != manifest_exists:
        raise ValueError("incomplete RT cache entry: payload and manifest must both exist")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("invalid RT cache manifest") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != _CACHE_SCHEMA
        or manifest.get("cache_key_sha256") != key
    ):
        raise ValueError("RT cache manifest schema or key mismatch")
    payload = payload_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != manifest.get("payload_sha256"):
        raise ValueError("RT cache payload hash mismatch")
    dataset = ChannelDataset.load(payload_path)
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
