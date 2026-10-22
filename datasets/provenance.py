"""Immutable, content-hashable generation provenance for UMCD artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SUBJECT_LAYERS = {
    "episode",
    "oracle_paths",
    "oracle_channels",
    "receiver_observations",
    "targets",
}


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _snapshot_mapping(value: Mapping[str, Any], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed mapping")
    try:
        snapshot = json.loads(
            json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
        )
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain finite JSON-compatible values") from exc
    return _freeze(snapshot)


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


@dataclass(frozen=True, slots=True)
class UMCDRunProvenance:
    """Provenance needed to interpret and reproduce one bundle layer."""

    record_id: str
    subject_layer: str
    source_family: str
    producer: str
    package_source_sha256: str
    software_versions: Mapping[str, str]
    backend: Mapping[str, Any]
    scene_asset_sha256: Mapping[str, str]
    seed: int | str | None
    physical_config: Mapping[str, Any]
    solver_config: Mapping[str, Any]
    material_config: Mapping[str, Any]
    run_identity_sha256: str | None = None

    def __post_init__(self) -> None:
        for name in ("record_id", "subject_layer", "source_family", "producer"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.subject_layer not in _SUBJECT_LAYERS:
            raise ValueError("subject_layer is not a supported UMCD data layer")
        if not _SHA256.fullmatch(self.package_source_sha256):
            raise ValueError("package_source_sha256 must be a lowercase SHA256 digest")
        if self.run_identity_sha256 is not None and not _SHA256.fullmatch(self.run_identity_sha256):
            raise ValueError("run_identity_sha256 must be a lowercase SHA256 digest")
        if isinstance(self.seed, bool) or (
            self.seed is not None and not isinstance(self.seed, (int, str))
        ):
            raise TypeError("seed must be a nonnegative integer, string, or None")
        if isinstance(self.seed, int) and self.seed < 0:
            raise ValueError("integer seed must be nonnegative")
        if isinstance(self.seed, str) and not self.seed.strip():
            raise ValueError("string seed must be non-empty")
        versions = dict(self.software_versions)
        if not versions or any(
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(version, str)
            or not version.strip()
            for name, version in versions.items()
        ):
            raise ValueError("software_versions must contain named, non-empty versions")
        assets = dict(self.scene_asset_sha256)
        for path, digest in assets.items():
            if (
                not isinstance(path, str)
                or not path.strip()
                or not isinstance(digest, str)
                or not _SHA256.fullmatch(digest)
            ):
                raise ValueError("scene asset entries need a path and lowercase SHA256 digest")
        object.__setattr__(self, "software_versions", MappingProxyType(versions))
        object.__setattr__(self, "scene_asset_sha256", MappingProxyType(assets))
        object.__setattr__(self, "backend", _snapshot_mapping(self.backend, name="backend"))
        object.__setattr__(
            self,
            "physical_config",
            _snapshot_mapping(self.physical_config, name="physical_config"),
        )
        object.__setattr__(
            self, "solver_config", _snapshot_mapping(self.solver_config, name="solver_config")
        )
        object.__setattr__(
            self,
            "material_config",
            _snapshot_mapping(self.material_config, name="material_config"),
        )
        if not self.backend:
            raise ValueError("backend must identify the execution backend")

    def to_dict(self) -> dict[str, Any]:
        """Return canonical JSON data with an integrity digest."""
        core: dict[str, Any] = {
            "schema": "umcd.run_provenance.v1",
            "schema_version": 1,
            "record_id": self.record_id,
            "subject_layer": self.subject_layer,
            "source_family": self.source_family,
            "producer": self.producer,
            "package_source_sha256": self.package_source_sha256,
            "software_versions": dict(sorted(self.software_versions.items())),
            "backend": _thaw(self.backend),
            "scene_asset_sha256": dict(sorted(self.scene_asset_sha256.items())),
            "seed": self.seed,
            "physical_config": _thaw(self.physical_config),
            "solver_config": _thaw(self.solver_config),
            "material_config": _thaw(self.material_config),
            "run_identity_sha256": self.run_identity_sha256,
        }
        return {**core, "record_sha256": hashlib.sha256(_canonical(core)).hexdigest()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> UMCDRunProvenance:
        """Parse a record and reject unsupported or tampered content."""
        if value.get("schema") != "umcd.run_provenance.v1" or value.get("schema_version") != 1:
            raise ValueError("unsupported UMCD provenance schema")
        core = {key: item for key, item in value.items() if key != "record_sha256"}
        if hashlib.sha256(_canonical(core)).hexdigest() != value.get("record_sha256"):
            raise ValueError("UMCD provenance hash is invalid")
        record = cls(
            record_id=value["record_id"],
            subject_layer=value["subject_layer"],
            source_family=value["source_family"],
            producer=value["producer"],
            package_source_sha256=value["package_source_sha256"],
            software_versions=value["software_versions"],
            backend=value["backend"],
            scene_asset_sha256=value["scene_asset_sha256"],
            seed=value["seed"],
            physical_config=value["physical_config"],
            solver_config=value["solver_config"],
            material_config=value["material_config"],
            run_identity_sha256=value["run_identity_sha256"],
        )
        if record.to_dict() != dict(value):
            raise ValueError("UMCD provenance representation is not canonical")
        return record
