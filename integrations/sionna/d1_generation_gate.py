"""Fail-closed workflow gate for bounded pilot D1 RT generation.

Production authorization is deliberately disabled: no trusted signature
keyring is configured. Approval names and SHA pins are not verified identity.
Pilot outputs are confined to /tmp, but this remains a workflow boundary,
not an OS sandbox against a caller with arbitrary Python execution.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any, Mapping

_APPROVAL_SCHEMA = "loslab.d1_qualification_approval.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_APPROVAL_BYTES = 256 * 1024
_REQUIRED_GATES = {f"g{index}" for index in range(6)}
_PUBLIC_RELEASE_GATES = {f"g7.{index:02d}" for index in range(1, 16)}


class D1GenerationAuthorization:
    """Record which policy allowed a dataset generation request."""

    __slots__ = (
        "_mode",
        "_request_sha256",
        "_approval_manifest_sha256",
        "_destinations",
        "_seal",
        "_operation",
    )

    def __init__(
        self,
        mode: str,
        request_sha256: str,
        approval_manifest_sha256: str | None,
        destinations: tuple[str, ...],
        operation: str,
        *,
        _seal: object,
    ) -> None:
        if _seal is not _AUTHORITY_SEAL:
            raise PermissionError("D1 authorization records can only be issued by the gate")
        self._mode = mode
        self._request_sha256 = request_sha256
        self._approval_manifest_sha256 = approval_manifest_sha256
        self._destinations = destinations
        self._operation = operation
        self._seal = _seal

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def request_sha256(self) -> str:
        return self._request_sha256

    @property
    def approval_manifest_sha256(self) -> str | None:
        return self._approval_manifest_sha256

    @property
    def destinations(self) -> tuple[str, ...]:
        return self._destinations

    @property
    def operation(self) -> str:
        return self._operation


_AUTHORITY_SEAL = object()
_MAX_PILOT_WORK_UNITS = 250_000
_PILOT_SCRATCH_ROOT = Path("/tmp").resolve()
_PUBLISHER_OPERATIONS = {
    "rt_dataset": ("rt_data",),
    "rt_cache": ("rt_data",),
    "failure_report": ("rt_data",),
    "trajectory_store": ("rt_nr_trajectory",),
    "umcd_bundle": ("rt_data", "rt_nr_trajectory"),
    "dmrs_dataset": ("rt_dmrs_observations",),
}


def canonical_sha256(value: Mapping[str, Any]) -> str:
    """Hash a JSON-compatible request using canonical UTF-8 JSON bytes."""
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("generation request must be finite JSON-compatible data") from exc
    return hashlib.sha256(payload).hexdigest()


def rt_scene_asset_identity(scene_name: str) -> dict[str, Any]:
    """Resolve and hash the complete local asset closure for a named RT scene."""
    try:
        import sionna.rt as rt
    except ImportError as exc:
        raise RuntimeError(
            "production RT authorization requires the configured Sionna RT runtime"
        ) from exc
    from integrations.sionna.scene_assets import scene_asset_manifest

    scene_ref = getattr(rt.scene, scene_name, scene_name)
    scene_path = Path(str(scene_ref)).expanduser()
    if not scene_path.is_file():
        raise ValueError(
            "production RT generation requires a locally resolvable scene asset closure"
        )
    manifest = scene_asset_manifest(scene_path)
    if manifest.get("complete") is not True:
        raise ValueError("production RT generation requires complete scene asset provenance")
    from artifacts.provenance import versions

    return {
        "scene_reference": str(scene_ref),
        "asset_set_sha256": manifest["asset_set_sha256"],
        "asset_count": manifest["asset_count"],
        "runtime_versions": versions(),
    }


def authorize_d1_generation(
    request: Mapping[str, Any],
    *,
    pilot: bool = False,
    approval_manifest: str | Path | None = None,
    expected_approval_sha256: str | None = None,
) -> D1GenerationAuthorization:
    """Issue a bounded-pilot record; production is denied until trust is configured."""
    request_hash = canonical_sha256(request)
    if pilot:
        if approval_manifest is not None or expected_approval_sha256 is not None:
            raise ValueError("pilot mode cannot also supply a production approval")
        _validate_pilot_request(request)
        destinations = _pilot_destinations(request)
        return D1GenerationAuthorization(
            "pilot",
            request_hash,
            None,
            destinations,
            str(request.get("operation")),
            _seal=_AUTHORITY_SEAL,
        )

    # No trusted signature root/keyring is configured in this repository. A
    # self-asserted manifest, even when pinned, cannot authorize production.
    raise PermissionError(
        "production D1 generation is disabled until a trusted approval-signature keyring is configured"
    )


def require_d1_authorization(
    authorization: D1GenerationAuthorization | None,
    request: Mapping[str, Any],
    *,
    expected_operation: str | None = None,
) -> D1GenerationAuthorization:
    """Require an authorization produced for this exact generation request."""
    if not isinstance(authorization, D1GenerationAuthorization):
        raise PermissionError("RT dataset builder requires D1 generation authorization")
    if getattr(authorization, "_seal", None) is not _AUTHORITY_SEAL:
        raise PermissionError("D1 authorization record was not issued by the gate")
    if authorization.request_sha256 != canonical_sha256(request):
        raise PermissionError("D1 generation authorization does not match this exact request")
    if authorization.mode != "pilot":
        raise PermissionError("unknown D1 generation authorization mode")
    if expected_operation is not None and authorization.operation != expected_operation:
        raise PermissionError(
            f"D1 authorization operation {authorization.operation!r} cannot authorize {expected_operation!r}"
        )
    _validate_pilot_request(request)
    if authorization.destinations != _pilot_destinations(request):
        raise PermissionError("D1 pilot authorization destinations do not match this request")
    return authorization


def _validate_pilot_request(request: Mapping[str, Any]) -> None:
    operation = request.get("operation")
    if operation == "rt_data":
        values = request.get("config")
        if not isinstance(values, Mapping):
            raise ValueError("pilot RT authorization requires the full scene config")
        limits = {
            "per_class": 20,
            "max_attempts": 2_000,
            "samples_per_src": 5_000_000,
            "max_depth": 3,
        }
        for field, maximum in limits.items():
            value = values.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                or value > maximum
            ):
                raise ValueError(f"pilot rt_data {field} must be an integer in [1, {maximum}]")
        attempts = values.get("max_attempts")
        samples = values.get("samples_per_src")
        work = None
        if (
            isinstance(attempts, int)
            and not isinstance(attempts, bool)
            and isinstance(samples, int)
            and not isinstance(samples, bool)
        ):
            work = attempts * samples
        if (
            isinstance(work, bool)
            or not isinstance(work, int)
            or not 1 <= work <= _MAX_PILOT_WORK_UNITS
        ):
            raise ValueError(
                "pilot RT total_work_units must be in [1, 250000] (attempts × samples/source)"
            )
        if values.get("total_work_units") != work:
            raise ValueError("pilot RT total_work_units must equal max_attempts × samples_per_src")
    elif operation == "rt_nr_trajectory":
        trajectory_values = request.get("config")
        if not isinstance(trajectory_values, Mapping):
            raise ValueError("pilot trajectory authorization requires its full config")
        values = trajectory_values
        limits = {
            "num_steps": 100,
            "samples_per_src": 5_000_000,
            "max_depth": 3,
            "max_num_paths_per_src": 1_000_000,
        }
        for field, maximum in limits.items():
            value = values.get(field)
            minimum = 0 if field == "max_depth" else 1
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < minimum
                or value > maximum
            ):
                raise ValueError(
                    f"pilot rt_nr_trajectory {field} must be an integer in [{minimum}, {maximum}]"
                )
        steps = trajectory_values.get("num_steps")
        samples = trajectory_values.get("samples_per_src")
        work = None
        if (
            isinstance(steps, int)
            and not isinstance(steps, bool)
            and isinstance(samples, int)
            and not isinstance(samples, bool)
        ):
            work = steps * samples
        if (
            isinstance(work, bool)
            or not isinstance(work, int)
            or not 1 <= work <= _MAX_PILOT_WORK_UNITS
        ):
            raise ValueError(
                "pilot trajectory total_work_units must be in [1, 250000] (steps × samples/source)"
            )
        if trajectory_values.get("total_work_units") != work:
            raise ValueError(
                "pilot trajectory total_work_units must equal num_steps × samples_per_src"
            )
        path_cap = trajectory_values.get("max_num_paths_per_src")
        if (
            isinstance(path_cap, bool)
            or not isinstance(path_cap, int)
            or not 1 <= path_cap <= 1_000_000
        ):
            raise ValueError("pilot trajectory max_num_paths_per_src must be in [1, 1000000]")
    elif operation == "rt_dmrs_observations":
        values = request
        sample_count = values.get("sample_count")
        artifact_count = values.get("input_artifact_count")
        if (
            isinstance(sample_count, bool)
            or not isinstance(sample_count, int)
            or sample_count < 1
            or sample_count > 1_000
        ):
            raise ValueError("pilot RT DM-RS sample_count must be an integer in [1, 1000]")
        if values.get("total_work_units") != sample_count:
            raise ValueError("pilot DM-RS total_work_units must equal sample_count")
        if (
            isinstance(artifact_count, bool)
            or not isinstance(artifact_count, int)
            or artifact_count < 0
            or artifact_count > 20
        ):
            raise ValueError("pilot RT DM-RS input_artifact_count must be an integer in [0, 20]")
        work = values.get("total_work_units")
        if (
            isinstance(work, bool)
            or not isinstance(work, int)
            or work < 1
            or work > _MAX_PILOT_WORK_UNITS
        ):
            raise ValueError("pilot DM-RS total_work_units must be an integer in [1, 250000]")
        return
    else:
        raise ValueError("pilot authorization is limited to recognized RT generation requests")
    for field, maximum in limits.items():
        value = values.get(field)
        minimum = 0 if operation == "rt_nr_trajectory" and field == "max_depth" else 1
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < minimum
            or value > maximum
        ):
            raise ValueError(
                f"pilot {operation} {field} must be an integer in [{minimum}, {maximum}]"
            )


def require_pilot_destination(
    authorization: D1GenerationAuthorization | None,
    destination: str | Path,
    *,
    publisher: str,
) -> Path:
    """Require this exact publisher path to be a gate-bound scratch destination."""
    if (
        not isinstance(authorization, D1GenerationAuthorization)
        or getattr(authorization, "_seal", None) is not _AUTHORITY_SEAL
    ):
        raise PermissionError("publisher requires a gate-issued D1 authorization")
    if authorization.mode != "pilot":
        raise PermissionError("production D1 publication is disabled without a trusted keyring")
    allowed_operations = _PUBLISHER_OPERATIONS.get(publisher)
    if allowed_operations is None or authorization.operation not in allowed_operations:
        raise PermissionError("authorization is not for this publication operation")
    resolved = Path(destination).expanduser().resolve()
    try:
        resolved.relative_to(_PILOT_SCRATCH_ROOT)
    except ValueError as exc:
        raise PermissionError(
            f"pilot publisher path must stay inside fixed scratch root {_PILOT_SCRATCH_ROOT}"
        ) from exc
    if str(resolved) not in authorization.destinations:
        raise PermissionError("publication destination is not bound by the D1 authorization")
    return resolved


def _pilot_destinations(request: Mapping[str, Any]) -> tuple[str, ...]:
    raw = request.get("destinations", [])
    if not isinstance(raw, list) or any(not isinstance(value, str) for value in raw):
        raise ValueError("pilot destinations must be a JSON array of paths")
    if not raw:
        raise PermissionError("pilot generation requires at least one scratch output destination")
    result: list[str] = []
    for value in raw:
        path = Path(value).expanduser().resolve()
        try:
            path.relative_to(_PILOT_SCRATCH_ROOT)
        except ValueError as exc:
            raise PermissionError(
                f"pilot output must stay inside the fixed scratch root {_PILOT_SCRATCH_ROOT}"
            ) from exc
        result.append(str(path))
    return tuple(result)


def _parse_json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key in {label}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must be valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be a JSON object")
    return value


def _validate_approval(manifest: dict[str, Any], path: Path, request_hash: str) -> None:
    required = {
        "schema",
        "decision",
        "approval_id",
        "approved_by",
        "approved_at",
        "scope_request_sha256",
        "gate_statuses",
        "g7_release_scope",
        "g7_release_statuses",
        "qualification_artifacts",
    }
    if set(manifest) != required:
        raise ValueError("approval manifest fields do not match the v1 contract")
    if manifest["schema"] != _APPROVAL_SCHEMA or manifest["decision"] != "approved":
        raise PermissionError("qualification approval schema or decision is not approved")
    for field in ("approval_id", "approved_by", "approved_at"):
        if not isinstance(manifest[field], str) or not manifest[field].strip():
            raise ValueError(f"approval manifest {field} must be a non-empty string")
    if manifest["scope_request_sha256"] != request_hash:
        raise PermissionError("approval does not bind this exact D1 generation request")

    statuses = manifest["gate_statuses"]
    if (
        not isinstance(statuses, dict)
        or set(statuses) != _REQUIRED_GATES
        or any(statuses[gate] != "passed" for gate in _REQUIRED_GATES)
    ):
        raise PermissionError("approval requires passed G0, G1, G2, G3, G4, and G5 statuses")
    release_scope = manifest["g7_release_scope"]
    release_statuses = manifest["g7_release_statuses"]
    if release_scope == "internal_research":
        if release_statuses != {}:
            raise ValueError("internal research scope must not claim G7 public-release checks")
    elif release_scope == "public_release":
        if (
            not isinstance(release_statuses, dict)
            or set(release_statuses) != _PUBLIC_RELEASE_GATES
            or any(release_statuses[gate] != "passed" for gate in _PUBLIC_RELEASE_GATES)
        ):
            raise PermissionError("public release approval requires passed G7.01-G7.15 statuses")
    else:
        raise ValueError("g7_release_scope must be internal_research or public_release")

    artifacts = manifest["qualification_artifacts"]
    if not isinstance(artifacts, list) or len(artifacts) < 2:
        raise ValueError("approval must bind qualification plan and report artifacts")
    labels: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict) or set(artifact) != {"role", "path", "sha256"}:
            raise ValueError("qualification artifacts require exactly role, path, and sha256")
        role, relative_path, digest = (artifact[key] for key in ("role", "path", "sha256"))
        if role not in {"qualification_plan", "qualification_report"} or role in labels:
            raise ValueError("qualification artifacts must include one plan and one report")
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError("qualification artifact path must be a non-empty relative path")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError("qualification artifact sha256 must be lowercase hexadecimal")
        observed = _hash_relative_file(path.parent, relative_path)
        if observed != digest:
            raise PermissionError(f"qualification artifact hash mismatch: {role}")
        labels.add(role)
    if labels != {"qualification_plan", "qualification_report"}:
        raise ValueError("approval must bind both qualification plan and report artifacts")


def _read_stable_file(path: Path, *, max_bytes: int | None = None) -> bytes:
    resolved = path.resolve(strict=True)
    before = resolved.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("approval manifest must be a regular file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(resolved, flags)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise PermissionError("approval manifest changed while opening")
        if max_bytes is not None and opened.st_size > max_bytes:
            raise ValueError("qualification approval manifest is too large")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if max_bytes is not None and total > max_bytes:
                raise ValueError("qualification approval manifest is too large")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        current = resolved.stat()
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) or (after.st_dev, after.st_ino) != (current.st_dev, current.st_ino):
            raise PermissionError("approval manifest changed while reading")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _hash_relative_file(root: Path, relative_path: str) -> str:
    relative = Path(relative_path)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError("qualification artifact paths must stay beneath the approval directory")
    root_resolved = root.resolve(strict=True)
    current = root_resolved
    for part in relative.parts[:-1]:
        current = current / part
        if current.is_symlink() or not current.is_dir():
            raise ValueError("qualification artifact path contains a symlink or non-directory")
    target = current / relative.parts[-1]
    if target.is_symlink() or not target.is_file():
        raise ValueError("qualification artifact must be a regular non-symlink file")
    before = target.stat(follow_symlinks=False)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(target, flags)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
            before.st_dev,
            before.st_ino,
        ):
            raise PermissionError("qualification artifact changed while opening")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
        current_stat = target.stat(follow_symlinks=False)
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) or (after.st_dev, after.st_ino) != (current_stat.st_dev, current_stat.st_ino):
            raise PermissionError("qualification artifact changed while hashing")
        return digest.hexdigest()
    finally:
        os.close(descriptor)
