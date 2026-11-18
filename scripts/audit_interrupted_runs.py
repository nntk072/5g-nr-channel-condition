"""Inventory interrupted receipts and staging directories without cleanup."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jsonschema
from artifacts.manifest import mark_failed
from artifacts.run_replay import transition_run_replay, validate_run_replay_provenance
from referencing import Registry, Resource

_MANIFEST_NAMES = {"manifest.json", "run_manifest.json"}
_SKIP_DIRS = {".git", ".venv", "__pycache__"}
_MAX_ENTRIES = 100_000
_CAPACITY_SCHEMA = (
    Path(__file__).resolve().parents[1]
    / "experiments/schema/capacity-sweep-manifest.v2.schema.json"
)
_REPLAY_SCHEMA = (
    Path(__file__).resolve().parents[1] / "datasets/schema/run-replay-provenance.v1.schema.json"
)
_OBSERVABLE_STAGE_SCHEMA = (
    Path(__file__).resolve().parent / "schema/observable-multipath-staging-receipt.v1.schema.json"
)
_OBSERVABLE_STAGE_SCHEMA_V2 = (
    Path(__file__).resolve().parent / "schema/observable-multipath-staging-receipt.v2.schema.json"
)


def _read_regular_file(path: Path) -> tuple[bytes, str]:
    before = os.lstat(path)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ValueError("manifest must be a single-link regular file")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0))
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError("manifest changed while opening")
        chunks = bytearray()
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.extend(chunk)
        after = os.fstat(descriptor)
        path_after = os.lstat(path)
    finally:
        os.close(descriptor)

    def identity(item: os.stat_result) -> tuple[int, ...]:
        return (
            item.st_dev,
            item.st_ino,
            item.st_mode,
            item.st_nlink,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )

    if identity(before) != identity(after) or identity(before) != identity(path_after):
        raise ValueError("manifest changed while reading")
    payload = bytes(chunks)
    return payload, hashlib.sha256(payload).hexdigest()


def _walk(root: Path):
    count = 0
    for directory, child_dirs, files in os.walk(root, followlinks=False):
        current = Path(directory)
        child_dirs[:] = sorted(
            name
            for name in child_dirs
            if name not in _SKIP_DIRS and not (current / name).is_symlink()
        )
        for name in sorted((*child_dirs, *files)):
            count += 1
            if count > _MAX_ENTRIES:
                raise ValueError(f"scan exceeded the {_MAX_ENTRIES}-entry safety limit")
            yield current / name


def _validate_capacity_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    replay_schema = json.loads(_REPLAY_SCHEMA.read_text(encoding="utf-8"))
    registry = Registry().with_resource(
        _REPLAY_SCHEMA.as_uri(), Resource.from_contents(replay_schema)
    )
    schema = json.loads(_CAPACITY_SCHEMA.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(
        schema, registry=registry, format_checker=jsonschema.FormatChecker()
    ).validate(payload)
    replay = validate_run_replay_provenance(payload["run_replay"])
    config_bytes = json.dumps(
        payload["config"], sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    if hashlib.sha256(config_bytes).hexdigest() != payload["config_sha256"]:
        raise ValueError("capacity sweep config digest mismatch")
    if replay["config_sha256"] != payload["config_sha256"]:
        raise ValueError("capacity sweep replay config digest mismatch")
    if replay["inputs"] != payload["inputs"]:
        raise ValueError("capacity sweep replay inputs mismatch")
    if replay["state"] != payload["state"]:
        raise ValueError("capacity sweep and replay states differ")
    return replay


def _validate_observable_stage_receipt(payload: dict[str, Any]) -> dict[str, Any]:
    replay_schema = json.loads(_REPLAY_SCHEMA.read_text(encoding="utf-8"))
    registry = Registry().with_resource(
        _REPLAY_SCHEMA.as_uri(), Resource.from_contents(replay_schema)
    )
    schema = json.loads(_OBSERVABLE_STAGE_SCHEMA.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(
        schema, registry=registry, format_checker=jsonschema.FormatChecker()
    ).validate(payload)
    replay = validate_run_replay_provenance(payload["run_replay"])
    config_bytes = json.dumps(
        payload["run_config"], sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    config_digest = hashlib.sha256(config_bytes).hexdigest()
    if payload["run_config_sha256"] != config_digest or replay["config_sha256"] != config_digest:
        raise ValueError("observable staging receipt config digest mismatch")
    if replay["state"] != payload["state"]:
        raise ValueError("observable staging and replay states differ")
    inputs = replay["inputs"]
    if len(inputs) != 1 or inputs[0]["path"] != "generated:grouped_synthetic_two_path_dataset":
        raise ValueError("observable staging receipt has an unexpected input roster")
    return replay


def _validate_observable_stage_receipt_v2(payload: dict[str, Any]) -> dict[str, Any] | None:
    replay_schema = json.loads(_REPLAY_SCHEMA.read_text(encoding="utf-8"))
    registry = Registry().with_resource(
        _REPLAY_SCHEMA.as_uri(), Resource.from_contents(replay_schema)
    )
    schema = json.loads(_OBSERVABLE_STAGE_SCHEMA_V2.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(
        schema, registry=registry, format_checker=jsonschema.FormatChecker()
    ).validate(payload)
    config_bytes = json.dumps(
        payload["run_config"], sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    config_digest = hashlib.sha256(config_bytes).hexdigest()
    if payload["run_config_sha256"] != config_digest:
        raise ValueError("observable staging receipt config digest mismatch")
    if payload["phase"] == "dataset_generation":
        if "run_replay" in payload:
            raise ValueError("dataset-generation receipt cannot claim replay inputs")
        return None

    replay = validate_run_replay_provenance(payload["run_replay"])
    if replay["config_sha256"] != config_digest:
        raise ValueError("observable staging receipt config digest mismatch")
    if replay["state"] != payload["state"]:
        raise ValueError("observable staging and replay states differ")
    if replay["source"]["git_commit"] != payload["source_identity"]["revision"]:
        raise ValueError("observable staging source revision mismatch")
    if (
        replay["source"]["package_source_sha256"]
        != payload["source_identity"]["package_source_sha256"]
    ):
        raise ValueError("observable staging source digest mismatch")
    backend = replay["backend_identity"]
    if (
        backend["platform"] != payload["runtime_identity"]["platform"]
        or backend["dependencies"] != payload["runtime_identity"]["dependencies"]
        or backend["requested_device"] != payload["runtime_identity"]["requested_device"]
    ):
        raise ValueError("observable staging runtime identity mismatch")
    if replay["command"]["argv"] != payload["command_argv"]:
        raise ValueError("observable staging command identity mismatch")
    if replay["rng_streams"] != payload["rng_streams"]:
        raise ValueError("observable staging RNG roster mismatch")
    inputs = replay["inputs"]
    if len(inputs) != 1 or inputs[0]["path"] != "generated:grouped_synthetic_two_path_dataset":
        raise ValueError("observable staging receipt has an unexpected input roster")
    return replay


def _recovery_supported(payload: dict[str, Any]) -> bool:
    schema = payload.get("schema")
    if schema == "loslab.run.v2":
        replay = validate_run_replay_provenance(payload.get("run_replay", {}))
        return payload.get("state") == "running" and replay["state"] == "running"
    if schema == "loslab.capacity_sweep.v2":
        replay = _validate_capacity_manifest(payload)
        return payload.get("state") == "running" and replay["state"] == "running"
    if schema == "loslab.observable_multipath_staging.v1":
        replay = _validate_observable_stage_receipt(payload)
        return payload.get("state") == "running" and replay["state"] == "running"
    if schema == "loslab.observable_multipath_staging.v2":
        replay = _validate_observable_stage_receipt_v2(payload)
        if payload.get("state") != "running":
            return False
        return replay is None or replay["state"] == "running"
    return False


def audit_root(root: str | Path) -> dict[str, Any]:
    """Return a bounded inventory; never mutate or remove discovered artifacts."""
    base = Path(root).expanduser().resolve(strict=True)
    if not base.is_dir():
        raise ValueError("root must be a directory")
    manifests: list[dict[str, Any]] = []
    staging: dict[str, dict[str, int]] = {}
    for path in _walk(base):
        if path.is_dir() and (".staging-" in path.name or path.name == ".staging"):
            staging.setdefault(path.relative_to(base).as_posix(), {"files": 0, "bytes": 0})
        if path.is_file() and path.name in _MANIFEST_NAMES:
            entry: dict[str, Any] = {"path": path.relative_to(base).as_posix()}
            try:
                raw, digest = _read_regular_file(path)
                payload = json.loads(raw)
                if not isinstance(payload, dict):
                    raise ValueError("manifest root must be an object")
                entry.update(
                    {
                        "sha256": digest,
                        "schema": payload.get("schema"),
                        "state": payload.get("state"),
                        "replay_state": None,
                        "recovery_supported": False,
                    }
                )
                replay = payload.get("run_replay")
                if replay is not None:
                    replay = validate_run_replay_provenance(replay)
                    entry["replay_state"] = replay["state"]
                entry["recovery_supported"] = _recovery_supported(payload)
            except (
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                jsonschema.ValidationError,
                ValueError,
            ) as error:
                entry["error"] = f"{type(error).__name__}: {error}"
            manifests.append(entry)
        if not path.is_file():
            continue
        for parent in path.parents:
            if parent == base.parent:
                break
            if ".staging-" in parent.name or parent.name == ".staging":
                relative = parent.relative_to(base).as_posix()
                item = staging.setdefault(relative, {"files": 0, "bytes": 0})
                try:
                    info = path.lstat()
                    if stat.S_ISREG(info.st_mode):
                        item["files"] += 1
                        item["bytes"] += info.st_size
                except OSError:
                    pass
                break
    return {
        "schema": "loslab.interrupted_run_audit.v1",
        "root": str(base),
        "manifests": sorted(manifests, key=lambda row: row["path"]),
        "staging_directories": [{"path": name, **staging[name]} for name in sorted(staging)],
        "policy": "inventory_only; staging data is never removed; receipt state does not establish scientific qualification",
    }


def mark_interrupted_failed(
    root: str | Path,
    manifest_relative_path: str,
    *,
    expected_sha256: str,
    operator_attestation: str,
) -> dict[str, Any]:
    """Terminalize a digest-pinned common or capacity run as interrupted."""
    base = Path(root).expanduser().resolve(strict=True)
    relative = Path(manifest_relative_path)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.as_posix() != manifest_relative_path
    ):
        raise ValueError("manifest path must be a safe relative POSIX path")
    if not isinstance(operator_attestation, str) or not operator_attestation.strip():
        raise ValueError("operator_attestation must state that the run is no longer active")
    target = base / relative
    if target.resolve(strict=True).parent != target.parent.resolve(strict=True):
        raise ValueError("manifest path may not traverse a symlink")
    raw, actual_digest = _read_regular_file(target)
    if actual_digest != expected_sha256:
        raise ValueError("manifest SHA-256 changed; re-audit before recovery")
    payload = json.loads(raw)
    if not isinstance(payload, dict) or not _recovery_supported(payload):
        raise ValueError("manifest is not a supported, schema-valid running receipt")
    schema = payload["schema"]
    if schema == "loslab.run.v2":
        failed = mark_failed(
            payload,
            error_type="InterruptedRunRecovery",
            error_message=operator_attestation.strip(),
            stage="operator_recovery",
        )
    elif schema == "loslab.capacity_sweep.v2":
        changed_utc = datetime.now(timezone.utc).isoformat()
        failed = {
            **payload,
            "state": "failed",
            "failed_utc": changed_utc,
            "error": operator_attestation.strip(),
            "failure": {
                "error_type": "InterruptedRunRecovery",
                "message": operator_attestation.strip(),
                "stage": "operator_recovery",
            },
            "run_replay": transition_run_replay(
                payload["run_replay"],
                state="failed",
                changed_utc=changed_utc,
                failure={
                    "error_type": "InterruptedRunRecovery",
                    "message": operator_attestation.strip(),
                    "stage": "operator_recovery",
                },
            ),
        }
        _validate_capacity_manifest(failed)
    elif schema == "loslab.observable_multipath_staging.v1":
        changed_utc = datetime.now(timezone.utc).isoformat()
        failed = {
            **payload,
            "state": "failed",
            "failed_utc": changed_utc,
            "failure": {
                "error_type": "InterruptedRunRecovery",
                "message": operator_attestation.strip(),
                "stage": "operator_recovery",
            },
            "run_replay": transition_run_replay(
                payload["run_replay"],
                state="failed",
                changed_utc=changed_utc,
                failure={
                    "error_type": "InterruptedRunRecovery",
                    "message": operator_attestation.strip(),
                    "stage": "operator_recovery",
                },
            ),
        }
        _validate_observable_stage_receipt(failed)
    else:
        changed_utc = datetime.now(timezone.utc).isoformat()
        failure = {
            "error_type": "InterruptedRunRecovery",
            "message": operator_attestation.strip(),
            "stage": "operator_recovery",
        }
        failed = {
            **payload,
            "state": "failed",
            "failed_utc": changed_utc,
            "failure": failure,
        }
        if "run_replay" in payload:
            failed["run_replay"] = transition_run_replay(
                payload["run_replay"],
                state="failed",
                changed_utc=changed_utc,
                failure=failure,
            )
        _validate_observable_stage_receipt_v2(failed)
    encoded = (json.dumps(failed, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    original_mode = stat.S_IMODE(os.stat(target, follow_symlinks=False).st_mode)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.recovery-", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            os.fchmod(stream.fileno(), original_mode)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        _, current_digest = _read_regular_file(target)
        if current_digest != expected_sha256:
            raise ValueError("manifest changed during recovery; re-audit before retrying")
        os.replace(temporary, target)
        directory_descriptor = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": manifest_relative_path,
        "state": "failed",
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="run-output root to inventory")
    parser.add_argument("--mark-failed", metavar="RELATIVE_MANIFEST")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--operator-attestation")
    args = parser.parse_args(argv)
    if args.mark_failed:
        if not args.expected_sha256 or not args.operator_attestation:
            parser.error("--mark-failed requires --expected-sha256 and --operator-attestation")
        result = mark_interrupted_failed(
            args.root,
            args.mark_failed,
            expected_sha256=args.expected_sha256,
            operator_attestation=args.operator_attestation,
        )
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0
    print(json.dumps(audit_root(args.root), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
