"""Strict, non-qualifying run replay provenance contracts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import jsonschema

_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "datasets"
    / "schema"
    / "run-replay-provenance.v1.schema.json"
)
_SCHEMA_NAME = "loslab.run_replay_provenance.v1"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _canonical_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a canonical nonempty string")
    return value


def _utc(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be an ISO-8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{label} must be an ISO-8601 UTC timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{label} must be an ISO-8601 UTC timestamp")
    return value


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _seed(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**63:
        raise ValueError(f"{label} must be an integer in [0, 2**63)")
    return value


def _normalise_inputs(inputs: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    if not isinstance(inputs, Sequence) or isinstance(inputs, (str, bytes)) or not inputs:
        raise ValueError("inputs must contain at least one artifact")
    result: list[dict[str, str]] = []
    for index, item in enumerate(inputs):
        if not isinstance(item, Mapping) or set(item) != {"path", "sha256"}:
            raise ValueError(f"inputs[{index}] must contain exactly path and sha256")
        path = _canonical_text(item["path"], f"inputs[{index}].path")
        digest = _digest(item["sha256"], f"inputs[{index}].sha256")
        result.append({"path": path, "sha256": digest})
    return result


def _normalise_backend(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "requested_device",
        "platform",
        "dependencies",
    }:
        raise ValueError("backend_identity has an invalid field roster")
    dependencies = value["dependencies"]
    if not isinstance(dependencies, Mapping) or not dependencies:
        raise ValueError("backend dependencies must be a nonempty mapping")
    normalised_dependencies: dict[str, str] = {}
    for name, version in sorted(dependencies.items()):
        normalised_dependencies[_canonical_text(name, "dependency name")] = _canonical_text(
            version, f"dependency {name!r} version"
        )
    return {
        "requested_device": _canonical_text(value["requested_device"], "requested_device"),
        "platform": _canonical_text(value["platform"], "platform"),
        "dependencies": normalised_dependencies,
    }


def _normalise_rng_streams(streams: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(streams, Sequence) or isinstance(streams, (str, bytes)) or not streams:
        raise ValueError("rng_streams must contain at least one stream")
    rows: list[dict[str, Any]] = []
    names: set[str] = set()
    expected = {"name", "library", "algorithm", "seed_material", "derived_seed", "purpose"}
    for index, item in enumerate(streams):
        if not isinstance(item, Mapping) or set(item) != expected:
            raise ValueError(f"rng_streams[{index}] has an invalid field roster")
        name = _canonical_text(item["name"], f"rng_streams[{index}].name")
        if name in names:
            raise ValueError(f"duplicate RNG stream name: {name}")
        names.add(name)
        material = item["seed_material"]
        if not isinstance(material, list) or not material:
            raise ValueError(f"rng_streams[{index}].seed_material must be a nonempty array")
        seed_material = [_seed(value, f"rng_streams[{index}].seed_material") for value in material]
        derived = item["derived_seed"]
        if derived is not None:
            derived = _seed(derived, f"rng_streams[{index}].derived_seed")
        rows.append(
            {
                "name": name,
                "library": _canonical_text(item["library"], f"rng_streams[{index}].library"),
                "algorithm": _canonical_text(item["algorithm"], f"rng_streams[{index}].algorithm"),
                "seed_material": seed_material,
                "derived_seed": derived,
                "purpose": _canonical_text(item["purpose"], f"rng_streams[{index}].purpose"),
            }
        )
    return sorted(rows, key=lambda row: row["name"])


def _source_identity(source_revision: Any, package_source_sha256: Any) -> dict[str, str]:
    package_digest = _digest(package_source_sha256, "package_source_sha256")
    if source_revision == "unavailable":
        return {
            "git_commit": "unavailable",
            "git_commit_status": "unavailable",
            "package_source_sha256": package_digest,
        }
    if (
        not isinstance(source_revision, str)
        or len(source_revision) not in {40, 64}
        or any(char not in "0123456789abcdef" for char in source_revision)
    ):
        raise ValueError("source_revision must be a lowercase Git object ID or unavailable")
    return {
        "git_commit": source_revision,
        "git_commit_status": "bound",
        "package_source_sha256": package_digest,
    }


def build_run_replay_provenance(
    *,
    created_utc: str,
    command_argv: Sequence[str],
    command_source: str,
    source_revision: str,
    package_source_sha256: str,
    config_sha256: str,
    inputs: Sequence[Mapping[str, Any]],
    backend_identity: Mapping[str, Any],
    rng_streams: Sequence[Mapping[str, Any]],
    attempt_index: int = 0,
    retry_of_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Build one deterministic replay envelope for a started run."""
    created = _utc(created_utc, "created_utc")
    if not isinstance(command_argv, Sequence) or isinstance(command_argv, (str, bytes)):
        raise ValueError("command_argv must be an argv sequence")
    argv = list(command_argv)
    if not argv or any(not isinstance(value, str) for value in argv):
        raise ValueError("command_argv must contain only strings and may not be empty")
    if isinstance(attempt_index, bool) or not isinstance(attempt_index, int) or attempt_index < 0:
        raise ValueError("attempt_index must be a non-negative integer")
    retry_digest = (
        None
        if retry_of_manifest_sha256 is None
        else _digest(retry_of_manifest_sha256, "retry_of_manifest_sha256")
    )
    if attempt_index == 0 and retry_digest is not None:
        raise ValueError("attempt zero may not declare a retry parent")
    if attempt_index > 0 and retry_digest is None:
        raise ValueError("retry attempts must bind the previous manifest SHA-256")
    backend = _normalise_backend(backend_identity)
    core = {
        "schema": _SCHEMA_NAME,
        "qualification_status": "NOT_QUALIFIED",
        "created_utc": created,
        "command": {
            "argv": argv,
            "source": _canonical_text(command_source, "command_source"),
        },
        "source": _source_identity(source_revision, package_source_sha256),
        "config_sha256": _digest(config_sha256, "config_sha256"),
        "inputs": _normalise_inputs(inputs),
        "backend_identity": backend,
        "backend_identity_sha256": hashlib.sha256(_canonical_json(backend)).hexdigest(),
        "rng_streams": _normalise_rng_streams(rng_streams),
        "attempt": {
            "index": attempt_index,
            "retry_of_manifest_sha256": retry_digest,
        },
        "state": "running",
        "state_changed_utc": created,
        "failure": None,
    }
    result = {**core, "contract_sha256": hashlib.sha256(_canonical_json(core)).hexdigest()}
    return validate_run_replay_provenance(result)


def validate_run_replay_provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate schema plus digest and cross-field invariants."""
    payload = json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(payload, schema)
    _utc(payload["created_utc"], "created_utc")
    _utc(payload["state_changed_utc"], "state_changed_utc")
    if payload["source"]["git_commit_status"] == "bound":
        _source_identity(
            payload["source"]["git_commit"], payload["source"]["package_source_sha256"]
        )
    elif payload["source"]["git_commit"] != "unavailable":
        raise ValueError("unavailable Git status requires git_commit=unavailable")
    expected_backend = hashlib.sha256(_canonical_json(payload["backend_identity"])).hexdigest()
    if payload["backend_identity_sha256"] != expected_backend:
        raise ValueError("backend identity digest mismatch")
    names = [row["name"] for row in payload["rng_streams"]]
    if names != sorted(set(names)):
        raise ValueError("RNG stream roster must be sorted and unique")
    if payload["attempt"]["index"] == 0 and payload["attempt"]["retry_of_manifest_sha256"]:
        raise ValueError("attempt zero may not declare a retry parent")
    if payload["attempt"]["index"] > 0 and not payload["attempt"]["retry_of_manifest_sha256"]:
        raise ValueError("retry attempts must bind the previous manifest SHA-256")
    state = payload["state"]
    if (state == "failed") != (payload["failure"] is not None):
        raise ValueError("failure details must be present exactly when state is failed")
    core = {key: val for key, val in payload.items() if key != "contract_sha256"}
    if payload["contract_sha256"] != hashlib.sha256(_canonical_json(core)).hexdigest():
        raise ValueError("run replay contract digest mismatch")
    return payload


def transition_run_replay(
    value: Mapping[str, Any],
    *,
    state: str,
    changed_utc: str | None = None,
    failure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Move one validated running contract to a terminal state."""
    current = validate_run_replay_provenance(value)
    if current["state"] != "running":
        raise ValueError("only a running replay contract may transition")
    if state not in {"complete", "failed"}:
        raise ValueError("terminal state must be complete or failed")
    failure_payload = None
    if state == "failed":
        if not isinstance(failure, Mapping) or set(failure) != {"error_type", "message", "stage"}:
            raise ValueError("failed state requires error_type, message, and stage")
        failure_payload = {
            "error_type": _canonical_text(failure["error_type"], "failure.error_type"),
            "message": failure["message"],
            "stage": _canonical_text(failure["stage"], "failure.stage"),
        }
        if not isinstance(failure_payload["message"], str):
            raise ValueError("failure.message must be a string")
    elif failure is not None:
        raise ValueError("complete state may not carry failure details")
    current.pop("contract_sha256")
    current["state"] = state
    current["state_changed_utc"] = _utc(changed_utc or _now_utc(), "state_changed_utc")
    current["failure"] = failure_payload
    result = {
        **current,
        "contract_sha256": hashlib.sha256(_canonical_json(current)).hexdigest(),
    }
    return validate_run_replay_provenance(result)
