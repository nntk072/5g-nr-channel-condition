"""Construction of signed joint-evidence run manifests."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from evaluation.acceptance.qualification_contract import verify_qualification_run_contract
from evaluation.acquisition_package import (
    MAX_FILE_BYTES,
    MAX_SCHEMA_BYTES,
    _json_bytes,
    _read_inventory_file,
    _safe_relative_path,
)
from evaluation.run_manifest.common import (
    _runtime_identity,
)
from evaluation.run_manifest.constants import (
    _MAX_CAPTURE_BYTES,
    _MAX_SOURCE_BUNDLE_BYTES,
    _RUN_ID_PATTERN,
    EVALUATOR_SOURCE_PATHS,
    RUNTIME_INVENTORY_SCHEMA_V3_RELATIVE_PATH,
    SCHEMA_V3_RELATIVE_PATH,
    SCHEMA_V5_RELATIVE_PATH,
    V5_EVALUATOR_SOURCE_PATHS,
)
from evaluation.run_manifest.contracts import (
    _validate_joint_evaluator_input_schema,
    _verify_qualification_contract_bindings,
)
from evaluation.run_manifest.sources import (
    _read_rooted_source_file,
    _rooted_file_identity,
    _source_bundle_sha256,
)


def build_joint_evidence_run_manifest(
    run_id: str,
    input_path: str | Path,
    artifact_path: str | Path,
    *,
    package_root: str | Path | None = None,
    source_root: str | Path | None = None,
    qualification_contract: dict[str, Any] | None = None,
    runtime_identity: dict[str, Any] | None = None,
    runtime_inventory_path: str | Path | None = None,
) -> dict[str, Any]:
    """Describe exact run files, optional preregistration, and scorer source identities."""
    if not isinstance(run_id, str) or _RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError(
            "run_id must be 3–64 lowercase letters, digits, dots, underscores, or hyphens"
        )
    input_file = Path(os.path.abspath(input_path))
    artifact_file = Path(os.path.abspath(artifact_path))
    package_dir = Path(package_root) if package_root is not None else input_file.parent
    package_dir = package_dir.resolve(strict=True)
    source_dir = (
        Path(source_root).resolve(strict=True)
        if source_root is not None
        else Path(__file__).resolve().parents[2]
    )
    if not package_dir.is_dir() or not source_dir.is_dir():
        raise ValueError("package_root and source_root must be directories")
    package_fd = os.open(
        package_dir, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
    )
    source_fd = os.open(
        source_dir, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
    )
    try:
        input_relative, input_bytes, input_size, input_digest = _rooted_file_identity(
            package_fd,
            package_dir,
            input_file,
            limit=_MAX_CAPTURE_BYTES,
            capture=True,
        )
        artifact_relative, artifact_bytes, artifact_size, artifact_digest = _rooted_file_identity(
            package_fd,
            package_dir,
            artifact_file,
            limit=_MAX_CAPTURE_BYTES,
            capture=True,
        )
        runtime_inventory_entry: dict[str, Any] | None = None
        if runtime_inventory_path is not None:
            inventory_file = Path(os.path.abspath(runtime_inventory_path))
            inventory_relative, inventory_bytes, inventory_size, inventory_digest = (
                _rooted_file_identity(
                    package_fd,
                    package_dir,
                    inventory_file,
                    limit=_MAX_CAPTURE_BYTES,
                    capture=True,
                )
            )
            if inventory_bytes is None:
                raise ValueError("runtime inventory bytes were not captured")
            inventory_payload = _json_bytes(inventory_bytes, "G3 runtime inventory")
            import jsonschema

            inventory_schema = _json_bytes(
                _read_rooted_source_file(
                    source_dir,
                    RUNTIME_INVENTORY_SCHEMA_V3_RELATIVE_PATH,
                    limit=MAX_SCHEMA_BYTES,
                ),
                "runtime inventory v3 schema",
            )
            inventory_errors = list(
                jsonschema.Draft202012Validator(inventory_schema).iter_errors(inventory_payload)
            )
            if inventory_errors:
                raise ValueError(
                    "runtime inventory violates v3 schema: "
                    + "; ".join(error.message for error in inventory_errors[:10])
                )
            if inventory_relative in (input_relative, artifact_relative):
                raise ValueError("runtime inventory, input, and artifact paths must be distinct")
            runtime_inventory_entry = {
                "path": inventory_relative,
                "size_bytes": inventory_size,
                "sha256": inventory_digest,
                "role": "runtime_inventory",
            }
        if qualification_contract is not None:
            if input_bytes is None:
                raise ValueError("joint evaluator input was not captured")
            _validate_joint_evaluator_input_schema(
                _json_bytes(input_bytes, "G3 joint evaluator input"),
                source_root=source_dir,
            )
            _verify_qualification_contract_bindings(qualification_contract, input_bytes)
        del input_bytes, artifact_bytes
        if input_relative == artifact_relative:
            raise ValueError("input and artifact paths must be distinct")
        sources: list[dict[str, Any]] = []
        source_bytes_total = 0
        source_roster = (
            V5_EVALUATOR_SOURCE_PATHS
            if runtime_inventory_entry is not None
            else EVALUATOR_SOURCE_PATHS
        )
        for relative_path in source_roster:
            parts = _safe_relative_path(relative_path)
            remaining_source_budget = _MAX_SOURCE_BUNDLE_BYTES - source_bytes_total
            if remaining_source_budget <= 0:
                raise ValueError("evaluator source roster exceeds its total byte limit")
            _, size, digest = _read_inventory_file(
                source_fd,
                parts,
                min(MAX_FILE_BYTES, remaining_source_budget),
                capture=False,
            )
            source_bytes_total += size
            sources.append({"path": relative_path, "size_bytes": size, "sha256": digest})
    finally:
        os.close(package_fd)
        os.close(source_fd)
    source_bundle_sha256 = _source_bundle_sha256(sources)
    if qualification_contract is not None and qualification_contract["source_identity"][
        "source_bundle_sha256"
    ] != _source_bundle_sha256(
        [item for item in sources if item["path"] in EVALUATOR_SOURCE_PATHS]
    ):
        raise ValueError("qualification contract source bundle differs from run receipt sources")
    manifest = {
        "schema": (
            "loslab.g3_joint_evidence_run_manifest.v5"
            if runtime_inventory_entry is not None
            else "loslab.g3_joint_evidence_run_manifest.v3"
        ),
        "run_id": run_id,
        "qualification_status": "NOT_QUALIFIED",
        "runtime_identity": (_runtime_identity() if runtime_identity is None else runtime_identity),
        "input": {
            "path": input_relative,
            "size_bytes": input_size,
            "sha256": input_digest,
            "role": "joint_evaluator_input",
        },
        "artifact": {
            "path": artifact_relative,
            "size_bytes": artifact_size,
            "sha256": artifact_digest,
            "role": "joint_evaluator_artifact",
        },
        "evaluator_sources": sources,
        "evaluator_source_bundle_sha256": source_bundle_sha256,
    }
    if runtime_inventory_entry is not None:
        manifest["runtime_inventory"] = runtime_inventory_entry
    if qualification_contract is not None:
        if not isinstance(qualification_contract, dict):
            raise TypeError("qualification_contract must be an object")
        verify_qualification_run_contract(qualification_contract)
        if qualification_contract["run_id"] != run_id:
            raise ValueError("qualification contract run_id differs from run receipt")
        manifest["qualification_contract"] = qualification_contract
    import jsonschema

    manifest_schema = _json_bytes(
        _read_rooted_source_file(
            source_dir,
            SCHEMA_V5_RELATIVE_PATH
            if runtime_inventory_entry is not None
            else SCHEMA_V3_RELATIVE_PATH,
            limit=MAX_SCHEMA_BYTES,
        ),
        "G3 run manifest schema",
    )
    schema_errors = list(jsonschema.Draft202012Validator(manifest_schema).iter_errors(manifest))
    if schema_errors:
        raise ValueError(
            "G3 run manifest violates selected schema: "
            + "; ".join(error.message for error in schema_errors[:10])
        )
    return manifest
